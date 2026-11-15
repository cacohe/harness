import itertools
from concurrent.futures import ThreadPoolExecutor

import pytest
from src.observability import CompositeEventSink, MetricsEventSink
from src.schema.event import (
    Event,
    EventChannel,
    EventType,
    LlmEventPayload,
    TaskEventPayload,
    ToolEventPayload,
)
from src.schema.llm import LlmUsage


def _task(event_type, task_id="task-1"):
    return Event(
        id=f"{event_type.value}-{task_id}",
        type=event_type,
        payload=TaskEventPayload(task_id=task_id),
        task_id=task_id,
    )


def _llm(event_type, *, task_id="task-1", phase="infer", usage=None):
    return Event(
        id=f"{event_type.value}-{task_id}-{phase}",
        type=event_type,
        payload=LlmEventPayload(phase=phase, usage=usage),
        task_id=task_id,
    )


def _tool(event_type, *, task_id="task-1", call_id="call-1"):
    return Event(
        id=f"{event_type.value}-{task_id}-{call_id}",
        type=event_type,
        payload=ToolEventPayload(tool_call_id=call_id, name="shell"),
        task_id=task_id,
    )


def test_metrics_pair_durations_count_outcomes_and_sum_llm_usage():
    ticks = iter([0.0, 1.0, 3.0, 4.0, 6.0, 10.0])
    metrics = MetricsEventSink(clock=lambda: next(ticks))

    for event in [
        _task(EventType.TASK_STARTED),
        _llm(EventType.LLM_STARTED),
        _llm(
            EventType.LLM_COMPLETED,
            usage=LlmUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5),
        ),
        _tool(EventType.TOOL_EXECUTION_STARTED),
        _tool(EventType.TOOL_EXECUTION_COMPLETED),
        _task(EventType.TASK_COMPLETED),
    ]:
        metrics.emit(event)

    snapshot = metrics.snapshot()
    assert snapshot["events"] == {
        "llm_completed": 1,
        "llm_started": 1,
        "task_completed": 1,
        "task_started": 1,
        "tool_execution_completed": 1,
        "tool_execution_started": 1,
    }
    assert snapshot["task_outcomes"] == {"completed": 1}
    assert snapshot["llm_tokens"] == {"prompt": 2, "completion": 3, "total": 5}
    assert snapshot["in_flight"] == {"tasks": 0, "llm": 0, "tools": 0}
    assert snapshot["durations"] == {
        "task": {"count": 1, "total_seconds": 10.0, "max_seconds": 10.0},
        "llm": {"count": 1, "total_seconds": 2.0, "max_seconds": 2.0},
        "tool": {"count": 1, "total_seconds": 2.0, "max_seconds": 2.0},
    }


def test_orphan_terminal_events_count_without_inventing_latency():
    metrics = MetricsEventSink(clock=lambda: 5.0)

    metrics.emit(_task(EventType.TASK_FAILED))
    metrics.emit(_llm(EventType.LLM_FAILED))
    metrics.emit(_tool(EventType.TOOL_EXECUTION_FAILED))

    snapshot = metrics.snapshot()
    assert snapshot["task_outcomes"] == {"failed": 1}
    assert all(summary["count"] == 0 for summary in snapshot["durations"].values())


def test_inflight_maps_are_bounded_and_evict_oldest_keys():
    ticks = itertools.count()
    metrics = MetricsEventSink(clock=lambda: float(next(ticks)), max_inflight=2)

    for task_id in ["one", "two", "three"]:
        metrics.emit(_task(EventType.TASK_STARTED, task_id))

    snapshot = metrics.snapshot()
    assert snapshot["in_flight"]["tasks"] == 2
    assert snapshot["inflight_evictions"] == {"tasks": 1, "llm": 0, "tools": 0}


def test_snapshot_is_detached_and_reset_clears_all_state():
    metrics = MetricsEventSink(clock=lambda: 1.0)
    metrics.emit(_task(EventType.TASK_STARTED))
    first = metrics.snapshot()
    first["events"]["task_started"] = 99

    assert metrics.snapshot()["events"]["task_started"] == 1
    metrics.reset()
    assert metrics.snapshot() == {
        "events": {},
        "task_outcomes": {},
        "llm_tokens": {"prompt": 0, "completion": 0, "total": 0},
        "in_flight": {"tasks": 0, "llm": 0, "tools": 0},
        "inflight_evictions": {"tasks": 0, "llm": 0, "tools": 0},
        "durations": {
            "task": {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0},
            "llm": {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0},
            "tool": {"count": 0, "total_seconds": 0.0, "max_seconds": 0.0},
        },
    }


def test_metrics_support_concurrent_emission_without_leaking_inflight_ids():
    metrics = MetricsEventSink(max_inflight=2000)

    def emit(index):
        task_id = f"task-{index}"
        metrics.emit(_task(EventType.TASK_STARTED, task_id))
        metrics.emit(_task(EventType.TASK_COMPLETED, task_id))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(emit, range(1000)))

    snapshot = metrics.snapshot()
    assert snapshot["events"] == {"task_completed": 1000, "task_started": 1000}
    assert snapshot["task_outcomes"] == {"completed": 1000}
    assert snapshot["in_flight"]["tasks"] == 0
    assert snapshot["durations"]["task"]["count"] == 1000


def test_composite_isolates_sink_failures_and_preserves_channel(caplog):
    received = []

    class Broken:
        def emit(self, event, channel=EventChannel.CLIENT):
            raise RuntimeError("offline")

    class Recording:
        def emit(self, event, channel=EventChannel.CLIENT):
            received.append((event, channel))

    composite = CompositeEventSink(Broken(), Recording())
    event = _task(EventType.TASK_STARTED)

    composite.emit(event, EventChannel.TRACING)

    assert received == [(event, EventChannel.TRACING)]
    assert composite.failure_count == 1
    assert "observability sink failed" in caplog.text


def test_composite_can_fail_fast_without_calling_later_sinks():
    received = []

    class Broken:
        def emit(self, event, channel=EventChannel.CLIENT):
            raise RuntimeError("offline")

    class Recording:
        def emit(self, event, channel=EventChannel.CLIENT):
            received.append(event)

    composite = CompositeEventSink(Broken(), Recording(), fail_fast=True)

    with pytest.raises(RuntimeError, match="offline"):
        composite.emit(_task(EventType.TASK_STARTED))

    assert received == []
