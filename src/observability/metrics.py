from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Hashable

from src.core.event import EventSink
from src.logger import get_logger
from src.schema.event import Event, EventChannel, EventType, LlmEventPayload, ToolEventPayload

logger = get_logger("observability")


@dataclass
class _Duration:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def observe(self, seconds: float) -> None:
        value = max(0.0, seconds)
        self.count += 1
        self.total_seconds += value
        self.max_seconds = max(self.max_seconds, value)

    def snapshot(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "total_seconds": self.total_seconds,
            "max_seconds": self.max_seconds,
        }


class MetricsEventSink:
    """Aggregate bounded, thread-safe in-process metrics from Harness events."""

    _TASK_TERMINALS = {
        EventType.TASK_COMPLETED: "completed",
        EventType.TASK_FAILED: "failed",
        EventType.TASK_PAUSED: "paused",
    }
    _LLM_TERMINALS = {EventType.LLM_COMPLETED, EventType.LLM_FAILED}
    _TOOL_STARTS = {EventType.TOOL_EXECUTION_STARTED, EventType.TOOL_EXECUTION_RESUMED}
    _TOOL_TERMINALS = {
        EventType.TOOL_EXECUTION_COMPLETED,
        EventType.TOOL_EXECUTION_FAILED,
        EventType.TOOL_EXECUTION_PAUSED,
    }

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_inflight: int = 10_000,
    ) -> None:
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight <= 0:
            raise ValueError("max_inflight must be a positive integer")
        self._clock = clock
        self._max_inflight = max_inflight
        self._lock = threading.RLock()
        self.reset()

    def emit(self, event: Event, channel: EventChannel = EventChannel.CLIENT) -> None:
        del channel
        now = float(self._clock())
        with self._lock:
            self._events[event.type.value] += 1
            self._observe_task(event, now)
            self._observe_llm(event, now)
            self._observe_tool(event, now)

    def _observe_task(self, event: Event, now: float) -> None:
        task_id = event.task_id or getattr(event.payload, "task_id", None)
        if event.type is EventType.TASK_STARTED and task_id:
            self._start(self._tasks, str(task_id), now, "tasks")
            return
        outcome = self._TASK_TERMINALS.get(event.type)
        if outcome is None:
            return
        self._outcomes[outcome] += 1
        if task_id:
            self._finish(self._tasks, str(task_id), now, self._durations["task"])

    def _observe_llm(self, event: Event, now: float) -> None:
        if not isinstance(event.payload, LlmEventPayload):
            return
        key = (event.task_id or "", event.payload.phase or "unknown")
        if event.type is EventType.LLM_STARTED:
            self._start(self._llm, key, now, "llm")
            return
        if event.type not in self._LLM_TERMINALS:
            return
        self._finish(self._llm, key, now, self._durations["llm"])
        usage = event.payload.usage
        if usage is not None:
            self._tokens["prompt"] += usage.prompt_tokens
            self._tokens["completion"] += usage.completion_tokens
            self._tokens["total"] += usage.total_tokens

    def _observe_tool(self, event: Event, now: float) -> None:
        if not isinstance(event.payload, ToolEventPayload):
            return
        key = (event.task_id or "", event.payload.tool_call_id)
        if event.type in self._TOOL_STARTS:
            self._start(self._tools, key, now, "tools")
        elif event.type in self._TOOL_TERMINALS:
            self._finish(self._tools, key, now, self._durations["tool"])

    def _start(
        self,
        active: OrderedDict[Hashable, float],
        key: Hashable,
        now: float,
        category: str,
    ) -> None:
        if key in active:
            active.pop(key)
        elif len(active) >= self._max_inflight:
            active.popitem(last=False)
            self._evictions[category] += 1
        active[key] = now

    @staticmethod
    def _finish(
        active: OrderedDict[Hashable, float],
        key: Hashable,
        now: float,
        duration: _Duration,
    ) -> None:
        started = active.pop(key, None)
        if started is not None:
            duration.observe(now - started)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "events": dict(sorted(self._events.items())),
                "task_outcomes": dict(sorted(self._outcomes.items())),
                "llm_tokens": dict(self._tokens),
                "in_flight": {
                    "tasks": len(self._tasks),
                    "llm": len(self._llm),
                    "tools": len(self._tools),
                },
                "inflight_evictions": dict(self._evictions),
                "durations": {
                    name: duration.snapshot() for name, duration in self._durations.items()
                },
            }

    def reset(self) -> None:
        with self._lock:
            self._events: Counter[str] = Counter()
            self._outcomes: Counter[str] = Counter()
            self._tokens = {"prompt": 0, "completion": 0, "total": 0}
            self._tasks: OrderedDict[Hashable, float] = OrderedDict()
            self._llm: OrderedDict[Hashable, float] = OrderedDict()
            self._tools: OrderedDict[Hashable, float] = OrderedDict()
            self._evictions = {"tasks": 0, "llm": 0, "tools": 0}
            self._durations = {
                "task": _Duration(),
                "llm": _Duration(),
                "tool": _Duration(),
            }


class CompositeEventSink:
    """Fan events out in order while optionally isolating observer failures."""

    def __init__(self, *sinks: EventSink, fail_fast: bool = False) -> None:
        self.sinks = tuple(sinks)
        self.fail_fast = fail_fast
        self.failure_count = 0
        self._lock = threading.Lock()

    def emit(self, event: Event, channel: EventChannel = EventChannel.CLIENT) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event, channel)
            except Exception:
                if self.fail_fast:
                    raise
                with self._lock:
                    self.failure_count += 1
                logger.exception(
                    "observability sink failed",
                    extra={
                        "context": {
                            "sink": type(sink).__name__,
                            "event_type": event.type.value,
                            "channel": channel.value,
                        }
                    },
                )
