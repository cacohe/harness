from datetime import datetime

from src.core.event import EventHandler
from src.schema.event import (
    Event,
    EventChannel,
    EventType,
    LlmEventPayload,
    TaskEventPayload,
    ToolEventPayload,
)


def _event() -> Event:
    return Event(
        id="event-1",
        type=EventType.TASK_COMPLETED,
        payload=TaskEventPayload(task_id="task-1", content="bought"),
        timestamp=datetime(2026, 11, 10, 16, 0, 0),
        task_id="task-1",
    )


def test_default_emitters_write_client_and_tracing_to_stdout(capsys):
    EventHandler().emit(_event(), EventChannel.ALL)

    output = capsys.readouterr().out
    assert "client task_completed" in output
    assert "tracing 2026-11-10T16:00:00" in output
    assert "type=task_completed" in output
    assert "task_id=task-1" in output
    assert "content=bought" in output


def test_llm_delta_is_written_without_a_newline(capsys):
    EventHandler().emit(
        Event(
            id="delta-1",
            type=EventType.LLM_DELTA,
            payload=LlmEventPayload(llm_id="delta-1", content="hel"),
        ),
        EventChannel.CLIENT,
    )

    assert capsys.readouterr().out == "hel"


def test_tool_event_shows_the_tool_name(capsys):
    EventHandler().emit(
        Event(
            id="tool-1",
            type=EventType.TOOL_EXECUTION_COMPLETED,
            payload=ToolEventPayload(
                tool_call_id="call_1",
                name="shell",
                arguments='{"command": "dir"}',
                content="listed",
            ),
        ),
        EventChannel.CLIENT,
    )

    output = capsys.readouterr().out
    assert 'client tool_execution_completed shell {"command": "dir"}' in output
    assert "listed" not in output


def test_channel_selects_one_emitter(capsys):
    EventHandler().emit(_event(), EventChannel.CLIENT)

    output = capsys.readouterr().out
    assert "client task_completed" in output
    assert "tracing " not in output
