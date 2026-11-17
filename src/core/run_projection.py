import json

from src.core.artifacts import ExternalizedContent
from src.core.errors import RunErrorCode, RunFailure
from src.core.journal import TranscriptEntry
from src.core.state import RunState, ToolCallState
from src.core.tools import ToolResult, ToolResultStatus


def tool_failure(result: ToolResult) -> RunFailure:
    if result.status is ToolResultStatus.DENIED:
        return RunFailure(RunErrorCode.TOOL_FAILED)
    if result.status is ToolResultStatus.PENDING_APPROVAL:
        return RunFailure(RunErrorCode.INVARIANT_VIOLATION)
    failure = result.failure
    return failure if failure is not None else RunFailure(RunErrorCode.TOOL_FAILED)


def tool_transcript(state: RunState) -> tuple[TranscriptEntry, ...]:
    step = state.current_step
    assert step is not None and step.model_call is not None
    calls = [
        {"id": item.tool_call_id, "name": item.name, "arguments": item.arguments_json}
        for item in step.model_call.tool_requests
    ]
    entries = [
        TranscriptEntry(
            state.session_id,
            state.run_id,
            state.submission_id,
            "assistant",
            json.dumps(
                {"text": step.model_call.result, "tool_calls": calls}, separators=(",", ":")
            ),
        )
    ]
    entries.extend(
        TranscriptEntry(
            state.session_id,
            state.run_id,
            state.submission_id,
            "tool",
            _tool_text(call),
        )
        for call in step.tool_calls
    )
    return tuple(entries)


def _tool_text(call: ToolCallState) -> str:
    if call.approval is not None and call.approval.approved is False:
        return json.dumps({"tool_call_id": call.tool_call_id, "status": "denied"})
    if type(call.result) is RunFailure:
        return json.dumps(
            {"tool_call_id": call.tool_call_id, "error": call.result.code.value},
            separators=(",", ":"),
        )
    assert type(call.result) is ExternalizedContent
    inline = call.result.inline_content
    if inline is not None:
        return json.dumps(
            {"tool_call_id": call.tool_call_id, "result": inline.value},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return json.dumps(
        {
            "tool_call_id": call.tool_call_id,
            "summary": call.result.summary,
            "artifacts": [ref.artifact_id for ref in call.result.artifacts],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
