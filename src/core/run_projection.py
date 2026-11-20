import hashlib
import json
from dataclasses import replace

from src.core.artifacts import ExternalizedContent
from src.core.commands import ToolCallRequest
from src.core.context import ContextRequest, summary_system_messages
from src.core.errors import RunErrorCode, RunFailure
from src.core.journal import TranscriptEntry, TranscriptEntryType
from src.core.model import ModelMessage, ModelResponse, ModelRole
from src.core.state import RunState, ToolCallState
from src.core.tools import ToolResult, ToolResultStatus, ToolSpec


def matches_trusted_context_shape(
    messages: tuple[ModelMessage, ...], request: ContextRequest
) -> bool:
    if not messages or messages[0] != request.system_message:
        return False
    prefix_end = 1
    while prefix_end < len(messages) and messages[prefix_end].role is ModelRole.SYSTEM:
        prefix_end += 1
    supplied_prefix = messages[1:prefix_end]
    trusted_groups = (
        *((message,) for message in request.memory_messages),
        *((message,) for message in request.skill_messages),
        *(summary_system_messages(checkpoint) for checkpoint in request.summary_checkpoints),
    )
    positions = {0}
    for group in trusted_groups:
        next_positions = set(positions)
        for position in positions:
            through = position + len(group)
            if supplied_prefix[position:through] == group:
                next_positions.add(through)
        positions = next_positions
    if len(supplied_prefix) not in positions:
        return False

    suffix_length = len(request.open_tool_messages) + 1
    if len(messages) < prefix_end + suffix_length:
        return False
    suffix = messages[-suffix_length:]
    if suffix[0] != request.current_input:
        return False
    if suffix[1:] != request.open_tool_messages:
        return False
    history = messages[prefix_end:-suffix_length]
    return all(message.role is not ModelRole.SYSTEM for message in history)


def model_tool_requests(
    response: ModelResponse, specs: tuple[ToolSpec, ...]
) -> tuple[ToolCallRequest, ...]:
    catalog = {spec.name: spec for spec in specs}
    requests = []
    for call in response.tool_calls:
        spec = catalog.get(call.name)
        if spec is None:
            requests.append(
                ToolCallRequest(
                    call.tool_call_id,
                    call.name,
                    call.arguments_json,
                    requires_approval=False,
                    approval_scope="unavailable",
                    workspace_id="unavailable",
                    available=False,
                )
            )
            continue
        requests.append(
            ToolCallRequest(
                call.tool_call_id,
                call.name,
                call.arguments_json,
                requires_approval=spec.requires_approval,
                approval_scope=spec.approval_scope,
                workspace_id=spec.workspace_id,
            )
        )
    return tuple(requests)


def scope_reused_tool_call_ids(
    requests: tuple[ToolCallRequest, ...],
    used: frozenset[str],
    run_id: str,
    step_id: str,
) -> tuple[ToolCallRequest, ...]:
    if not any(request.tool_call_id in used for request in requests):
        return requests
    occupied = set(used)
    scoped = []
    for request in requests:
        identifier = request.tool_call_id
        if identifier in occupied:
            nonce = 0
            while True:
                digest = hashlib.sha256(
                    f"{run_id}\0{step_id}\0{identifier}\0{nonce}".encode("utf-8")
                ).hexdigest()
                candidate = f"call-{digest}"
                if candidate not in occupied:
                    identifier = candidate
                    break
                nonce += 1
        occupied.add(identifier)
        scoped.append(replace(request, tool_call_id=identifier))
    return tuple(scoped)


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
            entry_type=TranscriptEntryType.ASSISTANT_TOOL_BATCH,
        )
    ]
    entries.extend(
        TranscriptEntry(
            state.session_id,
            state.run_id,
            state.submission_id,
            "tool",
            _tool_text(call),
            entry_type=TranscriptEntryType.TOOL_RESULT,
        )
        for call in step.tool_calls
    )
    return tuple(entries)


def _tool_text(call: ToolCallState) -> str:
    if call.approval is not None and call.approval.approved is False:
        return json.dumps({"tool_call_id": call.tool_call_id, "status": "denied"})
    if type(call.result) is RunFailure:
        payload = {"tool_call_id": call.tool_call_id, "error": call.result.code.value}
        if call.result.detail is not None:
            payload["detail"] = call.result.detail
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
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
