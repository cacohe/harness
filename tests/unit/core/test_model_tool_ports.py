from dataclasses import FrozenInstanceError
from decimal import Inexact, InvalidOperation, localcontext

import pytest
from src.core.artifacts import MAX_SUMMARY_BYTES, ArtifactRef, InlineContent
from src.core.commands import ToolCallRequest
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import RunId, SessionId
from src.core.model import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRole,
    ModelToolCall,
    ModelUsage,
)
from src.core.tools import ToolApproval, ToolRequest, ToolResult, ToolResultStatus, ToolSpec


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


def _call(call_id):
    return ModelToolCall(call_id, "read_file", "{}")


def _assistant(*ids):
    return ModelMessage(ModelRole.ASSISTANT, tool_calls=tuple(map(_call, ids)))


def _result(call_id):
    return ModelMessage(ModelRole.TOOL, "Found it", tool_call_id=call_id)


def _request(**overrides):
    fields = {
        "tool_call_id": "call-1",
        "name": "read_file",
        "arguments_json": '{"path":"./docs/example.md"}',
        "run_id": RunId("run-1"),
        "session_id": SessionId("session-1"),
        "step_id": "step-1",
        "approval_scope": "workspace-read",
        "workspace_id": "workspace-1",
        "approval": ToolApproval.UNDECIDED,
    }
    fields.update(overrides)
    if fields["approval"] in (ToolApproval.APPROVED, ToolApproval.DENIED):
        fields.setdefault(
            "original_call",
            ToolCallRequest(
                fields["tool_call_id"],
                fields["name"],
                fields["arguments_json"],
                approval_scope=fields["approval_scope"],
                workspace_id=fields["workspace_id"],
            ),
        )
    return ToolRequest(**fields)


def _artifact():
    digest = "a" * 64
    return ArtifactRef("sha256:" + digest, "text/plain", 20, digest, metadata={"lines": 2})


def test_model_request_accepts_tool_results_in_completion_order():
    history = (
        ModelMessage(ModelRole.SYSTEM, "Use tools when needed."),
        ModelMessage(ModelRole.USER, "Read both files."),
        ModelMessage(ModelRole.ASSISTANT, "Checking.", tool_calls=(_call("one"), _call("two"))),
        _result("two"),
        _result("one"),
        ModelMessage(ModelRole.ASSISTANT, "Both files were checked."),
    )
    assert ModelRequest(history).messages == history


def test_model_request_accepts_unsettled_calls_at_end_of_history():
    for history in ((_assistant("one"),), (_assistant("one", "two"), _result("one"))):
        assert ModelRequest(history).messages == history


@pytest.mark.parametrize(
    "history",
    [
        (_result("one"), _assistant("one")),
        (_assistant("one"), _result("wrong")),
        (_assistant("one"), _result("one"), _result("one")),
        (_assistant("one"), _result("one"), _assistant("one")),
        (_assistant("one"), _assistant("two"), _result("one")),
        (_assistant("one"), ModelMessage(ModelRole.USER, "Wait"), _result("one")),
        (_assistant("one"), ModelMessage(ModelRole.SYSTEM, "Rules"), _result("one")),
    ],
)
def test_model_request_rejects_unmatched_repeated_and_interleaved_calls(history):
    with pytest.raises(ValueError):
        ModelRequest(history)


@pytest.mark.parametrize(
    "role, content, fields",
    [
        ("assistant", "raw role string", {}),
        (ModelRole.SYSTEM, "rules", {"tool_calls": (_call("one"),)}),
        (ModelRole.USER, "question", {"tool_call_id": "one"}),
        (ModelRole.USER, None, {}),
        (ModelRole.USER, {"provider_payload": True}, {}),
        (ModelRole.ASSISTANT, None, {}),
        (ModelRole.ASSISTANT, "answer", {"tool_call_id": "one"}),
        (ModelRole.TOOL, "result", {}),
        (ModelRole.TOOL, None, {"tool_call_id": "one"}),
        (ModelRole.TOOL, "result", {"tool_call_id": "one", "tool_calls": (_call("one"),)}),
    ],
)
def test_model_message_rejects_invalid_role_content_combinations(role, content, fields):
    with pytest.raises(ValueError):
        ModelMessage(role, content, **fields)


def test_model_message_preserves_an_explicit_empty_tool_result():
    assert ModelMessage(ModelRole.TOOL, "", tool_call_id="one").content == ""


def test_model_ports_bound_text_message_tool_and_call_counts():
    user = ModelMessage(ModelRole.USER, "question")
    with pytest.raises(ValueError):
        ModelMessage(ModelRole.USER, "界" * 400_000)
    with pytest.raises(ValueError):
        ModelResponse(text="界" * 400_000)
    with pytest.raises(ValueError):
        ModelRequest((user,) * 10_000)
    with pytest.raises(ValueError):
        ModelRequest(())
    with pytest.raises(ValueError):
        ModelRequest((user,), tools=tuple(ToolSpec(f"tool_{i}", "Read", "{}") for i in range(1000)))
    with pytest.raises(ValueError):
        ModelResponse(tool_calls=tuple(_call(f"call-{i}") for i in range(1000)))
    with pytest.raises(ValueError):
        _assistant(*(f"call-{i}" for i in range(1000)))


def test_model_request_bounds_aggregate_message_text():
    large_message = ModelMessage(ModelRole.USER, "a" * 1_000_000)
    with pytest.raises(ValueError):
        ModelRequest((large_message,) * 9)


def test_model_request_rejects_duplicate_tool_names():
    tools = (ToolSpec("read", "First", "{}"), ToolSpec("read", "Second", "{}"))
    with pytest.raises(ValueError):
        ModelRequest((ModelMessage(ModelRole.USER, "Read"),), tools=tools)


def test_model_response_supports_direct_text_calls_and_both():
    assert ModelResponse(text="Done").text == "Done"
    assert ModelResponse(tool_calls=(_call("one"),)).text is None
    response = ModelResponse(text="Checking", tool_calls=(_call("one"),), usage=ModelUsage(23, 7))
    assert response.tool_calls == (_call("one"),)
    assert response.text == "Checking"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (23, 7)
    with pytest.raises(FrozenInstanceError):
        response.text = "changed"


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_model_response_rejects_empty_output(text):
    with pytest.raises(ValueError):
        ModelResponse(text=text)


def test_assistant_message_and_response_reject_repeated_call_ids():
    with pytest.raises(ValueError):
        ModelResponse(tool_calls=(_call("one"), _call("one")))
    with pytest.raises(ValueError):
        _assistant("one", "one")


@pytest.mark.parametrize("count", [True, -1, 1.5, IntSubclass(1), 1 << 80])
def test_model_usage_requires_bounded_exact_nonnegative_counts(count):
    with pytest.raises(ValueError):
        ModelUsage(count, 0)
    with pytest.raises(ValueError):
        ModelUsage(0, count)


def test_model_response_rejects_opaque_usage_object():
    with pytest.raises(ValueError):
        ModelResponse(text="Done", usage={"prompt_tokens": 1})


def test_ports_preserve_json_text_and_keep_tool_input_paths():
    raw = '{ "path" : "./docs/readme.md", "credentials": {"z":null,"a":true} }'
    spec = ToolSpec(
        "read", "Read a file", '{"type":"object","properties":{"path":{"type":"string"}}}'
    )
    assert ModelToolCall("call-1", "read", raw).arguments_json == raw
    assert _request(arguments_json=raw).arguments_json == raw
    assert spec.input_schema_json == '{"type":"object","properties":{"path":{"type":"string"}}}'
    assert ToolSpec("read", "Any input", " true ").input_schema_json == " true "


@pytest.mark.parametrize(
    "bad_json",
    [
        '{"nested":{"x":1,"x":2}}',
        '{"value":NaN}',
        '{"value":Infinity}',
        r'{"value":"\ud800"}',
        r'{"\ud800":0}',
        '{"value":' + "[" * 20 + "0" + "]" * 20 + "}",
        '{"value":"' + "a" * 70_000 + '"}',
        '["not an object"]',
        StringSubclass("{}"),
        {"not": "a JSON string"},
    ],
)
def test_port_json_rejects_invalid_or_unbounded_input(bad_json):
    for construct in (
        lambda: ModelToolCall("call-1", "read", bad_json),
        lambda: _request(arguments_json=bad_json),
        lambda: ToolSpec("read", "Read", bad_json),
    ):
        with pytest.raises(ValueError):
            construct()


@pytest.mark.parametrize("bad_json", ["null", "true", "17", '"text"'])
def test_tool_arguments_require_a_json_object(bad_json):
    with pytest.raises(ValueError):
        ModelToolCall("call-1", "read", bad_json)
    with pytest.raises(ValueError):
        _request(arguments_json=bad_json)


@pytest.mark.parametrize("approval", list(ToolApproval))
def test_tool_request_keeps_scope_workspace_and_explicit_approval(approval):
    request = _request(approval=approval)
    assert request.run_id == RunId("run-1")
    assert request.session_id == SessionId("session-1")
    assert request.step_id == "step-1"
    assert request.tool_call_id == "call-1"
    assert request.name == "read_file"
    assert request.approval is approval
    assert request.approval_scope == "workspace-read"
    assert request.workspace_id == "workspace-1"


@pytest.mark.parametrize(
    "identity",
    [
        "..",
        "a/b",
        r"C:\private",
        r"\\host\share",
        "C:relative",
        "file:root",
        "two words",
        "work\x00",
    ],
)
def test_tool_request_rejects_workspace_locations(identity):
    with pytest.raises(ValueError):
        _request(workspace_id=identity)


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_id": SessionId("run-1")},
        {"session_id": "session-1"},
        {"step_id": ""},
        {"approval_scope": ""},
        {"approval": "approved"},
    ],
)
def test_tool_request_rejects_wrong_identity_types_and_untyped_approval(overrides):
    with pytest.raises(ValueError):
        _request(**overrides)


def test_tool_result_returns_copy_isolated_inline_and_artifact_content():
    inline = InlineContent({"status": "ok"}, "application/json")
    artifact = _artifact()
    inline_result = ToolResult(ToolResultStatus.SUCCESS, inline_content=inline)
    artifact_result = ToolResult(ToolResultStatus.SUCCESS, artifacts=(artifact,), summary="2 lines")
    object.__setattr__(inline, "_encoded", b'{"status":"NO"}')
    object.__setattr__(artifact, "size_bytes", 999)
    assert inline_result.inline_content.value == {"status": "ok"}
    assert inline_result.artifacts == ()
    assert artifact_result.inline_content is None
    assert artifact_result.summary == "2 lines"
    assert artifact_result.artifacts[0].size_bytes == 20
    object.__setattr__(artifact_result.artifacts[0], "size_bytes", 999)
    assert artifact_result.artifacts[0].size_bytes == 20


def test_tool_result_revalidates_nested_instances_at_construction():
    inline = InlineContent("OK", "application/json")
    object.__setattr__(inline, "size_bytes", -1)
    with pytest.raises(ValueError):
        ToolResult(ToolResultStatus.SUCCESS, inline_content=inline)
    artifact = _artifact()
    object.__setattr__(artifact, "size_bytes", -1)
    with pytest.raises(ValueError):
        ToolResult(ToolResultStatus.SUCCESS, artifacts=(artifact,), summary="Artifact")


def test_tool_result_accepts_explicit_inline_json_null():
    result = ToolResult(
        ToolResultStatus.SUCCESS, inline_content=InlineContent(None, "application/json")
    )
    assert result.inline_content.value is None


def test_tool_result_success_requires_exactly_one_usable_content_variant():
    inline = InlineContent("OK", "application/json")
    for fields in (
        {},
        {"summary": "No refs"},
        {"artifacts": (_artifact(),)},
        {"artifacts": (_artifact(),), "summary": ""},
        {"artifacts": (_artifact(),), "summary": "界" * (MAX_SUMMARY_BYTES // 3 + 1)},
        {"artifacts": (_artifact(),), "summary": "Refs", "inline_content": inline},
        {"artifacts": [_artifact()], "summary": "Refs"},
        {"artifacts": (_artifact(),) * 17, "summary": "Refs"},
        {"inline_content": inline, "summary": "Inline"},
        {"inline_content": inline, "failure": RunFailure(RunErrorCode.TOOL_FAILED)},
    ):
        with pytest.raises(ValueError):
            ToolResult(ToolResultStatus.SUCCESS, **fields)


def test_pending_approval_is_payload_free():
    pending = ToolResult(ToolResultStatus.PENDING_APPROVAL)
    assert pending.inline_content is pending.summary is pending.failure is None
    assert pending.artifacts == ()
    for fields in (
        {"inline_content": InlineContent("secret", "text/plain")},
        {"artifacts": (_artifact(),)},
        {"summary": "secret"},
        {"failure": RunFailure(RunErrorCode.TOOL_FAILED)},
    ):
        with pytest.raises(ValueError):
            ToolResult(ToolResultStatus.PENDING_APPROVAL, **fields)


@pytest.mark.parametrize(
    "status, code, message",
    [
        (ToolResultStatus.FAILED, RunErrorCode.TOOL_FAILED, "Tool call failed."),
        (ToolResultStatus.TIMED_OUT, RunErrorCode.TOOL_TIMED_OUT, "Tool call timed out."),
        (ToolResultStatus.CANCELLED, RunErrorCode.CANCELLED, "The run was cancelled."),
    ],
)
def test_tool_failure_exposes_only_the_stable_code_and_message(status, code, message):
    failure = RunFailure(code)
    object.__setattr__(failure, "message", r"Traceback: C:\private\tool.py")
    outcome = ToolResult(status, failure=failure)
    assert outcome.failure == RunFailure(code)
    assert outcome.failure.message == message
    assert outcome.inline_content is outcome.summary is None
    assert outcome.artifacts == ()
    assert "Traceback" not in repr(outcome)


@pytest.mark.parametrize(
    "status, failure",
    [
        (ToolResultStatus.FAILED, None),
        (ToolResultStatus.FAILED, RuntimeError("Traceback with secret")),
        (ToolResultStatus.FAILED, RunFailure(RunErrorCode.TOOL_TIMED_OUT)),
        (ToolResultStatus.TIMED_OUT, RunFailure(RunErrorCode.TOOL_FAILED)),
        (ToolResultStatus.CANCELLED, RunFailure(RunErrorCode.TOOL_TIMED_OUT)),
    ],
)
def test_tool_failure_rejects_exception_objects_and_mismatched_codes(status, failure):
    with pytest.raises(ValueError):
        ToolResult(status, failure=failure)


def test_tool_failure_rejects_execution_content():
    with pytest.raises(ValueError):
        ToolResult(
            ToolResultStatus.FAILED,
            inline_content=InlineContent("secret", "text/plain"),
            failure=RunFailure(RunErrorCode.TOOL_FAILED),
        )


@pytest.mark.parametrize("number", ["9007199254740993.0", "1e-999", "1e999"])
def test_argument_and_schema_snapshots_preserve_exact_json_numbers(number):
    arguments = '{"value":' + number + "}"
    schema = '{"multipleOf":' + number + "}"
    assert ModelToolCall("one", "read", arguments).arguments_json == arguments
    assert _request(arguments_json=arguments).arguments_json == arguments
    assert ToolSpec("read", "Read", schema).input_schema_json == schema


@pytest.mark.parametrize(
    "inline",
    [
        InlineContent(b"\x00\xff", "application/octet-stream"),
        InlineContent("text", "text/plain"),
        InlineContent({"n": 1}, "application/example+json"),
    ],
)
def test_tool_result_rejects_inline_content_outside_application_json(inline):
    with pytest.raises(ValueError):
        ToolResult(ToolResultStatus.SUCCESS, inline_content=inline)


@pytest.mark.parametrize(
    "decision, executable, requires_approval",
    [
        (ToolApproval.UNDECIDED, False, True),
        (ToolApproval.DENIED, False, False),
        (ToolApproval.APPROVED, True, False),
        (ToolApproval.NOT_REQUIRED, True, False),
    ],
)
def test_tool_request_exposes_execution_and_pending_approval_guards(
    decision,
    executable,
    requires_approval,
):
    request = _request(approval=decision)
    assert request.is_executable is executable
    assert request.requires_approval is requires_approval


def test_denied_result_carries_no_payload():
    status = ToolResultStatus.DENIED
    result = ToolResult(status)
    assert result.failure is result.inline_content is result.summary is None
    assert result.artifacts == ()
    for payload in (
        {"inline_content": InlineContent(None, "application/json")},
        {"artifacts": (_artifact(),)},
        {"summary": "secret"},
        {"failure": RunFailure(RunErrorCode.TOOL_FAILED)},
    ):
        with pytest.raises(ValueError):
            ToolResult(status, **payload)


def test_model_request_counts_raw_arguments_and_schema_in_aggregate_utf8_limit():
    arguments = ' {"x":"' + "a" * 16_000 + '"}'
    schema = ' {"description":"' + "b" * 64_000 + '"}'
    call = ModelToolCall("one", "read_file", arguments)
    batch = ModelMessage(ModelRole.ASSISTANT, tool_calls=(call,))
    spec = ToolSpec("read_file", "Read", schema)
    full = ModelMessage(ModelRole.USER, "x" * 1_048_576)
    header_bytes = 8 * len("user") + len("assistantoneread_fileread_fileRead")
    tail_size = 8_388_608 - 7 * 1_048_576 - header_bytes - len(arguments) - len(schema)
    head = (full,) * 7
    exact = ModelMessage(ModelRole.USER, "x" * tail_size)
    over = ModelMessage(ModelRole.USER, "x" * (tail_size + 1))
    assert ModelRequest((*head, exact, batch), (spec,)).messages[-1] == batch
    with pytest.raises(ValueError):
        ModelRequest((*head, over, batch), (spec,))


def test_decided_approval_cannot_be_constructed_without_original_call():
    for decision in (ToolApproval.APPROVED, ToolApproval.DENIED):
        with pytest.raises(ValueError):
            _request(approval=decision, original_call=None)


@pytest.mark.parametrize("trap", [False, True])
def test_raw_json_numbers_ignore_decimal_context(trap):
    with localcontext() as context:
        context.traps[InvalidOperation] = trap
        context.flags[Inexact] = True
        saved_flags, saved_traps = dict(context.flags), dict(context.traps)
        for token in (
            "1e-10000000000000000000",
            "1e10000000000000000000",
            "9007199254740993.00000000000000000001",
            "9" * 4501,
        ):
            arguments = '{"number":' + token + "}"
            schema = '{"multipleOf":' + token + "}"
            assert ModelToolCall("one", "read", arguments).arguments_json == arguments
            assert _request(arguments_json=arguments).arguments_json == arguments
            assert ToolSpec("read", "Read", schema).input_schema_json == schema
        for token in ("NaN", "Infinity", "-Infinity", "01", "1e", "1."):
            with pytest.raises(ValueError):
                ModelToolCall("one", "read", '{"number":' + token + "}")
        assert dict(context.flags) == saved_flags
        assert dict(context.traps) == saved_traps
