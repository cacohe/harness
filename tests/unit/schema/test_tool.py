from src.schema.tool import (
    Tool,
    ToolExecution,
    ToolExecutionFailReason,
    ToolExecutionResult,
    ToolOrigin,
)
from src.workspace import Workspace


def test_tool_execution_treats_workspace_capability_as_an_opaque_runtime_type(tmp_path):
    workspace = Workspace(tmp_path)
    tool = Tool(
        id="read_file",
        name="read_file",
        description="Read a file.",
        input_schema={},
        executer=lambda inputs: "content",
        workspace=workspace,
    )

    execution = ToolExecution(tool=tool, inputs={"path": "README.md"})

    assert execution.tool is tool
    assert execution.tool.workspace is workspace


def test_tool_metadata_defaults_to_custom_without_forced_approval():
    tool = Tool("custom", "custom", "Custom.", {}, lambda inputs: "ok")

    assert tool.origin is ToolOrigin.CUSTOM
    assert tool.requires_approval is False


def test_remote_error_reason_round_trips_through_a_snapshot():
    result = ToolExecutionResult(
        tool_id="remote",
        tool_name="remote",
        state="failed",
        output={"is_error": True},
        fail_reason=ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REMOTE_ERROR,
    )

    restored = ToolExecutionResult.model_validate(result.model_dump(mode="json"))

    assert restored.fail_reason is ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REMOTE_ERROR
    assert restored.output == {"is_error": True}
