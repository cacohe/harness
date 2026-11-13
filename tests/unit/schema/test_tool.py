from src.core.workspace import Workspace
from src.schema.tool import Tool, ToolExecution


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
