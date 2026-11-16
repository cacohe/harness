import ast
import asyncio

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from src.application.agent import Agent
from src.config import Config
from src.core.tool import RemoteToolError
from src.schema.llm import FinishReason, LLMOutput, LlmToolCall
from src.schema.task import TaskPlan, TaskState


class MCPGateway:
    def __init__(self):
        self.infer_contexts = []

    async def aplanning(self, context, stream=True, response_format=None):
        return LLMOutput(
            FinishReason.STOP,
            None,
            [],
            None,
            None,
            structured=TaskPlan(goal=["add two numbers"]),
        )

    async def ainfer(self, context, stream=True, response_format=None):
        self.infer_contexts.append(context)
        if len(self.infer_contexts) == 1:
            return LLMOutput(
                FinishReason.TOOL_CALLS,
                None,
                [LlmToolCall("call-1", "mcp__math__add", '{"left":2,"right":3}')],
                None,
                None,
            )
        return LLMOutput(FinishReason.STOP, "The sum is 5.", [], None, None)


def test_agent_runs_real_mcp_tool_through_approval_and_resume():
    async def scenario():
        server = FastMCP("calculator")

        @server.tool
        def add(left: int, right: int) -> dict[str, int]:
            return {"sum": left + right}

        gateway = MCPGateway()
        async with Agent(
            config=Config(llm="test-model", max_steps=4, max_seconds=30),
            llm_gateway=gateway,
        ) as agent:
            catalog = agent.tool_registry
            await agent.add_mcp_server("math", server)
            assert catalog.find_tool("mcp__math__add") is not None

            paused = await agent.achat("Add 2 and 3")
            assert paused.state is TaskState.TASK_PAUSED
            [request] = agent.approval_requests()
            assert request.tool_name == "mcp__math__add"

            agent.decide_tool_call(request.tool_id, approved=True)
            completed = await agent.aresume()

            assert completed.state is TaskState.TASK_COMPLETED
            assert completed.result.output == "The sum is 5."
            tool_messages = [
                message
                for message in completed.history
                if message.role == "tool" and message.tool_call_id == "call-1"
            ]
            [tool_message] = tool_messages
            envelope = ast.literal_eval(tool_message.content)
            assert set(envelope) == {"content", "structured_content", "is_error"}
            assert envelope["structured_content"] == {"sum": 5}
            assert envelope["is_error"] is False

        assert catalog.find_tool("mcp__math__add") is None

    asyncio.run(scenario())


def test_real_mcp_error_preserves_the_remote_result_envelope():
    async def scenario():
        server = FastMCP("failing")

        @server.tool
        def reject() -> None:
            raise ToolError("rejected by server")

        async with Agent(
            config=Config(llm="test-model", max_steps=3, max_seconds=30),
            llm_gateway=MCPGateway(),
        ) as agent:
            await agent.add_mcp_server("failure", server)
            tool = agent.tool_registry.find_tool("mcp__failure__reject")
            assert tool is not None

            with pytest.raises(RemoteToolError) as caught:
                await tool.executer({})

            assert caught.value.output["is_error"] is True
            assert caught.value.output["content"]
            assert "rejected by server" in str(caught.value.output["content"])
            assert "MCP transport error" not in str(caught.value.output)

    asyncio.run(scenario())
