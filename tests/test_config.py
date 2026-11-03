from harness import (
    DEFAULT_SYSTEM_PROMPT,
    AgentKind,
    McpDependency,
    ModelConfig,
    OutputMode,
    OutputPhase,
    ToolDefinition,
    ToolSchedule,
    resolve_config,
    resolve_output_mode,
)

from support import config


def test_defaults_prompt_schedule_and_derived_output_mode():
    resolved = resolve_config(config())

    assert resolved.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert resolved.tool_schedule.concurrency == "serial"
    assert resolved.tool_schedule.completion == "sync"
    assert resolved.output_mode is None
    assert resolve_output_mode(resolved, OutputPhase.EXECUTION) is OutputMode.BUFFERED
    assert resolve_output_mode(resolved, OutputPhase.FINAL) is OutputMode.STREAM
    assert resolve_output_mode(resolved, OutputPhase.EVENT) is OutputMode.STREAM


def test_injected_prompt_schedule_tools_and_output_mode_override_defaults():
    tool = ToolDefinition(name="echo", description="say it back")
    mcp = McpDependency(server="files", tool="read")
    resolved = resolve_config(
        config(
            agent_kind=AgentKind.CONVERSATIONAL,
            model=ModelConfig(provider="local", model="small", options={"temperature": 0}),
            system_prompt="be terse",
            tools=(tool,),
            mcp=(mcp,),
            tool_schedule=ToolSchedule(concurrency="parallel"),
            output_mode=OutputMode.BUFFERED,
        )
    )

    assert resolved.system_prompt == "be terse"
    assert resolved.tools == (tool,)
    assert resolved.mcp == (mcp,)
    assert resolved.model.options["temperature"] == 0
    assert resolved.tool_schedule.concurrency == "parallel"
    assert resolved.tool_schedule.completion == "sync"
    assert resolve_output_mode(resolved, OutputPhase.EXECUTION) is OutputMode.BUFFERED
    assert resolve_output_mode(resolved, OutputPhase.FINAL) is OutputMode.BUFFERED


def test_conversational_agent_streams_every_phase():
    resolved = resolve_config(config(agent_kind=AgentKind.CONVERSATIONAL))

    assert resolve_output_mode(resolved, OutputPhase.EXECUTION) is OutputMode.STREAM
    assert resolve_output_mode(resolved, OutputPhase.FINAL) is OutputMode.STREAM
    assert resolve_output_mode(resolved, OutputPhase.EVENT) is OutputMode.STREAM
