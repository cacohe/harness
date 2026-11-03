from __future__ import annotations

from dataclasses import dataclass

from harness.types import (
    AgentKind,
    McpDependency,
    ModelConfig,
    OutputMode,
    OutputPhase,
    PermissionPolicy,
    ToolDefinition,
    ToolSchedule,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a task-focused agent. Use tools when needed and stop when the task is done."
)


@dataclass(frozen=True)
class HarnessConfig:
    agent_kind: AgentKind
    model: ModelConfig
    tools: tuple[ToolDefinition, ...] = ()
    mcp: tuple[McpDependency, ...] = ()
    permission_policy: PermissionPolicy = PermissionPolicy()
    system_prompt: str | None = None
    tool_schedule: ToolSchedule | None = None
    output_mode: OutputMode | None = None
    max_steps: int = 8
    execution_kind: str = "current"


@dataclass(frozen=True)
class ResolvedConfig:
    agent_kind: AgentKind
    model: ModelConfig
    tools: tuple[ToolDefinition, ...]
    mcp: tuple[McpDependency, ...]
    permission_policy: PermissionPolicy
    system_prompt: str
    tool_schedule: ToolSchedule
    output_mode: OutputMode | None
    max_steps: int
    execution_kind: str


def resolve_config(config: HarnessConfig) -> ResolvedConfig:
    prompt = DEFAULT_SYSTEM_PROMPT if config.system_prompt is None else config.system_prompt
    return ResolvedConfig(
        agent_kind=config.agent_kind,
        model=config.model,
        tools=tuple(config.tools),
        mcp=tuple(config.mcp),
        permission_policy=config.permission_policy,
        system_prompt=prompt,
        tool_schedule=config.tool_schedule or ToolSchedule(),
        output_mode=config.output_mode,
        max_steps=config.max_steps,
        execution_kind=config.execution_kind,
    )


def resolve_output_mode(config: ResolvedConfig, phase: OutputPhase) -> OutputMode:
    if config.output_mode is not None:
        return config.output_mode
    if config.agent_kind is AgentKind.CONVERSATIONAL:
        return OutputMode.STREAM
    if phase in (OutputPhase.FINAL, OutputPhase.EVENT):
        return OutputMode.STREAM
    return OutputMode.BUFFERED
