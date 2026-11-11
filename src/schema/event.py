from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from src.schema.llm import LlmUsage


class EventType(Enum):
    TASK_STARTED = "task_started"
    TASK_PLANNING = "task_planning"
    TASK_PLANNED = "task_planned"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    TASK_FAILED = "task_failed"
    TASK_COMPLETED = "task_completed"
    TASK_FINISHED = "task_finished"

    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_EXECUTION_PAUSED = "tool_execution_paused"
    TOOL_EXECUTION_RESUMED = "tool_execution_resumed"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"

    LLM_STARTED = "llm_started"
    LLM_DELTA = "llm_delta"
    LLM_RETRYING = "llm_retrying"
    LLM_FAILED = "llm_failed"
    LLM_COMPLETED = "llm_completed"


@dataclass
class TaskEventPayload:
    task_id: str
    reason: str | None = None
    content: str | None = None
    goal: list[str] | None = None


@dataclass
class ToolEventPayload:
    tool_call_id: str
    name: str | None = None
    arguments: str | None = None
    reason: str | None = None
    content: str | None = None


@dataclass
class LlmEventPayload:
    reason: str | None = None
    usage: LlmUsage | None = None
    attempt: int | None = None
    phase: str | None = None
    content: str | None = None


@dataclass
class Event:
    """One fact emitted by the harness loop."""

    id: str
    type: EventType
    payload: TaskEventPayload | ToolEventPayload | LlmEventPayload
    timestamp: datetime = field(default_factory=datetime.now)
    task_id: str | None = None
    step_id: str | None = None


class EventChannel(Enum):
    CLIENT = "client"
    TRACING = "tracing"
    ALL = "all"
