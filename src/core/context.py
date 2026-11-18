from collections.abc import Awaitable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from src.core.artifacts import ArtifactRef
from src.core.errors import RunErrorCode, RunFailure
from src.core.ids import ProfileId, SessionId
from src.core.journal import ConversationSummaryCheckpoint
from src.core.model import ModelMessage, ModelRequest, ModelRole
from src.core.tools import ToolSpec

_MAX_TOKEN_COUNT = (1 << 63) - 1
_MAX_OPTIONAL_MESSAGES = 128
_MAX_INTERACTION_MESSAGES = 128
_MAX_POLICY_BYTES = 256


class ContextBudgetExceeded(ValueError):
    def __init__(self) -> None:
        self.failure = RunFailure(RunErrorCode.CONTEXT_BUDGET_EXCEEDED)
        super().__init__(self.failure.message)


class EstimateQuality(StrEnum):
    EXACT = "exact"
    CONSERVATIVE = "conservative"


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_window_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int

    def __post_init__(self) -> None:
        values = (
            self.context_window_tokens,
            self.reserved_output_tokens,
            self.safety_margin_tokens,
        )
        if (
            any(type(value) is not int or not 0 <= value <= _MAX_TOKEN_COUNT for value in values)
            or self.context_window_tokens == 0
            or self.reserved_output_tokens + self.safety_margin_tokens >= self.context_window_tokens
        ):
            raise ValueError("Context budget requires a positive input allowance")

    @property
    def input_allowance_tokens(self) -> int:
        return self.context_window_tokens - self.reserved_output_tokens - self.safety_margin_tokens


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    tokens: int
    quality: EstimateQuality

    def __post_init__(self) -> None:
        if type(self.tokens) is not int or not 0 <= self.tokens <= _MAX_TOKEN_COUNT:
            raise ValueError("Token estimate requires a nonnegative integer count")
        if type(self.quality) is not EstimateQuality:
            raise ValueError("Token estimate requires an explicit quality")


@dataclass(frozen=True, slots=True)
class OmittedContextRef:
    source: str
    first_sequence: int
    last_sequence: int
    item_count: int

    def __post_init__(self) -> None:
        if (
            type(self.source) is not str
            or not self.source
            or len(self.source.encode("utf-8")) > 64
            or type(self.first_sequence) is not int
            or type(self.last_sequence) is not int
            or not 0 <= self.first_sequence <= self.last_sequence <= _MAX_TOKEN_COUNT
            or type(self.item_count) is not int
            or not 1 <= self.item_count <= _MAX_TOKEN_COUNT
        ):
            raise ValueError("Omitted context reference is invalid")


@dataclass(frozen=True, slots=True)
class ConversationCoverageSegment:
    """A fully inspected sequence interval and its transcript item count."""

    first_sequence: int
    last_sequence: int
    item_count: int

    def __post_init__(self) -> None:
        if (
            type(self.first_sequence) is not int
            or type(self.last_sequence) is not int
            or not 1 <= self.first_sequence <= self.last_sequence <= _MAX_TOKEN_COUNT
            or type(self.item_count) is not int
            or not 0 <= self.item_count <= self.last_sequence - self.first_sequence + 1
        ):
            raise ValueError("Conversation coverage segment is invalid")


@dataclass(frozen=True, slots=True)
class ConversationCoverage:
    """Reader witness for all sequences before a bound, optionally excluding one interval."""

    before_sequence: int
    excluded_from_sequence: int | None
    excluded_through_sequence: int | None
    segments: tuple[ConversationCoverageSegment, ...]

    def __post_init__(self) -> None:
        if (
            type(self.before_sequence) is not int
            or not 1 <= self.before_sequence <= _MAX_TOKEN_COUNT
            or (self.excluded_from_sequence is None) != (self.excluded_through_sequence is None)
            or self.excluded_from_sequence is not None
            and (
                type(self.excluded_from_sequence) is not int
                or type(self.excluded_through_sequence) is not int
                or not 1 <= self.excluded_from_sequence <= self.excluded_through_sequence
            )
            or type(self.segments) is not tuple
            or any(type(segment) is not ConversationCoverageSegment for segment in self.segments)
        ):
            raise ValueError("Conversation coverage witness is invalid")
        snapshot = tuple(
            ConversationCoverageSegment(
                segment.first_sequence,
                segment.last_sequence,
                segment.item_count,
            )
            for segment in self.segments
        )
        if any(
            newer.first_sequence <= older.last_sequence
            for newer, older in zip(snapshot, snapshot[1:])
        ):
            raise ValueError("Conversation coverage segments must be newest first")
        object.__setattr__(self, "segments", snapshot)


@dataclass(frozen=True, slots=True)
class ConversationInteraction:
    """One indivisible, settled transcript interaction."""

    sequence_from: int
    sequence_through: int
    messages: tuple[ModelMessage, ...]

    def __post_init__(self) -> None:
        if (
            type(self.sequence_from) is not int
            or type(self.sequence_through) is not int
            or not 1 <= self.sequence_from <= self.sequence_through <= _MAX_TOKEN_COUNT
            or type(self.messages) is not tuple
            or not 1 <= len(self.messages) <= _MAX_INTERACTION_MESSAGES
            or any(type(message) is not ModelMessage for message in self.messages)
        ):
            raise ValueError("Conversation interaction is invalid")
        messages = tuple(replace(message) for message in self.messages)
        # A following user turn proves that no tool-call batch is left half paired.
        ModelRequest(messages + (ModelMessage(ModelRole.USER, "context-boundary"),))
        object.__setattr__(self, "messages", messages)


@dataclass(frozen=True, slots=True)
class ConversationPage:
    """A newest-first page and the exclusive cursor for the next older page."""

    interactions: tuple[ConversationInteraction, ...]
    next_before_sequence: int | None
    remaining_omission: OmittedContextRef | None = None

    def __post_init__(self) -> None:
        if (
            type(self.interactions) is not tuple
            or any(type(item) is not ConversationInteraction for item in self.interactions)
            or self.next_before_sequence is not None
            and (
                type(self.next_before_sequence) is not int
                or self.next_before_sequence <= 0
                or not self.interactions
                or self.next_before_sequence != self.interactions[-1].sequence_from
            )
        ):
            raise ValueError("Conversation page is invalid")
        remaining = self.remaining_omission
        if remaining is not None and (
            type(remaining) is not OmittedContextRef
            or remaining.source != "transcript"
            or self.next_before_sequence is None
            or not self.interactions
            or remaining.last_sequence >= self.next_before_sequence
            or remaining.item_count > remaining.last_sequence - remaining.first_sequence + 1
        ):
            raise ValueError("Conversation page remaining omission is invalid")
        snapshot = tuple(
            ConversationInteraction(item.sequence_from, item.sequence_through, item.messages)
            for item in self.interactions
        )
        if any(
            newer.sequence_from <= older.sequence_through
            for newer, older in zip(snapshot, snapshot[1:])
        ):
            raise ValueError("Conversation page must be non-overlapping and newest first")
        object.__setattr__(self, "interactions", snapshot)
        if remaining is not None:
            object.__setattr__(
                self,
                "remaining_omission",
                OmittedContextRef(
                    remaining.source,
                    remaining.first_sequence,
                    remaining.last_sequence,
                    remaining.item_count,
                ),
            )


def _policy(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or len(value.encode("utf-8")) > _MAX_POLICY_BYTES
    ):
        raise ValueError("Context policy version is invalid")
    return value


def snapshot_summary_checkpoint(
    value: ConversationSummaryCheckpoint | None,
    *,
    session_id: SessionId | None = None,
    profile_id: ProfileId | None = None,
    policy_version: str | None = None,
) -> ConversationSummaryCheckpoint | None:
    if value is None:
        return None
    try:
        if type(value) is not ConversationSummaryCheckpoint:
            raise ValueError
        raw_session = object.__getattribute__(value, "session_id")
        raw_through = object.__getattribute__(value, "covered_through_sequence")
        raw_policy = object.__getattribute__(value, "policy_version")
        raw_profile = object.__getattribute__(value, "profile_id")
        raw_summary = object.__getattribute__(value, "summary")
        raw_from = object.__getattribute__(value, "covered_from_sequence")
        raw_artifacts = object.__getattribute__(value, "artifacts")
        if (
            type(raw_session) is not SessionId
            or type(raw_profile) is not ProfileId
            or type(raw_through) is not int
            or type(raw_from) is not int
            or type(raw_policy) is not str
            or type(raw_summary) is not str
            or type(raw_artifacts) is not tuple
            or any(type(ref) is not ArtifactRef for ref in raw_artifacts)
        ):
            raise ValueError
        result = ConversationSummaryCheckpoint(
            SessionId(object.__getattribute__(raw_session, "value")),
            raw_through,
            _policy(raw_policy),
            ProfileId(object.__getattribute__(raw_profile, "value")),
            raw_summary,
            raw_from,
            raw_artifacts,
        )
        if (
            type(result.covered_from_sequence) is not int
            or type(result.covered_through_sequence) is not int
            or not 1
            <= result.covered_from_sequence
            <= result.covered_through_sequence
            <= _MAX_TOKEN_COUNT
            or type(result.summary) is not str
            or not result.summary.strip()
            or len(result.summary.encode("utf-8")) > 1_048_576
            or session_id is not None
            and result.session_id != session_id
            or profile_id is not None
            and result.profile_id != profile_id
            or policy_version is not None
            and result.policy_version != policy_version
        ):
            raise ValueError
        return result
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Summary checkpoint does not match the context request") from None


def summary_system_messages(
    checkpoint: ConversationSummaryCheckpoint,
) -> tuple[ModelMessage, ...]:
    """Project a trusted durable summary into its model-facing system prefix."""

    messages = [ModelMessage(ModelRole.SYSTEM, checkpoint.summary)]
    if checkpoint.artifacts:
        messages.append(
            ModelMessage(
                ModelRole.SYSTEM,
                "Artifacts: " + ", ".join(ref.artifact_id for ref in checkpoint.artifacts),
            )
        )
    return tuple(messages)


def _optional_messages(messages: tuple[ModelMessage, ...], name: str) -> tuple[ModelMessage, ...]:
    if (
        type(messages) is not tuple
        or len(messages) > _MAX_OPTIONAL_MESSAGES
        or any(type(message) is not ModelMessage for message in messages)
    ):
        raise ValueError(f"{name} context requires a bounded tuple of messages")
    snapshot = tuple(replace(message) for message in messages)
    if any(message.role is not ModelRole.SYSTEM for message in snapshot):
        raise ValueError(f"{name} context must use system messages")
    return snapshot


@dataclass(frozen=True, slots=True)
class ContextRequest:
    session_id: SessionId
    profile_id: ProfileId
    policy_version: str
    system_message: ModelMessage
    current_input: ModelMessage
    tools: tuple[ToolSpec, ...] = ()
    open_tool_messages: tuple[ModelMessage, ...] = ()
    summary_checkpoints: tuple[ConversationSummaryCheckpoint, ...] = ()
    memory_messages: tuple[ModelMessage, ...] = ()
    skill_messages: tuple[ModelMessage, ...] = ()

    def __post_init__(self) -> None:
        if type(self.session_id) is not SessionId or type(self.profile_id) is not ProfileId:
            raise ValueError("Context request requires typed session and profile IDs")
        session_id = SessionId(self.session_id.value)
        profile_id = ProfileId(self.profile_id.value)
        policy_version = _policy(self.policy_version)
        if (
            type(self.system_message) is not ModelMessage
            or self.system_message.role is not ModelRole.SYSTEM
            or type(self.current_input) is not ModelMessage
            or self.current_input.role is not ModelRole.USER
        ):
            raise ValueError("Context request requires system and current user messages")
        if type(self.tools) is not tuple or any(type(tool) is not ToolSpec for tool in self.tools):
            raise ValueError("Context request requires a tuple of tool specs")
        if (
            type(self.open_tool_messages) is not tuple
            or len(self.open_tool_messages) > _MAX_INTERACTION_MESSAGES
            or any(type(message) is not ModelMessage for message in self.open_tool_messages)
        ):
            raise ValueError("Open tool context requires a bounded message tuple")
        system = replace(self.system_message)
        current = replace(self.current_input)
        open_messages = tuple(replace(message) for message in self.open_tool_messages)
        if open_messages and (
            open_messages[0].role is not ModelRole.ASSISTANT
            or not open_messages[0].tool_calls
            or any(message.role is not ModelRole.TOOL for message in open_messages[1:])
        ):
            raise ValueError("Open tool context requires one assistant call batch and its results")
        ModelRequest((system, *open_messages, current), tuple(replace(tool) for tool in self.tools))
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "policy_version", policy_version)
        object.__setattr__(self, "system_message", system)
        object.__setattr__(self, "current_input", current)
        object.__setattr__(self, "tools", tuple(replace(tool) for tool in self.tools))
        object.__setattr__(self, "open_tool_messages", open_messages)
        if (
            type(self.summary_checkpoints) is not tuple
            or len(self.summary_checkpoints) > 128
            or any(
                type(item) is not ConversationSummaryCheckpoint for item in self.summary_checkpoints
            )
        ):
            raise ValueError("Context request requires bounded summary checkpoints")
        summaries: list[ConversationSummaryCheckpoint] = []
        for item in self.summary_checkpoints:
            checkpoint = snapshot_summary_checkpoint(item)
            assert checkpoint is not None
            summaries.append(checkpoint)
        object.__setattr__(self, "summary_checkpoints", tuple(summaries))
        object.__setattr__(
            self, "memory_messages", _optional_messages(self.memory_messages, "Memory")
        )
        object.__setattr__(self, "skill_messages", _optional_messages(self.skill_messages, "Skill"))


@dataclass(frozen=True, slots=True)
class ContextWindow:
    messages: tuple[ModelMessage, ...]
    tools: tuple[ToolSpec, ...]
    estimate: TokenEstimate
    budget: ContextBudget
    omitted: tuple[OmittedContextRef, ...]
    policy_version: str
    new_checkpoint: ConversationSummaryCheckpoint | None = None

    def __post_init__(self) -> None:
        if type(self.budget) is not ContextBudget or type(self.estimate) is not TokenEstimate:
            raise ValueError("Context window requires a budget and token estimate")
        if self.estimate.tokens > self.budget.input_allowance_tokens:
            raise ValueError("Context window exceeds its input allowance")
        if type(self.omitted) is not tuple or any(
            type(item) is not OmittedContextRef for item in self.omitted
        ):
            raise ValueError("Context omissions require a tuple of references")
        request = ModelRequest(self.messages, self.tools)
        object.__setattr__(self, "messages", request.messages)
        object.__setattr__(self, "tools", request.tools)
        object.__setattr__(self, "omitted", tuple(replace(item) for item in self.omitted))
        object.__setattr__(self, "policy_version", _policy(self.policy_version))
        checkpoint = snapshot_summary_checkpoint(self.new_checkpoint)
        if checkpoint is not None and checkpoint.policy_version != self.policy_version:
            raise ValueError("Summary checkpoint does not match the context policy")
        object.__setattr__(self, "new_checkpoint", checkpoint)


@dataclass(frozen=True, slots=True)
class SummaryCompactionRequest:
    session_id: SessionId
    profile_id: ProfileId
    policy_version: str
    interactions: tuple[ConversationInteraction, ...]

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not SessionId
            or type(self.profile_id) is not ProfileId
            or type(self.interactions) is not tuple
            or not self.interactions
            or any(type(item) is not ConversationInteraction for item in self.interactions)
        ):
            raise ValueError("Summary compaction request is invalid")
        ordered = tuple(
            sorted(
                (
                    ConversationInteraction(
                        item.sequence_from, item.sequence_through, item.messages
                    )
                    for item in self.interactions
                ),
                key=lambda item: item.sequence_from,
            )
        )
        if any(
            earlier.sequence_through >= later.sequence_from
            for earlier, later in zip(ordered, ordered[1:])
        ):
            raise ValueError("Summary interactions must not overlap")
        object.__setattr__(self, "session_id", SessionId(self.session_id.value))
        object.__setattr__(self, "profile_id", ProfileId(self.profile_id.value))
        object.__setattr__(self, "policy_version", _policy(self.policy_version))
        object.__setattr__(self, "interactions", ordered)


class ConversationReader(Protocol):
    def read_page(
        self, session_id: SessionId, before_sequence: int | None, limit: int
    ) -> Awaitable[ConversationPage]: ...


class ConversationAggregateReader(ConversationReader, Protocol):
    def summarize_before(
        self,
        session_id: SessionId,
        before_sequence: int,
        excluded_from_sequence: int | None = None,
        excluded_through_sequence: int | None = None,
    ) -> Awaitable[ConversationCoverage]: ...


class TokenCounter(Protocol):
    def count(
        self, messages: tuple[ModelMessage, ...], tools: tuple[ToolSpec, ...]
    ) -> TokenEstimate: ...


class ConversationCompactor(Protocol):
    def compact(
        self, request: SummaryCompactionRequest
    ) -> Awaitable[ConversationSummaryCheckpoint]: ...


class ContextProvider(Protocol):
    def build(self, request: ContextRequest, budget: ContextBudget) -> Awaitable[ContextWindow]: ...


PENDING_TOOL_MESSAGE = "The tool was not executed because it is waiting for approval."
HELD_TOOL_MESSAGE = (
    "The tool was not executed because an earlier tool call is waiting for approval."
)
