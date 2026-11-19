from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NoReturn

from src.adapters.artifacts.externalizer import (
    ArtifactPolicy,
    PolicyArtifactExternalizer,
)
from src.adapters.artifacts.filesystem import FilesystemArtifactRepository
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.bounded import BoundedRunJournal
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.execution_store import SQLiteToolExecutionStore
from src.adapters.tools.local import LocalToolGateway
from src.adapters.tools.read_artifact import READ_ARTIFACT_TOOL_NAME, read_artifact_entry
from src.adapters.tools.router import ToolRouter
from src.application.agent import Agent
from src.application.artifact_retention import ArtifactRetentionService
from src.application.observability import PostCommitObserver
from src.application.recovery import RecoveryService
from src.application.runtime import RuntimeCoordinator, _close_resource
from src.application.submissions import SubmissionService
from src.core.artifacts import ArtifactExternalizer, ArtifactRepository
from src.core.commands import RunLimits
from src.core.context import ContextBudget, ContextProvider, ContextRequest, TokenCounter
from src.core.ids import Clock, IdGenerator, ProfileId
from src.core.journal import RunJournal
from src.core.model import ModelGateway, ModelMessage
from src.core.run import RunCoordinator
from src.core.state import RunState
from src.core.tools import ToolGateway, ToolSpec

ContextRequestFactory = Callable[[RunState, str], ContextRequest]
_PATH_TYPE = type(Path())


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    max_sessions: int = 128
    max_waiters_per_session: int = 16
    max_advances_per_operation: int = 1024
    limits: RunLimits = field(default_factory=lambda: RunLimits(32, 300.0, 1_048_576))
    context_budget: ContextBudget = field(
        default_factory=lambda: ContextBudget(128_000, 4_096, 1_024)
    )
    max_artifact_bytes: int = 64 * 1_048_576
    max_inline_artifact_bytes: int = 65_536
    max_artifact_summary_bytes: int = 16_384
    max_artifact_read_bytes: int = 65_536
    max_orphan_reference_ids: int = 1_000_000
    max_serialized_run_state_bytes: int = 16 * 1_048_576
    max_user_message_bytes: int = 1_048_576
    sqlite_path: Path | None = None
    tool_execution_path: Path | None = None
    artifact_root: Path | None = None
    observer_timeout_seconds: float = 0.1

    def __post_init__(self) -> None:
        integers = (
            object.__getattribute__(self, "max_sessions"),
            object.__getattribute__(self, "max_waiters_per_session"),
            object.__getattribute__(self, "max_advances_per_operation"),
            object.__getattribute__(self, "max_artifact_bytes"),
            object.__getattribute__(self, "max_inline_artifact_bytes"),
            object.__getattribute__(self, "max_artifact_summary_bytes"),
            object.__getattribute__(self, "max_artifact_read_bytes"),
            object.__getattribute__(self, "max_orphan_reference_ids"),
            object.__getattribute__(self, "max_serialized_run_state_bytes"),
            object.__getattribute__(self, "max_user_message_bytes"),
        )
        limits = _snapshot_limits(object.__getattribute__(self, "limits"))
        context_budget = _snapshot_context_budget(object.__getattribute__(self, "context_budget"))
        observer_timeout = object.__getattribute__(self, "observer_timeout_seconds")
        if (
            any(type(value) is not int for value in integers)
            or not 1 <= integers[0] <= 1_000_000
            or not 0 <= integers[1] <= 1_000_000
            or not 1 <= integers[2] <= 1_000_000
            or not 1 <= integers[3] <= (1 << 63) - 1
            or not 0 <= integers[4] <= integers[3]
            or not 1 <= integers[5] <= 1_048_576
            or not 1 <= integers[6] <= integers[3]
            or not 1 <= integers[7] <= 1_000_000
            or not 1_024 <= integers[8] <= 16 * 1_048_576
            or not 1 <= integers[9] <= 64 * 1_048_576
            or isinstance(observer_timeout, bool)
            or type(observer_timeout) not in (int, float)
            or not math.isfinite(observer_timeout)
            or not 0.001 <= observer_timeout <= 30
        ):
            raise ValueError("Application configuration limits are invalid")
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "context_budget", context_budget)
        object.__setattr__(self, "observer_timeout_seconds", float(observer_timeout))
        for name in ("sqlite_path", "tool_execution_path", "artifact_root"):
            value = object.__getattribute__(self, name)
            if value is None:
                continue
            if type(value) is not _PATH_TYPE:
                raise ValueError("Production paths must be explicit absolute paths")
            rebuilt = _PATH_TYPE(str(value))
            if not rebuilt.is_absolute():
                raise ValueError("Production paths must be explicit absolute paths")
            object.__setattr__(self, name, rebuilt)
        if (
            self.sqlite_path is not None
            and self.tool_execution_path is not None
            and self.sqlite_path == self.tool_execution_path
        ):
            raise ValueError("Application configuration limits are invalid")


async def create_development_agent(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway,
    context: ContextProvider,
    context_counter: TokenCounter,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal | None = None,
    artifacts: ArtifactExternalizer | None = None,
    artifact_repository: ArtifactRepository | None = None,
    owned_resources: tuple[object, ...] = (),
    observer: PostCommitObserver | None = None,
) -> Agent:
    config, profile_id = _validate_composition_inputs(
        config,
        model=model,
        tools=tools,
        context=context,
        context_counter=context_counter,
        context_request_factory=context_request_factory,
        ids=ids,
        clock=clock,
        profile_id=profile_id,
        journal=journal,
        artifacts=artifacts,
        artifact_repository=artifact_repository,
        owned_resources=owned_resources,
        observer=observer,
    )
    created: list[object] = []
    try:
        selected_journal = journal
        if selected_journal is None:
            selected_journal = InMemoryRunJournal()
            created.append(selected_journal)
        selected_artifacts = artifacts
        selected_repository = artifact_repository
        if selected_artifacts is None:
            if selected_repository is None:
                selected_repository = InMemoryArtifactRepository(config.max_artifact_bytes)
                created.append(selected_repository)
            selected_artifacts = PolicyArtifactExternalizer(
                selected_repository, _artifact_policy(config)
            )
            created.append(selected_artifacts)
        elif selected_repository is None:
            selected_repository = _externalizer_repository(selected_artifacts)
        _require_methods(selected_journal, _JOURNAL_METHODS)
        _require_methods(selected_artifacts, ("externalize",))
        return _compose(
            config,
            model=model,
            tools=tools,
            context=context,
            context_counter=context_counter,
            context_request_factory=context_request_factory,
            ids=ids,
            clock=clock,
            profile_id=profile_id,
            journal=selected_journal,
            artifacts=selected_artifacts,
            artifact_repository=selected_repository,
            owned_resources=(*owned_resources, *created),
            observer=observer,
        )
    except BaseException as error:
        await _cleanup_and_raise(error, created)


async def create_production_agent(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway | None,
    context: ContextProvider,
    context_counter: TokenCounter,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal | None = None,
    artifacts: ArtifactExternalizer | None = None,
    artifact_repository: ArtifactRepository | None = None,
    owned_resources: tuple[object, ...] = (),
    observer: PostCommitObserver | None = None,
) -> Agent:
    config, profile_id = _validate_composition_inputs(
        config,
        model=model,
        tools=tools,
        context=context,
        context_counter=context_counter,
        context_request_factory=context_request_factory,
        ids=ids,
        clock=clock,
        profile_id=profile_id,
        journal=journal,
        artifacts=artifacts,
        artifact_repository=artifact_repository,
        owned_resources=owned_resources,
        observer=observer,
        allow_missing_tools=True,
    )
    if journal is None and config.sqlite_path is None:
        raise ValueError("Production composition requires an absolute SQLite path")
    if config.artifact_root is None:
        raise ValueError("Production composition requires an absolute Artifact root")
    if (
        config.max_inline_artifact_bytes > config.limits.max_inline_result_bytes
        or config.max_artifact_summary_bytes > config.limits.max_inline_result_bytes
        or config.max_artifact_read_bytes > config.limits.max_inline_result_bytes
    ):
        raise ValueError("Production Artifact limits exceed the Run inline result limit")
    created: list[object] = []
    try:
        selected_journal: RunJournal
        if journal is None:
            assert config.sqlite_path is not None
            selected_journal = SQLiteRunJournal(config.sqlite_path)
            created.append(selected_journal)
        else:
            selected_journal = journal
        selected_artifacts: ArtifactExternalizer
        selected_repository = artifact_repository
        if artifacts is None:
            if selected_repository is None:
                assert config.artifact_root is not None
                selected_repository = FilesystemArtifactRepository(
                    config.artifact_root,
                    config.max_artifact_bytes,
                )
                created.append(selected_repository)
            selected_artifacts = PolicyArtifactExternalizer(
                selected_repository, _artifact_policy(config)
            )
            created.append(selected_artifacts)
        else:
            selected_artifacts = artifacts
            if selected_repository is None:
                selected_repository = _externalizer_repository(selected_artifacts)
        _require_methods(selected_journal, _JOURNAL_METHODS)
        _require_methods(selected_artifacts, ("externalize",))
        if selected_repository is None:
            raise ValueError("Production composition requires an Artifact repository")
        _require_methods(selected_repository, ("stat", "open", "delete", "collect_orphans"))
        _require_methods(selected_journal, ("load_referenced_artifact_ids",))
        execution_store = SQLiteToolExecutionStore(_tool_execution_path(config))
        await execution_store.recover_abandoned()
        read_entry = read_artifact_entry(
            selected_repository,
            workspace_id=_artifact_workspace(config.artifact_root),
            max_read_bytes=config.max_artifact_read_bytes,
        )
        read_gateway = LocalToolGateway(
            ToolCatalog((read_entry,)),
            execution_store,
            selected_artifacts,
            timeout_seconds=min(60.0, config.limits.max_elapsed_seconds),
        )
        selected_tools = ToolRouter(
            {READ_ARTIFACT_TOOL_NAME: read_gateway},
            fallback=tools,
        )
        maintenance = ArtifactRetentionService(
            journal=selected_journal,
            artifacts=selected_repository,
            max_reference_ids=config.max_orphan_reference_ids,
        )
        return _compose(
            config,
            model=model,
            tools=selected_tools,
            context=context,
            context_counter=context_counter,
            context_request_factory=context_request_factory,
            ids=ids,
            clock=clock,
            profile_id=profile_id,
            journal=selected_journal,
            artifacts=selected_artifacts,
            artifact_repository=selected_repository,
            owned_resources=(*owned_resources, *created),
            observer=observer,
            reserved_tool=read_entry.spec,
            artifact_services=maintenance,
        )
    except BaseException as error:
        await _cleanup_and_raise(error, created)


def _compose(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway,
    context: ContextProvider,
    context_counter: TokenCounter,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal,
    artifacts: ArtifactExternalizer,
    artifact_repository: ArtifactRepository | None,
    owned_resources: tuple[object, ...],
    observer: PostCommitObserver | None,
    reserved_tool: ToolSpec | None = None,
    artifact_services: ArtifactRetentionService | None = None,
) -> Agent:
    bounded_journal = BoundedRunJournal(journal, config.max_serialized_run_state_bytes)
    submissions = SubmissionService(
        bounded_journal, ids, clock, config.limits, config.max_user_message_bytes
    )

    async def coordinator_factory(state: RunState, user_input: str) -> RunCoordinator:
        request = context_request_factory(state, user_input)
        if type(request) is not ContextRequest:
            raise ValueError("Context request factory returned an invalid request")
        if request.session_id != state.session_id or request.profile_id != state.profile_id:
            raise ValueError("Context request factory returned mismatched identities")
        if reserved_tool is not None:
            if any(tool.name == reserved_tool.name for tool in request.tools):
                raise ValueError("Context request collides with a reserved production tool")
            request = replace(request, tools=(*request.tools, reserved_tool))
        summaries = await bounded_journal.load_context_summaries(
            state.session_id,
            state.profile_id,
            request.policy_version,
            state.journal_sequence,
            128,
            state.journal_version,
            state.journal_sequence,
        )
        input_entry = await bounded_journal.find_run_input(
            state.session_id, state.run_id, state.submission_id
        )
        if input_entry is None:
            raise ValueError("Active Run does not have a durable input")
        latest = await bounded_journal.read_page(state.session_id, None, 1)
        open_tool_messages: tuple[ModelMessage, ...] = ()
        open_tool_sequence_range = None
        if latest.interactions:
            interaction = latest.interactions[0]
            if (
                interaction.sequence_from > input_entry.sequence
                and interaction.messages[0].tool_calls
            ):
                open_tool_messages = interaction.messages
                open_tool_sequence_range = (
                    interaction.sequence_from,
                    interaction.sequence_through,
                )
        request = replace(
            request,
            summary_checkpoints=summaries,
            current_input_sequence=input_entry.sequence,
            open_tool_messages=open_tool_messages,
            open_tool_sequence_range=open_tool_sequence_range,
        )
        return RunCoordinator(
            context=context,
            model=model,
            tools=tools,
            artifacts=artifacts,
            journal=bounded_journal,
            clock=clock,
            ids=ids,
            context_request=request,
            context_budget=config.context_budget,
            context_counter=context_counter,
        )

    runtime = RuntimeCoordinator(
        submissions=submissions,
        journal=bounded_journal,
        coordinator_factory=coordinator_factory,
        ids=ids,
        clock=clock,
        max_sessions=config.max_sessions,
        max_waiters_per_session=config.max_waiters_per_session,
        max_advances_per_operation=config.max_advances_per_operation,
        owned_resources=owned_resources,
        observer=observer,
        observer_timeout_seconds=config.observer_timeout_seconds,
        recovery=(
            RecoveryService(
                journal=bounded_journal,
                artifacts=artifact_repository,
                clock=clock,
                ids=ids,
            )
            if artifact_repository is not None
            else None
        ),
    )
    return Agent(runtime, ids, profile_id, artifacts=artifact_services)


def _validate_composition_inputs(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway | None,
    context: ContextProvider,
    context_counter: TokenCounter,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal | None,
    artifacts: ArtifactExternalizer | None,
    artifact_repository: ArtifactRepository | None,
    owned_resources: tuple[object, ...],
    observer: PostCommitObserver | None,
    allow_missing_tools: bool = False,
) -> tuple[ApplicationConfig, ProfileId]:
    if (
        type(config) is not ApplicationConfig
        or type(owned_resources) is not tuple
        or len({id(resource) for resource in owned_resources}) != len(owned_resources)
    ):
        raise ValueError("Composition inputs are invalid")
    config = _snapshot_application_config(config)
    profile = _snapshot_profile_id(profile_id)
    _require_methods(model, ("complete",))
    if tools is None:
        if not allow_missing_tools:
            raise ValueError("Composition dependency ports are invalid")
    else:
        _require_methods(tools, ("execute",))
    _require_methods(context, ("build",))
    _require_methods(context_counter, ("count",))
    _require_callable(context_request_factory)
    _require_methods(
        ids, ("run", "session", "submission", "profile", "event", "step", "model_call")
    )
    _require_methods(clock, ("now", "monotonic"))
    if journal is not None:
        _require_methods(journal, _JOURNAL_METHODS)
    if artifacts is not None:
        _require_methods(artifacts, ("externalize",))
    if artifact_repository is not None:
        _require_methods(artifact_repository, ("stat",))
    try:
        max_output_tokens = object.__getattribute__(model, "max_output_tokens")
    except AttributeError:
        max_output_tokens = None
    if max_output_tokens is not None and (
        type(max_output_tokens) is not int
        or max_output_tokens < 1
        or max_output_tokens > config.context_budget.reserved_output_tokens
    ):
        raise ValueError("Model output limit exceeds the reserved output context")
    if observer is not None:
        _require_methods(observer, ("observe",))
    return config, profile


_JOURNAL_METHODS = (
    "commit",
    "load_head",
    "load_session",
    "load_context_summaries",
    "find_submission",
    "find_run_input",
    "read_page",
    "find_run_state",
    "find_run_state_by_id",
    "load_artifact_attachments",
    "artifact_has_active_attachments",
    "reserve_artifact_retirement",
    "complete_artifact_retirement",
    "abort_artifact_retirement",
    "load_artifact_facts",
)


def _require_methods(value: object, names: tuple[str, ...]) -> None:
    for name in names:
        try:
            member = inspect.getattr_static(value, name)
        except (AttributeError, TypeError):
            raise ValueError("Composition dependency ports are invalid") from None
        if isinstance(member, (classmethod, staticmethod)):
            member = member.__func__
        if not callable(member):
            raise ValueError("Composition dependency ports are invalid")


def _require_callable(value: object) -> None:
    try:
        member = inspect.getattr_static(value, "__call__")
    except (AttributeError, TypeError):
        raise ValueError("Composition dependency ports are invalid") from None
    if not callable(member):
        raise ValueError("Composition dependency ports are invalid")


def _externalizer_repository(value: ArtifactExternalizer) -> ArtifactRepository | None:
    try:
        repository = object.__getattribute__(value, "repository")
        inspect.getattr_static(repository, "stat")
    except (AttributeError, TypeError):
        return None
    return repository


def _snapshot_profile_id(value: object) -> ProfileId:
    try:
        if type(value) is not ProfileId:
            raise ValueError
        raw = object.__getattribute__(value, "value")
        if type(raw) is not str:
            raise ValueError
        return ProfileId(raw)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Composition inputs are invalid") from None


def _snapshot_application_config(value: object) -> ApplicationConfig:
    try:
        if type(value) is not ApplicationConfig:
            raise ValueError
        return ApplicationConfig(
            max_sessions=object.__getattribute__(value, "max_sessions"),
            max_waiters_per_session=object.__getattribute__(value, "max_waiters_per_session"),
            max_advances_per_operation=object.__getattribute__(value, "max_advances_per_operation"),
            limits=object.__getattribute__(value, "limits"),
            context_budget=object.__getattribute__(value, "context_budget"),
            max_artifact_bytes=object.__getattribute__(value, "max_artifact_bytes"),
            max_inline_artifact_bytes=object.__getattribute__(value, "max_inline_artifact_bytes"),
            max_artifact_summary_bytes=object.__getattribute__(value, "max_artifact_summary_bytes"),
            max_artifact_read_bytes=object.__getattribute__(value, "max_artifact_read_bytes"),
            max_orphan_reference_ids=object.__getattribute__(value, "max_orphan_reference_ids"),
            max_serialized_run_state_bytes=object.__getattribute__(
                value, "max_serialized_run_state_bytes"
            ),
            max_user_message_bytes=object.__getattribute__(value, "max_user_message_bytes"),
            sqlite_path=object.__getattribute__(value, "sqlite_path"),
            tool_execution_path=object.__getattribute__(value, "tool_execution_path"),
            artifact_root=object.__getattribute__(value, "artifact_root"),
            observer_timeout_seconds=object.__getattribute__(value, "observer_timeout_seconds"),
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Composition inputs are invalid") from None


def _snapshot_limits(value: object) -> RunLimits:
    try:
        if type(value) is not RunLimits:
            raise ValueError
        steps = object.__getattribute__(value, "max_steps")
        elapsed = object.__getattribute__(value, "max_elapsed_seconds")
        inline = object.__getattribute__(value, "max_inline_result_bytes")
        if type(steps) is not int or type(elapsed) not in (int, float) or type(inline) is not int:
            raise ValueError
        return RunLimits(steps, elapsed, inline)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Application configuration limits are invalid") from None


def _snapshot_context_budget(value: object) -> ContextBudget:
    try:
        if type(value) is not ContextBudget:
            raise ValueError
        window = object.__getattribute__(value, "context_window_tokens")
        output = object.__getattribute__(value, "reserved_output_tokens")
        margin = object.__getattribute__(value, "safety_margin_tokens")
        if any(type(item) is not int for item in (window, output, margin)):
            raise ValueError
        return ContextBudget(window, output, margin)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Application configuration limits are invalid") from None


async def _cleanup_resources(resources: list[object]) -> tuple[BaseException, ...]:
    failures: list[BaseException] = []
    for resource in reversed(resources):
        try:
            await _close_resource(resource)
        except BaseException as error:
            failures.append(error)
    return tuple(failures)


async def _cleanup_and_raise(original: BaseException, resources: list[object]) -> NoReturn:
    cleanup = asyncio.create_task(_cleanup_resources(resources))
    cancellation: asyncio.CancelledError | None = None
    try:
        failures = await asyncio.shield(cleanup)
    except asyncio.CancelledError as error:
        cancellation = error
        while True:
            try:
                failures = await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError as repeated:
                cancellation = repeated
    if cancellation is not None:
        cause: BaseException = original
        if failures:
            cause = BaseExceptionGroup(
                "Composition construction and cleanup failed",
                (original, *failures),
            )
        raise cancellation from cause
    if failures:
        original.add_note("One or more composition resources also failed to close")
        cleanup_group = BaseExceptionGroup("Composition resource cleanup failed", failures)
        raise original from cleanup_group
    raise original


def _artifact_policy(config: ApplicationConfig) -> ArtifactPolicy:
    return ArtifactPolicy(
        config.max_inline_artifact_bytes,
        config.max_artifact_bytes,
        config.max_artifact_summary_bytes,
        frozenset({"application/json", "text/plain", "application/octet-stream"}),
    )


def _tool_execution_path(config: ApplicationConfig) -> Path:
    if config.tool_execution_path is not None:
        return config.tool_execution_path
    if config.sqlite_path is None:
        raise ValueError("Production composition requires an absolute tool execution path")
    return config.sqlite_path.with_name(f"{config.sqlite_path.name}.tools")


def _artifact_workspace(root: Path) -> str:
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return f"artifact-{digest}"
