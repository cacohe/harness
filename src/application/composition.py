from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NoReturn

from src.adapters.artifacts.externalizer import (
    ArtifactPolicy,
    PolicyArtifactExternalizer,
)
from src.adapters.artifacts.filesystem import FilesystemArtifactRepository
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.agent_v3 import Agent
from src.application.runtime import RuntimeCoordinator, _close_resource
from src.application.submissions import SubmissionService
from src.core.artifacts import ArtifactExternalizer
from src.core.commands import RunLimits
from src.core.context import ContextBudget, ContextProvider, ContextRequest
from src.core.ids import Clock, IdGenerator, ProfileId
from src.core.journal import RunJournal
from src.core.model import ModelGateway
from src.core.run import RunCoordinator
from src.core.state import RunState
from src.core.tools import ToolGateway

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
    sqlite_path: Path | None = None
    artifact_root: Path | None = None

    def __post_init__(self) -> None:
        integers = (
            object.__getattribute__(self, "max_sessions"),
            object.__getattribute__(self, "max_waiters_per_session"),
            object.__getattribute__(self, "max_advances_per_operation"),
            object.__getattribute__(self, "max_artifact_bytes"),
            object.__getattribute__(self, "max_inline_artifact_bytes"),
            object.__getattribute__(self, "max_artifact_summary_bytes"),
        )
        limits = _snapshot_limits(object.__getattribute__(self, "limits"))
        context_budget = _snapshot_context_budget(object.__getattribute__(self, "context_budget"))
        if (
            any(type(value) is not int for value in integers)
            or not 1 <= integers[0] <= 1_000_000
            or not 0 <= integers[1] <= 1_000_000
            or not 1 <= integers[2] <= 1_000_000
            or not 1 <= integers[3] <= (1 << 63) - 1
            or not 0 <= integers[4] <= integers[3]
            or not 1 <= integers[5] <= 1_048_576
        ):
            raise ValueError("Application configuration limits are invalid")
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "context_budget", context_budget)
        for name in ("sqlite_path", "artifact_root"):
            value = object.__getattribute__(self, name)
            if value is None:
                continue
            if type(value) is not _PATH_TYPE:
                raise ValueError("Production paths must be explicit absolute paths")
            rebuilt = _PATH_TYPE(str(value))
            if not rebuilt.is_absolute():
                raise ValueError("Production paths must be explicit absolute paths")
            object.__setattr__(self, name, rebuilt)


async def create_development_agent(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway,
    context: ContextProvider,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal | None = None,
    artifacts: ArtifactExternalizer | None = None,
    owned_resources: tuple[object, ...] = (),
) -> Agent:
    config, profile_id = _validate_composition_inputs(
        config,
        model=model,
        tools=tools,
        context=context,
        context_request_factory=context_request_factory,
        ids=ids,
        clock=clock,
        profile_id=profile_id,
        journal=journal,
        artifacts=artifacts,
        owned_resources=owned_resources,
    )
    created: list[object] = []
    try:
        selected_journal = journal
        if selected_journal is None:
            selected_journal = InMemoryRunJournal()
            created.append(selected_journal)
        selected_artifacts = artifacts
        if selected_artifacts is None:
            repository = InMemoryArtifactRepository(config.max_artifact_bytes)
            created.append(repository)
            selected_artifacts = PolicyArtifactExternalizer(repository, _artifact_policy(config))
            created.append(selected_artifacts)
        _require_methods(selected_journal, _JOURNAL_METHODS)
        _require_methods(selected_artifacts, ("externalize",))
        return _compose(
            config,
            model=model,
            tools=tools,
            context=context,
            context_request_factory=context_request_factory,
            ids=ids,
            clock=clock,
            profile_id=profile_id,
            journal=selected_journal,
            artifacts=selected_artifacts,
            owned_resources=(*owned_resources, *created),
        )
    except BaseException as error:
        await _cleanup_and_raise(error, created)


async def create_production_agent(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway,
    context: ContextProvider,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal | None = None,
    artifacts: ArtifactExternalizer | None = None,
    owned_resources: tuple[object, ...] = (),
) -> Agent:
    config, profile_id = _validate_composition_inputs(
        config,
        model=model,
        tools=tools,
        context=context,
        context_request_factory=context_request_factory,
        ids=ids,
        clock=clock,
        profile_id=profile_id,
        journal=journal,
        artifacts=artifacts,
        owned_resources=owned_resources,
    )
    if journal is None and config.sqlite_path is None:
        raise ValueError("Production composition requires an absolute SQLite path")
    if artifacts is None and config.artifact_root is None:
        raise ValueError("Production composition requires an absolute Artifact root")
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
        if artifacts is None:
            assert config.artifact_root is not None
            repository = FilesystemArtifactRepository(
                config.artifact_root,
                config.max_artifact_bytes,
            )
            created.append(repository)
            selected_artifacts = PolicyArtifactExternalizer(repository, _artifact_policy(config))
            created.append(selected_artifacts)
        else:
            selected_artifacts = artifacts
        _require_methods(selected_journal, _JOURNAL_METHODS)
        _require_methods(selected_artifacts, ("externalize",))
        return _compose(
            config,
            model=model,
            tools=tools,
            context=context,
            context_request_factory=context_request_factory,
            ids=ids,
            clock=clock,
            profile_id=profile_id,
            journal=selected_journal,
            artifacts=selected_artifacts,
            owned_resources=(*owned_resources, *created),
        )
    except BaseException as error:
        await _cleanup_and_raise(error, created)


def _compose(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway,
    context: ContextProvider,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal,
    artifacts: ArtifactExternalizer,
    owned_resources: tuple[object, ...],
) -> Agent:
    submissions = SubmissionService(journal, ids, clock, config.limits)

    def coordinator_factory(state: RunState, user_input: str) -> RunCoordinator:
        request = context_request_factory(state, user_input)
        if type(request) is not ContextRequest:
            raise ValueError("Context request factory returned an invalid request")
        return RunCoordinator(
            context=context,
            model=model,
            tools=tools,
            artifacts=artifacts,
            journal=journal,
            clock=clock,
            ids=ids,
            context_request=request,
            context_budget=config.context_budget,
        )

    runtime = RuntimeCoordinator(
        submissions=submissions,
        journal=journal,
        coordinator_factory=coordinator_factory,
        ids=ids,
        clock=clock,
        max_sessions=config.max_sessions,
        max_waiters_per_session=config.max_waiters_per_session,
        max_advances_per_operation=config.max_advances_per_operation,
        owned_resources=owned_resources,
    )
    return Agent(runtime, ids, profile_id)


def _validate_composition_inputs(
    config: ApplicationConfig,
    *,
    model: ModelGateway,
    tools: ToolGateway,
    context: ContextProvider,
    context_request_factory: ContextRequestFactory,
    ids: IdGenerator,
    clock: Clock,
    profile_id: ProfileId,
    journal: RunJournal | None,
    artifacts: ArtifactExternalizer | None,
    owned_resources: tuple[object, ...],
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
    _require_methods(tools, ("execute",))
    _require_methods(context, ("build",))
    _require_callable(context_request_factory)
    _require_methods(
        ids, ("run", "session", "submission", "profile", "event", "step", "model_call")
    )
    _require_methods(clock, ("now", "monotonic"))
    if journal is not None:
        _require_methods(journal, _JOURNAL_METHODS)
    if artifacts is not None:
        _require_methods(artifacts, ("externalize",))
    return config, profile


_JOURNAL_METHODS = (
    "commit",
    "load_head",
    "load_session",
    "find_submission",
    "find_run_input",
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
            sqlite_path=object.__getattribute__(value, "sqlite_path"),
            artifact_root=object.__getattribute__(value, "artifact_root"),
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
