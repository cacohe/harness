from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.adapters.artifacts.externalizer import (
    ArtifactPolicy,
    PolicyArtifactExternalizer,
)
from src.adapters.artifacts.filesystem import FilesystemArtifactRepository
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.persistence.memory import InMemoryRunJournal
from src.adapters.persistence.sqlite import SQLiteRunJournal
from src.application.agent_v3 import Agent
from src.application.runtime import RuntimeCoordinator
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
            self.max_sessions,
            self.max_waiters_per_session,
            self.max_advances_per_operation,
            self.max_artifact_bytes,
            self.max_inline_artifact_bytes,
            self.max_artifact_summary_bytes,
        )
        if (
            any(type(value) is not int for value in integers)
            or not 1 <= self.max_sessions <= 1_000_000
            or not 0 <= self.max_waiters_per_session <= 1_000_000
            or not 1 <= self.max_advances_per_operation <= 1_000_000
            or not 1 <= self.max_artifact_bytes <= (1 << 63) - 1
            or not 0 <= self.max_inline_artifact_bytes <= self.max_artifact_bytes
            or not 1 <= self.max_artifact_summary_bytes <= 1_048_576
            or type(self.limits) is not RunLimits
            or type(self.context_budget) is not ContextBudget
        ):
            raise ValueError("Application configuration limits are invalid")
        for name in ("sqlite_path", "artifact_root"):
            value = object.__getattribute__(self, name)
            if value is None:
                continue
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError("Production paths must be explicit absolute paths")
            object.__setattr__(self, name, Path(value))


def create_development_agent(
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
    _validate_composition_inputs(config, profile_id, owned_resources)
    selected_journal = journal or InMemoryRunJournal()
    selected_artifacts = artifacts
    if selected_artifacts is None:
        repository = InMemoryArtifactRepository(config.max_artifact_bytes)
        selected_artifacts = PolicyArtifactExternalizer(repository, _artifact_policy(config))
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
        owned_resources=owned_resources,
    )


def create_production_agent(
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
    _validate_composition_inputs(config, profile_id, owned_resources)
    selected_journal: RunJournal
    if journal is None:
        if config.sqlite_path is None:
            raise ValueError("Production composition requires an absolute SQLite path")
        selected_journal = SQLiteRunJournal(config.sqlite_path)
    else:
        selected_journal = journal
    selected_artifacts: ArtifactExternalizer
    if artifacts is None:
        if config.artifact_root is None:
            raise ValueError("Production composition requires an absolute Artifact root")
        repository = FilesystemArtifactRepository(
            config.artifact_root,
            config.max_artifact_bytes,
        )
        selected_artifacts = PolicyArtifactExternalizer(repository, _artifact_policy(config))
    else:
        selected_artifacts = artifacts
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
        owned_resources=owned_resources,
    )


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
    profile_id: ProfileId,
    owned_resources: tuple[object, ...],
) -> None:
    if (
        type(config) is not ApplicationConfig
        or type(profile_id) is not ProfileId
        or type(owned_resources) is not tuple
        or len({id(resource) for resource in owned_resources}) != len(owned_resources)
    ):
        raise ValueError("Composition inputs are invalid")


def _artifact_policy(config: ApplicationConfig) -> ArtifactPolicy:
    return ArtifactPolicy(
        config.max_inline_artifact_bytes,
        config.max_artifact_bytes,
        config.max_artifact_summary_bytes,
        frozenset({"application/json", "text/plain", "application/octet-stream"}),
    )
