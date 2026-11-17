from __future__ import annotations

from pathlib import Path

import pytest
from src.core.ids import ProfileId, SessionId, SubmissionId

from .test_runtime_v3 import (
    Clock,
    CompletingCoordinator,
    Ids,
    Journal,
    Submissions,
    async_test,
)


@async_test
async def test_agent_exposes_only_async_first_session_and_submit_surface():
    from src.application.agent import Agent
    from src.application.runtime import RuntimeCoordinator

    journal = Journal()
    runtime = RuntimeCoordinator(
        submissions=Submissions(journal),
        journal=journal,
        coordinator_factory=lambda run, _message: CompletingCoordinator(journal, run, []),
        ids=Ids(),
        clock=Clock(),
        max_sessions=3,
        max_waiters_per_session=1,
    )
    agent = Agent(runtime, Ids(), ProfileId("profile"))
    session = await agent.open_session(SessionId("session"))
    result = await session.submit("hello", SubmissionId("submission"))
    assert result.status.value == "completed"
    assert not hasattr(agent, "chat")
    assert not hasattr(agent, "achat")
    await agent.close()


def test_application_config_is_immutable_strict_and_requires_absolute_production_paths(tmp_path):
    from src.application.composition import ApplicationConfig

    config = ApplicationConfig(
        max_sessions=4,
        max_waiters_per_session=2,
        max_advances_per_operation=32,
        sqlite_path=tmp_path / "journal.sqlite3",
        artifact_root=tmp_path / "artifacts",
    )
    assert config.sqlite_path.is_absolute()
    with pytest.raises(Exception):
        config.max_sessions = 8  # type: ignore[misc]
    with pytest.raises(ValueError):
        ApplicationConfig(max_sessions=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ApplicationConfig(sqlite_path=Path("relative.sqlite3"))


def test_production_composition_requires_explicit_durable_adapter_paths(tmp_path):
    from src.application.composition import ApplicationConfig, create_production_agent

    dependencies = {
        "model": object(),
        "tools": object(),
        "context": object(),
        "context_request_factory": lambda _state, _message: object(),
        "ids": Ids(),
        "clock": Clock(),
        "profile_id": ProfileId("profile"),
    }
    with pytest.raises(ValueError, match="SQLite path"):
        create_production_agent(ApplicationConfig(), **dependencies)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Artifact root"):
        create_production_agent(
            ApplicationConfig(),
            journal=object(),  # type: ignore[arg-type]
            **dependencies,  # type: ignore[arg-type]
        )
    sqlite_path = tmp_path / "must-not-exist.sqlite3"
    artifact_root = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="Composition inputs"):
        create_production_agent(
            ApplicationConfig(sqlite_path=sqlite_path, artifact_root=artifact_root),
            **{**dependencies, "profile_id": object()},  # type: ignore[arg-type]
        )
    assert not sqlite_path.exists()
    assert not artifact_root.exists()


def test_application_exports_do_not_reexport_legacy_harness_surface():
    import src.application as application

    assert application.Agent.__module__ == "src.application.agent"
    assert not hasattr(application, "Harness")
    assert not hasattr(application.Agent, "from_config")
