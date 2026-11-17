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

    class PathSubclass(type(tmp_path)):
        pass

    with pytest.raises(ValueError):
        ApplicationConfig(sqlite_path=PathSubclass(str(tmp_path / "subclass.sqlite3")))


def test_production_composition_requires_explicit_durable_adapter_paths(tmp_path):
    from src.adapters.persistence.memory import InMemoryRunJournal
    from src.application.composition import ApplicationConfig, create_production_agent

    dependencies = _valid_composition_dependencies()
    with pytest.raises(ValueError, match="SQLite path"):
        create_production_agent(ApplicationConfig(), **dependencies)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Artifact root"):
        create_production_agent(
            ApplicationConfig(),
            journal=InMemoryRunJournal(),
            **dependencies,
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


@pytest.mark.parametrize(
    "invalid_name",
    [
        "model",
        "tools",
        "context",
        "context_request_factory",
        "ids",
        "clock",
        "journal",
        "artifacts",
    ],
)
def test_composition_rejects_invalid_ports_before_filesystem_side_effects(tmp_path, invalid_name):
    from src.application.composition import ApplicationConfig, create_production_agent

    class Model:
        async def complete(self, _request):
            raise NotImplementedError

    class Tools:
        async def execute(self, _request):
            raise NotImplementedError

    class Context:
        async def build(self, _request, _budget):
            raise NotImplementedError

    class RequestFactory:
        def __call__(self, _state, _message):
            raise NotImplementedError

    class PortJournal:
        async def commit(self, *_args):
            raise NotImplementedError

        async def load_head(self, *_args):
            raise NotImplementedError

        async def load_session(self, *_args):
            raise NotImplementedError

        async def find_submission(self, *_args):
            raise NotImplementedError

        async def find_run_input(self, *_args):
            raise NotImplementedError

    class Artifacts:
        async def externalize(self, *_args):
            raise NotImplementedError

    sqlite_path = tmp_path / invalid_name / "journal.sqlite3"
    artifact_root = tmp_path / invalid_name / "artifacts"
    dependencies = {
        "model": Model(),
        "tools": Tools(),
        "context": Context(),
        "context_request_factory": RequestFactory(),
        "ids": Ids(),
        "clock": Clock(),
        "profile_id": ProfileId("profile"),
        "journal": PortJournal(),
        "artifacts": Artifacts(),
    }
    dependencies[invalid_name] = object()

    with pytest.raises(ValueError, match="Composition"):
        create_production_agent(
            ApplicationConfig(sqlite_path=sqlite_path, artifact_root=artifact_root),
            **dependencies,  # type: ignore[arg-type]
        )

    assert not sqlite_path.exists()
    assert not artifact_root.exists()


def test_composition_validation_does_not_execute_malicious_descriptors(tmp_path):
    from src.application.composition import ApplicationConfig, create_production_agent

    class HostileModel:
        @property
        def complete(self):
            raise AssertionError("descriptor executed")

    with pytest.raises(ValueError, match="Composition"):
        create_production_agent(
            ApplicationConfig(
                sqlite_path=tmp_path / "journal.sqlite3",
                artifact_root=tmp_path / "artifacts",
            ),
            model=HostileModel(),  # type: ignore[arg-type]
            tools=object(),  # type: ignore[arg-type]
            context=object(),  # type: ignore[arg-type]
            context_request_factory=object(),  # type: ignore[arg-type]
            ids=Ids(),
            clock=Clock(),
            profile_id=ProfileId("profile"),
        )


def test_partial_production_construction_closes_created_resources(monkeypatch, tmp_path):
    import src.application.composition as composition

    closed: list[str] = []

    class JournalPort:
        def __init__(self, _path):
            pass

        def close(self):
            closed.append("journal")

        async def commit(self, *_args):
            raise NotImplementedError

        async def load_head(self, *_args):
            raise NotImplementedError

        async def load_session(self, *_args):
            raise NotImplementedError

        async def find_submission(self, *_args):
            raise NotImplementedError

        async def find_run_input(self, *_args):
            raise NotImplementedError

    class FailingRepository:
        def __init__(self, *_args):
            raise RuntimeError("repository failed")

    monkeypatch.setattr(composition, "SQLiteRunJournal", JournalPort)
    monkeypatch.setattr(composition, "FilesystemArtifactRepository", FailingRepository)
    dependencies = _valid_composition_dependencies()

    with pytest.raises(RuntimeError, match="repository failed"):
        composition.create_production_agent(
            composition.ApplicationConfig(
                sqlite_path=tmp_path / "journal.sqlite3",
                artifact_root=tmp_path / "artifacts",
            ),
            **dependencies,
        )
    assert closed == ["journal"]


def test_successful_production_composition_owns_created_resources(monkeypatch, tmp_path):
    import src.application.composition as composition

    closed: list[str] = []

    class JournalPort:
        def __init__(self, _path):
            pass

        def close(self):
            closed.append("journal")

        async def commit(self, *_args):
            raise NotImplementedError

        async def load_head(self, *_args):
            raise NotImplementedError

        async def load_session(self, *_args):
            raise NotImplementedError

        async def find_submission(self, *_args):
            raise NotImplementedError

        async def find_run_input(self, *_args):
            raise NotImplementedError

    class Repository:
        def __init__(self, *_args):
            pass

        def close(self):
            closed.append("repository")

    class Externalizer:
        def __init__(self, _repository, _policy):
            pass

        def close(self):
            closed.append("externalizer")

        async def externalize(self, *_args):
            raise NotImplementedError

    monkeypatch.setattr(composition, "SQLiteRunJournal", JournalPort)
    monkeypatch.setattr(composition, "FilesystemArtifactRepository", Repository)
    monkeypatch.setattr(composition, "PolicyArtifactExternalizer", Externalizer)
    agent = composition.create_production_agent(
        composition.ApplicationConfig(
            sqlite_path=tmp_path / "journal.sqlite3",
            artifact_root=tmp_path / "artifacts",
        ),
        **_valid_composition_dependencies(),
    )

    import asyncio

    asyncio.run(agent.close())

    assert closed == ["externalizer", "repository", "journal"]


def _valid_composition_dependencies():
    class Model:
        async def complete(self, _request):
            raise NotImplementedError

    class Tools:
        async def execute(self, _request):
            raise NotImplementedError

    class Context:
        async def build(self, _request, _budget):
            raise NotImplementedError

    class RequestFactory:
        def __call__(self, _state, _message):
            raise NotImplementedError

    return {
        "model": Model(),
        "tools": Tools(),
        "context": Context(),
        "context_request_factory": RequestFactory(),
        "ids": Ids(),
        "clock": Clock(),
        "profile_id": ProfileId("profile"),
    }
