import asyncio
import sys
import threading
from pathlib import Path

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.tools import InMemoryToolExecutionStore, ToolCatalog
from src.adapters.tools.catalog import ToolCatalogEntry
from src.adapters.tools.docker import DockerInvocation, DockerToolGateway, _drain
from src.core.tools import ToolSpec


def test_invocation_rejects_nul() -> None:
    with pytest.raises(ValueError):
        DockerInvocation("echo\x00bad")


def test_gateway_requires_digest_pinned_image() -> None:
    with pytest.raises((TypeError, ValueError)):
        DockerToolGateway(  # type: ignore[call-arg]
            None,
            None,
            None,
            docker_executable=Path("docker"),
            image="latest",
            workspace_root=Path("."),
        )


def _gateway(tmp_path: Path) -> DockerToolGateway:
    externalizer = PolicyArtifactExternalizer(
        InMemoryArtifactRepository(1024),
        ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
    )
    return DockerToolGateway(
        ToolCatalog(()),
        InMemoryToolExecutionStore(),
        externalizer,
        docker_executable=Path(sys.executable),
        image="example.invalid/tool@sha256:" + "a" * 64,
        workspace_root=tmp_path.resolve(),
    )


def _configured_gateway(
    tmp_path: Path,
    catalog: ToolCatalog,
    *,
    builder_timeout_seconds: float = 0.02,
) -> DockerToolGateway:
    return DockerToolGateway(
        catalog,
        InMemoryToolExecutionStore(),
        PolicyArtifactExternalizer(
            InMemoryArtifactRepository(1024),
            ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
        ),
        docker_executable=Path(sys.executable),
        image="example.invalid/tool@sha256:" + "a" * 64,
        workspace_root=tmp_path.resolve(),
        builder_timeout_seconds=builder_timeout_seconds,
    )


def test_create_argv_contains_all_hardening_and_no_shell(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    args = gateway._create_arguments("container", DockerInvocation("echo ok"))
    joined = " ".join(args)
    for required in (
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges:true",
        "--pids-limit",
        "--memory",
        "--cpus",
        "--user 65534:65534",
        "--tmpfs",
        "--pull=never",
    ):
        assert required in joined
    assert args[-2:] == ["-lc", "echo ok"]
    assert gateway._host.startswith(("npipe://", "unix://"))


def test_command_timeout_and_cancel_kill_process(tmp_path: Path, monkeypatch) -> None:
    class Reader:
        async def read(self, _size):
            return b""

    class Process:
        returncode = None

        def __init__(self):
            self.killed = False
            self.stdout = Reader()
            self.stderr = Reader()

        def kill(self):
            self.killed = True

        async def wait(self):
            if not self.killed:
                await asyncio.sleep(10)
            return 0

    async def scenario():
        process = Process()

        async def create(*_args, **_kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        gateway = _gateway(tmp_path)
        with pytest.raises(TimeoutError):
            await gateway._command(["fake"], 0.01)
        assert process.killed

        process = Process()
        task = asyncio.create_task(gateway._command(["fake"], 10))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.killed

    asyncio.run(scenario())


def test_ambiguous_create_always_attempts_confirmed_cleanup(tmp_path: Path) -> None:
    async def scenario():
        gateway = _gateway(tmp_path)
        calls = []

        async def command(args, _timeout):
            calls.append(args)
            if "create" in args:
                raise TimeoutError
            return 0, b"", b"No such container"

        gateway._command = command  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            await gateway._run(DockerInvocation("echo ok"))
        assert any("rm" in args and "--force" in args for args in calls)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "host",
    ["tcp://localhost:2375", "unix://relative", "unix:///tmp/sock?x=1", "npipe://bad"],
)
def test_rejects_noncanonical_docker_hosts(tmp_path: Path, host: str) -> None:
    with pytest.raises(ValueError):
        DockerToolGateway(
            ToolCatalog(()),
            InMemoryToolExecutionStore(),
            PolicyArtifactExternalizer(
                InMemoryArtifactRepository(1024),
                ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
            ),
            docker_executable=Path(sys.executable),
            image="example.invalid/tool@sha256:" + "a" * 64,
            workspace_root=tmp_path.resolve(),
            docker_host=host,
        )


def test_sync_mutating_invocation_builder_is_rejected(tmp_path: Path) -> None:
    def builder(_arguments):
        return DockerInvocation("echo unsafe")

    catalog = ToolCatalog((ToolCatalogEntry(ToolSpec("docker", "docker", "true"), builder, True),))
    with pytest.raises(ValueError, match="Mutating Docker tools"):
        _configured_gateway(tmp_path, catalog)


def test_sync_nonmutating_builder_timeout_does_not_block_event_loop(tmp_path: Path) -> None:
    release = threading.Event()

    def builder(_arguments):
        release.wait()
        return DockerInvocation("echo late")

    gateway = _configured_gateway(
        tmp_path,
        ToolCatalog((ToolCatalogEntry(ToolSpec("docker", "docker", "true"), builder, False),)),
        builder_timeout_seconds=0.01,
    )

    async def scenario():
        try:
            with pytest.raises(TimeoutError):
                await gateway._build_invocation(builder, {})
        finally:
            release.set()

    asyncio.run(scenario())


def test_async_builder_timeout_and_caller_cancellation_are_bounded(tmp_path: Path) -> None:
    cancelled = asyncio.Event()

    async def builder(_arguments):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.set()

    gateway = _configured_gateway(
        tmp_path,
        ToolCatalog((ToolCatalogEntry(ToolSpec("docker", "docker", "true"), builder, True),)),
        builder_timeout_seconds=0.01,
    )

    async def scenario():
        with pytest.raises(TimeoutError):
            await gateway._build_invocation(builder, {})
        assert cancelled.is_set()
        cancelled.clear()
        task = asyncio.create_task(gateway._build_invocation(builder, {}))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_remove_failure_reader_failure_and_path_redaction(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)

    async def scenario():
        async def failed(_args, _timeout):
            return (1, b"", b"")

        gateway._command = failed  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="cleanup"):
            await gateway._remove("container")

        class BrokenReader:
            async def read(self, _size):
                raise OSError("reader failed")

        with pytest.raises(OSError):
            await _drain(BrokenReader(), 8)  # type: ignore[arg-type]

    asyncio.run(scenario())
    raw = f"failure at {tmp_path.resolve()}".encode()
    assert str(tmp_path.resolve()) not in gateway._sanitize(raw)
