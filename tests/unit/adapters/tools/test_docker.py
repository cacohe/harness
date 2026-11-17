import asyncio
import os
from pathlib import Path

import pytest
from src.adapters.artifacts.externalizer import ArtifactPolicy, PolicyArtifactExternalizer
from src.adapters.artifacts.memory import InMemoryArtifactRepository
from src.adapters.tools import InMemoryToolExecutionStore, ToolCatalog
from src.adapters.tools.docker import DockerInvocation, DockerToolGateway, _drain


def test_invocation_rejects_nul() -> None:
    with pytest.raises(ValueError):
        DockerInvocation("echo\x00bad")


def test_gateway_requires_digest_pinned_image() -> None:
    with pytest.raises((TypeError, ValueError)):
        DockerToolGateway(  # type: ignore[call-arg]
            None, None, None,
            docker_executable=Path("docker"), image="latest", workspace_root=Path("."),
        )


def _gateway(tmp_path: Path) -> DockerToolGateway:
    externalizer = PolicyArtifactExternalizer(
        InMemoryArtifactRepository(1024),
        ArtifactPolicy(512, 1024, 128, frozenset({"application/json"})),
    )
    return DockerToolGateway(
        ToolCatalog(()), InMemoryToolExecutionStore(), externalizer,
        docker_executable=Path(os.environ["COMSPEC"]),
        image="example.invalid/tool@sha256:" + "a" * 64,
        workspace_root=tmp_path.resolve(),
    )


def test_create_argv_contains_all_hardening_and_no_shell(tmp_path: Path) -> None:
    gateway = _gateway(tmp_path)
    args = gateway._create_arguments("container", DockerInvocation("echo ok"))
    joined = " ".join(args)
    for required in (
        "--network none", "--read-only", "--cap-drop ALL",
        "no-new-privileges:true", "--pids-limit", "--memory", "--cpus",
        "--user 65534:65534", "--tmpfs", "--pull=never",
    ):
        assert required in joined
    assert args[-2:] == ["-lc", "echo ok"]
    assert gateway._host.startswith(("npipe://", "unix://"))


def test_command_timeout_and_cancel_kill_process(tmp_path: Path, monkeypatch) -> None:
    class Process:
        returncode = None

        def __init__(self):
            self.killed = False

        async def communicate(self):
            await asyncio.sleep(10)

        def kill(self):
            self.killed = True

        async def wait(self):
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
