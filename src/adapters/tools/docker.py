from __future__ import annotations

import asyncio
import math
import os
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from src.adapters.tools.catalog import ToolCatalog
from src.adapters.tools.execution_store import ToolExecutionStore
from src.adapters.tools.local import LocalToolGateway
from src.core.artifacts.contracts import ArtifactExternalizer
from src.core.execution.errors import RunErrorCode, RunFailure
from src.core.execution.events import JsonValue
from src.core.tools.contracts import ToolRequest, ToolResult, ToolResultStatus

_IMAGE = re.compile(r"[^\s\x00-\x1f]+@sha256:[0-9a-f]{64}")
_USER = re.compile(r"[1-9][0-9]*:[1-9][0-9]*")


@dataclass(frozen=True, slots=True)
class DockerInvocation:
    command: str
    cwd: str = "."

    def __post_init__(self) -> None:
        if type(self.command) is not str or not self.command.strip() or "\x00" in self.command:
            raise ValueError("Docker command is invalid")
        if type(self.cwd) is not str or not self.cwd or "\x00" in self.cwd:
            raise ValueError("Docker working directory is invalid")


class DockerToolGateway(LocalToolGateway):
    """One-shot hardened Docker execution with confirmed cleanup."""

    def __init__(
        self,
        catalog: ToolCatalog,
        store: ToolExecutionStore,
        externalizer: ArtifactExternalizer,
        *,
        docker_executable: Path,
        image: str,
        workspace_root: Path,
        docker_host: str | None = None,
        timeout_seconds: float = 60.0,
        builder_timeout_seconds: float = 5.0,
        cleanup_timeout_seconds: float = 10.0,
        max_output_bytes: int = 1_048_576,
        memory_bytes: int = 536_870_912,
        pids_limit: int = 128,
        cpus: float = 1.0,
        tmpfs_bytes: int = 67_108_864,
        user: str = "65534:65534",
    ) -> None:
        super().__init__(catalog, store, externalizer, timeout_seconds=timeout_seconds)
        if any(
            entry.mutating and not asyncio.iscoroutinefunction(entry.handler)
            for entry in catalog.entries
        ):
            raise ValueError("Mutating Docker tools require async invocation builders")
        if type(docker_executable) is not type(Path()) or not docker_executable.is_absolute():
            raise ValueError("Docker CLI must be an explicit absolute path")
        resolved_cli = docker_executable.resolve(strict=True)
        if not resolved_cli.is_file() or os.name == "nt" and resolved_cli.suffix.lower() != ".exe":
            raise ValueError("Docker CLI must be a native executable")
        if type(image) is not str or image.startswith("-") or _IMAGE.fullmatch(image) is None:
            raise ValueError("Docker image must be digest pinned")
        if type(workspace_root) is not type(Path()) or not workspace_root.is_absolute():
            raise ValueError("Docker workspace must be an absolute path")
        root = workspace_root.resolve(strict=True)
        if not root.is_dir() or resolved_cli.is_relative_to(root):
            raise ValueError("Docker workspace is invalid")
        host = docker_host or (
            "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock"
        )
        if type(host) is not str or not _local_docker_host(host):
            raise ValueError("Docker host must be local")
        numeric = (memory_bytes, pids_limit, max_output_bytes, tmpfs_bytes)
        if any(type(item) is not int or item <= 0 for item in numeric):
            raise ValueError("Docker resource limits are invalid")
        if (
            isinstance(cpus, bool)
            or type(cpus) not in (int, float)
            or not math.isfinite(cpus)
            or cpus <= 0
        ):
            raise ValueError("Docker CPU limit is invalid")
        if type(user) is not str or _USER.fullmatch(user) is None:
            raise ValueError("Docker user must be non-root")
        self._docker = str(resolved_cli)
        self._image = image
        self._root = root
        self._host = host
        self._cleanup_timeout = _duration(cleanup_timeout_seconds, "Docker cleanup timeout")
        self._builder_timeout = _duration(builder_timeout_seconds, "Docker builder timeout")
        self._output_limit = max_output_bytes
        self._memory = memory_bytes
        self._pids = pids_limit
        self._cpus = float(cpus)
        self._tmpfs = tmpfs_bytes
        self._user = user

    async def _invoke(
        self,
        request: ToolRequest,
        handler: Callable[[dict[str, object]], object | Awaitable[object]],
        arguments: dict[str, object],
    ) -> ToolResult:
        try:
            candidate = await self._build_invocation(handler, arguments)
            output = await self._run(candidate)
            content = await self._externalizer.externalize(output, "application/json")
            return ToolResult(
                ToolResultStatus.SUCCESS,
                inline_content=content.inline_content,
                artifacts=content.artifacts,
                summary=content.summary,
                execution_key=request.idempotency_key,
            )
        except TimeoutError:
            return ToolResult(
                ToolResultStatus.TIMED_OUT,
                failure=RunFailure(RunErrorCode.TOOL_TIMED_OUT),
                execution_key=request.idempotency_key,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ToolResult(
                ToolResultStatus.FAILED,
                failure=RunFailure(RunErrorCode.TOOL_FAILED),
                execution_key=request.idempotency_key,
            )

    async def _build_invocation(
        self,
        handler: Callable[[dict[str, object]], object | Awaitable[object]],
        arguments: dict[str, object],
    ) -> DockerInvocation:
        if asyncio.iscoroutinefunction(handler):
            candidate = await asyncio.wait_for(handler(arguments), timeout=self._builder_timeout)
        else:
            candidate = await asyncio.wait_for(
                asyncio.to_thread(handler, arguments), timeout=self._builder_timeout
            )
        if type(candidate) is not DockerInvocation:
            raise ValueError("Docker invocation builder returned an invalid value")
        return candidate

    async def _run(self, invocation: DockerInvocation) -> dict[str, JsonValue]:
        name = f"harness-{secrets.token_hex(16)}"
        create = self._create_arguments(name, invocation)
        process: asyncio.subprocess.Process | None = None
        try:
            result = await self._command(create, self._cleanup_timeout)
            if result[0] != 0:
                raise RuntimeError("Docker create failed")
            process = await asyncio.create_subprocess_exec(
                *self._prefix(),
                "start",
                "--attach",
                name,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            assert process.stdout is not None and process.stderr is not None
            stdout_task = asyncio.create_task(_drain(process.stdout, self._output_limit))
            stderr_task = asyncio.create_task(_drain(process.stderr, self._output_limit))
            wait_task = asyncio.create_task(process.wait())
            try:
                async with asyncio.timeout(self._timeout):
                    await asyncio.wait(
                        (stdout_task, stderr_task, wait_task),
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    stdout, stderr, exit_code = await asyncio.gather(
                        stdout_task, stderr_task, wait_task
                    )
            except BaseException:
                process.kill()
                await process.wait()
                await asyncio.gather(stdout_task, stderr_task, wait_task, return_exceptions=True)
                raise
            return {
                "exit_code": exit_code,
                "stdout": self._sanitize(stdout[0]),
                "stderr": self._sanitize(stderr[0]),
                "stdout_truncated": stdout[1],
                "stderr_truncated": stderr[1],
            }
        finally:
            # A create timeout/error is ambiguous: the daemon may have created the name.
            cleanup = asyncio.create_task(self._remove(name))
            await _finish_cleanup(cleanup)

    def _create_arguments(self, name: str, invocation: DockerInvocation) -> list[str]:
        cwd = (self._root / invocation.cwd).resolve(strict=True)
        if not cwd.is_dir() or not cwd.is_relative_to(self._root):
            raise ValueError("Docker working directory escapes workspace")
        relative = cwd.relative_to(self._root).as_posix()
        container_cwd = "/workspace" if relative == "." else f"/workspace/{relative}"
        mount = str(self._root)
        if any(item in mount for item in (",", "\r", "\n", "\x00")):
            raise ValueError("Docker workspace cannot be mounted safely")
        return [
            *self._prefix(),
            "create",
            "--name",
            name,
            "--pull=never",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--init",
            "--pids-limit",
            str(self._pids),
            "--memory",
            str(self._memory),
            "--memory-swap",
            str(self._memory),
            "--cpus",
            f"{self._cpus:g}",
            "--ulimit",
            "nofile=1024:1024",
            "--user",
            self._user,
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self._tmpfs}",
            "--mount",
            f"type=bind,source={mount},target=/workspace",
            "--workdir",
            container_cwd,
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            "--entrypoint",
            "/bin/sh",
            self._image,
            "-lc",
            invocation.command,
        ]

    def _prefix(self) -> list[str]:
        return [self._docker, "--host", self._host]

    async def _command(self, args: list[str], timeout: float) -> tuple[int, bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            await process.wait()
            raise RuntimeError("Docker CLI output is unavailable")
        stdout_task = asyncio.create_task(_drain(process.stdout, self._output_limit))
        stderr_task = asyncio.create_task(_drain(process.stderr, self._output_limit))
        wait_task = asyncio.create_task(process.wait())
        try:
            async with asyncio.timeout(timeout):
                await asyncio.wait(
                    (stdout_task, stderr_task, wait_task),
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                stdout, stderr, returncode = await asyncio.gather(
                    stdout_task, stderr_task, wait_task
                )
        except BaseException:
            process.kill()
            await process.wait()
            await asyncio.gather(stdout_task, stderr_task, wait_task, return_exceptions=True)
            raise
        return returncode, stdout[0], stderr[0]

    async def _remove(self, name: str) -> None:
        code, _, stderr = await self._command(
            [*self._prefix(), "rm", "--force", name], self._cleanup_timeout
        )
        missing = b"No such container" in stderr or b"not found" in stderr.lower()
        if code != 0 and not missing:
            raise RuntimeError("Docker cleanup could not be confirmed")

    def _sanitize(self, value: bytes) -> str:
        text = value.decode("utf-8", errors="replace")
        for root in {str(self._root), str(self._root).replace("\\", "/")}:
            text = text.replace(root, "<workspace>")
        return text


async def _drain(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    captured = bytearray()
    truncated = False
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            return bytes(captured), truncated
        available = limit - len(captured)
        if available > 0:
            captured.extend(chunk[:available])
        if len(chunk) > available:
            truncated = True


def _local_docker_host(value: str) -> bool:
    if value == "npipe:////./pipe/docker_engine":
        return True
    if not value.startswith("unix:///") or any(character.isspace() for character in value):
        return False
    path = value[len("unix://") :]
    return path.startswith("/") and "?" not in path and "#" not in path and "\x00" not in path


def _duration(value: object, label: str) -> float:
    if isinstance(value, bool) or type(value) not in (int, float):
        raise ValueError(f"{label} is invalid")
    result = float(cast(int | float, value))
    if not math.isfinite(result) or not 0.001 <= result <= 300:
        raise ValueError(f"{label} is invalid")
    return result


async def _finish_cleanup(task: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
    # Propagate cleanup failure before cancellation: cleanup confirmation is mandatory.
    await task
    if cancellation is not None:
        raise cancellation
