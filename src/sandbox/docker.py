from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import BinaryIO

from src.core.workspace import Workspace, WorkspaceViolation
from src.schema.tool import SandboxKind, Tool

_DIGEST_IMAGE = re.compile(r"^[^\s\x00-\x1f]+@sha256:[0-9a-fA-F]{64}$")


class DockerSandboxError(RuntimeError):
    """A Docker sandbox operation failed without exposing host details."""


class _BoundedCapture:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self.error: Exception | None = None

    def drain(self, stream: BinaryIO) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                available = self.limit - len(self.data)
                if available > 0:
                    self.data.extend(chunk[:available])
                if len(chunk) > available:
                    self.truncated = True
        except Exception as error:
            self.error = error

    def text(self, workspace: Workspace) -> str:
        text = bytes(self.data).decode("utf-8", errors="replace")
        roots = {str(workspace.root), str(workspace.root).replace("\\", "/")}
        for root in roots:
            if root:
                text = text.replace(root, "<workspace>")
        return text


class DockerSandboxRunner:
    """Execute command-capability tools in locked-down one-shot Linux containers."""

    def __init__(
        self,
        image: str,
        *,
        docker_executable: str = "docker",
        require_digest: bool = True,
        default_timeout: float = 60.0,
        max_timeout: float = 300.0,
        cleanup_timeout: float = 10.0,
        memory_bytes: int = 512 * 1024 * 1024,
        tmpfs_bytes: int = 64 * 1024 * 1024,
        pids_limit: int = 128,
        cpus: float = 1.0,
        max_output_bytes: int = 1024 * 1024,
        user: str = "65534:65534",
        shell: str = "/bin/sh",
    ) -> None:
        if (
            not isinstance(image, str)
            or not image
            or any(character.isspace() or ord(character) < 32 for character in image)
        ):
            raise ValueError("Docker image reference is invalid")
        if require_digest and _DIGEST_IMAGE.fullmatch(image) is None:
            raise ValueError("Docker image must use an immutable sha256 digest")
        self.default_timeout = self._positive_number(default_timeout, "default_timeout")
        self.max_timeout = self._positive_number(max_timeout, "max_timeout")
        if self.default_timeout > self.max_timeout:
            raise ValueError("default_timeout must not exceed max_timeout")
        self.cleanup_timeout = self._positive_number(cleanup_timeout, "cleanup_timeout")
        self.memory_bytes = self._positive_integer(memory_bytes, "memory_bytes")
        self.tmpfs_bytes = self._positive_integer(tmpfs_bytes, "tmpfs_bytes")
        self.pids_limit = self._positive_integer(pids_limit, "pids_limit")
        self.max_output_bytes = self._positive_integer(max_output_bytes, "max_output_bytes")
        self.cpus = self._positive_number(cpus, "cpus")
        self.user = self._safe_config_text(user, "user")
        self.shell = self._safe_config_text(shell, "shell")
        if not self.shell.startswith("/"):
            raise ValueError("shell must be an absolute container path")
        executable = shutil.which(docker_executable)
        if executable is None:
            raise DockerSandboxError("Docker CLI is not available")
        self.executable = str(Path(executable).resolve(strict=False))
        self.image = image

    @staticmethod
    def _positive_number(value: object, name: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a positive finite number")
        return float(value)

    @staticmethod
    def _positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _safe_config_text(value: object, name: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"{name} is invalid")
        return value

    def supports(self, tool: Tool) -> bool:
        return tool.requires_sandbox and tool.sandbox_kind is SandboxKind.COMMAND

    def run(self, tool: Tool, inputs: dict, workspace: Workspace) -> dict[str, object]:
        if not self.supports(tool):
            raise DockerSandboxError("tool does not declare command sandbox capability")
        command = inputs.get("command")
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise ValueError("sandbox command must be a non-empty string without NUL bytes")
        timeout = self._call_timeout(inputs.get("timeout"))
        container_cwd = self._container_cwd(workspace, inputs.get("cwd", "."))
        source = str(workspace.root)
        if any(character in source for character in {",", "\r", "\n", "\x00"}):
            raise WorkspaceViolation("workspace path cannot be encoded as a Docker mount")
        container_name = f"agent-harness-{uuid.uuid4().hex}"
        arguments = self._arguments(container_name, source, container_cwd, command)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                creationflags=creationflags,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            raise DockerSandboxError("could not start Docker sandbox") from error

        assert process.stdout is not None
        assert process.stderr is not None
        stdout = _BoundedCapture(self.max_output_bytes)
        stderr = _BoundedCapture(self.max_output_bytes)
        readers = [
            threading.Thread(target=stdout.drain, args=(process.stdout,)),
            threading.Thread(target=stderr.drain, args=(process.stderr,)),
        ]
        for reader in readers:
            reader.start()

        timed_out = False
        wait_error: Exception | None = None
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            exit_code = process.wait()
        except Exception as error:
            wait_error = error
            process.kill()
            exit_code = process.wait()
        finally:
            for reader in readers:
                reader.join()

        if stdout.error is not None or stderr.error is not None:
            raise DockerSandboxError("could not capture Docker sandbox output")
        if timed_out or wait_error is not None:
            if not self._force_remove(container_name, creationflags):
                raise DockerSandboxError("sandbox container cleanup could not be confirmed")
        if timed_out:
            raise TimeoutError("sandbox command exceeded its timeout")
        if wait_error is not None:
            raise DockerSandboxError("Docker sandbox process failed") from wait_error
        return {
            "exit_code": int(exit_code),
            "stdout": stdout.text(workspace),
            "stderr": stderr.text(workspace),
            "stdout_truncated": stdout.truncated,
            "stderr_truncated": stderr.truncated,
        }

    def _call_timeout(self, value: object) -> float:
        if value is None:
            return self.default_timeout
        timeout = self._positive_number(value, "timeout")
        if timeout > self.max_timeout:
            raise ValueError("timeout exceeds the configured maximum")
        return timeout

    @staticmethod
    def _container_cwd(workspace: Workspace, value: object) -> str:
        if not isinstance(value, str):
            raise WorkspaceViolation("sandbox working directory must be workspace-relative")
        resolved = workspace.resolve(value)
        if not resolved.is_dir():
            raise WorkspaceViolation("sandbox working directory is not an existing directory")
        relative = workspace.relative(resolved)
        return "/workspace" if relative == "." else f"/workspace/{relative}"

    def _arguments(
        self, container_name: str, source: str, container_cwd: str, command: str
    ) -> list[str]:
        return [
            self.executable,
            "run",
            "--rm",
            "--name",
            container_name,
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
            str(self.pids_limit),
            "--memory",
            str(self.memory_bytes),
            "--memory-swap",
            str(self.memory_bytes),
            "--cpus",
            f"{self.cpus:g}",
            "--ulimit",
            "nofile=1024:1024",
            "--user",
            self.user,
            "--tmpfs",
            f"/tmp:rw,nosuid,nodev,size={self.tmpfs_bytes}",
            "--mount",
            f"type=bind,source={source},target=/workspace",
            "--workdir",
            container_cwd,
            "--env",
            "HOME=/tmp",
            "--env",
            "TMPDIR=/tmp",
            self.image,
            self.shell,
            "-lc",
            command,
        ]

    def _force_remove(self, container_name: str, creationflags: int) -> bool:
        try:
            result = subprocess.run(
                [self.executable, "rm", "--force", container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.cleanup_timeout,
                shell=False,
                creationflags=creationflags,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0
