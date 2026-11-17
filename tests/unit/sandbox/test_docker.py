import io
import os
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from src.sandbox import DockerSandboxError, DockerSandboxRunner
from src.schema.tool import SandboxKind
from src.tool.local_tools import LocalToolSuite
from src.workspace import Workspace, WorkspaceViolation

_IMAGE = "agent-runtime@sha256:" + "a" * 64
_EXECUTABLE = Path(sys.executable).resolve()


class _Process:
    def __init__(self, stdout=b"", stderr=b"", returncode=0, *, times_out=False):
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode
        self.times_out = times_out
        self.killed = False

    def wait(self, timeout=None):
        if self.times_out:
            self.times_out = False
            raise subprocess.TimeoutExpired("docker", timeout)
        return self.returncode

    def kill(self):
        self.killed = True


class _BlockingStream:
    def __init__(self):
        self.read_started = threading.Event()
        self.released = threading.Event()

    def read(self, _size):
        self.read_started.set()
        self.released.wait(10)
        return b""

    def close(self):
        self.released.set()


class _FailingStream:
    def read(self, _size):
        raise OSError("capture failed")

    def close(self):
        return None


def _runner(**kwargs):
    return DockerSandboxRunner(_IMAGE, docker_executable=_EXECUTABLE, **kwargs)


def _shell(tmp_path):
    workspace = Workspace(tmp_path)
    [shell] = [
        tool for tool in LocalToolSuite(workspace).tools(include_shell=True) if tool.name == "shell"
    ]
    return workspace, shell


def _control_success(args, **_kwargs):
    return subprocess.CompletedProcess(args, 0, stdout=b"created\n", stderr=b"")


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("agent-runtime:latest", "digest"),
        ("agent runtime@sha256:" + "a" * 64, "image"),
        ("--privileged@sha256:" + "a" * 64, "image"),
        ("agent-runtime@sha256:short", "digest"),
    ],
)
def test_constructor_rejects_mutable_or_invalid_image_references(image, message):
    with pytest.raises(ValueError, match=message):
        DockerSandboxRunner(image, docker_executable=_EXECUTABLE)


@pytest.mark.parametrize("executable", ["docker", Path("relative/docker"), Path("missing/docker")])
def test_constructor_requires_an_existing_absolute_docker_executable(executable):
    with pytest.raises(DockerSandboxError, match="absolute|available"):
        DockerSandboxRunner(_IMAGE, docker_executable=executable)


@pytest.mark.parametrize("docker_host", ["tcp://127.0.0.1:2375", "ssh://builder", "http://x"])
def test_constructor_rejects_nonlocal_docker_endpoints(docker_host):
    with pytest.raises(ValueError, match="local"):
        DockerSandboxRunner(
            _IMAGE,
            docker_executable=_EXECUTABLE,
            docker_host=docker_host,
        )


@pytest.mark.parametrize("user", ["0:0", "root", "1000:0", "0:1000", "1000"])
def test_constructor_requires_a_numeric_nonroot_uid_and_gid(user):
    with pytest.raises(ValueError, match="user"):
        DockerSandboxRunner(_IMAGE, docker_executable=_EXECUTABLE, user=user)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_timeout": 0},
        {"max_timeout": float("inf")},
        {"default_timeout": 10, "max_timeout": 5},
        {"memory_bytes": 0},
        {"tmpfs_bytes": True},
        {"pids_limit": 0},
        {"cpus": 0},
        {"max_output_bytes": 0},
        {"cleanup_timeout": 0},
    ],
)
def test_constructor_rejects_unsafe_limits(kwargs):
    with pytest.raises(ValueError):
        DockerSandboxRunner(_IMAGE, docker_executable=_EXECUTABLE, **kwargs)


def test_supports_only_explicit_command_sandbox_tools(tmp_path):
    runner = _runner()
    _workspace, shell = _shell(tmp_path)

    assert runner.supports(shell) is True
    assert runner.supports(replace(shell, requires_sandbox=False)) is False
    assert runner.supports(replace(shell, sandbox_kind=None)) is False
    assert runner.supports(replace(shell, sandbox_kind=SandboxKind.COMMAND)) is True


def test_run_rejects_a_docker_executable_inside_the_writable_workspace(tmp_path):
    executable = tmp_path / "docker.exe"
    executable.write_bytes(b"fake")
    executable.chmod(0o755)
    workspace, shell = _shell(tmp_path)
    runner = DockerSandboxRunner(_IMAGE, docker_executable=executable)

    with pytest.raises(WorkspaceViolation, match="executable") as caught:
        runner.run(shell, {"command": "echo ok"}, workspace)

    assert str(tmp_path) not in str(caught.value)


def test_run_creates_locked_down_container_then_starts_and_removes_it(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    (tmp_path / "nested").mkdir()
    process = _Process(
        stdout="完成\n".encode(),
        stderr=f"warning at {workspace.root}\n".encode(),
        returncode=7,
    )
    control_calls = []
    started = []

    def control(args, **kwargs):
        control_calls.append((args, kwargs))
        return _control_success(args)

    def popen(args, **kwargs):
        started.append((args, kwargs))
        return process

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", control)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", popen)
    runner = _runner(
        default_timeout=12,
        max_timeout=30,
        memory_bytes=128 * 1024 * 1024,
        tmpfs_bytes=8 * 1024 * 1024,
        pids_limit=32,
        cpus=0.5,
    )

    result = runner.run(shell, {"command": "printf ok", "cwd": "nested", "timeout": 20}, workspace)

    assert result == {
        "exit_code": 7,
        "stdout": "完成\n",
        "stderr": "warning at <workspace>\n",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }
    create_args, create_options = control_calls[0]
    assert "create" in create_args
    assert create_options["shell"] is False
    assert create_args[create_args.index("--network") + 1] == "none"
    assert "--read-only" in create_args
    assert create_args[create_args.index("--cap-drop") + 1] == "ALL"
    assert create_args[create_args.index("--security-opt") + 1] == "no-new-privileges:true"
    assert create_args[create_args.index("--pids-limit") + 1] == "32"
    assert create_args[create_args.index("--memory") + 1] == str(128 * 1024 * 1024)
    assert create_args[create_args.index("--memory-swap") + 1] == str(128 * 1024 * 1024)
    assert create_args[create_args.index("--cpus") + 1] == "0.5"
    assert create_args[create_args.index("--workdir") + 1] == "/workspace/nested"
    assert create_args[create_args.index("--entrypoint") + 1] == "/bin/sh"
    assert create_args[create_args.index("--host") + 1].startswith(("unix://", "npipe://"))
    mount = create_args[create_args.index("--mount") + 1]
    assert mount == f"type=bind,source={workspace.root},target=/workspace"
    assert create_args[-3:] == [_IMAGE, "-lc", "printf ok"]
    [start_call] = started
    start_args, start_options = start_call
    assert start_args[-3:-1] == ["start", "--attach"]
    assert start_options["shell"] is False
    assert start_options["stdin"] is subprocess.DEVNULL
    assert start_options["stdout"] is subprocess.PIPE
    assert start_options["stderr"] is subprocess.PIPE
    cleanup_args, _cleanup_options = control_calls[-1]
    assert cleanup_args[-3:-1] == ["rm", "--force"]


def test_run_revalidates_inputs_without_creating_a_container(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    runner = _runner(default_timeout=5, max_timeout=10)
    monkeypatch.setattr(
        "src.sandbox.docker.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("Docker must not start"),
    )

    for inputs in [
        {"command": ""},
        {"command": "echo ok\x00"},
        {"command": "echo ok", "timeout": 11},
        {"command": "echo ok", "timeout": True},
        {"command": "echo ok", "cwd": "../outside"},
        {"command": "echo ok", "cwd": "missing"},
    ]:
        with pytest.raises((ValueError, WorkspaceViolation)):
            runner.run(shell, inputs, workspace)


def test_run_rejects_ambiguous_workspace_mount_paths(tmp_path):
    root = tmp_path / "with,comma"
    root.mkdir()
    workspace, shell = _shell(root)
    runner = _runner()

    with pytest.raises(WorkspaceViolation, match="mount") as caught:
        runner.run(shell, {"command": "echo ok"}, workspace)

    assert str(root) not in str(caught.value)


def test_run_bounds_both_output_streams_while_draining_them(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process(stdout=b"abcdefgh", stderr=b"12345678")
    monkeypatch.setattr("src.sandbox.docker.subprocess.run", _control_success)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner(max_output_bytes=5)

    result = runner.run(shell, {"command": "noisy"}, workspace)

    assert result["stdout"] == "abcde"
    assert result["stderr"] == "12345"
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert process.stdout.tell() == 8
    assert process.stderr.tell() == 8


def test_timeout_kills_cli_removes_container_before_bounded_reader_join(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    blocked = _BlockingStream()
    process = _Process(times_out=True)
    process.stdout = blocked
    cleanup_called = threading.Event()

    def control(args, **kwargs):
        if "rm" in args:
            cleanup_called.set()
            blocked.close()
        return _control_success(args)

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", control)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner(default_timeout=0.01, max_timeout=1, cleanup_timeout=0.1)

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="exceeded"):
        runner.run(shell, {"command": "sleep 60"}, workspace)

    assert time.monotonic() - started < 1
    assert process.killed is True
    assert cleanup_called.is_set()
    assert blocked.read_started.is_set()


def test_capture_failure_after_timeout_cannot_skip_container_cleanup(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process(times_out=True)
    process.stdout = _FailingStream()
    cleanup_calls = []

    def control(args, **kwargs):
        if "rm" in args:
            cleanup_calls.append(args)
        return _control_success(args)

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", control)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner(default_timeout=0.01, max_timeout=1, cleanup_timeout=0.1)

    with pytest.raises(TimeoutError, match="exceeded"):
        runner.run(shell, {"command": "sleep 60"}, workspace)

    assert cleanup_calls


def test_output_setup_failure_cannot_skip_container_cleanup(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process()
    process.stdout = None
    cleanup_calls = []

    def control(args, **kwargs):
        if "rm" in args:
            cleanup_calls.append(args)
        return _control_success(args)

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", control)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner()

    with pytest.raises(DockerSandboxError, match="output"):
        runner.run(shell, {"command": "true"}, workspace)

    assert cleanup_calls


def test_buffered_reader_lock_cannot_defeat_the_cleanup_deadline(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "rb")
    writer = os.fdopen(write_fd, "wb")
    process = _Process()
    process.stdout = reader
    monkeypatch.setattr("src.sandbox.docker.subprocess.run", _control_success)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner(cleanup_timeout=0.02)
    finished = threading.Event()
    errors = []

    def execute():
        try:
            runner.run(shell, {"command": "wait for pipe"}, workspace)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=execute, daemon=True)
    worker.start()
    returned_before_release = finished.wait(0.3)
    writer.close()
    worker.join(1)
    reader.close()

    assert returned_before_release is True
    assert len(errors) == 1
    assert isinstance(errors[0], DockerSandboxError)
    assert "capture" in str(errors[0])


def test_cleanup_failure_overrides_a_successful_command_result(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process()

    def control(args, **kwargs):
        return subprocess.CompletedProcess(args, 1 if "rm" in args else 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", control)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner()

    with pytest.raises(DockerSandboxError, match="cleanup"):
        runner.run(shell, {"command": "true"}, workspace)


def test_process_start_failure_removes_created_container_and_is_sanitized(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    cleanup_calls = []

    def control(args, **kwargs):
        if "rm" in args:
            cleanup_calls.append(args)
        return _control_success(args)

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", control)
    monkeypatch.setattr(
        "src.sandbox.docker.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(f"denied {workspace.root}")),
    )
    runner = _runner()

    with pytest.raises(DockerSandboxError, match="start") as caught:
        runner.run(shell, {"command": "echo ok"}, workspace)

    assert cleanup_calls
    assert str(workspace.root) not in str(caught.value)
