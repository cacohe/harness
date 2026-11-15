import io
import subprocess
from dataclasses import replace

import pytest
from src.core.workspace import Workspace, WorkspaceViolation
from src.sandbox import DockerSandboxError, DockerSandboxRunner
from src.schema.tool import SandboxKind
from src.tool.local_tools import LocalToolSuite

_IMAGE = "agent-runtime@sha256:" + "a" * 64


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


def _runner(monkeypatch, **kwargs):
    monkeypatch.setattr("src.sandbox.docker.shutil.which", lambda _name: "docker")
    return DockerSandboxRunner(_IMAGE, **kwargs)


def _shell(tmp_path):
    workspace = Workspace(tmp_path)
    [shell] = [
        tool for tool in LocalToolSuite(workspace).tools(include_shell=True) if tool.name == "shell"
    ]
    return workspace, shell


@pytest.mark.parametrize(
    ("image", "message"),
    [
        ("agent-runtime:latest", "digest"),
        ("agent runtime@sha256:" + "a" * 64, "image"),
        ("agent-runtime@sha256:short", "digest"),
    ],
)
def test_constructor_rejects_mutable_or_invalid_image_references(monkeypatch, image, message):
    monkeypatch.setattr("src.sandbox.docker.shutil.which", lambda _name: "docker")

    with pytest.raises(ValueError, match=message):
        DockerSandboxRunner(image)


def test_constructor_rejects_an_unavailable_docker_cli(monkeypatch):
    monkeypatch.setattr("src.sandbox.docker.shutil.which", lambda _name: None)

    with pytest.raises(DockerSandboxError, match="Docker CLI"):
        DockerSandboxRunner(_IMAGE)


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
def test_constructor_rejects_unsafe_limits(monkeypatch, kwargs):
    monkeypatch.setattr("src.sandbox.docker.shutil.which", lambda _name: "docker")

    with pytest.raises(ValueError):
        DockerSandboxRunner(_IMAGE, **kwargs)


def test_supports_only_explicit_command_sandbox_tools(tmp_path, monkeypatch):
    runner = _runner(monkeypatch)
    _workspace, shell = _shell(tmp_path)

    assert runner.supports(shell) is True
    assert runner.supports(replace(shell, requires_sandbox=False)) is False
    assert runner.supports(replace(shell, sandbox_kind=None)) is False
    assert runner.supports(replace(shell, sandbox_kind=SandboxKind.COMMAND)) is True


def test_run_uses_locked_down_container_arguments_and_returns_json_safe_output(
    tmp_path, monkeypatch
):
    workspace, shell = _shell(tmp_path)
    (tmp_path / "nested").mkdir()
    process = _Process(
        stdout="完成\n".encode(),
        stderr=f"warning at {workspace.root}\n".encode(),
        returncode=7,
    )
    started = []

    def popen(args, **kwargs):
        started.append((args, kwargs))
        return process

    monkeypatch.setattr("src.sandbox.docker.shutil.which", lambda _name: "docker")
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", popen)
    runner = DockerSandboxRunner(
        _IMAGE,
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
    [invocation] = started
    args, options = invocation
    assert options["shell"] is False
    assert options["stdin"] is subprocess.DEVNULL
    assert options["stdout"] is subprocess.PIPE
    assert options["stderr"] is subprocess.PIPE
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert "--read-only" in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges:true"
    assert args[args.index("--pids-limit") + 1] == "32"
    assert args[args.index("--memory") + 1] == str(128 * 1024 * 1024)
    assert args[args.index("--memory-swap") + 1] == str(128 * 1024 * 1024)
    assert args[args.index("--cpus") + 1] == "0.5"
    assert args[args.index("--workdir") + 1] == "/workspace/nested"
    mount = args[args.index("--mount") + 1]
    assert mount == f"type=bind,source={workspace.root},target=/workspace"
    assert args[-4:] == [_IMAGE, "/bin/sh", "-lc", "printf ok"]


def test_run_revalidates_inputs_without_starting_docker(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    runner = _runner(monkeypatch, default_timeout=5, max_timeout=10)
    monkeypatch.setattr(
        "src.sandbox.docker.subprocess.Popen",
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


def test_run_rejects_ambiguous_workspace_mount_paths(tmp_path, monkeypatch):
    root = tmp_path / "with,comma"
    root.mkdir()
    workspace, shell = _shell(root)
    runner = _runner(monkeypatch)

    with pytest.raises(WorkspaceViolation, match="mount") as caught:
        runner.run(shell, {"command": "echo ok"}, workspace)

    assert str(root) not in str(caught.value)


def test_run_bounds_both_output_streams_while_draining_them(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process(stdout=b"abcdefgh", stderr=b"12345678")
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    runner = _runner(monkeypatch, max_output_bytes=5)

    result = runner.run(shell, {"command": "noisy"}, workspace)

    assert result["stdout"] == "abcde"
    assert result["stderr"] == "12345"
    assert result["stdout_truncated"] is True
    assert result["stderr_truncated"] is True
    assert process.stdout.tell() == 8
    assert process.stderr.tell() == 8


def test_timeout_kills_cli_force_removes_container_and_raises_timeout(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process(times_out=True)
    cleanup_calls = []
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)

    def cleanup(args, **kwargs):
        cleanup_calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr("src.sandbox.docker.subprocess.run", cleanup)
    runner = _runner(monkeypatch, default_timeout=0.1, max_timeout=1)

    with pytest.raises(TimeoutError, match="exceeded"):
        runner.run(shell, {"command": "sleep 60"}, workspace)

    assert process.killed is True
    assert cleanup_calls
    cleanup_args, cleanup_options = cleanup_calls[0]
    assert cleanup_args[1:3] == ["rm", "--force"]
    assert cleanup_options["shell"] is False


def test_timeout_fails_closed_when_container_cleanup_is_not_confirmed(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    process = _Process(times_out=True)
    monkeypatch.setattr("src.sandbox.docker.subprocess.Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(
        "src.sandbox.docker.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 1),
    )
    runner = _runner(monkeypatch, default_timeout=0.1, max_timeout=1)

    with pytest.raises(DockerSandboxError, match="cleanup"):
        runner.run(shell, {"command": "sleep 60"}, workspace)


def test_process_start_failure_is_sanitized(tmp_path, monkeypatch):
    workspace, shell = _shell(tmp_path)
    runner = _runner(monkeypatch)
    monkeypatch.setattr(
        "src.sandbox.docker.subprocess.Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(f"denied {workspace.root}")),
    )

    with pytest.raises(DockerSandboxError, match="start") as caught:
        runner.run(shell, {"command": "echo ok"}, workspace)

    assert str(workspace.root) not in str(caught.value)
