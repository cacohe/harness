# Docker Sandbox Runner Design

## Goal

Provide a concrete production-oriented `SandboxRunner` for local command tools without ever running
model-authored command text in a host shell.

## Trust boundary

The local Docker daemon, selected local image, Docker CLI, and host kernel are trusted deployment
dependencies. The container is the isolation boundary. This runner does not claim to protect a host
from a compromised Docker daemon or kernel, and it supports Linux containers only.

The Docker CLI must be supplied as an existing absolute native executable. The runner never searches
the current directory or `PATH`. Every invocation explicitly selects a local Unix socket or Windows
named pipe, overriding ambient Docker contexts and `DOCKER_HOST`; remote daemon transports are
rejected because a bind source would refer to the remote host rather than this Workspace.

The image is injected explicitly. By default it must use an immutable `@sha256:<digest>` reference
and the runner passes `--pull=never`; operators provision the image before starting the Agent.

## Capability routing

Add explicit `SandboxKind.COMMAND` metadata to `Tool`. Local and Skill shell tools retain that
capability when copied/renamed. `DockerSandboxRunner.supports()` accepts only sandbox-required
command tools, rather than guessing from a mutable tool name.

## Container policy

Each call creates a uniquely named, initially stopped container with:

- `--network none`, `--read-only`, `--cap-drop ALL`, and `no-new-privileges`;
- non-root UID/GID, PID, memory, swap, CPU, and file-descriptor limits;
- a bounded tmpfs for `/tmp`;
- exactly one read/write bind mount at `/workspace`;
- a workspace-relative, revalidated working directory;
- no host shell, host environment forwarding, Docker socket, or extra mounts.

The command is passed as one argument to `/bin/sh -lc` inside the container. Docker CLI invocation
uses an argument list with `shell=False`.

## Execution and cleanup

The synchronous runner is intentionally used through ToolExecutor's worker-thread bridge. It drains
stdout and stderr concurrently, keeps only configured byte limits, and returns a JSON-safe result
with exit code, decoded streams, and truncation flags.

Creation and attached start are separate Docker operations, so command execution begins only after
container identity is established. Every created container is force-removed before return, including
normal nonzero exits, timeouts, CLI failures, and output-capture failures. Cleanup happens before
bounded reader joins; remaining streams are closed at the shared cleanup deadline. Timeout and
cleanup failures cross a sanitized exception boundary; host paths and Docker command lines are not
included.

## Validation

Constructor limits and digest references fail closed. Per-call command, timeout, and cwd are
revalidated even when a caller bypasses the local tool validator. Workspace paths containing Docker
mount delimiter characters are rejected rather than ambiguously parsed.

## Non-goals

- Installing Docker, pulling images, or packaging this repository.
- Network-enabled commands or arbitrary container configuration from model input.
- Windows-container support.
- Treating approval as isolation; approval remains a separate policy layer.
