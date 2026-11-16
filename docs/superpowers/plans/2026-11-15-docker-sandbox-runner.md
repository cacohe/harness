# Docker Sandbox Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development task by task.

**Goal:** Add an explicit, resource-bounded Docker implementation of `SandboxRunner` for command
tools.

**Architecture:** Route using immutable capability metadata, invoke only Docker on the host, execute
model command text inside a locked-down one-shot Linux container, and fail closed on validation,
timeout, or cleanup errors.

**Spec:** `docs/superpowers/specs/2026-11-15-docker-sandbox-runner-design.md`

### Task 1: Explicit sandbox capability

**Files:** `src/schema/tool.py`, `src/tool/local_tools.py`, related unit tests.

- [x] Add failing tests for command capability on local and renamed Skill shell tools.
- [x] Add `SandboxKind.COMMAND` and preserve it through tool copies.
- [x] Run focused schema/local/Skill tests.
- [x] Commit.

### Task 2: Docker runner policy and execution

**Files:** create `src/sandbox/docker.py`, `src/sandbox/__init__.py`, and runner tests.

- [x] Add failing constructor/support/input/argument-policy tests.
- [x] Implement strict configuration, Docker executable resolution, capability routing, workspace
      revalidation, and locked-down argument construction.
- [x] Add failing execution tests for success, nonzero exit, bounded concurrent output, timeout,
      process-start failure, and forced cleanup.
- [x] Implement process execution, draining, JSON-safe results, sanitized errors, and cleanup.
- [x] Run focused tests and commit.

### Task 3: Composition documentation and integration

**Files:** `README.md`, only files required by review.

- [x] Document image provisioning, explicit Agent injection, guarantees, limits, and example usage.
- [x] Run full Ruff, Mypy, pytest, diff, and lock gates.
- [x] Dispatch one independent whole-branch reviewer.
- [x] Fix Critical/Important findings with RED→GREEN tests.
- [x] Push, verify CI, merge, compare the merged tree, and clean only this plan's scratch workspace.

完成状态：已通过 PR #12 合入 `main`，Docker 沙箱执行、资源限制、结果净化与清理路径均已实现。
