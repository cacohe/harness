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

- [ ] Add failing tests for command capability on local and renamed Skill shell tools.
- [ ] Add `SandboxKind.COMMAND` and preserve it through tool copies.
- [ ] Run focused schema/local/Skill tests.
- [ ] Commit.

### Task 2: Docker runner policy and execution

**Files:** create `src/sandbox/docker.py`, `src/sandbox/__init__.py`, and runner tests.

- [ ] Add failing constructor/support/input/argument-policy tests.
- [ ] Implement strict configuration, Docker executable resolution, capability routing, workspace
      revalidation, and locked-down argument construction.
- [ ] Add failing execution tests for success, nonzero exit, bounded concurrent output, timeout,
      process-start failure, and forced cleanup.
- [ ] Implement process execution, draining, JSON-safe results, sanitized errors, and cleanup.
- [ ] Run focused tests and commit.

### Task 3: Composition documentation and integration

**Files:** `README.md`, only files required by review.

- [ ] Document image provisioning, explicit Agent injection, guarantees, limits, and example usage.
- [ ] Run full Ruff, Mypy, pytest, diff, and lock gates.
- [ ] Dispatch one independent whole-branch reviewer.
- [ ] Fix Critical/Important findings with RED→GREEN tests.
- [ ] Push, verify CI, merge, compare the merged tree, and clean only this plan's scratch workspace.
