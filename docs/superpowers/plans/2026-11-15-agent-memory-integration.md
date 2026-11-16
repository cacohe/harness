# Agent Memory Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development task by task.

**Goal:** Wire the existing memory subsystem into Agent Sessions with safe prompt, concurrency,
budget, async, and failure behavior.

**Spec:** `docs/superpowers/specs/2026-11-15-agent-memory-integration-design.md`

### Task 1: Harden MemoryManager and built-in store

- [x] Add failing tests for token budgets, metadata exclusion, access persistence, validation,
      concurrent operations, and copy isolation.
- [x] Implement locked compound operations and a copy-isolated thread-safe `InMemoryStore`.
- [x] Run focused memory tests and commit.

### Task 2: Integrate AgentSession lifecycle

- [x] Add failing tests for cross-Session recall, same-Session exclusion, completed/resumed commit,
      failed/paused omission, prompt boundary, nonblocking async calls, failure recording, and
      terminal completion winning post-commit cancellation.
- [x] Add optional manager injection, worker-thread recall/commit, `MemoryFailure`, and stable prompt
      formatting.
- [x] Wire the shared manager through `Agent.new_session()` and preserve default behavior.
- [x] Run focused Session/Agent tests and commit.

### Task 3: Documentation, review, and integration

- [x] Document explicit manager construction, scope, budgets, failure inspection, and security.
- [x] Run full Ruff, Mypy, pytest, diff, and lock gates.
- [x] Dispatch an independent whole-branch reviewer and fix Critical/Important findings with tests.
- [x] Push, verify CI, merge, and compare the merged tree.

完成状态：已通过 PR #13 合入 `main`，记忆召回、提交、预算与失败可观测性均已集成。
