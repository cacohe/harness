# TaskPlan API Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the only concrete unimplemented placeholder from the runtime without inventing an
unused planning state machine.

**Architecture:** Keep `TaskPlan` as a Pydantic snapshot/LLM response model and keep whole-plan
replacement in `Task.set_plan()`.

**Spec:** `docs/superpowers/specs/2026-11-15-task-plan-api-cleanup-design.md`

### Task 1: Remove the phantom API

**Files:**
- Modify: `tests/unit/core/test_task.py`
- Modify: `src/schema/task.py`

- [x] Add a failing test that the unsupported method is absent.
- [x] Run the focused test and verify RED.
- [x] Remove `TaskPlan.advance_plan`.
- [x] Run focused Task tests and verify GREEN.

### Task 2: Verify and integrate

- [x] Run Ruff format/lint, Mypy, full pytest, and `uv lock --check`.
- [x] Commit implementation and documentation.
- [x] Dispatch an independent whole-branch review.
- [x] Fix Critical/Important findings, push, verify CI, and merge.

完成状态：已通过 PR #11 合入 `main`，未支持的 `TaskPlan.advance_plan` API 已移除并完成兼容性验证。
