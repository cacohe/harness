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

- [ ] Add a failing test that the unsupported method is absent.
- [ ] Run the focused test and verify RED.
- [ ] Remove `TaskPlan.advance_plan`.
- [ ] Run focused Task tests and verify GREEN.

### Task 2: Verify and integrate

- [ ] Run Ruff format/lint, Mypy, full pytest, and `uv lock --check`.
- [ ] Commit implementation and documentation.
- [ ] Dispatch an independent whole-branch review.
- [ ] Fix Critical/Important findings, push, verify CI, and merge.

