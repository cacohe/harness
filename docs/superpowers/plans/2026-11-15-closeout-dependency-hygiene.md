# Closeout and Dependency Hygiene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close implemented roadmap documentation and enforce a clean schema dependency direction.

**Architecture:** Move the layer-neutral async iterator into `src.utils` and the workspace boundary
to `src.workspace`, re-export both from their legacy core modules, and add an AST-level dependency
test. Treat plan/metadata/ignore changes as documentation and repository hygiene with explicit
command verification.

**Tech Stack:** Python 3.11, pytest, Ruff, Mypy, uv, Git

**Spec:** `docs/superpowers/specs/2026-11-15-closeout-dependency-hygiene-design.md`

## Global Constraints

- Work only in the existing isolated F: worktree and feature branch.
- Preserve runtime and import compatibility.
- Do not add dependencies or packaging metadata.
- Do not delete local environments or scratch directories owned by other plans.

## Review Focus

- `src/schema` must have no absolute dependency on `src.core` after the move.
- Sync iterators must still advance on worker threads and async iterators must stream directly.
- The core compatibility import must resolve to the canonical utility function, not a copy.
- Historical plan completion claims must correspond to code already merged into `main`.
- Ignore rules must not hide source, tests, docs, or lock files.

---

### Task 1: Correct the schema dependency direction

**Files:**
- Create: `src/utils/async_utils.py`
- Create: `src/workspace.py`
- Modify: `src/core/async_utils.py`
- Modify: `src/core/workspace.py`
- Modify: `src/schema/llm.py`
- Modify: `src/schema/tool.py`
- Modify: `tests/integration/test_project_layout.py`
- Modify: `tests/unit/core/test_async_utils.py`
- Modify: `tests/unit/core/test_workspace.py`

**Interfaces:**
- Produces: `src.utils.async_utils.iterate_maybe_async(value) -> AsyncIterator[T]`.
- Preserves: `src.core.async_utils.iterate_maybe_async` as the exact same function object.
- Produces: `src.workspace.Workspace` and `WorkspaceViolation`.
- Preserves: legacy core workspace imports as the exact same class objects.

- [x] **Step 1: Add a failing schema-layer dependency test and canonical utility import tests**
- [x] **Step 2: Run the focused tests and verify RED on the existing `schema -> core` import**
- [x] **Step 3: Move the iterator implementation, update schema, and re-export from core**
- [x] **Step 4: Run layout/async/LLM stream tests, Ruff, and Mypy and verify GREEN**
- [x] **Step 5: Commit the dependency-direction change**

### Task 2: Close roadmap documentation and repository hygiene

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Modify: completed plans under `docs/superpowers/plans/`
- Modify: this plan

**Interfaces:**
- Produces: accurate project positioning and auditable completed-plan status.
- Produces: root ignore rules for local generated artifacts.

- [x] **Step 1: Update README/project metadata and root ignore rules**
- [x] **Step 2: Mark the ten verified historical plans complete and append integration evidence**
- [x] **Step 3: Verify no scoped historical checklist item remains open**
- [x] **Step 4: Run lock, Ruff, Mypy, full pytest, and diff gates**
- [x] **Step 5: Commit closeout documentation and hygiene changes**
- [x] **Step 6: Dispatch one fresh whole-branch review and resolve actionable findings**
- [x] **Step 7: Mark this plan complete after final verification**
