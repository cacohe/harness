# src Package and uv Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src` the explicit application package and use uv for a non-packaged project without changing business behavior.

**Architecture:** Preserve the existing module layout and logic. Change import qualification and project tooling only; uv installs dependencies but does not build or install the repository as a distribution package.

**Tech Stack:** Python 3.11+, uv, pytest, Ruff

**Spec:** `docs/superpowers/specs/2026-11-06-src-uv-layout-design.md`

## Global Constraints

- Do not change business logic.
- All internal imports start with `src`.
- The project does not build a wheel.
- uv manages runtime and development dependencies through `pyproject.toml` and `uv.lock`.

## Review Focus

- Imports must work from the repository root without `PYTHONPATH` or `sys.path` mutation.
- Every Python subdirectory used as a package must have an explicit package marker.
- Test execution must use the same import paths as application execution.
- The lockfile must match `pyproject.toml`.
- CI must reject stale lock data.

---

### Task 1: Enforce and adopt the src import layout

**Files:**
- Create: `tests/test_project_layout.py`
- Create: `src/__init__.py` and package marker files under existing source directories
- Modify: import statements in `src/**/*.py` and `tests/**/*.py`

**Interfaces:**
- Consumes: repository-root Python module execution
- Produces: imports such as `from src.core.task import Task`

- [ ] Add a layout test that rejects legacy top-level project imports.
- [ ] Run the test and confirm it fails on the current imports.
- [ ] Add package markers and qualify internal imports with `src`.
- [ ] Remove path mutation and file-based module loading workarounds.
- [ ] Run the layout test and full test suite.

### Task 2: Adopt uv non-package dependency management

**Files:**
- Modify: `pyproject.toml`
- Create: `.python-version`
- Create: `uv.lock`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: uv project commands
- Produces: `uv sync --locked` and `uv run --locked ...`

- [ ] Add configuration assertions to the layout test and confirm they fail.
- [ ] Set `tool.uv.package = false`, remove build metadata, declare direct runtime dependencies,
      and move development tools to `dependency-groups.dev`.
- [ ] Pin the development Python version and generate `uv.lock`.
- [ ] Update CI to install uv, synchronize from the lockfile, and run checks through uv.
- [ ] Verify lock freshness, imports, tests, lint, format, and module compilation.

