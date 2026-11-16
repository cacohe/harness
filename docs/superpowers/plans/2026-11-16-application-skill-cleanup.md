# Application API, Skill Loading, and Low-Risk Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the existing production application API and Skill loading responsibilities in their proper packages and complete the selected low-risk structural cleanup without changing runtime behavior.

**Architecture:** `src.application` becomes the canonical Agent API while `src.demo` keeps compatibility re-exports and the CLI. `src.skill` loads a Skill and its resource tools from one document read. Neutral implementations remain canonical and core compatibility modules remain exact-object shims.

**Tech Stack:** Python 3.11, pytest, Ruff, Mypy, uv, Git

**Spec:** `docs/superpowers/specs/2026-11-16-application-skill-cleanup-design.md`

## Global Constraints

- Work only in `F:\CodeSpace\harness-worktrees\application-skill-cleanup` on `codex/application-skill-cleanup`.
- Preserve all public behavior and exact legacy import identities.
- Do not change Harness, event, ToolCatalog, schema, persistence, or configuration semantics.
- Do not add dependencies or packaging/distribution metadata.
- Do not copy the source checkout's uncommitted root files into this branch.

## Review Focus

- Legacy `src.demo` imports must resolve to the exact canonical application objects.
- Direct `src.application` callers must retain sync, async, session, MCP, memory, and close behavior.
- One Skill load must use one parsed name for both prompt text and resource tool names.
- Duplicate Skill names and partial tool registration must retain current failure behavior.
- Internal code must use neutral canonical modules without removing compatibility shims.

---

### Task 1: Move the production Agent API to `src.application`

**Files:**
- Create: `src/application/__init__.py`
- Move: `src/demo/agent.py` to `src/application/agent.py`
- Move: `src/demo/session.py` to `src/application/session.py`
- Create: `src/demo/agent.py`, `src/demo/session.py` compatibility re-exports
- Modify: `src/demo/main.py`, `README.md`, repository callers
- Move: `tests/unit/demo/test_agent.py`, `tests/unit/demo/test_session.py` to `tests/unit/application/`
- Test: `tests/unit/application/test_exports.py`

**Interfaces:**
- Produces: `src.application.Agent`, `AgentSession`, `MemoryFailure`, `MemoryRuntime`, and `TaskRun`.
- Preserves: all corresponding `src.demo.agent` and `src.demo.session` objects by identity.

- [x] **Step 1: Add canonical import and exact compatibility-identity tests**
- [x] **Step 2: Run the focused export test and verify RED because `src.application` is absent**
- [x] **Step 3: Move the implementations, add compatibility modules, and update repository-owned imports**
- [x] **Step 4: Run application, demo, integration, load, Ruff, and Mypy checks and verify GREEN**
- [x] **Step 5: Commit the application API migration**

### Task 2: Move Skill batch loading into `src.skill`

**Files:**
- Modify: `src/skill/directory.py`
- Create: `src/skill/loader.py`
- Modify: `src/skill/__init__.py`
- Modify: `src/application/agent.py`
- Modify: `tests/unit/skill/test_directory.py`
- Create: `tests/unit/skill/test_loader.py`

**Interfaces:**
- Produces: `SkillDirectory.load_bundle(*, include_shell: bool = False) -> tuple[Skill, list[Tool]]`.
- Produces: `load_skill_directories(roots, *, include_shell: bool) -> tuple[list[Skill], list[Tool]]`.
- Consumes: Task 1's canonical `src.application.agent.Agent`.

- [x] **Step 1: Add failing tests for the loader API, duplicate names, and one-read Skill/tool consistency**
- [x] **Step 2: Run focused Skill tests and verify RED on the missing APIs**
- [x] **Step 3: Implement bundle and batch loading and remove Agent's private loading helper**
- [x] **Step 4: Run Skill and application tests, Ruff, and Mypy and verify GREEN**
- [x] **Step 5: Commit the Skill ownership change**

### Task 3: Complete the selected low-risk cleanup

**Files:**
- Modify: internal imports and `README.md`
- Modify: `src/core/context.py`, `src/context/context_builder.py`
- Modify: `tests/integration/test_project_layout.py`
- Move: implementation-specific tests to matching package directories
- Delete: `src/tool/mcp_tools.py`, `src/utils/util.py`, obsolete `.gitkeep`
- Preserve: `src/core/workspace.py`, `src/core/async_utils.py`, their identity tests

**Interfaces:**
- Consumes: canonical application and Skill paths from Tasks 1 and 2.
- Preserves: core workspace and iterator compatibility exports by exact identity.

- [x] **Step 1: Add a failing layout test that rejects internal compatibility-path imports**
- [x] **Step 2: Run the layout test and verify RED on current internal imports**
- [x] **Step 3: Switch internal imports, move prompts/tests, and delete confirmed dead files**
- [x] **Step 4: Run focused compatibility and moved tests and verify GREEN**
- [x] **Step 5: Run uv lock, Ruff, Mypy, full pytest, and diff gates**
- [ ] **Step 6: Commit the cleanup and complete a fresh whole-branch review**
