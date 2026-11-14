# Skill Directory Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load directory-backed skills into `Agent` while routing resource reads through `Workspace` and all script execution through the existing sandbox-required shell tool.

**Architecture:** Keep `Skill` as a data-only schema. Add a `SkillDirectory` composition object that parses `SKILL.md` and derives renamed tools from `LocalToolSuite`, then have `Agent` transactionally register those tools and pass the loaded skills into `Harness`. Share provider-safe tool-name normalization with MCP so naming limits and collision behavior remain consistent.

**Tech Stack:** Python 3.11, PyYAML, Pydantic-based harness schemas, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-14-skill-directory-design.md`

## Global Constraints

- Do not package or publish the project.
- Do not execute host processes from the skill schema or loader.
- Every skill path is relative to an immutable `Workspace` rooted at that skill directory.
- Shell access exists only when a `SandboxRunner` is injected and remains `requires_sandbox`.
- Preserve direct `Skill(name, description, content)` construction.
- Provider-visible tool names are deterministic ASCII and at most 64 characters.

## Review Focus

- A `SKILL.md` with CRLF delimiters must parse exactly like LF without leaving frontmatter in content.
- Malformed YAML or non-string/blank metadata must fail with a relative, non-host-leaking error.
- Absolute, parent-traversal, reserved Windows, and escaping-symlink resource paths must be rejected by `Workspace`.
- Two distinct long or unsafe skill names must not silently produce the same tool id/name.
- Catalog registration failure must leave no subset of the skill tools registered.

---

### Task 1: Shared Provider-Safe Tool Names

**Files:**
- Create: `src/utils/tool_names.py`
- Modify: `src/mcp/tool_adapter.py`
- Create: `tests/unit/utils/test_tool_names.py`
- Modify: `tests/unit/mcp/test_tool_adapter.py`

**Interfaces:**
- Consumes: arbitrary namespace and name strings.
- Produces: `provider_tool_name(namespace: str, *segments: str) -> str`, deterministic ASCII with a 64-character ceiling and stable hash suffix on truncation.

- [ ] **Step 1: Write failing tests for normalization, empty segments, and collision-resistant truncation**

  Assert literal names for safe and punctuation-containing inputs, `ValueError` for names with no provider-safe ASCII, length `<= 64`, stable results, and different hashes for distinct long names.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run: `python -m pytest tests/unit/utils/test_tool_names.py -q -p no:cacheprovider`

  Expected: FAIL because `src.utils.tool_names` does not exist.

- [ ] **Step 3: Implement the shared helper and delegate MCP naming to it**

  Keep MCP's public `provider_tool_name(server_name, remote_name)` wrapper for compatibility; it calls the shared helper with namespace `"mcp"`.

- [ ] **Step 4: Run shared and MCP tests and verify GREEN**

  Run: `python -m pytest tests/unit/utils/test_tool_names.py tests/unit/mcp/test_tool_adapter.py -q -p no:cacheprovider`

  Expected: PASS.

- [ ] **Step 5: Commit**

  Commit message: `refactor: share provider-safe tool naming`

### Task 2: Secure Skill Directory Loader and Resource Tools

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/schema/skills.py`
- Create: `src/skill/__init__.py`
- Create: `src/skill/directory.py`
- Create: `tests/unit/skill/test_directory.py`

**Interfaces:**
- Consumes: `SkillDirectory(root: str | Path)`, a valid `SKILL.md`, and `include_shell`.
- Produces: `SkillDocumentError`, `SkillDirectory.workspace`, `load(*, include_shell: bool = False) -> Skill`, and `tools(*, include_shell: bool = False) -> list[Tool]`.

- [ ] **Step 1: Write failing document-loading tests**

  Cover LF and CRLF frontmatter, Unicode body text, missing closing delimiter, invalid YAML, non-mapping YAML, missing/blank/non-string `name` and `description`, missing document, invalid UTF-8, and error messages that omit the absolute root.

- [ ] **Step 2: Run document tests and verify RED**

  Run: `python -m pytest tests/unit/skill/test_directory.py -q -p no:cacheprovider`

  Expected: FAIL because `SkillDirectory` does not exist.

- [ ] **Step 3: Implement minimal parsing and restore the data-only `Skill` schema**

  Declare `pyyaml` as a direct dependency. Parse only a leading `---` frontmatter block, require a mapping with non-empty string metadata, and wrap read/parse failures in `SkillDocumentError` with `SKILL.md`-relative context.

- [ ] **Step 4: Run document tests and verify GREEN**

  Run: `python -m pytest tests/unit/skill/test_directory.py -q -p no:cacheprovider`

  Expected: document tests PASS.

- [ ] **Step 5: Write failing resource-tool tests**

  Assert exact generated names and resource notes, read-only default tools, immutable workspace binding, traversal/absolute/reserved-path rejection, escaping-symlink rejection, optional shell registration, `requires_sandbox=True`, normalized `cwd`, and a placeholder callable that cannot start a host process.

- [ ] **Step 6: Run resource-tool tests and verify RED**

  Run: `python -m pytest tests/unit/skill/test_directory.py -q -p no:cacheprovider`

  Expected: FAIL because `tools()` and resource notes are incomplete.

- [ ] **Step 7: Derive renamed tools from `LocalToolSuite` and append the resource note**

  Use `dataclasses.replace` on the real `read_file` and optional `shell` tools so their validators, executers, origin, workspace, and sandbox semantics remain unchanged.

- [ ] **Step 8: Run skill, workspace, local-tool, and executor tests and verify GREEN**

  Run: `python -m pytest tests/unit/skill/test_directory.py tests/unit/core/test_workspace.py tests/unit/tool/test_local_tools.py tests/unit/tool/test_tool_executor.py -q -p no:cacheprovider`

  Expected: PASS.

- [ ] **Step 9: Commit**

  Commit message: `feat: load sandbox-bound skill directories`

### Task 3: Agent Skill Composition and Documentation

**Files:**
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-11-14-skill-directory-design.md`

**Interfaces:**
- Consumes: `Agent(..., skill_directories: list[str | Path] | None = None)` and the existing catalog/executor/sandbox dependencies.
- Produces: loaded skills in `agent.harness.skills` plus atomically registered resource tools in the shared catalog.

- [ ] **Step 1: Write failing Agent composition tests**

  Cover skill content in the model system prompt, read tool registration, shell omission without a sandbox, shell routing to an injected sandbox with the skill workspace, duplicate skill names, pre-existing tool collisions, and rollback when catalog registration raises partway through a batch.

- [ ] **Step 2: Run Agent tests and verify RED**

  Run: `python -m pytest tests/unit/demo/test_agent.py -q -p no:cacheprovider`

  Expected: FAIL because `Agent` does not accept `skill_directories`.

- [ ] **Step 3: Implement preflight validation and transactional registration**

  Load all directories first, reject duplicate logical names and collisions from `find_tool`, register only after preflight, unregister only ids added by this batch on failure, and pass the loaded list to `Harness`.

- [ ] **Step 4: Run Agent and context tests and verify GREEN**

  Run: `python -m pytest tests/unit/demo/test_agent.py tests/unit/core/test_context.py -q -p no:cacheprovider`

  Expected: PASS.

- [ ] **Step 5: Document the public entry point and security behavior**

  Add a README example showing `skill_directories=["./skills/research"]`, optional sandbox injection for scripts, resource-tool naming, and explicit fail-closed semantics.

- [ ] **Step 6: Run all project gates**

  Run:

  - `ruff format --check src tests`
  - `ruff check --no-cache src tests`
  - `mypy src`
  - `python -m pytest -q -p no:cacheprovider`

  Expected: all commands exit 0 with no warnings attributable to this change.

- [ ] **Step 7: Commit**

  Commit message: `feat: compose skill directories into agent`

### Task 4: Whole-Branch Review and PR Update

**Files:**
- Modify: only files required by review findings.

**Interfaces:**
- Consumes: the complete branch diff against `origin/main`.
- Produces: a review-clean branch and an updated PR #8.

- [ ] **Step 1: Run a fresh whole-branch code review**

  Focus on authority boundaries, catalog rollback, naming collisions, document parser failure modes, sync/async safety, and regression risk.

- [ ] **Step 2: Reproduce every actionable finding with a failing test**

  Run the narrow test and confirm it fails for the reported behavior before changing production code.

- [ ] **Step 3: Fix findings minimally and rerun focused plus full gates**

  Expected: all focused tests and all project gates PASS.

- [ ] **Step 4: Commit review fixes, push the branch, and verify PR checks**

  Commit only if review produced changes; otherwise push the existing commits. Confirm PR #8 reports mergeable and all required checks pass.
