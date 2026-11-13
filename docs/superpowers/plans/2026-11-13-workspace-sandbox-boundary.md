# Workspace and Sandbox Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confine default local tools to an explicit workspace and make command execution fail closed unless an explicit sandbox runner handles it.

**Architecture:** A core `Workspace` capability owns canonical path resolution and identity. `LocalToolSuite` binds file tools to that capability, while `Tool` metadata and `ToolExecutor` enforce validated inputs, approval provenance, and sandbox-only routing for command tools. The default `Agent` advertises shell only when a sandbox is injected.

**Tech Stack:** Python 3.11+, `pathlib`, Pydantic, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-13-workspace-sandbox-boundary-design.md`

## Global Constraints

- Use only the Python standard library for workspace and sandbox-boundary infrastructure.
- Reject absolute paths, `..` traversal, and canonical symlink escapes; never rewrite them into the workspace.
- Guarantee canonical path containment only; hard links, mount points, TOCTOU races, network isolation, and OS/container sandboxing are out of scope.
- Approval never permits host-runner fallback for `requires_sandbox` tools.
- The six file/search tools remain available by default; shell is registered only when a sandbox runner is explicitly supplied.
- The project is not required to be packaged or distributed.
- Every production change follows RED → GREEN → refactor, then pytest, Ruff, and Mypy gates.

## Review Focus

- Windows drive, rooted, UNC, and mixed-separator paths must fail before filesystem access; Task 1 path tests cover them.
- A missing write leaf below an outside-pointing symlinked parent must fail; Task 1 and Task 2 symlink tests cover it.
- Glob patterns and search results must not escape or disclose the host root; Task 2 traversal/search tests cover them.
- A held approval replayed under another workspace or changed policy must fail rather than execute; Task 3 provenance tests cover it.
- Missing, unsupported, or falsey sandbox collaborators must never trigger local command fallback; Task 3 routing tests cover them.

---

### Task 1: Canonical Workspace Capability

**Files:**
- Create: `src/core/workspace.py`
- Create: `tests/unit/core/test_workspace.py`

**Interfaces:**
- Consumes: `pathlib.Path`; no project runtime dependencies.
- Produces: `WorkspaceViolation(ValueError)` and `Workspace(root: str | Path)` with `root: Path`, `identity: str`, `resolve(path: str | Path = ".") -> Path`, and `relative(path: Path) -> str`.

- [x] **Step 1: Write failing construction and containment tests**

Add tests named:

```python
def test_workspace_requires_an_existing_directory(tmp_path): ...
def test_workspace_resolves_relative_paths_and_has_stable_identity(tmp_path): ...
@pytest.mark.parametrize("path", ["../outside", "..\\outside", absolute_path, drive_path, unc_path])
def test_workspace_rejects_absolute_and_parent_paths(tmp_path, path): ...
def test_workspace_relative_never_returns_a_host_absolute_path(tmp_path): ...
```

Assert `WorkspaceViolation`, not a raw `ValueError`/`OSError`, for boundary inputs. Use literal expected relative paths and verify two distinct roots have distinct identities.

- [x] **Step 2: Run Task 1 tests and confirm RED**

Run: `python -m pytest tests/unit/core/test_workspace.py -q -p no:cacheprovider`

Expected: collection fails because `src.core.workspace` does not exist.

- [x] **Step 3: Implement `Workspace` canonical resolution**

Implement the exact interfaces above. Normalize the root once; compute identity with SHA-256 over `os.path.normcase(str(root))`; reject absolute syntax using both host `Path` and `PureWindowsPath`; reject `..` components before joining; resolve with `strict=False`; use `Path.is_relative_to` for canonical containment. Boundary errors must mention the user input, not an outside canonical host path.

- [x] **Step 4: Add and run symlink escape tests**

Add:

```python
def test_workspace_rejects_file_symlink_outside_root(tmp_path): ...
def test_workspace_rejects_missing_leaf_under_outside_symlink(tmp_path): ...
```

Create an outside directory beside the root. Skip only when `os.symlink` itself raises an OS privilege/capability error. Run the Task 1 test file and expect all tests to pass.

- [x] **Step 5: Run focused quality checks**

Run:

```bash
python -m ruff check --no-cache src/core/workspace.py tests/unit/core/test_workspace.py
python -m mypy src/core/workspace.py
```

Expected: both exit 0.

- [x] **Step 6: Commit Task 1**

```bash
git add src/core/workspace.py tests/unit/core/test_workspace.py
git commit -m "feat: add canonical workspace boundary"
```

### Task 2: Workspace-bound Local Tool Suite

**Files:**
- Modify: `src/schema/tool.py:1-16`
- Rewrite: `src/tool/local_tools.py:1-339`
- Rewrite: `tests/unit/tool/test_local_tools.py:1-79`

**Interfaces:**
- Consumes: `Workspace`, `WorkspaceViolation` from Task 1.
- Produces: `Tool.requires_sandbox: bool`, `Tool.input_validator: Callable[[dict], dict] | None`, `Tool.workspace: Workspace | None`, and `LocalToolSuite(workspace: Workspace)` with `tools(include_shell: bool = False) -> list[Tool]` plus bound file/search methods.

- [x] **Step 1: Write failing bound-tool behavior tests**

Replace ambient-`cwd` tests with `tmp_path`-backed tests that instantiate `LocalToolSuite(Workspace(root))`. Add tests:

```python
def test_local_suite_exposes_six_file_tools_without_shell(tmp_path): ...
def test_bound_write_read_edit_list_grep_and_glob_stay_relative(tmp_path): ...
def test_bound_tools_reject_absolute_and_parent_paths(tmp_path): ...
def test_search_skips_outside_symlink_targets(tmp_path): ...
def test_glob_rejects_absolute_and_parent_patterns(tmp_path): ...
```

Assert grep/glob output contains workspace-relative paths and never `str(root)` or the outside path.

- [x] **Step 2: Run Task 2 tests and confirm RED**

Run: `python -m pytest tests/unit/tool/test_local_tools.py -q -p no:cacheprovider`

Expected: import/constructor failures because `LocalToolSuite` is absent.

- [x] **Step 3: Extend `Tool` metadata with safe defaults**

Add the three fields from Interfaces after `executer`, preserving existing `Tool(...)` call sites. Use `TYPE_CHECKING` for the `Workspace` type to avoid a runtime schema-to-core import cycle.

- [x] **Step 4: Implement bound file and search methods**

Replace `_resolve`, ambient functions, and `LOCAL_TOOLS` with `LocalToolSuite`. Every path-bearing definition gets an input validator that returns a copied dictionary with canonical workspace-relative path fields. Bound methods resolve again immediately before I/O. Search validates every candidate before reading/reporting and skips an outside symlink; glob validates pattern syntax separately.

- [x] **Step 5: Add the sandbox-only shell definition**

Add tests and implementation for:

```python
def test_shell_tool_is_opt_in_and_requires_sandbox(tmp_path): ...
def test_shell_validator_normalizes_omitted_and_nested_cwd(tmp_path): ...
def test_shell_callable_never_executes_a_host_process(tmp_path): ...
```

`tools(include_shell=True)` adds one shell tool with `requires_sandbox=True`, the bound workspace, and an input validator that always supplies `cwd`. Its callable raises a clear sandbox-required exception and contains no `subprocess` call.

- [x] **Step 6: Run local-tool and existing schema tests**

Run:

```bash
python -m pytest tests/unit/tool/test_local_tools.py tests/unit/core/test_task.py -q -p no:cacheprovider
python -m ruff check --no-cache src/schema/tool.py src/tool/local_tools.py tests/unit/tool/test_local_tools.py
python -m mypy src/schema/tool.py src/tool/local_tools.py
```

Expected: all commands exit 0.

- [x] **Step 7: Commit Task 2**

```bash
git add src/schema/tool.py src/tool/local_tools.py tests/unit/tool/test_local_tools.py
git commit -m "feat: bind local tools to workspace"
```

### Task 3: Fail-closed Tool Execution and Approval Provenance

**Files:**
- Modify: `src/core/tool.py:1-224`
- Modify: `src/schema/tool.py:34-66`
- Modify: `src/tool/tool_executor.py:1-51`
- Modify: `tests/unit/tool/test_tool_executor.py:1-365`

**Interfaces:**
- Consumes: bound `Tool` metadata and `Workspace` from Tasks 1-2.
- Produces: `SandboxRunner.run(tool: Tool, inputs: dict, workspace: Workspace) -> Any`; fail reasons `workspace_violation` and `sandbox_required`; `ToolExecutionResult.workspace_id: str | None`; normalized pending arguments and workspace-bound resume enforcement.

- [x] **Step 1: Write failing input-validation and classification tests**

Add tests that assert:

```python
def test_validator_output_is_used_for_approval_and_execution(): ...
def test_workspace_violation_fails_before_policy_or_runner(): ...
def test_held_call_persists_normalized_arguments_and_workspace_identity(): ...
def test_resumed_call_revalidates_normalized_arguments(): ...
```

Use real `Workspace` objects and literal normalized JSON. Mutation counters must prove policy/runner were not called after a boundary violation.

- [x] **Step 2: Run the new Task 3 tests and confirm RED**

Run the four exact tests with `python -m pytest ... -q -p no:cacheprovider`.

Expected: missing metadata/result fields and current raw-argument behavior fail assertions.

- [x] **Step 3: Implement one validation path for direct, held, and resumed calls**

Add a private helper that copies inputs, invokes `tool.input_validator` when present, and maps `WorkspaceViolation` to a failed result. Store normalized JSON with stable separators/sorted keys in pending results. Store `tool.workspace.identity` as `workspace_id`.

- [x] **Step 4: Write failing sandbox-routing tests**

Replace the old “any supported tool runs in sandbox” assumption with:

```python
def test_sandbox_required_tool_fails_without_sandbox_and_never_uses_runner(): ...
def test_sandbox_required_tool_fails_when_sandbox_does_not_support_it(): ...
def test_sandbox_receives_tool_inputs_and_bound_workspace(): ...
def test_regular_tool_uses_bound_runner_even_when_sandbox_supports_it(): ...
def test_falsey_sandbox_is_preserved(): ...
def test_resume_rejects_approval_from_another_workspace(): ...
```

- [x] **Step 5: Implement fail-closed sandbox routing**

Change the protocol signature and test doubles. Route to sandbox only when `tool.requires_sandbox`; require a non-null `tool.workspace`; otherwise return the exact new fail reason. Ordinary tools always use `ToolRunner`. On resume, compare the stored/current workspace identities before execution and return `workspace_violation` on mismatch.

- [x] **Step 6: Run executor, loop, and demo regression tests**

Run:

```bash
python -m pytest tests/unit/tool/test_tool_executor.py tests/unit/core/test_loop.py tests/unit/demo/test_agent.py -q -p no:cacheprovider
python -m ruff check --no-cache src/core/tool.py src/schema/tool.py src/tool/tool_executor.py tests/unit/tool/test_tool_executor.py
python -m mypy src/core/tool.py src/schema/tool.py src/tool/tool_executor.py
```

Expected: all commands exit 0. Update existing sandbox doubles to the three-argument `run` signature without weakening assertions.

- [x] **Step 7: Commit Task 3**

```bash
git add src/core/tool.py src/schema/tool.py src/tool/tool_executor.py tests/unit/tool/test_tool_executor.py
git commit -m "feat: enforce fail-closed sandbox routing"
```

### Task 4: Secure Default Agent Composition and Documentation

**Files:**
- Modify: `src/demo/agent.py:1-82`
- Modify: `tests/unit/demo/test_agent.py:1-142`
- Modify: `README.md:38-91`
- Modify: `docs/superpowers/plans/2026-11-13-workspace-sandbox-boundary.md`

**Interfaces:**
- Consumes: `Workspace`, `LocalToolSuite`, `SandboxRunner`, and `DefaultToolExecutor` from Tasks 1-3.
- Produces: `Agent(..., workspace: Workspace | str | Path | None = None, sandbox_runner: SandboxRunner | None = None)` and a default catalog that excludes shell unless the sandbox is supplied.

- [x] **Step 1: Write failing default-composition tests**

Add:

```python
def test_default_agent_binds_file_tools_to_explicit_workspace(tmp_path): ...
def test_default_agent_does_not_advertise_shell_without_sandbox(tmp_path): ...
def test_default_agent_registers_shell_and_passes_workspace_to_sandbox(tmp_path): ...
def test_agent_preserves_falsey_workspace_and_sandbox_dependencies(tmp_path): ...
```

Use the existing fake LLM gateway; do not invoke a real model or process. Assert catalog names and object identity, not implementation internals.

- [x] **Step 2: Run Task 4 tests and confirm RED**

Run: `python -m pytest tests/unit/demo/test_agent.py -q -p no:cacheprovider`

Expected: constructor rejects the new keyword arguments or still advertises ambient shell.

- [x] **Step 3: Implement the secure composition root**

Capture `Path.cwd()` only during `Agent` construction. Normalize supplied roots to `Workspace`. When the catalog is defaulted, register `LocalToolSuite(workspace).tools(include_shell=sandbox_runner is not None)`. Pass the exact sandbox object to `DefaultToolExecutor`. Preserve explicitly injected catalog/executor behavior.

- [x] **Step 4: Update README composition and security contract**

Document explicit workspace construction, the six default tools, shell omission by default, the `SandboxRunner.run(tool, inputs, workspace)` contract, and the distinction between approval and isolation. Do not claim hard-link, mount, network, or process isolation.

- [x] **Step 5: Run the full verification gate**

Run:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff format --check --no-cache src tests
python -m ruff check --no-cache src tests
python -m mypy src
git diff --check
```

Expected: 0 failures/errors and no diff whitespace errors.

- [x] **Step 6: Mark this plan complete and commit**

Change every completed checkbox in this plan to `[x]`, then:

```bash
git add README.md src/demo/agent.py tests/unit/demo/test_agent.py docs/superpowers/plans/2026-11-13-workspace-sandbox-boundary.md
git commit -m "docs: describe workspace sandbox boundary"
```

The branch is ready for whole-branch code review only after this commit and the full verification gate.
