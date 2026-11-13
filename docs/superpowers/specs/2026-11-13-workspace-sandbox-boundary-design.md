# Workspace and Sandbox Boundary Design

## Intent

The default agent currently treats the process working directory as ambient authority. File tools may
resolve outside the project, symlinks can redirect reads and writes, and `shell` falls back to
`subprocess.run(..., shell=True)` on the host when no sandbox is installed. User approval reduces
accidental execution but is not an isolation boundary.

This phase makes the default composition fail closed. Local file tools receive an explicit workspace
capability, and command tools execute only through an explicitly supplied sandbox implementation.
The harness must never imply that an approval prompt, a path string, or a protocol declaration is a
security boundary by itself.

## Goals

- Confine every default local file operation to one immutable workspace root.
- Reject absolute paths, parent traversal, and existing symlinks that resolve outside the workspace.
- Avoid leaking host absolute paths in normal tool output; return workspace-relative paths instead.
- Mark command execution as sandbox-required and refuse it when no supporting sandbox is installed.
- Preserve the existing approval flow: authorization happens before execution, but approval never
  bypasses workspace or sandbox enforcement.
- Make the default `Agent` bind its local tools to an explicit root, defaulting to the process working
  directory captured at construction time.
- Keep the security ports structurally injectable so a later OS/container runner can implement the
  same `SandboxRunner` contract.

## Non-goals

- Implementing a Windows job object, Linux namespace, container, VM, seccomp, or network sandbox.
- Claiming protection from a time-of-check/time-of-use symlink race inside a hostile workspace.
- Detecting storage aliases such as hard links or mount points whose path remains inside the root while
  backing storage is shared elsewhere. This phase guarantees canonical path containment, not hostile
  filesystem-topology isolation.
- Restricting remote MCP tools; their trust and authorization model belongs to the MCP phase.
- Adding a package-distribution compatibility layer for the old ambient local-tool functions.
- Adding async execution, cancellation, durable persistence, or concurrent task ownership.

## Anti-goals

- Do not silently normalize an escaping path back into the workspace.
- Do not treat user approval as permission to fall back to host execution.
- Do not register an executable `shell` tool in the default demo when no sandbox runner is supplied.
- Do not catch a workspace violation and report the tool call as successfully completed text.
- Do not add a platform-specific sandbox and label it portable or production-ready.

## Constraints

- Python remains `>=3.11`; use `pathlib` and the standard library only.
- `ToolCatalog`, `ToolRunner`, `ApprovalPolicy`, and `SandboxRunner` remain structural ports; the
  `SandboxRunner.run` signature changes to receive the bound `Workspace` explicitly.
- The default local file set remains `read_file`, `write_file`, `edit_file`, `list_dir`, `grep`, and
  `glob`; `shell` is present only when sandbox execution is explicitly enabled.
- Tests, Ruff, and Mypy remain the quality gates.
- The project is not required to be packaged or distributed.

## Approaches Considered

### 1. Add checks to the existing module-level functions

This is the smallest diff, but it still depends on mutable process `cwd`, leaves global tool objects
bound to ambient state, and makes it easy for a new tool to omit the check. It improves symptoms
without making authority explicit.

### 2. Bind a local tool suite to an explicit workspace capability — selected

A `Workspace` value owns canonical path resolution. A `LocalToolSuite` owns the workspace and creates
bound `Tool` objects. The executor recognizes workspace violations and sandbox-required tools. This
keeps the default path portable, makes the secure boundary visible in composition, and leaves real
process isolation to injected runners.

### 3. Ship an OS or container sandbox in this phase

This provides stronger command isolation, but a correct cross-platform implementation needs process,
filesystem, network, lifecycle, and resource-limit decisions that are independent of workspace path
confinement. Shipping a partial implementation would create false confidence. The current phase
therefore establishes the fail-closed contract that a later runner must satisfy.

## Workspace Capability

Create `src/core/workspace.py` with:

- `WorkspaceViolation(ValueError)` for security-boundary failures.
- `Workspace(root: str | Path)`, which resolves the root once and requires it to be an existing
  directory.
- `identity: str`, a stable SHA-256 digest of the normalized canonical root used to bind persisted
  approvals without storing or displaying the host path.
- `resolve(path: str | Path = ".") -> Path`, which accepts only relative paths, rejects `..` path
  components (including Windows-style components), resolves existing symlinks, and verifies the
  canonical result is inside the canonical root.
- `relative(path: Path) -> str`, which returns a stable workspace-relative display path.

Absolute paths are rejected even when they point inside the root. This keeps tool inputs portable and
prevents the model from acquiring reusable host paths. File creation uses non-strict resolution so a
missing leaf can be created, while existing parent symlinks are still resolved and checked.

Directory traversal tools must validate every file before reading or returning it. A symlink that
points outside the root is skipped during search and rejected when addressed directly. Glob patterns
with absolute syntax or `..` components are rejected before enumeration.

## Bound Local Tools

Replace ambient module-level execution with `LocalToolSuite(workspace: Workspace)` in
`src/tool/local_tools.py`. Its methods implement the six file/search operations using the bound
workspace, and `tools(include_shell: bool = False) -> list[Tool]` creates the catalog entries.

Every path-bearing tool has an input validator. Validation converts accepted paths to canonical
workspace-relative strings before approval or execution. The bound method resolves them again at the
operation boundary. This both prevents bypass and ensures the approval display names the target that
will execute rather than an unresolved symlink alias.

The optional shell definition carries two properties:

- `requires_sandbox=True`, so the ordinary `ToolRunner` is never used for it.
- an input validator that verifies an optional `cwd` through the same workspace capability before a
  sandbox receives the request.

The normalized shell payload always includes `cwd`: omitted or empty input becomes `"."`; a nested
directory becomes its canonical workspace-relative path. `command` and `timeout` retain their input
values. The sandbox receives `run(tool, inputs, workspace)`, so it never has to infer the root from the
host process working directory.

The shell tool's attached callable is not a host-process fallback. Direct invocation fails with a
clear sandbox-required error.

## Tool Metadata and Execution

Extend `Tool` with backward-compatible defaults:

- `requires_sandbox: bool = False`
- `input_validator: Callable[[dict], dict] | None = None`
- `workspace: Workspace | None = None` (a runtime-only capability; `Tool` snapshots are not persisted)

Add `workspace_violation` and `sandbox_required` to `ToolExecutionFailReason`.

`ToolExecutor` validates inputs before approval and again on resumed execution. A
`WorkspaceViolation` becomes a failed result with reason `workspace_violation`. For a
`requires_sandbox` tool, execution succeeds only when a sandbox exists and `supports(tool)` is true;
otherwise it returns `sandbox_required` without calling the normal runner or the tool callable. A
supporting sandbox is called as `run(tool, validated_inputs, tool.workspace)`; a missing workspace is
a workspace violation. Exceptions raised by an installed sandbox remain ordinary failed tool results.

Sandbox routing applies only to `requires_sandbox` tools. The six default file tools always run through
their workspace-bound callables even if an injected sandbox claims to support them. A future tool that
should run in the sandbox must opt in explicitly with metadata.

Held tool calls persist normalized JSON arguments and the tool workspace identity for audit and
display. On resume, the executor compares the stored identity with the current tool. A mismatch fails
with `workspace_violation`; an approval is never replayed under a different root. Validation is
repeated after approval so persisted calls cannot bypass a changed policy. Because a symlink alias was
replaced by its canonical workspace-relative target before approval, retargeting that alias cannot
silently redirect the approved call.

## Default Composition

`Agent` accepts:

- `workspace: Workspace | str | Path | None`
- `sandbox_runner: SandboxRunner | None`

When it builds the default catalog, it captures `Path.cwd()` if no workspace is supplied, creates a
`LocalToolSuite`, and registers shell only when `sandbox_runner` is present. The same runner is passed
to `DefaultToolExecutor`. If callers inject their own catalog or executor, they own that composition;
the core executor still enforces metadata on any supplied tool.

README examples will show the explicit workspace boundary and state that command execution is absent
by default. A custom sandbox runner must be injected to advertise and execute shell.

## Error and Security Semantics

- Boundary violations are tool failures, not task crashes and not successful `"Error: ..."` outputs.
- Missing files and malformed regular expressions keep their current user-facing error behavior; this
  phase changes security classification, not every local-tool error contract.
- Approval rejection/timeout remains distinct from sandbox and workspace failures.
- An unsupported or absent sandbox never falls back to `LocalToolRunner`.
- Search output contains relative paths. Tool error messages may name the rejected user input but must
  not reveal the canonical host path outside the root.
- An injected sandbox is a trusted boundary implementation. Its stdout, stderr, and exception text are
  forwarded under the existing tool-result contract and may contain sandbox-specific paths; sanitizing
  third-party runner output is not part of workspace path confinement.
- The security claim assumes the workspace storage topology is trusted. Hard links, mount points, and
  TOCTOU mutation require an OS/container sandbox and remain explicitly outside this phase.

## Testing Strategy

Tests are written before implementation and must cover:

- relative file operations remain functional inside the root;
- `..`, absolute paths, Windows drive/UNC paths, and outside-pointing symlinks are rejected;
- write paths cannot escape through an existing symlinked parent;
- grep/glob do not read or report outside targets and return relative paths;
- shell is absent from the default `Agent` catalog without a sandbox;
- approval of a sandbox-required tool still fails closed when the sandbox is absent or unsupported;
- a supporting sandbox receives validated inputs and the normal runner is not called;
- resumed approved calls repeat workspace validation;
- a resumed approval is rejected after the workspace identity changes, and symlink retargeting cannot
  change the normalized approved target;
- ordinary file tools are not diverted merely because a sandbox reports that it supports them;
- falsey injected workspace/sandbox collaborators are preserved;
- baseline behavior for non-sandbox tools remains intact.

Symlink tests may skip only when the host cannot create symlinks; path traversal and absolute-path
tests are mandatory on every platform.

## Follow-up

The next phase can implement async-first Harness and MCP integration against these explicit security
ports. A later dedicated sandbox phase may add a concrete OS/container runner with process-tree
termination, resource limits, network policy, and filesystem mounting without changing tool metadata
or fail-closed executor behavior.
