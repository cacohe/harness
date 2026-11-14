# Skill Directory Integration Design

## Intent

Make directory-backed skills usable by the production harness without granting the schema
layer direct host filesystem or process authority. A skill directory contains a `SKILL.md`
document plus optional references and scripts. The harness must load the document, make the
skill visible to model context, and expose resource access through the same workspace and
sandbox boundaries already used by local tools.

## Goals

- Load `name`, `description`, and body content from a UTF-8 `SKILL.md` with YAML frontmatter.
- Bind every skill directory to an immutable `Workspace` so absolute paths, parent traversal,
  reserved Windows paths, and escaping symlinks are rejected consistently.
- Provide a read-only resource tool for files inside each skill directory.
- Provide an optional shell tool scoped to the skill directory; it must be marked
  `requires_sandbox` and must never start a host process itself.
- Let `Agent` accept skill directories and pass loaded `Skill` values to `Harness` while
  registering their resource tools in the shared catalog.
- Detect duplicate skill names and tool collisions before partially mutating the catalog.

## Non-goals

- Discovering skills from global user directories or plugin registries.
- Selecting skills dynamically from a large catalog; all configured skills remain visible to
  the context builder in this phase.
- Providing a concrete sandbox implementation.
- Defining a cross-platform script interpreter or bypassing the existing shell-tool contract.
- Packaging or publishing the project.

## Anti-goals

- No `subprocess` calls in `src/schema/skills.py` or the skill loader.
- No host-absolute paths in model-visible output.
- No silent overwrite when two skills normalize to the same provider-safe tool name.
- No script execution fallback when a compatible `SandboxRunner` is absent.

## Constraints

- Python 3.11 or newer.
- Reuse `Workspace`, `LocalToolSuite`, `ToolExecutor`, and `SandboxRunner` rather than creating
  parallel security or execution abstractions.
- Parse frontmatter with a declared YAML dependency; require a top-level mapping containing
  non-empty string `name` and `description` fields.
- Existing callers that construct `Skill(name, description, content)` remain compatible.
- Skill tools use deterministic provider-safe ASCII names no longer than 64 characters.

## Approaches Considered

### 1. Static helpers on `Skill`

This is the current PR implementation. It is small, but it mixes data with host I/O, permits
path traversal, launches `bash` directly, ignores exit status, and bypasses both approval and
sandbox routing. Rejected.

### 2. A separate skill-specific executor

This would keep the schema cleaner, but would duplicate workspace validation, tool schemas,
approval flow, async routing, and sandbox failure classification. Rejected because the existing
runtime already owns those responsibilities.

### 3. Directory loader plus existing local-tool capabilities

Selected. `SkillDirectory` owns one `Workspace`, parses its document, and derives renamed
read-only and optional shell tools from `LocalToolSuite`. `Agent` composes loaded skills and
their tools into the existing Harness and ToolExecutor. This adds only the missing skill
boundary while preserving one security and execution model.

## Design

### Data model

`src/schema/skills.py` returns to a data-only `Skill` class. It does not import filesystem,
regular-expression, or process modules.

### Directory loader

`src/skill/directory.py` defines:

- `SkillDocumentError`, for malformed or unreadable skill documents using only relative
  context in messages.
- `SkillDirectory(root)`, which constructs an immutable `Workspace` rooted at an existing
  directory.
- `load(*, include_shell: bool = False) -> Skill`, which reads `SKILL.md` as UTF-8, accepts LF
  or CRLF delimiters, parses the leading YAML mapping, validates `name` and `description`, and
  returns the remaining body as `content` without the delimiters plus the matching resource-tool
  note.
- `tools(include_shell: bool = False) -> list[Tool]`, which returns a renamed read-only file
  tool and, when requested, a renamed shell tool. Both remain bound to the same workspace.

Tool names are `skill__<normalized-name>__read_file` and
`skill__<normalized-name>__shell`. Normalization follows the same ASCII, stable-hash, 64-character
rules as MCP provider tools but lives in a shared naming helper so the rules do not diverge.
The read tool is backed by the real `LocalToolSuite.read_file` behavior. The shell tool is the
existing fail-closed local shell capability; its validator keeps `cwd` inside the skill root,
and only the injected `SandboxRunner` may execute it.

The loaded skill content includes a short generated resource note naming the read tool and,
when enabled, the shell tool. The authored body remains unchanged before that separated note.

### Agent composition

`Agent.__init__` accepts `skill_directories: list[str | Path] | None`. It loads all directories
before mutating the tool catalog, rejects duplicate skill names and any id/name collision with
existing tools, then registers the whole batch. Shell tools are included only when a
`SandboxRunner` is supplied. The resulting skills are passed to `Harness` instead of the current
empty list.

Registration is transactional using the catalog's public `find_tool`, `register`, and
`unregister` operations: on any registration exception, tools added by this batch are removed.
Custom callers may continue constructing `Harness` directly with `Skill` instances.

## Error and Security Semantics

- Missing `SKILL.md`, invalid UTF-8, invalid YAML, non-mapping frontmatter, absent/empty metadata,
  and missing closing delimiter raise `SkillDocumentError` without exposing the host root.
- Resource paths are validated by `Workspace` before execution. Traversal and absolute paths
  become the existing `WORKSPACE_VIOLATION` result when called through `ToolExecutor`.
- A generated shell tool never invokes its placeholder callable. With no compatible sandbox it
  returns the existing `SANDBOX_REQUIRED` failure.
- Symlinks resolving outside the skill root cannot be read or used as a working directory.

## Verification

- Unit tests cover LF/CRLF parsing, YAML errors, missing metadata, UTF-8 reads, path traversal,
  escaping symlinks, deterministic/collision-resistant names, and sandbox-required script tools.
- Agent tests cover skill prompt injection, resource-tool registration, no-shell behavior without
  a sandbox, duplicate names, collision rollback, and sandbox routing with the skill workspace.
- Full pytest, Ruff format/lint, and Mypy gates must pass.
