# Application API, Skill Loading, and Low-Risk Cleanup Design

## Intent

Make the existing production API and ownership boundaries visible in the directory structure without
adding behavior. `Agent` and `AgentSession` become application-layer APIs, Skill discovery and resource
tool construction become one `src.skill` responsibility, and the previously identified low-risk cleanup
removes internal compatibility-path usage, misplaced tests, and dead files.

## Scope

- Create `src.application` as the canonical home of `Agent`, `AgentSession`, and their supporting types.
- Preserve `src.demo.agent` and `src.demo.session` as exact-object compatibility re-exports; keep the CLI
  in `src.demo.main`.
- Add a Skill bundle load path that reads each `SKILL.md` once and returns its model plus bound resource
  tools; move multi-directory loading and duplicate-name validation into `src.skill`.
- Make production and ordinary tests import `Workspace` and the async iterator from their canonical
  neutral modules. Keep the two `src.core` shims and their identity tests.
- Move implementation-specific tests to `tests/unit/application`, `tests/unit/context`,
  `tests/unit/memory`, and `tests/unit/persistence`.
- Move default context prompts next to `DefaultContextBuilder`.
- Delete the tracked, unused `src/tool/mcp_tools.py` and `src/utils/util.py` files.
- Update README imports to the canonical application and workspace paths.

## Non-Goals

- Do not redesign `Harness`, LLM schemas, event output, ToolCatalog transactions, persistence behavior,
  configuration precedence, or any public runtime semantics.
- Do not remove legacy imports or change object identities.
- Do not add dependencies or packaging/distribution metadata.
- Do not include the source checkout's uncommitted `.gitignore`, `.env.example`, or `TODO.md` changes.

## Approaches

1. **Selected: application package plus compatibility modules.** This makes the production entry point
   explicit, keeps the CLI visibly demo-only, and avoids a breaking import migration.
2. Root-level `src/agent.py` and `src/session.py`. This is smaller but adds more root-module sprawl and
   gives no package home for future application-level composition code.
3. Rename `demo` outright. This is cleanest superficially but breaks every documented import and offers
   no compatibility period.

## Design

`src.application.agent.Agent` and `src.application.session.AgentSession` are canonical. The legacy demo
modules import and re-export the exact same objects. `src.application.__init__` exposes the primary API.
All repository-owned callers and documentation use the canonical path so the compatibility layer is not
an internal dependency.

`SkillDirectory.load_bundle(include_shell=...)` performs one document read, builds the `Skill`, and derives
resource tools from that same parsed name. `src.skill.loader.load_skill_directories(...)` owns directory
iteration and duplicate-name validation. Agent only consumes its returned `(skills, tools)` result.

Architecture tests enforce that internal modules do not depend on `src.core.workspace` or the core async
iterator compatibility export. Existing identity tests continue to prove legacy compatibility.

## Verification

- RED→GREEN tests for canonical application imports and exact legacy identity.
- RED→GREEN test proving one changing Skill document cannot produce mismatched Skill/tool names.
- RED→GREEN AST layout test for internal canonical imports.
- Focused application, skill, context, memory, persistence, workspace, async, integration, and load tests.
- Full Ruff, Mypy, pytest, uv lock, and diff checks.
