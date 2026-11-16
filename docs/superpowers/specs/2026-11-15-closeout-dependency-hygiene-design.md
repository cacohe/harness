# Closeout and Dependency Hygiene Design

## Intent

Close the documentation and repository-state gaps left after the production Harness roadmap was
implemented, and remove the remaining `schema -> core` dependency without breaking existing
imports or runtime behavior.

## Goals

- Make project metadata describe the implemented async-first Agent Harness rather than a sketch.
- Mark completed historical implementation-plan steps complete and record their integration state.
- Make the schema layer independent of the core runtime, including type-only imports.
- Preserve `src.core.async_utils.iterate_maybe_async` as a compatible import.
- Preserve `src.core.workspace.Workspace` and `WorkspaceViolation` as compatible imports.
- Ignore project-local tool, analysis, and coverage artifacts at the repository root.
- Leave the tree fully covered by the existing Ruff, Mypy, pytest, and lock gates.

## Non-goals

- Packaging or publishing the project.
- Changing public Agent, Harness, LLM, tool, or persistence behavior.
- Refactoring the rest of `src.core.async_utils`.
- Deleting virtual environments or another plan's local scratch data.
- Rewriting historical design decisions or claiming work without merged-code evidence.

## Approaches

### Move all async compatibility helpers to `utils`

This produces a cleaner utility module but creates unnecessary churn in every core import and a
larger compatibility surface. Rejected for this closeout.

### Duplicate the iterator helper in `schema`

This removes the import edge but creates two implementations whose cancellation and worker-thread
behavior can diverge. Rejected.

### Move only the layer-neutral iterator helper and re-export it from core

Selected. `src.utils.async_utils` owns `iterate_maybe_async` and its private synchronous-iterator
adapter. `src.schema.llm` imports it directly. `src.core.async_utils` re-exports the same object so
existing callers remain source-compatible, while core-only `run_sync` and `call_maybe_async` stay
where they are.

The same compatibility pattern applies to the workspace boundary exposed by `Tool`: the concrete,
layer-neutral boundary moves to `src.workspace`; `src.core.workspace` re-exports the exact classes,
and `src.schema.tool` uses only the neutral import for type checking. This avoids weakening the field
to `Any` or duplicating a partial workspace protocol.

## Verification

- A project-layout test rejects every absolute `src.core` import from `src/schema`.
- Utility tests exercise the canonical `src.utils.async_utils` implementation and verify the
  legacy core import resolves to the same function.
- Workspace tests verify the neutral and legacy imports resolve to the same classes.
- A completion audit reports no unchecked steps in the historical plans placed in scope.
- Root ignore rules cover Ruff, Superpowers, local-tool scratch, and coverage artifacts.
- Full Ruff, Mypy, pytest, `uv lock --check`, and `git diff --check` gates pass.
