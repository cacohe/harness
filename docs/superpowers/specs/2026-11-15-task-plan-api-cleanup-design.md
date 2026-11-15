# TaskPlan API Cleanup Design

## Context

`TaskPlan.advance_plan(current)` has contained only `...` since `TaskPlan` was introduced. The
runtime never calls it, no test or documentation defines its behavior, and the model already owns
the complete `completed/current/remaining` snapshot returned during planning.

## Decision

Remove the unsupported mutator instead of inventing transition semantics. `TaskPlan` remains a
validated transport and persistence model. Harness planning continues to replace the complete plan
through `Task.set_plan()`.

This keeps the public surface honest: callers cannot invoke a method that silently reports success
without changing state. A future incremental planner may add an operation only after defining step
identity, duplicate handling, ordering, retries, replanning, atomic validation, and persistence.

## Compatibility

No repository caller uses the method. External callers that invoked it only received `None` and no
state change; removing it converts that latent logic error into an immediate `AttributeError`.
Serialized Task/TaskPlan data and model-generated plan schemas are unchanged.

## Verification

- Add a public-surface regression test asserting `TaskPlan` does not advertise the unsupported
  method.
- Run Task model tests and the full quality gate.

