# Agent Session and TaskRun Isolation Design

## Intent

Allow one configured `Agent` to serve multiple independent conversations concurrently without
sharing transcript, paused-task, approval, or resume state. Preserve the existing `Agent.chat`
and `Agent.achat` API through a default session so current callers do not need an immediate
migration.

## Goals

- Add an explicit `AgentSession` handle created by `Agent.new_session()`.
- Give every session its own transcript and at most one current `TaskRun` waiting for approval.
- Permit `achat()` calls on different sessions to overlap on the same event loop.
- Reject concurrent/reentrant operations on the same session instead of racing or silently
  branching conversation history.
- Reject a new turn while that session has a paused run; callers must decide and resume it.
- Keep approval requests and decisions scoped to the session that owns the paused task.
- Make Agent closure immediately invalidate every existing session.
- Keep direct `Agent` methods and compatibility properties delegated to one default session.

## Non-goals

- Running two turns concurrently inside one ordered conversation.
- Persisting session metadata or transcript across process restarts; durable storage is the next
  independent phase.
- Providing distributed locks across processes.
- Changing `Harness.arun()` or `Task` serialization.
- Packaging or publishing the project.

## Anti-goals

- No global Agent lock that serializes unrelated sessions.
- No implicit `session_id` dictionary hidden inside every Agent call.
- No approval lookup by tool-call id across all sessions.
- No background task spawned without an explicit caller-owned awaitable.

## Constraints

- Python 3.11 or newer.
- Existing `Agent.chat/achat/resume/aresume`, `transcript`, `pending_task`, and
  `pending_message` behavior remains source-compatible for the default session.
- A session operation owns its lock across all awaits and releases it in `finally` on success,
  cancellation, or error.
- The guard acquisition is non-blocking; same-session reentry raises `RuntimeError` immediately
  and never blocks the event loop.
- Shared Harness dependencies are treated as stateless or task-keyed; per-run mutation stays on
  the `Task` object passed into Harness.

## Approaches Considered

### 1. Global Agent lock

This prevents corruption but serializes independent users and defeats the stated concurrency
goal. Rejected.

### 2. Agent-owned session-id registry

This can support concurrency but turns the composition root into a lifecycle registry with
implicit lookup, cleanup, and cross-session approval hazards. It is appropriate for a higher
level service, not the core Agent entry. Rejected.

### 3. Explicit AgentSession handles

Selected. The Agent continues to own shared model/tool/MCP composition. Each returned session
owns ordered conversation state, and callers retain or discard that handle using normal object
lifetime. This makes state ownership visible in the type and keeps unrelated sessions naturally
parallel.

## Design

### TaskRun

`src/demo/session.py` defines a small `TaskRun` dataclass containing the current `Task` and the
original user message for that turn. The message is retained only while the task is paused so a
successful resume can append exactly one user/assistant pair to the owning transcript.

### AgentSession

`AgentSession` receives the shared `Harness`, `TaskRepository`, and an `ensure_open` callback from
Agent. It owns:

- a stable generated `id`;
- `transcript: list[tuple[str, str]]`;
- `pending_run: TaskRun | None`;
- a private `threading.Lock` used only with non-blocking acquire/release.

`achat()` acquires the session guard, rejects an existing `pending_run`, creates a Task from only
that session's transcript, awaits `Harness.arun`, and either records the paused `TaskRun` or
commits the completed turn. `aresume()` requires a pending run, awaits the same task, clears the
pending run when it leaves paused state, and commits the turn once on completion. Cancellation
and exceptions release the operation guard in all cases.

`approval_requests()` and `decide_tool_call()` inspect only `pending_run.task`. A decision is
saved through the shared repository exactly as today. The synchronous methods use the existing
`run_sync` bridge and retain its active-event-loop rejection.

### Agent compatibility layer

Agent creates `_default_session` after the Harness is composed. `new_session()` returns another
session bound to the same Harness/repository/open check. Existing chat, resume, approval, and
transcript methods delegate to the default session. Compatibility properties proxy
`pending_task` and `pending_message`; setters exist only for legacy tests/callers and update the
default session's `TaskRun` without affecting other sessions.

### Concurrency semantics

- Different sessions: may execute concurrently and receive independent prompts/transcripts.
- Same session: a second chat/resume while one is active raises `RuntimeError("session operation
  already in progress")` immediately.
- Paused session: new chat raises `ValueError("session has a task waiting for approval")`.
- Wrong-session approval: behaves as no pending call and cannot mutate another session.
- Closed Agent: every session method that starts or mutates work raises `RuntimeError("Agent is
  closed")`; read-only transcript and approval listing remain safe snapshots.

## Verification

- Existing Agent tests remain green through the compatibility layer.
- New tests run two sessions through rendezvous gateways and prove actual overlap plus isolated
  prompts/transcripts.
- Tests cover same-session reentry, paused-turn rejection, wrong-session approvals, independent
  paused tasks, cancellation/error lock release, exactly-once transcript commit, and Agent-close
  invalidation.
- Full pytest, Ruff format/lint, and Mypy gates must pass.
