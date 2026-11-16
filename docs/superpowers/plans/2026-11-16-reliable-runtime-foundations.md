# Reliable Runtime Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add durable execution facts, strict lifecycle settlement, recovery, idempotent submissions, resource governance, runtime ownership, architecture gates, and reproducible Agent profiles without breaking the existing Python API.

**Architecture:** Keep `Task` snapshots as the read model and add append-only `SessionEvent` facts behind a new `ExecutionStore` port. The in-memory and SQLite repositories implement both the legacy `TaskRepository` and the new port; `Harness` records facts before emitting best-effort observations, while `AgentSession` owns submission idempotency and `Agent` owns resource lifecycle.

**Tech Stack:** Python 3.11, Pydantic 2, SQLite/WAL, asyncio, pytest, Ruff, Mypy, uv.

**Spec:** `docs/superpowers/specs/2026-11-16-reliable-runtime-foundations-design.md`

## Global Constraints

- Preserve `chat`, `achat`, `resume`, `aresume`, direct `Harness.run/arun`, and TaskRepository-only injection compatibility.
- SQLite events, Task snapshot, and submission updates must commit in one transaction.
- Use `sha256-raw-utf8-v1` for message digests; do not normalize Unicode, whitespace, or newlines.
- Never automatically replay a tool call whose side-effect status is unknown.
- Persist only strict JSON and never persist credentials, host absolute paths, or arbitrary exception objects.
- Keep `src/schema` independent of `src/core`; keep `src/core` independent of concrete adapters and SDKs.
- Use TDD for every task and keep each task in an independently reviewable commit.
- Run commands from `F:\CodeSpace\harness-worktrees\reliable-runtime-foundations`.

## Review Focus

- Empty, oversized, control-character, or reused IDs must be rejected before any fact is committed; Task 1 pins these boundaries.
- A stale writer racing a current SQLite writer must receive `SequenceConflictError` without a partial Task/submission update; Task 4 pins this transaction boundary.
- Cancellation between a started fact and its side effect must produce one terminal fact and must suppress late thread results; Tasks 6 and 7 pin both model and tool races.
- Concurrent identical `message_id` submissions must execute once, while different content with the same ID must conflict; Task 9 pins cross-instance behavior.
- Close timeout must reject new work, keep the fact store alive, and continue the same shielded cleanup task; Task 11 pins this state transition.

---

## File Map

### New files

- `src/schema/execution.py`: durable fact, submission, outcome, and profile value objects.
- `src/core/identifiers.py`: injectable execution ID factory and ID validation.
- `src/core/execution.py`: execution context, lifecycle tracker, fact recorder, and limits.
- `src/core/recovery.py`: history validation and interrupted-tail repair.
- `src/core/runtime.py`: owned-resource startup, flush, cancellation, and close coordination.
- `tests/unit/schema/test_execution.py`: durable schema contracts.
- `tests/unit/core/test_execution.py`: lifecycle and recorder contracts.
- `tests/unit/core/test_recovery.py`: recovery and corruption cases.
- `tests/unit/core/test_runtime.py`: owned-resource lifecycle.

### Primary modified files

- `src/core/task.py`, `src/schema/task.py`: Task schema v2 and interrupted reason.
- `src/core/persistence.py`: ExecutionStore port, records, errors, and compatibility adapter.
- `src/persistence/in_memory.py`: in-memory atomic fact/snapshot/submission store.
- `src/persistence/sqlite.py`: schema v2 migration and transactional store.
- `src/core/loop.py`: durable lifecycle facts, deadlines, budgets, and correlated observations.
- `src/schema/event.py`, `src/core/event.py`, `src/observability/metrics.py`: observation correlation only.
- `src/demo/session.py`: stable sessions, idempotent submissions, optional bounded inbox, restore.
- `src/demo/agent.py`: composition, profile, opening sessions, and runtime ownership.
- `src/config.py`, `.env.example`, `README.md`: new limits and operational documentation.
- `tests/integration/test_project_layout.py`: architecture dependency gate.

---

### Task 1: Durable execution schema and stable IDs

**Files:**
- Create: `src/schema/execution.py`
- Create: `src/core/identifiers.py`
- Create: `tests/unit/schema/test_execution.py`
- Modify: `src/schema/__init__.py`
- Modify: `src/core/__init__.py`

**Interfaces:**
- Produces: `SessionEventType`, `SessionOutcome`, `PendingSessionEvent`, `SessionEvent`, `SubmissionState`, `SubmissionRecord`, `AgentProfile`.
- Produces: `ExecutionIdFactory` Protocol, `UuidExecutionIds`, `validate_execution_id(value: str, name: str) -> str`.

- [ ] **Step 1: Write failing schema and ID tests**

  Add tests for frozen models, strict JSON payloads, UTC timestamps, sequence `>=1`, ID maximum 256 UTF-8 bytes, empty/control-character IDs, outcome/type validation, profile secret exclusion, and distinct deterministic fake IDs. Assert non-finite floats and non-string mapping keys fail.

- [ ] **Step 2: Verify the tests fail**

  Run: `uv run --locked python -m pytest tests/unit/schema/test_execution.py -q -p no:cacheprovider`

  Expected: FAIL because `src.schema.execution` and `src.core.identifiers` do not exist.

- [ ] **Step 3: Implement the schema and ID factory**

  Use frozen Pydantic models with `extra="forbid"`. Define all fact types from the approved spec, including execution start/settle, pause/resume transitions, `model.attempt.settled`, `tool.call.requested`, and `profile.changed`. Validate JSON recursively without coercion.

- [ ] **Step 4: Run focused tests and static checks**

  Run: `uv run --locked python -m pytest tests/unit/schema/test_execution.py -q -p no:cacheprovider`

  Run: `uv run --locked mypy src/schema/execution.py src/core/identifiers.py`

  Expected: all pass.

- [ ] **Step 5: Commit**

  ```powershell
  git add src/schema/execution.py src/schema/__init__.py src/core/identifiers.py src/core/__init__.py tests/unit/schema/test_execution.py
  git commit -m "feat: define durable execution facts"
  ```

### Task 2: Task schema v2 compatibility

**Files:**
- Modify: `src/core/task.py`
- Modify: `src/schema/task.py`
- Modify: `tests/unit/core/test_task.py`

**Interfaces:**
- Consumes: validated IDs from Task 1.
- Produces: Task fields `session_id`, `message_id`, `turn_id`, `last_sequence`; `TaskFailReason.TASK_FAILED_INTERRUPTED`.

- [ ] **Step 1: Write failing Task migration tests**

  Assert a schema-v1 snapshot loads with all new fields `None`, new Tasks serialize as schema version 2, `last_sequence` is non-negative, identifiers reject invalid values, and interrupted terminal Tasks require a result.

- [ ] **Step 2: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/core/test_task.py -q -p no:cacheprovider`

  Expected: new-field assertions fail.

- [ ] **Step 3: Implement Task v2**

  Add optional identity fields without changing existing constructor calls. Keep `Task.from_dict` responsible for v1 normalization and reuse atomic `_apply_updates` validation.

- [ ] **Step 4: Verify Task and legacy behavior**

  Run: `uv run --locked python -m pytest tests/unit/core/test_task.py tests/unit/core/test_legacy_exports.py -q -p no:cacheprovider`

  Expected: pass.

- [ ] **Step 5: Commit**

  ```powershell
  git add src/core/task.py src/schema/task.py tests/unit/core/test_task.py
  git commit -m "feat: add durable identities to tasks"
  ```

### Task 3: ExecutionStore port and in-memory contract

**Files:**
- Modify: `src/core/persistence.py`
- Modify: `src/persistence/in_memory.py`
- Modify: `src/persistence/__init__.py`
- Modify: `tests/unit/core/test_persistence.py`
- Create: `tests/unit/persistence/test_execution_contract.py`

**Interfaces:**
- Produces: `ExecutionStore.commit(task, events, *, expected_sequence, submission=None) -> list[SessionEvent]`.
- Produces: `ExecutionStore.events(session_id, *, after_sequence=0)`, `submission(session_id, message_id)`, and `unfinished(session_id=None)`.
- Produces: `ExecutionCommitError`, `SequenceConflictError`, `SubmissionConflictError`, `ExecutionRecord`, and `SnapshotExecutionStore`.
- Produces: `InMemoryTaskRepository` implementing both old and new ports.

- [ ] **Step 1: Write shared contract tests against InMemoryTaskRepository**

  Cover contiguous sequence assignment from zero, atomic Task/event/submission updates, stale expected sequence, event filtering, independent returned snapshots, message conflicts, unfinished records, and no mutation after a failed commit.

- [ ] **Step 2: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/persistence/test_execution_contract.py tests/unit/core/test_persistence.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement ports, errors, in-memory store, and legacy adapter**

  Protect the in-memory implementation with `threading.RLock`. `expected_sequence=None` is accepted only by `SnapshotExecutionStore`; the real in-memory store requires an integer. Validate encoded Task and facts before acquiring commit state.

- [ ] **Step 4: Verify both old and new contracts**

  Run: `uv run --locked python -m pytest tests/unit/persistence/test_execution_contract.py tests/unit/core/test_persistence.py tests/unit/core/test_ports.py -q -p no:cacheprovider`

  Expected: pass.

- [ ] **Step 5: Commit**

  ```powershell
  git add src/core/persistence.py src/persistence/in_memory.py src/persistence/__init__.py tests/unit/core/test_persistence.py tests/unit/persistence/test_execution_contract.py
  git commit -m "feat: add atomic execution store port"
  ```

### Task 4: SQLite schema v2 and atomic commits

**Files:**
- Modify: `src/persistence/sqlite.py`
- Modify: `tests/unit/persistence/test_sqlite.py`
- Modify: `tests/unit/persistence/test_execution_contract.py`

**Interfaces:**
- Consumes: ExecutionStore contract from Task 3.
- Produces: SQLite v2 tables `session_streams`, `session_events`, and `submissions`; v1-to-v2 migration.

- [ ] **Step 1: Add SQLite contract and migration tests**

  Parameterize the shared contract for SQLite. Add v1 databases with zero and multiple Tasks; assert transactional migration, exact table/column/PK/FK validation, rollback on injected DDL failure, higher-version rejection, corrupt event rejection, and reopening after commit.

- [ ] **Step 2: Add stale-writer concurrency tests**

  Open two repository instances. Assert only one commit at `expected_sequence=0` succeeds and the loser gets `SequenceConflictError` with no Task or submission partial update. Add concurrent identical and conflicting submission inserts.

- [ ] **Step 3: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/persistence/test_sqlite.py tests/unit/persistence/test_execution_contract.py -q -p no:cacheprovider --basetemp=.pytest-sqlite-v2`

- [ ] **Step 4: Implement schema migration and atomic commit/read methods**

  Use `BEGIN IMMEDIATE`, `PRAGMA foreign_keys=ON`, canonical JSON with `allow_nan=False`, and CAS against `session_streams.next_sequence`. Store `digest_algorithm` with each submission. Validate all rows through Pydantic on read.

- [ ] **Step 5: Verify repository tests**

  Run the Task 4 pytest command again.

  Expected: pass with no partial rows after any injected failure.

- [ ] **Step 6: Commit**

  ```powershell
  git add src/persistence/sqlite.py tests/unit/persistence/test_sqlite.py tests/unit/persistence/test_execution_contract.py
  git commit -m "feat: persist execution facts in sqlite"
  ```

### Task 5: Lifecycle tracker and fact recorder

**Files:**
- Create: `src/core/execution.py`
- Create: `tests/unit/core/test_execution.py`
- Modify: `src/core/__init__.py`

**Interfaces:**
- Consumes: ExecutionStore, durable schema, and ID factory.
- Produces: `ExecutionContext`, `ExecutionLimits`, `LifecycleTracker`, `FactRecorder.commit(task, events, submission=None) -> list[SessionEvent]`, and `PayloadLimitError`.

- [ ] **Step 1: Write failing lifecycle tests**

  Exercise valid completed, paused/resumed, failed, cancelled, and interrupted histories. Reject duplicate starts, duplicate terminal facts, settlement without start except requested-to-rejected tool calls, child identity mismatches, events after terminal turn, and resume without pause.

- [ ] **Step 2: Write failing fact-recorder tests**

  Assert it carries Task `last_sequence`, passes CAS, updates sequence only after success, enforces event and snapshot byte limits before store mutation, and preserves the original cause on commit failure.

- [ ] **Step 3: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/core/test_execution.py -q -p no:cacheprovider`

- [ ] **Step 4: Implement tracker, context, limits, and recorder**

  Keep lifecycle validation pure. Measure canonical UTF-8 JSON bytes. Use one recorder commit for every fact batch plus Task snapshot.

- [ ] **Step 5: Verify focused tests and Mypy**

  Run: `uv run --locked python -m pytest tests/unit/core/test_execution.py -q -p no:cacheprovider`

  Run: `uv run --locked mypy src/core/execution.py`

- [ ] **Step 6: Commit**

  ```powershell
  git add src/core/execution.py src/core/__init__.py tests/unit/core/test_execution.py
  git commit -m "feat: enforce durable lifecycle settlement"
  ```

### Task 6: Turn, step, and model facts in Harness

**Files:**
- Modify: `src/core/loop.py`
- Modify: `src/schema/event.py`
- Modify: `src/core/event.py`
- Modify: `src/observability/metrics.py`
- Modify: `tests/unit/core/test_loop.py`
- Modify: `tests/unit/core/test_event.py`
- Modify: `tests/unit/observability/test_metrics.py`

**Interfaces:**
- Consumes: FactRecorder and ExecutionContext from Task 5.
- Produces: `Harness.arun(..., execution_context: ExecutionContext | None = None)` compatibility extension.
- Produces: optional observation fields `session_id`, `execution_id`, `turn_id`, `attempt_id`, `tool_call_id`.

- [ ] **Step 1: Add successful and failing lifecycle tests**

  For planning and inference, assert fact order for completed, retry, exhausted retry, gateway error, cancellation, and caller task cancellation. A model attempt must always settle before assistant settlement or upper-level failure settlement.

- [ ] **Step 2: Add observation-separation tests**

  Assert durable commit precedes matching EventSink emission, LLM delta never enters ExecutionStore, observation failure cannot roll back facts, and correlated Metrics keys separate multiple attempts in one step.

- [ ] **Step 3: Add cancellation race test**

  Cancel after `model.attempt.started` commit but before the gateway returns; assert exactly one cancelled attempt settlement, one cancelled execution settlement, no assistant fact, and no late observation.

- [ ] **Step 4: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/core/test_loop.py tests/unit/core/test_event.py tests/unit/observability/test_metrics.py -q -p no:cacheprovider`

- [ ] **Step 5: Refactor Harness persistence through FactRecorder**

  Preserve direct TaskRepository-only construction by wrapping it with SnapshotExecutionStore. Add context-aware fact helpers and extend Event dataclasses with optional correlation fields without changing existing emitters.

- [ ] **Step 6: Verify core and metrics tests**

  Run the Task 6 pytest command again.

- [ ] **Step 7: Commit**

  ```powershell
  git add src/core/loop.py src/schema/event.py src/core/event.py src/observability/metrics.py tests/unit/core/test_loop.py tests/unit/core/test_event.py tests/unit/observability/test_metrics.py
  git commit -m "feat: record turn and model lifecycle facts"
  ```

### Task 7: Tool facts, approval pause, and resume

**Files:**
- Modify: `src/core/loop.py`
- Modify: `src/core/tool.py`
- Modify: `src/tool/tool_executor.py`
- Modify: `tests/unit/core/test_loop.py`
- Modify: `tests/unit/tool/test_tool_executor.py`

**Interfaces:**
- Consumes: Harness fact helpers from Task 6.
- Produces: requested/started/settled tool lifecycle and step/turn/submission pause-resume transitions.

- [ ] **Step 1: Write failing approval lifecycle tests**

  Assert model tool calls create requested facts; approval-required calls pause without started facts; rejection settles requested calls as rejected; approval resume uses a new execution ID and the original turn/step; final completion emits exactly one step/turn/submission settlement.

- [ ] **Step 2: Write failing side-effect ambiguity tests**

  Cancel or fail persistence immediately after `tool.call.started`. Assert no later tool is started, one interrupted/cancelled tool settlement is attempted, and a late synchronous tool result cannot publish facts or observations.

- [ ] **Step 3: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/core/test_loop.py tests/unit/tool/test_tool_executor.py -q -p no:cacheprovider`

- [ ] **Step 4: Implement tool and pause/resume facts**

  Persist requested calls with assistant settlement, persist started immediately before executor dispatch, and normalize every returned ToolExecutionResult before settled commit. Keep existing approval API and Task history format.

- [ ] **Step 5: Verify tool and loop tests**

  Run the Task 7 pytest command again.

- [ ] **Step 6: Commit**

  ```powershell
  git add src/core/loop.py src/core/tool.py src/tool/tool_executor.py tests/unit/core/test_loop.py tests/unit/tool/test_tool_executor.py
  git commit -m "feat: settle durable tool lifecycles"
  ```

### Task 8: Deadlines and payload budgets

**Files:**
- Modify: `src/config.py`
- Modify: `src/core/loop.py`
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/core/test_loop.py`
- Modify: `tests/unit/demo/test_agent.py`
- Create: `tests/unit/test_config.py`

**Interfaces:**
- Produces Config fields with defaults: `model_timeout_seconds=90`, `tool_timeout_seconds=30`, `max_model_output_bytes=1_048_576`, `max_tool_result_bytes=1_048_576`, `max_event_bytes=1_048_576`, `max_task_snapshot_bytes=8_388_608`, `max_pending_submissions=0`, `close_timeout_seconds=30`.

- [ ] **Step 1: Write config validation tests**

  Cover env/file aliases, positive finite seconds, positive byte integers, zero-only allowance for `max_pending_submissions`, booleans, NaN/Infinity, and default values.

- [ ] **Step 2: Write model/tool budget tests**

  Assert separate model and tool timeouts, cumulative streamed model bytes, tool result bytes for local and async tools, exact boundary acceptance, oversized failure classification, no invalid JSON truncation, and late-result suppression.

- [ ] **Step 3: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/test_config.py tests/unit/core/test_loop.py tests/unit/demo/test_agent.py -q -p no:cacheprovider`

- [ ] **Step 4: Implement validated Config fields and Harness deadlines**

  Use `asyncio.timeout` for native async operations and cancellation fencing for thread-backed calls. Apply output limits before Task mutation and FactRecorder commit.

- [ ] **Step 5: Verify focused tests**

  Run the Task 8 pytest command again.

- [ ] **Step 6: Commit**

  ```powershell
  git add src/config.py src/core/loop.py src/demo/agent.py tests/unit/test_config.py tests/unit/core/test_loop.py tests/unit/demo/test_agent.py
  git commit -m "feat: govern model and tool resources"
  ```

### Task 9: Stable sessions and idempotent submissions

**Files:**
- Modify: `src/demo/session.py`
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_session.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `tests/load/test_shared_agent_load.py`

**Interfaces:**
- Produces: `AgentSession(..., session_id: str | None = None, execution_store: ExecutionStore, id_factory: ExecutionIdFactory, max_pending_submissions: int = 0)`.
- Produces: `chat/achat(..., *, message_id: str | None = None)` and deterministic `message_digest(message: str) -> tuple[str, str]` returning algorithm and hex digest.
- Produces: optional bounded FIFO only when `max_pending_submissions > 0`.

- [ ] **Step 1: Write digest and retry tests**

  Assert raw UTF-8 semantics for composed/decomposed Unicode, CRLF/LF, leading whitespace, and empty messages. Assert same ID/content returns the stored Task without invoking recall/model/tools again and different content raises SubmissionConflictError.

- [ ] **Step 2: Write concurrent cross-instance tests**

  Create two AgentSession instances with the same session ID and shared SQLite repository. Submit identical message IDs concurrently; assert one model execution and the same Task ID. Repeat with conflicting content and assert one conflict.

- [ ] **Step 3: Write queue compatibility tests**

  Default zero retains immediate `session operation already in progress`. Positive capacity executes FIFO, rejects overflow, and cancellation before claim writes no submission fact.

- [ ] **Step 4: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/demo/test_session.py tests/unit/demo/test_agent.py tests/load/test_shared_agent_load.py -q -p no:cacheprovider --basetemp=.pytest-idempotency`

- [ ] **Step 5: Implement stable IDs, digest, CAS retry, and optional inbox**

  Accept submissions through ExecutionStore before memory recall or Harness invocation. On sequence conflict, reread submission and return or conflict deterministically.

- [ ] **Step 6: Verify Session and load tests**

  Run the Task 9 pytest command again.

- [ ] **Step 7: Commit**

  ```powershell
  git add src/demo/session.py src/demo/agent.py tests/unit/demo/test_session.py tests/unit/demo/test_agent.py tests/load/test_shared_agent_load.py
  git commit -m "feat: make session submissions idempotent"
  ```

### Task 10: Recovery and reopening sessions

**Files:**
- Create: `src/core/recovery.py`
- Create: `tests/unit/core/test_recovery.py`
- Modify: `src/demo/session.py`
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_session.py`

**Interfaces:**
- Produces: `ExecutionRecovery.recover_session(session_id: str) -> list[SessionEvent]`.
- Produces: `Agent.open_session(session_id: str) -> AgentSession` and transcript/pending-run reconstruction.

- [ ] **Step 1: Write history validation tests**

  Reject sequence gaps, duplicate terminal facts, mismatched hierarchy, impossible resume, unknown event schema, and Task/event identity disagreement without modifying storage.

- [ ] **Step 2: Write interrupted-tail tests**

  Cut valid histories after every started fact. Assert recovery appends the minimal reverse-order interrupted settlements in one commit, marks running Tasks failed/interrupted, preserves paused Tasks, never re-executes tools, and is idempotent on a second call.

- [ ] **Step 3: Write reopen tests**

  Reopen completed and paused sessions from SQLite. Assert ordered transcript reconstruction, original pending approvals, resume ownership, profile history retention, and no duplicate memory write for already completed Tasks.

- [ ] **Step 4: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/core/test_recovery.py tests/unit/demo/test_session.py -q -p no:cacheprovider --basetemp=.pytest-recovery`

- [ ] **Step 5: Implement recovery and Agent.open_session**

  Fold facts with LifecycleTracker. Build compensating events without guessing external results. Keep recovery synchronous because the current repository port is synchronous and bounded.

- [ ] **Step 6: Verify recovery tests**

  Run the Task 10 pytest command again.

- [ ] **Step 7: Commit**

  ```powershell
  git add src/core/recovery.py src/demo/session.py src/demo/agent.py tests/unit/core/test_recovery.py tests/unit/demo/test_session.py
  git commit -m "feat: recover interrupted agent sessions"
  ```

### Task 11: Owned runtime lifecycle and close semantics

**Files:**
- Create: `src/core/runtime.py`
- Create: `tests/unit/core/test_runtime.py`
- Modify: `src/demo/agent.py`
- Modify: `src/demo/session.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `tests/integration/test_mcp_agent.py`

**Interfaces:**
- Produces: `RuntimeResource` Protocol, `RuntimeCoordinator.start()`, `register_operation(task)`, and `aclose(timeout_seconds)`.
- Produces: `RuntimeClosedError`, `RuntimeCloseTimeoutError`.

- [ ] **Step 1: Write startup and ownership tests**

  Assert owned resources start in order, partial startup closes started resources in reverse, injected resources remain caller-owned unless explicitly listed, and lazy first operation performs one idempotent start.

- [ ] **Step 2: Write close-state tests**

  Assert close rejects new submissions/MCP changes, cancels active Session operations, waits for fact settlement, flushes then closes resources in reverse, remains idempotent, and keeps history readable.

- [ ] **Step 3: Write timeout and caller-cancellation tests**

  On timeout, assert `RuntimeCloseTimeoutError`, Agent state `closing`, fact store still open, and the same shielded cleanup task continues. Cancelling one aclose waiter must not cancel cleanup; a later aclose observes completion.

- [ ] **Step 4: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/core/test_runtime.py tests/unit/demo/test_agent.py tests/integration/test_mcp_agent.py -q -p no:cacheprovider`

- [ ] **Step 5: Implement coordinator and integrate Agent/Session**

  Track asyncio operation Tasks at the Session boundary. Ensure thread-backed late results remain fenced and are not awaited by close after the async operation settles.

- [ ] **Step 6: Verify lifecycle tests**

  Run the Task 11 pytest command again.

- [ ] **Step 7: Commit**

  ```powershell
  git add src/core/runtime.py src/demo/agent.py src/demo/session.py tests/unit/core/test_runtime.py tests/unit/demo/test_agent.py tests/integration/test_mcp_agent.py
  git commit -m "feat: coordinate agent runtime shutdown"
  ```

### Task 12: Agent Profile and capability digests

**Files:**
- Modify: `src/demo/agent.py`
- Modify: `src/demo/session.py`
- Modify: `src/tool/tool_registry.py`
- Modify: `src/skill/directory.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `tests/unit/demo/test_session.py`

**Interfaces:**
- Consumes: AgentProfile from Task 1.
- Produces: `build_agent_profile(...) -> AgentProfile`, deterministic sorted capability digests, and `profile.changed` facts.

- [ ] **Step 1: Write deterministic profile tests**

  Assert tool registration order does not change the digest, Skill CRLF/LF normalization is not silently applied, API keys/base credentials and absolute paths never appear, and identical configuration gives identical profile snapshots.

- [ ] **Step 2: Write profile-change tests**

  Assert `session.started` contains the initial profile, MCP tool changes alter the turn capability digest, reopening with a different profile appends one `profile.changed` before accepting a turn, and reopening unchanged appends nothing.

- [ ] **Step 3: Verify failure**

  Run: `uv run --locked python -m pytest tests/unit/demo/test_agent.py tests/unit/demo/test_session.py -q -p no:cacheprovider`

- [ ] **Step 4: Implement profile and capability digest integration**

  Hash canonical JSON of public model settings, sorted tool contracts, Skill name/description/body digests, memory policy version, and fact schema version. Do not include callable identity or process-specific object repr.

- [ ] **Step 5: Verify profile tests**

  Run the Task 12 pytest command again.

- [ ] **Step 6: Commit**

  ```powershell
  git add src/demo/agent.py src/demo/session.py src/tool/tool_registry.py src/skill/directory.py tests/unit/demo/test_agent.py tests/unit/demo/test_session.py
  git commit -m "feat: record reproducible agent profiles"
  ```

### Task 13: Architecture gate and operational documentation

**Files:**
- Modify: `tests/integration/test_project_layout.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `pyproject.toml`
- Create: `docs/reliable-runtime.md`

**Interfaces:**
- Consumes: all public APIs from Tasks 1-12.
- Produces: AST dependency gate and documented migration/operation contract.

- [ ] **Step 1: Write failing architecture-gate fixtures**

  Add parameterized fixtures for forbidden schema-to-core, core-to-SQLite/LiteLLM/FastMCP/Docker/demo imports, relative-import bypasses, and common `importlib.import_module` literals. Add allowed Protocol and composition-root cases.

- [ ] **Step 2: Verify the new negative fixtures fail**

  Run: `uv run --locked python -m pytest tests/integration/test_project_layout.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement the AST gate and fix any real violations**

  Keep an explicit allowlist containing a reason per entry. Do not move concrete dependencies into Core to satisfy convenience imports.

- [ ] **Step 4: Document config, SQLite v2 migration, recovery, facts vs observations, idempotency, limits, and close behavior**

  Update `.env.example` with all Task 8 settings and explain that v2 databases require backup before upgrade and are not writable by v1 code.

- [ ] **Step 5: Run docs-adjacent checks**

  Run: `uv run --locked ruff format --check src tests`

  Run: `uv run --locked ruff check --no-cache src tests`

  Run: `uv run --locked python -m pytest tests/integration/test_project_layout.py -q -p no:cacheprovider`

  Expected: pass.

- [ ] **Step 6: Commit**

  ```powershell
  git add tests/integration/test_project_layout.py README.md .env.example pyproject.toml docs/reliable-runtime.md
  git commit -m "docs: define reliable runtime operations"
  ```

### Task 14: Full regression, load validation, and closeout

**Files:**
- Modify as failures require: tests and implementation files from Tasks 1-13 only.
- Modify: `docs/superpowers/plans/2026-11-16-reliable-runtime-foundations.md`

**Interfaces:**
- Consumes: complete branch.
- Produces: verified release candidate with plan checkboxes and actual commands recorded.

- [ ] **Step 1: Run formatting and lint gates**

  Run: `uv run --locked ruff format --check src tests`

  Run: `uv run --locked ruff check --no-cache src tests`

  Expected: pass.

- [ ] **Step 2: Run type checking**

  Run: `uv run --locked mypy src`

  Expected: no issues.

- [ ] **Step 3: Run the full deterministic suite**

  Run: `uv run --locked python -m pytest -q -p no:cacheprovider --basetemp=.pytest-final`

  Expected: all tests pass; only tests explicitly requiring unavailable external services may skip.

- [ ] **Step 4: Run the shared-Agent load regression separately**

  Run: `uv run --locked python -m pytest tests/load/test_shared_agent_load.py -q -p no:cacheprovider --basetemp=.pytest-load`

  Expected: concurrent Session transcripts, sequences, submissions, memory, and metrics do not cross-contaminate.

- [ ] **Step 5: Inspect repository state and migration artifacts**

  Run: `git diff --check`

  Run: `git status --short`

  Expected: no generated SQLite, pytest, environment, or secret files; only intentional source/docs changes before the final commit.

- [ ] **Step 6: Update this plan with actual verification results and commit closeout**

  ```powershell
  git add docs/superpowers/plans/2026-11-16-reliable-runtime-foundations.md
  git commit -m "docs: complete reliable runtime plan"
  ```

---

## Self-Review Results

- **Spec coverage:** All ten approved design sections map to Tasks 1-13; Task 14 verifies the branch as a whole.
- **Step granularity:** Each task has a failing-test step, failure verification, one bounded implementation responsibility, passing verification, and a focused commit.
- **Type consistency:** Durable schema and IDs originate in Task 1; persistence in Tasks 3-4; recorder/context in Task 5; all later tasks consume those exact interfaces.
- **Review Focus:** ID boundaries, stale writers, cancellation races, concurrent idempotency, and close timeouts are each assigned explicit tests.
- **Proportion:** The plan names interfaces, tests, commands, and decisions without embedding implementation bodies already determined by the spec.
