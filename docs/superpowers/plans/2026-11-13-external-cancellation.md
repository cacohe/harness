# External Cancellation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-run, thread-safe cooperative cancellation to the synchronous Harness and Agent entry points without starting work after cancellation or corrupting tool-call order.

**Architecture:** A `threading.Event`-backed `CancellationToken` is passed into each `Harness.run` and never stored on the Harness. Tool batches accept a cancellation predicate and retain original call order; Harness cancellation has a dedicated terminal path that also closes paused held calls. Blocking third-party calls remain non-preemptive, but their results/errors are discarded when cancellation is observed afterward.

**Tech Stack:** Python 3.11+, `threading.Event`, Pydantic, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-13-external-cancellation-design.md`

## Global Constraints

- Cancellation is cooperative; do not claim or implement forced interruption of an in-flight gateway, tool, thread, process, or sandbox.
- One token controls one run and is never stored on `Harness` or `Agent`.
- Terminal tasks keep their existing result even when given a cancelled token.
- Repository and event-sink boundary exceptions must propagate instead of becoming cancellation.
- Stable task cancellation output is `"Task cancelled."`; stable tool cancellation output is `"Tool call cancelled."`.
- Existing synchronous APIs remain source-compatible when no token/predicate is supplied.
- Every production behavior follows RED → GREEN → refactor and each task ends with a commit.

## Review Focus

- A token cancelled between the last checkpoint and gateway exception handling must win over retry or `llm_error`; Task 3 tests this race.
- A paused task with undecided approvals must still be cancellable without running held tools or emitting `TASK_RESUMED`; Task 3 tests this path.
- A mixed tool batch (`completed`, then cancellation) must keep the exact original tool-message order; Task 2 tests ordered results and context construction.
- Reusing a cancelled token on a different run must cancel that run immediately without leaking mutation between tasks; Tasks 1 and 3 test this.
- Cancellation during timeout/max-step finalization must produce one persisted/event reason (`cancel`) rather than conflicting failure events; Task 3 tests finalization cancellation.

---

### Task 1: Thread-safe Cancellation Capability

**Files:**
- Create: `src/core/cancellation.py`
- Create: `tests/unit/core/test_cancellation.py`

**Interfaces:**
- Consumes: Python `threading.Event`.
- Produces: `CancellationToken.cancel() -> None`, `CancellationToken.is_cancelled() -> bool`, and `CancellationToken.wait(timeout: float) -> bool`.

- [ ] **Step 1: Write failing token lifecycle tests**

Add tests named:

```python
def test_token_starts_active_and_cancel_is_idempotent(): ...
def test_cancel_wakes_a_waiting_thread(): ...
def test_reusing_a_cancelled_token_remains_cancelled(): ...
```

Use a real helper thread plus bounded `threading.Event` coordination; do not use sleeps as synchronization. Assert both calls to `cancel()` are harmless and `wait()` returns `True` after cancellation.

- [ ] **Step 2: Run Task 1 tests and confirm RED**

Run: `python -m pytest tests/unit/core/test_cancellation.py -q -p no:cacheprovider`

Expected: collection fails because `src.core.cancellation` does not exist.

- [ ] **Step 3: Implement `CancellationToken`**

Create the exact interface above with one private `threading.Event`. `is_cancelled` delegates to `is_set`; `wait` preserves `Event.wait` return semantics; no reset API is added.

- [ ] **Step 4: Verify and commit Task 1**

Run:

```bash
python -m pytest tests/unit/core/test_cancellation.py -q -p no:cacheprovider
python -m ruff check --no-cache src/core/cancellation.py tests/unit/core/test_cancellation.py
python -m mypy src/core/cancellation.py
```

Expected: all exit 0, then commit:

```bash
git add src/core/cancellation.py tests/unit/core/test_cancellation.py
git commit -m "feat: add cancellation token"
```

### Task 2: Ordered and Cancellable Tool Batches

**Files:**
- Modify: `src/schema/tool.py`
- Modify: `src/core/tool.py`
- Modify: `src/core/context.py`
- Modify: `tests/unit/tool/test_tool_executor.py`
- Create: `tests/unit/core/test_context.py`

**Interfaces:**
- Consumes: optional `should_cancel: Callable[[], bool] | None`.
- Produces: `ToolCallResult.ordered_tools: list[ToolExecutionResult]`; `ToolExecutor.execute_tool_calls(..., should_cancel=None)`; stable cancelled results with fail reason `TOOL_EXECUTION_FAIL_REASON_CANCELLED` and output `"Tool call cancelled."`.

- [ ] **Step 1: Write failing ordered-result tests**

Add tests proving:

```python
def test_tool_call_result_keeps_original_mixed_call_order(): ...
def test_build_tool_messages_prefers_ordered_tools(): ...
def test_build_tool_messages_falls_back_for_legacy_results(): ...
```

Use three calls whose category order differs from input order. Assert `ordered_tools` and built tool messages match the model call IDs exactly.

- [ ] **Step 2: Run ordered-result tests and confirm RED**

Run the three exact tests with `python -m pytest ... -q -p no:cacheprovider`.

Expected: missing `ordered_tools` and current category concatenation fail.

- [ ] **Step 3: Implement the ordered result contract**

Add `ordered_tools` with a default factory. Populate it for every executor batch result. Change `build_tool_messages` to use it when non-empty and retain the existing completed/failed/pending fallback.

- [ ] **Step 4: Write failing cancellation-between-tools tests**

Add:

```python
def test_cancellation_stops_later_tools_and_preserves_order(): ...
def test_cancelled_tools_skip_before_call_but_reach_after_call(): ...
def test_precancelled_batch_never_invokes_policy_validator_or_runner(): ...
```

Cancel from the first runner after it completes. Assert every later call is failed/cancelled in order, has the stable output, and no later mutation hook ran.

- [ ] **Step 5: Implement cancellable batch routing**

Before each call, check `should_cancel` when supplied. For cancellation, create the failed result directly, call only `after_call`, append to `failed_tools` and `ordered_tools`, and continue so every model call receives a result.

- [ ] **Step 6: Verify and commit Task 2**

Run:

```bash
python -m pytest tests/unit/tool/test_tool_executor.py tests/unit/core/test_context.py tests/unit/core/test_loop.py -q -p no:cacheprovider
python -m ruff check --no-cache src/schema/tool.py src/core/tool.py src/core/context.py tests/unit/tool/test_tool_executor.py tests/unit/core/test_context.py
python -m mypy src/schema/tool.py src/core/tool.py src/core/context.py
```

Expected: all exit 0, then commit:

```bash
git add src/schema/tool.py src/core/tool.py src/core/context.py tests/unit/tool/test_tool_executor.py tests/unit/core/test_context.py
git commit -m "feat: cancel ordered tool batches"
```

### Task 3: Harness Cancellation Lifecycle

**Files:**
- Modify: `src/core/loop.py`
- Modify: `src/core/task.py`
- Modify: `tests/unit/core/test_loop.py`
- Modify: `tests/unit/core/test_task.py`

**Interfaces:**
- Consumes: `CancellationToken` from Task 1 and `ToolExecutor.execute_tool_calls(..., should_cancel=...)` plus `ordered_tools` from Task 2.
- Produces: `Harness.run(task: Task, cancellation_token: CancellationToken | None = None) -> TaskResult`; direct `TASK_PAUSED -> TASK_FAILED`; internal cancellation handling with exact task output `"Task cancelled."`.

- [ ] **Step 1: Write failing state and pre-cancel tests**

Add tests named:

```python
def test_paused_task_can_transition_directly_to_cancelled_failure(): ...
def test_precancelled_created_task_does_no_gateway_work(): ...
def test_precancelled_paused_task_cancels_held_calls_in_order(): ...
```

For the paused case assert cancelled tool messages replace held messages, pending calls clear, failed tool events keep order, no `TASK_RESUMED` is emitted, repository snapshot reason is `cancel`, and no runner/gateway/finalizer is called.

- [ ] **Step 2: Run Step 1 tests and confirm RED**

Run the three exact tests. Expected: constructor/signature/state-transition failures and current approval guard behavior.

- [ ] **Step 3: Implement the dedicated cancellation terminal path**

Add paused→failed to `Task.TRANSITIONS`. Add private Harness helpers to check/raise cancellation, close held calls, and persist/emit the deterministic cancelled task result. In `run`, terminal tasks win; otherwise pre-cancellation is checked before the approval guard.

- [ ] **Step 4: Write failing gateway and retry race tests**

Add tests proving:

```python
def test_cancel_during_planning_return_wins_before_plan_is_saved(): ...
def test_cancel_during_gateway_error_wins_over_llm_error(): ...
def test_cancel_interrupts_retry_backoff_without_another_attempt(): ...
def test_cancel_during_inference_prevents_tool_execution_and_completion(): ...
```

Use deterministic fake gateways and token callbacks; patch no sleeps. Assert abandoned LLM attempts emit `LLM_FAILED` reason `cancel` and never emit their `LLM_COMPLETED`.

- [ ] **Step 5: Propagate cancellation through model attempts and steps**

Pass the exact token through planning, inference, step, turn, and tool execution. Check before every attempt and after returns/exceptions. Replace retry `time.sleep` with `token.wait`; an interrupted wait enters the dedicated cancel path.

- [ ] **Step 6: Write failing finalization and boundary tests**

Add:

```python
def test_cancel_during_limit_finalization_emits_only_cancel_failure(): ...
def test_repository_failure_while_cancelling_still_propagates(): ...
def test_event_failure_while_cancelling_still_propagates(): ...
def test_finished_task_ignores_a_cancelled_token(): ...
```

Assert timeout/max-step `TASK_FAILED` is not emitted before finalization resolves and persisted/event reasons never disagree.

- [ ] **Step 7: Implement consistent failure-event ordering**

Delay the normal limit-failure event until finalization and the post-call cancellation check finish. Let internal cancellation escape normal LLM/failure classification but never catch `_TaskPersistenceError` or `_EventSinkError` as cancellation.

- [ ] **Step 8: Verify and commit Task 3**

Run:

```bash
python -m pytest tests/unit/core/test_task.py tests/unit/core/test_loop.py tests/unit/tool/test_tool_executor.py -q -p no:cacheprovider
python -m ruff check --no-cache src/core/loop.py src/core/task.py tests/unit/core/test_loop.py tests/unit/core/test_task.py
python -m mypy src/core/loop.py src/core/task.py
```

Expected: all exit 0, then commit:

```bash
git add src/core/loop.py src/core/task.py tests/unit/core/test_loop.py tests/unit/core/test_task.py
git commit -m "feat: cancel harness runs cooperatively"
```

### Task 4: Agent Entry Points and Documentation

**Files:**
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-11-13-external-cancellation.md`

**Interfaces:**
- Consumes: `CancellationToken` and the new `Harness.run` signature.
- Produces: `Agent.chat(message: str, cancellation_token: CancellationToken | None = None) -> Task` and `Agent.resume(cancellation_token: CancellationToken | None = None) -> Task`.

- [ ] **Step 1: Write failing Agent propagation tests**

Add:

```python
def test_chat_passes_the_exact_falsey_cancellation_token(): ...
def test_resume_passes_the_exact_falsey_cancellation_token(): ...
def test_cancelled_agent_turn_is_not_added_to_transcript(): ...
```

Use a recording Harness double or patched `run`; object identity is required and truthiness must not replace the injected token.

- [ ] **Step 2: Run Agent tests and confirm RED**

Run: `python -m pytest tests/unit/demo/test_agent.py -q -p no:cacheprovider`

Expected: `chat`/`resume` reject the keyword or do not forward it.

- [ ] **Step 3: Implement Agent propagation and README contract**

Add optional keyword parameters and pass them unchanged to `Harness.run`. Replace the README TODO with a controller-thread example and the explicit cooperative/non-preemptive limitation.

- [ ] **Step 4: Run the full verification gate**

Run:

```bash
python -m pytest -q -p no:cacheprovider
python -m ruff format --check --no-cache src tests
python -m ruff check --no-cache src tests
python -m mypy src
git diff --check
```

Expected: all exit 0.

- [ ] **Step 5: Mark the plan complete and commit**

Change every completed checkbox in this plan to `[x]`, then:

```bash
git add README.md src/demo/agent.py tests/unit/demo/test_agent.py docs/superpowers/plans/2026-11-13-external-cancellation.md
git commit -m "docs: document cooperative cancellation"
```

The branch is ready for whole-branch code review only after this commit and a fresh full verification gate.
