# External Cancellation Design

## Intent

Add a production-usable external cancellation boundary to the synchronous Agent Harness. A caller
must be able to cancel one running or resumed task from another thread, observe a deterministic
cancelled terminal state, and avoid starting additional model or tool work after cancellation.

This closes the explicit README TODO and activates the existing `TaskFailReason.TASK_FAILED_CANCEL`
and `ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_CANCELLED` contracts.

## Goals

- Cancellation authority is explicit and scoped to one `Harness.run` invocation.
- Cancellation is thread-safe and idempotent.
- Retry backoff wakes immediately when cancellation is requested.
- The harness checks cancellation before and after blocking model calls, and between tool calls.
- Tool calls not started after cancellation become cancelled results so every model tool call still
  receives a matching tool result.
- A cancelled task is persisted as failed with reason `cancel`, emits the normal task-failed event,
  and does not make a finalizing model call.
- `Agent.chat` and `Agent.resume` expose the same per-run cancellation entry point.
- Existing callers that do not supply a token keep their current behavior.

## Non-goals

- Converting the harness, gateway, tools, or Agent API to async.
- Preempting Python code or killing a thread that is already inside a model/tool call.
- Defining process/container cancellation for a concrete sandbox runner.
- Cancelling durable work from another process or after a process restart.
- Adding multi-task scheduling or making one `Agent` instance concurrently safe.

## Anti-goals

- A shared mutable cancellation flag on `Harness` or `Agent`; it would leak cancellation across
  concurrent or later runs.
- Treating cancellation as an exception that is rewritten to `unknown` or `llm_error`.
- Calling the model again to summarize a task after the caller explicitly cancelled it.
- Dropping remaining calls from a model tool-call batch and producing an invalid transcript.

## Constraints

- The execution core is synchronous today.
- Model gateways and tool runners may block without accepting a cancellation primitive.
- Task state transitions and repository snapshots must remain valid at every persistence boundary.
- Event and repository failures stay boundary failures and must not be disguised as cancellation.

## Approaches Considered

### 1. Per-run `CancellationToken` backed by `threading.Event` — selected

`CancellationToken.cancel()` sets an event; `is_cancelled()` reads it; `wait(timeout)` allows retry
backoff to be interrupted. The token is passed to `Harness.run`, then propagated through the run
without being stored on the Harness instance.

This matches the synchronous architecture, is safe to signal from another thread, and provides a
clean migration path to an async cancellation adapter later. It is cooperative: an in-flight call
returns before the harness can observe cancellation.

### 2. `should_cancel: Callable[[], bool]`

This is small and easy to fake in tests, but cannot interrupt retry sleeps without inventing a
polling loop, has no standard ownership semantics, and encourages closures over shared flags.

### 3. Convert the execution path to async and use task cancellation

Async cancellation can propagate through cooperative async gateways and MCP tools, but converting
the synchronous public API now would combine cancellation, scheduling, gateway migration, and tool
execution into one high-risk change. It remains a later phase.

## Core API

Create `src/core/cancellation.py`:

```python
class CancellationToken:
    def cancel(self) -> None: ...
    def is_cancelled(self) -> bool: ...
    def wait(self, timeout: float) -> bool: ...
```

The class owns a private `threading.Event`. `wait` returns `True` when cancelled, matching Event
semantics. A token may be cancelled before `run`; cancellation remains idempotent.

Extend public entry points:

```python
Harness.run(task, cancellation_token: CancellationToken | None = None) -> TaskResult
Agent.chat(message, cancellation_token: CancellationToken | None = None) -> Task
Agent.resume(cancellation_token: CancellationToken | None = None) -> Task
```

When omitted, `Harness.run` creates a fresh token for that invocation. The exact token is propagated
downward; it is never retained for later runs.

## Cancellation Checkpoints

The harness checks the token:

1. after the finished-task fast path and before starting new task work;
2. before each planning/inference attempt;
3. after each planning/inference call returns, before accepting output or emitting completion;
4. during retry backoff by calling `token.wait(delay)` instead of `time.sleep(delay)`;
5. before each tool call in a batch;
6. after a tool batch and before the next step, completion, pause, or limit handling;
7. before resuming held tool calls.

Cancellation requested while a blocking gateway or tool is running cannot preempt that call. When
the call returns, its return value is not used to start further work; the task follows cancellation
handling at the next checkpoint.

Planning, normal inference, and finalization inference all follow the same rule. After either a
normal gateway return or a non-boundary gateway exception, cancellation is checked before output
classification, retry, or `LlmCallError`; cancellation therefore wins until a terminal task snapshot
has been saved. The harness emits `LLM_FAILED` with reason `cancel` for an LLM attempt abandoned by
cancellation. Repository and event-sink boundary exceptions continue to propagate instead.

## Tool Batch Semantics

`ToolExecutor.execute_tool_calls` accepts an optional zero-argument cancellation predicate. Before
each call it checks the predicate. Once cancelled, the executor does not invoke policy, validator,
runner, or sandbox for that and all remaining calls. It creates a failed result for each remaining
call with reason `cancelled` and a stable cancellation message, invokes `after_call` so observers see
the result, and preserves call order.

Already completed calls remain completed. There is no attempt to roll back side effects.

`ToolCallResult` gains `ordered_tools: list[ToolExecutionResult]`, populated in original model call
order. Existing `pending_tools`, `completed_tools`, and `failed_tools` remain for compatibility.
Transcript construction uses `ordered_tools` when present and falls back to the legacy category
concatenation for restored/third-party results that omit it. This also fixes the pre-existing ordering
loss for mixed completed/failed/pending batches.

Held calls are checked by the Harness before `_finish_pending_tool_calls`; cancellation wins over
approval and no held tool starts. For a paused task, cancellation is checked before the
"waiting for approval" guard. Every pending held call becomes a cancelled tool result in its original
order, its held/pending history message is replaced with the matching cancelled tool message, a
`TOOL_EXECUTION_FAILED` event is emitted, and `pending_tool_calls` is cleared. The state machine allows
`TASK_PAUSED -> TASK_FAILED`; no `TASK_RESUMED` event is emitted for this direct cancellation path.

## Task and Event Semantics

Cancellation has a dedicated handler rather than using normal timeout/max-step finalization:

- update elapsed time without incrementing a step that never began;
- set `TaskResult(output="Task cancelled.")`;
- transition to `TASK_FAILED` with `TASK_FAILED_CANCEL`;
- save the task;
- emit one `TASK_FAILED` event with reason `cancel` and cancellation content;
- return the result without calling `finalize`.

If cancellation arrives after a task is already completed/failed, the existing terminal result wins.
If repository or event delivery fails while recording cancellation, the original boundary exception
continues to propagate.

Timeout/max-step failure does not emit its `TASK_FAILED` event until finalization finishes and the
harness confirms cancellation was not requested. If cancellation arrives during finalization, the
task instead follows the cancellation handler and emits only the `cancel` task-failed event. This
keeps event reason and persisted terminal reason consistent.

## Concurrency and Ownership

`CancellationToken` is the only new thread-safe object. Task mutation remains owned by the thread
running `Harness.run`; the cancelling thread only sets the token event. The design does not make
`Task`, `Harness`, `Agent`, repositories, or event sinks generally thread-safe.

One token should control one run. Reusing a cancelled token intentionally cancels a later run
immediately; the framework does not reset tokens.

## Testing

- Token starts active, cancels idempotently, and wakes a waiting thread.
- Pre-cancelled create and resume paths perform no gateway/tool work and persist `cancel`.
- Pre-cancelled paused tasks cancel every held call in order, repair transcript messages, clear
  pending calls, and do not emit `TASK_RESUMED`.
- Cancellation during planning/inference is observed after the blocking call returns and prevents
  later work.
- Cancellation requested while a gateway later raises an error wins over retry/LLM-error handling;
  repository and event failures still propagate.
- Cancellation interrupts transient-error retry backoff without another attempt.
- Cancellation between tool calls preserves completed results and marks all remaining calls
  cancelled without invoking their runners.
- Mixed tool results and their transcript messages retain original model call order.
- Cancelled tasks emit/persist the exact terminal state and never call finalization.
- Cancellation during timeout/max-step finalization produces one consistent cancel failure event.
- Agent passes the exact falsey token object through `chat` and `resume`.
- Existing timeout, approval, retry, event, persistence, and default Agent tests remain green.

## Documentation

README will show creating a token, passing it to `Agent.chat`/`resume`, and cancelling it from a
controller thread. It will state clearly that cancellation is cooperative and does not forcibly
interrupt an in-flight third-party call.
