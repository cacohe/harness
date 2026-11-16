# Async-first Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Make Harness, LiteLLM, tool execution, and Agent async-first while preserving safe synchronous entry points.

**Architecture:** Async methods own the control flow. Shared compatibility helpers await native async collaborators and offload legacy synchronous calls and iterator advancement to worker threads; sync public methods invoke the async implementation only when no event loop is active.

**Tech Stack:** Python 3.11, asyncio, LiteLLM, Pydantic, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-14-async-runtime-mcp-design.md`

## Global Constraints

- Python remains `>=3.11`; do not add dependencies.
- Preserve event order, task snapshots, approval order, workspace/sandbox enforcement, and external-token cancellation behavior.
- Keep one authoritative async implementation; sync wrappers must not duplicate the run loop.
- Do not parallelize tool calls within one model response.
- Context builders, event sinks, repositories, hooks, guards, and approval policies remain synchronous and must be short/non-blocking.
- No packaging/distribution work.

## Review Focus

- A synchronous gateway returning a lazy blocking iterator must advance that iterator off the event loop; Task 2 adds a ticker regression test.
- A sync adapter invoked inside an active event loop must fail before starting work and must close no leaked coroutine; Tasks 1, 3, and 4 test this.
- Caller-task cancellation during a worker-thread operation must not commit its late result; Task 4 tests persisted terminal state and re-raised cancellation.
- Persistence or event failure during caller-cancellation cleanup must win over `CancelledError`; Task 4 adds both boundary tests.
- A falsey async collaborator or cancellation token must be preserved by identity; Tasks 2–5 retain the existing falsey regression pattern.

---

### Task 1: Async compatibility and cancellation primitives

**Files:**
- Create: `src/core/async_utils.py`
- Modify: `src/core/cancellation.py`
- Create: `tests/unit/core/test_async_utils.py`
- Modify: `tests/unit/core/test_cancellation.py`

**Interfaces:**
- Produces: `run_sync(awaitable_factory: Callable[[], Awaitable[T]], *, async_name: str) -> T`
- Produces: `call_maybe_async(async_callable, sync_callable, /, *args, **kwargs) -> Awaitable[Any]`
- Produces: `iterate_maybe_async(value: AsyncIterator[T] | Iterator[T]) -> AsyncIterator[T]`
- Produces: `CancellationToken.await_cancelled(timeout: float | None = None) -> Awaitable[bool]`

- [x] **Step 1: Write failing async utility and token tests**

Add tests proving `run_sync` returns outside an event loop, raises `RuntimeError` naming the async API inside one, `call_maybe_async` prefers the native async callable, sync work runs on another thread, sync iteration does not block a ticker, and `await_cancelled` wakes after `cancel()` or returns `False` on timeout.

- [x] **Step 2: Run the focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/core/test_async_utils.py tests/unit/core/test_cancellation.py -q -p no:cacheprovider`
Expected: FAIL because the new interfaces do not exist.

- [x] **Step 3: Implement the primitives**

Use `asyncio.run` only after checking `asyncio.get_running_loop()`. Offload legacy calls and `next()` through `asyncio.to_thread`. Implement token async waiting with event-loop-safe notification rather than polling or a permanently blocked worker.

- [x] **Step 4: Run focused tests**

Run the Step 2 command.
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/core/async_utils.py src/core/cancellation.py tests/unit/core/test_async_utils.py tests/unit/core/test_cancellation.py
git commit -m "feat: add async compatibility primitives"
```

### Task 2: Native async LiteLLM gateway and stream assembly

**Files:**
- Modify: `src/core/llm.py`
- Modify: `src/llm/litellm.py`
- Modify: `src/schema/llm.py`
- Modify: `tests/unit/llm/test_llm.py`
- Modify: `tests/unit/schema/test_llm_stream.py`
- Modify: `tests/unit/core/test_ports.py`

**Interfaces:**
- Consumes: `iterate_maybe_async(...)` from Task 1.
- Produces: `LlmGateway` support for sync or async planning/inference results.
- Produces: `LiteLLMGateway.aplanning(...) -> Awaitable[LLMOutput | AsyncIterator[LLMOutput]]`
- Produces: `LiteLLMGateway.ainfer(...) -> Awaitable[LLMOutput | AsyncIterator[LLMOutput]]`
- Produces: `assemble_llm_async_stream(...) -> Awaitable[tuple[LLMOutput, list[str]]]`

- [x] **Step 1: Write failing native async gateway tests**

Patch `litellm.acompletion`, assert `aplanning`/`ainfer` await it, preserve structured output, tool calls, usage, and streamed deltas, and never call synchronous `completion`.

- [x] **Step 2: Write failing sync/async stream parity tests**

Feed identical sync and async chunks to the assemblers and assert equal `LLMOutput` and text parts, including split tool-call arguments and usage-only chunks.

- [x] **Step 3: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/llm/test_llm.py tests/unit/schema/test_llm_stream.py tests/unit/core/test_ports.py -q -p no:cacheprovider`
Expected: FAIL on missing async interfaces.

- [x] **Step 4: Implement async gateway methods and assembly**

Share request/output conversion with existing sync methods. Use `litellm.acompletion`; return an async generator for streaming and a complete `LLMOutput` otherwise. Keep direct legacy sync methods unchanged.

- [x] **Step 5: Run focused tests**

Run the Step 3 command.
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/core/llm.py src/llm/litellm.py src/schema/llm.py tests/unit/llm/test_llm.py tests/unit/schema/test_llm_stream.py tests/unit/core/test_ports.py
git commit -m "feat: add native async llm gateway"
```

### Task 3: Async-first tool execution

**Files:**
- Modify: `src/core/tool.py`
- Modify: `src/tool/tool_executor.py`
- Modify: `tests/unit/tool/test_tool_executor.py`
- Modify: `tests/unit/core/test_ports.py`

**Interfaces:**
- Consumes: `run_sync` and `call_maybe_async` from Task 1.
- Produces: `ToolExecutor.aexecute`, `aexecute_approved`, `aexecute_tool_call`, `aexecute_held_tool_call`, and `aexecute_tool_calls`.
- Preserves: synchronous method names as adapters with the same arguments and result types.

- [x] **Step 1: Write failing async runner/callable tests**

Cover native async runner success/error, legacy sync runner offloading without blocking a ticker, async tool callable through `LocalToolRunner`, ordered mixed results, approval hold/resume, and cancellation between calls.

- [x] **Step 2: Write failing sync-wrapper loop-safety tests**

Assert legacy sync APIs still work outside a loop and raise before execution inside a running loop.

- [x] **Step 3: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/tool/test_tool_executor.py tests/unit/core/test_ports.py -q -p no:cacheprovider`
Expected: FAIL on missing async executor methods.

- [x] **Step 4: Implement authoritative async methods**

Keep validation and approval synchronous. Await native runner/callable methods, offload legacy sync work, classify exceptions exactly as before, and process batches sequentially. Make sync methods delegate through `run_sync`.

- [x] **Step 5: Run focused tests**

Run the Step 3 command.
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/core/tool.py src/tool/tool_executor.py tests/unit/tool/test_tool_executor.py tests/unit/core/test_ports.py
git commit -m "feat: make tool execution async first"
```

### Task 4: Async-first Harness control flow

**Files:**
- Modify: `src/core/loop.py`
- Modify: `tests/unit/core/test_loop.py`

**Interfaces:**
- Consumes: Tasks 1–3 interfaces.
- Produces: `Harness.aplanning`, `ainfer`, `aexecute_tool_calls`, `astep`, `afinalize`, `aturn`, and `arun`.
- Preserves: `planning`, `infer`, `execute_tool_calls`, `step`, `finalize`, `turn`, and `run` as sync adapters.

- [x] **Step 1: Add failing async run parity tests**

Port representative completion, tool use, pause/resume, retry, timeout/finalization, event-order, and external cancellation scenarios to `arun`; assert snapshots match the synchronous contract.

- [x] **Step 2: Add failing concurrency and loop-safety tests**

Run two independent tasks with a native async gateway and prove one wait does not block the other. Assert `run()` inside an event loop raises before emitting or saving anything.

- [x] **Step 3: Add failing caller-cancellation boundary tests**

Cancel an `arun` task during model/tool await, assert terminal `failed/cancel` persistence and re-raised `CancelledError`; assert repository and event failures during cleanup replace it with the boundary error.

- [x] **Step 4: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/core/test_loop.py -q -p no:cacheprovider`
Expected: FAIL on missing async Harness methods.

- [x] **Step 5: Convert the control flow to async**

Async methods own all I/O paths and call only other async methods. Prefer gateway `a*` methods; offload legacy gateway methods and sync streams. Use `await_cancelled` for retry waits. Preserve current exception classification, callback race checks, state transitions, and event order.

- [x] **Step 6: Add sync adapters and legacy executor compatibility**

Delegate through `run_sync`. When an injected legacy executor overrides only synchronous `execute_tool_calls`, invoke it in a worker and retain current signature inspection for `should_cancel`.

- [x] **Step 7: Run focused and regression tests**

Run: `.venv\Scripts\python -m pytest tests/unit/core/test_loop.py tests/unit/tool/test_tool_executor.py -q -p no:cacheprovider`
Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add src/core/loop.py tests/unit/core/test_loop.py
git commit -m "feat: make harness runtime async first"
```

### Task 5: Async Agent surface and documentation

**Files:**
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `Harness.arun` and `run_sync`.
- Produces: `Agent.achat(message, cancellation_token=None) -> Awaitable[Task]`
- Produces: `Agent.aresume(cancellation_token=None) -> Awaitable[Task]`
- Preserves: `chat` and `resume` as sync adapters.

- [x] **Step 1: Write failing async Agent tests**

Cover completed chat memory, paused chat, resumed chat, failed-turn non-memory, token identity propagation, concurrent independent Agent instances, and sync-wrapper rejection inside a loop.

- [x] **Step 2: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/demo/test_agent.py -q -p no:cacheprovider`
Expected: FAIL on missing async methods.

- [x] **Step 3: Implement the async surface and thin sync adapters**

Move chat/resume behavior into async methods and delegate sync methods through `run_sync`; preserve pending-task and transcript behavior exactly.

- [x] **Step 4: Document async usage and compatibility limits**

Add `await agent.achat(...)`, sync wrapper guidance, active-loop error behavior, independent-run concurrency, and the non-preemptive worker-thread cancellation limitation.

- [x] **Step 5: Run all quality gates**

Run:

```powershell
.venv\Scripts\python -m pytest -q -p no:cacheprovider
.venv\Scripts\ruff format --check src tests
.venv\Scripts\ruff check --no-cache src tests
.venv\Scripts\mypy src
```

Expected: all tests pass; Ruff and Mypy report no errors.

- [x] **Step 6: Commit**

```bash
git add src/demo/agent.py tests/unit/demo/test_agent.py README.md
git commit -m "feat: expose async agent runtime"
```

完成状态：已通过 PR #7 合入 `main`，异步运行时与同步兼容入口均已由测试覆盖。
