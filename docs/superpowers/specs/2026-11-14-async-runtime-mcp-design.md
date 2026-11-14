# Async-first Runtime and MCP Integration Design

## Intent

The harness already depends on FastMCP and contains an asynchronous `MCPManager`, but that manager is
not connected to `Agent`, `ToolExecutor`, or the model-visible tool catalog. Its tool executors are
coroutines, while the current runner treats every callable as synchronous, so an MCP call would be
reported as a completed coroutine object instead of being awaited.

This phase makes the execution path async-first and turns MCP into a supported composition capability.
Existing synchronous entry points remain as compatibility adapters for command-line and simple Python
callers. The async API is the authoritative implementation and must support multiple independent runs
on one event loop without blocking it on synchronous LLM or local-tool adapters.

## Goals

- Add native `Harness.arun()` and `Agent.achat()` / `Agent.aresume()` entry points.
- Keep `Harness.run()` and `Agent.chat()` / `Agent.resume()` as thin synchronous adapters.
- Support both native async and legacy sync LLM gateways, streams, tool runners, and tool callables.
- Use LiteLLM's asynchronous completion API in the default gateway.
- Execute synchronous compatibility collaborators in worker threads so they do not block the event loop.
- Integrate connected MCP tools into the same catalog and approval/paused-task flow as local tools.
- Give every MCP server explicit connect, refresh, disconnect, and close lifecycle semantics.
- Avoid provider-invalid MCP function names and reject tool-name collisions deterministically.
- Require approval for MCP tools by default; connection is not authorization to execute arbitrary calls.
- Preserve current event ordering, persisted task states, ordered tool result messages, workspace rules,
  and cooperative external cancellation semantics.

## Non-goals

- No package/distribution compatibility work.
- No durable database repository, cross-process scheduler, or Session/TaskRun ownership model.
- No parallel execution of tool calls within one model response. Approval order and deterministic result
  ordering remain sequential in this phase.
- No forced termination of blocking synchronous work already running in a worker thread.
- No MCP resources, prompts, sampling, elicitation, roots, or OAuth implementation.
- No claim that a remote MCP server is sandboxed. Workspace and sandbox rules continue to apply only to
  explicitly bound local capabilities.
- No conversion of context building, event emission, task persistence, hooks, guards, or approval policy
  into async ports. Implementations supplied for those ports must remain short and non-blocking.

## Constraints and Invariants

- Python remains `>=3.11`; FastMCP remains the MCP transport dependency.
- There is one authoritative async implementation. Sync APIs must not duplicate the harness loop.
- A sync adapter called from a thread that already owns a running event loop fails with a clear message
  directing the caller to the async API; it must not nest an event loop or start a hidden helper thread.
- A legacy synchronous LLM call or tool runner is invoked with `asyncio.to_thread`.
- Native awaitables are awaited directly. Sync iterators are advanced in a worker thread; async iterators
  are consumed directly.
- One `CancellationToken` belongs to one run. Cancellation remains cooperative and run-local.
- Cancelling the caller's `asyncio.Task` performs terminal task cleanup, then re-raises
  `asyncio.CancelledError`; it is not converted into an ordinary successful return.
- Tool-call result order always matches the model's call order.
- A server name is unique within one manager. A model-visible MCP tool name is stable for the duration of
  that connection and unique within the complete catalog.
- Removing or closing a server unregisters all tools originating from it, even when server shutdown raises.
- A failed connection or failed tool refresh must not leave a partially registered server.

## Approaches Considered

### 1. Run the existing synchronous harness in `asyncio.to_thread`

This would expose an `arun()` method quickly, but MCP calls still need an event loop and native async LLM
streaming remains hidden behind a worker. Cancellation and backpressure would be weak, and concurrency
would consume one thread per run. It is an adapter, not an async-first architecture.

### 2. Replace every port with an async-only protocol

This is conceptually clean, but it would break all current gateways, tests, local tools, repositories,
event sinks, hooks, and user integrations at once. Many of those operations are intentionally small and
do not benefit from async I/O. The migration cost is not justified for version `0.1.0`.

### 3. Async orchestration with explicit sync compatibility adapters — selected

The Harness control flow, LLM calls, streamed output consumption, tool execution, and Agent/MCP lifecycle
become async. Compatibility helpers detect native async collaborators and otherwise offload synchronous
calls or iterator advancement to worker threads. Sync public methods call the async implementation only
when no event loop is running. This produces a real async runtime without an unnecessary all-ports rewrite.

## Async Runtime Surface

`Harness` gains async counterparts for every method that crosses an I/O boundary:

- `aplanning`, `ainfer`, `aexecute_tool_calls`, `astep`, `afinalize`, `aturn`, and `arun`;
- async pending-tool completion and retry waiting;
- private async helpers for invoking sync-or-async callables and consuming sync-or-async streams.

The existing public synchronous methods remain for compatibility and delegate to their async counterparts
through one shared `run_sync()` helper. Internal async code never calls a synchronous wrapper.

`LlmGateway` describes sync-or-async results rather than forcing one concrete implementation shape.
`LiteLLMGateway` implements native async `aplanning` and `ainfer` methods using
`litellm.acompletion`. Its existing sync methods remain available for direct legacy use. Harness prefers
the `a*` method when present and otherwise offloads the legacy method.

Streaming accepts either `Iterator[LLMOutput]` or `AsyncIterator[LLMOutput]`. Assembly retains the current
field-merging behavior and emits deltas in the same order. A sync iterator's `next()` runs in a worker so
a blocking provider adapter cannot stall unrelated tasks on the event loop.

## Tool Execution and Concurrency

`ToolExecutor` gains `aexecute`, `aexecute_approved`, `aexecute_tool_call`,
`aexecute_held_tool_call`, and `aexecute_tool_calls`. The async methods are authoritative. Native async
tool callables and runners are awaited directly; synchronous runners are offloaded with
`asyncio.to_thread`.

The batch remains sequential. Once one call pauses for approval, later calls are held exactly as today;
running calls concurrently would make that rule timing-dependent and could allow later side effects before
an earlier approval decision. Independent `Harness.arun()` invocations can run concurrently on the same
event loop, provided their tasks, cancellation tokens, and mutable collaborators are not shared.

The current synchronous ToolExecutor methods delegate to the async implementation when no event loop is
running. Legacy subclasses that override only synchronous batch execution remain supported by Harness via
signature inspection and worker-thread delegation; this compatibility path is documented as transitional.

## Cancellation Semantics

Retry delays use an async token wait rather than blocking the event loop. The token exposes an
`await_cancelled(timeout)` operation implemented without polling. Cancelling the token wakes that wait.

External token cancellation preserves the existing contract: after an in-flight third-party operation
returns, its result is discarded and the task is saved as `failed/cancel`. Native coroutine cancellation is
different: when the caller cancels the `arun()` task, Harness shields only the short terminal-state save and
event emission, records `failed/cancel`, then re-raises `asyncio.CancelledError`. This prevents callers from
mistaking cancellation for a normal result while still keeping repository state consistent.

Worker-thread calls cannot be force-stopped. If their awaiting coroutine is cancelled, the underlying thread
may finish in the background, but no result is committed and no subsequent harness step runs.

## MCP Composition

`MCPManager` becomes an async context manager and owns:

- the FastMCP client and exit stack for each named server;
- the set of catalog tool ids registered for that server;
- atomic connect/list/register and unregister/disconnect operations;
- idempotent `aclose()` that attempts to close every server and aggregates cleanup errors.

The manager receives a `ToolCatalog` rather than returning an unowned list that callers must synchronize.
`add_server()` connects, lists tools, converts all definitions, checks collisions, then registers them. Any
failure closes the candidate client and leaves the catalog unchanged. `refresh_server()` computes a complete
replacement set and updates it only after validation succeeds. `remove_server()` unregisters tools first and
then closes the client so a failed transport shutdown cannot leave callable stale tools advertised.

`Agent` accepts an optional manager or creates one bound to its catalog. It exposes
`add_mcp_server`, `refresh_mcp_server`, `remove_mcp_server`, `aclose`, and async-context-manager methods.
MCP connection methods are async-only because hiding persistent transport lifecycle behind repeated event-loop
creation is unsafe. `Agent.achat()` can then use registered MCP tools normally.

Agent composition has exactly one authoritative catalog. When `tool_catalog` is supplied it is authoritative;
otherwise an injected executor's `catalog` is reused; otherwise Agent creates a registry. Supplying both a
catalog and an executor bound to a different catalog raises `ValueError`. An injected MCP manager must expose
and use the same catalog or construction fails. This prevents a tool from being model-visible but unavailable
to execution, or executable without being advertised.

Lifecycle mutations are serialized by an async lock. Each server also tracks active calls and a closing flag.
Remove and close mark the server closing, unregister its tools immediately, reject new calls, wait for calls
that already acquired a lease to finish, and only then close the transport. Refresh is serialized with other
lifecycle mutations but does not interrupt calls already using the unchanged client. This gives catalog
atomicity without terminating a remote side effect whose execution already began.

## MCP Tool Identity and Approval

Provider-facing function names use `mcp__<server>__<tool>`, normalized to ASCII letters, digits,
underscores, and hyphens and capped to the provider-safe length. The original server/tool names are retained
as runtime metadata and in the description. Normalization collisions fail the whole connection rather than
silently renaming a tool after the model has seen it.

`Tool` gains runtime origin metadata and `requires_approval`. The default approval policy returns a pending
decision for every MCP tool regardless of its advertised annotations. A caller may inject a more permissive
policy after evaluating server identity and MCP annotations. MCP tools never receive a local `Workspace` and
never route through `SandboxRunner`; remote execution is a separate trust boundary.

MCP results are normalized into one stable JSON-safe mapping before entering task history:

```python
{
    "content": [/* MCP content blocks serialized as JSON objects */],
    "structured_content": /* JSON value or null */,
    "is_error": False,
}
```

All three keys are present. Opaque FastMCP result objects and its convenience `data` view are not persisted.
An MCP result with `is_error=True` uses the same mapping but becomes a failed `ToolExecutionResult` with a new
`remote_error` fail reason. Transport exceptions also become `remote_error` failures with a JSON-safe error
mapping; they are not reported as successful tool output. The core tool layer defines a small
`RemoteToolError` carrying that output so the executor can classify remote failures without importing the MCP
adapter or depending on FastMCP types.

## Error and Lifecycle Semantics

- Calling a sync adapter inside an active event loop raises `RuntimeError` before starting work.
- Duplicate server names, invalid names, provider-name collisions, and catalog collisions fail before
  registration.
- Connection/list failure leaves neither a server entry nor catalog entries.
- Disconnect removes advertised tools even if transport close fails; the close error is still surfaced.
- Calling a stale MCP `Tool` after removal fails clearly and does not reconnect implicitly.
- `Agent.aclose()` is idempotent. Chat after close and new server connections after close fail clearly.
- Harness boundary errors from persistence and event sinks keep their current propagation behavior.
- If caller-task cancellation cleanup cannot save the terminal state or emit its event, that boundary error
  wins and `CancelledError` remains its exception context. `CancelledError` is re-raised only after cleanup
  succeeds; the harness never claims durable cancellation when its boundary rejected the update.

## Testing Strategy

Tests are written before implementation and cover:

- `Harness.arun()` completes with a native async gateway and does not block an unrelated coroutine;
- legacy sync gateways and sync streams execute off the event loop and preserve event ordering;
- async and sync tool callables produce identical persisted results;
- sync wrappers reject use from an active event loop and work normally outside one;
- independent async runs do not share cancellation state;
- external token cancellation wakes async retry waiting and preserves `failed/cancel` state;
- cancellation of the caller task persists cancellation and re-raises `CancelledError`;
- MCP connect atomically registers provider-safe tools, and calls are awaited rather than persisted as
  coroutine objects;
- MCP tools pause for approval by default and resume through the normal task flow;
- duplicate normalized names and existing catalog collisions leave the catalog unchanged;
- refresh replaces one server's tools atomically;
- remove and close unregister tools, including when client shutdown fails;
- `Agent` async context management closes MCP connections;
- existing workspace, sandbox, task-state, event-order, and sync API regression suites remain green.

Full gates remain pytest, Ruff formatting, Ruff lint, and Mypy over `src`.

## Follow-up

After this phase, the next priority is Session/TaskRun concurrency isolation: remove Agent's single
`pending_task` ownership, add per-run handles and optimistic task ownership, and make concurrent conversations
explicit. Durable repositories, memory/skills, observability, and a concrete sandbox runner follow that work.
