# Task 12 Report — Production Adapters

## Scope and architecture

Task 12 moved production model, tool, MCP, Artifact-read, and observability capabilities into
`src/adapters/`. Core was not given provider, Docker, MCP, SQLite-execution, or metrics concerns.
The application layer gained only a bounded post-commit `Observation` port and explicit optional
composition injection; its default remains disabled and produces no stdout/log content.

The implementation intentionally does not preserve the legacy `Agent`, MCP manager, sandbox, or
observability APIs. Task 14 remains responsible for deleting those directories.

## RED / GREEN evidence by group

### Model

- RED: the plan's `tests/unit/adapters/model/test_litellm.py` and
  `src/adapters/model/litellm.py` did not exist; initial collection reported file-not-found/import
  absence.
- GREEN: native async completion, all role/tool-result wire conversion, ordered tool batches,
  usage conversion, sanitized provider failure, timeout/caller-cancellation distinction, bounded
  streaming, and duplicate/missing/out-of-order fragment rejection are covered.
- Secrets are stored privately and omitted from `repr`; provider objects and exception text are
  never returned through Core values.

### Durable idempotency, Local, and ReadArtifact

- RED: there was no process-external tool execution claim store and no Core `ToolGateway`
  production adapter.
- GREEN: `SQLiteToolExecutionStore` uses an independent strict schema (`user_version=1`) and WAL.
  Its primary identity is `(session_id, run_id, tool_call_id)` and every row binds the complete
  `ToolRequest.idempotency_key`. Claim precedes the side effect. Completed results use canonical
  JSON encoding and survive reopen. Binding changes, corrupt results, extra/unknown schemas, and
  completion divergence are rejected. Startup converts an abandoned `started` row into the same
  stable failed result and never replays it. `InMemoryToolExecutionStore` is explicitly documented
  as development-only.
- Concurrent retries within a gateway share a keyed lock; SQLite claims coordinate independent
  gateway instances. Live approval can tighten but never erase the persisted gate. Scope and
  workspace are revalidated before claim.
- Local callables support native async and sync `asyncio.to_thread`, bounded timeout, caller
  cancellation, stable sanitized failure, strict JSON-object/schema checks, and policy-driven
  Artifact externalization.
- `ReadArtifactTool` accepts exactly `artifact_id/offset/limit`, rejects bool/negative/overflow and
  excess limit, rebuilds untrusted refs, streams skip/read with only `limit + 1` bytes retained,
  uses strict UTF-8 or base64, and returns only safe ref/range metadata without repository paths.

### Docker

- RED: only the legacy schema-coupled sandbox existed.
- GREEN: the new adapter requires an explicit absolute native CLI, local socket, digest-pinned
  image, absolute contained workspace, and non-root numeric user. Generated argv includes
  `--network none`, read-only root, `--cap-drop ALL`, `no-new-privileges`, PID/memory/CPU/nofile
  limits, bounded tmpfs, and explicit mount/workdir. It uses `create_subprocess_exec` (no shell
  expansion), drains both streams with hard bounds, redacts workspace paths, kills on timeout,
  cancellation, or reader failure, and shield-removes the named container with confirmation.

### MCP

- RED: only the legacy catalog/manager API existed.
- GREEN: add/refresh/remove publish complete catalogs atomically. The server generation is included
  in both catalog entries and the frozen workspace binding. Every call acquires a generation lease;
  refresh/remove/close wait for active calls, stale generation calls are rejected, refresh failure
  restores the previous published state, add failure closes without publication, and owned cleanup
  is shielded and exactly once even under caller cancellation.

### Observability and runtime

- RED: Task 11 runtime had no application post-commit observer port.
- GREEN: runtime emits metadata-only observations after successful submission acceptance and after
  each successful coordinator advance. A runtime test reads the journal head from inside the
  observer and proves its version already equals the observation. Observer failure cannot change
  the durable fact or stop later work.
- `CompositeObserver` preserves order and isolates failures. `MetricsObserver` is async-safe and
  caps series cardinality with an overflow counter. `EventObserver` requires an explicit injected
  sink. Default composition has `observer=None`; there is no stdout emitter.

## Protocol and schema decisions

- Tool execution persistence is deliberately separate from `SQLiteRunJournal` v3.
- The execution result codec stores only `ToolResultStatus`, stable failure code, execution key,
  canonical inline JSON, safe `ArtifactRef` descriptors, and summary. It never uses pickle.
- MCP generation is encoded as an opaque `workspace-generation` identity so the existing Core
  approval binding carries it durably without adding provider concepts to Core.
- `DockerInvocation` is a capability value returned by a registered adapter-side handler; arbitrary
  imports/eval are not supported.
- Observations contain only kind, Session/Run IDs, status, journal version, and sequence. Prompt,
  tool arguments/results, and Artifact bodies are not part of the port.

## Verification commands and results

Environment: locked CPython 3.11 environment, `LITELLM_LOCAL_MODEL_COST_MAP=True`, pytest cache
disabled, and short absolute basetemp paths under `F:/CodeSpace/.tmp`.

```text
python -m pytest -q -p no:cacheprovider \
  tests/unit/adapters/model tests/unit/adapters/tools \
  tests/unit/adapters/observability tests/unit/application/test_runtime_v3.py
54 passed in 15.81s
```

After the focused run, the attack suites were expanded further; their final counts are included in
the non-legacy and full-suite results below.

```text
ruff check --no-cache <Task 12 source and tests>
All checks passed!

mypy --cache-dir F:/CodeSpace/.tmp/mypy-task12 \
  src/adapters/model src/adapters/tools src/adapters/observability \
  src/application/observability.py src/application/runtime.py src/application/composition.py
Success: no issues found in 15 source files

python -m pytest -q -p no:cacheprovider tests \
  --ignore=tests/unit/application/test_agent.py \
  --ignore=tests/integration/test_mcp_agent.py \
  --ignore=tests/load/test_shared_agent_load.py \
  --ignore=tests/unit/core/test_ports.py
1738 passed, 7 skipped in 243.52s

python -m pytest -q -p no:cacheprovider tests
58 failed, 1743 passed, 7 skipped in 247.76s
```

The 58 full-suite failures are exactly the pre-existing legacy groups approved for Task 14:

- `tests/unit/application/test_agent.py`: 54
- `tests/integration/test_mcp_agent.py`: 2
- `tests/load/test_shared_agent_load.py`: 1
- `tests/unit/core/test_ports.py`: 1

No non-legacy regression was introduced.

## Fix round 1/5 — production ownership and lifecycle review

Baseline: `b15d58a`. Two Critical and seven Important findings were reproduced with attacking
tests. The two reviewer Minor findings remain explicitly deferred and were not mixed into this
round.

### RED evidence

- A second `SQLiteToolExecutionStore` constructor could settle a live owner's `started` row, and a
  contender's old five-second wait timeout could overwrite the live result with `FAILED`.
- Docker cleanup ran only after an acknowledged create; a create timeout/cancellation could leave an
  ambiguously created container. CLI `communicate()` could also buffer without the adapter bound.
- SQLite accepted weakly similar schemas/results and did not require the stored execution key to
  equal the claim binding before the first write.
- MCP cancellation while waiting for an active lease could leave `closing=True`; cancellation of the
  first `aclose()` caller could make later close calls return without finishing owned cleanup.
- Artifact reads trusted declared length/iterator chunks, leaked a partial UTF-8 decode exception,
  and had no chunk-size corruption bound.
- LiteLLM rejected valid parallel fragment continuation (`[0,1] -> [0,1]`), accepted missing finish,
  and did not guarantee explicit async-provider iterator close.
- Replay acceptance produced the same observation as a new commit, and a hung observer had no hard
  application bound. A synchronous event sink ran on the event loop.
- A mutating synchronous local handler could continue in its worker thread after timeout/cancel;
  stale live binding returned a generic failure rather than denial.

### GREEN decisions and coverage

- Durable execution now has explicit `OWNER`, `WAIT`, and `COMPLETED` claims. Constructors never
  inspect or settle `started`; contenders only poll or cancel and never mutate owner state. SQLite
  busy timeout, poll interval, and optional caller wait bound are separate. `recover_abandoned()` is
  an explicit exclusive startup operation whose precondition is that the caller has excluded all
  live owners. Tests cover three independent instances, execution longer than the SQLite timeout,
  wait expiry with zero mutation, owner cancellation, exact shared result, and explicit recovery.
- The SQLite store atomically creates and fingerprints two strict tables, checks exact SQL, columns,
  PK/not-null layout, metadata/version, and rejects extra/future objects. Result writes rebuild
  `ToolResult`, require the exact request execution key before mutation, use bounded canonical JSON,
  and reject duplicate/unknown/noncanonical/corrupt data.
- Docker chooses the random name before create and unconditionally runs one shielded confirmed
  `rm --force` task from `finally`, including create timeout/cancel. Repeated cancellation cannot
  interrupt owned cleanup. Create/start/remove all use concurrently drained bounded streams; reader
  failure kills the process. Image leading dashes, noncanonical/nonlocal hosts, and invalid cleanup
  bounds are rejected; not-found removal is treated as confirmed idempotence.
- MCP refresh/remove rollback `closing` in `finally` before publication. Server close tasks and the
  aggregate `aclose` task persist independently of any caller; a canceled caller can return, while a
  later `aclose` awaits the same exactly-once cleanup. Tests cover active leases, refresh/remove
  cancellation, stale generation, list/call/close failures, and later-close completion.
- Artifact reads validate exact refs and declared length, reject short/long streams and oversized
  chunks, retain only the requested slice plus the truncation byte, use `memoryview`, explicitly
  close the iterator, and translate partial UTF-8 to base64.
- LiteLLM requires and forwards bounded `max_output_tokens`, caps chunks/bytes/calls/usage, supports
  parallel per-index builders, rejects same-chunk duplicate/illegal indices and post-finish/missing
  finish, and closes the provider iterator under success, timeout, and cancellation.
- `AcceptedRun.committed` is exact-bool trusted receipt metadata excluded from value equality. Runtime
  emits acceptance only for a new commit, applies a configurable hard observer timeout, and isolates
  timeout/failure. `CompositeObserver` bounds every child; synchronous event sinks use `to_thread`.
- `LocalToolGateway` rejects synchronous mutating handlers at registration, persists a stable
  cancellation result before propagating caller cancellation, and denies stale live bindings before
  creating a claim.

### Fix-round verification

```text
python -m pytest -q -p no:cacheprovider \
  tests/unit/adapters/model tests/unit/adapters/tools \
  tests/unit/adapters/observability tests/unit/application/test_runtime_v3.py \
  tests/unit/application/test_submissions.py
123 passed in 39.26s

ruff check --no-cache src tests/unit/adapters tests/unit/application/test_runtime_v3.py
All checks passed!

mypy --cache-dir F:/CodeSpace/.tmp/mypy-task12-fix2 src
Success: no issues found in 117 source files

python -m pytest -qq -p no:cacheprovider tests <four legacy excludes>
100% with no failures (1757 passed, 7 skipped)

python -m pytest -q -p no:cacheprovider tests
58 failed, 1762 passed, 7 skipped in 288.71s
```

The 58 failures remain exactly the four approved legacy groups (54 + 2 + 1 + 1); fix round 1 adds
no non-legacy regression.
