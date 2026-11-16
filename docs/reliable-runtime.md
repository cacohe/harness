# Reliable runtime operations

## Durable facts and observations

`SessionEvent` is the recovery/audit record. Task snapshots and fact batches are committed together with sequence CAS. Client/tracing `Event` objects are observations: sink failure cannot roll back facts, and streamed deltas are never durable facts.

SQLite schema v2 adds `session_streams`, `session_events`, and `submissions`. Back up the database before first opening it with this release. Migration from v1 is transactional and preserves legacy Task snapshots without inventing history. A v2 database is not writable by v1 code; rollback requires restoring the backup.

## Idempotency and recovery

Supply a stable `message_id` to `AgentSession.chat/achat`. The digest is SHA-256 of the exact raw UTF-8 message (no trimming or Unicode/newline normalization). Same ID and content reuse the stored Task; different content raises `SubmissionConflictError`.

`Agent.open_session(session_id)` validates history, compensates interrupted open lifecycles without replaying tools, and restores transcript or pending approval ownership. Started-but-unsettled tools recover as interrupted/unknown. Paused approvals remain paused.

## Limits and shutdown

Model/tool deadlines and byte budgets come from `.env.example`. Oversized values fail before Task/fact mutation; output is never byte-truncated into invalid JSON. `MAX_PENDING_SUBMISSIONS=0` preserves immediate same-Session reentry failure; positive values enable a bounded FIFO.

`Agent.aclose()` rejects new Session/MCP work, cancels active async Session operations, waits for settlement, flushes, then closes owned resources in reverse order. Injected resources remain caller-owned unless passed in `owned_resources`. A close timeout leaves shielded cleanup running; call `aclose()` again to observe completion.

## Compatibility

Legacy `TaskRepository` implementations remain supported through `SnapshotExecutionStore`, but their facts are process-local and cannot provide restart recovery or cross-process idempotency. Production adapters should implement `ExecutionStore` and CAS exactly as documented by the protocol.
