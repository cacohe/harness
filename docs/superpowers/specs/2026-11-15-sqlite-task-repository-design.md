# SQLite Task Repository Design

## Intent

Provide a durable, production-usable local `TaskRepository` that survives process restarts and
concurrent Agent sessions without changing the core Task snapshot contract or making persistent
storage an implicit default.

## Goals

- Persist complete JSON-safe `Task.to_dict()` snapshots in SQLite.
- Restore them through `Task.from_dict()` so state invariants and schema compatibility remain
  centralized in the Task model.
- Make every save an atomic upsert transaction.
- Support concurrent threads, processes, and multiple repository instances targeting one file.
- Detect unsupported database schema versions before reading or writing tasks.
- Surface stable `TaskRepositoryError` failures without leaking the host database path.
- Document explicit Agent injection and restart recovery.

## Non-goals

- Making SQLite the default Agent repository.
- A task-list/search/query API beyond the current `save` and `get` port.
- Distributed database support or an async network-database client.
- Session transcript persistence; this phase persists Task snapshots only.
- Packaging or publishing the project.

## Anti-goals

- No pickle or arbitrary Python object deserialization.
- No long-lived SQLite connection shared across threads.
- No silent reset, downgrade, or migration of a newer database schema.
- No partial file writes or ad-hoc JSON files.

## Constraints

- Use Python 3.11 standard-library `sqlite3`; add no runtime dependency.
- Keep `TaskRepository.save/get` source-compatible and synchronous.
- The repository path must identify a filesystem database; `:memory:` is rejected because the
  connection-per-operation design would not be durable.
- Default Agent composition continues to use `InMemoryTaskRepository` unless a repository is
  explicitly injected.

## Approaches Considered

### JSON files per task

Atomic rename can protect one writer, but multi-process coordination, index evolution, cleanup,
and crash behavior require recreating database features. Rejected.

### External service database

PostgreSQL or another service is appropriate for distributed deployments, but choosing one
would impose infrastructure and a client dependency on the Harness. The existing port remains
the extension point for those deployments. Rejected as the built-in implementation.

### SQLite

Selected. SQLite is available in the standard library and provides transactions, process-safe
locking, WAL, schema versioning, and durable local recovery without deployment infrastructure.

## Design

### Public API

`src/persistence/sqlite.py` defines:

- `TaskRepositoryError(RuntimeError)`, the stable storage failure boundary.
- `SQLiteTaskRepository(path: str | Path, *, timeout: float = 5.0)`.
- `save(task: Task) -> None`, validating/encoding before opening a write transaction.
- `get(task_id: str) -> Task | None`, returning an independent reconstructed Task.

The constructor canonicalizes the explicit file path, creates its parent directory when needed,
opens a short initialization connection, verifies `PRAGMA user_version`, creates the table for a
new version-0 database, and sets version 1. Version values above or below the supported nonzero
version raise `TaskRepositoryError`; the repository never rewrites them.

### Connections and transactions

Each operation opens and closes its own connection, making instances safe to call from different
threads without `check_same_thread=False`. Connections configure a bounded busy timeout,
`journal_mode=WAL`, and `synchronous=FULL`.

`save` encodes a strict JSON snapshot first, then runs `BEGIN IMMEDIATE` and an
`INSERT ... ON CONFLICT(id) DO UPDATE` statement. Context-manager commit/rollback guarantees an
old complete snapshot or a new complete snapshot, never a partial value.

`get` performs one indexed lookup, decodes strict JSON, and reconstructs `Task`. Missing ids
return `None`. Invalid JSON or an invariant-breaking snapshot is reported as
`TaskRepositoryError("stored task snapshot is invalid")`.

### Schema

Version 1 contains one table:

```sql
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,
    snapshot TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
```

The upsert refreshes `updated_at`. SQLite's `user_version` is the migration gate; future schema
changes must add an explicit migration rather than silently accepting a mismatch.

### Error semantics

- Invalid constructor path, directory creation, connection, locking timeout, disk-full, schema,
  SQL, encoding, and decode failures are wrapped in `TaskRepositoryError`.
- Public error messages name the operation or invariant, not the absolute host path or raw SQL.
- `KeyboardInterrupt`, `SystemExit`, and cancellation-like `BaseException` subclasses are never
  swallowed.

## Concurrency and async behavior

Repository methods are synchronous because that is the existing port and Harness save points are
synchronous. SQLite calls are bounded by the configured timeout and suitable for local embedded
durability. A deployment using a network database or requiring nonblocking persistence should
provide a dedicated repository implementation and schedule it at its service boundary; changing
the core port to async is intentionally a separate architecture project.

## Verification

- Tests cover restart recovery through a new instance, overwrite, independent returned objects,
  Unicode, missing ids, corrupt snapshots, unsupported versions, invalid paths, and no host-path
  leakage.
- Thread and multi-instance stress tests save distinct tasks plus repeated versions of the same
  task and verify every stored row reconstructs.
- Existing in-memory persistence and full Harness suites remain green.
- Ruff format/lint, Mypy, full pytest, and `uv lock --check` pass.
