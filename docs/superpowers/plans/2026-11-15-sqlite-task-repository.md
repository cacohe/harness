# SQLite Task Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable, concurrent, restart-safe SQLite implementation of the existing TaskRepository port.

**Architecture:** Use one short-lived standard-library sqlite3 connection per operation, version the database with `PRAGMA user_version`, store strict Task JSON snapshots, and atomically upsert under SQLite transactions. Keep Agent's default in-memory behavior and expose SQLite through explicit dependency injection.

**Tech Stack:** Python 3.11 sqlite3/json/pathlib, Pydantic Task snapshots, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-15-sqlite-task-repository-design.md`

## Global Constraints

- Do not package or publish the project.
- Add no runtime dependency.
- Never deserialize pickle or executable objects.
- Never silently reset or migrate an unsupported nonzero schema version.
- Public errors do not expose the absolute database path or SQL text.
- Default Agent composition remains in-memory unless a repository is injected.

## Review Focus

- Two repository instances initialized concurrently against a new file must not race schema creation.
- A write timeout/disk/SQL failure must roll back and preserve the previously committed snapshot.
- Corrupt JSON and invariant-invalid task data must cross the stable TaskRepositoryError boundary.
- Repeated concurrent upserts of one task id must leave one complete valid version, never mixed JSON.
- A newer schema version must be rejected without changing its version or tables.

---

### Task 1: Database Initialization and Version Boundary

**Files:**
- Create: `src/persistence/sqlite.py`
- Modify: `src/persistence/__init__.py`
- Create: `tests/unit/persistence/test_sqlite.py`

**Interfaces:**
- Consumes: `path: str | Path`, `timeout: float = 5.0`.
- Produces: `TaskRepositoryError`, `SQLiteTaskRepository`, version-1 `tasks` table, safe short-lived connections.

- [ ] **Step 1: Write failing initialization tests**

  Cover parent creation, version/table creation, idempotent second instance, concurrent first-time instances, `:memory:` rejection, directory-as-file failure, timeout validation, unsupported nonzero versions, and sanitized messages without the host path.

- [ ] **Step 2: Run focused tests and verify RED**

  Run: `python -m pytest tests/unit/persistence/test_sqlite.py -q -p no:cacheprovider`

  Expected: FAIL because `src.persistence.sqlite` does not exist.

- [ ] **Step 3: Implement constructor, connection configuration, schema verification, and error boundary**

  Use `Path.resolve(strict=False)`, create parents, reject `:memory:`, require finite positive timeout, configure WAL/FULL/busy timeout, and atomically claim only version 0.

- [ ] **Step 4: Run initialization tests and verify GREEN**

  Run: `python -m pytest tests/unit/persistence/test_sqlite.py -q -p no:cacheprovider`

  Expected: initialization tests PASS.

- [ ] **Step 5: Commit**

  Commit message: `feat: initialize sqlite task repository`

### Task 2: Atomic Task Snapshots and Concurrency

**Files:**
- Modify: `src/persistence/sqlite.py`
- Modify: `tests/unit/persistence/test_sqlite.py`

**Interfaces:**
- Consumes: `save(task: Task) -> None`, `get(task_id: str) -> Task | None`.
- Produces: atomic strict-JSON upsert, independent reconstruction, stable invalid-snapshot/storage failures.

- [ ] **Step 1: Write failing persistence tests**

  Cover missing id, Unicode round-trip, restart through a new instance, overwrite, returned-object independence, caller mutation after save, corrupt JSON, invariant-invalid JSON, and simulated failed upsert preserving the previous snapshot.

- [ ] **Step 2: Run persistence tests and verify RED**

  Run: `python -m pytest tests/unit/persistence/test_sqlite.py -q -p no:cacheprovider`

  Expected: FAIL because `save/get` are absent.

- [ ] **Step 3: Implement strict encoding, transactional upsert, lookup, decode, and sanitized wrapping**

  Encode and validate before opening the write transaction; catch ordinary sqlite/JSON/Pydantic failures while allowing BaseException subclasses through.

- [ ] **Step 4: Write failing thread/multi-instance stress tests**

  Use bounded worker counts to initialize multiple instances concurrently, save distinct ids, and race complete versions of one id. Assert no worker error, all distinct tasks restore, and the shared id equals one complete submitted version.

- [ ] **Step 5: Run stress tests and verify RED or existing behavior**

  If they pass immediately, record that SQLite transaction semantics already satisfy the test; otherwise fix only the reproduced concurrency defect.

- [ ] **Step 6: Run SQLite and existing persistence tests and verify GREEN**

  Run: `python -m pytest tests/unit/persistence/test_sqlite.py tests/unit/core/test_persistence.py -q -p no:cacheprovider`

  Expected: PASS.

- [ ] **Step 7: Commit**

  Commit message: `feat: persist task snapshots in sqlite`

### Task 3: Public Documentation, Full Gates, Review, and Integration

**Files:**
- Modify: `README.md`
- Modify: only files required by review findings.

**Interfaces:**
- Consumes: complete SQLiteTaskRepository API and branch diff.
- Produces: documented explicit injection/recovery contract and review-clean merged change.

- [ ] **Step 1: Document explicit Agent injection and restart recovery**

  Show constructing `SQLiteTaskRepository("./state/tasks.db")`, passing it to Agent, retrieving by task id from a new instance, default in-memory behavior, and local synchronous-I/O limits.

- [ ] **Step 2: Run Ruff format/lint, Mypy, uv lock check, and full pytest**

  Expected: all commands exit 0.

- [ ] **Step 3: Commit documentation**

  Commit message: `docs: explain durable task persistence`

- [ ] **Step 4: Dispatch one fresh whole-branch reviewer**

  Focus on transactions, schema races, rollback, corrupt data, error sanitization, thread/process safety, and scope accuracy.

- [ ] **Step 5: Fix Critical/Important findings with RED→GREEN tests and rerun full gates**

  Record all rulings and deferred minors.

- [ ] **Step 6: Push, verify CI, merge, compare merged tree, and remove only this plan's ignored workspace**
