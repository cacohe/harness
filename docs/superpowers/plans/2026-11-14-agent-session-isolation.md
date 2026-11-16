# Agent Session and TaskRun Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Let one Agent run multiple isolated conversations concurrently through explicit `AgentSession` handles while preserving the current default-session API.

**Architecture:** Move transcript and paused-run ownership into `AgentSession`; represent a paused turn with `TaskRun`; keep Harness, tools, repository, and MCP lifecycle shared by Agent. A non-blocking per-session guard rejects same-session reentry while different sessions run concurrently.

**Tech Stack:** Python 3.11, asyncio, threading locks, Pydantic Task model, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-14-agent-session-isolation-design.md`

## Global Constraints

- Do not package or publish the project.
- Different sessions may overlap; one ordered session never runs two operations concurrently.
- Session state never lives in shared Agent mutable fields except through the default-session compatibility proxy.
- Existing Agent APIs and return types remain compatible.
- All operation guards release in `finally`, including cancellation and exceptions.

## Review Focus

- Same-session reentry during an await must fail immediately without cancelling or corrupting the first turn.
- A cancelled or failing operation must release the session guard for the next call.
- Two paused sessions with identical tool-call ids must approve/resume independently.
- A new message must not overwrite a paused task awaiting approval.
- Closing Agent must invalidate sessions created before closure without affecting read-only state inspection.

---

### Task 1: AgentSession and TaskRun Ownership

**Files:**
- Create: `src/demo/session.py`
- Create: `tests/unit/demo/test_session.py`

**Interfaces:**
- Consumes: shared `Harness`, `TaskRepository`, and `ensure_open: Callable[[], None]`.
- Produces: `TaskRun(task: Task, message: str)`, `AgentSession.chat/achat/resume/aresume`, `approval_requests`, `decide_tool_call`, `transcript`, `pending_run`, `pending_task`, and `pending_message`.

- [x] **Step 1: Write failing tests for isolated turn and paused-run ownership**

  Test one completed turn, transcript carry-forward, one paused task, scoped approval decision/save, resume commit, rejection of a new chat while paused, and no transcript commit for failed/cancelled tasks.

- [x] **Step 2: Run the focused tests and verify RED**

  Run: `python -m pytest tests/unit/demo/test_session.py -q -p no:cacheprovider`

  Expected: FAIL because `src.demo.session` does not exist.

- [x] **Step 3: Implement TaskRun and AgentSession with the non-blocking operation guard**

  Use `threading.Lock.acquire(blocking=False)` before every chat/resume operation and release it in `finally`. Build Task input from only the session transcript and keep pending ownership in `TaskRun`.

- [x] **Step 4: Run session tests and verify GREEN**

  Run: `python -m pytest tests/unit/demo/test_session.py -q -p no:cacheprovider`

  Expected: PASS.

- [x] **Step 5: Commit**

  Commit message: `feat: add isolated agent sessions`

### Task 2: Agent Compatibility and Concurrency Composition

**Files:**
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `tests/unit/demo/test_session.py`

**Interfaces:**
- Consumes: Task 1 `AgentSession` and `TaskRun`.
- Produces: `Agent.new_session() -> AgentSession`, default-session delegation, compatibility properties, and shared-Agent/different-session concurrency.

- [x] **Step 1: Write failing Agent composition and concurrency tests**

  Cover two sessions overlapping through a rendezvous gateway with isolated prompts/transcripts, same-session reentry failure, independent paused tasks sharing a tool-call id, wrong-session decisions, cancellation/error lock release, Agent close invalidation, and compatibility setters for `pending_task/pending_message`.

- [x] **Step 2: Run focused tests and verify RED**

  Run: `python -m pytest tests/unit/demo/test_session.py tests/unit/demo/test_agent.py -q -p no:cacheprovider`

  Expected: FAIL because Agent has no session composition/delegation.

- [x] **Step 3: Compose the default session, add `new_session`, and delegate legacy APIs**

  Remove shared transcript/pending mutation from Agent methods. Keep MCP lifecycle and shared dependencies on Agent; sessions call Agent's existing `_ensure_open` callback.

- [x] **Step 4: Run Agent/session tests and verify GREEN**

  Run: `python -m pytest tests/unit/demo/test_session.py tests/unit/demo/test_agent.py -q -p no:cacheprovider`

  Expected: PASS with actual rendezvous overlap.

- [x] **Step 5: Commit**

  Commit message: `feat: isolate concurrent agent conversations`

### Task 3: Documentation, Full Gates, and Whole-Branch Review

**Files:**
- Modify: `README.md`
- Modify: only files required by review findings.

**Interfaces:**
- Consumes: complete session API and branch diff.
- Produces: documented concurrency contract and review-clean branch.

- [x] **Step 1: Document session usage and exact concurrency semantics**

  Show `first = agent.new_session()` / `second = agent.new_session()` with `asyncio.gather`, same-session rejection, paused-run ownership, default-session compatibility, and Agent-close behavior.

- [x] **Step 2: Run all project gates**

  Run Ruff format/lint without worktree cache, Mypy with the task cache, `uv lock --check`, and full pytest.

  Expected: all commands exit 0.

- [x] **Step 3: Commit documentation**

  Commit message: `docs: explain agent session concurrency`

- [x] **Step 4: Generate a review package and dispatch one fresh whole-branch reviewer**

  Review state ownership, same-session guard release, task/approval isolation, cancellation, sync bridge behavior, backward compatibility, and README accuracy.

- [x] **Step 5: Reproduce Critical/Important findings with failing tests, fix once, and rerun full gates**

  Expected: every fix records RED→GREEN and the full suite passes.

- [x] **Step 6: Push, verify CI, merge the PR, and verify the merged tree**

  Preserve the host-owned worktree; delete only this plan's ignored SDD workspace after recording rulings and deferred minors.

完成状态：已通过 PR #9 合入 `main`，会话状态、审批、取消与任务隔离均已有回归测试。
