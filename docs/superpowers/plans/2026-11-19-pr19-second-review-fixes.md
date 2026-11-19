# PR #19 Second Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 PR #19 第二轮评审确认的阻塞问题，并完成紧邻边界的一致性修复。

**Architecture:** Context 通过一个统一的强制消息后缀契约保证顺序；Artifact 通过每次写入独立的 publication ID 隔离永久墓碑与新对象；MCP async context 由单一 owner task actor 持有；composition 只在应用层收敛默认 adapter 策略。

**Tech Stack:** Python 3.11, asyncio, anyio, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-11-19-pr19-second-review-design.md`

## Global Constraints

- 所有修改只发生在 `F:\CodeSpace\harness-worktrees\lean-core-context-artifacts`。
- `src/core` 不导入 application 或 adapters。
- 不提供向后兼容分支。
- 每个行为修改先写失败测试并观察预期失败。

## Review Focus

- 同一 Run 中 current user 与 open tool batch 的相对顺序由 integration test 固定。
- Artifact 的旧 publication ID 永远不能越过 `RETIRED` 墓碑。
- 相同内容重新写入必须得到新 ID，旧 delete 与新对象互不影响。
- MCP owner task 在超时、remove 和 gateway close 时都必须有界清理。
- 名称规范化后的碰撞不得将调用路由到错误的远端工具。

---

### Task 1: Context message order

**Files:**
- Modify: `src/adapters/context/bounded.py`
- Modify: `src/core/run_projection.py`
- Test: `tests/unit/adapters/context/test_bounded_context.py`
- Test: `tests/integration/test_agent_v3.py`

**Interfaces:**
- Consumes: `ContextRequest.current_input`, `ContextRequest.open_tool_messages`
- Produces: messages ending in `(current_input, *open_tool_messages)` and matching core validation

- [ ] Write exact-order failing tests for the provider and real multi-step agent flow.
- [ ] Run the focused tests and confirm the old suffix order fails.
- [ ] Change adapter packing and core shape validation to the new suffix.
- [ ] Run focused context/integration tests and commit.

### Task 2: Independent Artifact publication identities

**Files:**
- Modify: `src/core/artifacts.py`
- Modify: `src/adapters/artifacts/memory.py`
- Modify: `src/adapters/artifacts/filesystem.py`
- Test: `tests/unit/core/test_journal_contract.py`
- Test: `tests/unit/application/test_artifact_retention.py`
- Test: `tests/integration/test_artifact_flow.py`
- Modify: `README.md`

**Interfaces:**
- Produces: every successful `ArtifactRepository.put()` returns a unique immutable publication ID
- Preserves: `ArtifactRef.sha256` as the content digest; journal attachment and tombstone rules remain unchanged

- [ ] Add tests proving equivalent writes have distinct IDs and deleting the old ID cannot affect the new publication.
- [ ] Run memory/filesystem tests and confirm deterministic-ID assumptions fail.
- [ ] Separate publication identity validation from content digest validation in core, then update repository adapters.
- [ ] Add end-to-end retire/reproduce test, update README, run focused suites, and commit.

### Task 3: MCP owner task and normalized names

**Files:**
- Modify: `src/adapters/tools/mcp.py`
- Test: `tests/unit/adapters/tools/test_mcp.py`

**Interfaces:**
- Produces: owner-task proxy supporting `list_tools`, `call_tool`, and `aclose`
- Produces: descriptor tuples retaining `(remote_name, public_segment, description, schema_json)`

- [ ] Add an anyio task-group client test asserting enter/call/exit task identity.
- [ ] Add dotted, slashed, long-name, original-routing, and collision tests.
- [ ] Run focused tests and confirm current task split/name rejection failures.
- [ ] Implement queued owner task and stable provider-safe aliases without changing sync client behavior.
- [ ] Run the complete MCP test file and commit.

### Task 4: Development composition consistency

**Files:**
- Modify: `src/application/composition.py`
- Test: `tests/unit/application/test_agent_v3.py`

**Interfaces:**
- Consumes: `ApplicationConfig.limits.max_inline_result_bytes`
- Produces: default development maintenance service and a clamped internal `ArtifactPolicy`

- [ ] Add failing tests for `agent.artifacts` and a small Run inline limit.
- [ ] Implement compatible maintenance discovery and development-only policy clamping.
- [ ] Run application composition tests and commit.

### Task 5: Verification and branch review

**Files:**
- Modify only if verification exposes a regression.

**Interfaces:**
- Consumes: all prior task outputs
- Produces: green locked pytest/type/format/lint CI evidence

- [ ] Run focused cross-subsystem tests, full `pytest`, and static checks available without launching broken Windows `ruff.exe`.
- [ ] Build a whole-branch review package and perform one fresh review pass.
- [ ] Fix Critical/Important findings through RED→GREEN, then push and verify GitHub Actions.

