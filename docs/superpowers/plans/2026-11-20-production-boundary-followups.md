# Production Boundary Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让未知工具调用可恢复、让 MCP `isError` 的安全详情进入模型 transcript，并让 Artifact GC 安全回收崩溃 staging。

**Architecture:** core 只增加通用的 unavailable tool-call 语义和有界 failure detail；MCP 可见错误的认定、截断及文件系统租约/回收均留在 adapters。当前无兼容要求，journal codec 与值对象同步升级。

**Tech Stack:** Python 3.11、pytest、asyncio、SQLite、Windows/POSIX filesystem primitives。

**Spec:** `docs/superpowers/specs/2026-11-20-production-boundary-followups-design.md`

## File structure

- Modify: `src/core/commands.py` — 持久化未知工具 availability。
- Modify: `src/core/run_projection.py`、`src/core/run.py` — 投影并短路未知工具，输出 failure detail。
- Modify: `src/core/errors.py`、`src/core/tools.py`、`src/core/state.py`、`src/core/receipt_snapshot.py`、`src/core/state_validation.py` — 有界 failure detail 的值对象与快照。
- Modify: `src/adapters/tools/local.py`、`src/adapters/tools/mcp.py` — 显式 model-safe adapter 错误。
- Modify: `src/adapters/persistence/sqlite_codec.py` — detail 的 durable round-trip。
- Modify: `src/adapters/artifacts/filesystem_platform.py`、`src/adapters/artifacts/filesystem.py` — 分片 staging 租约和 GC。
- Modify tests under matching `tests/unit/...` paths — 行为、持久化、安全与并发回归。

## Task 1: Unknown tools fail locally without failing the Run

**Interfaces**

- `ToolCallRequest(..., *, available: bool = True, ...)`；unavailable 请求必须 `requires_approval=False`。
- `model_tool_requests()` 为缺失 catalog entry 的调用返回 `available=False` 的请求。
- `RunCoordinator._execute()` 在 gateway 前把 unavailable 请求映射为 `RunFailure(TOOL_FAILED)`。

- [ ] 在 `tests/unit/core/test_run_coordinator.py` 写回归测试：unknown + known batch 中 unknown 不调用 gateway、known 正常执行、transcript 含 unknown `tool_failed`、Run 可进入下一模型 step。
- [ ] 运行该测试并确认它因当前 Run 变为 `MODEL_FAILED` 而失败。
  Expected: 新测试失败，状态为 `MODEL_FAILED` 或 gateway 行为与验收不符。
- [ ] 最小实现 availability、snapshot/validation 和 coordinator short-circuit。
- [ ] 运行 core command/state/coordinator 与 SQLite codec 定向测试并确认通过。
  Expected: 所有定向测试通过，退出码为 0。
- [ ] Commit: `fix(core): isolate unknown tool failures`

## Task 2: Preserve explicitly safe MCP error detail

**Interfaces**

- `RunFailure(code: RunErrorCode, detail: str | None = None)`，detail 最大 65,536 UTF-8 bytes。
- `ToolReportedError(detail: str)` 只供 adapter 显式声明模型可见错误。
- `LocalToolGateway` 对 `ToolReportedError` 返回带 detail 的 failed `ToolResult`，其他异常无 detail。

- [ ] 在 local、MCP、core coordinator 和 SQLite codec tests 中分别写失败测试，覆盖 MCP 文本端到端保留、普通异常脱敏、超大 detail 有界、journal round-trip。
- [ ] 运行新增测试并确认当前实现分别丢失 detail 或拒绝新接口。
  Expected: 新测试因 detail 丢失、接口不存在或 codec 不保留 detail 而失败。
- [ ] 最小实现 `RunFailure`/`ToolResult`/state/receipt/codec detail 通道与 transcript 字段。
- [ ] 在 MCP adapter 中将规范化 `isError` 转成有界 `ToolReportedError`；保持其他异常路径不变。
- [ ] 运行 adapters/tools、core 和 persistence 定向测试并确认通过。
  Expected: 所有定向测试通过，退出码为 0。
- [ ] Commit: `fix(tools): preserve bounded reported errors`

## Task 3: Reclaim abandoned filesystem staging safely

**Interfaces**

- filesystem platform 增加显式 acquire/release object-lock handle；现有 context manager 复用同一实现。
- `.put-<uuid>` 映射到 `stage-<first-byte>` 固定分片锁。
- `collect_orphans(referenced_ids, older_than)` 回收通过租约与严格成员验证的旧 staging，并把数量计入返回值。

- [ ] 在 filesystem artifact tests 中先写失败测试：旧 crash stage 被删、new stage 保留、活跃租约 stage 保留、symlink/hard-link/未知成员保留、并发 put 与 GC 不互删。
- [ ] 运行新增测试并确认旧 staging 当前被忽略。
  Expected: 新测试显示旧 `.put-*` 仍存在或回收数量不正确。
- [ ] 最小实现可持有锁 handle、writer 分片租约及 GC 二次校验。
- [ ] 运行完整 filesystem artifact test file 并确认通过。
  Expected: 文件中的全部测试通过，退出码为 0。
- [ ] Commit: `fix(artifacts): reclaim abandoned staging`

## Task 4: Verification and review

- [ ] 运行 `ruff check .`。
  Expected: `All checks passed!`，退出码为 0。
- [ ] 运行 `mypy src`。
  Expected: `Success: no issues found`，退出码为 0。
- [ ] 使用 `.venv\\tmp` 作为 `TEMP`/`TMP` 运行 `python -m pytest` 全量测试。
  Expected: 全部测试通过，仅允许项目已有的显式 skip，退出码为 0。
- [ ] 审查 `git diff --check`、分支 diff 和工作区副作用；清理本轮临时目录，但保留 worktree 供用户验收。
  Expected: `git diff --check` 无输出；无计划外文件或残留进程。

## Review Focus

- unavailable 标记是否可能被伪造为可执行调用，或触发审批等待。
- failure detail 是否存在普通异常泄漏、无界 UTF-8 增长或 codec 丢失。
- MCP 超大/嵌套错误结果是否仍满足既有 JSON 规范化边界。
- staging GC 是否可能删除活跃 writer，或被 symlink/hard link/TOCTOU 绕过。
- 分片锁获取是否会阻塞 asyncio event loop，锁文件数量是否保持固定上界。
