# Task 11 实施报告

## 结果

Task 11 已将 application facade 重建为 async-first `Agent` / `AgentSession`，新增
`RuntimeCoordinator`、严格 immutable `ApplicationConfig` 以及 development/production
composition factories。Application 只负责接受 submission、Session 并发治理和反复调用 Core
单步 `RunCoordinator.advance()`；terminal transcript 继续由 Core 的原子 `JournalCommit` 写入。

## TDD 与 RED

先新增 `test_agent_v3.py` 和 `test_runtime_v3.py`。最初 RED 为 7 项：
`src.application.runtime`、`src.application.composition` 不存在，且公开 application surface
仍为旧同步 facade。修正异步测试驱动器后，未在实现前借助 legacy API 获得伪通过。

最终 14 个 v3 测试覆盖：

- 同 Session 有界 FIFO、不同 Session 并发、waiter 上限与取消转交；
- active 异常释放、close/submit race、queued waiter 拒绝、active cancellation、重复 close；
- close caller cancellation 不取消 shared cleanup；owned resource 逆序 exactly-once close，
  caller-injected resource 不关闭；
- Session registry 容量与显式回收；
- approval 的 Session/Run/tool durable ownership，以及 `DecideToolCall -> ResumeRun -> AdvanceRun`
  单步顺序；
- duplicate terminal submission 不二次 drive、不由 application 追加 terminal facts；
- async-only facade、严格配置、显式 production paths、lazy application exports。

## API、gate 与 ownership

- `Agent.submit/open_session/close` 与 `AgentSession.submit/decide_tool_call/close` 均为 async API；
  不提供 `chat`、`achat`、`from_config` 或 `Harness` compatibility branch。
- 每个 Session 使用显式 `_Gate`：deque 保存有限 waiter；取消尚未授予的 waiter 会摘除，
  取消刚获 owner 的 waiter 会把 owner 转交下一个 FIFO waiter；异常路径统一 release。
- Runtime 的 closing 转换在 lifecycle lock 内完成。shared close task 用 `asyncio.shield()` 防止
  某个调用方取消导致资源清理中断；所有 active tasks 均 cancel 后 await。
- approval 在 coordinator I/O 前从 durable journal head 校验 Session、Run、当前 pending
  tool_call，并从 canonical journal snapshot 恢复唯一 user input。
- composition 的 model/tool/context 仍是显式 ports；Task 12 之前未复用 legacy concrete wiring。
  injected dependencies 默认 caller-owned，仅显式 `owned_resources` 交给 Runtime。

## 验证

锁定 CPython 3.11 通过工作树 `.venv/Lib/site-packages` 加载依赖。由于 worktree 内 Ruff
临时替换被 Windows ACL 拒绝，使用同版本 `F:/CodeSpace/harness/.venv/Scripts/ruff.exe`
做只读检查，并按 `--diff` 用 `apply_patch` 格式化。

| 检查 | 结果 |
| --- | --- |
| Task 11 + submissions/export related | `66 passed in 17.46s` |
| 排除旧 `test_agent.py` 的全仓 pytest | `1675 passed, 7 skipped, 4 failed in 213.63s` |
| mypy `src` | `Success: no issues found in 103 source files` |
| Ruff lint `src tests` | `All checks passed!` |
| Ruff format `src tests` | `165 files already formatted` |
| `git diff --check` | 通过（仅 Git 的 LF/CRLF 提示） |

全仓剩余失败均为与“不保留 v2 compatibility”直接冲突的旧测试，共 58 项，交 Task 14
删除/替换：

- `tests/unit/application/test_agent.py`：54 项；
- `tests/integration/test_mcp_agent.py`：2 项；
- `tests/load/test_shared_agent_load.py`：1 项；
- `tests/unit/core/test_ports.py`：1 项。

前三类要求旧 `config=` 构造、`chat/achat/new_session` 等 API；最后一项直接断言旧
`Agent.__init__` 的 `llm_gateway` annotation。没有为这些测试恢复兼容层。

## 已知文件约束

Windows ACL 拒绝物理删除/替换已跟踪的 `src/application/agent.py`。因此公开模块末尾将
`Agent`/`AgentSession` 名称切换为 v3 subclass，旧构造和方法不再可达，但旧文件主体暂时仍在；
Task 14 应在删除 legacy application 文件时物理清除该主体。新实现独立位于
`src/application/agent_v3.py`，公开 `src.application.Agent.__module__` 仍保持
`src.application.agent`。

## Review fix round 1/5

基于 `128a3f7` 的第一轮复审修复了四个 Important finding，未处理已明确 deferred 到
Task 14 的 legacy `agent.py` 主体。

### RED

- Runtime 攻击组初始 `2 failed, 2 passed`：close/reopen 后旧 gate owner 仍可进入 I/O；
  approval 仍调用 `load_session()`。
- shared journal 新 contract 初始 `4 failed`：memory/SQLite 均缺少 `find_run_input()`。
- composition 攻击组初始 `11 failed`：8 类无效 port 未拒绝、恶意 descriptor 未拒绝、
  partial construction 未 cleanup、成功 composition 未接管 factory-created resources。
- `ApplicationConfig` 的 concrete `Path` subclass 拒绝测试在实现前失败。

### 修复

- 所有公共 Session/Run/Submission/Profile ID 边界先验证 exact class，通过
  `object.__getattribute__` 取得 exact `str`，再用可信构造器重建；攻击测试确认不执行恶意
  `.value`、`__eq__` 或 `__hash__` hook。
- active operation 记录改为 `_SessionSlot` identity。gate owner 在 registry→lifecycle 固定锁序下
  原子验证当前 slot 并登记 active；`close_session()` 先移除 slot，再按相同锁序捕获并
  cancel/await active。精确 barrier 覆盖 close/reopen、queued、active 与 caller cancellation。
- `RunJournal` 新增 bounded `find_run_input(session_id, run_id, submission_id)`。memory 维护
  O(1) identity index；SQLite strict v3 schema 新增
  `transcript_run_input(session_id, run_id, submission_id, role, sequence)`，使用 `LIMIT 2`
  检测重复，并验证 row/payload/Run identity 与 immediate hash-chain anchor。Runtime 不再为审批
  materialize 完整 Session history。
- `ApplicationConfig` 重建 `RunLimits`、`ContextBudget` 和 exact platform concrete Path。
  Composition 使用 `inspect.getattr_static()` 验证 ports，不执行 property descriptor；全部输入
  验证通过后才创建路径资源。factory-created journal/repository/externalizer 纳入 Runtime ownership，
  构造失败时已创建资源逆序 cleanup，不删除持久数据。

### 验证

| 检查 | 结果 |
| --- | --- |
| Runtime + composition v3 | `30 passed in 15.13s` |
| 10k bounded journal input / duplicate contract | `4 passed` |
| SQLite row/payload/identity tamper | `3 passed` |
| Task 10 shared journal/schema/recovery + Task 11 related | `297 passed in 172.21s` |
| 排除旧 `test_agent.py` 的全仓 pytest | `1699 passed, 7 skipped, 4 legacy failures in 225.52s` |
| mypy `src` | `Success: no issues found in 103 source files` |
| Ruff lint `src tests` | `All checks passed!` |
| Ruff format `src tests` | `165 files already formatted` |

全仓 4 个失败仍是报告前述 legacy 清单中的非 `test_agent.py` 部分：
`test_mcp_agent.py` 2 项、`test_shared_agent_load.py` 1 项、`test_ports.py` 1 项。
