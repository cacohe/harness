# Final review C：Artifact durability 与 SQLite projection authentication

基线为 `db2cd6e`。本轮在专用 worktree
`F:\CodeSpace\harness-worktrees\lean-core-context-artifacts` 实施；无兼容迁移，SQLite schema
由 v5 直接升级为 v6。

## Durable Artifact attachment/index

RED 首先复现了完成态工具结果的问题：Artifact descriptor 只出现在已经清空的
`current_step`，重启后的 Recovery 无法再发现已删除对象。新增独立的 append-only lifecycle
事实与 active index：

- `ArtifactAttachmentFact` 保存完整 canonical `ArtifactRef`，并绑定
  `session_id/run_id/submission_id/owner_id/role/action`；
- role 覆盖 `tool_result`、`final_output`、`summary`，action 覆盖 `attached/detached`；
- `RunCoordinator` 在 tool/final externalization 后及 summary checkpoint 提交时自动写事实；
- Memory 与 SQLite journal 都维护按 Run 查询的 active index，以及跨 Session 的 active 引用判断；
- SQLite 同时保留 append-only `artifact_attachment_facts` 和派生
  `artifact_attachments`，读取时重放并核对两者；fact/index 的插入、更新、删除和跨 identity
  移动均被受控写通道或一致性验证拒绝；
- Recovery 先查询目标 Run 的 active attachments，再对每个完整 descriptor 做 repository
  `stat` 比对；缺失、替换、metadata/media type/size/digest 不一致都稳定映射为
  `ARTIFACT_FAILED`；journal 不保存 payload 或 host path；
- application 层新增 `ArtifactRetentionService`：任何 active attachment 都阻止 delete；共享
  digest 必须等全部 Run/Session owner detach 后才能 retire。Core repository port 保持存储无关。

retirement 不信任 active index 单点：它按 `artifact_facts_by_artifact(artifact_id)` 重放该
Artifact 的 lifecycle，并与 active index 比较。SQL cursor 流式读取且显式 `LIMIT 1,000,001`；
超过 1,000,000 facts 稳定拒绝。复杂度为
`O(min(F_artifact, 1,000,001))` 时间与 `O(active owners)` 内存，不扫描无关 journal。

覆盖了 large tool → complete → state/current_step 清空 → SQLite reopen、对象缺失、descriptor
冲突、detach 非 active、重复/冲突 attach、fact/index 直接篡改、共享 digest 与 retention gate。

## SQLite v6 authenticated projection

RED 证明 v5 下同时修改 `sessions.state_json` 与 `runs.state_json` 为同一合法状态后，
`load_head()` 会接受伪造。v6 增加：

- `sessions.session_state_hash`；
- `runs.run_index/state_hash`；
- 固定 32 层 `run_projection_nodes` sparse Merkle tree；
- `sessions.run_projection_hash/run_count`；
- authenticated Session root 同时绑定 ledger、coverage、summary、当前 Session state hash、
  Run projection root 与 count。

commit 对当前 Run 单叶增量更新，无需扫描所有 Run；`load_head()` 为 O(1) row + 固定 32 层
proof，`find_run_state()` 对历史 Run 做固定深度 proof；`load_session()` 继续承担完整历史认证。
Run/Session identity 不可移动，投影不可删除。approval/checkpoint 等字段包含在 canonical
`RunState` 中，因此同步修改 approval binding 也会因 state hash 不匹配而失败。v5 已认证的
summary tree 继续作为 authenticated root 的独立组成部分。

攻击测试覆盖同步 Session+Run payload、历史 Run 中间 payload/state_hash/leaf/node/root/count、
approval decision、Session root、跨 Session 移动、codec malformed、冗余物理字段与现有
ledger/summary/coverage corruption。

`JournalSnapshot`、SQLite codec、Memory canonical copy 与 commit receipt snapshot/validation
全部 exact-copy 并认证 `artifact_facts`；伪造 receipt descriptor 不能被 equality hook 或字段遗漏
绕过。

## TDD 与验证

关键 RED：

- attachment contract 在 Memory/SQLite 均因 canonical codec 不认识新事实而失败：`2 failed`；
- 同步修改两份合法 `state_json` 时 `load_head()` 未抛错：`Failed: DID NOT RAISE`。

GREEN/回归：

| 检查 | 结果 |
| --- | --- |
| attachment contract / lifecycle / tamper | `10 passed` |
| historical Run payload/hash/node/root/count attacks | `5 passed` |
| approval binding + cross-Session attack | `2 passed` |
| journal contract + SQLite 全集（最终） | `250 passed in 343.34s` |
| receipt/runtime/recovery/Artifact/load E2E（最终） | `288 passed, 2 skipped in 29.01s` |
| Ruff | `All checks passed!` |
| mypy `src` | `Success: no issues found in 58 source files` |
| CPython 3.11 全仓 | `1261 passed, 2 skipped in 432.28s` |
| `git diff --check` | 通过，仅 LF/CRLF 提示 |

全仓使用 `cpython-3.11.16-windows-x86_64-none` 与 worktree `.venv` dependencies。

## Fix round 1/5：公开 detach 生命周期与 Memory facts history

本轮 RED 为 `ModuleNotFoundError: No module named 'src.application.artifact_lifecycle'`：生产
attachment 虽然 durable，却没有调用方可达的释放入口。

新增 `ArtifactLifecycleService`，并通过 `RuntimeCoordinator` 暴露
`AgentSession.detach_artifacts(run_id, roles=None)`。调用方只能选择 Run 和 role，不能提交
descriptor；服务从 journal active index 取得完整 owner/role/descriptor，确认目标 Run 属于当前
Session 且已经 terminal，再以当前 Session head 做 CAS state-only commit，原子追加精确
`DETACHED` facts。目标可以是历史 Run，commit 不会把 Session head 回退到旧 Run。CAS 冲突会重读
durable state 后重试；空 active set、重复调用和并发调用都幂等。observer 仍在 commit 后运行，
失败不回滚 detach，随后 retry 返回空集合。

`RunJournal` 新增 authenticated `find_run_state_by_id()` 和有界
`load_artifact_facts(session_id, run_id, after_index, limit<=1024)`。Memory journal 现在除 active
派生索引外，还保存 Session-global、append-only `ArtifactFactRecord(index, fact)` 历史；SQLite
使用既有 `fact_index` 并新增 `(session_id, run_id, fact_index)` 索引分页。facts 不复制到
`RunState`。

覆盖：Memory/SQLite reopen、tool/final/summary role 筛选、整 Run 释放、重复与并发 detach、真实
stale CAS、wrong Run/non-terminal、两 Run 共享 digest、detach/retire 并发、observer failure 后
retry，以及公开 AgentSession 生产生成 Artifact 流程。README 与 reliable runtime 文档已同步；
Session/Agent close 不隐式释放 durable ownership。

本轮最终验证：

| 检查 | 结果 |
| --- | --- |
| lifecycle/retention/public API focused | `44 passed` |
| journal contract + SQLite | `250 passed in 331.12s` |
| application/runtime/Artifact/recovery/load E2E | `90 passed in 14.02s` |
| Ruff `src tests` | `All checks passed!` |
| mypy `src` | `Success: no issues found in 59 source files` |
| CPython 3.11 全仓 | `1269 passed, 2 skipped in 461.19s` |

## Fix round 2/5：active-only Recovery、retirement reservation 与有界分页

三个 RED 分别确认：detached final ref 在 SQLite reopen 后仍被 Recovery 当成 dangling；Memory
和 SQLite 都没有原子 retirement reservation；Memory `load_artifact_facts()` 在应用 limit 前
迭代完整 lifecycle history。

Recovery 现在只验证目标 Run 的 durable active attachments。RunState、tool terminal outcome、
final output 与 summary checkpoint 中保留的历史 ref 在明确 detach/retire 后只作为审计数据，
不再要求 repository 对象继续存在；未 detach 的 active descriptor 缺失或篡改仍稳定失败。

journal 新增 generation-bound retirement saga：`reserve` 在同一 Memory lock 或 SQLite
`BEGIN IMMEDIATE` 中检查所有 active owners 并写入 `RETIRING`，ATTACHED commit 同一原子边界
拒绝 `RETIRING`；delete 成功后 `complete` 为 `RETIRED`，明确失败且对象仍存在时 `abort` 为
`ABORTED`。reserve/delete/complete 任一点崩溃后重复 retire 都能收敛，重复完成幂等。content-
addressed payload 再次 put 后，ATTACHED 会原子开启下一 generation 的 `AVAILABLE`，因此不会
永久 tombstone 某 digest；共享 digest 仍必须逐 owner detach。

Memory journal 为每个 `(session, run)` 维护有序 fact-index tuple，cursor 用 `bisect_right`，先
slice `limit` 再 canonical copy，复杂度为 `O(log F_run + limit)`，不扫描其他 Run 或完整事实
历史。100,000 条、两个 Run 的 guard 测试观测到只访问 5 条记录。

RED/GREEN 证据：Recovery RED `1 failed`；reservation contract RED `2 failed`；分页 guard RED
`1 failed`。最小实现后四项参数化用例 `4 passed`；Recovery/generation `20 passed`；retirement
竞态、delete failure、restart 与重复调用 `7 passed`；schema/retention/Recovery focused
`23 passed`。最终验证：wide contract/SQLite `293 passed in 362.24s`；Artifact/runtime/load
E2E `39 passed in 46.93s`；Ruff lint 与 format check 通过；mypy `src` 为
`Success: no issues found in 59 source files`；CPython 3.11 全仓
`1279 passed, 2 skipped in 446.90s`。

## Fix round 3/5：retirement authenticated projection

审查 RED 证明两个信任缺口：raw SQL 注册既有 lifecycle UDF 后删除 derived active row 并清除
sticky violation，reserve 会错误成功；直接改写 retirement generation/state 后 reopen reserve
会错误接受。两项复现为 `2 failed`。

SQLite reserve 现在在同一个 `BEGIN IMMEDIATE` 内，使用
`artifact_facts_by_artifact(artifact_id)` 有界读取（上限 1,000,000+1），重放 append-only
attachment facts 得到 active owners，并与 derived index 做完整双向 descriptor/identity 核对；
缺行、多行、跨 Session 移动或 descriptor 不一致均拒绝，owner 判断与 reservation 写入之间
没有事务窗口。

retirement transition 改为 per-artifact append-only canonical fact/hash chain。current projection
绑定 `artifact_index/generation/state/fact_count/fact_root/state_hash`；每个 current state 是固定
32 层 sparse Merkle tree 的 leaf，singleton database meta 用独立 authenticated root 绑定
projection root 与 artifact count。reserve、complete、abort、restart，以及 retired digest
re-put 后开启下一 generation，都会先验证 meta、32 层 point proof、目标完整事实链与 current
projection，再增量追加 transition 并更新一条 Merkle path。普通 SQL 即使绕过现有 UDF guard，
修改 fact/current/leaf/internal node/root/count/authenticated root 也不能成为权威。

攻击测试覆盖 active index+sticky row 同删、retirement fact/current/leaf/node/root/count/auth-root、
跨 Session authority、SQLite reserve-vs-attach 并发序列化、restart/rollback/re-put，以及 Memory
等价 contract。审查两项 RED 与 generation contract 转为 `6 passed`；新增投影/跨 Session 攻击
`8 passed`；focused lifecycle/race `14 passed`。最终 wide 为
`287 passed in 370.08s`，E2E 为 `39 passed in 38.74s`；Ruff lint/format 与 mypy `src`
通过；CPython 3.11 全仓为 `1290 passed, 2 skipped in 468.07s`。

## Fix round 4/5：新 Artifact 的空路径认证与 transition 上限

本轮从 `1460392` 接手，只处理两个开放 Important 及其相邻容量边界。RED 为
`165 failed, 6 passed in 170.22s`：每层 orphan/empty digest-count/超容量节点都可被正常
新 reservation 吸入；删除旧路径节点也可被覆盖修复；reserve、complete、abort、re-put
四条追加路径在 `fact_count == MAX` 时都会错误提交 `MAX + 1`。另外的 frontier RED 为
`3 failed, 3 passed in 12.98s`，确认远端孤儿、树范围外节点和 allocation index gap 也未拒绝。

新 artifact 现在必须在同一 `BEGIN IMMEDIATE` 内证明 `artifact_index == artifact_count`
对应的叶子在旧 root 下为空。每一层验证旧节点和 sibling、连续分配前缀对应的精确 subtree
count，以及最终 root/count；缺失的空子树使用 canonical empty digest/count，空父节点也
按 canonical empty 规则折叠。显式节点必须满足该层容量和 empty digest/count 一致性。
allocation 还检查 `artifact_index` 索引的末值，并按每层 `(level, node_index)` 索引执行
`LIMIT 1` 尾部探测，拒绝藏在缺失空子树根之下的孤儿和越界节点。验证不依赖全表扫描，
固定 32 层证明与 33 次有界索引探测使用常量内存；`_set_retirement_leaf` 只在认证后写入。

所有 retirement transition 共用的 append 入口在任何 retirement 写入前拒绝
`fact_index >= _MAX_ARTIFACT_FACTS_PER_LOOKUP`。`MAX - 1 -> MAX` 仍可提交并重启读取，
`MAX -> MAX + 1` 稳定返回 `INVARIANT_VIOLATION`，事务内已暂写的 Session、attachment
facts/index 也一起回滚。重复 reserve/complete/abort 不追加事实，保持幂等。相邻树容量
测试还发现最后一个合法叶子会使旧 meta 验证器自拒绝：现在允许 `artifact_count == capacity`
读取及更新已有叶子，只在下一次新分配前拒绝。

新增 `test_sqlite_retirement_boundaries.py` 覆盖 0..31 每层空路径攻击、0..32 每层旧路径
缺失、root/count/meta、远端孤儿和 index gap、完整数据库快照零变化、重复拒绝/reopen、
四类 transition 的小 limit、两新 artifact 并发分配，以及小树满容量后的读取/完成。
现有 attachment facts/index 双向核对、generation、restart、saga 和 lookup 上限保持不变。

最终验证：

| 检查 | 结果 |
| --- | --- |
| 新 retirement 边界 targeted | `177 passed in 285.63s` |
| schema/SQLite/shared journal/Memory/retention/lifecycle/recovery wide | `304 passed in 698.93s` |
| Artifact/Agent/bounded growth/load E2E | `23 passed in 54.87s` |
| Ruff `check --no-cache src tests` | `All checks passed!` |
| Ruff `format --check src tests` | `95 files already formatted` |
| mypy `src` | `Success: no issues found in 59 source files` |
| CPython 3.11 全仓 | `1467 passed, 2 skipped in 891.03s` |
| `git diff --check` | 通过，仅 LF/CRLF 提示 |

全仓两个 skip 均来自 `test_filesystem_artifacts.py:44`：
`Windows symlink privilege is unavailable`。测试使用真实
`cpython-3.11.16-windows-x86_64-none/python.exe` 与 worktree `.venv/Lib/site-packages`；
全仓命令为 `python -m pytest -q -p no:cacheprovider --basetemp .superpowers/c4-full --tb=short -rs`。
wide 覆盖 `test_journal_contract.py`、`test_sqlite_journal.py`、`test_memory_journal.py`、
`test_artifact_retention.py`、`test_artifact_lifecycle.py` 和 `test_recovery_v3.py`；
E2E 覆盖 integration 的 `test_artifact_flow.py`、`test_agent_v3.py`、`test_bounded_growth.py`
以及 `tests/load/test_shared_agent_load.py`。原始输出保留于 `.superpowers/c4-{red,frontier-red,
green,wide,e2e,full}.log`。本轮未修改 attachment 重放/index 双向核对或 saga/generation 状态机。
