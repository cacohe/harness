# PR 19 Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 PR #19 新增评审确认的所有正确性、安全性、资源边界和生产组合缺陷，同时保持 `src/core` 精简。

**Architecture:** Core 只强化 deadline、状态和端口契约；上下文选择、恢复、Artifact 生命周期、SQLite、MCP、LiteLLM 和默认工具组合继续位于 application/adapters。所有行为先用真实边界测试复现，再做最小实现修复。

**Tech Stack:** Python 3.11、asyncio、SQLite、pytest、uv、ruff、mypy。

**Spec:** `docs/superpowers/specs/2026-11-17-lean-core-context-artifacts-design.md`

## Global Constraints

- `src/core` 不导入 application、adapters 或具体基础设施。
- 不保留旧 API 或 SQLite 旧 schema 兼容层。
- 所有随会话增长的数据结构和读取成本必须有显式上限。
- Artifact 本体不进入 RunState、journal event 或模型上下文。
- 每项生产代码修改先有失败测试并观察 RED。

## Review Focus

- caller cancellation 发生在 model/tool/commit 任意 await 边界时，Session 必须可再次提交。
- 不可信 MCP server 不得通过 annotations 或名称碰撞改变审批语义。
- SQLite busy/locked、commit 后返回值构造失败和大 receipt 不得伪装为 invariant corruption。
- Artifact 删除与并发重新附加不得产生 durable dangling reference。
- 10,000 条历史和任意合法 Artifact ID 探测不得触发无界扫描或永久文件增长。

---

### Task 1: Context correctness and bounded reads

**Files:**
- Modify: `src/application/submissions.py`
- Modify: `src/application/composition.py`
- Modify: `src/adapters/context/bounded.py`
- Modify: `src/adapters/context/summarization.py`
- Test: `tests/unit/adapters/context/`
- Test: `tests/integration/test_agent_v3.py`
- Test: `tests/integration/test_bounded_growth.py`

- [ ] 写真实 composition 测试，复现当前输入重复、开放 tool batch 丢失、memory/skills/summary 饥饿、摘要不累积和少量 compaction 全表扫描。
- [ ] 运行上述测试并观察预期失败。
- [ ] 明确 transcript cursor 与 `current_input` 的去重所有权，生产 request factory 填充开放工具关系；改为确定性优先级和硬分页上限；摘要纳入上一个 checkpoint。
- [ ] 运行 context 和集成测试至通过。

### Task 2: Recovery and model execution boundaries

**Files:**
- Modify: `src/core/run.py`
- Modify: `src/application/agent.py`
- Modify: `src/application/runtime.py`
- Modify: `src/application/composition.py`
- Modify: `src/adapters/model/litellm.py`
- Test: `tests/unit/core/test_run_coordinator.py`
- Test: `tests/unit/application/`
- Test: `tests/unit/adapters/model/test_litellm.py`
- Test: `tests/integration/test_recovery_v3.py`

- [ ] 写中断后可恢复提交、RunLimits model deadline、finish reason 截断、错误分类和 output reservation 一致性失败测试。
- [ ] 观察 RED 后实现启动恢复、当前 run 查询/取消边界、core deadline、LiteLLM 结束原因和错误分类。
- [ ] 运行相关测试至通过。

### Task 3: Artifact identity, retirement, locking and production lifecycle

**Files:**
- Modify: `src/adapters/artifacts/filesystem.py`
- Modify: `src/adapters/artifacts/memory.py`
- Modify: `src/adapters/persistence/memory.py`
- Modify: `src/adapters/persistence/sqlite.py`
- Modify: `src/application/artifact_retention.py`
- Modify: `src/application/composition.py`
- Test: `tests/unit/adapters/artifacts/`
- Test: `tests/unit/application/test_artifact_retention.py`
- Test: `tests/unit/adapters/persistence/test_sqlite_retirement_boundaries.py`

- [ ] 写 media type 身份冲突、`stat` lock 泄漏、删除/重新附加竞态、阈值不一致和 orphan GC 组合测试并观察 RED。
- [ ] 将 blob identity 与 descriptor 元数据分离；只为实际对象操作创建可回收 lock；retirement 使用 generation/CAS，不静默复活。
- [ ] 接入 retention/orphan GC/ReadArtifactTool 并运行相关测试。

### Task 4: SQLite atomicity, retry taxonomy and hot-path cost

**Files:**
- Modify: `src/adapters/persistence/sqlite.py`
- Modify: `src/adapters/persistence/sqlite_codec.py`
- Test: `tests/unit/adapters/persistence/test_sqlite_journal.py`

- [ ] 写大 receipt 原子性、busy/locked 分类和提交/查询不重放全链测试并观察 RED。
- [ ] 在 commit 前构造并验证 receipt；区分 retryable operational failure；将全链校验移到显式 integrity API，热路径只校验局部锚点。
- [ ] 运行 SQLite 完整性与并发测试。

### Task 5: MCP trust boundary and real adapter lifecycle

**Files:**
- Modify: `src/adapters/tools/mcp.py`
- Modify: `src/adapters/tools/catalog.py`
- Modify: `pyproject.toml` only if a real transport dependency is required
- Test: `tests/unit/adapters/tools/test_mcp.py`
- Test: `tests/integration/` for real transport lifecycle

- [ ] 写 annotation 绕过、稳定 workspace、供应商合法名称、context-manager client、timeout cancellation 和 `isError` 归一化测试并观察 RED。
- [ ] 默认所有远端工具需要审批，仅本地可信策略可降级；使用稳定 schema identity；拥有 client lifecycle；超时取消底层 task；统一 MCP result。
- [ ] 运行工具测试和真实 transport integration test。

### Task 6: Production composition, cleanup and full verification

**Files:**
- Modify: `src/application/composition.py`
- Modify: `src/adapters/tools/__init__.py`
- Modify: `README.md`
- Delete: tracked `.superpowers/sdd/**` reports
- Test: `tests/integration/test_project_layout.py`
- Test: production composition integration tests

- [ ] 写 production builder 的 recovery、router、ReadArtifactTool、retention、orphan GC 和配置交叉校验测试。
- [ ] 实现 adapter 层默认组合与可运行入口，不向 core 增加具体能力。
- [ ] 删除会话报告并保留 `.superpowers/` ignore。
- [ ] 运行 `ruff format --check`、`ruff check`、`mypy src`、完整 pytest、架构与规模检查。

