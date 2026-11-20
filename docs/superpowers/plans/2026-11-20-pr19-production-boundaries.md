# PR #19 生产边界加固实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复真实 LiteLLM、MCP、长上下文和运行失败条件下的合并阻塞问题。

**Architecture:** 在 adapters 边界规范化第三方对象，在 context provider 内实施多维预算与连续摘要，在状态机入口隔离 provider tool-call identity，并由 application runtime 保证任何推进失败都落到终态。core 保持精简且不依赖具体基础设施。

**Tech Stack:** Python 3.11, asyncio, LiteLLM 1.108.1, MCP-compatible values, pytest

**Spec:** `docs/superpowers/specs/2026-11-20-pr19-production-boundaries-design.md`

## Global Constraints

- 所有修改只发生在 `F:\CodeSpace\harness-worktrees\lean-core-context-artifacts`。
- 不恢复旧 API，不向 `src/core` 引入 adapter/application 依赖。
- LiteLLM 普通 tools 使用 `_skip_mcp_handler=True`，不新增 `mcp` 运行依赖。
- 每个行为修改先写失败测试并观察预期失败。
- 不运行本机已知会触发 `0xc0000142` 弹窗的 `ruff.exe`；静态检查以 CI 或安全 Python 路径执行。

## Review Focus

- 真实第三方对象的 `None`、Pydantic/dataclass 和可选字段形状。
- 历史在 token 足够但消息数或字节数超限时的裁剪行为。
- durable summary 与新 compaction slice 之间存在空洞时不得宣称覆盖。
- provider 在不同 step/run 重复 tool-call ID 时的状态与历史一致性。
- factory/advance/journal failure 与取消并发时仍只能产生一个合法终态。

---

### Task 1: LiteLLM 真实对象兼容

**Files:**
- Modify: `src/adapters/model/litellm.py`
- Test: `tests/unit/adapters/model/test_litellm.py`

**Interfaces:**
- Produces: normal/stream response parser accepting `tool_calls=None`
- Produces: adapter-owned `_skip_mcp_handler=True`

- [ ] 用真实 LiteLLM response 类型添加流式、非流式和带 tools 的失败测试。
- [ ] 运行聚焦测试，确认分别因 `None` 和 MCP handler 导入失败。
- [ ] 实现最小规范化与 adapter-owned 参数。
- [ ] 运行完整 LiteLLM adapter 测试。

### Task 2: MCP 结果与 refresh 生命周期

**Files:**
- Modify: `src/adapters/tools/mcp.py`
- Test: `tests/unit/adapters/tools/test_mcp.py`

**Interfaces:**
- Produces: JSON-only MCP result envelope
- Produces: refresh-safe transport termination callback

- [ ] 添加 `CallToolResult`/Pydantic/dataclass、`isError` 文本和 refresh 后终止测试。
- [ ] 运行测试并观察当前失败。
- [ ] 实现递归 JSON 规范化和 client-identity lifecycle 绑定。
- [ ] 运行完整 MCP 测试。

### Task 3: 上下文预算、摘要连续性与工具调用身份

**Files:**
- Modify: `src/adapters/context/bounded.py`
- Modify: `src/adapters/context/summarization.py`
- Modify: `src/core/run.py`
- Modify: `src/core/run_projection.py`
- Test: `tests/unit/adapters/context/test_bounded_context.py`
- Test: `tests/unit/adapters/context/test_summarization.py`
- Test: `tests/unit/core/test_run_coordinator.py`
- Test: `tests/integration/test_bounded_growth.py`

**Interfaces:**
- Produces: history selection bounded by token/message/byte limits
- Produces: only contiguous summary coverage
- Produces: run/step-scoped internal tool-call IDs

- [ ] 添加消息数、字节数和摘要空洞回归测试并观察失败。
- [ ] 实现统一候选窗口有效性判断和连续 compaction 选择。
- [ ] 添加同一/跨 Run 重复 provider ID 测试并观察失败。
- [ ] 在状态机入口派生内部 ID，运行 context/core/integration 测试。

### Task 4: 运行失败终态与 Artifact 读回预算

**Files:**
- Modify: `src/application/runtime.py`
- Modify: `src/application/composition.py`
- Modify: `src/adapters/tools/read_artifact.py`
- Test: `tests/unit/application/test_runtime_v3.py`
- Test: `tests/integration/test_production_composition.py`
- Test: `tests/unit/adapters/tools/test_read_artifact.py`

**Interfaces:**
- Produces: non-cancellation drive failures committed as terminal failures
- Produces: encoded read result within inline result budget

- [ ] 添加 factory、advance、journal bound、open-tool overflow 的终态测试并观察失败。
- [ ] 实现统一失败收口和 `max_steps`/advance 配置联合校验。
- [ ] 添加满尺寸二进制读回不二次 externalize 的测试并观察失败。
- [ ] 以编码后大小裁剪 read result，运行 runtime/composition/artifact 测试。

### Task 5: 交叉验证与整分支审查

**Files:**
- Modify only if verification exposes a regression.

**Interfaces:**
- Consumes: all prior task outputs
- Produces: focused and full-suite evidence

- [ ] 运行模型、MCP、上下文、runtime、production composition 聚焦测试。
- [ ] 运行锁定依赖下的完整 `pytest`。
- [ ] 检查 diff、类型/格式和工作区状态。
- [ ] 对照本设计逐项复核，修复 Critical/Important 回归后再提交。
