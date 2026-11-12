# Task State Pydantic Design

## 意图

将 Harness 的任务运行状态统一为可验证、可序列化、可恢复的 Pydantic 模型，消除
`Task`、dataclass、普通类和 Pydantic 混用导致的终态不一致与持久化失败。本阶段只建立
正确的状态契约和进程内仓库语义，为后续 durable repository、sandbox 和 async runtime
提供稳定基础。

## 目标

- `Task`、`TaskResult`、`Message`、`ToolExecutionResult` 及其嵌套状态均可通过
  `model_dump(mode="json")` 生成 JSON-safe 数据，并可通过 `model_validate()` 完整恢复。
- 终态满足严格不变量：`COMPLETED` 和 `FAILED` 必须拥有 `TaskResult`；`TaskResult.output`
  始终为 `str`。
- 每次完成、失败、暂停、澄清和异常收尾都先更新完整任务状态，再保存同一份状态。
- `step_count` 与 `elapsed_seconds` 在暂停、完成和失败时都准确记录。
- `InMemoryTaskRepository` 使用快照语义，保存后调用方继续修改原对象不会隐式改变仓库内容，
  读取结果也不会暴露仓库内部可变对象。
- 保持现有同步 Harness、审批流程、事件顺序和 CLI 用户体验不变。

## 非目标

- 本阶段不实现数据库、文件或远程持久化。
- 不引入事件溯源、乐观锁或 schema migration 框架；仅预留 `schema_version` 字段。
- 不处理 sandbox、MCP 异步化、任务取消或多会话并发。
- 不拆分 `Harness` 大类，也不调整项目打包方式。

## 约束与不变量

- `TaskState.TASK_COMPLETED`：`result` 必须存在，`result.output` 为字符串。
- `TaskState.TASK_FAILED`：`result` 必须存在，并保留面向调用方的错误或收尾文本。
- `TaskState.TASK_CREATED`、`TASK_RUNNING`、`TASK_PAUSED`：允许 `result=None`。
- `state_reason` 必须与状态类别匹配；完成态清除旧的 running/pause reason。
- `step_count >= 0`，`elapsed_seconds >= 0`。
- 模型默认值使用 `Field(default_factory=...)`，不同任务之间不得共享可变对象。
- 仓库接口仍接收和返回 `Task`，以限制本阶段对调用方的影响。

## 方案比较

### 方案 1：渐进式显式序列化

保留现有类型，只为每种类型补充 `to_dict/from_dict`。改动较小，但会继续维护多套验证与序列化
规则，新增字段时容易遗漏。

### 方案 2：统一 Pydantic 模型（已选择）

将任务状态图中的数据对象统一为 Pydantic，利用同一套验证、JSON schema、序列化和恢复机制。
迁移面较大，但能一次消除当前持久化和类型不变量问题，并为后续 durable repository 提供清晰边界。

### 方案 3：事件溯源与版本化 checkpoint

保存不可变事件流并重放任务。长期能力最强，但当前缺少真实持久化后端和并发运行时，现阶段会引入
不必要的复杂度。

## 模型设计

### `src/schema/context.py`

`Message` 与 `Context` 改为 `BaseModel`。消息中的 `tool_calls`、`thinking_blocks` 继续允许当前
LiteLLM 适配器所需的宽松结构，但所有外层容器都由 Pydantic 复制和序列化。

### `src/schema/llm.py`

`LlmToolCall`、`LlmUsage`、`LLMOutput` 改为 `BaseModel`。流式合并仍创建新的 `LLMOutput`，不改变
当前调用协议。

### `src/schema/tool.py`

`Tool` 包含不可序列化 callable，不属于任务快照；它保留运行时类型。`ToolExecutionResult` 与
`ToolCallResult` 改为 Pydantic 模型，因为它们属于暂停和恢复状态。`ToolExecutionResult.output`
使用 `Any`，序列化时若工具返回不可 JSON 化对象应明确失败，不静默转换为字符串。

### `src/schema/task.py` 与 `src/core/task.py`

`TaskPlan` 保持 Pydantic。`TaskResult` 改为 Pydantic，`output: str = ""`。

`Task` 继续位于 `src/core/task.py`，但继承 `BaseModel`，保留现有行为方法和状态转换 API。
新增：

- `schema_version: int = 1`
- 对终态结果、reason 类型、非负进度的 model validator
- `to_dict()` 作为 `model_dump(mode="json")` 的兼容包装
- `from_dict()` 作为 `model_validate()` 的明确恢复入口

内部状态转换先完成字段更新，再运行不变量检查。为避免 running 中间态被终态 validator 阻断，
终态 result 在转换前创建。

## Harness 状态写入顺序

所有终止路径采用统一顺序：

1. 更新 `step_count`、`elapsed_seconds`。
2. 创建并写入最终 `TaskResult`。
3. 转换状态。
4. 保存任务快照。
5. 发送对应事件。
6. 返回 `task.result`。

澄清仍保持当前“完成本轮任务”的外部语义，但必须保存 completed 快照。将澄清改为
`WAITING_INPUT` 留给后续会话状态设计。

异常路径把当前公开错误文本写入 `task.result` 后再保存，确保调用返回值、Task 对象和仓库快照一致。

## 仓库快照语义

`InMemoryTaskRepository.save(task)` 保存 `task.model_dump(mode="json")`；`get(task_id)` 使用
`Task.model_validate()` 返回新对象。这样可在不引入数据库的情况下验证真实持久化边界，并避免引用
共享掩盖遗漏的 `save()` 调用。

## 兼容性

- 保留 `Task(...)`、`TaskResult(...)`、`to_dict()`、状态判断和状态转换方法。
- `Harness.run()` 恢复为 `TaskResult` 返回类型；终态无结果被视为无效状态，而不是向调用方传播
  `None`。
- 工具 callable、LLM gateway、ContextBuilder 和 EventSink 接口不在本阶段改变。
- 旧的非 JSON-safe `to_dict()` 输出不作为兼容契约保留。

## 测试策略

- 先增加 JSON round-trip 测试，证明包含 history、pending tool calls、plan、result 和进度的任务可恢复。
- 增加终态不变量和非法进度验证测试。
- 为完成、失败、澄清、异常路径增加“返回值 = Task.result = repository snapshot”的行为测试。
- 增加仓库快照隔离测试。
- 每个生产改动先观察对应测试失败，再实施最小代码。
- 完整门禁：pytest、ruff format、ruff check、mypy。

## 后续阶段

本阶段完成后，下一优先级依次为：workspace/sandbox 安全边界、async-first Harness 与 MCP 接入、
deadline/cancellation、Session/TaskRun 并发隔离、memory/skills/observability。
