# 可靠运行时基础设计

## 背景与意图

当前 Harness 已具备异步优先的 Agent 循环、Pydantic Task 状态机、暂停审批、SQLite Task 快照、MCP、Sandbox、长期记忆和进程内指标。现有持久化回答“Task 当前是什么状态”，但不能完整回答“执行如何到达该状态”，也不能可靠地区分进程中断、调用方重试、模型调用失败、工具副作用状态不明和观测系统失败。

本设计在保留 Python API、Task 模型和现有组合方式的前提下，引入持久执行事实、严格生命周期、恢复、幂等提交、资源治理、统一关闭和可复现 Profile。目标是提升生产可靠性，而不是将项目改写为通用事件溯源框架。

## 已选方案

采用增量混合迁移：持久事件日志作为执行事实，Task 快照作为查询模型和 checkpoint；SQLite 在同一事务中追加事件并更新 Task 快照。现有 `chat`、`achat`、`resume`、`aresume` 和 `TaskRepository` 调用保持兼容，新能力通过新的执行存储端口逐步接入。

### 考虑过的替代方案

1. **纯事件溯源重写**：Task 完全由事件重建。模型最统一，但会集中重写主循环、审批恢复和全部持久化测试，迁移风险过高。
2. **外层 Journal 包装**：不修改主循环，只记录已有 EventSink 输出。实现快，但无法保证事实事件与 Task 快照原子一致，也不能建立严格结算不变量。
3. **增量混合迁移（已选）**：修改核心提交边界，但保留 Task 快照和公开 API；能够逐阶段验证，并允许旧的自定义 TaskRepository 继续工作。

## 目标

- 为 Session、Turn、Step、ModelAttempt 和 ToolCall 建立稳定身份和严格生命周期。
- 持久化可重放、可审计、按 Session 严格排序的执行事实。
- 在 SQLite 中原子提交事实事件、Task 快照和消息提交状态。
- 进程重启后识别并结算中断执行，不自动重放状态不明的副作用。
- 让调用方以稳定 `message_id` 安全重试，不重复执行模型或工具。
- 将持久事实、执行观测和流式输出明确分离。
- 对时间、队列和外部载荷施加分层预算。
- 让 Agent 统一拥有活动运行和外部资源的关闭生命周期。
- 用自动 gate 维护 schema、core、adapter 和 composition root 的依赖方向。
- 持久记录可解释历史行为的 Agent Profile 与能力目录摘要。

## 非目标

- 不以 JSONL 替换 SQLite。
- 不提供跨节点分布式锁、共识或多主写入。
- 不保证任意外部工具副作用的 exactly-once；框架只保证自身事实结算 exactly-once。
- 不自动重试状态不明的工具调用。
- 不把项目拆成多个 Python distribution。
- 不在本项目中实现 OpenTelemetry exporter、远程数据库或多租户控制面。
- 不改变 Memory、MCP 或 Sandbox 的权限边界。

## 约束与原则

- 保持 Python 3.11、Pydantic、uv 和当前异步优先接口。
- 同一 Session 的持久事件只有一个逻辑 writer，序号严格单调。
- 持久事实必须先于依赖该事实的后续阶段；外部副作用前必须持久化足够信息，使恢复时能够拒绝静默重放。
- Task 快照不是审计来源；事件日志不是查询 API 的替代品。
- 持久化失败是边界失败，不能伪装成普通 Task 业务失败。
- EventSink、metrics 和客户端输出保持 best-effort，不参与事实事务。
- 所有不可信 JSON、数据库内容、工具载荷和外部 adapter 返回值在核心边界校验。

## 总体架构

```text
Agent / AgentSession / Host
          |
          v
       Harness  -----------------> ObservationSink / stream
          |
          v
    ExecutionStore
      |       |
      |       +------------------> Task snapshot read model
      +--------------------------> append-only session facts
          |
          v
 SQLiteExecutionRepository / InMemoryExecutionRepository
```

新增持久事实不会复用现有 `schema.event.EventType`。现有 Event/EventSink 继续表示客户端通知和 tracing 观测；新的 `SessionEvent` 只表示能够恢复执行状态的事实。

## 1. 持久事件日志与 Task 快照

### 事件 Envelope

在 `src/schema/execution.py` 定义冻结的 Pydantic 值对象：

```python
class SessionEvent(BaseModel):
    schema_version: int
    event_id: str
    session_id: str
    sequence: int
    type: SessionEventType
    task_id: str
    execution_id: str
    turn_id: str | None
    step_id: str | None
    attempt_id: str | None
    tool_call_id: str | None
    occurred_at: datetime
    payload: dict[str, JsonValue]
```

调用方创建的是没有 `sequence` 的 `PendingSessionEvent`；Repository 在提交事务中分配连续序号并返回已持久化 Envelope。载荷使用严格 JSON 值，拒绝 NaN、Infinity、非字符串 key 和隐式字符串化对象。

### 初始事实类型

- `session.started`
- `submission.accepted`
- `execution.started`
- `turn.started`
- `user.message`
- `step.started`
- `model.attempt.started`
- `model.attempt.settled`
- `assistant.settled`
- `tool.call.requested`
- `tool.call.started`
- `tool.call.settled`
- `step.paused`
- `step.resumed`
- `step.settled`
- `turn.paused`
- `turn.resumed`
- `turn.settled`
- `submission.paused`
- `submission.resumed`
- `submission.settled`
- `execution.settled`
- `profile.changed`
- `session.interrupted`

settled 事件的 payload 统一包含 `outcome`。Execution 可以以 `completed`、`failed`、`cancelled`、`paused` 或 `interrupted` 结算；Turn、Step、ModelAttempt、ToolCall 和 Submission 的最终 outcome 不包含 `paused`，暂停使用独立状态转换事件表达。错误只保存稳定分类、公开消息和可选错误类型，不持久化宿主绝对路径、凭据或任意异常对象。

### 执行存储端口

在 `src/core/persistence.py` 新增 `ExecutionStore` Protocol：

```python
def commit(
    self,
    task: Task,
    events: Sequence[PendingSessionEvent],
    *,
    expected_sequence: int | None,
    submission: SubmissionRecord | None = None,
) -> list[SessionEvent]: ...


def events(self, session_id: str, *, after_sequence: int = 0) -> list[SessionEvent]: ...
def submission(self, session_id: str, message_id: str) -> SubmissionRecord | None: ...
def unfinished(self, session_id: str | None = None) -> list[ExecutionRecord]: ...
```

`ExecutionCommitError` 保留事务失败原因，`SequenceConflictError` 表示并发 writer 或过期状态。Core 不直接导入 SQLite。

`expected_sequence` 精确定义为调用方最后观察到的最高已提交 sequence；空 stream 为 `0`。Repository 仅在当前最高 sequence 等于该值时提交，并将新事件依次编号为 `expected_sequence + 1...n`。`None` 只允许数据库迁移和旧 TaskRepository 兼容 adapter 禁用 CAS；Core 的正常提交始终传入整数。

### 兼容旧 TaskRepository

- `TaskRepository.save/get` 保留。
- 新的 InMemory 和 SQLite repository 同时实现 TaskRepository 与 ExecutionStore。
- 注入只实现旧 TaskRepository 的自定义对象时，由 `SnapshotExecutionStore` 适配：继续保存 Task，并在进程内保存事实事件；它明确不提供跨进程恢复和事务幂等。
- 新代码不使用 `hasattr` 猜测能力，而以 `runtime_checkable ExecutionStore` 做结构化判断。
- README 标记旧 TaskRepository-only 注入为兼容路径，并建议生产实现迁移到 ExecutionStore。

## 2. 严格生命周期与结算

### 身份层级

```text
session_id
  submission/message_id
    turn_id
      step_id
        attempt_id
        tool_call_id
  execution_id -> 本次 API 调用所处理的 turn_id
```

- `session_id` 标识一条有序对话。
- `execution_id` 标识一次 `achat` 或 `aresume` 调用；暂停后的 resume 创建新 execution_id，但继续原 turn。
- `turn_id` 标识一条用户输入的完整处理；暂停后 resume 继续原 turn。
- 每次模型推理创建新的 `step_id` 和 `attempt_id`。
- 模型提供的 tool-call ID 作为外部 ID保留，同时生成框架范围内稳定的关联 ID；二者都进入事实载荷。

### 生命周期规则

- 每个 `*.started` 必须有且只有一个对应 `*.settled`。
- `model.attempt.started` 始终由独立的 `model.attempt.settled` 闭合。模型正常结束时先结算 attempt，再持久化 `assistant.settled`；模型失败、取消或超时时只结算 attempt 和上层生命周期，不伪造 assistant 消息。
- assistant 的流式 delta 不是事实；完整 assistant 消息只在 `assistant.settled` 中持久化一次。
- 模型公布工具调用时持久化 `tool.call.requested`。只有真正执行副作用前才持久化 `tool.call.started`；被策略拒绝的调用可以从 requested 直接进入 settled/rejected。调用开始后，无论成功、失败、取消还是超时，都持久化一个 `tool.call.settled`。
- ToolCall 已开始但没有 settled 时，恢复结果是 `interrupted/unknown`，绝不自动重新调用。
- Task 暂停审批时追加 step/turn/submission 的 `paused` 状态转换，并以 `execution.settled(outcome=paused)` 结束本次 API 调用。Step、turn 和 submission 保持打开且没有最终 settled；resume 追加对应 `resumed` 事实，创建新 execution_id 并继续原 turn，最终只产生一个 step/turn/submission settled。
- 终态持久化失败优先暴露为边界错误，并以原执行错误作为 cause。

### Core 组织

主循环仍由 `Harness` 负责，但把以下逻辑提取成高内聚模块：

- `ExecutionContext`：冻结身份、sequence 和 deadline。
- `FactRecorder`：构造、校验并提交事实与快照。
- `LifecycleState`：跟踪已经开始/结算的 turn、step、attempt 和 tool call。
- `PayloadLimiter`：在持久化前执行统一字节预算。

这些模块位于 `src/core/`，只依赖 schema 和 Protocol，不依赖 adapter。

## 3. 启动恢复与中断修复

新增 `ExecutionRecovery`，从持久事件重建未完成生命周期。恢复执行以下规则：

1. 校验序号连续、event_id 唯一、身份层级一致和生命周期合法。
2. 如果发现结构矛盾，例如同一 step 两个终态，失败关闭并报告 `RecoveryError`，不改写数据库。
3. 如果只是有效尾部中断，在一个事务中追加补偿事实并更新 Task：
   - 未结算 tool call -> `tool.call.settled(outcome=interrupted, result=unknown)`；
   - 未结算 attempt/step -> 对应 interrupted settlement；
   - 未结算 turn/submission -> interrupted settlement；
   - running Task -> failed/cancel 类之外新增稳定的 interrupted 失败原因。
4. 补偿是幂等的；重复恢复不追加第二组终态。
5. 已暂停等待审批的 Task 不视为中断，保留 resume 所有权。

`Agent.open_session(session_id)` 在暴露 Session 前恢复并读取 transcript；默认 `new_session()` 仍创建新 ID。直接调用 `Harness.arun(Task)` 时使用 `task:<task_id>` 作为兼容 stream identity。

## 4. 消息提交幂等性

`AgentSession.chat/achat` 增加可选 keyword-only `message_id`：

```python
await session.achat("生成报告", message_id="request-001")
```

未传入时生成 UUID，保持现有调用方式。提交记录包含：

- session_id
- message_id
- 规范化后的原始消息摘要（SHA-256）
- task_id
- turn_id
- 状态：accepted、paused、completed、failed、cancelled、interrupted
- 创建和更新时间

规则：

- 相同 session/message_id 和相同摘要返回已有 Task 快照，不再次运行模型或工具。
- 相同 ID、不同摘要抛出 `SubmissionConflictError`。
- accepted 记录与 `submission.accepted`、初始 Task 快照同事务提交。
- 暂停后的重复提交返回 paused Task；调用方必须通过该 Session resume。
- 并发相同提交由数据库 unique key 和 expected_sequence 仲裁，失败方重新读取已存在提交。

消息摘要协议命名为 `sha256-raw-utf8-v1`：直接对 Python `str.encode("utf-8")` 的字节计算 SHA-256，不执行 strip、换行转换或 Unicode normalization。提交记录保存算法版本。未来结构化消息使用新的 canonical-JSON 协议版本，不能静默改变 v1 规则。

## 5. 持久事实、观测和流分离

- `SessionEvent`：可靠事实，只通过 ExecutionStore 提交。
- 现有 `Event`/`EventSink`：执行观测和客户端通知，保持 best-effort。
- `LLM_DELTA`：临时流，不持久化。
- MetricsEventSink 只消费观测事件，不读取数据库。
- 事实成功提交后再发出相应观测；observer 失败只累计 failure，不回滚事实。
- 为观测事件补齐 execution_id、turn_id、step_id、attempt_id 和 tool_call_id；旧字段继续保留一个兼容周期。
- 可靠远程 telemetry 应由 Host 实现 outbox；本次不实现网络 exporter。

## 6. 分层资源预算

在 `Config` 和 `Harness` 增加经过校验的预算：

- `max_seconds`：整个 turn 的累计运行预算，沿用现有语义。
- `model_timeout_seconds`：单次模型 attempt deadline。
- `tool_timeout_seconds`：单次工具调用 deadline。
- `max_model_output_bytes`：assistant 文本、thinking 和工具调用 JSON 的累计 UTF-8 大小。
- `max_tool_result_bytes`：所有本地、MCP 和 sandbox 工具结果进入 Task 前的统一上限。
- `max_event_bytes`：单个持久事件编码后的上限。
- `max_task_snapshot_bytes`：Task 快照上限。
- `max_pending_submissions`：可选的 Session FIFO 容量；默认 `0`，继续保持同 Session 重入立即失败。大于 0 时才启用有界排队，排队取消不写入 submission 事实。

超限不截断协议 JSON；生成稳定、可审计的失败结果。文本流可以在达到上限时停止消费，但最终事实只保存预算内的已验证内容和失败分类。

第三方同步 gateway/tool 在线程池运行时无法抢占；deadline 到达后框架停止等待、丢弃晚到结果，并禁止晚到回调追加事实或观测。

## 7. Runtime 生命周期

保留 `Agent` 作为默认 composition root，并在内部增加显式 runtime ownership，不新增面向用户的必选包装层。

### 资源协议

可拥有资源实现可选的异步 `start()`、`aclose()` 和 `flush()` Protocol。Agent 记录由自己创建或明确转交 ownership 的资源；注入对象默认由调用方拥有，除非通过 `owned_resources` 标记。

### 启动

- `async with Agent(...)` 进入时显式 start owned resources。
- 部分启动失败时按相反顺序关闭已经启动的资源。
- 保持当前惰性调用兼容；首次异步操作可以触发一次幂等 start。

### 关闭

1. 原子停止接纳新 Session 操作和 MCP 生命周期变更。
2. 取消所有活动 Session operation。
3. 在独立 `close_timeout_seconds` 内等待严格结算。
4. flush 事实存储和观测 sink。
5. 逆序关闭 MCP、repository、sink 和其他 owned resources。

`aclose()` 幂等并受 shield 保护。调用方取消等待不会取消后台清理。关闭后 transcript 和历史审批仍可只读，不能产生新事实。

关闭等待超过 `close_timeout_seconds` 时，首次 `aclose()` 抛出 `RuntimeCloseTimeoutError`，Agent 保持 `closing` 并继续一个受 shield 保护的后台清理任务；事实存储在活动操作完成结算前保持打开。后续 `aclose()` 继续等待同一任务。开始关闭后不再接纳新 submission；只有关闭任务真正完成后才保证不会再追加清理事实。同步第三方调用的晚到结果仍被丢弃，因此后台清理不等待其线程自然退出。

## 8. 架构依赖 Gate

延续现有单一 Python project，不拆包。自动 gate 扩展为：

```text
schema <- core <- adapters/runtime composition <- demo/hosts
```

具体规则：

- `src/schema` 不得导入 `src.core`、adapter、demo 或外部 SDK。
- `src/core` 不得导入 LiteLLM、FastMCP、SQLite、Docker、具体 memory/tool adapter 或 demo。
- `src/persistence`、`src/llm`、`src/mcp`、`src/sandbox`、`src/tool`、`src/memory` 实现 core Protocol。
- `src/demo/agent.py` 是默认 composition root，可以导入所有下层。
- 禁止循环导入、相对路径绕过和延迟字符串构造绕过明显依赖。
- 架构 gate 使用 AST，覆盖 `import`、`from ... import` 和常见动态 import；误报场景通过明确 allowlist 记录原因。

## 9. Agent Profile 与可复现性

新增冻结 `AgentProfile`：

- `id`
- `version`
- system prompt digest
- model identifier 与公开参数摘要（不含 key/base credential）
- Skill 名称、版本或内容 digest
- 静态工具目录 digest
- memory policy version
- Harness 事实 schema version

`session.started` 保存完整 Profile snapshot。每个 `turn.started` 额外保存当时动态能力摘要，包括 MCP 工具变更后的目录 digest。Profile 不保存密钥、绝对路径或未脱敏环境变量。

默认 Agent 从已解析配置和能力目录确定性构造 Profile；调用方也可注入显式 Profile。打开既有 Session 时，历史 Profile 只用于审计，不覆盖当前运行配置；如果当前 Profile 与历史不一致，追加 `profile.changed` 事实后才能接受新 turn。

## 10. 稳定关联 ID

新增可注入 `ExecutionIdFactory`，生产默认 UUID，测试使用顺序 ID。所有 ID 在进入事件前校验为非空、长度有界的字符串。

- schema、数据库、Task 和 Event 使用相同关联 ID。
- EventSink 输出和 Metrics key 使用 execution/turn/step/attempt/tool identity，避免仅以 task_id + phase 合并重试。
- ID 一旦分配即保留；prepare/validation 在不产生执行事实前失败时不消耗下级生命周期 ID。
- 外部 tool-call ID 与内部 ID 分开存储，避免不同模型 provider 的格式差异污染内部身份。

## SQLite Schema v2

从 user_version 1 迁移至 2，迁移在 `BEGIN IMMEDIATE` 中完成：

```sql
CREATE TABLE session_streams (
    session_id TEXT PRIMARY KEY,
    next_sequence INTEGER NOT NULL CHECK (next_sequence >= 1)
);

CREATE TABLE session_events (
    session_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    event_id TEXT NOT NULL UNIQUE,
    task_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

CREATE TABLE submissions (
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    digest_algorithm TEXT NOT NULL,
    message_digest TEXT NOT NULL,
    task_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, message_id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);
```

`session_events.payload` 保存完整 Envelope 的可变部分，索引列只保存查询和约束需要的字段。读取后必须重新进行 Pydantic 校验。Schema v1 数据迁移时只保留已有 Task 快照，不伪造过去的详细执行历史；第一次访问旧 Task 时可以追加明确的 `legacy.snapshot.imported` 事实。

迁移必须：

- 支持空 v1 和已有 Task 的 v1 数据库。
- 失败时整体回滚，保持 user_version 1 可继续由旧代码读取。
- 拒绝未知更高版本。
- 验证表、列、主键、外键和必要索引，而不只检查 user_version。

## Task Schema 兼容

Task schema_version 升至 2，新增可选：

- `session_id`
- `message_id`
- `turn_id`
- `last_sequence`

旧快照读取时设置兼容默认值；首次新提交时补齐。Task 的现有状态枚举保持稳定，新增 `TASK_FAILED_INTERRUPTED` 作为失败原因，不新增顶层 TaskState。

快照中的历史消息继续服务模型上下文；持久事件服务恢复和审计。两者不允许独立修改：Core 只能通过 FactRecorder 同时提交。

## 错误语义

- `ExecutionCommitError`：事实或快照无法原子提交。
- `SequenceConflictError`：同 Session 并发 writer。
- `SubmissionConflictError`：相同 message_id 对应不同内容。
- `RecoveryError`：持久历史结构矛盾或无法安全恢复。
- `PayloadLimitError`：模型、工具、事件或快照超过预算。
- `RuntimeClosedError`：关闭后尝试写操作。

这些错误保留 cause，公开消息不包含数据库路径、Workspace 根目录、凭据或完整外部响应。

## 测试策略

每项实现遵循测试先行。新增测试至少覆盖：

### Schema 与状态机

- 事件严格 JSON、冻结和身份层级校验。
- 每种非法生命周期序列。
- Task v1 -> v2 快照迁移。

### Repository

- SQLite v1 -> v2 原子迁移和失败回滚。
- 同事务写 Task、events 和 submission。
- expected_sequence 冲突。
- 多线程/多实例并发相同 message_id。
- 损坏事件、序号缺口、重复终态和未知 schema。
- InMemory 与 SQLite 的共享契约测试。

### Core

- 成功、模型失败、工具失败、审批暂停、拒绝、resume、超时和取消的 started/settled 配对。
- 持久化失败时不继续外部副作用。
- observer 失败不影响事实。
- 晚到线程结果不能发布事件。
- 各级载荷和 deadline 边界。

### 恢复与幂等

- 在每个生命周期边界模拟进程中断。
- 状态不明工具被结算为 interrupted 且不重放。
- 重复恢复不产生新事件。
- 相同 message_id 相同内容不执行第二次。
- 相同 message_id 不同内容拒绝。
- 暂停 Task 重启后由原 Session resume。

### 生命周期与架构

- 部分 start 失败逆序清理。
- close 取消、结算、flush、逆序关闭和调用方取消等待。
- AST dependency gate 的正反例。
- Profile 不泄漏 secret，并在能力变更时产生 profile.changed。

完整 gate 保持：Ruff format、Ruff lint、Mypy、全部 Pytest；负载测试继续验证共享 Agent 下 Session、事实序号、幂等表和指标不会串扰。

## 实施阶段

### 阶段 A：事实模型与存储

- Execution schema、ID factory 和严格 JSON。
- ExecutionStore、InMemory 实现、兼容 adapter。
- SQLite v2 迁移和事务提交。
- Repository contract tests。

### 阶段 B：核心生命周期

- FactRecorder、LifecycleState、ExecutionContext。
- Harness turn/step/attempt/tool 事实接入。
- 事实与 EventSink 分离。
- 资源预算和关联 ID 观测。

### 阶段 C：Session、幂等与恢复

- AgentSession message_id。
- submission 表和并发去重。
- ExecutionRecovery、open_session 和 transcript 重建。
- 暂停/resume 跨重启恢复。

### 阶段 D：Runtime、Profile 与治理

- owned resource lifecycle 和统一 aclose。
- Profile snapshot、动态能力 digest。
- 架构 gate 扩展。
- README、配置示例、迁移说明和 load regression。

每个阶段必须保持公开 API 兼容、通过全量 gate，并产生独立可审查提交。后续阶段不得依赖未通过测试的临时状态。

## 交付与回滚

- SQLite v2 迁移不可由旧代码写入，因此发布说明必须指出数据库升级边界，并建议升级前备份。
- 代码层面通过保留 TaskRepository-only compatibility adapter 降低调用方迁移成本。
- 每个阶段单独提交；若后续阶段出现问题，可回滚代码但不能让 v1 二进制重新写入已升级 v2 的数据库。
- 首次发布将新事实能力标记为稳定，旧 TaskRepository-only 路径标记为兼容但不提供重启恢复保证。

## 验收标准

- 现有公开调用方式和测试保持通过。
- SQLite 事务保证 Task、事实和 submission 不会部分提交。
- 每个已开始的 turn、step、attempt 和 tool call 在所有受测路径上恰好一个终态。
- 任意受测中断点恢复后不存在 running 幽灵状态，也不会自动重放未知副作用。
- 调用方重复提交相同 message_id 不会重复调用模型或工具。
- facts、observations 和 stream 在类型、存储和失败语义上明确分离。
- 时间、输出、工具结果、事件和快照预算均有边界测试。
- Agent 开始关闭后没有新 submission；关闭任务完成后没有新事实，owned resources 被逆序、幂等关闭。首次有界等待超时时会报告错误，但同一后台清理任务继续运行。
- 架构 gate 阻止 Core 导入具体 SDK/adapter。
- Session 和 Turn 事实能够说明使用了哪个 Agent Profile 与工具目录摘要。
- Ruff、Mypy、506 项既有测试及新增测试全部通过。
