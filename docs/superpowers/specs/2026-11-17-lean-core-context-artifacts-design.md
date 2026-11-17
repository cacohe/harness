# 精简 Harness 内核、上下文治理与 Artifact 子系统设计

## 背景与意图

当前项目已经具备异步推理、工具调用、暂停审批、持久事实、恢复和运行时关闭等能力，
但核心控制流集中在接近两千行的 `src/core/loop.py` 中。该类同时承担状态转换、上下文
构建、模型重试、工具执行、事实持久化和观测发布，导致扩展一种能力时必须修改核心循环。

同时，默认上下文构建会把完整历史、全部 Skill 和工具结果持续加入每次模型请求；Task
快照也保存完整执行历史。长会话会产生无界 token 增长、重复序列化和突然触发 provider
上下文限制的问题。项目也没有一等 Artifact 模型，较大的工具结果只能内嵌进 Task、事件和
模型上下文。

本次改造面向尚无生产用户的 `0.x` 阶段，不保留现有 Python API、内部类型或 SQLite v2
数据兼容。目标是一次性建立精简、可扩展的 Harness 内核，并删除旧兼容层。

## 已选方案

采用“一次性重建精简内核”：重新定义 Core 的领域模型、状态机和端口，应用层负责用例
编排，adapter 负责某一种能力的生产级实现。借鉴事件驱动设计，让核心状态转换产生领域
事实，但不要求所有读取都通过完整事件重放。

### 考虑过的方案

1. **一次性重建精简内核（已选）**：边界最清晰，不保留双轨和兼容代码；代价是一次改动
   范围较大，必须用状态机、恢复和资源边界测试约束行为。
2. **渐进式绞杀旧 Harness**：迁移风险较低，但会长期保留代理类、兼容端口和重复模型，
   与精简 Core 的目标冲突。
3. **完全事件溯源**：所有状态均由事件 fold 获得，理论边界最纯粹，但会把 projection、
   snapshot 和查询模型同时引入本阶段，超出解决三个优先问题的必要范围。

## 目标

- `src/core` 只包含 Harness 不可替代的执行语义、领域状态和扩展协议。
- Core 不导入 `application`、`adapters`、`llm`、`persistence`、`sandbox`、`mcp`、
  `observability` 或其他具体实现。
- 将旧 `Harness` 拆成可独立测试的状态机和小型协调器。
- 所有模型调用在发送前经过明确、可测试的上下文预算决策。
- 大内容可以转为 Artifact 引用，Task、事件和模型上下文不持久化 Artifact 本体。
- Artifact 存储、摘要、token 估算和持久化均通过端口扩展。
- 保留现有的重要语义：异步优先、取消、暂停审批、调用超时、幂等提交、原子事实提交、
  中断恢复和资源关闭。

## 非目标

- 不在本阶段实现 A2A、HTTP 服务、分布式 scheduler、worker lease 或多租户认证。
- 不在本阶段实现 S3、PostgreSQL、向量数据库等所有生产 adapter；只提供可验证的本地
  reference adapter 和清晰端口。
- 不保留 `src.*` 现有公共导入路径、旧 `Harness` 方法或 SQLite v2 schema。
- 不实现完全事件溯源、任意历史时间点 projection 或跨版本数据迁移。
- 不把 prompt injection 检测、内容安全扫描或 Artifact 病毒扫描写进 Core。

## 架构原则

### 1. Core 只表达机制，不表达某一种策略

Core 可以定义 `ContextBudget`、`ArtifactRef`、`ModelGateway`、`ToolGateway`、
`RunJournal` 等协议和不可变值对象，但不能实现 tokenizer、摘要模型、SQLite、文件存储、
LiteLLM 或 Docker。

### 2. 依赖方向固定

```text
interfaces (CLI / future HTTP / future A2A)
                  |
                  v
application (use cases / session ownership / composition)
                  |
                  v
core (state machine / coordinator / values / ports)
                  ^
                  |
adapters (LLM / tools / context / artifact / persistence / telemetry)
```

`application` 依赖 Core 端口并组合 adapter；adapter 实现 Core 端口。Core 不知道 adapter
目录，也不提供默认生产实现。

### 3. 默认组合不是 Core 职责

“使用 LiteLLM、SQLite、本地 Artifact 目录和某一种 context policy”属于 application
composition root。Core 构造函数不创建具体依赖，不读取环境变量，也不选择默认 workspace。

### 4. 有界状态

Core 中所有随执行增长的集合必须有明确上限或外置路径。Task snapshot 仅保存继续执行所需
的 projection，不保存完整原始 Artifact 内容，也不把完整 Session transcript 复制到每个
Task。

## 目标目录与职责

以下结构是职责约束，不要求每个文件机械对应一个类；实施时可以在不改变依赖方向的前提下
合并极小值对象。

```text
src/
  core/
    run.py                 # RunCoordinator：推进一次执行，不组装具体 adapter
    state.py               # RunState、StepState、状态转换与不变量
    commands.py            # Submit、Resume、Cancel、Approve 等核心命令
    events.py              # 领域事实，不包含传输/日志格式
    context.py             # ContextRequest/ContextWindow/ContextBudget 及端口
    artifacts.py           # ArtifactRef/ArtifactMetadata 及 ArtifactRepository 端口
    model.py               # ModelGateway、模型请求/响应值对象
    tools.py               # ToolGateway、审批和工具调用值对象
    journal.py             # RunJournal 原子提交端口
    clock.py               # Clock/IdGenerator 等确定性端口
    errors.py              # 稳定错误分类
  application/
    agent.py               # 对外 use-case facade
    session.py             # Session 所有权、幂等提交与 transcript projection
    runtime.py             # 资源启动、关闭和运行中 operation 管理
    composition.py         # 根据配置组装 adapters
  adapters/
    context/
      bounded.py           # 有界上下文选择策略
      tokenizers.py        # provider/tokenizer 适配
      summarization.py     # 可选摘要策略
    artifacts/
      memory.py            # 测试/reference adapter
      filesystem.py        # 本地生产级 reference adapter
    model/
      litellm.py
    tools/
      local.py
      docker.py
      mcp.py
    persistence/
      memory.py
      sqlite.py
    observability/
      events.py
      metrics.py
```

旧的 `src/schema` 不再作为与 Core 平行的领域层。核心值对象归入 `core`；adapter 专用 wire
schema 留在对应 adapter。这样避免 `core` 与 `schema` 之间形成两个事实来源。

## 精简 Core 的允许与禁止内容

### 允许

- 状态枚举、不可变值对象和状态转换不变量。
- 运行步骤的协调顺序。
- 模型、工具、上下文、Artifact、journal、clock、ID 的 Protocol。
- timeout/cancellation/budget 的语义值，不包含具体 provider 策略。
- 领域事件及其最小关联 ID。

### 禁止

- 环境变量、配置文件、日志配置和 dependency injection 默认组合。
- SQLite、文件系统、Docker、LiteLLM、FastMCP 导入。
- stdout、OpenTelemetry、Prometheus 等观测实现。
- token 估算算法、具体摘要 prompt 和 provider model table。
- Artifact 路径布局、对象存储 SDK、签名 URL 和清理线程。
- 为旧 API 或旧 schema 提供的 re-export、fallback 和 compatibility branch。

使用 AST 架构测试强制执行以上边界。

## RunCoordinator 与状态机拆分

### Core 状态

`RunState` 是一次用户提交的有界 projection，包含：

- `run_id`、`session_id`、`submission_id`、`profile_id`；
- 当前状态与原因；
- 当前 step、开放的 model/tool 调用和待审批项；
- budget 消耗摘要；
- 最终文本与 `ArtifactRef` 列表；
- journal sequence/version。

完整 transcript、模型 delta、历史工具大结果和 Artifact 本体不属于 `RunState`。

同一 Session 在 application 层最多只有一个 active Run；不同 Session 可以并发。journal 的
sequence/version 以 `session_id` 为作用域，从而同时确定 submission 仲裁、Run 事实和
transcript 的全序。`run_id` 用于关联一次执行，但不形成独立的并发写序列。

### TaskStateMachine

纯状态机接收 command/result，返回新的 `RunState` 与领域事件。它不调用任何 I/O，也不读取
时间或生成 ID。典型转换包括：

- `CREATED -> RUNNING`
- `RUNNING -> WAITING_APPROVAL`
- `WAITING_APPROVAL -> RUNNING`
- `RUNNING -> COMPLETED | FAILED | CANCELLED`

非法转换在任何 adapter 被调用前失败。时间和 ID 由端口提供后作为显式输入传入。

### RunCoordinator

`RunCoordinator` 是 Core 中唯一允许协调多个端口的组件，职责仅限：

1. 向 `ContextProvider` 请求一个已经满足预算的 `ContextWindow`；
2. 调用 `ModelGateway`；
3. 把模型决定交给状态机；
4. 根据状态机输出调用 `ToolGateway`；
5. 将大结果交给 `ArtifactExternalizer` 外置；
6. 通过 `RunJournal.commit(expected_version, state, events)` 原子提交；
7. 返回 `RunOutcome` 或 `WAITING_APPROVAL`。

它不负责 Session 排队、message-id 幂等、资源启动关闭、配置加载、retry 策略和 telemetry
export；这些由 application 或 adapter wrapper 负责。

每个核心方法保持单次状态推进，循环停止条件由显式 `RunLimits` 控制。模型和工具 retry
在 gateway decorator/adapter 中完成，Core 只观察最终成功或分类失败。

## 上下文预算子系统

### Core 协议

Core 定义：

```python
class ContextProvider(Protocol):
    async def build(self, request: ContextRequest, budget: ContextBudget) -> ContextWindow: ...
```

`ContextRequest` 只包含构建上下文需要的引用和当前执行事实；`ContextBudget` 至少包含
`max_input_tokens`、`reserved_output_tokens` 和可选分区；`ContextWindow` 返回有序消息、
工具定义、使用量估计、被省略内容的引用及所用策略版本。

Core 在调用模型前验证：

```text
estimated_input_tokens <= max_input_tokens - reserved_output_tokens
```

adapter 若无法在不删除强制内容的情况下满足预算，返回稳定的
`ContextBudgetExceeded`，Core 将其结算为分类失败，而不是把超限请求发送给 provider。

`ContextBudget` 使用 `context_window_tokens` 表示 provider 的完整窗口，使用
`reserved_output_tokens` 表示为回答预留的空间；允许的输入上限固定计算为
`context_window_tokens - reserved_output_tokens - safety_margin_tokens`，避免
`max_input_tokens` 含义不清。

### BoundedContextAdapter

首个 reference adapter 使用确定性的优先级装箱：

1. 必须保留 system instruction、当前用户输入和当前待处理工具调用关系；
2. 保留最新的完整交互；
3. 按预算加入较旧摘要、memory 和 Skill 内容；
4. 大型工具结果只加入 Artifact 摘要和引用；
5. 仍然超限则明确失败。

不能从字符串长度直接假装得到精确 token。`TokenCounter` 是 adapter 端口；无法获得 provider
tokenizer 时使用显式标记为 estimate 的保守实现，并留出安全余量。

### 摘要边界

摘要不是 Core 能力。`ConversationCompactor` 属于 context adapter，可以使用规则或模型生成
摘要。摘要必须记录：来源 message/event 范围、策略版本、生成 profile 和 Artifact 引用；
原始事实仍由 journal/Artifact 保存。摘要失败不能静默删除历史。

摘要允许跨 Run 复用，但不参与运行正确性。`ContextWindow` 可以携带一个新的
`ConversationSummaryCheckpoint`；只有下一次成功 journal commit 才会把 checkpoint 与
Run 事实一起持久化。模型调用或 commit 失败时，下次允许重新生成同一摘要。已持久化摘要
以 `session_id + covered_through_sequence + policy_version + profile_id` 标识；源序列继续增长
不会修改旧摘要，只会生成覆盖更远位置的新 checkpoint。

### 防止持久化写放大

Session transcript 作为独立、分页读取的记录保存。Run snapshot 只持有 transcript cursor 或
引用，不复制整个 Session 历史。Context adapter 通过只读 `ConversationReader` 获取所需窗口。

## Artifact 子系统

### Core 值对象

`ArtifactRef` 是模型、工具和 Task 之间传递 Artifact 的唯一形式，至少包含：

- `artifact_id`
- `media_type`
- `size_bytes`
- `sha256`
- `name`（可选、非路径）
- `metadata`（有界、严格 JSON）

`ArtifactRef` 不包含主机绝对路径、bucket 凭证或可长期访问 URL。

### Core 端口

```python
class ArtifactRepository(Protocol):
    async def put(self, request: ArtifactWrite) -> ArtifactRef: ...
    async def stat(self, artifact_id: str) -> ArtifactRef | None: ...
    def open(self, artifact_id: str) -> AsyncIterator[bytes]: ...
    async def delete(self, artifact_id: str) -> None: ...
```

`put()` 返回成功时必须保证 Artifact 已完整写入、hash/size 已验证，并且可以立即通过返回的
`ArtifactRef` 读取。adapter 可以在内部使用临时对象和 finalize，但 finalize 必须发生在
`put()` 返回前，不能把一个尚不可读的引用交给 journal。

Core 定义 `ArtifactExternalizer` 端口，把潜在大内容转换为 `ExternalizedContent`：有界
`inline_content` 或 `ArtifactRef + summary`。它仅表达 Harness 所需的外置结果；存储位置、
摘要算法、分片、加密、签名 URL、病毒扫描、retention 和租户 ACL 属于 adapter 或上层
policy。

### 工具结果外置

`ToolResult` 支持两种互斥内容：

- 小型、严格 JSON 的 `inline_content`；
- 一个或多个 `ArtifactRef`，外加供模型使用的有界 `summary`。

由 adapter 中的 `ArtifactPolicy` 决定阈值和允许的 MIME。`ArtifactExternalizer` 负责生成
有界 summary：结构化结果优先保留 schema/key 和计数，文本使用确定性的 head/tail 摘要，
也可以委托可选 `ArtifactSummarizer`。Core 只要求最终进入状态、事件和上下文的 inline 内容
与 summary 不超过 `RunLimits.max_inline_result_bytes`。

模型不会获得 ArtifactRepository 的直接权限。application 可以显式注册一个普通
`read_artifact` Tool adapter；该工具遵循同样的审批、授权、timeout 和 context budget，按
范围读取 Artifact 并返回有界片段或新的派生 Artifact。当 summary 不足以继续推理时，模型
通过这个工具按需回读，而不是让 context adapter 自动注入完整本体。

### Artifact 事实

领域事件只保存引用：

- `artifact.created`
- `artifact.attached`
- `artifact.detached`

删除是 application/retention 用例，不由一次 Run 自动执行。journal commit 与对象存储无法
天然形成跨系统事务，因此顺序固定为：`put()` 完成并使 Artifact 可读，然后提交 journal
引用。journal 失败只会产生没有引用的 orphan，不会产生指向不可读对象的 durable ref；
orphan 由 adapter garbage collector 按宽限期清理。内存和文件 adapter 必须用故障测试固定
这一语义。

### 文件系统 reference adapter

`FilesystemArtifactRepository` 使用 content-addressed 路径、临时文件、原子 rename、hash
校验、最大流大小、路径穿越防护和有界读取。写入接口接受 stream，禁止先把全部内容读入
内存。它是单机生产级 reference adapter，不宣称支持多节点共享。

## 持久事实与观测

Core 只产生领域事件。`RunJournal` 的一次 expected-version CAS 必须原子保存同一 Session
的有界 `RunState`、领域事件、submission 幂等记录、transcript append，以及可选的
`ConversationSummaryCheckpoint`。这些数据共同组成 `JournalCommit`，不能由 application
拆成多个独立写入。这样恢复时不需要用 outbox 补偿“Task 已完成但 transcript 未提交”等
中间状态。

日志、metrics、trace 和客户端流是 application 在 journal 成功后发布的 observation；
observer 失败不回滚领域事实。

领域事件 payload 只能包含有界 JSON 和 `ArtifactRef`，不得包含任意二进制、完整流式 delta
或主机路径。

## 错误语义

Core 使用稳定分类而非公开 adapter 异常文本：

- `context_budget_exceeded`
- `model_failed`
- `tool_failed`
- `tool_timed_out`
- `artifact_failed`
- `journal_conflict`
- `cancelled`
- `limit_exceeded`
- `invariant_violation`

adapter 原始异常作为 cause 进入受控 telemetry，不直接写入用户结果、领域事件或 Artifact
metadata。

## 配置与组合

配置模型移到 application。Core 构造必须显式接收所有端口和 `RunLimits`。应用层提供
development composition：内存 journal、内存 Artifact、保守 token estimate；CLI 或未来
服务入口显式选择 SQLite、filesystem Artifact、LiteLLM 等 adapter。

不存在“Core 默认使用当前目录”“Core 默认打印 stdout”或“Core 默认创建 LiteLLM”的行为。

## 删除与迁移

本阶段允许破坏性内部迁移，最终删除：

- 旧 `src/core/loop.py` 单体实现；
- 旧 `src/schema` 领域模型；
- `SnapshotExecutionStore` 和旧 `TaskRepository` compatibility path；
- `src/demo` compatibility exports；
- SQLite v1/v2 migration 代码和旧 schema；
- 指向旧 API 的 README 示例与测试。

迁移通过替换测试和调用方完成，不提供旧数据库转换器。仓库不保留同时可运行的新旧核心。

## 测试策略

### Core

- 纯状态机表驱动测试覆盖全部合法与非法转换。
- property-based 或生成式序列测试验证 terminal state 不可重新打开、版本单调和审批所有权。
- fake ports 验证 `RunCoordinator` 的调用顺序、CAS 冲突、取消边界及 adapter 失败分类。
- 架构测试禁止 Core 导入具体 adapter 和配置/日志模块。

### Context

- 精确边界测试：刚好满足预算、超一 token、必需内容自身超限。
- 历史选择确定性、最新优先、工具调用/结果不被拆散。
- Artifact 摘要替代大结果，省略项可追溯。
- tokenizer estimate 与 exact 模式都保留安全余量。
- compactor 失败不会丢失原始历史。

### Artifact

- streaming put/open、hash/size 校验、重复内容、空文件和超限。
- 临时写失败、journal commit 失败和 finalize 失败后的垃圾回收。
- 文件路径穿越、symlink、部分写、并发同内容写入和取消。
- Task、event、context 中只出现 `ArtifactRef`，不出现本体或主机路径。

### 集成与回归

- 完整 direct answer、tool call、approval/resume、cancel、timeout、restart recovery。
- 长会话上下文始终满足预算，Task snapshot 大小保持有界。
- 大型工具输出被外置并仍能生成最终回答。
- application composition 可以替换 context、artifact、journal 和 model adapters。

## 实施阶段

### 阶段 A：建立新 Core 骨架

定义领域值、状态机、端口、错误和架构 gate；不接入旧 `Harness` compatibility layer。

### 阶段 B：Artifact 端口与 reference adapters

先实现内存 adapter，再实现 filesystem adapter 和故障语义；让新 ToolResult 使用引用。

### 阶段 C：有界上下文 adapter

建立 transcript reader、token counter、优先级选择和可选 compaction；证明长会话有界。

### 阶段 D：RunCoordinator

按单次推进模型组合 context、model、tool、artifact 和 journal 端口，迁移暂停/恢复/取消语义。

### 阶段 E：Application 与 adapter 迁移

迁移 Agent/Session/runtime、LiteLLM、MCP、Docker、SQLite、memory 和 observability。

### 阶段 F：删除旧实现

删除旧 Harness、schema、兼容代码和旧测试，更新 README，运行全套质量门禁。

## 验收标准

- `src/core` 不导入任何具体 adapter、配置、日志或环境模块。
- Core 中不存在超过 500 行的单一职责模块；`RunCoordinator` 不超过 350 行，状态机独立。
- 每次模型调用都有 `ContextBudget`，adapter 无法满足时不会调用 provider。
- 10,000 条历史记录的测试中，模型请求 token 和 Run snapshot 大小保持配置上限内。
- 大型工具结果只以 `ArtifactRef + summary` 进入状态、事件和上下文。
- filesystem Artifact 写入流式、原子、可校验，并覆盖失败清理。
- direct answer、tool、approval/resume、cancel、timeout、CAS 冲突和 restart recovery 集成测试通过。
- 仓库中不保留旧 `Harness` compatibility 实现和 SQLite v1/v2 migration。
- `ruff format --check`、`ruff check`、`mypy` 和完整 `pytest` 全部通过。


