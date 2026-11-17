# Lean Harness Core, Bounded Context, and Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一次不保留旧 API 和 SQLite schema 的重构，建立精简可扩展的 Harness Core、有界上下文能力和独立 Artifact 子系统。

**Architecture:** Core 只保留不可变领域值、纯状态机、单次推进协调器及端口；application 管理 Session、幂等提交和资源生命周期；adapters 实现 context、artifact、model、tool、journal 与 observability。RunJournal 以 Session 为 CAS/序列作用域，原子提交 RunState、领域事件、submission、transcript append 和可选 summary checkpoint。

**Tech Stack:** Python 3.11、asyncio、Pydantic 2、SQLite、LiteLLM、FastMCP、pytest、mypy、Ruff。

**Spec:** `docs/superpowers/specs/2026-11-17-lean-core-context-artifacts-design.md`

## Global Constraints

- 所有工作只在 `F:\CodeSpace\harness-worktrees\lean-core-context-artifacts` 中进行。
- 不保留旧 `Harness`、旧 `src/schema`、旧 Python API 或 SQLite v1/v2 数据兼容。
- `src/core` 不得导入 `src.application`、`src.adapters`、配置、日志或任何具体实现。
- Core 构造必须显式接收所有端口与 `RunLimits`，不得读取环境或创建默认 adapter。
- 同一 Session 最多一个 active Run，不同 Session 可以并发；journal version 按 Session 作用域。
- 每次模型调用前必须验证 `estimated_input_tokens <= context_window_tokens - reserved_output_tokens - safety_margin_tokens`。
- Artifact `put()` 返回时对象必须完整、可校验并可立即读取；journal 失败只允许产生可回收 orphan。
- Task、领域事件和模型上下文不得内嵌 Artifact 本体或主机绝对路径。
- 新行为一律遵循 TDD：先运行并观察失败测试，再写最小实现。
- 每个任务完成后运行该任务测试与受影响回归；最终运行完整质量门禁。

## Review Focus

- 同一 Session 的两个并发 submission：只能有一个 active Run，失败者不得留下半提交 transcript；由 Task 4、Task 10、Task 11 覆盖。
- Artifact `put()` 成功而 journal commit 失败：对象可读但无 durable ref，并可被 orphan GC 清理；由 Task 6 覆盖。
- 工具结果恰好跨过 inline 阈值且 summary 本身接近上限：必须外置且最终状态仍满足 byte limit；由 Task 5、Task 9 覆盖。
- 必需 prompt 本身超过 context budget：不得调用 ModelGateway，必须产生 `context_budget_exceeded`；由 Task 8、Task 9 覆盖。
- summary checkpoint 已持久化但 policy/profile 改变：不得错误复用旧摘要；由 Task 8、Task 10 覆盖。

---

## File Structure

### Core

- `src/core/errors.py`：稳定错误代码与 Core 异常。
- `src/core/ids.py`：强类型关联 ID 与 ID/Clock 端口。
- `src/core/state.py`：`RunState`、step/call/approval 状态和纯状态转换。
- `src/core/commands.py`：Core command/result 值对象。
- `src/core/events.py`：有界领域事件。
- `src/core/context.py`：上下文预算、窗口、summary checkpoint 和 `ContextProvider`。
- `src/core/artifacts.py`：Artifact 值对象、repository/externalizer 端口。
- `src/core/model.py`：模型请求/响应和 `ModelGateway`。
- `src/core/tools.py`：工具请求/结果、审批和 `ToolGateway`。
- `src/core/journal.py`：`JournalCommit` 与 `RunJournal` 原子提交端口。
- `src/core/run.py`：`RunCoordinator` 单次推进。

### Application

- `src/application/session.py`：Session 串行化、submission 幂等和 transcript projection。
- `src/application/agent.py`：用例 facade。
- `src/application/runtime.py`：operation/resource 生命周期。
- `src/application/composition.py`：显式组装 adapters。

### Adapters

- `src/adapters/artifacts/memory.py`：内存 Artifact reference adapter。
- `src/adapters/artifacts/filesystem.py`：流式、原子、content-addressed 文件 Artifact adapter。
- `src/adapters/artifacts/externalizer.py`：阈值、summary 与外置策略。
- `src/adapters/context/bounded.py`：确定性上下文选择。
- `src/adapters/context/tokenizers.py`：exact/estimate token counter。
- `src/adapters/context/summarization.py`：summary checkpoint 生成与复用。
- `src/adapters/persistence/memory.py`：原子 journal reference adapter。
- `src/adapters/persistence/sqlite.py`：全新 SQLite schema。
- `src/adapters/model/litellm.py`：LiteLLM gateway。
- `src/adapters/tools/local.py`、`docker.py`、`mcp.py`：工具 adapters。
- `src/adapters/observability/events.py`、`metrics.py`：事实提交后的 observations。

---

### Task 1: Core Values, Limits, Errors, and Dependency Gate

**Files:**
- Create: `src/core/errors.py`
- Create: `src/core/ids.py`
- Create: `src/core/commands.py`
- Create: `src/core/events.py`
- Create: `tests/unit/core/test_core_values.py`
- Replace: `tests/integration/test_project_layout.py`

**Interfaces:**
- Produces: `RunErrorCode`, `RunFailure`, `RunId`, `SessionId`, `SubmissionId`, `ProfileId`, `EventId`, `IdGenerator`, `Clock`, `RunLimits`, `DomainEvent`, `DomainEventType`, `SubmitRun`, `ResumeRun`, `CancelRun`, `DecideToolCall`。
- `RunLimits(max_steps: int, max_elapsed_seconds: float, max_inline_result_bytes: int)` 的每个值必须为正数。
- `DomainEvent.payload` 必须是有界、严格 JSON；event 不允许主机路径或二进制值。

- [ ] **Step 1: 写失败测试固定不可变值、严格 JSON、正数限制和依赖边界**

测试至少断言：ID 拒绝空值/控制字符/超过 256 UTF-8 bytes；`RunLimits` 拒绝 `0`、负数、`bool` 和非有限值；event 拒绝 `bytes`、非字符串 key、NaN；AST gate 先覆盖本任务新建的 Core 模块并拒绝其导入 `src.adapters`、`src.application`、`src.config`、`src.logger`。现有 legacy Core 模块使用显式临时 allowlist，Task 14 删除旧模块后必须删除 allowlist 并对整个 `src/core` 生效。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_core_values.py tests/integration/test_project_layout.py`

Expected: FAIL，因为新模块和类型尚不存在。

- [ ] **Step 3: 实现最小 Core values 与架构 gate**

所有值对象使用 frozen Pydantic model 或 frozen dataclass；`RunFailure` 只暴露稳定 code 和安全 message，不序列化 adapter exception。

- [ ] **Step 4: 运行 Task 1 测试与类型检查**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_core_values.py tests/integration/test_project_layout.py`

Run: `uv run --locked mypy src/core`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/core/errors.py src/core/ids.py src/core/commands.py src/core/events.py tests/unit/core/test_core_values.py tests/integration/test_project_layout.py
git commit -m "refactor: establish lean core values"
```

### Task 2: Pure Run State Machine

**Files:**
- Create: `src/core/state.py`
- Create: `tests/unit/core/test_run_state.py`

**Interfaces:**
- Consumes: Task 1 IDs、commands、events、errors、`RunLimits`。
- Produces: `RunStatus`, `RunState`, `StepState`, `ModelCallState`, `ToolCallState`, `ApprovalState`, `StateTransition`, `TaskStateMachine.apply(state: RunState, command: CoreCommand) -> StateTransition`。
- `StateTransition` 包含新的 frozen `RunState` 和按顺序产生的 `tuple[DomainEvent, ...]`，不执行 I/O。

- [ ] **Step 1: 写状态转换表失败测试**

覆盖 `CREATED -> RUNNING -> WAITING_APPROVAL -> RUNNING -> COMPLETED`、failed/cancelled terminal 不可重开、错误 run/session 所有权、重复 approval、step/model/tool 父子层级、max_steps 与 elapsed limit。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_run_state.py`

Expected: FAIL，因为 `TaskStateMachine` 不存在。

- [ ] **Step 3: 实现纯状态机**

状态机方法不得是 async，不读取 clock，不生成 ID；所有时间、ID、model/tool result 都通过 command 显式输入。

- [ ] **Step 4: 添加生成式序列测试**

使用确定性 command 序列生成器覆盖至少 1,000 条有效/无效序列，断言 version 不回退、terminal 不重开、同一 tool call 只结算一次、开放子生命周期不能超过一个当前 step。

- [ ] **Step 5: 运行 Task 2 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_run_state.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/core/state.py tests/unit/core/test_run_state.py
git commit -m "refactor: add pure run state machine"
```

### Task 3: Atomic Journal Contract and In-Memory Adapter

**Files:**
- Create: `src/core/journal.py`
- Create: `src/adapters/__init__.py`
- Create: `src/adapters/persistence/__init__.py`
- Create: `src/adapters/persistence/memory.py`
- Create: `tests/unit/core/test_journal_contract.py`
- Create: `tests/unit/adapters/persistence/test_memory_journal.py`

**Interfaces:**
- Consumes: `RunState`, `DomainEvent`。
- Produces: `SubmissionRecord`, `TranscriptEntry`, `ConversationSummaryCheckpoint`, `JournalCommit`, `JournalSnapshot`, `RunJournal.commit(batch: JournalCommit, expected_version: int) -> Awaitable[JournalSnapshot]`, `load_session(session_id, after_sequence=0)`, `find_submission(session_id, submission_id)`。
- `JournalCommit` 原子包含 state、events、submission update、transcript append 和可选 summary checkpoint。

- [ ] **Step 1: 写共享 contract 失败测试**

断言连续 Session sequence、expected-version CAS、事件/run/submission identity 一致、commit 任一成员无效则零写入、重复相同 submission 返回原记录、相同 ID 不同 digest 冲突、summary key 包含 covered sequence/policy/profile。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_journal_contract.py tests/unit/adapters/persistence/test_memory_journal.py`

Expected: FAIL，因为 journal contract 尚不存在。

- [ ] **Step 3: 实现 journal port 和线程安全内存 adapter**

内存 adapter 必须 copy-on-read/copy-on-write，并在一个 lock 内验证和应用整个 `JournalCommit`。

- [ ] **Step 4: 运行并发与原子性测试**

增加两个 writer 使用同一 expected version 的测试，断言仅一个成功且另一个得到 `journal_conflict`，失败者没有 event/transcript 残留。

- [ ] **Step 5: 运行 Task 3 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_journal_contract.py tests/unit/adapters/persistence/test_memory_journal.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/core/journal.py src/adapters tests/unit/core/test_journal_contract.py tests/unit/adapters/persistence/test_memory_journal.py
git commit -m "feat: add atomic run journal contract"
```

### Task 4: Submission Ownership and Idempotent Acceptance

**Files:**
- Create: `src/application/submissions.py`
- Create: `tests/unit/application/test_submissions.py`

**Interfaces:**
- Consumes: `RunJournal`, `TaskStateMachine`, submission/transcript values。
- Produces: `SubmissionService.accept(session_id: SessionId, message: str, submission_id: SubmissionId, profile_id: ProfileId) -> Awaitable[AcceptedRun]`；`AcceptedRun` 包含新建或幂等复用的 `RunState` 与 journal version。
- 该服务只负责 submission 接受和 active Run 所有权，不执行 model/tool；Task 11 的 `AgentSession` 组合它与 `RunCoordinator`。

- [ ] **Step 1: 写失败测试固定 Session 所有权**

覆盖相同 ID/相同内容幂等、相同 ID/不同内容冲突、两个不同 submission 并发只接受一个、CAS 失败者不写 submission/event、terminal Session 可以接受下一 Run。approval ownership、排队 waiter 和 transcript terminal commit 留给 Task 11。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/application/test_submissions.py`

Expected: FAIL，因为新 Session API 未实现。

- [ ] **Step 3: 实现最小 submission 接受服务**

服务只调用 Core 和 journal 端口，不导入 SQLite/LiteLLM/MCP；不使用进程内 lock 代替 journal CAS。

- [ ] **Step 4: 运行 Task 4 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/application/test_submissions.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/application/submissions.py tests/unit/application/test_submissions.py
git commit -m "feat: enforce submission ownership"
```

### Task 5: Artifact Values, Externalizer, and In-Memory Repository

**Files:**
- Create: `src/core/artifacts.py`
- Create: `src/adapters/artifacts/__init__.py`
- Create: `src/adapters/artifacts/memory.py`
- Create: `src/adapters/artifacts/externalizer.py`
- Create: `tests/unit/core/test_artifacts.py`
- Create: `tests/unit/adapters/artifacts/test_memory_artifacts.py`
- Create: `tests/unit/adapters/artifacts/test_externalizer.py`

**Interfaces:**
- Produces: `ArtifactRef`, `ArtifactWrite`, `ArtifactRepository`, `ArtifactExternalizer`, `ExternalizedContent(inline_content, artifacts, summary)`, `ArtifactPolicy(max_inline_bytes, max_artifact_bytes, max_summary_bytes, allowed_media_types)`。
- `ArtifactRepository.put(request) -> Awaitable[ArtifactRef]`；`open(artifact_id) -> AsyncIterator[bytes]`；`delete()` 和 `stat()` 为 async。
- `ArtifactExternalizer.externalize(content, media_type, name=None) -> Awaitable[ExternalizedContent]`。

- [ ] **Step 1: 写 Artifact schema 与 repository 失败测试**

覆盖 strict metadata、sha256、非负/上限 size、name 不能是路径、stream put/open、空内容、重复内容 dedupe、写入取消、超过最大流大小不产生对象。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_artifacts.py tests/unit/adapters/artifacts/test_memory_artifacts.py`

Expected: FAIL，因为 Artifact 类型与 adapter 不存在。

- [ ] **Step 3: 实现 Artifact core port 和内存 adapter**

`put()` 仅在流完全读取、hash/size 校验完成后发布对象；content address 使用完整 SHA-256。

- [ ] **Step 4: 写并运行 externalizer 失败测试**

测试 `size == max_inline_bytes` 保持 inline，`size == max_inline_bytes + 1` 外置；summary 使用结构 key/count 或文本 head/tail，且 UTF-8 byte 长度不超过 `max_summary_bytes`；summary 接近上限时不得破坏 Unicode。

- [ ] **Step 5: 实现 externalizer 并运行 Task 5 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_artifacts.py tests/unit/adapters/artifacts`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/core/artifacts.py src/adapters/artifacts tests/unit/core/test_artifacts.py tests/unit/adapters/artifacts
git commit -m "feat: add artifact subsystem ports"
```

### Task 6: Filesystem Artifact Repository and Orphan Collection

**Files:**
- Create: `src/adapters/artifacts/filesystem.py`
- Create: `tests/unit/adapters/artifacts/test_filesystem_artifacts.py`

**Interfaces:**
- Consumes: Task 5 Artifact types。
- Produces: `FilesystemArtifactRepository(root: Path, max_artifact_bytes: int)`, `collect_orphans(referenced_ids: Collection[str], older_than: datetime) -> Awaitable[int]`。
- content-addressed layout 只能由内部 hash 构造；caller name 不参与路径。

- [ ] **Step 1: 写文件 adapter 失败测试**

覆盖 chunked streaming、临时文件+原子 rename、并发相同内容、hash/size、部分写/取消清理、symlink/path traversal、防绝对路径泄露、open 中途取消、只读已提交对象。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/artifacts/test_filesystem_artifacts.py`

Expected: FAIL，因为 filesystem adapter 不存在。

- [ ] **Step 3: 实现 filesystem repository**

在 root 内创建随机临时文件，流式计算 hash/size，`fsync` 后使用原子 rename 发布；异常路径删除精确临时文件。

- [ ] **Step 4: 添加 orphan 场景测试**

模拟 `put()` 成功、journal commit 失败；断言对象立即可读、没有 durable ref，并且只有超过宽限期且不在 `referenced_ids` 的对象被删除。

- [ ] **Step 5: 运行 Task 6 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/artifacts/test_filesystem_artifacts.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/adapters/artifacts/filesystem.py tests/unit/adapters/artifacts/test_filesystem_artifacts.py
git commit -m "feat: add filesystem artifact repository"
```

### Task 7: Model and Tool Ports with Artifact-Aware Results

**Files:**
- Create: `src/core/model.py`
- Create: `src/core/tools.py`
- Create: `tests/unit/core/test_model_tool_ports.py`

**Interfaces:**
- Produces: `ModelMessage`, `ModelRequest`, `ModelResponse`, `ModelToolCall`, `ModelGateway.complete(request) -> Awaitable[ModelResponse]`, `ToolSpec`, `ToolRequest`, `ToolResult`, `ToolGateway.execute(request) -> Awaitable[ToolResult]`。
- `ToolResult` 必须二选一：有界 `inline_content`，或 `artifacts: tuple[ArtifactRef, ...] + summary`；pending approval 不携带执行结果。

- [ ] **Step 1: 写失败测试固定 wire-neutral 模型**

覆盖 tool call ID 唯一性、assistant tool call 与 tool result 配对、Artifact 结果不允许同时携带完整 inline、本地异常文本不进入公开 failure、approval scope 与 workspace identity 属于 ToolRequest。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_model_tool_ports.py`

Expected: FAIL，因为新端口不存在。

- [ ] **Step 3: 实现 model/tool Core ports**

这些模块不包含 provider role 转换、LiteLLM 参数或本地 callable runner。

- [ ] **Step 4: 运行 Task 7 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_model_tool_ports.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/core/model.py src/core/tools.py tests/unit/core/test_model_tool_ports.py
git commit -m "refactor: define model and tool core ports"
```

### Task 8: Bounded Context Adapter and Summary Checkpoints

**Files:**
- Create: `src/core/context.py`
- Create: `src/adapters/context/__init__.py`
- Create: `src/adapters/context/tokenizers.py`
- Create: `src/adapters/context/bounded.py`
- Create: `src/adapters/context/summarization.py`
- Create: `tests/unit/core/test_context_contract.py`
- Create: `tests/unit/adapters/context/test_bounded_context.py`
- Create: `tests/unit/adapters/context/test_summarization.py`

**Interfaces:**
- Produces: `ContextBudget(context_window_tokens, reserved_output_tokens, safety_margin_tokens)`, `ContextRequest`, `ContextWindow`, `OmittedContextRef`, `ContextProvider.build(request, budget)`, `ConversationReader`, `TokenCounter.count(messages, tools) -> TokenEstimate`, `ConversationCompactor.compact(...) -> ConversationSummaryCheckpoint`。
- `ContextWindow` 包含 ordered messages、tools、estimate、omitted refs、policy version 和可选新 checkpoint。

- [ ] **Step 1: 写 Core budget 失败测试**

断言 input allowance 计算、保留空间不能超过 window、负值/bool 拒绝、`ContextWindow` 超预算拒绝、estimate 明确标记 `exact` 或 `conservative`。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_context_contract.py`

Expected: FAIL，因为 context contract 不存在。

- [ ] **Step 3: 实现 Core context contract 与 token counters**

提供测试用 exact counter 和生产 fallback conservative counter；fallback 必须应用配置 safety factor，不能声称 exact。

- [ ] **Step 4: 写 bounded selection 失败测试**

固定优先级：system/current input/开放 tool 配对必保留；之后最新完整交互；再 summary、memory、skills。测试刚好满足、超一 token、工具调用不可拆分、Artifact 只放 summary/ref、必需内容自身超限时抛 `ContextBudgetExceeded` 且不产生窗口。

- [ ] **Step 5: 实现 bounded adapter 并验证 10,000 条历史**

10,000 条 transcript 的测试断言输出 token 不超过 allowance、结果确定、读取分页且不把完整 transcript 放入 `RunState`。

- [ ] **Step 6: 写 summary checkpoint 失败测试**

断言相同 session/covered sequence/policy/profile 可复用；policy 或 profile 改变不复用；compactor 失败保留原始记录；新 checkpoint 尚未 journal commit 时不会成为 durable summary。

- [ ] **Step 7: 实现 summarization adapter 并运行 Task 8 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_context_contract.py tests/unit/adapters/context`

Expected: PASS。

- [ ] **Step 8: 提交**

```powershell
git add src/core/context.py src/adapters/context tests/unit/core/test_context_contract.py tests/unit/adapters/context
git commit -m "feat: add bounded context assembly"
```

### Task 9: RunCoordinator Single-Step Orchestration

**Files:**
- Create: `src/core/run.py`
- Create: `tests/unit/core/test_run_coordinator.py`

**Interfaces:**
- Consumes: `TaskStateMachine`, `ContextProvider`, `ModelGateway`, `ToolGateway`, `ArtifactExternalizer`, `RunJournal`, `Clock`, `IdGenerator`, `RunLimits`。
- Produces: `RunCoordinator.advance(state: RunState, command: CoreCommand) -> Awaitable[RunOutcome]`。
- 一次 `advance()` 最多执行一次 model decision 或一批有序 tool calls，并在返回前完成一次原子 journal commit。

- [ ] **Step 1: 写 direct-answer 失败测试**

断言调用顺序 `context -> model -> state machine -> journal`；journal 成功后返回 completed；context 超预算时 model 未调用且提交分类失败。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_run_coordinator.py -k "direct or context"`

Expected: FAIL，因为 `RunCoordinator` 不存在。

- [ ] **Step 3: 实现 direct-answer 最小路径**

adapter exception 映射为稳定 `RunFailure`；journal exception/cancellation 不得伪造 terminal success。

- [ ] **Step 4: 写工具、Artifact 与审批失败测试**

覆盖工具请求、mutating tool 暂停、approve/reject/resume、tool timeout、large result externalize、`read_artifact` 普通工具路径、summary byte limit、多个工具保持原顺序。

- [ ] **Step 5: 实现 tool path 和 Artifact 外置**

外置顺序固定为 `ArtifactExternalizer -> state/events -> journal`；journal 失败的 Artifact 交给 adapter orphan policy，不从 Core 直接删除。

- [ ] **Step 6: 写取消、CAS 与 checkpoint 失败测试**

覆盖 context/model/tool 前取消、native async 执行中取消、journal CAS conflict、summary checkpoint 只随成功 commit 持久化、terminal state 重复 advance 不调用 adapter。

- [ ] **Step 7: 运行 Task 9 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/core/test_run_coordinator.py`

Expected: PASS。

- [ ] **Step 8: 检查模块规模并提交**

Run: `(Get-Content src/core/run.py | Measure-Object -Line).Lines`

Expected: 不超过 350 行；超出时先按 model/tool/journal coordination 提取私有小模块，但 Core 总职责不扩张。

```powershell
git add src/core/run.py tests/unit/core/test_run_coordinator.py
git commit -m "refactor: orchestrate runs through core ports"
```

### Task 10: SQLite Journal v3 and Recovery

**Files:**
- Create: `src/adapters/persistence/sqlite.py`
- Create: `tests/unit/adapters/persistence/test_sqlite_journal.py`
- Create: `tests/integration/test_recovery_v3.py`

**Interfaces:**
- Consumes: Task 3 journal contract。
- Produces: `SQLiteRunJournal(path: Path, timeout_seconds: float)`；全新 schema 包含 sessions、runs、events、submissions、transcript、summary_checkpoints，不含 v1/v2 migration。

- [ ] **Step 1: 将共享 journal contract 参数化到 SQLite 并观察失败**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/persistence/test_sqlite_journal.py`

Expected: FAIL，因为 SQLite v3 adapter 不存在。

- [ ] **Step 2: 实现全新事务 schema 与索引**

一个 `BEGIN IMMEDIATE` 事务验证 Session expected version，写入整个 `JournalCommit`，提交后返回 snapshot。索引至少覆盖 active run、session sequence、submission identity 和 transcript pagination。

- [ ] **Step 3: 写并发 Session 和恢复失败测试**

覆盖同 Session 两个进程/实例竞争只能一个 commit、不同 Session 并发、paused approval 恢复、started-but-unsettled model/tool 结算 interrupted、terminal transcript 不重复、policy/profile 改变不复用 summary。

- [ ] **Step 4: 实现 recovery application service**

恢复只根据 journal 事实关闭开放生命周期，不重放 model/tool/Artifact side effect；dangling durable Artifact ref 视为 repository corruption 并返回稳定失败。

- [ ] **Step 5: 运行 Task 10 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/persistence/test_sqlite_journal.py tests/integration/test_recovery_v3.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/adapters/persistence/sqlite.py tests/unit/adapters/persistence/test_sqlite_journal.py tests/integration/test_recovery_v3.py
git commit -m "feat: add sqlite v3 run journal"
```

### Task 11: Migrate Application Facade and Runtime

**Files:**
- Replace: `src/application/agent.py`
- Create: `src/application/runtime.py`
- Create: `src/application/composition.py`
- Replace: `src/application/__init__.py`
- Create: `tests/unit/application/test_agent_v3.py`
- Create: `tests/unit/application/test_runtime_v3.py`

**Interfaces:**
- Consumes: Tasks 1-10 Core 与 adapter 接口。
- Produces: 新 `Agent`, `AgentSession`, `RuntimeCoordinator`, `ApplicationConfig`, development/production composition factories。
- `Agent` 对外提供 async-first `submit/open_session/close`；sync compatibility 方法不保留。

- [ ] **Step 1: 写新 facade 失败测试**

覆盖显式依赖注入、不同 Session 并发、同 Session 串行、closed runtime 拒绝新工作、close 取消 active operations、injected resource caller-owned、composition 不在 Core 创建默认依赖。

- [ ] **Step 2: 运行失败测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/application/test_agent_v3.py tests/unit/application/test_runtime_v3.py`

Expected: FAIL，因为新 facade/runtime 不存在。

- [ ] **Step 3: 实现 application runtime 与 Agent facade**

application 通过 Task 4 `SubmissionService` 接受请求，并负责循环调用 `RunCoordinator.advance()` 直到 terminal/waiting approval；Core 的 `advance()` 保持单次推进。`AgentSession` 使用 async operation lock 和有界 FIFO waiters，异常/caller cancellation 必须释放 lock；approval 必须校验 Session/Run 所有权，terminal commit 必须在同一个 `JournalCommit` 中追加 transcript。

- [ ] **Step 4: 实现 composition root**

配置选择 LiteLLM、SQLiteRunJournal、FilesystemArtifactRepository、BoundedContextAdapter 和 tool adapters；development factory 使用内存 adapters。配置值不得进入 Core 全局状态。

- [ ] **Step 5: 运行 Task 11 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/application/test_agent_v3.py tests/unit/application/test_runtime_v3.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/application tests/unit/application/test_agent_v3.py tests/unit/application/test_runtime_v3.py
git commit -m "refactor: rebuild application facade"
```

### Task 12: Migrate Concrete Model, Tool, MCP, and Observability Adapters

**Files:**
- Create: `src/adapters/model/__init__.py`
- Move/Rewrite: `src/llm/litellm.py` -> `src/adapters/model/litellm.py`
- Create: `src/adapters/tools/__init__.py`
- Move/Rewrite: `src/tool/local_tools.py` and `tool_executor.py` -> `src/adapters/tools/local.py`
- Move/Rewrite: `src/sandbox/docker.py` -> `src/adapters/tools/docker.py`
- Move/Rewrite: `src/mcp/*` -> `src/adapters/tools/mcp.py`
- Create: `src/adapters/observability/events.py`
- Move/Rewrite: `src/observability/metrics.py` -> `src/adapters/observability/metrics.py`
- Create: `tests/unit/adapters/model/test_litellm.py`
- Create: `tests/unit/adapters/tools/test_local.py`
- Create: `tests/unit/adapters/tools/test_docker.py`
- Create: `tests/unit/adapters/tools/test_mcp.py`
- Create: `tests/unit/adapters/observability/test_observability.py`

**Interfaces:**
- Consumes: Core model/tool ports和 application post-commit observation hook。
- Produces: `LiteLLMModelGateway`, `LocalToolGateway`, `DockerToolGateway`, `MCPToolGateway`, `ReadArtifactTool`, `CompositeObserver`, `MetricsObserver`。

- [ ] **Step 1: 迁移 LiteLLM contract 测试并观察失败**

固定 role/tool wire 转换、stream assembly、usage、provider error 分类、timeout/cancel、不把 provider object 泄漏进 Core。

- [ ] **Step 2: 实现 LiteLLM adapter 并运行测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/model/test_litellm.py`

Expected: PASS。

- [ ] **Step 3: 迁移 local/Docker/MCP 工具测试并观察失败**

保留 workspace containment、approval revalidation、Docker hardening、MCP catalog 原子更新和 active call lease；所有结果转换为 Task 7 `ToolResult`。

- [ ] **Step 4: 实现工具 adapters 与 `ReadArtifactTool`**

`ReadArtifactTool` 只接受 artifact ID、offset/limit；调用 repository 流式读取有界片段，超出部分返回新的 ArtifactRef 或明确 truncated metadata。

- [ ] **Step 5: 运行工具 adapter 测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/tools`

Expected: PASS。

- [ ] **Step 6: 迁移 observations 并验证默认不输出 stdout**

observer 只在 journal commit 成功后收到 observation；单个 observer 失败不影响事实；默认 composition 不启用内容 stdout emitter。

- [ ] **Step 7: 运行 Task 12 测试并提交**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/unit/adapters/model tests/unit/adapters/tools tests/unit/adapters/observability`

Expected: PASS。

```powershell
git add src/adapters tests/unit/adapters
git commit -m "refactor: migrate production adapters"
```

### Task 13: End-to-End Scenarios and Bounded Growth

**Files:**
- Create: `tests/integration/test_agent_v3.py`
- Create: `tests/integration/test_bounded_growth.py`
- Create: `tests/integration/test_artifact_flow.py`
- Replace: `tests/load/test_shared_agent_load.py`

**Interfaces:**
- Consumes: 完整新架构。
- Produces: 可执行的 end-to-end contract，供删除旧实现时防回归。

- [ ] **Step 1: 写 end-to-end 失败测试**

覆盖 direct answer、tool、approval/reject/resume、cancel、model/tool timeout、restart recovery、message idempotency、不同 Session 并发。

- [ ] **Step 2: 运行失败测试并补齐最小集成缺口**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/integration/test_agent_v3.py`

Expected: 初次 FAIL；只修复新架构 wiring，不添加兼容层。

- [ ] **Step 3: 写并通过 bounded growth 测试**

生成 10,000 条 transcript 与大型 Skill/tool history，断言每次 ModelRequest 不超 token allowance、RunState serialized size 保持配置上限、journal 使用分页读取、没有完整 transcript copy。

- [ ] **Step 4: 写并通过 Artifact flow 测试**

大型工具结果外置，ModelRequest 只含 summary/ref；模型可调用 `read_artifact` 获取有界片段；最终 RunOutcome 带 ArtifactRef；journal/event 不含本体和主机路径。

- [ ] **Step 5: 更新并发 load test**

至少 64 Session 并发，断言无状态泄漏、每个 Session version 独立、Artifact 和 summary ownership 正确；增加同 Session submission 竞争。

- [ ] **Step 6: 运行 Task 13 测试并提交**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/integration/test_agent_v3.py tests/integration/test_bounded_growth.py tests/integration/test_artifact_flow.py tests/load/test_shared_agent_load.py`

Expected: PASS。

```powershell
git add tests/integration tests/load/test_shared_agent_load.py
git commit -m "test: verify lean harness end to end"
```

### Task 14: Delete Legacy Architecture and Update Documentation

**Files:**
- Delete: `src/core/loop.py`, old core compatibility modules superseded by new files
- Delete: `src/schema/`
- Delete: `src/context/`, `src/llm/`, `src/tool/`, `src/sandbox/`, `src/mcp/`, `src/persistence/`, `src/observability/`, `src/demo/`
- Delete/Replace: legacy tests under `tests/unit` and `tests/integration`
- Modify: `src/config.py`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `TODO.md` if it is intentionally added to Git
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Tasks 1-13 的新架构和 end-to-end contract。
- Produces: 单轨仓库，不存在旧 API/schema/compatibility path。

- [ ] **Step 1: 列出旧模块的所有 import 引用并建立删除清单**

Run: `rg -n "src\.(schema|context|llm|tool|sandbox|mcp|persistence|observability|demo)|core\.loop|SnapshotExecutionStore|TaskRepository" src tests README.md`

Expected: 输出只用于迁移，不直接批量替换。

- [ ] **Step 2: 逐目录删除已被新测试覆盖的旧实现和测试**

不得保留 re-export、deprecated alias 或 SQLite migration。每删除一组后运行 Task 13 end-to-end contract。

- [ ] **Step 3: 更新配置、README 和 TODO**

README 只展示新 async API、显式 composition、context budget 和 Artifact 生命周期；`TODO.md` 将 LLM/tool timeout 标为已完成，将 context limit 替换为已实现说明，保留 A2A/eval 后续项。若主分支的 `TODO.md` 仍未跟踪，不擅自把它纳入本分支；改为在交付说明中提供建议 patch。

- [ ] **Step 4: 运行架构和 dead-code gate**

Run: `rg -n "src\.(schema|context|llm|tool|sandbox|mcp|persistence|observability|demo)|core\.loop|SnapshotExecutionStore|TaskRepository" src tests README.md`

Expected: 无旧架构引用；同时删除 Task 1 架构 gate 的 legacy allowlist，使规则覆盖整个 `src/core`。

Run: `uv run --locked python -m pytest -q -p no:cacheprovider tests/integration/test_project_layout.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add -A
git commit -m "refactor: remove legacy harness architecture"
```

### Task 15: Full Verification and Closeout

**Files:**
- Modify only if verification exposes a tested defect.

**Interfaces:**
- Produces: 可交付 branch 和验证记录。

- [ ] **Step 1: 运行格式、lint 和类型检查**

Run: `uv run --locked ruff format --check src tests`

Run: `uv run --locked ruff check --no-cache src tests`

Run: `uv run --locked mypy src`

Expected: 全部 PASS。

- [ ] **Step 2: 运行完整测试**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider`

Expected: 全部 PASS；任何 skip 必须逐项说明原因。

- [ ] **Step 3: 运行规模与依赖检查**

Run: `Get-ChildItem src/core -Filter *.py | ForEach-Object { [PSCustomObject]@{File=$_.Name; Lines=(Get-Content $_.FullName | Measure-Object -Line).Lines} } | Sort-Object Lines -Descending`

Expected: Core 单文件不超过 500 行，`run.py` 不超过 350 行。

Run: `git diff --check; git status --short`

Expected: 无 whitespace error，只有预期变更。

- [ ] **Step 4: 对照设计验收标准逐项记录证据**

在最终报告中列出 context 10,000 history、Artifact externalization、approval/recovery、CAS conflict、architecture gate 的测试名称和结果。

- [ ] **Step 5: 提交必要的验证修复**

```powershell
git add -A
git commit -m "chore: complete lean harness verification"
```

仅在确有修复时创建该提交；没有变更则不创建空提交。
