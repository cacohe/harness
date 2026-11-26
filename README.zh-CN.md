# agent-harness

[English](README.md) | 简体中文

生产级、异步优先的 Harness 内核。项目将执行语义、应用用例和具体基础设施明确分层：

- `src/core`：按执行、上下文、工具、Artifact、持久化划分的五个能力包，以及共享的有界值；
- `src/application`：Session 所有权、幂等提交、运行时生命周期和 composition root；
- `src/adapters`：SQLite、文件 Artifact、LiteLLM、Docker、MCP、上下文策略与观测实现。

项目处于 `0.x` 阶段，只提供当前架构，不保留旧 Harness API 或旧 SQLite schema。

## 环境

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

```powershell
uv sync --locked
```

Harness 不隐式读取环境变量，也不在 Core 中选择模型、数据库或工作目录。调用方可以自行读取
`.env`，然后显式构造 adapter 和 `ApplicationConfig`。

## 异步 API 与显式组合

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.application import ApplicationConfig, create_production_agent
from src.core.context.contracts import ContextBudget
from src.core.execution.commands import RunLimits
from src.core.ids import ProfileId
from src.core.artifacts.lifecycle import ArtifactAttachmentRole

config = ApplicationConfig(
    limits=RunLimits(max_steps=32, max_elapsed_seconds=300.0, max_inline_result_bytes=65_536),
    context_budget=ContextBudget(
        context_window_tokens=128_000,
        reserved_output_tokens=4_096,
        safety_margin_tokens=1_024,
    ),
    sqlite_path=Path("C:/absolute/state/harness.db"),
    tool_execution_path=Path("C:/absolute/state/tool-executions.db"),
    artifact_root=Path("C:/absolute/state/artifacts"),
    max_artifact_read_bytes=65_536,
    max_orphan_reference_ids=1_000_000,
)
orphan_cutoff_utc = datetime.now(timezone.utc) - timedelta(days=1)

agent = await create_production_agent(
    config,
    model=model_gateway,
    tools=tool_gateway,
    context=context_provider,
    context_counter=token_counter,
    context_request_factory=request_factory,
    ids=id_generator,
    clock=clock,
    profile_id=ProfileId("default"),
)

async with agent:
    session = await agent.open_session()
    state = await session.submit("分析这批数据")
    if state.status.value == "waiting_approval":
        state = await session.decide_tool_call(state.run_id, "call-1", approved=True)
    detached = await session.detach_artifacts(state.run_id)
    if detached:
        await agent.artifacts.retire(detached[0].artifact.artifact_id)
    collected = await agent.artifacts.collect_orphans(orphan_cutoff_utc)
```

`create_development_agent()` 使用内存 journal 与内存 Artifact repository。只有未注入
`artifact_repository` 时，生产组合才会创建文件系统 repository，并要求绝对 `artifact_root`。
注入的 repository 可以是远程实现，不需要本地 root，但必须提供显式、不透明的
`artifact_workspace_id`、externalizer，以及部署实际需要的维护能力。Artifact ownership、retirement
和 reference-index port 已与 `RunJournal` 分离；生产组合仍要求这些 port 的适配器对象与 journal
是同一个原子持久化权威。生产组合使用独立的 `tool_execution_path` 保存工具 exactly-once claim；未显式
提供时会从显式 `sqlite_path` 派生同目录的 `*.tools` 文件，不会使用当前工作目录。两个 factory
都是 async-only，资源在调用方 event loop 中创建和关闭。

builder 会在启动时对该 tool execution store 执行 abandoned-claim recovery；同一个
`tool_execution_path` 的生产启动必须串行且单进程独占。需要多进程或多节点执行时，应替换为带
租约/协调能力的外部 ToolGateway，而不是让多个 builder 共享这个 reference SQLite store。

生产组合始终保留 `read_artifact` 名称，并将它路由到有界 `ReadArtifactTool`。传入的
`ToolGateway` 是其他工具的 fallback；`ContextRequestFactory` 不得重复声明 `read_artifact`，
composition 会把可信的内置 spec 注入每次请求。传入 `tools=None` 可得到只允许 Artifact 回读的
最小工具组合。

## 有界执行与上下文

每次 `RunCoordinator.advance()` 最多执行一次模型调用或一个持久化工具批次，并通过 journal
CAS 原子提交。外层循环由 application runtime 驱动，受以下边界约束：

- `RunLimits.max_steps`、`max_elapsed_seconds` 和 `max_inline_result_bytes`；
- `ApplicationConfig.max_advances_per_operation`；
- `ApplicationConfig.max_serialized_run_state_bytes`；
- `ContextBudget.input_allowance_tokens`。

如果最后一个允许的模型决策返回任何工具调用，Core 会在持久化或执行该批次之前以
`limit_exceeded` 终止 Run。普通工具、需要审批的工具、会写 Artifact 的工具以及 `load_skill`
都遵循这一规则，因此不会产生没有后续模型步骤可消费的副作用。

`BoundedContextProvider` 通过分页 reader 和认证 coverage 构建请求。10,000 条以上历史不会被
复制到 `RunState`，SQLite reader 使用索引分页与有界认证聚合；摘要 checkpoint 记录来源范围、
策略版本、profile 和 Artifact 引用。无法满足预算时返回稳定的
`context_budget_exceeded`，不会把超限请求发送给模型。

## 渐进式 Skills

两个 composition factory 都接受可选 `SkillProvider`。配置后，每次模型决策会收到一个有界、确定性
的 catalog SYSTEM message，其中仅包含每个 Skill 的 `name`、`description` 和 `version`，同时暴露保留
工具 `load_skill`。模型只能选择经过验证的逻辑名称；该保留调用由 Core 处理，绝不会转发给普通
`ToolGateway`。

激活成功后，`RunState` 只持久化 `name + version + content_sha256`。完整 SYSTEM 指令会在该 Run
余下生命周期持续生效：每次后续模型决策（包括进程重启后）都必须由 provider 精确解析已持久化绑定，
且返回内容的规范摘要必须一致。内容缺失或变化会在下一次模型请求前使 Run 失败，不会静默热更新；
新 Run 可以选择新版本。catalog、激活绑定、消息和确认结果都有明确上限，激活内容也计入普通上下文预算。

`FilesystemSkillProvider` 是单机参考 loader，要求
`<absolute-root>/<logical-name>/SKILL.md`，并使用精确的 `name`、`version`、`description`
frontmatter。root 是排他且 fail-closed 的 Skill 容器：每个直接子项都必须是合法 Skill 目录，且每个目录
只能包含 `SKILL.md`；`README.md`、`.gitkeep`、`desktop.ini` 等额外文件会使 catalog 失败，而不会
被静默忽略。`max_catalog_bytes` 按实际发送给模型的规范 JSON catalog 计量，包括字段名、标点和转义。
文件读取在线程中完成；路径穿越名称、symlink/reparse point、硬链接或非普通文件、错误或重复
metadata、非法 UTF-8、NUL、空/超限 body，以及验证读取期间发生变化的 entry 都会被拒绝。每个
provider 实例代表一个由调用方定义的信任域：它会校验 `profile_id`，但不会据此派生文件路径；需要
profile 隔离时，调用方必须注入不同的 provider 或 root。它不做递归发现、远程 registry 查询、自动
激活或热更新。

## Artifact 生命周期

大工具结果和大最终回答不会进入 RunState、领域事件或模型上下文本体。流程固定为：

1. `ArtifactExternalizer` 根据 policy 决定 inline 或外置；
2. repository 完整写入并校验 hash/size；
3. repository 为每次成功写入分配独立、不可变的 publication ID；journal 随后仅提交
   `ArtifactRef + bounded summary`；
4. 模型需要内容时，通过普通 `read_artifact` 工具按范围读取；
5. journal 失败可能留下未引用对象，由 adapter 的宽限期回收处理。

terminal Run 的 ownership 不会随 Session handle 关闭而隐式消失。调用方应显式使用
`await session.detach_artifacts(run_id)` 释放整 Run，或通过
`roles=(ArtifactAttachmentRole.TOOL_RESULT,)` 只释放指定角色。该 API 不接受调用方构造的
`ArtifactRef`，而是从 durable active index 读取完整 descriptor，并用 Session CAS 原子追加
`DETACHED` facts；重复或并发调用幂等。同一 publication ID 只有在所有 owner 都 detach 后才允许
`ArtifactRetentionService.retire()`。

Recovery 只把 journal 中仍然 active 的 attachment 当作待校验权威；已经显式 detach 的历史
`RunState`、tool result、final output 或 summary 引用不会因对象随后被回收而被误判为损坏。
retire 先在 journal 中原子取得 `RETIRING` reservation：该步骤与“没有 active owner”的判断
处于同一事务，并阻止新的 attachment；repository delete 成功后再标记 `RETIRED`，明确失败且
对象仍存在时标记 `ABORTED`。进程可在任一阶段重启并通过重复 retire 收敛。普通 attachment
不能越过 `RETIRED` 墓碑；相同内容重新 `put()` 会得到新的 publication ID，而 `sha256` 仍表示
内容摘要。因此相同 digest 可以再次使用，旧 tombstone 或迟到的 delete 也不会影响新对象。
`artifact_id` 是 repository 签发的不透明 publication identity，不等于内容 hash；`ArtifactRef`
分别校验 ID 与 `sha256` 的格式，而 repository 在签发 ref 前校验实际内容的 hash/size。该隔离会
牺牲跨 publication 的内容去重，但避免退役、重发与迟到删除之间的跨存储竞态。
SQLite 不把 derived active index 或 retirement current row 当成单点权威：reserve 在同一
`BEGIN IMMEDIATE` 中按 `artifact_id` 有界重放 append-only attachment facts 并与 index 双向
核对；retirement transition 另有 append-only hash chain、固定深度 Merkle projection 与数据库
authenticated root。普通 SQL 改写 current/fact/leaf/node/root/count 会在下一次生命周期操作时
稳定拒绝。

`FilesystemArtifactRepository` 使用 publication-addressed 路径、临时写入、原子 finalize、大小/hash
校验和路径防护；它是单机 reference adapter，不宣称支持多节点共享。

`agent.artifacts.collect_orphans(cutoff)` 只回收从未出现在 append-only Artifact facts 中且早于
UTC cutoff 的对象，不会把 detach 当成 orphan。引用清单读取受
`max_orphan_reference_ids` 限制；超过上限时安全失败，调用方必须提高显式配置或使用外部 GC。

## Adapter 边界

| 能力 | Core 端口 | Reference adapter |
|---|---|---|
| 模型 | `ModelGateway` | `LiteLLMModelGateway` |
| 工具 | `ToolGateway` | local / Docker / MCP |
| 上下文 | `ContextProvider` | bounded + tokenizer/compactor |
| Skills | `SkillProvider` | 有界文件系统 loader 或注入的 provider |
| Artifact | `ArtifactRepository`, `ArtifactExternalizer` | memory / filesystem |
| 持久化 | `RunJournal` | memory / SQLite |
| 观测 | post-commit observer | event / metrics sinks |

工具副作用的 exactly-once 边界由 gateway 的稳定 `idempotency_key` 和 execution store 保证；
Core 不在同一步中伪造“started + completed”双提交。观测只在 journal commit 后发生，sink 超时或
失败不回滚领域事实。

## 验证

```powershell
uv run --locked python -m pytest -q -p no:cacheprovider
uv run --locked python -m mypy src
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

架构测试确保 `src/core` 不导入 application、adapter、SQLite、LiteLLM、Docker、MCP 或观测实现，
并确保旧 schema、旧 Harness 和兼容路径不会重新出现。

## 许可证

本项目采用 [MIT License](LICENSE)。
