# agent-harness

生产级、异步优先的 Harness 内核。项目将执行语义、应用用例和具体基础设施明确分层：

- `src/core`：有界状态机、单步协调器、领域值与扩展端口；
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
from pathlib import Path

from src.application import ApplicationConfig, create_production_agent
from src.core.commands import RunLimits
from src.core.context import ContextBudget
from src.core.ids import ProfileId
from src.core.journal import ArtifactAttachmentRole

config = ApplicationConfig(
    limits=RunLimits(max_steps=32, max_elapsed_seconds=300.0, max_inline_result_bytes=65_536),
    context_budget=ContextBudget(
        context_window_tokens=128_000,
        reserved_output_tokens=4_096,
        safety_margin_tokens=1_024,
    ),
    sqlite_path=Path("C:/absolute/state/harness.db"),
    artifact_root=Path("C:/absolute/state/artifacts"),
)

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
```

`create_development_agent()` 使用内存 journal 与内存 Artifact repository；生产组合在未显式注入
实现时要求绝对 `sqlite_path` 和 `artifact_root`。两个 factory 都是 async-only，资源在调用方
event loop 中创建和关闭。

## 有界执行与上下文

每次 `RunCoordinator.advance()` 最多执行一次模型调用或一个持久化工具批次，并通过 journal
CAS 原子提交。外层循环由 application runtime 驱动，受以下边界约束：

- `RunLimits.max_steps`、`max_elapsed_seconds` 和 `max_inline_result_bytes`；
- `ApplicationConfig.max_advances_per_operation`；
- `ApplicationConfig.max_serialized_run_state_bytes`；
- `ContextBudget.input_allowance_tokens`。

`BoundedContextProvider` 通过分页 reader 和认证 coverage 构建请求。10,000 条以上历史不会被
复制到 `RunState`，SQLite reader 使用索引分页与有界认证聚合；摘要 checkpoint 记录来源范围、
策略版本、profile 和 Artifact 引用。无法满足预算时返回稳定的
`context_budget_exceeded`，不会把超限请求发送给模型。

## Artifact 生命周期

大工具结果和大最终回答不会进入 RunState、领域事件或模型上下文本体。流程固定为：

1. `ArtifactExternalizer` 根据 policy 决定 inline 或外置；
2. repository 完整写入并校验 hash/size；
3. journal 仅提交 `ArtifactRef + bounded summary`；
4. 模型需要内容时，通过普通 `read_artifact` 工具按范围读取；
5. journal 失败可能留下未引用对象，由 adapter 的宽限期回收处理。

terminal Run 的 ownership 不会随 Session handle 关闭而隐式消失。调用方应显式使用
`await session.detach_artifacts(run_id)` 释放整 Run，或通过
`roles=(ArtifactAttachmentRole.TOOL_RESULT,)` 只释放指定角色。该 API 不接受调用方构造的
`ArtifactRef`，而是从 durable active index 读取完整 descriptor，并用 Session CAS 原子追加
`DETACHED` facts；重复或并发调用幂等。共享 digest 只有在所有 owner 都 detach 后才允许
`ArtifactRetentionService.retire()`。

`FilesystemArtifactRepository` 使用 content-addressed 路径、临时写入、原子 finalize、大小/hash
校验和路径防护；它是单机 reference adapter，不宣称支持多节点共享。

## Adapter 边界

| 能力 | Core 端口 | Reference adapter |
|---|---|---|
| 模型 | `ModelGateway` | `LiteLLMModelGateway` |
| 工具 | `ToolGateway` | local / Docker / MCP |
| 上下文 | `ContextProvider` | bounded + tokenizer/compactor |
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
