# agent-harness

Agent harness 的接口与调用顺序草图。运行时依赖由 uv 管理，应用模块从仓库根目录以 `python -m src.<module>` 启动。

## 环境

- Python 3.11 及以上（见 `.python-version`）
- [uv](https://docs.astral.sh/uv/)

## 安装

在仓库根目录执行：

```powershell
uv sync --locked
```

## 配置

复制环境变量模板并填入模型与密钥：

```powershell
Copy-Item .env.example .env
```

`.env` 示例：

```env
LLM_MODEL=openai/gpt-4o
LLM_API_KEY=
LLM_API_BASE=
MAX_STEPS=10
MAX_SECONDS=1800
```

`LLM_MODEL` 必填。`LLM_API_KEY` 与 `LLM_API_BASE` 按所选模型填写。进程环境变量优先于 `.env`，`.env` 优先于 `config.toml` / `config.json`。也可用 `HARNESS_ENV_FILE`、`HARNESS_CONFIG_FILE` 指定配置文件路径。

## 组装方式

`Agent` 是默认 composition root。普通调用方可以让它按配置创建 LiteLLM、上下文构建器、
本地工具目录、工具执行器和内存任务仓库：

```python
from src.config import Config
from src.demo.agent import Agent

agent = Agent.from_config(Config.load(), workspace="./my-project")
task = agent.chat("帮我比较两款手机")
```

`workspace` 是本地工具唯一的文件系统权限边界，可传入现有目录路径或 `Workspace` 对象；
未提供时，`Agent` 只在构造时捕获当时的当前目录。默认目录包含 `read_file`、
`write_file`、`edit_file`、`list_dir`、`grep` 和 `glob` 六个绑定到该 workspace 的工具。
工具参数只接受 workspace 相对路径；绝对路径、父目录穿越、Windows 设备名/数据流以及解析后
越界的符号链接会被拒绝。本地 I/O 错误只报告相对路径，不暴露宿主 workspace 前缀。

默认不会注册 shell。要启用 shell，调用方必须注入真正提供隔离能力的 `SandboxRunner`：

```python
from src.config import Config
from src.core.workspace import Workspace
from src.demo.agent import Agent

workspace = Workspace("./my-project")
agent = Agent.from_config(
    Config.load(),
    workspace=workspace,
    sandbox_runner=my_sandbox_runner,
)
```

`SandboxRunner` 实现 `supports(tool)` 与 `run(tool, inputs, workspace)`。标记为
`requires_sandbox` 的工具在未配置兼容 runner 时失败关闭，绝不回退到宿主进程执行。
审批只表示用户允许某次调用，不提供隔离，也不能绕过 workspace 或 sandbox 检查。
注入的 runner 是受信任的隔离边界；当前框架本身不声明提供硬链接、挂载、网络或进程隔离。

## 外部取消

同步 Harness 支持每次运行独立的协作式取消。控制线程创建 `CancellationToken`，将它传给
`Agent.chat` 或 `Agent.resume`，再从其他线程调用 `cancel()`：

```python
from threading import Thread

from src.core.cancellation import CancellationToken

token = CancellationToken()
worker = Thread(
    target=lambda: agent.chat("分析这个项目", cancellation_token=token),
)
worker.start()

# 控制器决定停止本次运行。
token.cancel()
worker.join()
```

取消是线程安全、幂等且按单次运行隔离的。Harness 会立即唤醒模型重试等待，并在规划、
推理和工具调用边界停止后续工作；任务最终保存为 `failed/cancel`。取消不会强制终止已经
进入第三方模型、工具、线程、进程或 sandbox 的阻塞调用，而会在该调用返回后丢弃其结果并
停止下一步。一个已经取消的 token 不会自动重置，重复用于其他运行会立即取消该运行。

## 异步运行时

`Harness` 和 `Agent` 以异步实现为主。在已有事件循环的服务、Notebook 或异步测试中，直接
调用 `arun`、`achat` 和 `aresume`：

```python
task = await agent.achat("分析这个项目")

if task.is_paused():
    for request in agent.approval_requests():
        agent.decide_tool_call(request.tool_id, approved=True)
    task = await agent.aresume()
```

同步的 `Harness.run()`、`Agent.chat()` 和 `Agent.resume()` 是兼容入口，只在调用线程没有
运行中的事件循环时可用；在异步代码里调用会立即抛出 `RuntimeError`，应改为 `await` 对应的
异步方法。默认 LiteLLM gateway 使用原生异步请求和流式响应；旧的同步 gateway、同步流和本地
工具会在线程池中运行，避免阻塞事件循环。

多个独立 `Harness.arun()` 或独立 `Agent` 实例可以在同一事件循环并发运行。当前单个 `Agent`
仍只拥有一个 `pending_task` 和一份 transcript，不应同时启动多个会话；按会话隔离并发所有权属于
下一阶段。

取消等待中的原生协程会停止后续 Harness 工作并保存 `failed/cancel`。已经进入同步 gateway 或
同步工具的线程池调用无法被强制终止；其结果会被丢弃，后台线程可能在短时间内继续运行。外部
`CancellationToken` 仍采用上一节说明的协作式、非抢占语义。


## MCP 工具

MCP 连接属于持久异步资源，应通过 `Agent` 的异步上下文管理生命周期。连接成功后，远程工具
进入 Agent 与工具执行器共享的同一目录；MCP 工具默认逐次要求审批：

```python
import asyncio

from src.config import Config
from src.demo.agent import Agent


async def main() -> None:
    async with Agent.from_config(Config.load(), workspace="./my-project") as agent:
        await agent.add_mcp_server("weather", "https://example.com/mcp")

        task = await agent.achat("查询北京天气")
        if task.is_paused():
            for request in agent.approval_requests():
                agent.decide_tool_call(request.tool_id, approved=True)
            task = await agent.aresume()

        print(task.result.output)


asyncio.run(main())
```

也可以用 `refresh_mcp_server(name)` 原子刷新工具定义，或用 `remove_mcp_server(name)` 主动断开。
退出 `async with` 会调用幂等的 `aclose()`，立即撤下工具，等待已开始的远程调用结束，再关闭
传输。关闭后的 Agent 不再接受聊天或新的 MCP 生命周期操作。

模型可见名称采用 `mcp__<server>__<tool>` 格式。非 ASCII 与标点会被规范化，超长名称带稳定
哈希后缀；规范化后发生冲突时连接或刷新会失败，目录不会留下部分注册结果。调用方应使用
Agent 目录中实际公布的名称，不应自行拼接未经规范化的远程名称。

连接 MCP 服务不等于授权该服务执行操作，因此默认审批策略会暂停每一次 MCP 工具调用。更重要
的是，远程 MCP 工具不在本地 `Workspace` 内运行，也不会经过本地 `SandboxRunner`；本地文件和
进程隔离保证不适用于远程服务。服务身份、网络访问、凭据、数据披露以及远端执行环境必须由部署
方单独评估和控制。

需要替换模型、工具执行或持久化时，构造函数接收的是 `core` 中定义的端口，运行时注入
相应实现：

```python
from src.context.context_builder import DefaultContextBuilder
from src.core.guard import NoOpGuard
from src.core.loop import Harness
from src.core.tool import ToolExecutor
from src.persistence.in_memory import InMemoryTaskRepository
from src.tool.tool_executor import LocalToolRunner
from src.tool.tool_registry import DefaultToolRegistry

catalog = DefaultToolRegistry()
executor = ToolExecutor(catalog, LocalToolRunner())
harness = Harness(
    "You are a helpful assistant.",
    my_llm_gateway,
    [],
    executor,
    DefaultContextBuilder(),
    harness_guard=NoOpGuard(),
    task_repository=InMemoryTaskRepository(),
)
```

`LlmGateway`、`ContextBuilder`、`EventSink`、`ToolCatalog`、`ToolRunner`、`ApprovalPolicy`、
`SandboxRunner`、`MemoryStore`、`MemoryRetriever`、`TaskRepository` 和
`HarnessGuard` 使用 `Protocol` 描述。自定义实现只需满足方法签名，不必继承这些类型。

`Harness`、`Task`、`ToolExecutor` 和 `MemoryManager` 是核心具体类；扩展行为应优先通过
注入端口、hook 和策略完成，而不是继承并覆盖主循环。

## 任务状态与持久化

`Task` 及其消息、计划、工具执行和结果状态使用 Pydantic 模型。`Task.to_dict()` 生成
JSON-safe 快照，`Task.from_dict()` 可恢复完整状态；完成和失败任务始终包含字符串结果。

默认的 `InMemoryTaskRepository` 仅适用于单进程生命周期，但保存和读取使用相互独立的
可序列化快照，不会因调用方继续修改原 Task 或读取结果而隐式改变仓库内容。需要跨进程恢复时，
可实现同一 `TaskRepository` 端口，将这些快照保存到数据库或其他 durable storage。

## 验证

```powershell
uv run --locked ruff format --check src tests
uv run --locked ruff check --no-cache src tests
uv run --locked mypy src
uv run --locked python -m pytest -q -p no:cacheprovider
```

## 启动

在仓库根目录启动交互式 CLI。它会读取当前目录的配置，然后等待输入；每一行发给 agent，回答打印在下一行。

```powershell
uv run python -m src.demo.main
```

```text
agent-harness. Enter a message, or /exit to quit.
> 帮我比较两款手机
...
> /exit
```

`/help` 查看命令。`/exit`、`/quit`、`exit`、`quit` 或 Ctrl+C 退出。空行会被忽略。同一会话里，后面的输入会带上之前的问答。
