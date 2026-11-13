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
