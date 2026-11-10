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
