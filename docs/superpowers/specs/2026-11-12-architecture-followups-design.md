# Architecture Follow-ups Design

## Intent

完成 core 架构重构后的四项收尾工作：建立可持续的静态类型门禁、统一事件注入端口、
同步实施文档状态，并在项目仍处于 0.1.0 阶段时删除不再需要的兼容层和占位协议。

## Goals

- 使用 mypy 检查整个 src 包，并把检查命令纳入项目验证说明。
- 用运行时可检查的 EventSink Protocol 表达 Harness 的事件输出依赖。
- 保留 EventHandler 作为默认具体实现，同时使用显式 is None 选择默认值。
- 删除 ToolRegistry、Sandbox 和 src.schema.hook 兼容导出。
- 删除未使用的 CheckpointStore、ArtifactStore 和 TraceStore Protocol。
- 更新引用、测试、README、设计文档和原实施计划状态。
- 保持工具审批、暂停恢复、CLI 和 Agent 的用户可见行为不变。

## Non-goals

- 不实现外部取消机制、真实沙箱、数据库存储或新的 provider。
- 不要求 mypy --strict；本次建立覆盖整个 src 的可执行基线，后续再逐步收紧。
- 不删除仍由核心流程实际使用的 TaskRepository、SandboxRunner 或 HarnessHook。

## Selected Approach

采用一次性清理方案。项目版本仍为 0.1.0，直接移除兼容导出比继续维护弃用层更清晰。
类型检查覆盖整个 src，通过必要而有限的类型修正建立零错误基线，不使用大范围
ignore_errors 掩盖问题。事件边界采用 EventSink Protocol，EventHandler 继续负责
默认 stdout emitter 与容错日志。

## Compatibility and Migration

- ToolRegistry 调用方改用 ToolCatalog。
- Sandbox 调用方改用 SandboxRunner。
- src.schema.hook.HarnessHook 调用方改用 src.core.hook.HarnessHook。
- 当前仓库没有这些兼容导出的生产调用方；删除后属于有意的 0.1.0 API 整理。

## Worktree Policy

OpenAI 官方 config.toml 没有 worktree 根目录配置，Codex 原生 worktree 工具也不接受
目标路径。使用全局 AGENTS.md 持久记录规则：新 worktree 必须创建在
F:\CodeSpace\harness-worktrees 下；同时把该目录加入 Codex sandbox 的额外可写根目录。

## Verification

- 新行为遵循测试先行，验证测试在实现前按预期失败。
- python -m pytest -q -p no:cacheprovider
- python -m ruff check --no-cache src tests
- python -m ruff format --check src tests
- python -m mypy src
- git diff --check

## Cleanup

新 PR 创建后，删除已合入的旧远程分支、本地
codex/core-architecture-refactor 分支和 C 盘旧 worktree。新的 F 盘 worktree 保留，
用于处理 PR 审批反馈。
