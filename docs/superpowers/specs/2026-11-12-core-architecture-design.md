# Core Architecture Design

## Intent

将项目整理为“核心编排 + 端口契约 + 外部适配器”的 harness：`Harness` 依赖 `core`
定义的稳定契约，`Agent` 负责注入 `src` 各子目录中的默认实现，同时允许调用方替换任一外部能力。

## Goals

- 修复 `Task`、工具注册表和构造函数默认对象的实例共享问题。
- 保持 `Harness`、`Task`、`ToolExecutor`、`MemoryManager` 等核心流程为具体类。
- 用 `Protocol` 表达 LLM、上下文、工具目录、工具运行、审批、沙箱、内存、持久化、事件和 guard 端口。
- 让工具审批和沙箱执行只有一条清晰路径，不在 `Harness` 与执行器之间重复编排。
- 由 `Agent` 提供开箱即用的默认组装，同时保留显式依赖注入。

## Non-goals

- 本次不实现真实容器沙箱、数据库持久化或新的 LLM provider。
- 本次不改变现有 CLI 的用户可见行为。
- 本次不把每个 `core` 类型都改成抽象类，也不允许通过继承 `Harness` 修改主循环。

## Architecture

`core` 包含业务状态、编排服务和端口契约。外部目录包含可替换实现：

```text
Agent (composition root)
  -> Harness (core orchestration)
       -> LlmGateway / ContextBuilder
       -> ToolExecutor
            -> ToolCatalog / ToolRunner / ApprovalPolicy / SandboxRunner
       -> TaskRepository / HarnessGuard / EventHandler
```

纯依赖注入契约使用 `Protocol`；只有共享模板算法且必须禁止直接实例化时才使用 `ABC`。
未实现能力不得通过只有 `...` 的可实例化类伪装成成功。默认空策略必须显式命名为
`NoOp*`，默认内存实现必须真正保存数据。

## Compatibility

- 现有公开导入在本次改造中尽量保留；新端口可从现有模块重新导出。
- `DefaultToolExecutor` 在迁移期保留名称，但内部委托给新的核心 `ToolExecutor`。
- 先稳定行为和契约，再做必要的文件移动；不进行纯粹为了目录美观的改名。

## Verification

- 每个行为变化都遵循测试先行，先观察失败再实现。
- 完整验证包括 `pytest`、`ruff`，并为核心模块增加静态类型检查。
- 工具执行重点覆盖未知工具、错误参数、审批、拒绝、沙箱和执行异常。
