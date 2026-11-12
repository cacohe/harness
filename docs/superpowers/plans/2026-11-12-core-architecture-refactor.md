# Core Architecture Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复实例共享和空实现风险，并把 harness 整理为核心编排依赖稳定端口、`Agent` 注入默认适配器的结构。

**Architecture:** `Harness`、`Task`、`ToolExecutor` 和 `MemoryManager` 保持具体核心类型；外部能力用 `Protocol` 表达。工具执行由一个核心编排器协调目录、审批、普通 runner 和沙箱 runner，`Agent` 是默认 composition root。

**Tech Stack:** Python 3.11、pytest、ruff、Pydantic、uv

**Spec:** `docs/superpowers/specs/2026-11-12-core-architecture-design.md`

## Global Constraints

- 所有修改只发生在基于最新 `origin/main` 的 `codex/core-architecture-refactor` worktree。
- 不引入真实容器沙箱、数据库或新的 LLM provider。
- 保持现有 CLI 用户可见行为和已有公开导入，除非兼容层明确覆盖。
- 每项行为修改先写失败测试，观察预期失败后再写生产代码。

## Review Focus

- 连续创建两个 `Task` 时 ID 和可变集合必须隔离。
- 两个工具目录实例不得共享注册状态。
- 未知工具、非对象 JSON、执行异常必须返回明确失败结果。
- 需要审批或沙箱的工具不得绕过相应端口在进程内重复执行。
- 未配置可选能力时不得通过只有 `...` 的空实现静默成功。

---

### Task 1: 实例隔离与安全默认值

**Files:**
- Modify: `src/core/task.py`
- Modify: `src/core/loop.py`
- Modify: `src/tool/tool_registry.py`
- Modify: `src/demo/agent.py`
- Test: `tests/unit/core/test_task.py`
- Test: `tests/unit/core/test_loop.py`
- Test: `tests/unit/tool/test_tool_registry.py`

**Interfaces:**
- Produces: 独立的 `Task` 状态、独立的 `DefaultToolRegistry`、按实例创建的 hook/guard。

- [ ] 写回归测试并确认在远程基线上因共享 ID、共享列表、共享 registry 或默认对象而失败。
- [ ] 用 `None`/实例初始化修复默认值，并让 `Task.to_dict()` 支持空计划。
- [ ] 删除 `Agent` 对 `registry.tools = {}` 的补丁式重置。
- [ ] 运行目标测试和完整测试套件。
- [ ] 提交 `fix: isolate harness runtime state`。

### Task 2: 将纯扩展边界统一为 Protocol

**Files:**
- Modify: `src/core/context.py`
- Modify: `src/core/llm.py`
- Modify: `src/core/guard.py`
- Modify: `src/core/memory.py`
- Modify: `src/context/context_builder.py`
- Modify: `src/llm/litellm.py`
- Modify: `src/memory/store.py`
- Modify: `src/memory/retriever.py`
- Test: `tests/unit/core/test_ports.py`
- Test: `tests/unit/core/test_memory.py`

**Interfaces:**
- Produces: `ContextBuilder`、`LlmGateway`、`HarnessGuard`、`MemoryStore`、`MemoryRetriever` 结构化端口；`NoOpGuard` 默认策略。

- [ ] 写测试证明不显式继承的对象也能注入，并确认当前 ABC/具体 guard 结构不满足目标。
- [ ] 将纯端口改为 `Protocol`，删除无意义的空构造函数和显式继承。
- [ ] 让 `MemoryManager` 的 store/retriever 成为必需依赖并覆盖缺失依赖场景。
- [ ] 运行目标测试和完整测试套件。
- [ ] 提交 `refactor: define core dependency ports`。

### Task 3: 重构工具执行边界

**Files:**
- Modify: `src/core/tool.py`
- Modify: `src/core/sandbox.py`
- Modify: `src/tool/tool_executor.py`
- Modify: `src/tool/tool_registry.py`
- Modify: `src/core/loop.py`
- Test: `tests/unit/tool/test_tool_executor.py`
- Test: `tests/unit/core/test_loop.py`

**Interfaces:**
- Produces: `ToolCatalog`、`ToolRunner`、`ApprovalPolicy`、`SandboxRunner` Protocol；具体核心 `ToolExecutor`。
- Compatibility: `ToolRegistry` 和 `DefaultToolExecutor` 保留兼容名称。

- [ ] 分别为未知工具、错误 JSON、异常、审批、拒绝和沙箱路径写失败测试。
- [ ] 实现端口和唯一的工具执行编排路径，删除 bool/Enum、Tool/id 和沙箱签名冲突。
- [ ] 让 `Harness` 委托批量执行，不再复制结果分类算法，并从 catalog 动态获得工具。
- [ ] 运行目标测试和完整测试套件。
- [ ] 提交 `refactor: separate tool execution ports`。

### Task 4: 引入真实持久化边界

**Files:**
- Modify: `src/core/persistence.py`
- Modify: `src/core/task.py`
- Modify: `src/core/loop.py`
- Create: `src/persistence/__init__.py`
- Create: `src/persistence/in_memory.py`
- Test: `tests/unit/core/test_persistence.py`
- Test: `tests/unit/core/test_loop.py`

**Interfaces:**
- Produces: `TaskRepository.save(task) -> None`、`TaskRepository.get(task_id) -> Task | None`、`InMemoryTaskRepository`。

- [ ] 写测试证明状态变化由 repository 保存而不是 `Task.persist()` 空调用。
- [ ] 删除 `Task.persist()`，将保存时机放入 `Harness`。
- [ ] 用 Protocol 替代可实例化的持久化占位类；删除未使用的通用 kind 分发外观。
- [ ] 运行目标测试和完整测试套件。
- [ ] 提交 `refactor: inject task repository`。

### Task 5: 完成 Agent 默认组装和行为对象归位

**Files:**
- Modify: `src/demo/agent.py`
- Create: `src/core/hook.py`
- Modify: `src/schema/hook.py`
- Modify: `src/core/event.py`
- Test: `tests/unit/demo/test_agent.py`
- Test: `tests/unit/core/test_event.py`

**Interfaces:**
- Produces: `Agent.from_config(config)` 默认组装；`HarnessHook` 从 core 导出；事件失败可通过 logger 观测。

- [ ] 写测试覆盖显式假值依赖、默认组装和事件 emitter 异常日志。
- [ ] 将 `Agent` 中的默认选择改为显式 `is None`，集中组装端口实现。
- [ ] 将 hook 行为移出 schema，并保留兼容导出。
- [ ] 记录 emitter 异常后继续，不改变事件分发的容错语义。
- [ ] 运行目标测试和完整测试套件。
- [ ] 提交 `refactor: clarify agent composition root`。

### Task 6: 文档、清理与最终验证

**Files:**
- Modify: `README.md`
- Modify: `pyproject.toml` only if an existing verification tool requires configuration
- Modify: affected modules to remove dead parameters/imports

**Interfaces:**
- Produces: 默认使用和自定义注入示例；无死代码的公共契约。

- [ ] 删除确认未使用的字段、参数和过时兼容实现，保留必要迁移别名。
- [ ] 更新 README，展示 `Agent.from_config` 与显式 `Harness` 注入。
- [ ] 运行 `pytest -q -p no:cacheprovider`。
- [ ] 运行 `ruff check --no-cache src tests` 和格式检查。
- [ ] 检查 `git diff --check`、分支状态和相对 `origin/main` 的完整 diff。
- [ ] 提交 `docs: describe harness extension architecture`。
