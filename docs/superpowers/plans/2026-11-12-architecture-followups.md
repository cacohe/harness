# Architecture Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 建立全项目静态类型门禁、统一事件端口、删除已确认不再需要的兼容层，并让架构文档准确反映实现状态。

**Architecture:** `Harness` 依赖结构化的 `EventSink`，默认由具体 `EventHandler` 实现。`mypy` 检查整个 `src` 包并进入 CI；0.1.0 阶段直接删除没有仓库内生产调用方的兼容导出和未使用协议。

**Tech Stack:** Python 3.11、pytest、ruff、mypy、uv、GitHub Actions

**Spec:** `docs/superpowers/specs/2026-11-12-architecture-followups-design.md`

## Global Constraints

- 所有仓库修改只发生在 `F:\CodeSpace\harness-worktrees\architecture-followups`。
- 保持工具审批、暂停恢复、CLI 和 Agent 的用户可见行为不变。
- `mypy` 覆盖整个 `src`，不使用全局 `ignore_errors`。
- 兼容导出本次直接删除，不增加新的弃用层。
- 新 PR 创建后才清理旧 C 盘 worktree 和旧功能分支。

## Review Focus

- 假值事件实现必须原样注入，不能被默认 `EventHandler` 替换。
- 结构匹配但不继承的事件实现必须满足 `EventSink`。
- 删除兼容模块后，仓库内不得残留旧导入。
- `mypy` 必须检查整个 `src`，CI 与本地命令必须一致。
- 文档复选框和验证命令必须与实际完成状态一致。

---

### Task 1: 建立全项目 mypy 门禁

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: mypy 报告的 `src/**/*.py` 文件，仅做必要类型修正

**Interfaces:**
- Produces: `uv run --locked mypy src` 零错误的 CI 门禁。

- [x] **Step 1: 运行缺失的类型检查命令并记录预期失败**

Run: `uv run --locked mypy src`
Observed: FAIL；环境中可发现未锁定的 `mypy`，并报告 14 个项目类型错误。随后把
`mypy` 加入项目锁定依赖，避免 CI 依赖外部 PATH 状态。

- [x] **Step 2: 添加 mypy 开发依赖和项目配置**

在 `pyproject.toml` 增加 `mypy>=1.15`；配置 Python 3.11、`check_untyped_defs = true`、`no_implicit_optional = true`、`warn_unused_ignores = true`，不启用全局忽略。

- [x] **Step 3: 更新锁文件并运行全项目检查**

Run: `uv lock && uv run --locked mypy src`
Observed: 锁定版首次检查报告 13 个真实类型问题。

- [x] **Step 4: 最小化修正 mypy 报告的问题**

只修正类型签名、窄化和第三方边界注解；不改变运行时行为。

- [x] **Step 5: 把 mypy 加入 CI 和 README 验证命令**

CI command: `uv run --locked mypy src`

- [x] **Step 6: 验证并提交**

Run: `uv run --locked mypy src && uv run --locked python -m pytest -q -p no:cacheprovider`
Expected: PASS。

Commit: `build: add project-wide mypy gate`

### Task 2: 引入 EventSink 端口

**Files:**
- Modify: `src/core/event.py`
- Modify: `src/core/loop.py`
- Modify: `tests/unit/core/test_event.py`
- Modify: `tests/unit/core/test_ports.py`
- Modify: `tests/unit/core/test_loop.py`

**Interfaces:**
- Produces: `EventSink.emit(event: Event, channel: EventChannel = EventChannel.CLIENT) -> None`。
- Consumes: Task 1 的全项目 mypy 门禁。

- [x] **Step 1: 写结构化端口和假值注入的失败测试**

增加 `test_event_sink_accepts_structural_implementation` 和 `test_harness_preserves_falsey_event_sink`；后者断言 Harness 保存并调用同一个假值 sink。

- [x] **Step 2: 运行测试并确认按预期失败**

Run: `uv run --locked python -m pytest tests/unit/core/test_ports.py tests/unit/core/test_loop.py -q`
Expected: FAIL，因为 `EventSink` 不存在且假值 sink 被替换。

- [x] **Step 3: 实现 EventSink 并修正默认选择**

在 `src/core/event.py` 添加 `@runtime_checkable class EventSink(Protocol)`；将 Harness 参数改为 `EventSink | None`，并使用 `is not None` 选择注入值。

- [x] **Step 4: 运行目标测试、mypy 和完整测试**

Run: `uv run --locked python -m pytest tests/unit/core/test_event.py tests/unit/core/test_ports.py tests/unit/core/test_loop.py -q && uv run --locked mypy src && uv run --locked python -m pytest -q -p no:cacheprovider`
Expected: PASS。

- [x] **Step 5: 提交**

Commit: `refactor: define event sink port`

### Task 3: 删除兼容导出和未使用协议

**Files:**
- Modify: `src/core/tool.py`
- Delete: `src/core/sandbox.py`
- Modify: `src/core/persistence.py`
- Delete: `src/schema/hook.py`
- Modify: `tests/unit/demo/test_agent.py`
- Create: `tests/unit/core/test_legacy_exports.py`
- Modify: references found by `rg` in `src`, `tests`, `README.md`, and `docs`

**Interfaces:**
- Removes: `ToolRegistry`、`Sandbox`、`src.schema.hook.HarnessHook`、`CheckpointStore`、`ArtifactStore`、`TraceStore`。
- Retains: `ToolCatalog`、`SandboxRunner`、`src.core.hook.HarnessHook`、`TaskRepository`。

- [x] **Step 1: 写兼容导出仍存在的失败测试**

测试断言 `src.core.tool` 不再暴露 `ToolRegistry`，两个兼容模块无法发现，三个未使用 Store 名称不存在。

- [x] **Step 2: 运行测试并确认按预期失败**

Run: `uv run --locked python -m pytest tests/unit/core/test_legacy_exports.py -q`
Expected: FAIL，列出当前仍存在的兼容导出。

- [x] **Step 3: 删除兼容导出、模块和未使用协议**

同步将测试与文档中的 hook 导入改为 `src.core.hook`，清除不再使用的导入。

- [x] **Step 4: 验证没有旧引用**

Run: `rg -n "\bToolRegistry\b|core\.sandbox|schema\.hook|\bCheckpointStore\b|\bArtifactStore\b|\bTraceStore\b" src tests README.md docs`
Expected: 只允许删除回归测试、迁移说明和历史计划中明确描述已删除名称的文本。

- [x] **Step 5: 运行目标测试、mypy 和完整测试**

Run: `uv run --locked python -m pytest tests/unit/core/test_legacy_exports.py tests/unit/demo/test_agent.py -q && uv run --locked mypy src && uv run --locked python -m pytest -q -p no:cacheprovider`
Expected: PASS。

- [x] **Step 6: 提交**

Commit: `refactor: remove obsolete compatibility exports`

### Task 4: 同步文档、主机规则并完成验证

**Files:**
- Modify: `docs/superpowers/plans/2026-11-12-core-architecture-refactor.md`
- Modify: `docs/superpowers/specs/2026-11-12-core-architecture-design.md`
- Modify: `docs/superpowers/plans/2026-11-12-architecture-followups.md`
- Modify: `README.md`
- Host config: `C:\Users\25685\.codex\AGENTS.md`
- Host config: `C:\Users\25685\.codex\config.toml`

**Interfaces:**
- Produces: 准确的完成状态和持久的 F 盘 worktree 目录规则。

- [x] **Step 1: 回填原计划并更新架构说明**

把已完成的原计划复选框改为 `[x]`；将事件端口和 mypy 命令更新为实际实现；记录已删除的兼容导出。

- [x] **Step 2: 持久化 worktree 目录规则**

在全局 `AGENTS.md` 写明新 worktree 必须位于 `F:\CodeSpace\harness-worktrees`；在现有 `config.toml` 的 `sandbox_workspace_write.writable_roots` 中追加该目录，不覆盖其他配置。

- [x] **Step 3: 运行全部验证**

Run: `uv run --locked python -m pytest -q -p no:cacheprovider`
Run: `uv run --locked ruff check --no-cache src tests`
Run: `uv run --locked ruff format --check src tests`
Run: `uv run --locked mypy src`
Run: `git diff --check origin/main..HEAD`
Expected: 全部退出码为 0。

- [x] **Step 4: 提交文档**

Commit: `docs: close architecture follow-up plan`

- [ ] **Step 5: 创建 PR 并清理旧工作区**

推送 `codex/architecture-followups` 并创建面向 `main` 的 PR。确认旧 C 盘 worktree 干净后，从主工作区删除它；删除本地和远程 `codex/core-architecture-refactor` 分支。保留新的 F 盘 worktree 处理 PR 反馈。
