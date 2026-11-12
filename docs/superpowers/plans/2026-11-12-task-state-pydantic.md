# Task State Pydantic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Harness 的任务状态统一为可验证、可 JSON 序列化和可恢复的 Pydantic 模型，并确保所有终止路径的返回值、Task 对象与仓库快照一致。

**Architecture:** 先把任务快照依赖的 schema 对象迁移为 Pydantic，再把 `Task` 迁移为带状态不变量的 Pydantic 聚合根。`InMemoryTaskRepository` 存储 JSON-safe 快照，Harness 在每个终止路径完整更新状态后再保存。

**Tech Stack:** Python 3.11、Pydantic 2、pytest、mypy、ruff

**Spec:** `docs/superpowers/specs/2026-11-12-task-state-pydantic-design.md`

## Global Constraints

- 保持现有同步 Harness、审批流程、事件顺序和 CLI 用户体验不变。
- `Task(...)`、`Task(original_input=...)`、`TaskResult(...)`、`to_dict()` 与现有状态方法继续可用。
- `COMPLETED` 和 `FAILED` 必须拥有 `TaskResult`，且 `TaskResult.output` 始终为 `str`。
- `step_count >= 0`，`elapsed_seconds >= 0`，不同任务不得共享可变默认值。
- `Tool` callable 不进入任务快照；不可 JSON 序列化的工具输出必须明确失败。
- 每个生产代码改动严格执行 RED、GREEN、REFACTOR。

## Review Focus

- JSON 数据中的 `state_reason="unknown"` 同时存在于暂停和失败枚举时，必须依据 `state` 恢复成正确枚举；Task 2 覆盖。
- 工具输出包含不可 JSON 序列化对象时，仓库保存必须明确失败且不得留下部分快照；Task 3 覆盖。
- 已保存对象和读取对象被调用方修改时，仓库内部快照不得变化；Task 3 覆盖。
- finalization 自身失败时，Task、返回结果和仓库快照必须保存同一错误文本；Task 4 覆盖。
- LLM 以 `STOP` 结束但 `content=None` 时，完成结果和后续 transcript 必须使用空字符串而不是 `None`；Task 4 覆盖。

---

### Task 1: 将快照依赖的 schema 统一为 Pydantic

**Files:**
- Modify: `src/schema/context.py`
- Modify: `src/schema/llm.py`
- Modify: `src/schema/tool.py`
- Test: `tests/unit/schema/test_llm_stream.py`
- Test: `tests/unit/core/test_task.py`

**Interfaces:**
- Consumes: 现有 `Message(...)`、`Context(...)`、`LlmToolCall(...)`、`LlmUsage(...)`、`LLMOutput(...)`、`ToolExecutionResult(...)` 构造方式。
- Produces: 上述状态对象继承 `BaseModel`，支持 `model_dump(mode="json")` 与 `model_validate()`；`Tool` 继续保留 callable 运行时模型。

- [ ] **Step 1: 为嵌套状态模型增加 round-trip 失败测试**

在 `tests/unit/core/test_task.py` 增加测试，构造包含 assistant tool call、reasoning、usage 和 pending tool result 的数据，分别断言 `Message`、`LLMOutput`、`ToolExecutionResult` 的 JSON dump 是基础 JSON 类型，且 `model_validate()` 恢复后字段相等。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_task.py -q -p no:cacheprovider`

Expected: FAIL，现有 dataclass 没有 `model_dump` 或 `model_validate`。

- [ ] **Step 3: 迁移 schema 模型**

在对应文件中将 `Message`、`Context`、`LlmToolCall`、`LlmUsage`、`LLMOutput`、`ToolExecutionResult` 改为 `BaseModel`。使用精确容器类型和 `Field(default_factory=list)`；`Context` 允许 `Tool` 等运行时任意类型；保留位置参数调用兼容所需的显式构造处理，或同步将仓库内调用点改为关键字参数。

- [ ] **Step 4: 运行 schema、LLM、tool 测试并确认 GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\schema tests\unit\llm tests\unit\tool tests\unit\core\test_task.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 运行 mypy 并修正本任务引入的类型问题**

Run: `.\.venv\Scripts\mypy.exe src`

Expected: `Success: no issues found`。

- [ ] **Step 6: 提交 schema 迁移**

```powershell
git add src/schema/context.py src/schema/llm.py src/schema/tool.py tests/unit/schema/test_llm_stream.py tests/unit/core/test_task.py
git commit -m "refactor: model task state schemas with pydantic"
```

### Task 2: 将 Task 聚合根迁移为 Pydantic 并建立不变量

**Files:**
- Modify: `src/schema/task.py`
- Modify: `src/core/task.py`
- Test: `tests/unit/core/test_task.py`

**Interfaces:**
- Consumes: Task 1 的 Pydantic `Message` 与 `ToolExecutionResult`。
- Produces: `TaskResult(BaseModel)`；`Task(BaseModel)`；`Task.from_dict(data: Mapping[str, Any]) -> Task`；JSON-safe `Task.to_dict() -> dict[str, Any]`。

- [ ] **Step 1: 增加 Task round-trip 与不变量失败测试**

测试必须覆盖：完整 Task 经 `json.dumps(task.to_dict())` 后可 `Task.from_dict()` 恢复；位置与关键字 `original_input` 均可构造；负进度被拒绝；completed/failed 缺少 result 被拒绝；`state=failed,state_reason="unknown"` 恢复为 `TaskFailReason.TASK_FAILED_UNKNOWN`；不同任务集合互不共享。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_task.py -q -p no:cacheprovider`

Expected: FAIL，当前 `Task.to_dict()` 不是 JSON-safe，且不存在 `from_dict()`/Pydantic 验证。

- [ ] **Step 3: 实现 `TaskResult` 与 `Task` Pydantic 模型**

`TaskResult.output: str = ""`。`Task` 增加 `schema_version: int = 1`、显式字段、非负约束、按 state 解析 `state_reason` 的 validator、终态 result validator、兼容现有位置参数的 `__init__`、`to_dict()` 与 `from_dict()`。保留状态方法并在完成态清除旧 reason。

- [ ] **Step 4: 运行 Task 测试并确认 GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_task.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 运行 core 测试确认兼容性**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 6: 提交 Task 聚合根迁移**

```powershell
git add src/schema/task.py src/core/task.py tests/unit/core/test_task.py
git commit -m "refactor: enforce pydantic task state invariants"
```

### Task 3: 为 InMemoryTaskRepository 建立快照语义

**Files:**
- Modify: `src/persistence/in_memory.py`
- Test: `tests/unit/core/test_persistence.py`

**Interfaces:**
- Consumes: `Task.to_dict()` 与 `Task.from_dict()`。
- Produces: `InMemoryTaskRepository.save(task: Task) -> None` 保存独立 JSON-safe 快照；`get(task_id: str) -> Task | None` 返回新 Task。

- [ ] **Step 1: 增加仓库快照失败测试**

测试必须断言：保存后修改原 Task 不影响仓库；修改 `get()` 返回的 Task 不影响下一次读取；包含不可 JSON 序列化工具输出时 `save()` 抛出序列化错误且不覆盖之前的有效快照。

- [ ] **Step 2: 运行测试并确认 RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_persistence.py -q -p no:cacheprovider`

Expected: FAIL，当前仓库保存并返回同一对象引用。

- [ ] **Step 3: 实现快照存储**

将 `_tasks` 改为 `dict[str, dict[str, Any]]`。`save()` 先完整生成并通过 `json.dumps()` 验证快照，再原子替换字典项；`get()` 通过 `Task.from_dict()` 恢复新对象。

- [ ] **Step 4: 运行 persistence 测试并确认 GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_persistence.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交仓库快照语义**

```powershell
git add src/persistence/in_memory.py tests/unit/core/test_persistence.py
git commit -m "fix: persist independent task snapshots"
```

### Task 4: 统一 Harness 终止路径的状态、结果和持久化

**Files:**
- Modify: `src/core/loop.py`
- Modify: `src/demo/agent.py`
- Test: `tests/unit/core/test_loop.py`
- Test: `tests/unit/demo/test_agent.py`

**Interfaces:**
- Consumes: Task 2 的终态不变量与 Task 3 的快照仓库。
- Produces: `Harness.run(task: Task) -> TaskResult`；所有终止路径返回 `task.result` 同一对象语义，仓库读取值与其字段相等；Agent transcript 只包含字符串。

- [ ] **Step 1: 增加完成、失败、澄清和异常一致性失败测试**

测试分别断言：完成和 max-step 失败会记录进度；失败 finalization 输出写回 Task 与仓库；澄清完成态被保存；普通异常的返回结果、Task.result 和仓库快照文本一致；`STOP + content=None` 归一为空字符串。

- [ ] **Step 2: 增加 finalization 失败与 transcript 失败测试**

测试断言 finalization 抛错时仍保存统一错误结果；完成空内容后下一轮 transcript 不包含字面量 `None`。

- [ ] **Step 3: 运行目标测试并确认 RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_loop.py tests\unit\demo\test_agent.py -q -p no:cacheprovider`

Expected: FAIL，当前终止路径保存顺序和 nullable output 不满足断言。

- [ ] **Step 4: 重排终止路径写入顺序**

在 `turn()` 每个返回分支前记录进度。完成、失败、澄清和异常路径都创建字符串 `TaskResult`、写入 Task、转换状态、保存、发事件并返回 `task.result`。`Harness.run()` 返回类型恢复为 `TaskResult`，遇到非法 finished task 由 Task 模型验证阻止。

- [ ] **Step 5: 修正 Agent transcript**

将 transcript 恢复为 `list[tuple[str, str]]`，只记录满足终态不变量的字符串结果。

- [ ] **Step 6: 运行目标测试并确认 GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\unit\core\test_loop.py tests\unit\demo\test_agent.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 7: 提交 Harness 一致性修复**

```powershell
git add src/core/loop.py src/demo/agent.py tests/unit/core/test_loop.py tests/unit/demo/test_agent.py
git commit -m "fix: persist consistent terminal task results"
```

### Task 5: 完整验证与文档收尾

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-11-12-task-state-pydantic.md`

**Interfaces:**
- Consumes: Tasks 1-4 的完整实现。
- Produces: 通过全部质量门禁的第一阶段交付物和更新后的状态持久化说明。

- [ ] **Step 1: 将 README 的持久化说明更新为快照语义**

说明默认仓库仅限单进程，但保存/读取使用可序列化独立快照；真实 durable repository 可实现同一 `TaskRepository` 端口。

- [ ] **Step 2: 运行完整质量门禁**

Run:

```powershell
.\.venv\Scripts\ruff.exe format --check . --no-cache
.\.venv\Scripts\ruff.exe check . --no-cache
.\.venv\Scripts\mypy.exe src
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected: 所有命令退出码为 0，无 warnings；测试数不少于当前 115。

- [ ] **Step 3: 更新计划复选框并提交文档**

```powershell
git add README.md docs/superpowers/specs/2026-11-12-task-state-pydantic-design.md docs/superpowers/plans/2026-11-12-task-state-pydantic.md
git commit -m "docs: describe pydantic task snapshots"
```
