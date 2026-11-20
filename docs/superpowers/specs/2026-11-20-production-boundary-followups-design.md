# 生产边界遗留问题修复设计

## 意图

修复 PR #19 合入后仍存在的三项生产问题：模型请求未知工具会终止整个 Run、MCP `isError` 的错误内容无法进入模型上下文，以及进程崩溃遗留的 `.put-*` Artifact staging 目录无法回收。继续保持 `src/core` 精简：core 只表达通用执行语义，MCP 和文件系统策略留在 adapters。

## 目标与非目标

- 未知工具调用只失败该次工具调用，Run 保持可继续，且未知调用绝不进入 ToolGateway。
- MCP 协议明确返回的错误内容以有界、持久化、模型可见的详情保留；任意 adapter 异常仍只暴露稳定错误码，避免泄露内部异常。
- Artifact GC 能回收超过调用方 cutoff 的崩溃 staging，同时不会删除仍由 writer 持有租约的目录，也不会跟随 symlink、reparse point 或 hard link。
- 不改变历史 Run 的取消语义，不改变最后一个模型 step 产生工具后的执行语义。
- 不恢复旧 API、动态 ToolRouter 或 core 内重试。
- 当前没有生产兼容负担，因此允许同步更新当前 SQLite journal 编码，不增加旧编码兼容分支。

## 方案比较

### 未知工具

1. **将未知名填入伪造 catalog binding 后交给 gateway。** 改动小，但把“绝不能执行”的安全结论依赖给每个 ToolGateway 实现，拒绝。
2. **将整个模型响应判为失败。** 是当前行为，破坏模型自修复能力，拒绝。
3. **在持久化 `ToolCallRequest` 上记录 availability。** core 能在 gateway 之前生成稳定失败结果，同时 transcript 保留原调用。选择此方案。

### MCP 错误详情

1. **把 `isError` 当成功结果。** 内容能到模型，但丢失 Harness 的失败语义和观测分类，拒绝。
2. **把任意异常文本放入 `RunFailure`。** 会泄露路径、凭据和内部异常，拒绝。
3. **增加显式的 model-safe failure detail 通道。** 只有 adapter 主动抛出的 `ToolReportedError` 可进入该通道；详情有 UTF-8 字节上限并随 ToolResult、RunState 和 journal 保持。选择此方案。

### Artifact staging 回收

1. **启动时无条件删除 `.put-*`。** 会误删其他进程的活跃写入，拒绝。
2. **仅按 mtime 删除。** 对停顿中的长写入仍有竞态，拒绝。
3. **分片租约锁加 cutoff。** `.put-<uuid>` 按 UUID 前缀映射到固定 256 个锁；writer 在整个 staging 生命周期持锁，GC 获取同一锁后重新校验名称、目录身份、成员类型及 mtime，再执行删除。选择此方案，锁文件数量有固定上界。

## 详细设计

### 未知工具投影

`ToolCallRequest` 增加 kw-only `available: bool = True`。已知工具继续冻结 catalog 中的 approval binding；未知工具保存原始 name/arguments，使用不可执行的内部 binding，设置 `available=False` 和 `requires_approval=False`。`RunCoordinator._execute()` 在构造 `ToolRequest` 前检查该字段，直接返回 `TOOL_FAILED`。因此未知工具不会请求人工批准、不会调用 adapter，也不会终止 Run；下一模型回合会看到该工具调用的失败 transcript。

### 有界错误详情

`RunFailure` 增加可选 `detail`，只接受不超过 64 KiB 的 UTF-8 文本。`message` 仍由稳定错误码生成，公共错误文案不受 adapter 控制。`ToolResult`、state snapshot、receipt snapshot 和 SQLite codec 必须完整保留 detail；tool transcript 在存在 detail 时增加 `detail` 字段。

adapter 内新增 `ToolReportedError`，表示其 payload 已被 adapter 判断为可以返回模型。`LocalToolGateway` 只对该异常构造带 detail 的 `TOOL_FAILED`；`except Exception` 仍返回无 detail 的稳定失败。MCP adapter 对规范化后的 `isError` 信封进行 JSON 编码并抛出 `ToolReportedError`。超过 64 KiB 时由 MCP adapter 生成有界预览，避免无界状态和 journal 增长。

### staging 租约与回收

filesystem platform 提供可跨异步阶段持有的 object-lock handle，现有 `object_lock()` 复用它。Artifact writer 在创建 staging 前异步获取 `stage-00` 至 `stage-ff` 之一，并持有到发布或清理结束；不会同步阻塞 event loop。`collect_orphans()` 扫描严格匹配 `\.put-[0-9a-f]{32}` 的条目，对 mtime 早于 cutoff 的候选获取相同分片锁，然后从 pinned root 重新验证。

只有真实目录、成员集合属于 `data`/`descriptor.json`、成员都是单链接普通文件且所有相关 mtime 都早于 cutoff 时才删除。活跃 writer、发生变化的对象、symlink/reparse point、hard link、未知成员和损坏目录全部保留供重试或检查。成功回收的 staging 计入 `collect_orphans()` 返回数量。

## 验收

- 模型同时请求未知和已知工具时，未知调用不进入 gateway、已知调用照常执行，Run 不进入终态失败，后续模型能看到 `tool_failed`。
- 普通 handler `RuntimeError("secret")` 的文本仍不会进入 ToolResult 或 transcript。
- MCP `isError` 的规范化内容经过 LocalToolGateway、RunCoordinator、journal round-trip 后仍出现在 tool transcript 中，并受 64 KiB 上限约束。
- 旧的 `.put-*` 崩溃目录可由 `collect_orphans()` 删除；持有同分片租约的活跃 staging、过新的 staging 和不安全文件系统对象不会被删除。
- unit tests、integration tests、ruff 和 mypy 全部通过。
