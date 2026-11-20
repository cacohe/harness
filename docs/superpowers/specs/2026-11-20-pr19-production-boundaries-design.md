# PR #19 生产边界加固设计

## 意图

修复全面复审确认的真实运行缺陷，使精简后的 Harness 在真实 LiteLLM、MCP、长上下文和运行失败条件下仍保持可用、可恢复且有界。此次不恢复旧 `Harness` API，也不把某一种工具、模型或持久化实现放回 `src/core`。

## 目标与约束

- `src/core` 只保存执行语义、值对象与扩展端口；provider、MCP 和 Artifact 编码细节留在 adapters/application。
- 真实 LiteLLM 的 `tool_calls=None` 必须按空列表处理；Harness 传入的是普通 provider tools，因此适配器必须绕过 LiteLLM 自带的 MCP handler。
- MCP 返回值必须在适配器边界转换为稳定 JSON 信封，成功与失败都保留远端内容。
- 上下文选择同时满足 token、消息数和序列化字节数约束；摘要检查点只能覆盖连续、实际参与压缩的交互。
- provider 工具调用 ID 只作为外部 ID；进入 Harness 后使用 run/step 命名空间，避免同一或跨 Run 冲突。
- factory、advance、journal bound 等异常不能把 Run 留在 `running`；失败码尽可能保留原始语义。
- MCP catalog refresh 后，transport 终止仍必须使当前 server generation 下线。
- `read_artifact` 以最终 JSON 编码大小为准限制返回，不能把读回结果再次外置成另一个 Artifact。

## 已选方案

1. **LiteLLM 边界规范化。** 在非流式与流式解析处把 `None` 归一为空列表；适配器拥有 `_skip_mcp_handler=True`，调用方不能覆盖，并用真实 LiteLLM response 类型做回归测试。
2. **MCP JSON 信封。** 递归支持 JSON、Pydantic `model_dump()`、dataclass 和常见 MCP content 项，输出 `content`、`structuredContent`、`isError`；协议错误文本保留在工具失败结果中。
3. **多维上下文预算。** 选择候选历史时用最终 `ModelRequest` 同级的消息数和 UTF-8 JSON 字节上限判定；压缩只接受与 durable checkpoint 首尾相接的连续片段，否则不生成会虚报覆盖范围的新 checkpoint。
4. **内部工具调用身份。** 在模型响应进入状态机前，以 `run_id`、模型 step 和 provider call ID 派生稳定内部 ID；历史投影与工具结果始终使用内部 ID，原始 ID只保留为适配器/观测信息（若当前值对象没有该字段则不扩张公共契约）。
5. **失败终态收口。** `_drive` 捕获非取消异常并尝试提交 `FailRun`；`JournalError` 保留其 code，其余边界异常映射为明确的 invariant/context/model failure。取消路径不得依赖已经无法构造的 model context。
6. **MCP generation 生命周期。** transport 终止回调按 client identity 关联当前 generation；refresh 只替换 catalog generation，不丢失 client 终止所有权。
7. **编码后 Artifact 预算。** `read_artifact` 在构造 JSON 值后检查编码大小并缩短 payload；配置校验继续提供防御，但不依赖固定的 4/3 估算。

## 非目标

- 不恢复 CLI、旧同步 API、旧 SQLite schema、内置文件/ shell 工具或 skill loader。
- 不在 core 中实现 provider 重试、MCP transport 或 Artifact repository。
- 不实现动态/前缀 `ToolRouter`。
- 本轮不补 reasoning delta、结构化输出或 `include_usage`；这些属于模型适配器后续能力，而不是本轮已确认的运行阻塞。

## 验收

- 真实 LiteLLM mock response 在流式/非流式和带普通 tools 时均可完成。
- 真实形状的 MCP `CallToolResult` 成功返回 JSON；`isError` 保留错误文本。
- 超过 token、1024 条消息、8 MB 或 open tool message 上限时，历史被裁剪或 Run 明确失败，不留下 `running`。
- 摘要 checkpoint 的覆盖范围连续且每段内容实际进入 compactor。
- provider 重复 `call_0` 不会导致同一或后续 Run 冲突。
- refresh 后 transport 终止会清理 catalog。
- 读满配置上限的二进制 Artifact 不会产生二次 Artifact。

