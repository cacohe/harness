# PR #19 第二轮评审修复设计

## 意图

修复第二轮评审确认的三个阻塞问题，并一并处理与这些边界直接相关的低风险一致性问题：同一 Run 的消息顺序、Artifact 退役后的再次发布、MCP async context 的 task 所有权、远端工具名规范化，以及 development 组合的 Artifact 能力一致性。

## 目标与约束

- `src/core` 继续只保存 Harness 执行语义和端口，不引入某种 MCP、文件系统或 SQLite 实现。
- 工具调用后的下一次模型请求固定为 `system, ...history, current user, assistant tool call, tool result`。
- 已退役内容可以再次发布，但 `RETIRING` 期间仍禁止附加，且不能通过恢复旧墓碑的方式绕过对象可读性边界。
- async context 的 `__aenter__`、客户端方法和 `__aexit__` 必须在同一个 owner task 中执行。
- 任意合法 MCP 远端名称都映射为 provider 可接受、稳定且不超过 64 ASCII bytes 的公开名称；调用仍使用原始远端名称；规范化碰撞必须拒绝整个 catalog。
- development 默认组合提供 Artifact maintenance，并将内部 externalizer 的 inline/summary 阈值收敛到 `RunLimits.max_inline_result_bytes`。
- 不实现动态或前缀 `ToolRouter`；调用方仍可按公开端口组合多个 gateway。

## 已选方案

1. **上下文顺序：修正强制后缀契约。** `current_input` 位于 open tool batch 之前，core 的 shape validator 与 adapter 使用同一顺序。
2. **Artifact：显式发布代次。** journal 增加“确认新发布”的端口；外置器在 repository `put()` 成功且可读后登记发布。`RETIRED -> AVAILABLE` 只允许由这一显式动作开启下一代，普通 attach 继续拒绝墓碑。发布和 attach 之间若被新一轮 retirement 抢占，attach 仍冲突而不会提交悬空引用。
3. **MCP 生命周期：owner task actor。** async context wrapper 创建一个常驻 task，由队列接收方法调用和 close；enter/call/exit 全部在该 task 内串行执行。同步客户端继续使用现有有界 worker。
4. **工具名：稳定别名。** 保留安全字符，将其他连续字符替换为 `_`，必要时使用远端名 SHA-256 短后缀截断；公开长度根据 server 前缀动态计算。规范化后重复即拒绝。
5. **development：能力对齐。** 仅对内部构造的 policy 做阈值收敛；调用方注入的 externalizer 不被改写。repository/journal 具备维护端口时暴露 `agent.artifacts`。

## 考虑过但未采用

- **Artifact attach 时直接复活墓碑：** 改动最小，但无法证明对象仍存在，正是上一轮已修复的悬空引用缺陷。
- **为每次 put 生成随机 Artifact ID：** 可以彻底隔离代次，但会破坏当前 content-addressed 去重契约并扩大文件系统迁移范围。
- **MCP 只让 enter/exit 同 task、调用仍外置：** anyio 客户端的方法通常也依赖 enter task 的 cancel scope，因此生命周期必须整体归 owner task。

## 验收

- 真实 bounded context 的多步工具请求消息顺序正确。
- memory 与 SQLite journal 都能在 retire 后通过显式 republish 再次 attach；未 republish 的 attach 仍冲突。
- anyio task group 模拟客户端可 add/call/remove，且 enter/call/exit task identity 相同。
- `server.tool`、`path/to/tool`、超长 Unicode 名可调用，且规范化碰撞被拒绝。
- development 默认 `agent.artifacts` 可用，小 Run inline limit 会正常外置而不是 `ARTIFACT_FAILED`。

