# MCP Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Integrate FastMCP servers as lifecycle-managed, provider-safe, approval-gated Agent tools.

**Architecture:** `MCPManager` owns FastMCP clients, atomic catalog registration, active-call leases, and shutdown. Agent and ToolExecutor share one authoritative catalog; MCP tools execute through the async tool path and return a stable JSON-safe result envelope.

**Tech Stack:** Python 3.11, asyncio, FastMCP, Pydantic, pytest, Ruff, Mypy

**Spec:** `docs/superpowers/specs/2026-11-14-async-runtime-mcp-design.md`

## Global Constraints

- This plan starts only after `docs/superpowers/plans/2026-11-14-async-runtime.md` is complete.
- Python remains `>=3.11`; use the existing FastMCP dependency.
- MCP tools require approval by default and are never routed through local Workspace or SandboxRunner.
- Tool registration/removal is atomic from model/catalog consumers' perspective.
- Tool calls within one model response remain sequential.
- No MCP resources, prompts, OAuth, sampling, elicitation, or packaging/distribution work.

## Review Focus

- Unicode, punctuation, overlength, and normalization-colliding server/tool names must fail or normalize deterministically before catalog mutation; Task 2 tests them.
- Connect/list/register failure at every boundary must close the candidate and leave no manager/catalog residue; Task 2 tests each injection point.
- Remove/close racing an in-flight call must reject new calls, retain the client until the lease exits, and unregister immediately; Task 3 tests the race.
- Non-JSON structured content or content blocks must fail as `remote_error`, never corrupt task persistence; Task 2 tests normalization rejection.
- Injected Agent catalog/executor/manager mismatches must fail at construction rather than advertise an unexecutable tool; Task 4 tests every combination.

---

### Task 1: MCP tool metadata and default approval boundary

**Files:**
- Modify: `src/schema/tool.py`
- Modify: `src/core/tool.py`
- Modify: `src/tool/local_tools.py`
- Modify: `src/tool/tool_executor.py`
- Modify: `tests/unit/schema/test_tool.py`
- Modify: `tests/unit/tool/test_tool_executor.py`

**Interfaces:**
- Produces: `ToolOrigin` enum with `LOCAL`, `MCP`, and `CUSTOM`.
- Extends: `Tool.origin: ToolOrigin = ToolOrigin.CUSTOM`
- Extends: `Tool.requires_approval: bool = False`
- Produces: `ToolExecutionFailReason.TOOL_EXECUTION_FAIL_REASON_REMOTE_ERROR` with value `"remote_error"`.
- Produces: `RemoteToolError(output: Any)` in `src/core/tool.py`; async execution maps it to the
  `remote_error` fail reason while preserving its JSON-safe output.

- [x] **Step 1: Write failing metadata and policy tests**

Assert defaults preserve custom/local behavior, `requires_approval=True` returns pending under
`DefaultApprovalPolicy`, an explicit injected approval policy can approve it, and `RemoteToolError` becomes a
failed result that round-trips through task snapshots.

- [x] **Step 2: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/schema/test_tool.py tests/unit/tool/test_tool_executor.py -q -p no:cacheprovider`
Expected: FAIL on missing metadata and fail reason.

- [x] **Step 3: Implement metadata and policy enforcement**

Mark tools created by `LocalToolSuite` as `LOCAL`. Make `DefaultApprovalPolicy` return `None` for `requires_approval` before name-based rules; injected policies remain authoritative.

- [x] **Step 4: Run focused tests**

Run the Step 2 command.
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/schema/tool.py src/core/tool.py src/tool/local_tools.py src/tool/tool_executor.py tests/unit/schema/test_tool.py tests/unit/tool/test_tool_executor.py
git commit -m "feat: add remote tool approval metadata"
```

### Task 2: MCP identity and result normalization

**Files:**
- Create: `src/mcp/tool_adapter.py`
- Create: `tests/unit/mcp/test_tool_adapter.py`

**Interfaces:**
- Produces: `provider_tool_name(server_name: str, remote_name: str) -> str`
- Produces: `normalize_mcp_result(result: Any) -> dict[str, Any]`
- Consumes: `RemoteToolError` from Task 1 for `is_error=True` or invalid remote data.
- Produces: `make_mcp_tool(server_name, remote_tool, invoke) -> Tool` with origin MCP and approval required.

- [x] **Step 1: Write failing provider-name tests**

Assert exact `mcp__weather__forecast` output, allowed character set, deterministic maximum length, rejection of empty normalized segments, and detection-ready equality for normalization collisions.

- [x] **Step 2: Write failing result-envelope tests**

Assert exactly `content`, `structured_content`, and `is_error` keys; Pydantic content blocks serialize to JSON objects; `is_error=True`, opaque values, and non-JSON data raise `RemoteToolError` with useful JSON-safe output.

- [x] **Step 3: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/mcp/test_tool_adapter.py -q -p no:cacheprovider`
Expected: FAIL because the module does not exist.

- [x] **Step 4: Implement the adapter**

Use stable ASCII normalization plus a hash suffix when truncating. Keep all envelope keys. Validate with `json.dumps` rather than coercing opaque objects to strings. The produced Tool calls the supplied async `invoke` function and raises `RemoteToolError` for the executor to classify.

- [x] **Step 5: Run focused tests**

Run the Step 3 command.
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/mcp/tool_adapter.py tests/unit/mcp/test_tool_adapter.py
git commit -m "feat: normalize mcp tools and results"
```

### Task 3: Atomic MCP manager lifecycle

**Files:**
- Rewrite: `src/mcp/mcp_client.py`
- Modify: `tests/unit/mcp/test_mcp_client.py`
- Modify: `src/tool/tool_registry.py`
- Modify: `tests/unit/tool/test_tool_registry.py`

**Interfaces:**
- Consumes: adapter interfaces from Task 2 and `ToolCatalog`.
- Produces: `MCPManager(catalog: ToolCatalog, *, client_factory=Client)` and read-only `catalog`.
- Produces: `add_server`, `refresh_server`, `remove_server`, `aclose`, `__aenter__`, and `__aexit__` async methods.
- Produces: catalog collision preflight and exact unregister behavior without partially mutating the registry.

- [x] **Step 1: Replace current shallow tests with failing fake-client lifecycle tests**

Test successful atomic registration/call, duplicate server, catalog/name collision, failed connect, failed list, failed registration rollback, refresh replacement/rollback, remove, idempotent close, and stale tool invocation.

- [x] **Step 2: Add failing in-flight lease race tests**

Start a remote call, invoke remove/close, assert catalog removal and new-call rejection happen immediately, client close waits, and the started call may finish before transport close.

- [x] **Step 3: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/mcp/test_mcp_client.py tests/unit/tool/test_tool_registry.py -q -p no:cacheprovider`
Expected: FAIL against the existing unbound manager.

- [x] **Step 4: Add registry preflight support**

Add only the minimal catalog queries/strict registration needed for atomic MCP batches; preserve existing replacement behavior where current callers depend on it.

- [x] **Step 5: Implement manager state and lifecycle**

Use one lifecycle lock plus per-server state containing client, exit stack, registered ids, closing flag, active count, and condition. Build and validate the complete tool set before registration. On removal unregister first, wait for active leases, then close; always clear manager state even if close raises.

- [x] **Step 6: Run focused tests**

Run the Step 3 command.
Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add src/mcp/mcp_client.py src/tool/tool_registry.py tests/unit/mcp/test_mcp_client.py tests/unit/tool/test_tool_registry.py
git commit -m "feat: manage mcp server lifecycle atomically"
```

### Task 4: Authoritative Agent composition and MCP lifecycle surface

**Files:**
- Modify: `src/demo/agent.py`
- Modify: `tests/unit/demo/test_agent.py`
- Modify: `src/mcp/__init__.py`

**Interfaces:**
- Consumes: `MCPManager` from Task 3 and async Agent from the prerequisite plan.
- Produces: optional `mcp_manager` constructor parameter.
- Produces: `Agent.add_mcp_server`, `refresh_mcp_server`, `remove_mcp_server`, `aclose`, `__aenter__`, and `__aexit__`.
- Enforces: one authoritative catalog across Agent, executor, and manager.

- [x] **Step 1: Write failing composition tests**

Cover default catalog creation, executor-catalog reuse, explicit matching injection, and rejection of catalog/executor, catalog/manager, and executor/manager identity mismatches.

- [x] **Step 2: Write failing lifecycle tests**

Assert async context exit closes the manager, close is idempotent, operations/chat after close fail, and all Agent server methods delegate without hidden sync-loop creation.

- [x] **Step 3: Run focused tests and verify failure**

Run: `.venv\Scripts\python -m pytest tests/unit/demo/test_agent.py -q -p no:cacheprovider`
Expected: FAIL on missing composition/lifecycle behavior.

- [x] **Step 4: Implement authoritative composition and delegation**

Choose catalog in this order: explicit catalog, injected executor catalog, new registry. Reject every identity mismatch before registering local or MCP tools. Create a manager only when none is injected.

- [x] **Step 5: Run focused tests**

Run the Step 3 command.
Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add src/demo/agent.py src/mcp/__init__.py tests/unit/demo/test_agent.py
git commit -m "feat: integrate mcp lifecycle with agent"
```

### Task 5: End-to-end MCP approval, resume, and documentation

**Files:**
- Create: `tests/integration/test_mcp_agent.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: all preceding async runtime and MCP interfaces.
- Produces: documented supported MCP composition path.

- [x] **Step 1: Write a failing in-process FastMCP integration test**

Connect an in-process server with one structured tool, run `Agent.achat`, assert the MCP call pauses by default, approve it, call `aresume`, and assert JSON-safe output reaches model history and the final task completes. Verify context exit removes the tool.

- [x] **Step 2: Run the integration test and verify failure**

Run: `.venv\Scripts\python -m pytest tests/integration/test_mcp_agent.py -q -p no:cacheprovider`
Expected: FAIL until the real FastMCP path matches the fake-client unit contract.

- [x] **Step 3: Fix only adapter discrepancies exposed by the real server**

Keep the public envelope and lifecycle contracts unchanged; adapt FastMCP field access in `src/mcp/tool_adapter.py` or `src/mcp/mcp_client.py`.

- [x] **Step 4: Document MCP usage and trust boundaries**

Show `async with Agent(...)`, `await agent.add_mcp_server(...)`, `await agent.achat(...)`, approval/resume, shutdown, provider-safe names, and the explicit statement that remote MCP is not local sandbox execution.

- [x] **Step 5: Run all quality gates**

Run:

```powershell
.venv\Scripts\python -m pytest -q -p no:cacheprovider
.venv\Scripts\ruff format --check src tests
.venv\Scripts\ruff check --no-cache src tests
.venv\Scripts\mypy src
```

Expected: all tests pass; Ruff and Mypy report no errors.

- [x] **Step 6: Commit**

```bash
git add src/mcp src/demo/agent.py src/schema/tool.py src/tool tests/integration/test_mcp_agent.py README.md
git commit -m "feat: complete mcp agent integration"
```

完成状态：已通过 PR #7 合入 `main`，MCP 工具发现、调用与 Agent 集成路径均已落地。
