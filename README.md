# agent-harness

English | [简体中文](README.zh-CN.md)

A production-grade, async-first Harness core. The project clearly separates execution semantics, application use cases, and concrete infrastructure:

- `src/core`: five capability packages for execution, context, tools, Artifacts, and persistence, plus shared bounded values;
- `src/application`: Session ownership, idempotent submission, runtime lifecycle, and the composition root;
- `src/adapters`: SQLite, filesystem Artifacts, LiteLLM, Docker, MCP, context policies, and observability implementations.

The project is in the `0.x` stage. It exposes only the current architecture and does not retain the legacy Harness API or legacy SQLite schema.

## Environment

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

```powershell
uv sync --locked
```

The Harness does not read environment variables implicitly, nor does Core choose a model, database, or working directory. Callers may load `.env` themselves and then explicitly construct adapters and `ApplicationConfig`.

## Async API and Explicit Composition

```python
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.application import ApplicationConfig, create_production_agent
from src.core.context.contracts import ContextBudget
from src.core.execution.commands import RunLimits
from src.core.ids import ProfileId
from src.core.artifacts.lifecycle import ArtifactAttachmentRole

config = ApplicationConfig(
    limits=RunLimits(max_steps=32, max_elapsed_seconds=300.0, max_inline_result_bytes=65_536),
    context_budget=ContextBudget(
        context_window_tokens=128_000,
        reserved_output_tokens=4_096,
        safety_margin_tokens=1_024,
    ),
    sqlite_path=Path("C:/absolute/state/harness.db"),
    tool_execution_path=Path("C:/absolute/state/tool-executions.db"),
    artifact_root=Path("C:/absolute/state/artifacts"),
    max_artifact_read_bytes=65_536,
    max_orphan_reference_ids=1_000_000,
)
orphan_cutoff_utc = datetime.now(timezone.utc) - timedelta(days=1)

agent = await create_production_agent(
    config,
    model=model_gateway,
    tools=tool_gateway,
    context=context_provider,
    context_counter=token_counter,
    context_request_factory=request_factory,
    ids=id_generator,
    clock=clock,
    profile_id=ProfileId("default"),
)

async with agent:
    session = await agent.open_session()
    state = await session.submit("Analyze this dataset")
    if state.status.value == "waiting_approval":
        state = await session.decide_tool_call(state.run_id, "call-1", approved=True)
    detached = await session.detach_artifacts(state.run_id)
    if detached:
        await agent.artifacts.retire(detached[0].artifact.artifact_id)
    collected = await agent.artifacts.collect_orphans(orphan_cutoff_utc)
```

`create_development_agent()` uses an in-memory journal and an in-memory Artifact repository. Production creates a filesystem repository only when `artifact_repository` is omitted; that default requires an absolute `artifact_root`. An injected repository may be remote and needs no local root, but it requires an explicit opaque `artifact_workspace_id`, an externalizer, and any maintenance capabilities the deployment needs. Artifact ownership, retirement, and reference-index ports are separate from `RunJournal`, while production requires their adapter object to be the same atomic persistence authority as the journal. Production uses a separate `tool_execution_path` to persist exactly-once tool claims. When it is not provided explicitly, a sibling `*.tools` file is derived from the explicit `sqlite_path`; the current working directory is never used. Both factories are async-only, and resources are created and closed on the caller's event loop.

At startup, the builder performs abandoned-claim recovery on the tool execution store. Production startup for the same `tool_execution_path` must be serialized and owned by a single process. For multi-process or multi-node execution, replace the reference SQLite store with an external ToolGateway that provides leases or coordination instead of sharing it across builders.

Production composition always reserves the `read_artifact` name and routes it to a bounded `ReadArtifactTool`. The supplied `ToolGateway` is the fallback for all other tools. `ContextRequestFactory` must not declare `read_artifact` again; composition injects the trusted built-in spec into every request. Passing `tools=None` creates a minimal tool composition that allows only Artifact reads.

## Bounded Execution and Context

Each `RunCoordinator.advance()` performs at most one model call or one durable tool batch and commits it atomically through journal CAS. The outer loop is driven by the application runtime and constrained by:

- `RunLimits.max_steps`, `max_elapsed_seconds`, and `max_inline_result_bytes`;
- `ApplicationConfig.max_advances_per_operation`;
- `ApplicationConfig.max_serialized_run_state_bytes`;
- `ContextBudget.input_allowance_tokens`.

If the last allowed model decision returns any tool calls, Core terminates the Run with `limit_exceeded` before persisting or executing that batch. This applies to ordinary tools, approval-bound tools, Artifact-producing tools, and `load_skill`; a result that no later model step can consume therefore causes no tool-side effect.

`BoundedContextProvider` builds requests through a paginated reader and authenticated coverage. Histories with 10,000 or more records are not copied into `RunState`; the SQLite reader uses indexed pagination and bounded authenticated aggregation. Summary checkpoints record the source range, policy version, profile, and Artifact references. If the budget cannot be satisfied, the provider returns the stable `context_budget_exceeded` failure and does not send an oversized request to the model.

## Progressive Skills

Both composition factories accept an optional `SkillProvider`. When configured, every model decision receives one bounded, deterministic catalog SYSTEM message containing only each Skill's `name`, `description`, and `version`, plus a reserved `load_skill` tool. The model selects a validated logical name; the reserved call is handled by Core and is never sent to the ordinary `ToolGateway`.

A successful activation persists only `name + version + content_sha256` in `RunState`. The full SYSTEM instructions remain active for the rest of that Run: before every later model decision, including after restart, the provider must resolve exactly the persisted binding and return content whose canonical digest matches it. Changed or missing content fails the Run before another model request instead of silently hot-reloading. New Runs may select newer versions. Catalogs, active bindings, messages, and confirmation results are bounded, and activated content still counts against the normal context budget.

`FilesystemSkillProvider` is the single-machine reference loader. It expects `<absolute-root>/<logical-name>/SKILL.md` with exact `name`, `version`, and `description` frontmatter. Reads run off the event loop and reject traversal names, symlinks/reparse points, hard-linked or non-regular files, malformed or duplicate metadata, invalid UTF-8, NUL, empty/oversized bodies, and entries changed during a validated read. It performs no recursive discovery, remote registry lookup, automatic activation, or hot reload.

## Artifact Lifecycle

Large tool results and large final answers never enter `RunState`, domain events, or the model context body. The flow is fixed:

1. `ArtifactExternalizer` applies policy to choose inline storage or externalization;
2. the repository writes the complete object and verifies its hash and size;
3. the repository assigns a distinct, immutable publication ID to every successful write, after which the journal commits only `ArtifactRef + bounded summary`;
4. when the model needs the content, it reads a bounded range through the ordinary `read_artifact` tool;
5. a journal failure may leave an unreferenced object, which is handled by the adapter's grace-period collection.

Ownership of a terminal Run does not disappear implicitly when a Session handle is closed. Callers should explicitly use `await session.detach_artifacts(run_id)` to release the entire Run, or pass `roles=(ArtifactAttachmentRole.TOOL_RESULT,)` to release selected roles only. This API does not accept caller-constructed `ArtifactRef` values. It reads the full descriptor from the durable active index and atomically appends `DETACHED` facts through Session CAS; repeated or concurrent calls are idempotent. A publication ID may be retired by `ArtifactRetentionService.retire()` only after every owner has detached it.

Recovery treats only attachments that remain active in the journal as authoritative validation targets. Historical `RunState`, tool-result, final-output, or summary references that were explicitly detached are not misclassified as corrupt after their objects are collected. Retirement first obtains a `RETIRING` reservation atomically in the journal. That operation checks for the absence of active owners in the same transaction and blocks new attachments. A successful repository delete is then marked `RETIRED`; an explicit failure while the object still exists is marked `ABORTED`. A process may restart at any stage and converge by retrying retirement. Ordinary attachments cannot cross a `RETIRED` tombstone. Writing the same content again with `put()` produces a new publication ID, while `sha256` continues to represent the content digest. The same digest can therefore be reused without old tombstones or late deletes affecting the new object.

`artifact_id` is an opaque publication identity issued by the repository, not a content hash. `ArtifactRef` validates the ID and `sha256` formats separately, and the repository verifies the actual content hash and size before issuing the reference. This isolation gives up cross-publication content deduplication but prevents races between retirement, republication, and late deletion.

SQLite does not treat the derived active index or the retirement current row as a single source of truth. A reservation performs a bounded replay of append-only attachment facts for the `artifact_id` inside the same `BEGIN IMMEDIATE` transaction and cross-checks both directions against the index. Retirement transitions additionally use an append-only hash chain, a fixed-depth Merkle projection, and an authenticated database root. Ordinary SQL rewrites of current/fact/leaf/node/root/count data are rejected deterministically by the next lifecycle operation.

`FilesystemArtifactRepository` uses publication-addressed paths, staged writes, atomic finalization, hash and size validation, and path protections. It is a single-machine reference adapter and does not claim to support multi-node shared storage.

`agent.artifacts.collect_orphans(cutoff)` collects only objects older than the UTC cutoff that have never appeared in append-only Artifact facts; detached objects are not treated as orphans. Reading the reference inventory is bounded by `max_orphan_reference_ids`. Exceeding that limit fails safely, and the caller must either raise the explicit limit or use an external garbage collector.

## Adapter Boundaries

| Capability | Core port | Reference adapter |
|---|---|---|
| Model | `ModelGateway` | `LiteLLMModelGateway` |
| Tools | `ToolGateway` | local / Docker / MCP |
| Context | `ContextProvider` | bounded + tokenizer/compactor |
| Skills | `SkillProvider` | bounded filesystem loader or injected provider |
| Artifact | `ArtifactRepository`, `ArtifactExternalizer` | memory / filesystem |
| Persistence | `RunJournal` | memory / SQLite |
| Observability | post-commit observer | event / metrics sinks |

The exactly-once boundary for tool side effects is enforced by the gateway's stable `idempotency_key` and execution store. Core does not fabricate a paired “started + completed” commit in a single step. Observations are emitted only after the journal commit; sink timeouts or failures do not roll back domain facts.

## Verification

```powershell
uv run --locked python -m pytest -q -p no:cacheprovider
uv run --locked python -m mypy src
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
```

Architecture tests ensure that `src/core` does not import application code, adapters, SQLite, LiteLLM, Docker, MCP, or observability implementations, and that legacy schemas, the legacy Harness, and compatibility paths do not reappear.

## License

This project is licensed under the [MIT License](LICENSE).
