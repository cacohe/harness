# Reliable runtime operations

## Single-track runtime

The repository exposes one runtime architecture. There is no legacy Harness facade, snapshot
repository adapter, schema migration, or compatibility import path. Databases are created and opened
only with the current SQLite schema; an incompatible or drifted schema is rejected instead of being
migrated.

Composition is explicit and async:

```python
agent = await create_production_agent(
    config,
    model=model_gateway,
    tools=tool_gateway,
    context=context_provider,
    context_counter=token_counter,
    context_request_factory=request_factory,
    ids=id_generator,
    clock=clock,
    profile_id=profile_id,
)

async with agent:
    session = await agent.open_session(session_id)
    state = await session.submit("run this", submission_id)
```

`create_development_agent()` has the same required ports and supplies memory-backed journal and
Artifact implementations when they are omitted. `create_production_agent()` requires an explicit
absolute SQLite path. It requires an absolute Artifact root only when it creates the filesystem
repository; an injected remote repository instead requires an explicit opaque
`artifact_workspace_id`. Artifact ownership, retirement, and reference-index capabilities are
separate ports, and production binds them to the same adapter object as the journal so attachment
facts and Run state retain one atomic authority.

## Atomic facts and authenticated recovery

`RunJournal.commit(batch, expected_version)` is the sole durable mutation boundary. The `batch`
argument is a `JournalCommit`; one CAS
atomically records the bounded `RunState`, domain events, transcript append, submission receipt and
optional summary checkpoint for a Session. A conflict never publishes a partial fact set.

SQLite stores a canonical record-hash chain and authenticated transcript coverage. Indexed page
reads and Merkle range proofs keep context recovery independent of total transcript length, while
schema signatures, sticky integrity violations and authenticated roots reject ordinary storage
rewrites. Recovery validates Session/version/order, state/event identity, Artifact descriptors,
submission bindings, checkpoints and coverage before returning facts.

There is intentionally no database migration path. Back up or discard databases before changing
schema versions; this `0.x` library rejects any non-current schema.

## Submission and Session ownership

Use a stable typed `SubmissionId` when a caller may retry. Repeating the same Session,
`SubmissionId` and exact message returns the committed Run without another model call. Reusing the
ID with different content is rejected.

A Session serializes operations; different Sessions can run concurrently. The application API is
async-only:

- `await session.submit(message, submission_id=None)`
- `await session.decide_tool_call(run_id, tool_call_id, approved=True)` to approve
- `await session.decide_tool_call(run_id, tool_call_id, approved=False)` to reject
- `await session.cancel(run_id)`
- `await session.detach_artifacts(run_id, roles=None)` for a terminal Run
- `await session.close()`
- `await agent.close()`

Runtime ownership checks prevent a Run from being resumed, decided or cancelled through a different
Session. Terminal facts remain durable across restart.

Artifact detach never accepts caller-supplied descriptors. It reads the Run's authoritative active
attachments, appends exact `DETACHED` lifecycle facts with the current Session CAS, and retries a
stale CAS from durable state. `roles` can select tool results, final outputs, or summaries; omitting
it releases the whole Run. Repeated and concurrent calls are idempotent, and detaching one owner
does not release another owner of the same publication. Lifecycle facts are available
through the journal's bounded `load_artifact_facts(..., after_index, limit)` index.

## Tool side effects

Core performs one model call or one durable tool batch per advance. The Tool gateway owns
exactly-once side effects using the stable request `idempotency_key`. Production execution stores
persist the binding of Session, Run and tool-call identity, return the original result on retry, and
reject binding changes. Approval may only become stricter after persistence.

Model, builder and tool deadlines are adapter boundaries. Cancellation propagates while required
container/client cleanup is shielded and bounded. Adapter exceptions are mapped to stable Core error
codes before they reach durable state.

The last permitted model step may complete with text, but it may not schedule tools. If its response
contains any tool call, Core records `limit_exceeded` before persisting or executing that batch.
Ordinary tools, approval requests, Artifact-producing tools, and `load_skill` all obey this rule.

## Bounded context and state

Every model request is built through `ContextProvider.build(request, budget)`. The provider must
return an estimate no larger than:

```text
context_window_tokens - reserved_output_tokens - safety_margin_tokens
```

Conversation history is read in bounded pages. Durable summary checkpoints record source range,
policy version, profile and Artifact references. Authenticated coverage witnesses allow older
history to be omitted without scanning or silently forgetting it. `BoundedRunJournal` rejects a
canonical serialized RunState larger than `ApplicationConfig.max_serialized_run_state_bytes`.
Before each model advance, application composition reloads matching durable checkpoints from the
authenticated journal snapshot; a checkpoint's summary and Artifact projection enter the trusted
SYSTEM prefix as one indivisible group, including after process restart.

A budget or state-envelope violation is a stable failure; the runtime does not send an oversized
request or truncate JSON into an invalid value.

## Progressive Skill loading

An optional `SkillProvider` exposes a bounded catalog before each model decision. The canonical
catalog message contains only `name`, `description`, and `version`; full instructions are absent
until the model invokes the reserved `load_skill` control tool with one logical name. Core handles
that tool directly and never routes it to the general Tool gateway.

Activation durably appends a `SkillBinding(name, version, content_sha256)` to `RunState`. Every later
model decision resolves each binding in activation order and verifies that the returned canonical
SYSTEM messages still hash to the persisted digest. Resolution happens again after restart. Missing
or changed content is an invariant failure before the next model request, not a silent version
switch. Catalog and resolution are model-path operations only; cancellation, approval decisions,
and resumption of already-persisted ordinary tools do not need the provider.

`FilesystemSkillProvider` performs pinned, validated reads of `<root>/<name>/SKILL.md` off the event
loop. Its trust boundary rejects non-absolute or replaced roots, traversal names, symlinks and
Windows reparse points, hard links and non-regular files, invalid or duplicate frontmatter, invalid
UTF-8, NUL, empty or oversized content, and replacement during a read. It is a single-machine
adapter, not a remote registry, recursive discovery service, automatic selector, or hot-reload
mechanism.

## Artifact durability

Large tool results and final answers are externalized before journal commit. Durable state, events
and context contain only bounded summaries and publication-scoped `ArtifactRef` values. A successful
repository `put()` is readable before its ref can be committed.

If journal commit fails after an Artifact write, the result is an unreferenced orphan, never a
durable dangling ref. The filesystem adapter can retire such objects after a grace period.
`ReadArtifactTool` provides authorized, bounded range reads; the model never receives repository
access directly.

Referenced objects are retired only after explicit detach. `ArtifactRetentionService` rejects a
delete while any durable owner remains; after the last owner detaches, retirement becomes reachable.
Recovery validates only the authoritative active attachment set for the target Run. Historical
terminal outcome, tool, final-answer, and summary refs that were explicitly detached may remain as
audit data without keeping the repository object alive.

Retirement is a journal-coordinated saga rather than a cross-store transaction. The journal
atomically reserves `RETIRING` only when no active owner exists, and attachment commits reject that
generation until delete completes. Successful delete records `RETIRED`; a confirmed failed delete
with the object still present records `ABORTED`. Repeating retirement after a crash resumes safely.
Every successful `put()` creates a new immutable publication ID while `sha256` remains the content
digest. A retired ID stays tombstoned permanently; putting the same bytes again creates a distinct
readable publication, so a late delete of the old ID cannot affect it. This intentionally forgoes
cross-publication content deduplication. SQLite persists retirement reservations; the memory journal
uses the same state machine under one lock.
Memory lifecycle paging keeps a per-Run ordered index and performs `bisect + limit` access, so its
lookup cost is `O(log F_run + limit)` and does not scan other Runs or copy full fact history.

The SQLite retirement gate does not trust the derived active-owner table. Inside the same
`BEGIN IMMEDIATE`, it replays at most 1,000,000 lifecycle facts selected by `artifact_id` and
requires exact bidirectional agreement with the index. Retirement transitions have their own
per-artifact append-only canonical hash chain. The current state is a leaf in a fixed-depth sparse
Merkle projection whose root and count are bound by a database retirement authenticated root.
Reserve, complete, abort, restart, and content-generation reopening verify the selected chain and
32-level proof before use. This detects partial raw-SQL rewrites of lifecycle facts, current state,
generation, leaf, internal node, root, or count. As with the Session journal authentication model,
an administrator that recomputes and rewrites every anchor is outside this corruption-detection
boundary.

## Observability and shutdown

Observers run only after a successful journal commit. Sink failure or timeout does not roll back
domain facts, and idempotent submission replay does not emit a second acceptance observation.

Factories close partially created resources in reverse order if composition fails. A successful
factory transfers only explicitly owned resources to `RuntimeCoordinator`; injected resources stay
caller-owned. `await agent.close()` rejects new work, waits for in-flight operations, and closes
owned resources exactly once on the same event loop. Repeated close calls are safe.
