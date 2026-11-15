# Observability and Load Validation Design

## Goal

Expose Harness events through Agent composition, provide dependency-free production metrics, and
exercise the complete shared-Agent concurrency path under deterministic bounded load.

## Composition

`Agent` accepts an optional `EventSink` and passes the exact object to `Harness`. The default remains
the current `EventHandler`. Operators that need client streaming plus metrics use a
`CompositeEventSink(EventHandler(), metrics)`.

The composite emits in registration order. By default one failed observer is logged and isolated so
telemetry outages do not fail user Tasks; strict deployments may opt into fail-fast behavior.

## Metrics sink

`MetricsEventSink` is thread-safe and uses an injected monotonic clock. It maintains:

- counters by event type and terminal task outcome;
- accumulated LLM prompt/completion/total tokens;
- active task, LLM, and tool operation counts;
- count, total seconds, and maximum seconds for task runs, LLM phases, and tool calls.

Task IDs and call IDs exist only in bounded in-flight maps and are removed on terminal events.
Duplicate starts replace the previous timestamp; orphan terminal events increment counters but do not
invent latency. A configurable maximum evicts the oldest in-flight key and increments an eviction
counter, preventing missing terminal events from causing unbounded cardinality.

Snapshots are detached JSON-safe dictionaries. Reading or resetting metrics does not expose mutable
internal state.

## Load validation

A deterministic test runs many Sessions concurrently on one Agent with an async gateway, shared
repository, MemoryManager, and metrics sink. It verifies completion, transcript isolation, persisted
Task identity, long-term memory count, event counts, and zero remaining in-flight metrics. Separate
threaded tests stress concurrent `emit()` and bounded orphan handling.

The test is a repeatable concurrency regression gate, not a hardware benchmark or throughput SLA.
No external service, packaging, or dependency is added.

