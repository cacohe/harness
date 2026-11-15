# Observability and Load Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and superpowers:test-driven-development task by task.

**Goal:** Add a production-usable in-process metrics sink and prove shared-Agent concurrency under
bounded deterministic load.

**Spec:** `docs/superpowers/specs/2026-11-15-observability-load-design.md`

### Task 1: Metrics and composite event sinks

- [ ] Add failing tests for event/outcome/token counters, monotonic duration pairing, concurrent
      emission, detached snapshots, reset, orphan events, bounded in-flight eviction, and composite
      failure policy.
- [ ] Implement `MetricsEventSink` and `CompositeEventSink` without external dependencies.
- [ ] Run focused event/metrics tests and commit.

### Task 2: Agent injection and concurrency load gate

- [ ] Add failing tests proving Agent preserves the injected sink and default behavior.
- [ ] Wire the exact sink into Harness.
- [ ] Add a deterministic multi-Session load test covering shared repository, memory, metrics, and
      transcript isolation.
- [ ] Run focused integration/load tests and commit.

### Task 3: Documentation, review, and integration

- [ ] Document composition, metric schema, failure policy, cardinality bound, and load-test scope.
- [ ] Run full Ruff, Mypy, pytest, diff, and lock gates.
- [ ] Dispatch independent whole-branch review and fix Critical/Important findings with tests.
- [ ] Push, verify CI, merge, and compare the merged tree.
