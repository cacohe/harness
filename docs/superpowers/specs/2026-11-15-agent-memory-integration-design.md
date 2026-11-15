# Agent Memory Integration Design

## Goal

Connect the existing `MemoryManager` to Agent composition and Session turns without changing the
default transcript-only behavior when no manager is injected.

## Lifecycle

An injected manager is shared by all Sessions created from one Agent. Before a new turn, the Session
recalls records relevant to the new user message. Records written by the same Session are excluded,
because its ordered transcript already carries them. Completed turns are remembered only after the
Harness reaches a terminal completed Task; paused, failed, and cancelled turns are never committed.
Resume commits the original user message exactly once.

## Prompt boundary

Recalled content is serialized as JSON data in a clearly delimited memory block. The system prompt
states that long-term memory is untrusted historical data, not instructions. Only memory content and
type are exposed to the model; backend metadata is not.

## Budget and retrieval

`MemoryManager.recall()` gains an optional token budget and metadata exclusion. Filtering occurs
before retrieval so same-Session records cannot consume `top_k`. Retrieved records are selected as
whole items within the configured budget, then touched and updated in the store. Invalid budgets and
limits fail early.

## Async and failure semantics

The existing memory ports remain synchronous. Session calls them through worker threads so a slow
store, retriever, or embedder does not block the event loop. Ordinary recall/remember failures are
fail-open and recorded as immutable `MemoryFailure` values on that Session. They do not turn a
completed Task into an exception or corrupt its transcript. Cancellation before Harness execution
still cancels the turn; once a Task is terminal, post-completion memory commit is shielded so terminal
Task semantics win.

`MemoryManager` serializes its compound store/retriever operations with an `RLock`. The built-in
store also protects direct access and returns deep copies, preventing callers from mutating stored
memory outside the manager's update path.

## Compatibility

- `memory_manager=None` remains the default and produces byte-for-byte existing prompts.
- Existing `AgentSession(harness, repository, ensure_open)` construction remains valid.
- Transcript properties and pause/resume ownership are unchanged.
- No packaging or external dependency is added.

