from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from src.application.observability import Observation, PostCommitObserver

EventSink = Callable[[dict[str, str | int]], None | Awaitable[None]]


class CompositeObserver:
    """Ordered best-effort fan-out; one observer can never affect another."""

    def __init__(self, observers: tuple[PostCommitObserver, ...]) -> None:
        if type(observers) is not tuple or len(observers) > 64:
            raise ValueError("Observer collection is invalid")
        self._observers = observers

    async def observe(self, observation: Observation) -> None:
        if type(observation) is not Observation:
            raise ValueError("Observation is invalid")
        for observer in self._observers:
            try:
                await observer.observe(observation)
            except asyncio.CancelledError:
                raise
            except Exception:
                continue


class EventObserver:
    """Emit metadata-only events to an explicitly injected sink."""

    def __init__(self, sink: EventSink) -> None:
        if not callable(sink):
            raise ValueError("Event sink is invalid")
        self._sink = sink

    async def observe(self, observation: Observation) -> None:
        event: dict[str, str | int] = {
            "kind": observation.kind,
            "session_id": observation.session_id,
            "run_id": observation.run_id,
            "status": observation.status,
            "journal_version": observation.journal_version,
            "journal_sequence": observation.journal_sequence,
        }
        result = self._sink(event)
        if inspect.isawaitable(result):
            await result
