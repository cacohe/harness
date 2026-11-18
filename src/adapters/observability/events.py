from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable

from src.application.observability import Observation, PostCommitObserver

EventSink = Callable[[dict[str, str | int]], None | Awaitable[None]]


class CompositeObserver:
    """Ordered best-effort fan-out; one observer can never affect another."""

    def __init__(
        self,
        observers: tuple[PostCommitObserver, ...],
        *,
        observer_timeout_seconds: float = 0.05,
    ) -> None:
        if type(observers) is not tuple or len(observers) > 64:
            raise ValueError("Observer collection is invalid")
        self._observers = observers
        if (
            isinstance(observer_timeout_seconds, bool)
            or type(observer_timeout_seconds) not in (int, float)
            or not 0.001 <= observer_timeout_seconds <= 30
        ):
            raise ValueError("Observer timeout is invalid")
        self._timeout = float(observer_timeout_seconds)

    async def observe(self, observation: Observation) -> None:
        if type(observation) is not Observation:
            raise ValueError("Observation is invalid")
        for observer in self._observers:
            try:
                await asyncio.wait_for(
                    observer.observe(observation), timeout=self._timeout
                )
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
        if inspect.iscoroutinefunction(self._sink):
            await self._sink(event)
            return
        result = await asyncio.to_thread(self._sink, event)
        if inspect.isawaitable(result):
            await result
