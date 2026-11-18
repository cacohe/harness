from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.application.observability import Observation


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    counts: tuple[tuple[str, str, int], ...]
    overflow: int


class MetricsObserver:
    """Async-safe counters with a hard cardinality ceiling."""

    def __init__(self, *, max_series: int = 256) -> None:
        if type(max_series) is not int or not 1 <= max_series <= 65_536:
            raise ValueError("Metric cardinality limit is invalid")
        self._max_series = max_series
        self._counts: dict[tuple[str, str], int] = {}
        self._overflow = 0
        self._lock = asyncio.Lock()

    async def observe(self, observation: Observation) -> None:
        if type(observation) is not Observation:
            raise ValueError("Observation is invalid")
        key = (observation.kind, observation.status)
        async with self._lock:
            if key not in self._counts and len(self._counts) >= self._max_series:
                self._overflow += 1
            else:
                self._counts[key] = self._counts.get(key, 0) + 1

    async def snapshot(self) -> MetricsSnapshot:
        async with self._lock:
            return MetricsSnapshot(
                tuple(
                    (kind, status, count) for (kind, status), count in sorted(self._counts.items())
                ),
                self._overflow,
            )
