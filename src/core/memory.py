from __future__ import annotations

import math
import threading
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from src.schema.memory import MemoryItem, MemoryType


@runtime_checkable
class MemoryStore(Protocol):
    def add(self, item: MemoryItem) -> None: ...
    def update(self, item: MemoryItem) -> None: ...
    def remove(self, item_id: str) -> bool: ...
    def get(self, item_id: str) -> MemoryItem | None: ...
    def all(self) -> list[MemoryItem]: ...


@runtime_checkable
class MemoryRetriever(Protocol):
    def retrieve(self, query: str, items: Sequence[MemoryItem], top_k: int) -> list[MemoryItem]: ...


class MemoryManager:
    """Memory manager for the agent."""

    def __init__(
        self,
        store: MemoryStore,
        retriever: MemoryRetriever,
        embedder: Callable[[str], list[float]] | None = None,
        token_counter: Callable[[str], int] | None = None,
        default_token_budget: int = 2000,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.token_counter = token_counter or (lambda s: max(1, len(s) // 4))
        self.default_token_budget = self._positive_integer(
            default_token_budget, "default token budget"
        )
        self._lock = threading.RLock()

    def remember(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("memory content must be a non-empty string")
        if (
            isinstance(importance, bool)
            or not isinstance(importance, (int, float))
            or not math.isfinite(importance)
        ):
            raise ValueError("memory importance must be a finite number")
        with self._lock:
            item = MemoryItem(
                content=content,
                type=type,
                importance=max(0.0, min(1.0, float(importance))),
                metadata=metadata or {},
                embedding=self.embedder(content) if self.embedder else None,
            )
            self.store.add(item)
            return item

    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        token_budget: int | None = None,
        exclude_metadata: dict[str, Any] | None = None,
    ) -> list[MemoryItem]:
        if not isinstance(query, str):
            raise ValueError("memory query must be a string")
        limit = self._positive_integer(top_k, "top_k")
        budget = self._positive_integer(
            self.default_token_budget if token_budget is None else token_budget,
            "token budget",
        )
        excluded = exclude_metadata or {}
        with self._lock:
            items = [
                item
                for item in self.store.all()
                if not excluded
                or not all(item.metadata.get(key) == value for key, value in excluded.items())
            ]
            ranked = self.retriever.retrieve(query, items, len(items))
            selected: list[MemoryItem] = []
            remaining = budget
            seen: set[str] = set()
            for item in ranked:
                if item.id in seen:
                    continue
                tokens = self.token_counter(item.content)
                if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
                    raise ValueError("token counter must return a positive integer")
                if tokens <= remaining:
                    item.touch()
                    self.store.update(item)
                    selected.append(item)
                    seen.add(item.id)
                    remaining -= tokens
                    if len(selected) == limit:
                        break
            return selected

    def recent(self, n: int = 10) -> list[MemoryItem]:
        limit = self._positive_integer(n, "recent count")
        with self._lock:
            items = sorted(self.store.all(), key=lambda x: x.created_at, reverse=True)
            return items[:limit]

    def forget(self, item_id: str) -> bool:
        with self._lock:
            return self.store.remove(item_id)

    @staticmethod
    def _positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value
