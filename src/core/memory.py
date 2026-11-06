from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Protocol, Sequence


class MemoryType(str, Enum):
    EPISODIC = "episodic"  # conversation and events
    SEMANTIC = "semantic"  # facts and knowledge
    PROCEDURAL = "procedural"  # experiences and rules


@dataclass
class MemoryItem:
    content: str
    type: MemoryType = MemoryType.EPISODIC
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5  # 0~1, manually or model scored
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    access_count: int = 0

    def touch(self) -> None:
        self.access_count += 1


class MemoryStore(Protocol):
    def add(self, item: MemoryItem) -> None: ...
    def update(self, item: MemoryItem) -> None: ...
    def remove(self, item_id: str) -> bool: ...
    def get(self, item_id: str) -> MemoryItem | None: ...
    def all(self) -> list[MemoryItem]: ...


class MemoryRetriever(Protocol):
    def retrieve(self, query: str, items: Sequence[MemoryItem], top_k: int) -> list[MemoryItem]: ...


class MemoryManager:
    """Memory manager for the agent."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        retriever: MemoryRetriever | None = None,
        embedder: Callable[[str], list[float]] | None = None,
        token_counter: Callable[[str], int] | None = None,
        default_token_budget: int = 2000,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.retriever = retriever
        self.token_counter = token_counter or (lambda s: max(1, len(s) // 4))
        self.default_token_budget = default_token_budget

    def remember(
        self,
        content: str,
        *,
        type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            content=content,
            type=type,
            importance=max(0.0, min(1.0, importance)),
            metadata=metadata or {},
            embedding=self.embedder(content) if self.embedder else None,
        )
        self.store.add(item)
        return item

    def recall(self, query: str, *, top_k: int = 5) -> list[MemoryItem]:
        results = self.retriever.retrieve(query, self.store.all(), top_k)
        for it in results:
            it.touch()
        return results

    def recent(self, n: int = 10) -> list[MemoryItem]:
        items = sorted(self.store.all(), key=lambda x: x.created_at, reverse=True)
        return items[:n]

    def forget(self, item_id: str) -> bool:
        return self.store.remove(item_id)
