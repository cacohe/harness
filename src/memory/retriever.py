import math
from datetime import datetime, timezone
from typing import Callable, Sequence

from src.core.memory import MemoryRetriever
from src.schema.memory import MemoryItem


class HybridRetriever(MemoryRetriever):
    """Relevance + recency + importance weighted scoring."""

    def __init__(
        self,
        embedder: Callable[[str], list[float]] | None = None,
        now: Callable[[], datetime] | None = None,
        w_relevance: float = 0.6,
        w_recency: float = 0.2,
        w_importance: float = 0.2,
        half_life_hours: float = 24.0,
    ) -> None:
        self.embedder = embedder
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.w_relevance = w_relevance
        self.w_recency = w_recency
        self.w_importance = w_importance
        self.half_life_hours = half_life_hours

    def retrieve(self, query: str, items: Sequence[MemoryItem], top_k: int) -> list[MemoryItem]:
        if not items or top_k <= 0:
            return []
        now = self.now()
        q_emb = self.embedder(query) if self.embedder else None

        scored: list[tuple[float, MemoryItem]] = []
        for it in items:
            rel = self._relevance(query, q_emb, it)
            rec = self._recency(now, it.created_at)
            score = (
                self.w_relevance * rel + self.w_recency * rec + self.w_importance * it.importance
            )
            scored.append((score, it))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in scored[:top_k]]

    def _relevance(self, query: str, q_emb: list[float] | None, item: MemoryItem) -> float:
        if q_emb is not None and item.embedding is not None:
            return _cosine(q_emb, item.embedding)
        # 无向量时的兜底：查询词命中率
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return 0.0
        i_tokens = set(_tokenize(item.content))
        return len(q_tokens & i_tokens) / len(q_tokens)

    def _recency(self, now: datetime, created_at: datetime) -> float:
        age_h = max((now - created_at).total_seconds() / 3600.0, 0.0)
        return math.exp(-math.log(2) * age_h / self.half_life_hours)


def _tokenize(text: str) -> list[str]:
    # 中文可按字/词，英文按空白；这里给最小实现
    return [t for t in text.lower().replace("\n", " ").split(" ") if t]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(0.0, min(1.0, dot / (na * nb)))
