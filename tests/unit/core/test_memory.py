import math
from concurrent.futures import ThreadPoolExecutor

import pytest
from src.core.memory import MemoryManager
from src.memory.retriever import HybridRetriever
from src.memory.store import InMemoryStore


def test_memory_manager_requires_store_and_retriever():
    with pytest.raises(TypeError):
        MemoryManager()


def test_memory_manager_uses_injected_store_and_retriever():
    manager = MemoryManager(InMemoryStore(), HybridRetriever())

    remembered = manager.remember("remember this")

    assert [item.id for item in manager.recall("remember", top_k=1)] == [remembered.id]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_memory_manager_rejects_invalid_default_token_budgets(budget):
    with pytest.raises(ValueError, match="token budget"):
        MemoryManager(InMemoryStore(), HybridRetriever(), default_token_budget=budget)


@pytest.mark.parametrize(
    ("content", "importance"),
    [("", 0.5), ("   ", 0.5), (None, 0.5), ("valid", math.nan), ("valid", True)],
)
def test_remember_rejects_invalid_content_or_importance(content, importance):
    manager = MemoryManager(InMemoryStore(), HybridRetriever())

    with pytest.raises(ValueError):
        manager.remember(content, importance=importance)


@pytest.mark.parametrize(
    "kwargs",
    [{"top_k": 0}, {"top_k": True}, {"token_budget": 0}, {"token_budget": 1.5}],
)
def test_recall_rejects_invalid_limits(kwargs):
    manager = MemoryManager(InMemoryStore(), HybridRetriever())

    with pytest.raises(ValueError):
        manager.recall("query", **kwargs)


def test_recall_filters_metadata_before_top_k_and_persists_access_count():
    store = InMemoryStore()
    manager = MemoryManager(store, HybridRetriever())
    own = manager.remember("project alpha", metadata={"session_id": "current"})
    shared = manager.remember("project alpha", metadata={"session_id": "other"})

    recalled = manager.recall(
        "project alpha",
        top_k=1,
        exclude_metadata={"session_id": "current"},
    )

    assert [item.id for item in recalled] == [shared.id]
    assert store.get(shared.id).access_count == 1
    assert store.get(own.id).access_count == 0


def test_recall_selects_complete_items_within_token_budget():
    class InOrderRetriever:
        def retrieve(self, query, items, top_k):
            return list(items)[:top_k]

    manager = MemoryManager(
        InMemoryStore(),
        InOrderRetriever(),
        token_counter=len,
        default_token_budget=5,
    )
    too_large = manager.remember("123456")
    fits = manager.remember("123")

    recalled = manager.recall("query", top_k=2)

    assert [item.id for item in recalled] == [fits.id]
    assert too_large.access_count == 0


def test_memory_manager_and_store_support_concurrent_remember_and_recall():
    store = InMemoryStore()
    manager = MemoryManager(store, HybridRetriever())

    def remember(index):
        return manager.remember(f"shared topic {index}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        remembered = list(pool.map(remember, range(80)))
        recalled = list(pool.map(lambda _index: manager.recall("shared", top_k=5), range(20)))

    assert len(store.all()) == 80
    assert len({item.id for item in remembered}) == 80
    assert all(len(items) == 5 for items in recalled)
