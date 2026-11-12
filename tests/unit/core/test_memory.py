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

    assert manager.recall("remember", top_k=1) == [remembered]
