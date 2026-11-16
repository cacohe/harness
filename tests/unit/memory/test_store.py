from src.memory.store import InMemoryStore
from src.schema.memory import MemoryItem


def test_in_memory_store_isolates_added_returned_and_nested_values():
    store = InMemoryStore()
    original = MemoryItem(
        content="original",
        metadata={"nested": ["value"]},
        embedding=[1.0],
    )
    store.add(original)
    original.content = "caller mutation"
    original.metadata["nested"].append("changed")
    original.embedding.append(2.0)

    first = store.get(original.id)
    assert first is not None
    assert first.content == "original"
    assert first.metadata == {"nested": ["value"]}
    assert first.embedding == [1.0]
    first.content = "loaded mutation"

    second = store.get(original.id)
    assert second is not None
    assert second.content == "original"
