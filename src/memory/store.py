from src.core.memory import MemoryItem, MemoryStore


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}

    def add(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def update(self, item: MemoryItem) -> None:
        self._items[item.id] = item

    def remove(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    def get(self, item_id: str) -> MemoryItem | None:
        return self._items.get(item_id)

    def all(self) -> list[MemoryItem]:
        return list(self._items.values())
