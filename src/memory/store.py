import copy
import threading

from src.schema.memory import MemoryItem


class InMemoryStore:
    def __init__(self) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._lock = threading.RLock()

    def add(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.id] = copy.deepcopy(item)

    def update(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.id] = copy.deepcopy(item)

    def remove(self, item_id: str) -> bool:
        with self._lock:
            return self._items.pop(item_id, None) is not None

    def get(self, item_id: str) -> MemoryItem | None:
        with self._lock:
            item = self._items.get(item_id)
            return copy.deepcopy(item) if item is not None else None

    def all(self) -> list[MemoryItem]:
        with self._lock:
            return copy.deepcopy(list(self._items.values()))
