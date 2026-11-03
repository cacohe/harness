from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class InputQueue(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []
        self._lock = threading.Lock()

    def push(self, item: T) -> None:
        with self._lock:
            self._items.append(item)

    def drain(self) -> list[T]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
            return items
