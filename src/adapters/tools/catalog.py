from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from src.core.tools import ToolSpec

ToolHandler = Callable[[dict[str, object]], object | Awaitable[object]]


@dataclass(frozen=True, slots=True)
class ToolCatalogEntry:
    spec: ToolSpec
    handler: ToolHandler
    mutating: bool
    generation: str = "static"

    def __post_init__(self) -> None:
        if type(self.spec) is not ToolSpec or not callable(self.handler):
            raise ValueError("Tool catalog entry is invalid")
        if type(self.mutating) is not bool or type(self.generation) is not str:
            raise ValueError("Tool catalog entry is invalid")
        if not self.generation or len(self.generation.encode("utf-8")) > 256:
            raise ValueError("Tool catalog generation is invalid")


class ToolCatalog:
    def __init__(self, entries: tuple[ToolCatalogEntry, ...]) -> None:
        if type(entries) is not tuple or any(type(item) is not ToolCatalogEntry for item in entries):
            raise ValueError("Tool catalog requires a tuple of entries")
        if len({item.spec.name for item in entries}) != len(entries):
            raise ValueError("Tool catalog names must be unique")
        self._entries = {item.spec.name: item for item in entries}

    def get(self, name: str) -> ToolCatalogEntry | None:
        return self._entries.get(name)

    def replace(self, entries: tuple[ToolCatalogEntry, ...]) -> None:
        self._entries = ToolCatalog(entries)._entries

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(entry.spec for entry in self._entries.values())
