from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Observation:
    kind: str
    session_id: str
    run_id: str
    status: str
    journal_version: int
    journal_sequence: int

    def __post_init__(self) -> None:
        texts = (self.kind, self.session_id, self.run_id, self.status)
        if any(
            type(item) is not str or not item or len(item.encode("utf-8")) > 256 for item in texts
        ):
            raise ValueError("Observation identity is invalid")
        if any(
            type(item) is not int or not 0 <= item <= (1 << 63) - 1
            for item in (self.journal_version, self.journal_sequence)
        ):
            raise ValueError("Observation position is invalid")


class PostCommitObserver(Protocol):
    def observe(self, observation: Observation) -> Awaitable[None]: ...
