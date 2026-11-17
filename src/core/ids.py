from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from unicodedata import category

_MAX_ID_BYTES = 256


@dataclass(frozen=True)
class _Id:
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not self.value:
            raise ValueError("ID must be a non-empty string")
        if any(category(character) == "Cc" for character in self.value):
            raise ValueError("ID must not contain control characters")
        if len(self.value.encode("utf-8")) > _MAX_ID_BYTES:
            raise ValueError(f"ID must not exceed {_MAX_ID_BYTES} UTF-8 bytes")


@dataclass(frozen=True)
class RunId(_Id):
    pass


@dataclass(frozen=True)
class SessionId(_Id):
    pass


@dataclass(frozen=True)
class SubmissionId(_Id):
    pass


@dataclass(frozen=True)
class ProfileId(_Id):
    pass


@dataclass(frozen=True)
class EventId(_Id):
    pass


class IdGenerator(Protocol):
    def run(self) -> RunId: ...

    def session(self) -> SessionId: ...

    def submission(self) -> SubmissionId: ...

    def profile(self) -> ProfileId: ...

    def event(self) -> EventId: ...


class Clock(Protocol):
    def now(self) -> datetime:
        """Return the wall time used to timestamp domain facts."""
        ...

    def monotonic(self) -> float:
        """Return a reading in seconds for measuring elapsed time."""
        ...
