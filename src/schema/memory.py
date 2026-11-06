from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EPISODIC = "episodic"  # conversation and events
    SEMANTIC = "semantic"  # facts and knowledge
    PROCEDURAL = "procedural"  # experiences and rules


@dataclass
class MemoryItem:
    content: str
    type: MemoryType = MemoryType.EPISODIC
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5  # 0~1, manually or model scored
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    access_count: int = 0

    def touch(self) -> None:
        self.access_count += 1
