from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Callable


class HookPoint(Enum):
    BEFORE_STEP = "before_step"
    AFTER_VALIDATE = "after_validate"
    BEFORE_INFER = "before_infer"
    AFTER_INFER = "after_infer"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_APPROVAL = "on_approval"
    ON_STATE_CHANGE = "on_state_change"
    BEFORE_OUTPUT = "before_output"


Hook = Callable[[dict], dict | None]


class HookBus:
    def __init__(self) -> None:
        self._hooks: dict[HookPoint, list[Hook]] = defaultdict(list)

    def register(self, point: HookPoint, fn: Hook) -> None:
        self._hooks[point].append(fn)

    def emit(self, point: HookPoint, ctx: dict) -> dict:
        current = dict(ctx)
        for fn in self._hooks[point]:
            result = fn(current)
            if result is not None:
                current = result
        return current
