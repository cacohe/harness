from dataclasses import dataclass
from typing import Any, Callable


def _noop(**_kwargs: Any) -> None:
    """Do nothing. The default so every hook can be called."""


@dataclass
class HarnessHook:
    """Callbacks around model thinking and tool execution."""

    before_think: Callable[..., Any] = _noop
    after_think: Callable[..., Any] = _noop
    before_execute_tool_calls: Callable[..., Any] = _noop
    after_execute_tool_calls: Callable[..., Any] = _noop
