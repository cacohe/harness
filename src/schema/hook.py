from dataclasses import dataclass
from typing import Any, Callable


def _noop(**_kwargs: Any) -> None:
    """Do nothing. The default so every hook can be called."""


@dataclass
class HarnessHook:
    """Harness hooks. Each one is called with keyword arguments and may accept any subset.

    before_think may receive task and context.
    after_think may receive task and output.
    before_execute_tool_calls may receive task and output.
    after_execute_tool_calls may receive task, result.
    """

    before_think: Callable[..., Any] = _noop
    after_think: Callable[..., Any] = _noop
    before_execute_tool_calls: Callable[..., Any] = _noop
    after_execute_tool_calls: Callable[..., Any] = _noop
