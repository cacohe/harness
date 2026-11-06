from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class HarnessHook:
    """Optional callbacks. Each one is called with keyword arguments and may accept any subset.

    before_think may receive task and context.
    after_think may receive task and output.
    before_execute_tool_calls may receive task and output.
    after_execute_tool_calls may receive task, result, and messages.
    """

    before_think: Optional[Callable[..., Any]] = None
    after_think: Optional[Callable[..., Any]] = None
    before_execute_tool_calls: Optional[Callable[..., Any]] = None
    after_execute_tool_calls: Optional[Callable[..., Any]] = None
