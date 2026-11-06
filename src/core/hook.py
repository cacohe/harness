from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class HarnessHook:
    """Optional callbacks around one model call and its tool calls.

    before_think(task, context)
    after_think(task, output)
    before_execute_tool_calls(task, output)
    after_execute_tool_calls(task, tool_messages)
    """

    before_think: Optional[Callable] = None
    after_think: Optional[Callable] = None
    before_execute_tool_calls: Optional[Callable] = None
    after_execute_tool_calls: Optional[Callable] = None
