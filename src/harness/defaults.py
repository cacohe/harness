from __future__ import annotations

from harness.types import (
    FailDecision,
    FinishDecision,
    GoalUpdate,
    LlmOutput,
    RunSnapshot,
    ToolsDecision,
)


class DefaultGoalAdjuster:
    def adjust(self, state: RunSnapshot, incoming) -> GoalUpdate:
        return GoalUpdate(goal=state.goal, strategy=state.strategy)


class StructuralDiscriminator:
    def decide(self, output: LlmOutput, state: RunSnapshot | None):
        if output.tool_calls:
            return ToolsDecision(calls=tuple(output.tool_calls))
        if output.text:
            return FinishDecision(text=output.text)
        return FailDecision(reason="empty model output")


class EmptyMemoryStore:
    def save(self, record) -> None:
        return None

    def search(self, query):
        return ()


class ListObserver:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)
