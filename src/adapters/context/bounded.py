from __future__ import annotations

from dataclasses import dataclass

from src.core.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextRequest,
    ContextWindow,
    ConversationInteraction,
    ConversationReader,
    OmittedContextRef,
    TokenCounter,
)
from src.core.model import ModelMessage, ModelRole


@dataclass
class _Omission:
    first: int
    last: int
    count: int

    def include(self, item: ConversationInteraction) -> None:
        self.first = min(self.first, item.sequence_from)
        self.last = max(self.last, item.sequence_through)
        self.count += 1


class BoundedContextProvider:
    """Select a deterministic, newest-first transcript window without loading it whole."""

    def __init__(
        self, reader: ConversationReader, counter: TokenCounter, *, page_size: int = 128
    ) -> None:
        if type(page_size) is not int or not 1 <= page_size <= 1_024:
            raise ValueError("Context page size is outside its supported range")
        self._reader = reader
        self._counter = counter
        self._page_size = page_size

    async def build(self, request: ContextRequest, budget: ContextBudget) -> ContextWindow:
        if type(request) is not ContextRequest or type(budget) is not ContextBudget:
            raise ValueError("Context provider requires a request and budget")
        required = (
            request.system_message,
            *request.open_tool_messages,
            request.current_input,
        )
        estimate = self._counter.count(required, request.tools)
        if estimate.tokens > budget.input_allowance_tokens:
            raise ContextBudgetExceeded

        selected: list[ConversationInteraction] = []
        omission: _Omission | None = None
        cursor: int | None = None
        seen_cursors: set[int | None] = set()
        while True:
            if cursor in seen_cursors:
                raise ValueError("Conversation reader repeated a pagination cursor")
            seen_cursors.add(cursor)
            page = await self._reader.read_page(request.session_id, cursor, self._page_size)
            if cursor is not None and any(
                item.sequence_through >= cursor for item in page.interactions
            ):
                raise ValueError("Conversation reader returned an item outside its cursor")
            for item in page.interactions:
                candidate = [*selected, item]
                messages = self._assemble(request, candidate, (), (), ())
                candidate_estimate = self._counter.count(messages, request.tools)
                if candidate_estimate.tokens <= budget.input_allowance_tokens:
                    selected.append(item)
                    estimate = candidate_estimate
                elif omission is None:
                    omission = _Omission(item.sequence_from, item.sequence_through, 1)
                else:
                    omission.include(item)
            cursor = page.next_before_sequence
            if cursor is None:
                break

        summary: tuple[ModelMessage, ...] = ()
        memory: tuple[ModelMessage, ...] = ()
        skills: tuple[ModelMessage, ...] = ()
        omitted: list[OmittedContextRef] = []
        if omission is not None:
            omitted.append(
                OmittedContextRef("transcript", omission.first, omission.last, omission.count)
            )

        if request.summary_checkpoint is not None:
            message = ModelMessage(ModelRole.SYSTEM, request.summary_checkpoint.summary)
            summary, estimate = self._include_optional(
                request, budget, selected, summary, memory, skills, (message,), estimate
            )
            if not summary:
                checkpoint = request.summary_checkpoint
                omitted.append(
                    OmittedContextRef(
                        "summary",
                        checkpoint.covered_from_sequence,
                        checkpoint.covered_through_sequence,
                        1,
                    )
                )

        memory, memory_omitted, estimate = self._include_messages(
            request, budget, selected, summary, memory, skills, request.memory_messages, "memory"
        )
        omitted.extend(memory_omitted)
        skills, skill_omitted, estimate = self._include_messages(
            request, budget, selected, summary, memory, skills, request.skill_messages, "skills"
        )
        omitted.extend(skill_omitted)
        messages = self._assemble(request, selected, summary, memory, skills)
        estimate = self._counter.count(messages, request.tools)
        return ContextWindow(
            messages,
            request.tools,
            estimate,
            budget,
            tuple(omitted),
            request.policy_version,
            request.new_checkpoint,
        )

    def _include_optional(
        self,
        request: ContextRequest,
        budget: ContextBudget,
        selected: list[ConversationInteraction],
        summary: tuple[ModelMessage, ...],
        memory: tuple[ModelMessage, ...],
        skills: tuple[ModelMessage, ...],
        candidate: tuple[ModelMessage, ...],
        prior_estimate,
    ):
        messages = self._assemble(request, selected, candidate, memory, skills)
        estimate = self._counter.count(messages, request.tools)
        if estimate.tokens <= budget.input_allowance_tokens:
            return candidate, estimate
        return summary, prior_estimate

    def _include_messages(
        self,
        request: ContextRequest,
        budget: ContextBudget,
        selected: list[ConversationInteraction],
        summary: tuple[ModelMessage, ...],
        current: tuple[ModelMessage, ...],
        later: tuple[ModelMessage, ...],
        candidates: tuple[ModelMessage, ...],
        source: str,
    ):
        included = list(current)
        omitted_count = 0
        estimate = self._counter.count(
            self._assemble(request, selected, summary, current, later), request.tools
        )
        for message in candidates:
            proposed = (*included, message)
            if source == "memory":
                assembled = self._assemble(request, selected, summary, proposed, later)
            else:
                assembled = self._assemble(request, selected, summary, current, proposed)
            candidate_estimate = self._counter.count(assembled, request.tools)
            if candidate_estimate.tokens <= budget.input_allowance_tokens:
                included.append(message)
                estimate = candidate_estimate
            else:
                omitted_count += 1
        references = (OmittedContextRef(source, 0, 0, omitted_count),) if omitted_count else ()
        return tuple(included), references, estimate

    @staticmethod
    def _assemble(
        request: ContextRequest,
        selected: list[ConversationInteraction],
        summary: tuple[ModelMessage, ...],
        memory: tuple[ModelMessage, ...],
        skills: tuple[ModelMessage, ...],
    ) -> tuple[ModelMessage, ...]:
        history = tuple(
            message
            for interaction in sorted(selected, key=lambda item: item.sequence_from)
            for message in interaction.messages
        )
        return (
            request.system_message,
            *summary,
            *memory,
            *skills,
            *history,
            *request.open_tool_messages,
            request.current_input,
        )
