from __future__ import annotations

from dataclasses import dataclass

from src.adapters.context.summarization import SummaryCheckpointService
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
from src.core.journal import ConversationSummaryCheckpoint
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
        self,
        reader: ConversationReader,
        counter: TokenCounter,
        *,
        page_size: int = 128,
        summary_service: SummaryCheckpointService | None = None,
        max_compaction_interactions: int = 128,
    ) -> None:
        if (
            type(page_size) is not int
            or not 1 <= page_size <= 1_024
            or type(max_compaction_interactions) is not int
            or not 1 <= max_compaction_interactions <= 1_024
        ):
            raise ValueError("Context page size is outside its supported range")
        self._reader = reader
        self._counter = counter
        self._page_size = page_size
        self._summary_service = summary_service or SummaryCheckpointService()
        self._max_compaction_interactions = max_compaction_interactions

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
        compaction: list[ConversationInteraction] = []
        omission: _Omission | None = None
        durable = self._summary_service.find_latest_reusable(request.summary_checkpoints, request)
        covered_through = durable.covered_through_sequence if durable is not None else 0
        suffix_open = True
        cursor: int | None = None
        seen_cursors: set[int | None] = set()
        reached_covered_history = False
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
                if item.sequence_through <= covered_through:
                    reached_covered_history = True
                    break
                if item.sequence_from <= covered_through:
                    raise ValueError("Summary checkpoint splits a transcript interaction")
                if suffix_open:
                    candidate = [*selected, item]
                    messages = self._assemble(request, candidate, (), (), ())
                    candidate_estimate = self._counter.count(messages, request.tools)
                    if candidate_estimate.tokens <= budget.input_allowance_tokens:
                        selected.append(item)
                        estimate = candidate_estimate
                        continue
                    suffix_open = False
                if omission is None:
                    omission = _Omission(item.sequence_from, item.sequence_through, 1)
                else:
                    omission.include(item)
                if len(compaction) < self._max_compaction_interactions:
                    compaction.append(item)
            if reached_covered_history:
                break
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

        candidate_checkpoint: ConversationSummaryCheckpoint | None = None
        if compaction:
            outcome = await self._summary_service.compact_or_retain(request, tuple(compaction))
            candidate_checkpoint = outcome.checkpoint

        checkpoints = tuple(
            checkpoint for checkpoint in (candidate_checkpoint, durable) if checkpoint is not None
        )
        included_checkpoints: list[ConversationSummaryCheckpoint] = []
        for checkpoint in sorted(
            checkpoints, key=lambda item: item.covered_through_sequence, reverse=True
        ):
            proposed = [*included_checkpoints, checkpoint]
            proposed_messages = self._summary_messages(proposed)
            messages = self._assemble(request, selected, proposed_messages, memory, skills)
            candidate_estimate = self._counter.count(messages, request.tools)
            if candidate_estimate.tokens <= budget.input_allowance_tokens:
                included_checkpoints.append(checkpoint)
                summary = proposed_messages
                estimate = candidate_estimate
            else:
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
            candidate_checkpoint,
        )

    @staticmethod
    def _summary_messages(
        checkpoints: list[ConversationSummaryCheckpoint],
    ) -> tuple[ModelMessage, ...]:
        messages: list[ModelMessage] = []
        for checkpoint in sorted(checkpoints, key=lambda item: item.covered_through_sequence):
            messages.append(ModelMessage(ModelRole.SYSTEM, checkpoint.summary))
            if checkpoint.artifacts:
                messages.append(
                    ModelMessage(
                        ModelRole.SYSTEM,
                        "Artifacts: " + ", ".join(ref.artifact_id for ref in checkpoint.artifacts),
                    )
                )
        return tuple(messages)

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
