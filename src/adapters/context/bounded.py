from __future__ import annotations

from dataclasses import dataclass

from src.adapters.context.summarization import SummaryCheckpointService
from src.core.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextRequest,
    ContextWindow,
    ConversationCoverage,
    ConversationInteraction,
    ConversationReader,
    OmittedContextRef,
    TokenCounter,
    summary_system_messages,
)
from src.core.journal import ConversationSummaryCheckpoint
from src.core.model import ModelMessage


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
            request.current_input,
            *request.open_tool_messages,
        )
        estimate = self._counter.count(required, request.tools)
        if estimate.tokens > budget.input_allowance_tokens:
            raise ContextBudgetExceeded

        durable = self._summary_service.find_latest_reusable(request.summary_checkpoints, request)
        selected: list[ConversationInteraction] = []
        included_checkpoints: list[ConversationSummaryCheckpoint] = []
        omitted_checkpoints: list[ConversationSummaryCheckpoint] = []
        omitted: list[OmittedContextRef] = []
        omission_segments: list[_Omission] = []
        current_omission: _Omission | None = None
        compaction: list[ConversationInteraction] = []
        compaction_open = True
        suffix_open = True
        memory: tuple[ModelMessage, ...] = ()
        skills: tuple[ModelMessage, ...] = ()
        memory, memory_omitted, estimate = self._include_messages(
            request,
            budget,
            selected,
            (),
            memory,
            skills,
            request.memory_messages,
            "memory",
        )
        omitted.extend(memory_omitted)
        skills, skill_omitted, estimate = self._include_messages(
            request,
            budget,
            selected,
            (),
            memory,
            skills,
            request.skill_messages,
            "skills",
        )
        omitted.extend(skill_omitted)
        if durable is not None:
            suffix_open, estimate = self._consider_checkpoint(
                request,
                budget,
                selected,
                included_checkpoints,
                omitted_checkpoints,
                durable,
                suffix_open,
                estimate,
                memory,
                skills,
            )
        cursor: int | None = None
        seen_cursors: set[int | None] = set()
        scanned = 0
        skipped_current = False
        skipped_open_tools = False

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
                scanned += 1
                if self._is_current_input(request, item, skipped_current):
                    skipped_current = True
                    continue
                if self._is_open_tool_batch(request, item, skipped_open_tools):
                    skipped_open_tools = True
                    continue
                relation = self._checkpoint_relation(item, durable)
                if relation == "split":
                    raise ValueError("Summary checkpoint splits a transcript interaction")
                if relation == "covered":
                    continue
                if suffix_open:
                    proposed = [*selected, item]
                    candidate_estimate = self._counter.count(
                        self._assemble(
                            request,
                            proposed,
                            tuple(included_checkpoints),
                            memory,
                            skills,
                        ),
                        request.tools,
                    )
                    if candidate_estimate.tokens <= budget.input_allowance_tokens:
                        selected.append(item)
                        estimate = candidate_estimate
                        continue
                    suffix_open = False
                if current_omission is None:
                    current_omission = _Omission(item.sequence_from, item.sequence_through, 1)
                    omission_segments.append(current_omission)
                else:
                    current_omission.include(item)
                if compaction_open and len(compaction) < self._max_compaction_interactions:
                    compaction.append(item)
            if (
                scanned >= self._max_compaction_interactions
                and page.next_before_sequence is not None
            ):
                aggregate = getattr(self._reader, "summarize_before", None)
                if not callable(aggregate):
                    raise ValueError("Bounded conversation reads require aggregate coverage")
                older = await aggregate(
                    request.session_id,
                    page.next_before_sequence,
                    durable.covered_from_sequence if durable is not None else None,
                    durable.covered_through_sequence if durable is not None else None,
                )
                self._append_coverage(
                    omission_segments,
                    current_omission,
                    older,
                    page.next_before_sequence,
                    durable,
                )
                cursor = None
                break
            cursor = page.next_before_sequence
            if cursor is None:
                break

        candidate_checkpoint: ConversationSummaryCheckpoint | None = None
        if compaction:
            prior = (
                durable
                if durable is not None
                and min(item.sequence_from for item in compaction)
                > durable.covered_through_sequence
                else None
            )
            outcome = await self._summary_service.compact_or_retain(
                request, tuple(compaction), prior_checkpoint=prior
            )
            candidate_checkpoint = outcome.checkpoint

        omitted[0:0] = [
            OmittedContextRef("transcript", item.first, item.last, item.count)
            for item in reversed(omission_segments)
        ]
        omitted.extend(
            OmittedContextRef(
                "summary",
                checkpoint.covered_from_sequence,
                checkpoint.covered_through_sequence,
                1,
            )
            for checkpoint in omitted_checkpoints
        )

        checkpoints = tuple(included_checkpoints)
        messages = self._assemble(request, selected, checkpoints, memory, skills)
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
    def _append_coverage(
        omissions: list[_Omission],
        current: _Omission | None,
        coverage: object,
        before_sequence: int,
        excluded: ConversationSummaryCheckpoint | None,
    ) -> None:
        excluded_from = excluded.covered_from_sequence if excluded is not None else None
        excluded_through = excluded.covered_through_sequence if excluded is not None else None
        if (
            type(coverage) is not ConversationCoverage
            or coverage.before_sequence != before_sequence
            or coverage.excluded_from_sequence != excluded_from
            or coverage.excluded_through_sequence != excluded_through
        ):
            raise ValueError("Conversation aggregate has the wrong coverage bound")
        expected: list[tuple[int, int]] = []
        target_last = before_sequence - 1
        if target_last > 0:
            if excluded_from is None or excluded_through is None or excluded_from > target_last:
                expected.append((1, target_last))
            else:
                if excluded_through < target_last:
                    expected.append((excluded_through + 1, target_last))
                if excluded_from > 1:
                    expected.append((1, excluded_from - 1))
        actual = [(segment.first_sequence, segment.last_sequence) for segment in coverage.segments]
        if actual != expected:
            raise ValueError("Conversation aggregate does not completely cover its target")
        for index, item in enumerate(coverage.segments):
            separated_by_checkpoint = (
                current is not None
                and excluded is not None
                and item.last_sequence < excluded.covered_from_sequence
                and current.first > excluded.covered_through_sequence
            )
            if index == 0 and current is not None and not separated_by_checkpoint:
                current.first = min(current.first, item.first_sequence)
                current.last = max(current.last, item.last_sequence)
                current.count += item.item_count
            elif item.item_count:
                omissions.append(
                    _Omission(item.first_sequence, item.last_sequence, item.item_count)
                )

    @staticmethod
    def _checkpoint_relation(
        item: ConversationInteraction,
        checkpoint: ConversationSummaryCheckpoint | None,
    ) -> str:
        if checkpoint is None:
            return "none"
        if item.sequence_from > checkpoint.covered_through_sequence:
            return "newer"
        if item.sequence_through < checkpoint.covered_from_sequence:
            return "older"
        if (
            item.sequence_from >= checkpoint.covered_from_sequence
            and item.sequence_through <= checkpoint.covered_through_sequence
        ):
            return "covered"
        return "split"

    def _consider_checkpoint(
        self,
        request: ContextRequest,
        budget: ContextBudget,
        selected: list[ConversationInteraction],
        included: list[ConversationSummaryCheckpoint],
        omitted: list[ConversationSummaryCheckpoint],
        checkpoint: ConversationSummaryCheckpoint | None,
        suffix_open: bool,
        prior_estimate,
        memory: tuple[ModelMessage, ...] = (),
        skills: tuple[ModelMessage, ...] = (),
    ):
        assert checkpoint is not None
        proposed = (*included, checkpoint)
        estimate = self._counter.count(
            self._assemble(request, selected, proposed, memory, skills), request.tools
        )
        if suffix_open and estimate.tokens <= budget.input_allowance_tokens:
            included.append(checkpoint)
            return True, estimate
        omitted.append(checkpoint)
        return suffix_open, prior_estimate

    @staticmethod
    def _is_current_input(
        request: ContextRequest, item: ConversationInteraction, already_skipped: bool
    ) -> bool:
        sequence = request.current_input_sequence
        if sequence is not None:
            if item.sequence_from <= sequence <= item.sequence_through:
                if item.sequence_from != sequence or item.sequence_through != sequence:
                    raise ValueError("Current input sequence splits a transcript interaction")
                return True
            return False
        return not already_skipped and item.messages == (request.current_input,)

    @staticmethod
    def _is_open_tool_batch(
        request: ContextRequest, item: ConversationInteraction, already_skipped: bool
    ) -> bool:
        sequence_range = request.open_tool_sequence_range
        if sequence_range is not None:
            if (
                item.sequence_from >= sequence_range[0]
                and item.sequence_through <= sequence_range[1]
            ):
                return True
            if not (
                item.sequence_through < sequence_range[0] or item.sequence_from > sequence_range[1]
            ):
                raise ValueError("Open tool sequence range splits a transcript interaction")
            return False
        return (
            bool(request.open_tool_messages)
            and not already_skipped
            and item.messages == request.open_tool_messages
        )

    @staticmethod
    def _checkpoint_messages(
        checkpoint: ConversationSummaryCheckpoint,
    ) -> tuple[ModelMessage, ...]:
        return summary_system_messages(checkpoint)

    def _include_messages(
        self,
        request: ContextRequest,
        budget: ContextBudget,
        selected: list[ConversationInteraction],
        checkpoints: tuple[ConversationSummaryCheckpoint, ...],
        current: tuple[ModelMessage, ...],
        later: tuple[ModelMessage, ...],
        candidates: tuple[ModelMessage, ...],
        source: str,
    ):
        included = list(current if source == "memory" else later)
        omitted_count = 0
        estimate = self._counter.count(
            self._assemble(request, selected, checkpoints, current, later), request.tools
        )
        for message in candidates:
            proposed = (*included, message)
            if source == "memory":
                assembled = self._assemble(request, selected, checkpoints, proposed, later)
            else:
                assembled = self._assemble(request, selected, checkpoints, current, proposed)
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
        checkpoints: tuple[ConversationSummaryCheckpoint, ...],
        memory: tuple[ModelMessage, ...],
        skills: tuple[ModelMessage, ...],
    ) -> tuple[ModelMessage, ...]:
        segments = [
            (interaction.sequence_from, interaction.sequence_through, interaction.messages)
            for interaction in selected
        ]
        ordered = sorted(segments, key=lambda item: item[0])
        if any(earlier[1] >= later[0] for earlier, later in zip(ordered, ordered[1:])):
            raise ValueError("Context history segments overlap")
        history = tuple(message for _, _, messages in ordered for message in messages)
        summaries = tuple(
            message
            for checkpoint in checkpoints
            for message in BoundedContextProvider._checkpoint_messages(checkpoint)
        )
        return (
            request.system_message,
            *memory,
            *skills,
            *summaries,
            *history,
            request.current_input,
            *request.open_tool_messages,
        )
