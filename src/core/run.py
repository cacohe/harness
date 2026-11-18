from __future__ import annotations

from dataclasses import dataclass, replace

from src.core.artifacts import ArtifactExternalizer, ArtifactRef, ExternalizedContent
from src.core.commands import (
    AdvanceRun,
    CancelRun,
    CommandContext,
    CompleteRun,
    CoreCommand,
    DecideToolCall,
    FailRun,
    FinishModelCall,
    FinishStep,
    FinishToolCall,
    ResumeRun,
    StartModelCall,
    StartStep,
    StartToolCall,
    SubmitRun,
)
from src.core.context import (
    ContextBudget,
    ContextBudgetExceeded,
    ContextProvider,
    ContextRequest,
    TokenCounter,
    summary_system_messages,
)
from src.core.errors import InvalidTransition, RunErrorCode, RunFailure
from src.core.events import DomainEvent
from src.core.ids import Clock, IdGenerator
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactAttachmentRole,
    ConversationSummaryCheckpoint,
    JournalCommit,
    RunJournal,
    TranscriptEntry,
)
from src.core.model import (
    ModelGateway,
    ModelMessage,
    ModelRequest,
    ModelRole,
    snapshot_model_response,
)
from src.core.run_content import externalize_final_text, externalize_tool_result
from src.core.run_projection import model_tool_requests, tool_failure, tool_transcript
from src.core.run_receipt import validate_commit_receipt
from src.core.state import RunState, RunStatus, StateTransition, TaskStateMachine, ToolCallState
from src.core.tools import (
    ToolApproval,
    ToolGateway,
    ToolRequest,
    ToolResultStatus,
    snapshot_tool_result,
)


@dataclass(frozen=True, slots=True)
class RunOutcome:
    state: RunState
    journal_version: int
    sequence: int
    events: tuple[DomainEvent, ...] = ()


def _matches_trusted_context_shape(
    messages: tuple[ModelMessage, ...], request: ContextRequest
) -> bool:
    if not messages or messages[0] != request.system_message:
        return False
    prefix_end = 1
    while prefix_end < len(messages) and messages[prefix_end].role is ModelRole.SYSTEM:
        prefix_end += 1
    supplied_prefix = messages[1:prefix_end]
    trusted_groups = (
        *((message,) for message in request.memory_messages),
        *((message,) for message in request.skill_messages),
        *(summary_system_messages(checkpoint) for checkpoint in request.summary_checkpoints),
    )
    positions = {0}
    for group in trusted_groups:
        next_positions = set(positions)
        for position in positions:
            through = position + len(group)
            if supplied_prefix[position:through] == group:
                next_positions.add(through)
        positions = next_positions
    if len(supplied_prefix) not in positions:
        return False

    suffix_length = len(request.open_tool_messages) + 1
    if len(messages) < prefix_end + suffix_length:
        return False
    if messages[-1] != request.current_input:
        return False
    if request.open_tool_messages and (messages[-suffix_length:-1] != request.open_tool_messages):
        return False
    history = messages[prefix_end:-suffix_length]
    return all(message.role is not ModelRole.SYSTEM for message in history)


class RunCoordinator:
    """Coordinate one bounded model decision or one recoverable tool batch."""

    def __init__(
        self,
        *,
        context: ContextProvider,
        model: ModelGateway,
        tools: ToolGateway,
        artifacts: ArtifactExternalizer,
        journal: RunJournal,
        clock: Clock,
        ids: IdGenerator,
        context_request: ContextRequest,
        context_budget: ContextBudget,
        context_counter: TokenCounter,
    ) -> None:
        self._context = context
        self._model = model
        self._tools = tools
        self._artifacts = artifacts
        self._journal = journal
        self._clock = clock
        self._ids = ids
        self._request = context_request
        self._budget = context_budget
        self._counter = context_counter

    async def advance(self, state: RunState, command: CoreCommand) -> RunOutcome:
        if type(state) is not RunState or command.run_id != state.run_id:
            raise InvalidTransition
        if command.session_id != state.session_id:
            raise InvalidTransition
        if (
            self._request.session_id != state.session_id
            or self._request.profile_id != state.profile_id
        ):
            raise InvalidTransition
        if state.status.terminal:
            return RunOutcome(state, state.journal_version, state.journal_sequence)

        driver = _Driver(self, state, command.context)
        if isinstance(command, AdvanceRun):
            if state.status is not RunStatus.RUNNING:
                raise InvalidTransition
            if state.current_step is not None:
                return await self._continue_tools(driver)
        else:
            driver.apply(command)
            if driver.state.status.terminal:
                return await driver.commit()
        if isinstance(command, (CancelRun, DecideToolCall)):
            return await driver.commit()
        if isinstance(command, ResumeRun):
            return await self._continue_tools(driver)
        if not isinstance(command, (SubmitRun, AdvanceRun)):
            raise InvalidTransition
        return await self._model_step(driver)

    async def _model_step(self, driver: _Driver) -> RunOutcome:
        step_id = self._ids.step()
        model_call_id = self._ids.model_call()
        if driver.apply(
            StartStep(driver.run_id, driver.session_id, step_id, context=driver.context())
        ):
            return await driver.commit()
        if driver.apply(
            StartModelCall(
                driver.run_id,
                driver.session_id,
                step_id,
                model_call_id,
                context=driver.context(),
            )
        ):
            return await driver.commit()
        try:
            window = await self._context.build(self._request, self._budget)
            estimate = self._counter.count(window.messages, window.tools)
            if (
                window.policy_version != self._request.policy_version
                or window.tools != self._request.tools
                or not _matches_trusted_context_shape(window.messages, self._request)
            ):
                raise ValueError("Context window does not match its request binding")
            if (
                window.budget != self._budget
                or estimate.tokens > self._budget.input_allowance_tokens
            ):
                raise ContextBudgetExceeded
        except ContextBudgetExceeded:
            return await driver.fail(RunErrorCode.CONTEXT_BUDGET_EXCEEDED)
        except Exception:
            return await driver.fail(RunErrorCode.INVARIANT_VIOLATION)
        try:
            response = snapshot_model_response(
                await self._model.complete(ModelRequest(window.messages, window.tools))
            )
        except Exception:
            return await driver.fail(RunErrorCode.MODEL_FAILED)
        try:
            requests = model_tool_requests(response, window.tools)
        except (InvalidTransition, ValueError):
            return await driver.fail(RunErrorCode.MODEL_FAILED)
        model_text = response.text or ""
        final_artifacts: tuple[ArtifactRef, ...] = ()
        if (
            requests
            and len(model_text.encode("utf-8")) > driver.state.limits.max_inline_result_bytes
        ):
            return await driver.fail(RunErrorCode.MODEL_FAILED)
        if not requests:
            try:
                model_text, final_artifacts = await externalize_final_text(
                    model_text, driver.state.limits.max_inline_result_bytes, self._artifacts
                )
                driver.attach_refs(final_artifacts, ArtifactAttachmentRole.FINAL_OUTPUT, "final")
            except Exception:
                return await driver.fail(RunErrorCode.ARTIFACT_FAILED)
        if driver.apply(
            FinishModelCall(
                driver.run_id,
                driver.session_id,
                step_id,
                model_call_id,
                model_text,
                requests,
                context=driver.context(),
            )
        ):
            return await driver.commit()
        driver.checkpoint = window.new_checkpoint
        if requests:
            return await driver.commit()
        if driver.apply(
            FinishStep(driver.run_id, driver.session_id, step_id, context=driver.context())
        ):
            return await driver.commit()
        driver.apply(
            CompleteRun(
                driver.run_id,
                driver.session_id,
                model_text,
                final_artifacts,
                context=driver.context(),
            )
        )
        driver.transcript = (driver.entry("assistant", model_text),)
        return await driver.commit()

    async def _continue_tools(self, driver: _Driver) -> RunOutcome:
        step = driver.state.current_step
        if step is None or step.model_call is None:
            raise InvalidTransition
        step_id = step.step_id
        model_call_id = step.model_call.model_call_id
        while step.next_tool_index < len(step.model_call.tool_requests):
            index = step.next_tool_index
            request = step.model_call.tool_requests[index]
            current = step.tool_calls[index] if index < len(step.tool_calls) else None
            if current is None:
                if driver.apply(
                    StartToolCall(
                        driver.run_id,
                        driver.session_id,
                        step_id,
                        model_call_id,
                        request.tool_call_id,
                        request.requires_approval,
                        context=driver.context(),
                    )
                ):
                    return await driver.commit()
                step = driver.state.current_step
                assert step is not None
                current = step.tool_calls[index]
                if driver.state.status is RunStatus.WAITING_APPROVAL:
                    return await driver.commit()
            result = await self._execute(driver, current)
            if driver.apply(
                FinishToolCall(
                    driver.run_id,
                    driver.session_id,
                    step_id,
                    model_call_id,
                    request.tool_call_id,
                    result,
                    context=driver.context(),
                )
            ):
                return await driver.commit()
            step = driver.state.current_step
            assert step is not None
        driver.transcript = tool_transcript(driver.state)
        if driver.apply(
            FinishStep(driver.run_id, driver.session_id, step_id, context=driver.context())
        ):
            return await driver.commit()
        return await driver.commit()

    async def _execute(
        self, driver: _Driver, call: ToolCallState
    ) -> ExternalizedContent | RunFailure:
        approval = ToolApproval.NOT_REQUIRED
        if call.approval is not None:
            if call.approval.approved is False:
                raise InvalidTransition
            approval = ToolApproval.APPROVED
        active_step = driver.state.current_step
        if active_step is None:
            raise InvalidTransition
        request = ToolRequest.from_call(
            call.request,
            run_id=driver.run_id,
            session_id=driver.session_id,
            step_id=active_step.step_id,
            approval=approval,
        )
        try:
            result = snapshot_tool_result(await self._tools.execute(request))
            if result.execution_key != request.idempotency_key:
                raise ValueError
        except Exception:
            driver.checkpoint = None
            return RunFailure(RunErrorCode.TOOL_FAILED)
        if result.status is not ToolResultStatus.SUCCESS:
            driver.checkpoint = None
            return tool_failure(result)
        try:
            externalized = await externalize_tool_result(
                result, driver.state.limits.max_inline_result_bytes, self._artifacts
            )
            driver.attach_refs(
                externalized.artifacts,
                ArtifactAttachmentRole.TOOL_RESULT,
                call.request.tool_call_id,
            )
            return externalized
        except Exception:
            driver.checkpoint = None
            return RunFailure(RunErrorCode.ARTIFACT_FAILED)


class _Driver:
    def __init__(self, owner: RunCoordinator, state: RunState, supplied: CommandContext) -> None:
        self.owner = owner
        self.state = state
        self.events: list[DomainEvent] = []
        self.transcript: tuple[TranscriptEntry, ...] = ()
        self.checkpoint: ConversationSummaryCheckpoint | None = None
        self.artifact_facts: list[ArtifactAttachmentFact] = []
        self.base_elapsed = supplied.elapsed_seconds
        self.started = owner._clock.monotonic()

    @property
    def run_id(self):
        return self.state.run_id

    @property
    def session_id(self):
        return self.state.session_id

    def context(self) -> CommandContext:
        elapsed = self.base_elapsed + max(0.0, self.owner._clock.monotonic() - self.started)
        return CommandContext(self.owner._clock.now(), elapsed, (self.owner._ids.event(),))

    def apply(self, command: CoreCommand) -> bool:
        transition: StateTransition = TaskStateMachine.apply(self.state, command)
        self.state = transition.state
        self.events.extend(transition.events)
        return self.state.status.terminal

    async def fail(self, code: RunErrorCode) -> RunOutcome:
        self.checkpoint = None
        self.apply(FailRun(self.run_id, self.session_id, RunFailure(code), context=self.context()))
        return await self.commit()

    def entry(self, role: str, content: str) -> TranscriptEntry:
        return TranscriptEntry(
            self.state.session_id,
            self.state.run_id,
            self.state.submission_id,
            role,
            content,
        )

    def attach_refs(
        self, refs: tuple[ArtifactRef, ...], role: ArtifactAttachmentRole, owner_id: str
    ) -> None:
        self.artifact_facts.extend(
            ArtifactAttachmentFact(
                self.state.session_id,
                self.state.run_id,
                self.state.submission_id,
                owner_id,
                role,
                ArtifactAttachmentAction.ATTACHED,
                ref,
            )
            for ref in refs
        )

    async def commit(self) -> RunOutcome:
        expected = self.state.journal_version
        sequence = self.state.journal_sequence + len(self.events) + len(self.transcript)
        persisted = replace(self.state, journal_version=expected + 1, journal_sequence=sequence)
        facts = list(self.artifact_facts)
        if self.checkpoint is not None:
            owner = (
                f"{self.checkpoint.policy_version}:"
                f"{self.checkpoint.covered_from_sequence}-{self.checkpoint.covered_through_sequence}"
            )
            known = {(fact.role, fact.owner_id, fact.artifact.artifact_id) for fact in facts}
            for ref in self.checkpoint.artifacts:
                key = (ArtifactAttachmentRole.SUMMARY, owner, ref.artifact_id)
                if key not in known:
                    facts.append(
                        ArtifactAttachmentFact(
                            persisted.session_id,
                            persisted.run_id,
                            persisted.submission_id,
                            owner,
                            ArtifactAttachmentRole.SUMMARY,
                            ArtifactAttachmentAction.ATTACHED,
                            ref,
                        )
                    )
        receipt = await self.owner._journal.commit(
            JournalCommit(
                persisted,
                events=tuple(self.events),
                transcript=self.transcript,
                summary=self.checkpoint,
                artifact_facts=tuple(facts),
            ),
            expected,
        )
        receipt = validate_commit_receipt(
            receipt,
            JournalCommit(
                persisted,
                events=tuple(self.events),
                transcript=self.transcript,
                summary=self.checkpoint,
                artifact_facts=tuple(facts),
            ),
            expected_version=expected,
            previous_sequence=self.state.journal_sequence,
        )
        return RunOutcome(persisted, receipt.version, receipt.sequence, tuple(self.events))
