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
)
from src.core.errors import InvalidTransition, RunErrorCode, RunFailure
from src.core.events import DomainEvent
from src.core.ids import Clock, IdGenerator
from src.core.journal import (
    ConversationSummaryCheckpoint,
    JournalCommit,
    RunJournal,
    TranscriptEntry,
)
from src.core.model import ModelGateway, ModelRequest, snapshot_model_response
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
                or window.messages[0] != self._request.system_message
                or window.messages[-1] != self._request.current_input
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
            return await externalize_tool_result(
                result, driver.state.limits.max_inline_result_bytes, self._artifacts
            )
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

    async def commit(self) -> RunOutcome:
        expected = self.state.journal_version
        sequence = self.state.journal_sequence + len(self.events) + len(self.transcript)
        persisted = replace(self.state, journal_version=expected + 1, journal_sequence=sequence)
        receipt = await self.owner._journal.commit(
            JournalCommit(
                persisted,
                events=tuple(self.events),
                transcript=self.transcript,
                summary=self.checkpoint,
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
            ),
            expected_version=expected,
            previous_sequence=self.state.journal_sequence,
        )
        return RunOutcome(persisted, receipt.version, receipt.sequence, tuple(self.events))
