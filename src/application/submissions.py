import hashlib
import json
from dataclasses import dataclass

from src.core.commands import CommandContext, RunLimits, SubmitRun
from src.core.errors import InvalidTransition, RunErrorCode
from src.core.ids import Clock, IdGenerator, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    JournalCommit,
    JournalError,
    RunJournal,
    SubmissionAcceptance,
    SubmissionRecord,
    TranscriptEntry,
)
from src.core.state import RunState, RunStatus, TaskStateMachine


@dataclass(frozen=True)
class AcceptedRun:
    """The Run state and Session CAS version from the original acceptance."""

    state: RunState
    version: int


class SubmissionService:
    def __init__(
        self, journal: RunJournal, ids: IdGenerator, clock: Clock, limits: RunLimits
    ) -> None:
        self._journal = journal
        self._ids = ids
        self._clock = clock
        self._limits = limits

    async def accept(
        self,
        session_id: SessionId,
        message: str,
        submission_id: SubmissionId,
        profile_id: ProfileId,
    ) -> AcceptedRun:
        digest = _message_digest(session_id, submission_id, profile_id, message)
        head = await self._journal.load_head(session_id)
        accepted = await self._replay(session_id, submission_id, profile_id, digest)
        if accepted is not None:
            return accepted
        if head.state is not None and not head.state.status.terminal:
            raise JournalError(RunErrorCode.JOURNAL_CONFLICT)

        run_id = self._ids.run()
        command = SubmitRun(
            run_id,
            session_id,
            submission_id,
            profile_id,
            message,
            context=CommandContext(self._clock.now(), 0, (self._ids.event(),)),
        )
        transition = TaskStateMachine.apply(RunState.create(command, self._limits), command)
        batch = JournalCommit(
            state=transition.state,
            events=transition.events,
            submission=SubmissionRecord(session_id, run_id, submission_id, digest),
            transcript=(TranscriptEntry(session_id, run_id, submission_id, "user", message),),
        )
        try:
            receipt = await self._journal.commit(batch, expected_version=head.version)
        except JournalError as error:
            if error.failure.code is RunErrorCode.JOURNAL_CONFLICT:
                # Another writer may have accepted this submission after the lookup.
                accepted = await self._replay(session_id, submission_id, profile_id, digest)
                if accepted is not None:
                    return accepted
            raise
        return AcceptedRun(transition.state, receipt.version)

    async def _replay(
        self,
        session_id: SessionId,
        submission_id: SubmissionId,
        profile_id: ProfileId,
        digest: str,
    ) -> AcceptedRun | None:
        accepted = await self._journal.find_submission(session_id, submission_id)
        if accepted is None:
            return None
        state = _validate_acceptance(accepted, session_id, submission_id)
        if accepted.submission.digest != digest:
            raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
        if state.profile_id != profile_id:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return AcceptedRun(state, accepted.version)


def _validate_acceptance(
    receipt: SubmissionAcceptance, session_id: SessionId, submission_id: SubmissionId
) -> RunState:
    try:
        if type(receipt) is not SubmissionAcceptance:
            raise ValueError
        record, state, version = receipt.submission, receipt.state, receipt.version
        if (
            type(record) is not SubmissionRecord
            or type(state) is not RunState
            or type(version) is not int
            or version < 1
            or type(record.digest) is not str
            or len(record.digest) != 64
            or any(character not in "0123456789abcdef" for character in record.digest)
            or type(record.session_id) is not SessionId
            or type(record.session_id.value) is not str
            or record.session_id != session_id
            or type(record.submission_id) is not SubmissionId
            or type(record.submission_id.value) is not str
            or record.submission_id != submission_id
            or type(record.run_id) is not RunId
            or type(record.run_id.value) is not str
            or type(state.profile_id) is not ProfileId
            or type(state.profile_id.value) is not str
            or type(state.limits) is not RunLimits
        ):
            raise ValueError
        limits = RunLimits(
            state.limits.max_steps,
            state.limits.max_elapsed_seconds,
            state.limits.max_inline_result_bytes,
        )
        original = RunState(
            RunId(record.run_id.value),
            session_id,
            submission_id,
            ProfileId(state.profile_id.value),
            limits,
            version=1,
            status=RunStatus.RUNNING,
        )
        if (
            type(state.session_id) is not SessionId
            or type(state.session_id.value) is not str
            or type(state.submission_id) is not SubmissionId
            or type(state.submission_id.value) is not str
            or type(state.run_id) is not RunId
            or type(state.run_id.value) is not str
            or state.status is not RunStatus.RUNNING
            or type(state.version) is not int
            or type(state.elapsed_seconds) not in (int, float)
            or type(state.used_step_ids) is not frozenset
            or type(state.used_tool_call_ids) is not frozenset
            or state.current_step is not None
            or state.final_text is not None
            or state.failure is not None
            or state != original
        ):
            raise ValueError
        return original
    except (AttributeError, TypeError, ValueError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


def _message_digest(
    session_id: SessionId, submission_id: SubmissionId, profile_id: ProfileId, message: str
) -> str:
    """Hash the UTF-8 JSON tuple of identities and exact user text."""
    if (
        type(session_id) is not SessionId
        or type(submission_id) is not SubmissionId
        or type(profile_id) is not ProfileId
        or type(message) is not str
    ):
        raise InvalidTransition
    try:
        for identifier in (session_id, submission_id, profile_id):
            if type(identifier.value) is not str:
                raise ValueError
            type(identifier)(identifier.value)
        payload = json.dumps(
            [session_id.value, submission_id.value, profile_id.value, message],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (AttributeError, ValueError):
        raise InvalidTransition from None
    return hashlib.sha256(payload).hexdigest()
