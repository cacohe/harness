import hashlib
import json
from dataclasses import dataclass

from src.core.commands import CommandContext, RunLimits, SubmitRun
from src.core.errors import InvalidTransition, RunErrorCode
from src.core.ids import Clock, IdGenerator, ProfileId, SessionId, SubmissionId
from src.core.journal import (
    JournalCommit,
    JournalError,
    RunJournal,
    SubmissionRecord,
    TranscriptEntry,
)
from src.core.state import RunState, TaskStateMachine


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
        head = await self._journal.load_session(session_id)
        accepted = await self._replay(session_id, submission_id, digest)
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
                accepted = await self._replay(session_id, submission_id, digest)
                if accepted is not None:
                    return accepted
            raise
        return AcceptedRun(transition.state, receipt.version)

    async def _replay(
        self, session_id: SessionId, submission_id: SubmissionId, digest: str
    ) -> AcceptedRun | None:
        accepted = await self._journal.find_submission(session_id, submission_id)
        if accepted is None:
            return None
        if accepted.submission.digest != digest:
            raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
        return AcceptedRun(accepted.state, accepted.version)


def _message_digest(
    session_id: SessionId, submission_id: SubmissionId, profile_id: ProfileId, message: str
) -> str:
    """Hash the UTF-8 JSON tuple of identities and exact user text."""
    if type(message) is not str:
        raise InvalidTransition
    try:
        payload = json.dumps(
            [session_id.value, submission_id.value, profile_id.value, message],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except UnicodeEncodeError:
        raise InvalidTransition from None
    return hashlib.sha256(payload).hexdigest()
