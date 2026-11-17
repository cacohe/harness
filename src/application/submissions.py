import hashlib
import json
import math
from dataclasses import dataclass, replace

from src.core.commands import CommandContext, RunLimits, SubmitRun
from src.core.errors import InvalidTransition, RunErrorCode
from src.core.ids import Clock, IdGenerator, ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    JournalCommit,
    JournalError,
    JournalHead,
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
        accepted = await self._replay(session_id, submission_id, profile_id, digest, head)
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
        persisted_state = replace(
            transition.state,
            journal_version=head.version + 1,
            journal_sequence=head.sequence + len(transition.events) + 1,
        )
        batch = JournalCommit(
            state=persisted_state,
            events=transition.events,
            submission=SubmissionRecord(session_id, run_id, submission_id, digest),
            transcript=(TranscriptEntry(session_id, run_id, submission_id, "user", message),),
        )
        try:
            receipt = await self._journal.commit(batch, expected_version=head.version)
        except JournalError as error:
            if error.failure.code is RunErrorCode.JOURNAL_CONFLICT:
                # Another writer may have accepted this submission after the lookup.
                accepted = await self._replay(session_id, submission_id, profile_id, digest, head)
                if accepted is not None:
                    return accepted
            raise
        return AcceptedRun(persisted_state, receipt.version)

    async def _replay(
        self,
        session_id: SessionId,
        submission_id: SubmissionId,
        profile_id: ProfileId,
        digest: str,
        head: JournalHead,
    ) -> AcceptedRun | None:
        accepted = await self._journal.find_submission(session_id, submission_id)
        if accepted is None:
            return None
        state, version, saved_digest = _validate_acceptance(accepted, session_id, submission_id)
        head_version = _head_version(head)
        if version > head_version:
            # A writer may have accepted the submission between the head and lookup reads.
            head = await self._journal.load_head(session_id)
            head_version = _head_version(head)
        if version > head_version:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        if version == head_version:
            head_state = _initial_state(getattr(head, "state", None))
            if not _same_initial_state(state, head_state):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        if saved_digest != digest:
            raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
        if state.profile_id != profile_id:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return AcceptedRun(state, version)


def _head_version(head: JournalHead) -> int:
    version = getattr(head, "version", None) if type(head) is JournalHead else None
    if type(version) is not int or version < 0:
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
    return version


def _validate_acceptance(
    receipt: SubmissionAcceptance, session_id: SessionId, submission_id: SubmissionId
) -> tuple[RunState, int, str]:
    try:
        if type(receipt) is not SubmissionAcceptance:
            raise ValueError
        record, version = receipt.submission, receipt.version
        if (
            type(record) is not SubmissionRecord
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
        ):
            raise ValueError
        state = _initial_state(receipt.state)
        if (
            state.session_id != session_id
            or state.submission_id != submission_id
            or state.run_id != record.run_id
        ):
            raise ValueError
        return state, version, record.digest
    except (AttributeError, TypeError, ValueError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


def _initial_state(state: RunState | None) -> RunState:
    """Decode only the fields an acceptance commit can contain."""
    try:
        if (
            type(state) is not RunState
            or type(state.session_id) is not SessionId
            or type(state.session_id.value) is not str
            or type(state.submission_id) is not SubmissionId
            or type(state.submission_id.value) is not str
            or type(state.run_id) is not RunId
            or type(state.run_id.value) is not str
            or type(state.profile_id) is not ProfileId
            or type(state.profile_id.value) is not str
            or type(state.limits) is not RunLimits
            or state.status is not RunStatus.RUNNING
            or type(state.version) is not int
            or state.version != 1
            or not _is_exact_number(state.elapsed_seconds)
            or state.elapsed_seconds != 0
            or type(state.used_step_ids) is not frozenset
            or state.used_step_ids
            or type(state.used_tool_call_ids) is not frozenset
            or state.used_tool_call_ids
            or state.current_step is not None
            or state.final_text is not None
            or state.final_artifacts
            or state.failure is not None
            or type(state.journal_version) is not int
            or state.journal_version < 1
            or type(state.journal_sequence) is not int
            or state.journal_sequence < 2
        ):
            raise ValueError
        limits = state.limits
        elapsed = limits.max_elapsed_seconds
        if (
            type(limits.max_steps) is not int
            or type(limits.max_inline_result_bytes) is not int
            or not _is_exact_number(elapsed)
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError
        return RunState(
            RunId(state.run_id.value),
            SessionId(state.session_id.value),
            SubmissionId(state.submission_id.value),
            ProfileId(state.profile_id.value),
            RunLimits(limits.max_steps, elapsed, limits.max_inline_result_bytes),
            version=1,
            status=RunStatus.RUNNING,
            elapsed_seconds=state.elapsed_seconds,
            journal_version=state.journal_version,
            journal_sequence=state.journal_sequence,
        )
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


def _is_exact_number(value: object) -> bool:
    value_type = type(value)
    return value_type is int or value_type is float


def _same_initial_state(left: RunState, right: RunState) -> bool:
    # Both states have been decoded. Preserve the journal's distinction between numeric types.
    return (
        left == right
        and type(left.limits.max_elapsed_seconds) is type(right.limits.max_elapsed_seconds)
        and repr(left.elapsed_seconds) == repr(right.elapsed_seconds)
    )


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
