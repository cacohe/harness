from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from src.core.artifacts import ArtifactRef, snapshot_artifact_ref
from src.core.errors import RunErrorCode, RunFailure
from src.core.events import DomainEvent
from src.core.ids import ProfileId, RunId, SessionId, SubmissionId
from src.core.state import RunState


class JournalError(ValueError):
    """A journal failure with a stable, public classification."""

    def __init__(self, code: RunErrorCode) -> None:
        self.failure = RunFailure(code)
        super().__init__(self.failure.message)


@dataclass(frozen=True)
class SubmissionRecord:
    session_id: SessionId
    run_id: RunId
    submission_id: SubmissionId
    digest: str


@dataclass(frozen=True)
class SubmissionAcceptance:
    """The immutable record, Run state, and Session CAS version at acceptance."""

    submission: SubmissionRecord
    state: RunState
    version: int


@dataclass(frozen=True)
class TranscriptEntry:
    session_id: SessionId
    run_id: RunId
    submission_id: SubmissionId
    role: str
    content: str
    sequence: int = 0  # Assigned by the journal on append.


@dataclass(frozen=True)
class ConversationSummaryCheckpoint:
    session_id: SessionId
    covered_through_sequence: int
    policy_version: str
    profile_id: ProfileId
    summary: str
    covered_from_sequence: int = 1
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not SessionId
            or type(self.profile_id) is not ProfileId
            or type(self.covered_from_sequence) is not int
            or type(self.covered_through_sequence) is not int
            or not 1 <= self.covered_from_sequence <= self.covered_through_sequence <= (1 << 63) - 1
            or type(self.policy_version) is not str
            or not self.policy_version
            or self.policy_version.strip() != self.policy_version
            or len(self.policy_version.encode("utf-8")) > 256
            or type(self.summary) is not str
            or not self.summary.strip()
            or len(self.summary.encode("utf-8")) > 1_048_576
            or type(self.artifacts) is not tuple
            or len(self.artifacts) > 16
            or any(type(ref) is not ArtifactRef for ref in self.artifacts)
        ):
            raise ValueError("Summary checkpoint is invalid")
        object.__setattr__(self, "session_id", SessionId(self.session_id.value))
        object.__setattr__(self, "profile_id", ProfileId(self.profile_id.value))
        object.__setattr__(
            self, "artifacts", tuple(snapshot_artifact_ref(ref) for ref in self.artifacts)
        )

    @property
    def key(self) -> tuple[SessionId, int, str, ProfileId]:
        return self.session_id, self.covered_through_sequence, self.policy_version, self.profile_id


@dataclass(frozen=True)
class JournalCommit:
    state: RunState
    events: tuple[DomainEvent, ...] = ()
    submission: SubmissionRecord | None = None
    transcript: tuple[TranscriptEntry, ...] = ()
    summary: ConversationSummaryCheckpoint | None = None


@dataclass(frozen=True)
class JournalEvent:
    sequence: int
    event: DomainEvent


@dataclass(frozen=True)
class JournalHead:
    version: int
    sequence: int
    state: RunState | None = None


@dataclass(frozen=True)
class JournalSnapshot:
    session_id: SessionId
    version: int
    sequence: int
    state: RunState | None = None
    events: tuple[JournalEvent, ...] = ()
    transcript: tuple[TranscriptEntry, ...] = ()
    summaries: tuple[ConversationSummaryCheckpoint, ...] = ()
    submission: SubmissionRecord | None = None


class RunJournal(Protocol):
    def commit(self, batch: JournalCommit, expected_version: int) -> Awaitable[JournalSnapshot]:
        """Atomically compare and advance the Session's commit version by one.

        The Session sequence starts at one and advances for each event, then each
        transcript entry, in the supplied order. RunState.version tracks state
        transitions independently. The submitted RunState must project the successful
        commit's journal_version and journal_sequence. Return a receipt containing
        this batch's entries.

        An identical acceptance batch for a recorded submission returns its
        original receipt even after the Session advances. A changed digest or
        acceptance batch and a stale CAS raise JournalError with JOURNAL_CONFLICT.
        Invalid members and duplicate event IDs raise INVARIANT_VIOLATION.
        Event timestamps must be plain UTC datetimes with fold zero.
        A rejected or replayed batch makes no writes.
        """
        ...

    def load_head(self, session_id: SessionId) -> Awaitable[JournalHead]:
        """Read the current projection and CAS counters without reading Session history.

        A missing Session has version and sequence zero and no state.
        """
        ...

    def load_session(
        self, session_id: SessionId, after_sequence: int = 0
    ) -> Awaitable[JournalSnapshot]:
        """Read the current state and version and entries strictly after the cursor.

        Each entry collection is ordered by the shared sequence. Summaries are
        indexed independently of the cursor. A missing Session has version and
        sequence zero and no state. Zero is the cursor before the first entry.
        """
        ...

    def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> Awaitable[SubmissionAcceptance | None]:
        """Read the original acceptance, independent of subsequent Session changes.

        Persist the acceptance with its commit, so a durable journal can return
        the same snapshot and version after reopening.
        """
        ...
