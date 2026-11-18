from __future__ import annotations

import asyncio
import math
import sqlite3
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import TypeVar

from src.adapters.context.journal_history import (
    MAX_TRANSCRIPT_ROWS_PER_INTERACTION,
    newest_transcript_interactions,
)
from src.adapters.persistence.memory import (
    _canonical_copy,
    _coverage_ranges,
    _validate_aggregate_request,
    _validate_members,
)
from src.adapters.persistence.sqlite_codec import decode, encode
from src.adapters.persistence.sqlite_integrity import (
    authenticated_root_hash,
    coverage_block_content_hash,
    coverage_empty_hash,
    coverage_leaf_hash,
    coverage_parent_hash,
    genesis_hash,
    record_hash,
    retirement_authenticated_root_hash,
    retirement_empty_hash,
    retirement_fact_hash,
    retirement_genesis_hash,
    retirement_leaf_hash,
    retirement_parent_hash,
    retirement_state_hash,
    run_empty_hash,
    run_leaf_hash,
    run_parent_hash,
    state_record_hash,
    summary_empty_hash,
    summary_leaf_hash,
    summary_parent_hash,
    summary_record_hash,
    validate_hash,
)
from src.core.context import (
    ConversationCoverage,
    ConversationCoverageSegment,
    ConversationPage,
)
from src.core.errors import RunErrorCode
from src.core.ids import ProfileId, RunId, SessionId, SubmissionId
from src.core.journal import (
    ArtifactAttachmentAction,
    ArtifactAttachmentFact,
    ArtifactFactRecord,
    ArtifactRetirementReservation,
    ArtifactRetirementState,
    ConversationSummaryCheckpoint,
    JournalCommit,
    JournalError,
    JournalEvent,
    JournalHead,
    JournalSnapshot,
    SubmissionAcceptance,
    TranscriptEntry,
    TranscriptEntryType,
)
from src.core.run_receipt import validate_commit_receipt
from src.core.state import RunState

_SCHEMA_VERSION = 6
_COVERAGE_BLOCK_SIZE = 128
_COVERAGE_DEPTH = 32
_SUMMARY_DEPTH = 32
_RUN_DEPTH = 32
_RETIREMENT_DEPTH = 32
_MAX_ARTIFACT_FACTS_PER_LOOKUP = 1_000_000
_T = TypeVar("_T")
_IdT = TypeVar("_IdT", SessionId, RunId, SubmissionId, ProfileId)

_DDL = """
CREATE TABLE sessions (
  session_id TEXT PRIMARY KEY,
  version INTEGER NOT NULL CHECK(version >= 0),
  sequence INTEGER NOT NULL CHECK(sequence >= 0),
  state_json TEXT NOT NULL,
  ledger_hash TEXT NOT NULL,
  coverage_hash TEXT NOT NULL,
  transcript_count INTEGER NOT NULL CHECK(transcript_count >= 0),
  summary_hash TEXT NOT NULL,
  summary_count INTEGER NOT NULL CHECK(summary_count >= 0),
  session_state_hash TEXT NOT NULL,
  run_projection_hash TEXT NOT NULL,
  run_count INTEGER NOT NULL CHECK(run_count >= 0),
  authenticated_root TEXT NOT NULL
) STRICT;
CREATE TABLE journal_entries (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  kind TEXT NOT NULL CHECK(kind IN ('event', 'transcript')),
  previous_hash TEXT NOT NULL,
  record_hash TEXT NOT NULL,
  PRIMARY KEY(session_id, sequence)
) STRICT;
CREATE TABLE runs (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  status TEXT NOT NULL,
  state_version INTEGER NOT NULL CHECK(state_version >= 0),
  run_index INTEGER NOT NULL CHECK(run_index >= 0),
  state_hash TEXT NOT NULL,
  state_json TEXT NOT NULL,
  PRIMARY KEY(session_id, run_id),
  UNIQUE(session_id, submission_id)
) STRICT;
CREATE INDEX runs_active_by_session ON runs(session_id, status);
CREATE TRIGGER runs_identity_immutable BEFORE UPDATE OF session_id, run_id, submission_id, profile_id, run_index ON runs BEGIN
  SELECT RAISE(ABORT, 'Run projection identity is immutable');
END;
CREATE TRIGGER runs_no_delete BEFORE DELETE ON runs BEGIN
  SELECT RAISE(ABORT, 'Run projections cannot be deleted');
END;
CREATE TRIGGER sessions_no_delete BEFORE DELETE ON sessions BEGIN
  SELECT RAISE(ABORT, 'Sessions cannot be deleted');
END;
CREATE TABLE run_projection_nodes (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  level INTEGER NOT NULL CHECK(level >= 0 AND level <= 32),
  node_index INTEGER NOT NULL CHECK(node_index >= 0),
  digest TEXT NOT NULL,
  item_count INTEGER NOT NULL CHECK(item_count >= 0),
  PRIMARY KEY(session_id, level, node_index)
) STRICT;
CREATE TABLE events (
  event_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  run_id TEXT NOT NULL,
  event_json TEXT NOT NULL,
  UNIQUE(session_id, sequence)
) STRICT;
CREATE INDEX events_by_session_sequence ON events(session_id, sequence);
CREATE TABLE transcript (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  sequence INTEGER NOT NULL CHECK(sequence > 0),
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  entry_type TEXT NOT NULL CHECK(entry_type IN ('message', 'assistant_tool_batch', 'tool_result')),
  format_version INTEGER NOT NULL CHECK(format_version = 1),
  entry_json TEXT NOT NULL,
  PRIMARY KEY(session_id, sequence)
) STRICT;
CREATE INDEX transcript_by_session_sequence ON transcript(session_id, sequence);
CREATE INDEX transcript_run_input ON transcript(session_id, run_id, submission_id, role, sequence);
CREATE TABLE transcript_coverage_blocks (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  block_index INTEGER NOT NULL CHECK(block_index >= 0),
  first_sequence INTEGER NOT NULL CHECK(first_sequence > 0),
  last_sequence INTEGER NOT NULL CHECK(last_sequence >= first_sequence),
  item_count INTEGER NOT NULL CHECK(item_count > 0 AND item_count <= 128),
  content_hash TEXT NOT NULL,
  PRIMARY KEY(session_id, block_index)
) STRICT;
CREATE INDEX transcript_coverage_blocks_by_sequence
ON transcript_coverage_blocks(session_id, first_sequence, last_sequence);
CREATE TABLE transcript_coverage_nodes (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  level INTEGER NOT NULL CHECK(level >= 0 AND level <= 32),
  node_index INTEGER NOT NULL CHECK(node_index >= 0),
  digest TEXT NOT NULL,
  item_count INTEGER NOT NULL CHECK(item_count >= 0),
  first_sequence INTEGER NOT NULL CHECK(first_sequence >= 0),
  last_sequence INTEGER NOT NULL CHECK(last_sequence >= 0),
  PRIMARY KEY(session_id, level, node_index)
) STRICT;
CREATE TABLE integrity_violations (
  session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
  reason TEXT NOT NULL
) STRICT, WITHOUT ROWID;
CREATE TRIGGER integrity_violations_keep_first BEFORE INSERT ON integrity_violations
WHEN EXISTS(SELECT 1 FROM integrity_violations WHERE session_id=NEW.session_id)
BEGIN
  SELECT RAISE(IGNORE);
END;
CREATE TRIGGER integrity_violations_no_update BEFORE UPDATE ON integrity_violations
BEGIN
  SELECT RAISE(ABORT, 'integrity violations are permanent');
END;
CREATE TRIGGER integrity_violations_no_delete BEFORE DELETE ON integrity_violations
BEGIN
  SELECT RAISE(ABORT, 'integrity violations are permanent');
END;
CREATE TRIGGER journal_entries_insert_guard BEFORE INSERT ON journal_entries
WHEN harness_journal_append_allowed() != 1
BEGIN
  SELECT RAISE(ABORT, 'journal entries require the commit append channel');
END;
CREATE TRIGGER transcript_insert_guard BEFORE INSERT ON transcript
WHEN harness_journal_append_allowed() != 1
BEGIN
  SELECT RAISE(ABORT, 'transcript entries require the commit append channel');
END;
CREATE TRIGGER journal_entries_update_violation AFTER UPDATE ON journal_entries
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'journal_entries updated');
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (NEW.session_id, 'journal_entries updated');
END;
CREATE TRIGGER journal_entries_delete_violation AFTER DELETE ON journal_entries
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'journal_entries deleted');
END;
CREATE TRIGGER transcript_update_violation AFTER UPDATE ON transcript
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'transcript updated');
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (NEW.session_id, 'transcript updated');
END;
CREATE TRIGGER transcript_delete_violation AFTER DELETE ON transcript
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'transcript deleted');
END;
CREATE TRIGGER sealed_coverage_block_update_violation
AFTER UPDATE ON transcript_coverage_blocks WHEN OLD.item_count = 128
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'sealed coverage block updated');
END;
CREATE TRIGGER coverage_block_delete_violation AFTER DELETE ON transcript_coverage_blocks
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'coverage block deleted');
END;
CREATE TABLE summary_checkpoints (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  summary_index INTEGER NOT NULL CHECK(summary_index >= 0),
  covered_from_sequence INTEGER NOT NULL,
  covered_through_sequence INTEGER NOT NULL,
  policy_version TEXT NOT NULL,
  profile_id TEXT NOT NULL,
  checkpoint_json TEXT NOT NULL,
  checkpoint_hash TEXT NOT NULL,
  PRIMARY KEY(session_id, covered_through_sequence, policy_version, profile_id),
  UNIQUE(session_id, summary_index)
) STRICT;
CREATE INDEX summary_context_lookup ON summary_checkpoints(
  session_id, profile_id, policy_version, covered_through_sequence DESC, summary_index DESC
);
CREATE TABLE summary_nodes (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  level INTEGER NOT NULL CHECK(level >= 0 AND level <= 32),
  node_index INTEGER NOT NULL CHECK(node_index >= 0),
  digest TEXT NOT NULL,
  item_count INTEGER NOT NULL CHECK(item_count >= 0),
  PRIMARY KEY(session_id, level, node_index)
) STRICT;
CREATE TRIGGER summary_checkpoints_insert_guard BEFORE INSERT ON summary_checkpoints
WHEN harness_journal_append_allowed() != 1
BEGIN
  SELECT RAISE(ABORT, 'summary checkpoints require the commit append channel');
END;
CREATE TRIGGER summary_checkpoints_update_violation AFTER UPDATE ON summary_checkpoints
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'summary checkpoint updated');
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (NEW.session_id, 'summary checkpoint updated');
END;
CREATE TRIGGER summary_checkpoints_delete_violation AFTER DELETE ON summary_checkpoints
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'summary checkpoint deleted');
END;
CREATE TRIGGER summary_nodes_insert_guard BEFORE INSERT ON summary_nodes
WHEN harness_journal_append_allowed() != 1
BEGIN
  SELECT RAISE(ABORT, 'summary nodes require the commit append channel');
END;
CREATE TRIGGER summary_nodes_update_violation AFTER UPDATE ON summary_nodes
WHEN harness_journal_append_allowed() != 1
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'summary node updated');
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (NEW.session_id, 'summary node updated');
END;
CREATE TRIGGER summary_nodes_delete_violation AFTER DELETE ON summary_nodes
BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason)
  VALUES (OLD.session_id, 'summary node deleted');
END;
CREATE TABLE submissions (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  submission_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  digest TEXT NOT NULL,
  intent_json TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  acceptance_state_json TEXT NOT NULL,
  acceptance_version INTEGER NOT NULL,
  PRIMARY KEY(session_id, submission_id)
) STRICT;
CREATE INDEX submissions_by_identity ON submissions(session_id, submission_id, digest);
CREATE TABLE artifact_attachment_facts (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  fact_index INTEGER NOT NULL CHECK(fact_index >= 0),
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('tool_result', 'final_output', 'summary')),
  action TEXT NOT NULL CHECK(action IN ('attached', 'detached')),
  artifact_id TEXT NOT NULL,
  fact_json TEXT NOT NULL,
  PRIMARY KEY(session_id, fact_index)
) STRICT;
CREATE TABLE artifact_attachments (
  session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
  run_id TEXT NOT NULL,
  submission_id TEXT NOT NULL,
  owner_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('tool_result', 'final_output', 'summary')),
  artifact_id TEXT NOT NULL,
  fact_json TEXT NOT NULL,
  PRIMARY KEY(session_id, run_id, owner_id, role, artifact_id)
) STRICT;
CREATE INDEX artifact_facts_by_artifact ON artifact_attachment_facts(artifact_id);
CREATE INDEX artifact_facts_by_run ON artifact_attachment_facts(session_id, run_id, fact_index);
CREATE INDEX artifact_attachments_by_artifact ON artifact_attachments(artifact_id);
CREATE TABLE artifact_retirements (
  artifact_id TEXT PRIMARY KEY,
  artifact_index INTEGER NOT NULL UNIQUE CHECK(artifact_index >= 0),
  generation INTEGER NOT NULL CHECK(generation > 0),
  state TEXT NOT NULL CHECK(state IN ('available', 'retiring', 'retired', 'aborted')),
  fact_count INTEGER NOT NULL CHECK(fact_count > 0),
  fact_root TEXT NOT NULL,
  state_hash TEXT NOT NULL
) STRICT;
CREATE TABLE artifact_retirement_facts (
  artifact_id TEXT NOT NULL,
  fact_index INTEGER NOT NULL CHECK(fact_index >= 0),
  generation INTEGER NOT NULL CHECK(generation > 0),
  state TEXT NOT NULL CHECK(state IN ('available', 'retiring', 'retired', 'aborted')),
  previous_hash TEXT NOT NULL,
  fact_hash TEXT NOT NULL,
  fact_json TEXT NOT NULL,
  PRIMARY KEY(artifact_id, fact_index)
) STRICT;
CREATE TABLE artifact_retirement_nodes (
  level INTEGER NOT NULL CHECK(level >= 0 AND level <= 32),
  node_index INTEGER NOT NULL CHECK(node_index >= 0),
  digest TEXT NOT NULL,
  item_count INTEGER NOT NULL CHECK(item_count >= 0),
  PRIMARY KEY(level, node_index)
) STRICT;
CREATE TABLE artifact_retirement_meta (
  singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
  projection_root TEXT NOT NULL,
  artifact_count INTEGER NOT NULL CHECK(artifact_count >= 0),
  authenticated_root TEXT NOT NULL
) STRICT;
CREATE TABLE artifact_retirement_allocation_damage (
  singleton INTEGER PRIMARY KEY CHECK(singleton = 1)
) STRICT;
CREATE TRIGGER artifact_retirement_damage_update_guard
BEFORE UPDATE ON artifact_retirement_allocation_damage BEGIN
  SELECT RAISE(ABORT, 'retirement allocation damage is permanent');
END;
CREATE TRIGGER artifact_retirement_damage_delete_guard
BEFORE DELETE ON artifact_retirement_allocation_damage BEGIN
  SELECT RAISE(ABORT, 'retirement allocation damage is permanent');
END;
CREATE TRIGGER artifact_retirements_delete_damage AFTER DELETE ON artifact_retirements BEGIN
  INSERT INTO artifact_retirement_allocation_damage
  SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM artifact_retirement_allocation_damage WHERE singleton=1);
END;
CREATE TRIGGER artifact_retirements_index_damage AFTER UPDATE OF artifact_index ON artifact_retirements
WHEN NEW.artifact_index != OLD.artifact_index BEGIN
  INSERT INTO artifact_retirement_allocation_damage
  SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM artifact_retirement_allocation_damage WHERE singleton=1);
END;
CREATE TRIGGER artifact_retirements_replace_damage BEFORE INSERT ON artifact_retirements
WHEN EXISTS (
  SELECT 1 FROM artifact_retirements
  WHERE artifact_id=NEW.artifact_id AND artifact_index != NEW.artifact_index
) BEGIN
  INSERT INTO artifact_retirement_allocation_damage
  SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM artifact_retirement_allocation_damage WHERE singleton=1);
END;
CREATE INDEX artifact_retirement_facts_by_artifact
ON artifact_retirement_facts(artifact_id, fact_index);
CREATE TRIGGER artifact_facts_insert_guard BEFORE INSERT ON artifact_attachment_facts
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact facts require the commit append channel');
END;
CREATE TRIGGER artifact_facts_update_violation AFTER UPDATE ON artifact_attachment_facts BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason) VALUES(OLD.session_id, 'artifact fact updated');
END;
CREATE TRIGGER artifact_facts_delete_violation AFTER DELETE ON artifact_attachment_facts BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason) VALUES(OLD.session_id, 'artifact fact deleted');
END;
CREATE TRIGGER artifact_attachments_write_guard BEFORE INSERT ON artifact_attachments
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact attachments require the commit channel');
END;
CREATE TRIGGER artifact_attachments_update_violation AFTER UPDATE ON artifact_attachments
WHEN harness_journal_append_allowed() != 1 BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason) VALUES(OLD.session_id, 'artifact attachment updated');
END;
CREATE TRIGGER artifact_attachments_delete_violation AFTER DELETE ON artifact_attachments
WHEN harness_journal_append_allowed() != 1 BEGIN
  INSERT OR IGNORE INTO integrity_violations(session_id, reason) VALUES(OLD.session_id, 'artifact attachment deleted');
END;
CREATE TRIGGER artifact_retirements_insert_guard BEFORE INSERT ON artifact_retirements
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirements require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirements_update_guard BEFORE UPDATE ON artifact_retirements
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirements require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirements_delete_guard BEFORE DELETE ON artifact_retirements
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirements are append-only');
END;
CREATE TRIGGER artifact_retirement_facts_write_guard BEFORE INSERT ON artifact_retirement_facts
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement facts require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_facts_update_guard BEFORE UPDATE ON artifact_retirement_facts
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement facts require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_facts_delete_guard BEFORE DELETE ON artifact_retirement_facts
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement facts require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_nodes_write_guard BEFORE INSERT ON artifact_retirement_nodes
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement nodes require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_nodes_update_guard BEFORE UPDATE ON artifact_retirement_nodes
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement nodes require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_nodes_delete_guard BEFORE DELETE ON artifact_retirement_nodes
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement nodes require the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_meta_write_guard BEFORE INSERT ON artifact_retirement_meta
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement meta requires the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_meta_update_guard BEFORE UPDATE ON artifact_retirement_meta
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement meta requires the lifecycle channel');
END;
CREATE TRIGGER artifact_retirement_meta_delete_guard BEFORE DELETE ON artifact_retirement_meta
WHEN harness_journal_append_allowed() != 1 BEGIN
  SELECT RAISE(ABORT, 'artifact retirement meta requires the lifecycle channel');
END;
"""

_EXPECTED_COLUMNS = {
    "sessions": (
        "session_id",
        "version",
        "sequence",
        "state_json",
        "ledger_hash",
        "coverage_hash",
        "transcript_count",
        "summary_hash",
        "summary_count",
        "session_state_hash",
        "run_projection_hash",
        "run_count",
        "authenticated_root",
    ),
    "journal_entries": ("session_id", "sequence", "kind", "previous_hash", "record_hash"),
    "runs": (
        "session_id",
        "run_id",
        "submission_id",
        "profile_id",
        "status",
        "state_version",
        "run_index",
        "state_hash",
        "state_json",
    ),
    "run_projection_nodes": ("session_id", "level", "node_index", "digest", "item_count"),
    "events": ("event_id", "session_id", "sequence", "run_id", "event_json"),
    "transcript": (
        "session_id",
        "sequence",
        "run_id",
        "submission_id",
        "role",
        "content",
        "entry_type",
        "format_version",
        "entry_json",
    ),
    "transcript_coverage_blocks": (
        "session_id",
        "block_index",
        "first_sequence",
        "last_sequence",
        "item_count",
        "content_hash",
    ),
    "transcript_coverage_nodes": (
        "session_id",
        "level",
        "node_index",
        "digest",
        "item_count",
        "first_sequence",
        "last_sequence",
    ),
    "integrity_violations": ("session_id", "reason"),
    "summary_checkpoints": (
        "session_id",
        "summary_index",
        "covered_from_sequence",
        "covered_through_sequence",
        "policy_version",
        "profile_id",
        "checkpoint_json",
        "checkpoint_hash",
    ),
    "summary_nodes": ("session_id", "level", "node_index", "digest", "item_count"),
    "submissions": (
        "session_id",
        "submission_id",
        "run_id",
        "digest",
        "intent_json",
        "receipt_json",
        "acceptance_state_json",
        "acceptance_version",
    ),
    "artifact_attachment_facts": (
        "session_id",
        "fact_index",
        "run_id",
        "submission_id",
        "owner_id",
        "role",
        "action",
        "artifact_id",
        "fact_json",
    ),
    "artifact_attachments": (
        "session_id",
        "run_id",
        "submission_id",
        "owner_id",
        "role",
        "artifact_id",
        "fact_json",
    ),
    "artifact_retirements": (
        "artifact_id",
        "artifact_index",
        "generation",
        "state",
        "fact_count",
        "fact_root",
        "state_hash",
    ),
    "artifact_retirement_facts": (
        "artifact_id",
        "fact_index",
        "generation",
        "state",
        "previous_hash",
        "fact_hash",
        "fact_json",
    ),
    "artifact_retirement_nodes": ("level", "node_index", "digest", "item_count"),
    "artifact_retirement_allocation_damage": ("singleton",),
    "artifact_retirement_meta": (
        "singleton",
        "projection_root",
        "artifact_count",
        "authenticated_root",
    ),
}
_EXPECTED_INDEXES = {
    "runs_active_by_session": ("session_id", "status"),
    "events_by_session_sequence": ("session_id", "sequence"),
    "transcript_by_session_sequence": ("session_id", "sequence"),
    "transcript_run_input": (
        "session_id",
        "run_id",
        "submission_id",
        "role",
        "sequence",
    ),
    "transcript_coverage_blocks_by_sequence": (
        "session_id",
        "first_sequence",
        "last_sequence",
    ),
    "summary_context_lookup": (
        "session_id",
        "profile_id",
        "policy_version",
        "covered_through_sequence",
        "summary_index",
    ),
    "submissions_by_identity": ("session_id", "submission_id", "digest"),
    "artifact_attachments_by_artifact": ("artifact_id",),
    "artifact_facts_by_artifact": ("artifact_id",),
    "artifact_facts_by_run": ("session_id", "run_id", "fact_index"),
    "artifact_retirement_facts_by_artifact": ("artifact_id", "fact_index"),
}


def _typed_id_key(value: object, expected: type[_IdT]) -> str:
    try:
        if type(value) is not expected:
            raise ValueError
        raw = object.__getattribute__(value, "value")
        if type(raw) is not str:
            raise ValueError
        canonical = expected(raw)
        return object.__getattribute__(canonical, "value")
    except (AttributeError, TypeError, ValueError):
        raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None


class SQLiteRunJournal:
    """SQLite v6 Session-scoped CAS journal using one connection per operation."""

    def __init__(self, path: Path, timeout_seconds: float = 5.0) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("SQLite journal path must be an absolute Path")
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("SQLite journal timeout must be positive and finite")
        if path.exists() and path.is_dir():
            raise ValueError("SQLite journal path cannot be a directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.timeout_seconds = float(timeout_seconds)
        self._initialize()

    async def commit(self, batch: JournalCommit, expected_version: int) -> JournalSnapshot:
        if (
            type(batch) is not JournalCommit
            or type(expected_version) is not int
            or expected_version < 0
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        try:
            canonical = _canonical_copy(batch)
        except JournalError:
            raise
        except (MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        return await self._run(
            lambda cancelled: self._commit_sync(canonical, expected_version, cancelled)
        )

    async def load_head(self, session_id: SessionId) -> JournalHead:
        key = self._session_key(session_id)
        return await self._run(lambda _cancelled: self._load_head_sync(key))

    async def load_session(self, session_id: SessionId, after_sequence: int = 0) -> JournalSnapshot:
        key = self._session_key(session_id)
        if type(after_sequence) is not int or after_sequence < 0:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(lambda _cancelled: self._load_session_sync(key, after_sequence))

    async def load_context_summaries(
        self,
        session_id: SessionId,
        profile_id: ProfileId,
        policy_version: str,
        covered_through_sequence: int,
        limit: int,
        expected_version: int,
        expected_sequence: int,
    ) -> tuple[ConversationSummaryCheckpoint, ...]:
        key = self._session_key(session_id)
        if (
            type(profile_id) is not ProfileId
            or not _valid_policy_version(policy_version)
            or type(covered_through_sequence) is not int
            or covered_through_sequence < 0
            or type(limit) is not int
            or not 1 <= limit <= 128
            or type(expected_version) is not int
            or expected_version < 0
            or type(expected_sequence) is not int
            or expected_sequence < 0
            or covered_through_sequence > expected_sequence
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        profile_key = _typed_id_key(profile_id, ProfileId)
        return await self._run(
            lambda _cancelled: self._load_context_summaries_sync(
                key,
                profile_key,
                policy_version,
                covered_through_sequence,
                limit,
                expected_version,
                expected_sequence,
            )
        )

    async def read_page(
        self, session_id: SessionId, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        key = self._session_key(session_id)
        if (
            before_sequence is not None
            and (type(before_sequence) is not int or before_sequence <= 0)
            or type(limit) is not int
            or not 1 <= limit <= 1_024
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(lambda _cancelled: self._read_page_sync(key, before_sequence, limit))

    async def summarize_before(
        self,
        session_id: SessionId,
        before_sequence: int,
        excluded_from_sequence: int | None = None,
        excluded_through_sequence: int | None = None,
    ) -> ConversationCoverage:
        _validate_aggregate_request(
            session_id,
            before_sequence,
            excluded_from_sequence,
            excluded_through_sequence,
        )
        key = self._session_key(session_id)
        return await self._run(
            lambda _cancelled: self._summarize_before_sync(
                key,
                before_sequence,
                excluded_from_sequence,
                excluded_through_sequence,
            )
        )

    async def find_submission(
        self, session_id: SessionId, submission_id: SubmissionId
    ) -> SubmissionAcceptance | None:
        session_key = self._session_key(session_id)
        submission_key = _typed_id_key(submission_id, SubmissionId)
        return await self._run(
            lambda _cancelled: self._find_submission_sync(session_key, submission_key)
        )

    async def find_run_input(
        self,
        session_id: SessionId,
        run_id: RunId,
        submission_id: SubmissionId,
    ) -> TranscriptEntry | None:
        session_key = self._session_key(session_id)
        run_key = _typed_id_key(run_id, RunId)
        submission_key = _typed_id_key(submission_id, SubmissionId)
        return await self._run(
            lambda _cancelled: self._find_run_input_sync(session_key, run_key, submission_key)
        )

    async def find_run_state(
        self,
        session_id: SessionId,
        run_id: RunId,
        submission_id: SubmissionId,
    ) -> RunState | None:
        session_key = self._session_key(session_id)
        run_key = _typed_id_key(run_id, RunId)
        submission_key = _typed_id_key(submission_id, SubmissionId)
        return await self._run(
            lambda _cancelled: self._find_run_state_sync(session_key, run_key, submission_key)
        )

    async def find_run_state_by_id(self, session_id: SessionId, run_id: RunId) -> RunState | None:
        session_key = self._session_key(session_id)
        run_key = _typed_id_key(run_id, RunId)
        return await self._run(
            lambda _cancelled: self._find_run_state_by_id_sync(session_key, run_key)
        )

    async def load_artifact_attachments(
        self, session_id: SessionId, run_id: RunId
    ) -> tuple[ArtifactAttachmentFact, ...]:
        session_key = self._session_key(session_id)
        run_key = _typed_id_key(run_id, RunId)
        return await self._run(
            lambda _cancelled: self._load_artifact_attachments_sync(session_key, run_key)
        )

    async def artifact_has_active_attachments(self, artifact_id: str) -> bool:
        if type(artifact_id) is not str or not artifact_id:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(
            lambda _cancelled: self._artifact_has_active_attachments_sync(artifact_id)
        )

    async def reserve_artifact_retirement(
        self, artifact_id: str
    ) -> ArtifactRetirementReservation | None:
        if type(artifact_id) is not str or not artifact_id:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(
            lambda _cancelled: self._reserve_artifact_retirement_sync(artifact_id)
        )

    async def complete_artifact_retirement(
        self, reservation: ArtifactRetirementReservation
    ) -> ArtifactRetirementReservation:
        return await self._run(
            lambda _cancelled: self._finish_artifact_retirement_sync(
                reservation, ArtifactRetirementState.RETIRED
            )
        )

    async def abort_artifact_retirement(
        self, reservation: ArtifactRetirementReservation
    ) -> ArtifactRetirementReservation:
        return await self._run(
            lambda _cancelled: self._finish_artifact_retirement_sync(
                reservation, ArtifactRetirementState.ABORTED
            )
        )

    async def load_artifact_facts(
        self,
        session_id: SessionId,
        run_id: RunId,
        after_index: int,
        limit: int,
    ) -> tuple[ArtifactFactRecord, ...]:
        session_key = self._session_key(session_id)
        run_key = _typed_id_key(run_id, RunId)
        if (
            type(after_index) is not int
            or after_index < -1
            or type(limit) is not int
            or not 1 <= limit <= 1_024
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        return await self._run(
            lambda _cancelled: self._load_artifact_facts_sync(
                session_key, run_key, after_index, limit
            )
        )

    async def _run(self, operation: Callable[[Event], _T]) -> _T:
        cancelled = Event()
        task = asyncio.create_task(asyncio.to_thread(operation, cancelled))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise

    def _initialize(self) -> None:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            objects = self._application_objects(connection)
            if not objects and version == 0:
                selected = str(
                    connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                ).lower()
                if selected != "wal":
                    raise ValueError("SQLite journal requires WAL mode")
                connection.executescript(
                    "BEGIN IMMEDIATE;\n"
                    + _DDL
                    + f"\nPRAGMA user_version = {_SCHEMA_VERSION};\nCOMMIT;"
                )
            elif version != _SCHEMA_VERSION:
                raise ValueError("SQLite journal requires an exact schema v6 database")
            self._validate_schema(connection)
            if str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower() != "wal":
                raise ValueError("SQLite journal requires WAL mode")
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.timeout_seconds)
        connection.row_factory = sqlite3.Row
        connection.create_function("harness_journal_append_allowed", 0, lambda: 0)
        connection.execute(f"PRAGMA busy_timeout = {max(1, int(self.timeout_seconds * 1000))}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        selected = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if selected != "wal":
            connection.close()
            raise ValueError("SQLite journal requires WAL mode")
        return connection

    def _commit_sync(
        self, batch: JournalCommit, expected_version: int, cancelled: Event
    ) -> JournalSnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            state = batch.state
            session_key = state.session_id.value
            row, _current = _preflight_existing_ledger(connection, session_key)
            # This capability exists only on this private commit connection. It is
            # neither persistent SQL state nor shared with concurrent readers/writers.
            connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
            version = int(row["version"]) if row is not None else 0
            previous_sequence = int(row["sequence"]) if row is not None else 0
            ledger_hash = row["ledger_hash"] if row is not None else genesis_hash(session_key)
            coverage_hash = (
                row["coverage_hash"]
                if row is not None
                else coverage_empty_hash(session_key, _COVERAGE_DEPTH, 0)
            )
            transcript_count = int(row["transcript_count"]) if row is not None else 0
            summary_hash = (
                row["summary_hash"]
                if row is not None
                else summary_empty_hash(session_key, _SUMMARY_DEPTH, 0)
            )
            summary_count = int(row["summary_count"]) if row is not None else 0
            state_json = encode(state)
            session_state_hash = (
                row["session_state_hash"]
                if row is not None
                else state_record_hash(
                    session_key, state.run_id.value, state.submission_id.value, state_json
                )
            )
            run_projection_hash = (
                row["run_projection_hash"]
                if row is not None
                else run_empty_hash(session_key, _RUN_DEPTH, 0)
            )
            run_count = int(row["run_count"]) if row is not None else 0
            if not 0 <= summary_count < (1 << _SUMMARY_DEPTH):
                raise ValueError("SQLite summary capacity is exhausted")
            authenticated_root = authenticated_root_hash(
                session_key,
                ledger_hash,
                coverage_hash,
                transcript_count,
                summary_hash,
                summary_count,
                session_state_hash,
                run_projection_hash,
                run_count,
            )
            existing = connection.execute(
                "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
                "acceptance_state_json, acceptance_version FROM submissions "
                "WHERE session_id = ? AND submission_id = ?",
                (session_key, state.submission_id.value),
            ).fetchone()
            intent_json = encode(batch) if batch.submission is not None else None
            if existing is not None and intent_json is not None:
                prior_receipt, _, _ = _decode_submission_row(
                    existing, session_key, state.submission_id.value
                )
                if existing["intent_json"] != intent_json:
                    raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                connection.rollback()
                return prior_receipt
            if existing is not None and intent_json is None:
                prior_receipt, _, _ = _decode_submission_row(
                    existing, session_key, state.submission_id.value
                )
                if prior_receipt.state is None or prior_receipt.state.run_id != state.run_id:
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            if version != expected_version:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            _validate_members(batch, previous_sequence, expected_version)
            prior_run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, run_index, state_hash, state_json "
                "FROM runs WHERE session_id = ? AND run_id = ?",
                (session_key, state.run_id.value),
            ).fetchone()
            if prior_run_row is not None:
                prior = decode(prior_run_row["state_json"], RunState)
                _validate_run_row(prior_run_row, prior)
                if (
                    prior.submission_id != state.submission_id
                    or prior.profile_id != state.profile_id
                    or prior.session_id != state.session_id
                    or state.version < prior.version
                ):
                    raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            cursor = previous_sequence
            final_sequence = previous_sequence + len(batch.events) + len(batch.transcript)
            persisted_version = version + 1
            connection.execute(
                "INSERT INTO sessions(session_id, version, sequence, state_json, ledger_hash, "
                "coverage_hash, transcript_count, summary_hash, summary_count, session_state_hash, "
                "run_projection_hash, run_count, authenticated_root) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET version=excluded.version, sequence=excluded.sequence, "
                "state_json=excluded.state_json, session_state_hash=excluded.session_state_hash",
                (
                    session_key,
                    persisted_version,
                    final_sequence,
                    state_json,
                    ledger_hash,
                    coverage_hash,
                    transcript_count,
                    summary_hash,
                    summary_count,
                    session_state_hash,
                    run_projection_hash,
                    run_count,
                    authenticated_root,
                ),
            )
            run_index = int(prior_run_row["run_index"]) if prior_run_row is not None else run_count
            if prior_run_row is None:
                run_count += 1
            current_state_hash = state_record_hash(
                session_key, state.run_id.value, state.submission_id.value, state_json
            )
            connection.execute(
                "INSERT INTO runs(session_id, run_id, submission_id, profile_id, status, state_version, run_index, state_hash, state_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(session_id, run_id) DO UPDATE SET status=excluded.status, state_version=excluded.state_version, state_hash=excluded.state_hash, state_json=excluded.state_json",
                (
                    session_key,
                    state.run_id.value,
                    state.submission_id.value,
                    state.profile_id.value,
                    state.status.value,
                    state.version,
                    run_index,
                    current_state_hash,
                    state_json,
                ),
            )
            run_projection_hash = _set_run_projection_leaf(
                connection, session_key, run_index, current_state_hash
            )
            session_state_hash = current_state_hash
            fact_index = int(
                connection.execute(
                    "SELECT COALESCE(MAX(fact_index) + 1, 0) FROM artifact_attachment_facts WHERE session_id=?",
                    (session_key,),
                ).fetchone()[0]
            )
            for fact in batch.artifact_facts:
                target_run = connection.execute(
                    "SELECT submission_id FROM runs WHERE session_id=? AND run_id=?",
                    (session_key, fact.run_id.value),
                ).fetchone()
                if (
                    fact.session_id != state.session_id
                    or target_run is None
                    or target_run["submission_id"] != fact.submission_id.value
                ):
                    raise ValueError("Artifact attachment identity disagrees")
                fact_json = encode(fact)
                values = (
                    session_key,
                    fact.run_id.value,
                    fact.submission_id.value,
                    fact.owner_id,
                    fact.role.value,
                    fact.artifact.artifact_id,
                    fact_json,
                )
                connection.execute(
                    "INSERT INTO artifact_attachment_facts(session_id, fact_index, run_id, submission_id, owner_id, role, action, artifact_id, fact_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_key,
                        fact_index,
                        fact.run_id.value,
                        fact.submission_id.value,
                        fact.owner_id,
                        fact.role.value,
                        fact.action.value,
                        fact.artifact.artifact_id,
                        fact_json,
                    ),
                )
                fact_index += 1
                if fact.action is ArtifactAttachmentAction.ATTACHED:
                    retirement = _load_verified_retirement(connection, fact.artifact.artifact_id)
                    if (
                        retirement is not None
                        and retirement.state is ArtifactRetirementState.RETIRING
                    ):
                        raise ValueError("Artifact retirement blocks attachment")
                    if (
                        retirement is not None
                        and retirement.state is ArtifactRetirementState.RETIRED
                    ):
                        _append_retirement_transition(
                            connection,
                            fact.artifact.artifact_id,
                            retirement.generation + 1,
                            ArtifactRetirementState.AVAILABLE,
                        )
                    existing_attachment = connection.execute(
                        "SELECT fact_json FROM artifact_attachments WHERE session_id=? AND run_id=? AND owner_id=? AND role=? AND artifact_id=?",
                        (
                            session_key,
                            fact.run_id.value,
                            fact.owner_id,
                            fact.role.value,
                            fact.artifact.artifact_id,
                        ),
                    ).fetchone()
                    if existing_attachment is not None:
                        attached = decode(existing_attachment["fact_json"], ArtifactAttachmentFact)
                        if attached.artifact != fact.artifact:
                            raise ValueError("Artifact attachment descriptor disagrees")
                    connection.execute(
                        "INSERT INTO artifact_attachments(session_id, run_id, submission_id, owner_id, role, artifact_id, fact_json) VALUES (?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(session_id, run_id, owner_id, role, artifact_id) DO UPDATE SET fact_json=excluded.fact_json",
                        values,
                    )
                else:
                    row_to_detach = connection.execute(
                        "SELECT fact_json FROM artifact_attachments WHERE session_id=? AND run_id=? AND owner_id=? AND role=? AND artifact_id=?",
                        (
                            session_key,
                            fact.run_id.value,
                            fact.owner_id,
                            fact.role.value,
                            fact.artifact.artifact_id,
                        ),
                    ).fetchone()
                    if row_to_detach is None:
                        raise ValueError("Artifact attachment does not exist")
                    attached = decode(row_to_detach["fact_json"], ArtifactAttachmentFact)
                    if attached.artifact != fact.artifact:
                        raise ValueError("Artifact attachment descriptor disagrees")
                    connection.execute(
                        "DELETE FROM artifact_attachments WHERE session_id=? AND run_id=? AND owner_id=? AND role=? AND artifact_id=?",
                        (
                            session_key,
                            fact.run_id.value,
                            fact.owner_id,
                            fact.role.value,
                            fact.artifact.artifact_id,
                        ),
                    )
            events: list[JournalEvent] = []
            for event in batch.events:
                cursor += 1
                record = JournalEvent(cursor, event)
                record_json = encode(record)
                connection.execute(
                    "INSERT INTO events(event_id, session_id, sequence, run_id, event_json) VALUES (?, ?, ?, ?, ?)",
                    (event.event_id.value, session_key, cursor, event.run_id.value, record_json),
                )
                ledger_hash = _append_anchor(
                    connection, session_key, cursor, "event", record_json, ledger_hash
                )
                events.append(record)
            transcript: list[TranscriptEntry] = []
            for entry in batch.transcript:
                cursor += 1
                transcript_record = replace(entry, sequence=cursor)
                record_json = encode(transcript_record)
                connection.execute(
                    "INSERT INTO transcript(session_id, sequence, run_id, submission_id, role, "
                    "content, entry_type, format_version, entry_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_key,
                        cursor,
                        transcript_record.run_id.value,
                        transcript_record.submission_id.value,
                        transcript_record.role,
                        transcript_record.content,
                        transcript_record.entry_type.value,
                        transcript_record.format_version,
                        record_json,
                    ),
                )
                ledger_hash = _append_anchor(
                    connection, session_key, cursor, "transcript", record_json, ledger_hash
                )
                transcript.append(transcript_record)
            if transcript:
                first_block = transcript_count // _COVERAGE_BLOCK_SIZE
                transcript_count += len(transcript)
                last_block = (transcript_count - 1) // _COVERAGE_BLOCK_SIZE
                for block_index in range(first_block, last_block + 1):
                    _refresh_coverage_block(connection, session_key, block_index)
                coverage_hash = _coverage_root(connection, session_key)[0]
            authenticated_root = authenticated_root_hash(
                session_key,
                ledger_hash,
                coverage_hash,
                transcript_count,
                summary_hash,
                summary_count,
                session_state_hash,
                run_projection_hash,
                run_count,
            )
            connection.execute(
                "UPDATE sessions SET ledger_hash=?, coverage_hash=?, transcript_count=?, "
                "session_state_hash=?, run_projection_hash=?, run_count=?, authenticated_root=? WHERE session_id=?",
                (
                    ledger_hash,
                    coverage_hash,
                    transcript_count,
                    session_state_hash,
                    run_projection_hash,
                    run_count,
                    authenticated_root,
                    session_key,
                ),
            )
            receipt = JournalSnapshot(
                state.session_id,
                persisted_version,
                cursor,
                state,
                tuple(events),
                tuple(transcript),
                (batch.summary,) if batch.summary is not None else (),
                batch.submission,
                batch.artifact_facts,
            )
            if batch.summary is not None:
                summary = batch.summary
                summary_json = encode(summary)
                prior_summary = connection.execute(
                    "SELECT session_id, summary_index, covered_from_sequence, "
                    "covered_through_sequence, policy_version, profile_id, checkpoint_json, "
                    "checkpoint_hash FROM summary_checkpoints WHERE session_id = ? "
                    "AND covered_through_sequence = ? AND policy_version = ? AND profile_id = ?",
                    (
                        session_key,
                        summary.covered_through_sequence,
                        summary.policy_version,
                        summary.profile_id.value,
                    ),
                ).fetchone()
                if prior_summary is None:
                    checkpoint_hash = summary_record_hash(session_key, summary_count, summary_json)
                    connection.execute(
                        "INSERT INTO summary_checkpoints(session_id, summary_index, "
                        "covered_from_sequence, covered_through_sequence, policy_version, "
                        "profile_id, checkpoint_json, checkpoint_hash) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            session_key,
                            summary_count,
                            summary.covered_from_sequence,
                            summary.covered_through_sequence,
                            summary.policy_version,
                            summary.profile_id.value,
                            summary_json,
                            checkpoint_hash,
                        ),
                    )
                    summary_hash = _append_summary_leaf(
                        connection,
                        session_key,
                        summary_count,
                        checkpoint_hash,
                    )
                    summary_count += 1
                else:
                    persisted_summary = _decode_summary_row(prior_summary, session_key)
                    if (
                        persisted_summary != summary
                        or prior_summary["checkpoint_json"] != summary_json
                    ):
                        raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            authenticated_root = authenticated_root_hash(
                session_key,
                ledger_hash,
                coverage_hash,
                transcript_count,
                summary_hash,
                summary_count,
                session_state_hash,
                run_projection_hash,
                run_count,
            )
            connection.execute(
                "UPDATE sessions SET summary_hash=?, summary_count=?, authenticated_root=? "
                "WHERE session_id=?",
                (summary_hash, summary_count, authenticated_root, session_key),
            )
            if batch.submission is not None and intent_json is not None:
                connection.execute(
                    "INSERT INTO submissions(session_id, submission_id, run_id, digest, intent_json, receipt_json, acceptance_state_json, acceptance_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_key,
                        state.submission_id.value,
                        state.run_id.value,
                        batch.submission.digest,
                        intent_json,
                        encode(receipt),
                        state_json,
                        persisted_version,
                    ),
                )
            if cancelled.is_set():
                raise asyncio.CancelledError
            connection.commit()
            return decode(encode(receipt), JournalSnapshot)
        except asyncio.CancelledError:
            connection.rollback()
            raise
        except JournalError:
            connection.rollback()
            raise
        except sqlite3.IntegrityError:
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        except (
            sqlite3.Error,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OverflowError,
            MemoryError,
            RecursionError,
        ):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_head_sync(self, session_key: str) -> JournalHead:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _reject_integrity_violation(connection, session_key)
            row = connection.execute(
                "SELECT session_id, version, sequence, state_json, ledger_hash, coverage_hash, "
                "transcript_count, summary_hash, summary_count, session_state_hash, run_projection_hash, run_count, authenticated_root "
                "FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                return JournalHead(0, 0)
            state = decode(row["state_json"], RunState)
            version, sequence = _validate_session_row(row, session_key, state)
            run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, run_index, state_hash, state_json "
                "FROM runs WHERE session_id = ? AND run_id = ?",
                (session_key, state.run_id.value),
            ).fetchone()
            if run_row is None or decode(run_row["state_json"], RunState) != state:
                raise ValueError
            _validate_run_row(run_row, state)
            _verify_authenticated_root(row, session_key)
            if validate_hash(row["session_state_hash"]) != validate_hash(run_row["state_hash"]):
                raise ValueError("SQLite Session state hash disagrees with current Run")
            _verify_run_projection_path(
                connection,
                session_key,
                run_row["run_index"],
                run_row["state_hash"],
                row["run_projection_hash"],
            )
            _verify_ledger_tail(connection, session_key, sequence, row["ledger_hash"])
            return JournalHead(version, sequence, state)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_session_sync(self, session_key: str, after_sequence: int) -> JournalSnapshot:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row, state = _preflight_existing_ledger(connection, session_key)
            if row is None:
                connection.commit()
                return JournalSnapshot(SessionId(session_key), 0, 0)
            event_rows = tuple(
                connection.execute(
                    "SELECT event_id, session_id, sequence, run_id, event_json FROM events "
                    "WHERE session_id = ? AND sequence > ? ORDER BY sequence",
                    (session_key, after_sequence),
                )
            )
            events = tuple(_decode_event_row(item, session_key) for item in event_rows)
            transcript_rows = tuple(
                connection.execute(
                    "SELECT session_id, sequence, run_id, submission_id, role, content, "
                    "entry_type, format_version, entry_json "
                    "FROM transcript WHERE session_id = ? AND sequence > ? ORDER BY sequence",
                    (session_key, after_sequence),
                )
            )
            transcript = tuple(
                _decode_transcript_row(item, session_key) for item in transcript_rows
            )
            summary_rows = tuple(
                connection.execute(
                    "SELECT session_id, covered_from_sequence, covered_through_sequence, "
                    "policy_version, profile_id, checkpoint_json FROM summary_checkpoints "
                    "WHERE session_id = ? ORDER BY covered_through_sequence, policy_version, profile_id",
                    (session_key,),
                )
            )
            summaries = tuple(_decode_summary_row(item, session_key) for item in summary_rows)
            assert state is not None
            version, sequence = _validate_session_row(row, session_key, state)
            result = JournalSnapshot(
                SessionId(session_key),
                version,
                sequence,
                state,
                events,
                transcript,
                summaries,
            )
            connection.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_context_summaries_sync(
        self,
        session_key: str,
        profile_key: str,
        policy_version: str,
        covered_through_sequence: int,
        limit: int,
        expected_version: int,
        expected_sequence: int,
    ) -> tuple[ConversationSummaryCheckpoint, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _reject_integrity_violation(connection, session_key)
            row = connection.execute(
                "SELECT session_id, version, sequence, state_json, ledger_hash, coverage_hash, "
                "transcript_count, summary_hash, summary_count, session_state_hash, run_projection_hash, run_count, authenticated_root "
                "FROM sessions WHERE session_id = ?",
                (session_key,),
            ).fetchone()
            if row is None:
                if expected_version != 0 or expected_sequence != 0:
                    raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
                return ()
            state = decode(row["state_json"], RunState)
            version, sequence = _validate_session_row(row, session_key, state)
            _verify_authenticated_root(row, session_key)
            _verify_ledger_tail(connection, session_key, sequence, row["ledger_hash"])
            if version != expected_version or sequence != expected_sequence:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            rows = tuple(
                connection.execute(
                    "SELECT session_id, summary_index, covered_from_sequence, "
                    "covered_through_sequence, policy_version, profile_id, checkpoint_json, "
                    "checkpoint_hash FROM summary_checkpoints "
                    "WHERE session_id=? AND profile_id=? AND policy_version=? "
                    "AND covered_through_sequence<=? "
                    "ORDER BY covered_through_sequence DESC, summary_index DESC LIMIT ?",
                    (
                        session_key,
                        profile_key,
                        policy_version,
                        covered_through_sequence,
                        limit,
                    ),
                )
            )
            summaries = []
            expected_root = validate_hash(row["summary_hash"])
            for item in rows:
                summary = _decode_summary_row(item, session_key)
                summary_index = item["summary_index"]
                if type(summary_index) is not int or not 0 <= summary_index < (1 << _SUMMARY_DEPTH):
                    raise ValueError("SQLite summary index is invalid")
                digest = summary_record_hash(session_key, summary_index, item["checkpoint_json"])
                if validate_hash(item["checkpoint_hash"]) != digest:
                    raise ValueError("SQLite summary record hash disagrees")
                _verify_summary_path(
                    connection,
                    session_key,
                    summary_index,
                    (summary_leaf_hash(session_key, summary_index, digest), 1),
                    expected_root,
                )
                summaries.append(summary)
            return tuple(summaries)
        except JournalError:
            raise
        except (
            sqlite3.Error,
            ValueError,
            TypeError,
            KeyError,
            MemoryError,
            RecursionError,
        ):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _read_page_sync(
        self, session_key: str, before_sequence: int | None, limit: int
    ) -> ConversationPage:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _reject_integrity_violation(connection, session_key)
            if before_sequence is None:
                rows = tuple(
                    connection.execute(
                        "SELECT session_id, sequence, run_id, submission_id, role, content, "
                        "entry_type, format_version, entry_json FROM transcript WHERE session_id=? "
                        "ORDER BY sequence DESC LIMIT ?",
                        (session_key, (limit + 1) * MAX_TRANSCRIPT_ROWS_PER_INTERACTION),
                    )
                )
            else:
                rows = tuple(
                    connection.execute(
                        "SELECT session_id, sequence, run_id, submission_id, role, content, "
                        "entry_type, format_version, entry_json FROM transcript "
                        "WHERE session_id=? AND sequence<? "
                        "ORDER BY sequence DESC LIMIT ?",
                        (
                            session_key,
                            before_sequence,
                            (limit + 1) * MAX_TRANSCRIPT_ROWS_PER_INTERACTION,
                        ),
                    )
                )
            entries = tuple(_decode_transcript_row(row, session_key) for row in rows)
            for entry in entries:
                _verify_point_anchor(connection, session_key, entry.sequence)
            chronological = tuple(reversed(entries))
            raw_limit = (limit + 1) * MAX_TRANSCRIPT_ROWS_PER_INTERACTION
            if len(rows) == raw_limit:
                boundary = next(
                    (
                        index
                        for index, entry in enumerate(chronological)
                        if entry.entry_type is not TranscriptEntryType.TOOL_RESULT
                    ),
                    len(chronological),
                )
                chronological = chronological[boundary:]
            interactions = newest_transcript_interactions(chronological, limit + 1)
            selected = interactions[:limit]
            cursor = (
                selected[-1].sequence_from
                if selected and (len(interactions) > limit or len(rows) == raw_limit)
                else None
            )
            result = ConversationPage(
                selected,
                cursor,
            )
            connection.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _summarize_before_sync(
        self,
        session_key: str,
        before_sequence: int,
        excluded_from_sequence: int | None,
        excluded_through_sequence: int | None,
    ) -> ConversationCoverage:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _reject_integrity_violation(connection, session_key)
            session = connection.execute(
                "SELECT sequence, ledger_hash, coverage_hash, transcript_count, summary_hash, "
                "summary_count, session_state_hash, run_projection_hash, run_count, authenticated_root "
                "FROM sessions WHERE session_id=?",
                (session_key,),
            ).fetchone()
            if session is None:
                if connection.execute(
                    "SELECT EXISTS(SELECT 1 FROM transcript WHERE session_id=?)",
                    (session_key,),
                ).fetchone()[0]:
                    raise ValueError("SQLite transcript rows exist without a Session")
                coverage_root = coverage_empty_hash(session_key, _COVERAGE_DEPTH, 0)
            else:
                _verify_authenticated_root(session, session_key)
                _verify_ledger_tail(
                    connection, session_key, session["sequence"], session["ledger_hash"]
                )
                coverage = _coverage_root(connection, session_key)
                coverage_root = validate_hash(session["coverage_hash"])
                if coverage[0] != coverage_root or coverage[1] != session["transcript_count"]:
                    raise ValueError("SQLite coverage root disagrees with its Session")
            segments = []
            for first, last in _coverage_ranges(
                before_sequence,
                excluded_from_sequence,
                excluded_through_sequence,
            ):
                segments.append(
                    ConversationCoverageSegment(
                        first,
                        last,
                        _authenticated_transcript_count(
                            connection,
                            session_key,
                            first,
                            last,
                            coverage_root,
                        ),
                    )
                )
            result = ConversationCoverage(
                before_sequence,
                excluded_from_sequence,
                excluded_through_sequence,
                tuple(segments),
            )
            connection.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _find_submission_sync(
        self, session_key: str, submission_key: str
    ) -> SubmissionAcceptance | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _preflight_existing_ledger(connection, session_key)
            row = connection.execute(
                "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
                "acceptance_state_json, acceptance_version FROM submissions "
                "WHERE session_id = ? AND submission_id = ?",
                (session_key, submission_key),
            ).fetchone()
            if row is None:
                return None
            receipt, state, version = _decode_submission_row(row, session_key, submission_key)
            assert receipt.submission is not None
            return SubmissionAcceptance(receipt.submission, state, version)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _find_run_input_sync(
        self, session_key: str, run_key: str, submission_key: str
    ) -> TranscriptEntry | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            rows = tuple(
                connection.execute(
                    "SELECT session_id, sequence, run_id, submission_id, role, content, "
                    "entry_type, format_version, entry_json FROM transcript "
                    "WHERE session_id=? AND run_id=? "
                    "AND submission_id=? AND role='user' ORDER BY sequence LIMIT 2",
                    (session_key, run_key, submission_key),
                )
            )
            if len(rows) > 1:
                raise ValueError("SQLite Run has duplicate user inputs")
            if not rows:
                return None
            entry = _decode_transcript_row(rows[0], session_key)
            if (
                entry.run_id.value != run_key
                or entry.submission_id.value != submission_key
                or entry.role != "user"
            ):
                raise ValueError("SQLite Run input identity disagrees")
            run_row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, "
                "run_index, state_hash, state_json FROM runs WHERE session_id=? AND run_id=?",
                (session_key, run_key),
            ).fetchone()
            if run_row is None:
                raise ValueError("SQLite Run input has no Run projection")
            state = decode(run_row["state_json"], RunState)
            _validate_run_row(run_row, state)
            if state.submission_id.value != submission_key:
                raise ValueError("SQLite Run input disagrees with Run projection")
            _verify_point_anchor(connection, session_key, entry.sequence)
            return entry
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _find_run_state_sync(
        self, session_key: str, run_key: str, submission_key: str
    ) -> RunState | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, "
                "run_index, state_hash, state_json FROM runs WHERE session_id=? AND run_id=?",
                (session_key, run_key),
            ).fetchone()
            if row is None:
                return None
            state = decode(row["state_json"], RunState)
            _validate_run_row(row, state)
            if state.submission_id.value != submission_key:
                raise ValueError("SQLite Run identity disagrees")
            session_row = connection.execute(
                "SELECT session_id, version, sequence, state_json, ledger_hash, coverage_hash, "
                "transcript_count, summary_hash, summary_count, session_state_hash, "
                "run_projection_hash, run_count, authenticated_root FROM sessions WHERE session_id=?",
                (session_key,),
            ).fetchone()
            if session_row is None:
                raise ValueError("SQLite Run has no Session")
            _verify_authenticated_root(session_row, session_key)
            _verify_run_projection_path(
                connection,
                session_key,
                row["run_index"],
                row["state_hash"],
                session_row["run_projection_hash"],
            )
            submission_row = connection.execute(
                "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
                "acceptance_state_json, acceptance_version FROM submissions "
                "WHERE session_id=? AND submission_id=?",
                (session_key, submission_key),
            ).fetchone()
            if submission_row is None:
                raise ValueError("SQLite Run has no acceptance")
            receipt, _, _ = _decode_submission_row(submission_row, session_key, submission_key)
            if (
                receipt.submission is None
                or receipt.submission.run_id.value != run_key
                or len(receipt.transcript) != 1
            ):
                raise ValueError("SQLite Run acceptance identity disagrees")
            _verify_point_anchor(connection, session_key, receipt.transcript[0].sequence)
            return state
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _find_run_state_by_id_sync(self, session_key: str, run_key: str) -> RunState | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT session_id, run_id, submission_id, profile_id, status, state_version, "
                "run_index, state_hash, state_json FROM runs WHERE session_id=? AND run_id=?",
                (session_key, run_key),
            ).fetchone()
            if row is None:
                return None
            state = decode(row["state_json"], RunState)
            _validate_run_row(row, state)
            session_row = connection.execute(
                "SELECT session_id, version, sequence, state_json, ledger_hash, coverage_hash, "
                "transcript_count, summary_hash, summary_count, session_state_hash, "
                "run_projection_hash, run_count, authenticated_root FROM sessions WHERE session_id=?",
                (session_key,),
            ).fetchone()
            if session_row is None:
                raise ValueError("SQLite Run has no Session")
            _reject_integrity_violation(connection, session_key)
            _verify_authenticated_root(session_row, session_key)
            _verify_run_projection_path(
                connection,
                session_key,
                row["run_index"],
                row["state_hash"],
                session_row["run_projection_hash"],
            )
            return state
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_artifact_facts_sync(
        self,
        session_key: str,
        run_key: str,
        after_index: int,
        limit: int,
    ) -> tuple[ArtifactFactRecord, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _reject_integrity_violation(connection, session_key)
            records: list[ArtifactFactRecord] = []
            for row in connection.execute(
                "SELECT fact_index, session_id, run_id, submission_id, owner_id, role, action, "
                "artifact_id, fact_json FROM artifact_attachment_facts WHERE session_id=? "
                "AND run_id=? AND fact_index>? ORDER BY fact_index LIMIT ?",
                (session_key, run_key, after_index, limit),
            ):
                fact = decode(row["fact_json"], ArtifactAttachmentFact)
                if (
                    row["session_id"] != fact.session_id.value
                    or row["run_id"] != fact.run_id.value
                    or row["submission_id"] != fact.submission_id.value
                    or row["owner_id"] != fact.owner_id
                    or row["role"] != fact.role.value
                    or row["action"] != fact.action.value
                    or row["artifact_id"] != fact.artifact.artifact_id
                ):
                    raise ValueError("Artifact lifecycle fact row disagrees")
                records.append(ArtifactFactRecord(int(row["fact_index"]), fact))
            return tuple(records)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _load_artifact_attachments_sync(
        self, session_key: str, run_key: str
    ) -> tuple[ArtifactAttachmentFact, ...]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            _reject_integrity_violation(connection, session_key)
            result: list[ArtifactAttachmentFact] = []
            for row in connection.execute(
                "SELECT session_id, run_id, submission_id, owner_id, role, artifact_id, fact_json "
                "FROM artifact_attachments WHERE session_id=? AND run_id=? "
                "ORDER BY owner_id, role, artifact_id",
                (session_key, run_key),
            ):
                fact = decode(row["fact_json"], ArtifactAttachmentFact)
                if (
                    fact.action is not ArtifactAttachmentAction.ATTACHED
                    or row["session_id"] != fact.session_id.value
                    or row["run_id"] != fact.run_id.value
                    or row["submission_id"] != fact.submission_id.value
                    or row["owner_id"] != fact.owner_id
                    or row["role"] != fact.role.value
                    or row["artifact_id"] != fact.artifact.artifact_id
                ):
                    raise ValueError("Artifact attachment row disagrees")
                lifecycle = connection.execute(
                    "SELECT fact_json FROM artifact_attachment_facts WHERE session_id=? AND run_id=? "
                    "AND owner_id=? AND role=? AND artifact_id=? ORDER BY fact_index DESC LIMIT 1",
                    (
                        session_key,
                        run_key,
                        fact.owner_id,
                        fact.role.value,
                        fact.artifact.artifact_id,
                    ),
                ).fetchone()
                if (
                    lifecycle is None
                    or decode(lifecycle["fact_json"], ArtifactAttachmentFact) != fact
                ):
                    raise ValueError("Artifact active index disagrees with lifecycle facts")
                result.append(fact)
            expected: dict[tuple[str, str, str], ArtifactAttachmentFact] = {}
            fact_count = 0
            for fact_row in connection.execute(
                "SELECT session_id, run_id, submission_id, owner_id, role, action, artifact_id, fact_json "
                "FROM artifact_attachment_facts WHERE session_id=? AND run_id=? ORDER BY fact_index LIMIT ?",
                (session_key, run_key, _MAX_ARTIFACT_FACTS_PER_LOOKUP + 1),
            ):
                fact_count += 1
                if fact_count > _MAX_ARTIFACT_FACTS_PER_LOOKUP:
                    raise ValueError("Artifact lifecycle exceeds its lookup bound")
                lifecycle_fact = decode(fact_row["fact_json"], ArtifactAttachmentFact)
                if (
                    fact_row["session_id"] != lifecycle_fact.session_id.value
                    or fact_row["run_id"] != lifecycle_fact.run_id.value
                    or fact_row["submission_id"] != lifecycle_fact.submission_id.value
                    or fact_row["owner_id"] != lifecycle_fact.owner_id
                    or fact_row["role"] != lifecycle_fact.role.value
                    or fact_row["action"] != lifecycle_fact.action.value
                    or fact_row["artifact_id"] != lifecycle_fact.artifact.artifact_id
                ):
                    raise ValueError("Artifact lifecycle fact row disagrees")
                key = lifecycle_fact.key
                if lifecycle_fact.action is ArtifactAttachmentAction.ATTACHED:
                    prior = expected.get(key)
                    if prior is not None and prior.artifact != lifecycle_fact.artifact:
                        raise ValueError("Artifact lifecycle descriptor changed")
                    expected[key] = lifecycle_fact
                elif key not in expected or expected[key].artifact != lifecycle_fact.artifact:
                    raise ValueError("Artifact detach has no matching active fact")
                else:
                    del expected[key]
            actual = {fact.key: fact for fact in result}
            if actual != expected:
                raise ValueError("Artifact active index disagrees with lifecycle facts")
            return tuple(result)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _artifact_has_active_attachments_sync(self, artifact_id: str) -> bool:
        connection = self._connect()
        try:
            return _verify_artifact_active_index(connection, artifact_id)
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _reserve_artifact_retirement_sync(
        self, artifact_id: str
    ) -> ArtifactRetirementReservation | None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if (
                connection.execute("SELECT 1 FROM integrity_violations LIMIT 1").fetchone()
                is not None
            ):
                raise ValueError("SQLite journal has an integrity violation")
            if _verify_artifact_active_index(connection, artifact_id):
                connection.rollback()
                return None
            current = _load_verified_retirement(connection, artifact_id)
            if current is not None and current.state in (
                ArtifactRetirementState.RETIRING,
                ArtifactRetirementState.RETIRED,
            ):
                connection.rollback()
                return current
            generation = (
                1
                if current is None
                else current.generation
                if current.state is ArtifactRetirementState.AVAILABLE
                else current.generation + 1
            )
            connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
            reservation = _append_retirement_transition(
                connection, artifact_id, generation, ArtifactRetirementState.RETIRING
            )
            connection.commit()
            return reservation
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    def _finish_artifact_retirement_sync(
        self,
        reservation: ArtifactRetirementReservation,
        target: ArtifactRetirementState,
    ) -> ArtifactRetirementReservation:
        if type(reservation) is not ArtifactRetirementReservation:
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if _verify_artifact_active_index(connection, reservation.artifact_id):
                raise JournalError(RunErrorCode.INVARIANT_VIOLATION)
            current_reservation = _load_verified_retirement(connection, reservation.artifact_id)
            if (
                current_reservation is None
                or current_reservation.generation != reservation.generation
            ):
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            current = current_reservation.state
            if current is target:
                connection.rollback()
                return ArtifactRetirementReservation(
                    reservation.artifact_id, reservation.generation, target
                )
            if current is not ArtifactRetirementState.RETIRING:
                raise JournalError(RunErrorCode.JOURNAL_CONFLICT)
            connection.create_function("harness_journal_append_allowed", 0, lambda: 1)
            completed = _append_retirement_transition(
                connection, reservation.artifact_id, reservation.generation, target
            )
            connection.commit()
            return completed
        except JournalError:
            connection.rollback()
            raise
        except (sqlite3.Error, ValueError, TypeError, KeyError, MemoryError, RecursionError):
            connection.rollback()
            raise JournalError(RunErrorCode.INVARIANT_VIOLATION) from None
        finally:
            connection.close()

    @staticmethod
    def _session_key(session_id: SessionId) -> str:
        return _typed_id_key(session_id, SessionId)

    @staticmethod
    def _application_objects(connection: sqlite3.Connection) -> set[tuple[str, str]]:
        return {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT type, name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
            )
        }

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        if _schema_signature(connection) != _expected_schema_signature():
            raise ValueError("SQLite journal schema v6 definition has drifted")
        for table, expected in _EXPECTED_COLUMNS.items():
            columns = tuple(
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if columns != expected:
                raise ValueError("SQLite journal schema v6 has drifted")
        indexes = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            )
        }
        if not set(_EXPECTED_INDEXES) <= indexes:
            raise ValueError("SQLite journal schema v6 is missing required indexes")
        for index, expected_columns in _EXPECTED_INDEXES.items():
            actual_columns = tuple(
                str(row[2]) for row in connection.execute(f"PRAGMA index_info({index})")
            )
            if actual_columns != expected_columns:
                raise ValueError("SQLite journal schema v6 index has drifted")
        for table in (
            "journal_entries",
            "runs",
            "run_projection_nodes",
            "events",
            "transcript",
            "transcript_coverage_blocks",
            "transcript_coverage_nodes",
            "integrity_violations",
            "summary_checkpoints",
            "summary_nodes",
            "submissions",
            "artifact_attachment_facts",
            "artifact_attachments",
        ):
            foreign = tuple(connection.execute(f"PRAGMA foreign_key_list({table})"))
            if not any(
                str(row[2]) == "sessions"
                and str(row[3]) == "session_id"
                and str(row[4]) == "session_id"
                and str(row[6]).upper() == "CASCADE"
                for row in foreign
            ):
                raise ValueError("SQLite journal schema v6 is missing a Session foreign key")


def _schema_signature(connection: sqlite3.Connection) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema "
            "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY type, name"
        )
    )


def _expected_schema_signature() -> tuple[tuple[str, str, str, str], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_DDL)
        return _schema_signature(connection)
    finally:
        connection.close()


_CHAIN_ROWS = (
    "SELECT j.session_id, j.sequence, j.kind, j.previous_hash, j.record_hash, "
    "e.event_id, e.event_json, t.entry_json, t.submission_id, t.role, t.content, "
    "t.entry_type, t.format_version, "
    "CASE j.kind WHEN 'event' THEN e.run_id ELSE t.run_id END AS run_id "
    "FROM journal_entries AS j "
    "LEFT JOIN events AS e ON e.session_id=j.session_id AND e.sequence=j.sequence "
    "LEFT JOIN transcript AS t ON t.session_id=j.session_id AND t.sequence=j.sequence"
)


def _append_anchor(
    connection: sqlite3.Connection,
    session_key: str,
    sequence: int,
    kind: str,
    canonical_json: str,
    previous_hash: str,
) -> str:
    digest = record_hash(session_key, sequence, kind, canonical_json, previous_hash)
    connection.execute(
        "INSERT INTO journal_entries(session_id, sequence, kind, previous_hash, record_hash) "
        "VALUES (?, ?, ?, ?, ?)",
        (session_key, sequence, kind, previous_hash, digest),
    )
    return digest


def _verify_anchor(anchor: sqlite3.Row, session_key: str, previous_hash: str) -> str:
    if validate_hash(anchor["previous_hash"]) != previous_hash:
        raise ValueError("SQLite ledger predecessor hash disagrees")
    if anchor["kind"] == "event" and anchor["entry_json"] is None:
        _decode_event_row(anchor, session_key)
        canonical_json = anchor["event_json"]
    elif anchor["kind"] == "transcript" and anchor["event_json"] is None:
        _decode_transcript_row(anchor, session_key)
        canonical_json = anchor["entry_json"]
    else:
        raise ValueError("SQLite ledger anchor kind disagrees with its record")
    # The decoders reject non-canonical JSON before the stored text is hashed.
    digest = record_hash(
        session_key, anchor["sequence"], anchor["kind"], canonical_json, previous_hash
    )
    if validate_hash(anchor["record_hash"]) != digest:
        raise ValueError("SQLite ledger record hash disagrees with its payload")
    return digest


def _node_value(row: sqlite3.Row | None, session_key: str, level: int, node_index: int):
    if row is None:
        return coverage_empty_hash(session_key, level, node_index), 0, 0, 0
    digest = validate_hash(row["digest"])
    count = row["item_count"]
    first = row["first_sequence"]
    last = row["last_sequence"]
    if (
        type(count) is not int
        or type(first) is not int
        or type(last) is not int
        or count < 0
        or (count == 0 and (first != 0 or last != 0))
        or (count > 0 and not 1 <= first <= last)
    ):
        raise ValueError("SQLite coverage node is invalid")
    return digest, count, first, last


def _read_coverage_node(
    connection: sqlite3.Connection, session_key: str, level: int, node_index: int
):
    row = connection.execute(
        "SELECT digest, item_count, first_sequence, last_sequence "
        "FROM transcript_coverage_nodes WHERE session_id=? AND level=? AND node_index=?",
        (session_key, level, node_index),
    ).fetchone()
    return _node_value(row, session_key, level, node_index)


def _write_coverage_node(
    connection: sqlite3.Connection,
    session_key: str,
    level: int,
    node_index: int,
    value: tuple[str, int, int, int],
) -> None:
    connection.execute(
        "INSERT INTO transcript_coverage_nodes(session_id, level, node_index, digest, "
        "item_count, first_sequence, last_sequence) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(session_id, level, node_index) DO UPDATE SET digest=excluded.digest, "
        "item_count=excluded.item_count, first_sequence=excluded.first_sequence, "
        "last_sequence=excluded.last_sequence",
        (session_key, level, node_index, *value),
    )


def _refresh_coverage_block(
    connection: sqlite3.Connection, session_key: str, block_index: int
) -> None:
    first_row = connection.execute(
        "SELECT sequence FROM transcript WHERE session_id=? ORDER BY sequence LIMIT 1 OFFSET ?",
        (session_key, block_index * _COVERAGE_BLOCK_SIZE),
    ).fetchone()
    if first_row is None or type(first_row["sequence"]) is not int:
        raise ValueError("SQLite coverage block has no first transcript row")
    rows = tuple(
        connection.execute(
            "SELECT t.session_id, t.sequence, t.run_id, t.submission_id, t.role, t.content, "
            "t.entry_type, t.format_version, t.entry_json, j.record_hash "
            "FROM transcript AS t JOIN journal_entries AS j "
            "ON j.session_id=t.session_id AND j.sequence=t.sequence AND j.kind='transcript' "
            "WHERE t.session_id=? AND t.sequence>=? ORDER BY t.sequence LIMIT ?",
            (session_key, first_row["sequence"], _COVERAGE_BLOCK_SIZE),
        )
    )
    records = []
    for row in rows:
        _decode_transcript_row(row, session_key)
        records.append((row["sequence"], validate_hash(row["record_hash"]), row["entry_json"]))
    if not records:
        raise ValueError("SQLite coverage block is empty")
    content_hash = coverage_block_content_hash(session_key, block_index, tuple(records))
    first = records[0][0]
    last = records[-1][0]
    count = len(records)
    connection.execute(
        "INSERT INTO transcript_coverage_blocks(session_id, block_index, first_sequence, "
        "last_sequence, item_count, content_hash) VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(session_id, block_index) DO UPDATE SET first_sequence=excluded.first_sequence, "
        "last_sequence=excluded.last_sequence, item_count=excluded.item_count, "
        "content_hash=excluded.content_hash",
        (session_key, block_index, first, last, count, content_hash),
    )
    index = block_index
    value = (
        coverage_leaf_hash(session_key, block_index, first, last, count, content_hash),
        count,
        first,
        last,
    )
    _write_coverage_node(connection, session_key, 0, index, value)
    for level in range(1, _COVERAGE_DEPTH + 1):
        sibling = _read_coverage_node(connection, session_key, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        parent_count = left[1] + right[1]
        parent_first = left[2] if left[1] else right[2]
        parent_last = right[3] if right[1] else left[3]
        value = (
            coverage_parent_hash(session_key, level, parent_index, left, right),
            parent_count,
            parent_first,
            parent_last,
        )
        _write_coverage_node(connection, session_key, level, parent_index, value)
        index = parent_index


def _coverage_root(connection: sqlite3.Connection, session_key: str):
    return _read_coverage_node(connection, session_key, _COVERAGE_DEPTH, 0)


def _verify_artifact_active_index(connection: sqlite3.Connection, artifact_id: str) -> bool:
    expected: dict[tuple[str, str, str, str, str], ArtifactAttachmentFact] = {}
    fact_count = 0
    for row in connection.execute(
        "SELECT session_id, run_id, submission_id, owner_id, role, action, artifact_id, fact_json "
        "FROM artifact_attachment_facts WHERE artifact_id=? ORDER BY session_id, fact_index LIMIT ?",
        (artifact_id, _MAX_ARTIFACT_FACTS_PER_LOOKUP + 1),
    ):
        fact_count += 1
        if fact_count > _MAX_ARTIFACT_FACTS_PER_LOOKUP:
            raise ValueError("Artifact lifecycle exceeds its lookup bound")
        _reject_integrity_violation(connection, str(row["session_id"]))
        fact = decode(row["fact_json"], ArtifactAttachmentFact)
        if (
            row["session_id"] != fact.session_id.value
            or row["run_id"] != fact.run_id.value
            or row["submission_id"] != fact.submission_id.value
            or row["owner_id"] != fact.owner_id
            or row["role"] != fact.role.value
            or row["action"] != fact.action.value
            or row["artifact_id"] != fact.artifact.artifact_id
        ):
            raise ValueError("Artifact lifecycle fact row disagrees")
        key = (
            fact.session_id.value,
            fact.run_id.value,
            fact.owner_id,
            fact.role.value,
            fact.artifact.artifact_id,
        )
        if fact.action is ArtifactAttachmentAction.ATTACHED:
            prior = expected.get(key)
            if prior is not None and prior.artifact != fact.artifact:
                raise ValueError("Artifact lifecycle descriptor changed")
            expected[key] = fact
        elif key not in expected or expected[key].artifact != fact.artifact:
            raise ValueError("Artifact detach has no matching active fact")
        else:
            del expected[key]
    actual: dict[tuple[str, str, str, str, str], ArtifactAttachmentFact] = {}
    for row in connection.execute(
        "SELECT session_id, run_id, submission_id, owner_id, role, artifact_id, fact_json "
        "FROM artifact_attachments WHERE artifact_id=?",
        (artifact_id,),
    ):
        _reject_integrity_violation(connection, str(row["session_id"]))
        fact = decode(row["fact_json"], ArtifactAttachmentFact)
        key = (
            str(row["session_id"]),
            str(row["run_id"]),
            str(row["owner_id"]),
            str(row["role"]),
            str(row["artifact_id"]),
        )
        if (
            fact.action is not ArtifactAttachmentAction.ATTACHED
            or fact.session_id.value != key[0]
            or fact.run_id.value != key[1]
            or fact.submission_id.value != row["submission_id"]
            or fact.owner_id != key[2]
            or fact.role.value != key[3]
            or fact.artifact.artifact_id != key[4]
        ):
            raise ValueError("Artifact active index row disagrees")
        actual[key] = fact
    if actual != expected:
        raise ValueError("Artifact active index disagrees with lifecycle facts")
    return bool(expected)


def _retirement_node_value(row: sqlite3.Row | None, level: int, node_index: int) -> tuple[str, int]:
    if not 0 <= level <= _RETIREMENT_DEPTH or not 0 <= node_index < (
        1 << (_RETIREMENT_DEPTH - level)
    ):
        raise ValueError("SQLite retirement projection node is out of range")
    empty_digest = retirement_empty_hash(level, node_index)
    if row is None:
        return empty_digest, 0
    digest = validate_hash(row["digest"])
    count = row["item_count"]
    if (
        type(count) is not int
        or not 0 <= count <= (1 << level)
        or (count == 0) != (digest == empty_digest)
    ):
        raise ValueError("SQLite retirement projection node is invalid")
    return digest, count


def _read_retirement_node(
    connection: sqlite3.Connection, level: int, node_index: int
) -> tuple[str, int]:
    row = connection.execute(
        "SELECT digest, item_count FROM artifact_retirement_nodes WHERE level=? AND node_index=?",
        (level, node_index),
    ).fetchone()
    return _retirement_node_value(row, level, node_index)


def _write_retirement_node(
    connection: sqlite3.Connection,
    level: int,
    node_index: int,
    value: tuple[str, int],
) -> None:
    connection.execute(
        "INSERT INTO artifact_retirement_nodes(level, node_index, digest, item_count) "
        "VALUES (?, ?, ?, ?) ON CONFLICT(level, node_index) DO UPDATE "
        "SET digest=excluded.digest, item_count=excluded.item_count",
        (level, node_index, *value),
    )


def _set_retirement_leaf(
    connection: sqlite3.Connection, artifact_index: int, state_digest: str
) -> str:
    if not 0 <= artifact_index < (1 << _RETIREMENT_DEPTH):
        raise ValueError("SQLite retirement projection index exceeds capacity")
    index = artifact_index
    value = (retirement_leaf_hash(artifact_index, state_digest), 1)
    _write_retirement_node(connection, 0, index, value)
    for level in range(1, _RETIREMENT_DEPTH + 1):
        sibling = _read_retirement_node(connection, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        value = (
            retirement_parent_hash(level, parent_index, left, right),
            left[1] + right[1],
        )
        _write_retirement_node(connection, level, parent_index, value)
        index = parent_index
    return value[0]


def _verify_retirement_meta(connection: sqlite3.Connection) -> tuple[str, int]:
    # Current rows are append-only with immutable indices. These unconditional SQL
    # triggers attest that no deletion/index rewrite broke their dense prefix. The
    # singleton witness cannot be cleared even by the lifecycle write capability;
    # schema authentication protects its triggers. This primary-key lookup is O(1).
    if (
        connection.execute(
            "SELECT 1 FROM artifact_retirement_allocation_damage WHERE singleton=1"
        ).fetchone()
        is not None
    ):
        raise ValueError("SQLite retirement allocation has permanent damage")
    row = connection.execute(
        "SELECT projection_root, artifact_count, authenticated_root "
        "FROM artifact_retirement_meta WHERE singleton=1"
    ).fetchone()
    if row is None:
        if any(
            connection.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None
            for table in (
                "artifact_retirements",
                "artifact_retirement_facts",
                "artifact_retirement_nodes",
            )
        ):
            raise ValueError("SQLite retirement authenticated root is missing")
        return retirement_empty_hash(_RETIREMENT_DEPTH, 0), 0
    projection_root = validate_hash(row["projection_root"])
    artifact_count = row["artifact_count"]
    if type(artifact_count) is not int or not 0 <= artifact_count <= (1 << _RETIREMENT_DEPTH):
        raise ValueError("SQLite retirement artifact count is invalid")
    if row["authenticated_root"] != retirement_authenticated_root_hash(
        projection_root, artifact_count
    ):
        raise ValueError("SQLite retirement authenticated root disagrees")
    root = _read_retirement_node(connection, _RETIREMENT_DEPTH, 0)
    if root != (projection_root, artifact_count):
        raise ValueError("SQLite retirement projection root disagrees")
    return projection_root, artifact_count


def _verify_retirement_path(
    connection: sqlite3.Connection,
    artifact_index: int,
    state_digest: str | None,
    expected_root: str,
    artifact_count: int,
) -> None:
    index = artifact_index
    value = (
        (retirement_empty_hash(0, index), 0)
        if state_digest is None
        else (retirement_leaf_hash(artifact_index, state_digest), 1)
    )
    if _read_retirement_node(connection, 0, index) != value:
        raise ValueError("SQLite retirement projection leaf disagrees")
    for level in range(1, _RETIREMENT_DEPTH + 1):
        sibling = _read_retirement_node(connection, level - 1, index ^ 1)
        size = 1 << (level - 1)
        if sibling[1] != min(size, max(0, artifact_count - (index ^ 1) * size)):
            raise ValueError("SQLite retirement projection subtree count disagrees")
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        count = left[1] + right[1]
        value = (
            retirement_empty_hash(level, parent_index)
            if count == 0
            else retirement_parent_hash(level, parent_index, left, right),
            count,
        )
        if _read_retirement_node(connection, level, parent_index) != value:
            raise ValueError("SQLite retirement projection Merkle path is invalid")
        index = parent_index
    if value != (validate_hash(expected_root), artifact_count):
        raise ValueError("SQLite retirement projection path disagrees with root")


def _verify_retirement_allocation(
    connection: sqlite3.Connection, projection_root: str, artifact_count: int
) -> None:
    if artifact_count >= (1 << _RETIREMENT_DEPTH):
        raise ValueError("SQLite retirement projection index exceeds capacity")
    # Allocation is a dense prefix. Indexed tail probes reject unreachable nodes,
    # including descendants hidden below an absent empty-subtree root, without a scan.
    last = connection.execute(
        "SELECT artifact_index FROM artifact_retirements ORDER BY artifact_index DESC LIMIT 1"
    ).fetchone()
    if (None if last is None else last["artifact_index"]) != (
        None if artifact_count == 0 else artifact_count - 1
    ):
        raise ValueError("SQLite retirement allocation frontier disagrees")
    for level in range(_RETIREMENT_DEPTH + 1):
        size = 1 << level
        first_empty = (artifact_count + size - 1) // size
        if (
            connection.execute(
                "SELECT 1 FROM artifact_retirement_nodes WHERE level=? AND node_index>=? LIMIT 1",
                (level, first_empty),
            ).fetchone()
            is not None
        ):
            raise ValueError("SQLite retirement projection has an orphan node")
    _verify_retirement_path(connection, artifact_count, None, projection_root, artifact_count)


def _load_verified_retirement(
    connection: sqlite3.Connection, artifact_id: str
) -> ArtifactRetirementReservation | None:
    projection_root, artifact_count = _verify_retirement_meta(connection)
    row = connection.execute(
        "SELECT artifact_index, generation, state, fact_count, fact_root, state_hash "
        "FROM artifact_retirements WHERE artifact_id=?",
        (artifact_id,),
    ).fetchone()
    if row is None:
        if (
            connection.execute(
                "SELECT 1 FROM artifact_retirement_facts WHERE artifact_id=? LIMIT 1",
                (artifact_id,),
            ).fetchone()
            is not None
        ):
            raise ValueError("SQLite retirement current projection is missing")
        return None
    artifact_index = row["artifact_index"]
    generation = row["generation"]
    fact_count = row["fact_count"]
    state = ArtifactRetirementState(str(row["state"]))
    if (
        type(artifact_index) is not int
        or not 0 <= artifact_index < artifact_count
        or type(generation) is not int
        or generation < 1
        or type(fact_count) is not int
        or not 1 <= fact_count <= _MAX_ARTIFACT_FACTS_PER_LOOKUP
    ):
        raise ValueError("SQLite retirement current projection is invalid")
    fact_root = validate_hash(row["fact_root"])
    state_digest = retirement_state_hash(
        artifact_id, artifact_index, generation, state.value, fact_count, fact_root
    )
    if row["state_hash"] != state_digest:
        raise ValueError("SQLite retirement state hash disagrees")
    _verify_retirement_path(
        connection, artifact_index, state_digest, projection_root, artifact_count
    )
    previous = retirement_genesis_hash(artifact_id)
    facts = connection.execute(
        "SELECT fact_index, generation, state, previous_hash, fact_hash, fact_json "
        "FROM artifact_retirement_facts WHERE artifact_id=? ORDER BY fact_index LIMIT ?",
        (artifact_id, _MAX_ARTIFACT_FACTS_PER_LOOKUP + 1),
    ).fetchall()
    if len(facts) != fact_count:
        raise ValueError("SQLite retirement fact count disagrees")
    last: ArtifactRetirementReservation | None = None
    for expected_index, fact in enumerate(facts):
        decoded = decode(fact["fact_json"], ArtifactRetirementReservation)
        if (
            fact["fact_index"] != expected_index
            or decoded.artifact_id != artifact_id
            or fact["generation"] != decoded.generation
            or fact["state"] != decoded.state.value
            or fact["previous_hash"] != previous
        ):
            raise ValueError("SQLite retirement fact disagrees")
        expected_hash = retirement_fact_hash(
            artifact_id,
            expected_index,
            decoded.generation,
            decoded.state.value,
            previous,
        )
        if fact["fact_hash"] != expected_hash:
            raise ValueError("SQLite retirement fact hash disagrees")
        previous = expected_hash
        last = decoded
    if previous != fact_root or last != ArtifactRetirementReservation(
        artifact_id, generation, state
    ):
        raise ValueError("SQLite retirement fact root disagrees")
    return last


def _append_retirement_transition(
    connection: sqlite3.Connection,
    artifact_id: str,
    generation: int,
    state: ArtifactRetirementState,
) -> ArtifactRetirementReservation:
    projection_root, artifact_count = _verify_retirement_meta(connection)
    current = connection.execute(
        "SELECT artifact_index, fact_count, fact_root FROM artifact_retirements "
        "WHERE artifact_id=?",
        (artifact_id,),
    ).fetchone()
    if current is None:
        _verify_retirement_allocation(connection, projection_root, artifact_count)
        artifact_index = artifact_count
        fact_index = 0
        previous = retirement_genesis_hash(artifact_id)
        next_count = artifact_count + 1
    else:
        artifact_index = int(current["artifact_index"])
        fact_index = int(current["fact_count"])
        previous = validate_hash(current["fact_root"])
        next_count = artifact_count
    if fact_index >= _MAX_ARTIFACT_FACTS_PER_LOOKUP:
        raise ValueError("Artifact retirement exceeds its transition bound")
    reservation = ArtifactRetirementReservation(artifact_id, generation, state)
    fact_digest = retirement_fact_hash(artifact_id, fact_index, generation, state.value, previous)
    connection.execute(
        "INSERT INTO artifact_retirement_facts(artifact_id, fact_index, generation, state, "
        "previous_hash, fact_hash, fact_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            artifact_id,
            fact_index,
            generation,
            state.value,
            previous,
            fact_digest,
            encode(reservation),
        ),
    )
    fact_count = fact_index + 1
    state_digest = retirement_state_hash(
        artifact_id, artifact_index, generation, state.value, fact_count, fact_digest
    )
    connection.execute(
        "INSERT INTO artifact_retirements(artifact_id, artifact_index, generation, state, "
        "fact_count, fact_root, state_hash) VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(artifact_id) DO UPDATE SET generation=excluded.generation, "
        "state=excluded.state, fact_count=excluded.fact_count, fact_root=excluded.fact_root, "
        "state_hash=excluded.state_hash",
        (
            artifact_id,
            artifact_index,
            generation,
            state.value,
            fact_count,
            fact_digest,
            state_digest,
        ),
    )
    projection_root = _set_retirement_leaf(connection, artifact_index, state_digest)
    authenticated_root = retirement_authenticated_root_hash(projection_root, next_count)
    connection.execute(
        "INSERT INTO artifact_retirement_meta(singleton, projection_root, artifact_count, "
        "authenticated_root) VALUES (1, ?, ?, ?) ON CONFLICT(singleton) DO UPDATE SET "
        "projection_root=excluded.projection_root, artifact_count=excluded.artifact_count, "
        "authenticated_root=excluded.authenticated_root",
        (projection_root, next_count, authenticated_root),
    )
    return reservation


def _run_node_value(
    row: sqlite3.Row | None, session_key: str, level: int, node_index: int
) -> tuple[str, int]:
    if row is None:
        return run_empty_hash(session_key, level, node_index), 0
    digest = validate_hash(row["digest"])
    count = row["item_count"]
    if type(count) is not int or count < 0:
        raise ValueError("SQLite Run projection node is invalid")
    return digest, count


def _read_run_node(
    connection: sqlite3.Connection, session_key: str, level: int, node_index: int
) -> tuple[str, int]:
    row = connection.execute(
        "SELECT digest, item_count FROM run_projection_nodes "
        "WHERE session_id=? AND level=? AND node_index=?",
        (session_key, level, node_index),
    ).fetchone()
    return _run_node_value(row, session_key, level, node_index)


def _write_run_node(
    connection: sqlite3.Connection,
    session_key: str,
    level: int,
    node_index: int,
    value: tuple[str, int],
) -> None:
    connection.execute(
        "INSERT INTO run_projection_nodes(session_id, level, node_index, digest, item_count) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id, level, node_index) DO UPDATE "
        "SET digest=excluded.digest, item_count=excluded.item_count",
        (session_key, level, node_index, *value),
    )


def _set_run_projection_leaf(
    connection: sqlite3.Connection, session_key: str, run_index: int, state_hash: str
) -> str:
    if not 0 <= run_index < (1 << _RUN_DEPTH):
        raise ValueError("SQLite Run projection index exceeds capacity")
    index = run_index
    value = (run_leaf_hash(session_key, run_index, state_hash), 1)
    _write_run_node(connection, session_key, 0, index, value)
    for level in range(1, _RUN_DEPTH + 1):
        sibling = _read_run_node(connection, session_key, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        value = (
            run_parent_hash(session_key, level, parent_index, left, right),
            left[1] + right[1],
        )
        _write_run_node(connection, session_key, level, parent_index, value)
        index = parent_index
    return value[0]


def _verify_run_projection_path(
    connection: sqlite3.Connection,
    session_key: str,
    run_index: int,
    state_hash: str,
    expected_root: str,
) -> None:
    index = run_index
    value = (run_leaf_hash(session_key, run_index, state_hash), 1)
    if _read_run_node(connection, session_key, 0, index) != value:
        raise ValueError("SQLite Run projection leaf disagrees")
    for level in range(1, _RUN_DEPTH + 1):
        sibling = _read_run_node(connection, session_key, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        value = (
            run_parent_hash(session_key, level, parent_index, left, right),
            left[1] + right[1],
        )
        if _read_run_node(connection, session_key, level, parent_index) != value:
            raise ValueError("SQLite Run projection Merkle path is invalid")
        index = parent_index
    if value[0] != validate_hash(expected_root):
        raise ValueError("SQLite Run projection path disagrees with Session root")


def _summary_node_value(
    row: sqlite3.Row | None, session_key: str, level: int, node_index: int
) -> tuple[str, int]:
    if row is None:
        return summary_empty_hash(session_key, level, node_index), 0
    digest = validate_hash(row["digest"])
    count = row["item_count"]
    if type(count) is not int or count < 0:
        raise ValueError("SQLite summary node count is invalid")
    return digest, count


def _read_summary_node(
    connection: sqlite3.Connection, session_key: str, level: int, node_index: int
) -> tuple[str, int]:
    row = connection.execute(
        "SELECT digest, item_count FROM summary_nodes "
        "WHERE session_id=? AND level=? AND node_index=?",
        (session_key, level, node_index),
    ).fetchone()
    return _summary_node_value(row, session_key, level, node_index)


def _write_summary_node(
    connection: sqlite3.Connection,
    session_key: str,
    level: int,
    node_index: int,
    value: tuple[str, int],
) -> None:
    connection.execute(
        "INSERT INTO summary_nodes(session_id, level, node_index, digest, item_count) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id, level, node_index) DO UPDATE "
        "SET digest=excluded.digest, item_count=excluded.item_count",
        (session_key, level, node_index, *value),
    )


def _append_summary_leaf(
    connection: sqlite3.Connection,
    session_key: str,
    summary_index: int,
    checkpoint_hash: str,
) -> str:
    if type(summary_index) is not int or not 0 <= summary_index < (1 << _SUMMARY_DEPTH):
        raise ValueError("SQLite summary index exceeds its authenticated tree")
    index = summary_index
    value = (summary_leaf_hash(session_key, summary_index, checkpoint_hash), 1)
    _write_summary_node(connection, session_key, 0, index, value)
    for level in range(1, _SUMMARY_DEPTH + 1):
        sibling = _read_summary_node(connection, session_key, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        value = (
            summary_parent_hash(session_key, level, parent_index, left, right),
            left[1] + right[1],
        )
        _write_summary_node(connection, session_key, level, parent_index, value)
        index = parent_index
    return value[0]


def _verify_summary_path(
    connection: sqlite3.Connection,
    session_key: str,
    summary_index: int,
    leaf: tuple[str, int],
    expected_root: str,
) -> None:
    index = summary_index
    value = leaf
    if _read_summary_node(connection, session_key, 0, index) != value:
        raise ValueError("SQLite summary leaf disagrees with its checkpoint")
    for level in range(1, _SUMMARY_DEPTH + 1):
        sibling = _read_summary_node(connection, session_key, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        value = (
            summary_parent_hash(session_key, level, parent_index, left, right),
            left[1] + right[1],
        )
        if _read_summary_node(connection, session_key, level, parent_index) != value:
            raise ValueError("SQLite summary Merkle path is invalid")
        index = parent_index
    if value[0] != expected_root:
        raise ValueError("SQLite summary path disagrees with its Session root")


def _verify_authenticated_root(row: sqlite3.Row, session_key: str) -> None:
    ledger = validate_hash(row["ledger_hash"])
    coverage = validate_hash(row["coverage_hash"])
    count = row["transcript_count"]
    if type(count) is not int or count < 0:
        raise ValueError("SQLite transcript coverage count is invalid")
    summary = validate_hash(row["summary_hash"])
    summary_count = row["summary_count"]
    if type(summary_count) is not int or not 0 <= summary_count <= (1 << _SUMMARY_DEPTH):
        raise ValueError("SQLite summary count is invalid")
    run_count = row["run_count"]
    if type(run_count) is not int or not 0 <= run_count <= (1 << _RUN_DEPTH):
        raise ValueError("SQLite Run projection count is invalid")
    expected = authenticated_root_hash(
        session_key,
        ledger,
        coverage,
        count,
        summary,
        summary_count,
        validate_hash(row["session_state_hash"]),
        validate_hash(row["run_projection_hash"]),
        run_count,
    )
    if validate_hash(row["authenticated_root"]) != expected:
        raise ValueError("SQLite authenticated Session root disagrees")


def _verify_coverage_path(
    connection: sqlite3.Connection,
    session_key: str,
    block_index: int,
    leaf: tuple[str, int, int, int],
    expected_root: str,
) -> None:
    index = block_index
    value = leaf
    if _read_coverage_node(connection, session_key, 0, index) != value:
        raise ValueError("SQLite coverage leaf disagrees with its block")
    for level in range(1, _COVERAGE_DEPTH + 1):
        sibling = _read_coverage_node(connection, session_key, level - 1, index ^ 1)
        left, right = (value, sibling) if index % 2 == 0 else (sibling, value)
        parent_index = index // 2
        count = left[1] + right[1]
        value = (
            coverage_parent_hash(session_key, level, parent_index, left, right),
            count,
            left[2] if left[1] else right[2],
            right[3] if right[1] else left[3],
        )
        if _read_coverage_node(connection, session_key, level, parent_index) != value:
            raise ValueError("SQLite coverage Merkle path is invalid")
        index = parent_index
    if value[0] != expected_root:
        raise ValueError("SQLite coverage path disagrees with its Session root")


def _verify_physical_coverage_block(
    connection: sqlite3.Connection,
    session_key: str,
    block_index: int,
    expected_root: str,
) -> tuple[int, ...]:
    block = connection.execute(
        "SELECT first_sequence, last_sequence, item_count, content_hash "
        "FROM transcript_coverage_blocks WHERE session_id=? AND block_index=?",
        (session_key, block_index),
    ).fetchone()
    if block is None:
        raise ValueError("SQLite coverage block is missing")
    rows = tuple(
        connection.execute(
            "SELECT t.session_id, t.sequence, t.run_id, t.submission_id, t.role, t.content, "
            "t.entry_type, t.format_version, t.entry_json, j.record_hash "
            "FROM transcript AS t JOIN journal_entries AS j "
            "ON j.session_id=t.session_id AND j.sequence=t.sequence AND j.kind='transcript' "
            "WHERE t.session_id=? AND t.sequence>=? ORDER BY t.sequence LIMIT ?",
            (session_key, block["first_sequence"], _COVERAGE_BLOCK_SIZE),
        )
    )
    records = []
    for row in rows:
        _decode_transcript_row(row, session_key)
        _verify_point_anchor(connection, session_key, row["sequence"])
        records.append((row["sequence"], validate_hash(row["record_hash"]), row["entry_json"]))
    if (
        not records
        or records[0][0] != block["first_sequence"]
        or records[-1][0] != block["last_sequence"]
        or len(records) != block["item_count"]
    ):
        raise ValueError("SQLite coverage block disagrees with physical transcript rows")
    content_hash = coverage_block_content_hash(session_key, block_index, tuple(records))
    if validate_hash(block["content_hash"]) != content_hash:
        raise ValueError("SQLite coverage block content hash disagrees")
    leaf = (
        coverage_leaf_hash(
            session_key,
            block_index,
            records[0][0],
            records[-1][0],
            len(records),
            content_hash,
        ),
        len(records),
        records[0][0],
        records[-1][0],
    )
    _verify_coverage_path(connection, session_key, block_index, leaf, expected_root)
    return tuple(record[0] for record in records)


def _coverage_range_count(
    connection: sqlite3.Connection,
    session_key: str,
    level: int,
    node_index: int,
    first_block: int,
    last_block: int,
) -> tuple[tuple[str, int, int, int], int]:
    value = _read_coverage_node(connection, session_key, level, node_index)
    node_first = node_index << level
    node_last = node_first + (1 << level) - 1
    if node_last < first_block or node_first > last_block:
        return value, 0
    if first_block <= node_first and node_last <= last_block:
        return value, value[1]
    if level == 0:
        return value, value[1]
    left, left_count = _coverage_range_count(
        connection, session_key, level - 1, node_index * 2, first_block, last_block
    )
    right, right_count = _coverage_range_count(
        connection, session_key, level - 1, node_index * 2 + 1, first_block, last_block
    )
    expected = (
        coverage_parent_hash(session_key, level, node_index, left, right),
        left[1] + right[1],
        left[2] if left[1] else right[2],
        right[3] if right[1] else left[3],
    )
    if value != expected:
        raise ValueError("SQLite coverage range proof is invalid")
    return value, left_count + right_count


def _authenticated_transcript_count(
    connection: sqlite3.Connection,
    session_key: str,
    first_sequence: int,
    last_sequence: int,
    coverage_root: str,
) -> int:
    first_row = connection.execute(
        "SELECT block_index, last_sequence FROM transcript_coverage_blocks WHERE session_id=? "
        "AND first_sequence<=? ORDER BY first_sequence DESC LIMIT 1",
        (session_key, first_sequence),
    ).fetchone()
    if first_row is None or first_row["last_sequence"] < first_sequence:
        first_row = connection.execute(
            "SELECT block_index FROM transcript_coverage_blocks WHERE session_id=? "
            "AND first_sequence>? AND first_sequence<=? ORDER BY first_sequence LIMIT 1",
            (session_key, first_sequence, last_sequence),
        ).fetchone()
    last_row = connection.execute(
        "SELECT block_index FROM transcript_coverage_blocks WHERE session_id=? "
        "AND first_sequence<=? ORDER BY first_sequence DESC LIMIT 1",
        (session_key, last_sequence),
    ).fetchone()
    physical_first = connection.execute(
        "SELECT sequence FROM transcript WHERE session_id=? AND sequence>=? AND sequence<=? "
        "ORDER BY sequence LIMIT 1",
        (session_key, first_sequence, last_sequence),
    ).fetchone()
    physical_last = connection.execute(
        "SELECT sequence FROM transcript WHERE session_id=? AND sequence>=? AND sequence<=? "
        "ORDER BY sequence DESC LIMIT 1",
        (session_key, first_sequence, last_sequence),
    ).fetchone()
    if first_row is None or last_row is None:
        if physical_first is not None or physical_last is not None:
            raise ValueError("SQLite coverage index omitted physical transcript rows")
        return 0
    first_block = int(first_row["block_index"])
    last_block = int(last_row["block_index"])
    first_sequences = _verify_physical_coverage_block(
        connection, session_key, first_block, coverage_root
    )
    last_sequences = (
        first_sequences
        if last_block == first_block
        else _verify_physical_coverage_block(connection, session_key, last_block, coverage_root)
    )
    if physical_first is None and physical_last is None:
        if first_block != last_block or any(
            first_sequence <= sequence <= last_sequence for sequence in first_sequences
        ):
            raise ValueError("SQLite empty range disagrees with authenticated transcript rows")
        return 0
    if (
        physical_first is None
        or physical_last is None
        or physical_first["sequence"] not in first_sequences
        or physical_last["sequence"] not in last_sequences
    ):
        raise ValueError("SQLite coverage boundary blocks disagree with physical history")
    if first_block == last_block:
        return sum(first_sequence <= sequence <= last_sequence for sequence in first_sequences)
    count = sum(sequence >= first_sequence for sequence in first_sequences)
    count += sum(sequence <= last_sequence for sequence in last_sequences)
    if first_block + 1 <= last_block - 1:
        root, interior = _coverage_range_count(
            connection,
            session_key,
            _COVERAGE_DEPTH,
            0,
            first_block + 1,
            last_block - 1,
        )
        if root[0] != coverage_root:
            raise ValueError("SQLite coverage range root disagrees with its Session")
        count += interior
    return count


def _verify_ledger_tail(
    connection: sqlite3.Connection, session_key: str, sequence: int, ledger_hash: object
) -> None:
    """Check the head boundary in O(1) indexed reads, without authenticating older history."""
    root = validate_hash(ledger_hash)
    previous_hash = genesis_hash(session_key)
    if sequence == 0:
        if root != previous_hash:
            raise ValueError("SQLite empty ledger root disagrees with its Session")
        return
    if sequence > 1:
        previous = connection.execute(
            "SELECT record_hash FROM journal_entries WHERE session_id=? AND sequence=?",
            (session_key, sequence - 1),
        ).fetchone()
        if previous is None:
            raise ValueError("SQLite ledger tail predecessor is missing")
        previous_hash = validate_hash(previous["record_hash"])
    anchor = connection.execute(
        _CHAIN_ROWS + " WHERE j.session_id=? AND j.sequence=?", (session_key, sequence)
    ).fetchone()
    if anchor is None or _verify_anchor(anchor, session_key, previous_hash) != root:
        raise ValueError("SQLite ledger tail disagrees with its Session root")


def _verify_point_anchor(connection: sqlite3.Connection, session_key: str, sequence: int) -> None:
    """Authenticate one indexed record and its immediate ledger boundary in O(1)."""
    session = connection.execute(
        "SELECT sequence, ledger_hash, coverage_hash, transcript_count, summary_hash, "
        "summary_count, session_state_hash, run_projection_hash, run_count, authenticated_root "
        "FROM sessions WHERE session_id=?",
        (session_key,),
    ).fetchone()
    if session is None or type(session["sequence"]) is not int or sequence > session["sequence"]:
        raise ValueError("SQLite point anchor is outside its Session")
    _verify_authenticated_root(session, session_key)
    previous_hash = genesis_hash(session_key)
    if sequence > 1:
        previous = connection.execute(
            "SELECT record_hash FROM journal_entries WHERE session_id=? AND sequence=?",
            (session_key, sequence - 1),
        ).fetchone()
        if previous is None:
            raise ValueError("SQLite point anchor predecessor is missing")
        previous_hash = validate_hash(previous["record_hash"])
    anchor = connection.execute(
        _CHAIN_ROWS + " WHERE j.session_id=? AND j.sequence=?", (session_key, sequence)
    ).fetchone()
    if anchor is None:
        raise ValueError("SQLite point anchor is missing")
    digest = _verify_anchor(anchor, session_key, previous_hash)
    if sequence == session["sequence"] and digest != validate_hash(session["ledger_hash"]):
        raise ValueError("SQLite point anchor disagrees with Session root")


def _reject_integrity_violation(connection: sqlite3.Connection, session_key: str) -> None:
    if connection.execute(
        "SELECT EXISTS(SELECT 1 FROM integrity_violations WHERE session_id=?)",
        (session_key,),
    ).fetchone()[0]:
        raise ValueError("SQLite immutable journal history was modified")


def _preflight_existing_ledger(
    connection: sqlite3.Connection, session_key: str
) -> tuple[sqlite3.Row | None, RunState | None]:
    """Validate one Session's immutable ledger before a commit reads its CAS counters."""
    row = connection.execute(
        "SELECT session_id, version, sequence, state_json, ledger_hash, coverage_hash, "
        "transcript_count, summary_hash, summary_count, session_state_hash, run_projection_hash, run_count, authenticated_root "
        "FROM sessions WHERE session_id = ?",
        (session_key,),
    ).fetchone()
    _reject_integrity_violation(connection, session_key)
    anchor_stats = connection.execute(
        "SELECT COUNT(*), MIN(sequence), MAX(sequence) FROM journal_entries WHERE session_id = ?",
        (session_key,),
    ).fetchone()
    event_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?", (session_key,)
        ).fetchone()[0]
    )
    transcript_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM transcript WHERE session_id = ?", (session_key,)
        ).fetchone()[0]
    )
    anchor_count = int(anchor_stats[0])
    if row is None:
        if anchor_count or event_count or transcript_count:
            raise ValueError("SQLite ledger has records without a Session")
        return None, None

    state = decode(row["state_json"], RunState)
    _verify_authenticated_root(row, session_key)
    coverage_root = _coverage_root(connection, session_key)
    if (
        coverage_root[0] != validate_hash(row["coverage_hash"])
        or coverage_root[1] != row["transcript_count"]
    ):
        raise ValueError("SQLite coverage root disagrees with its Session")
    _, head_sequence = _validate_session_row(row, session_key, state)
    current_run_row = connection.execute(
        "SELECT session_id, run_id, submission_id, profile_id, status, state_version, run_index, state_hash, state_json "
        "FROM runs WHERE session_id = ? AND run_id = ?",
        (session_key, state.run_id.value),
    ).fetchone()
    if current_run_row is None or decode(current_run_row["state_json"], RunState) != state:
        raise ValueError("SQLite Session head disagrees with its Run projection")
    _validate_run_row(current_run_row, state)
    if validate_hash(row["session_state_hash"]) != validate_hash(current_run_row["state_hash"]):
        raise ValueError("SQLite Session state hash disagrees with current Run")
    _verify_run_projection_path(
        connection,
        session_key,
        current_run_row["run_index"],
        current_run_row["state_hash"],
        row["run_projection_hash"],
    )

    minimum = anchor_stats[1]
    maximum = anchor_stats[2]
    if (
        anchor_count != head_sequence
        or event_count + transcript_count != head_sequence
        or (head_sequence == 0 and (minimum is not None or maximum is not None))
        or (head_sequence > 0 and (minimum != 1 or maximum != head_sequence))
    ):
        raise ValueError("SQLite ledger sequence range disagrees with its Session head")

    bad_anchor = int(
        connection.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM journal_entries AS j "
            "LEFT JOIN events AS e ON e.session_id=j.session_id AND e.sequence=j.sequence "
            "LEFT JOIN transcript AS t ON t.session_id=j.session_id AND t.sequence=j.sequence "
            "WHERE j.session_id=? AND ("
            "j.kind NOT IN ('event', 'transcript') OR "
            "(j.kind='event' AND (e.sequence IS NULL OR t.sequence IS NOT NULL)) OR "
            "(j.kind='transcript' AND (t.sequence IS NULL OR e.sequence IS NOT NULL))"
            ") LIMIT 1)",
            (session_key,),
        ).fetchone()[0]
    )
    orphan_event = int(
        connection.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM events AS e "
            "LEFT JOIN journal_entries AS j "
            "ON j.session_id=e.session_id AND j.sequence=e.sequence AND j.kind='event' "
            "WHERE e.session_id=? AND j.sequence IS NULL LIMIT 1)",
            (session_key,),
        ).fetchone()[0]
    )
    orphan_transcript = int(
        connection.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM transcript AS t "
            "LEFT JOIN journal_entries AS j "
            "ON j.session_id=t.session_id AND j.sequence=t.sequence AND j.kind='transcript' "
            "WHERE t.session_id=? AND j.sequence IS NULL LIMIT 1)",
            (session_key,),
        ).fetchone()[0]
    )
    if bad_anchor or orphan_event or orphan_transcript:
        raise ValueError("SQLite ledger anchors disagree with physical records")

    previous_hash = genesis_hash(session_key)
    for anchor in connection.execute(
        _CHAIN_ROWS + " WHERE j.session_id=? ORDER BY j.sequence", (session_key,)
    ):
        previous_hash = _verify_anchor(anchor, session_key, previous_hash)
    if previous_hash != validate_hash(row["ledger_hash"]):
        raise ValueError("SQLite ledger root disagrees with its history")

    for submission_row in connection.execute(
        "SELECT session_id, submission_id, run_id, digest, intent_json, receipt_json, "
        "acceptance_state_json, acceptance_version FROM submissions "
        "WHERE session_id = ? ORDER BY submission_id",
        (session_key,),
    ):
        receipt, _, _ = _decode_submission_row(
            submission_row, session_key, str(submission_row["submission_id"])
        )
        for expected_event in receipt.events:
            physical = connection.execute(
                "SELECT event_id, session_id, sequence, run_id, event_json FROM events "
                "WHERE session_id = ? AND sequence = ?",
                (session_key, expected_event.sequence),
            ).fetchone()
            if physical is None or _decode_event_row(physical, session_key) != expected_event:
                raise ValueError("SQLite acceptance Event disagrees with physical history")
        for expected_entry in receipt.transcript:
            physical = connection.execute(
                "SELECT session_id, sequence, run_id, submission_id, role, content, "
                "entry_type, format_version, entry_json "
                "FROM transcript WHERE session_id = ? AND sequence = ?",
                (session_key, expected_entry.sequence),
            ).fetchone()
            if physical is None or _decode_transcript_row(physical, session_key) != expected_entry:
                raise ValueError("SQLite acceptance transcript disagrees with physical history")
    return row, state


def _validate_session_row(row: sqlite3.Row, session_key: str, state: RunState) -> tuple[int, int]:
    version = row["version"]
    sequence = row["sequence"]
    if (
        row["session_id"] != session_key
        or state.session_id.value != session_key
        or type(version) is not int
        or version < 1
        or type(sequence) is not int
        or sequence < 0
        or state.journal_version != version
        or state.journal_sequence != sequence
    ):
        raise ValueError("SQLite Session row disagrees with its state")
    return version, sequence


def _validate_run_row(row: sqlite3.Row, state: RunState) -> None:
    if (
        row["session_id"] != state.session_id.value
        or row["run_id"] != state.run_id.value
        or row["submission_id"] != state.submission_id.value
        or row["profile_id"] != state.profile_id.value
        or row["status"] != state.status.value
        or type(row["state_version"]) is not int
        or row["state_version"] != state.version
        or type(row["run_index"]) is not int
        or row["run_index"] < 0
        or validate_hash(row["state_hash"])
        != state_record_hash(
            state.session_id.value,
            state.run_id.value,
            state.submission_id.value,
            row["state_json"],
        )
    ):
        raise ValueError("SQLite Run row disagrees with its state")


def _decode_event_row(row: sqlite3.Row, session_key: str) -> JournalEvent:
    record = decode(row["event_json"], JournalEvent)
    if (
        row["session_id"] != session_key
        or type(row["sequence"]) is not int
        or row["sequence"] != record.sequence
        or row["event_id"] != record.event.event_id.value
        or row["run_id"] != record.event.run_id.value
        or record.event.session_id.value != session_key
    ):
        raise ValueError("SQLite Event row disagrees with its payload")
    return record


def _decode_transcript_row(row: sqlite3.Row, session_key: str) -> TranscriptEntry:
    record = decode(row["entry_json"], TranscriptEntry)
    if (
        row["session_id"] != session_key
        or type(row["sequence"]) is not int
        or row["sequence"] != record.sequence
        or row["run_id"] != record.run_id.value
        or row["submission_id"] != record.submission_id.value
        or row["role"] != record.role
        or row["content"] != record.content
        or row["entry_type"] != record.entry_type.value
        or row["format_version"] != record.format_version
        or record.session_id.value != session_key
    ):
        raise ValueError("SQLite Transcript row disagrees with its payload")
    return record


def _decode_summary_row(row: sqlite3.Row, session_key: str) -> ConversationSummaryCheckpoint:
    record = decode(row["checkpoint_json"], ConversationSummaryCheckpoint)
    if (
        row["session_id"] != session_key
        or row["covered_from_sequence"] != record.covered_from_sequence
        or row["covered_through_sequence"] != record.covered_through_sequence
        or row["policy_version"] != record.policy_version
        or row["profile_id"] != record.profile_id.value
        or record.session_id.value != session_key
    ):
        raise ValueError("SQLite Summary row disagrees with its payload")
    return record


def _valid_policy_version(value: object) -> bool:
    if type(value) is not str or not value or value.strip() != value:
        return False
    try:
        return len(value.encode("utf-8")) <= 256
    except UnicodeEncodeError:
        return False


def _decode_submission_row(
    row: sqlite3.Row, session_key: str, submission_key: str
) -> tuple[JournalSnapshot, RunState, int]:
    receipt = decode(row["receipt_json"], JournalSnapshot)
    state = decode(row["acceptance_state_json"], RunState)
    intent = decode(row["intent_json"], JournalCommit)
    version = row["acceptance_version"]
    submission = receipt.submission
    span = len(intent.events) + len(intent.transcript)
    previous_sequence = receipt.sequence - span
    if previous_sequence < 0:
        raise ValueError("SQLite Submission receipt has an invalid sequence range")
    validate_commit_receipt(
        receipt,
        intent,
        expected_version=version - 1,
        previous_sequence=previous_sequence,
    )
    if (
        submission is None
        or type(version) is not int
        or version < 1
        or receipt.version != version
        or receipt.session_id.value != session_key
        or receipt.state != state
        or state.session_id.value != session_key
        or state.submission_id.value != submission_key
        or intent.state != state
        or intent.submission != submission
        or state.journal_version != version
        or state.journal_sequence != receipt.sequence
        or row["session_id"] != session_key
        or row["submission_id"] != submission_key
        or row["run_id"] != state.run_id.value
        or row["digest"] != submission.digest
        or submission.session_id.value != session_key
        or submission.run_id != state.run_id
        or submission.submission_id.value != submission_key
    ):
        raise ValueError("SQLite Submission row disagrees with its receipt")
    return receipt, state, version
