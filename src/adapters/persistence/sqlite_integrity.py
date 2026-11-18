"""Versioned storage-corruption detection, not authentication against a DB administrator.

All frames use UTF-8 fields with unsigned 64-bit big-endian byte lengths. SHA-256
digests are lowercase hexadecimal. Schema/hash versions and domains are explicit.
An administrator able to rewrite every anchor and the root can forge a new chain.
"""

from hashlib import sha256

_DOMAIN = "agent-harness/sqlite-journal/schema-6/hash-1/sha256"


def _digest(*fields: str) -> str:
    digest = sha256()
    for field in (_DOMAIN, *fields):
        data = field.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def genesis_hash(session_key: str) -> str:
    return _digest("genesis", session_key)


def record_hash(
    session_key: str, sequence: int, kind: str, canonical_json: str, previous_hash: str
) -> str:
    """Hash only already decoded/re-encoded canonical database JSON text."""
    return _digest("record", session_key, str(sequence), kind, canonical_json, previous_hash)


def coverage_empty_hash(session_key: str, level: int, node_index: int) -> str:
    return _digest("coverage-empty", session_key, str(level), str(node_index))


def coverage_block_content_hash(
    session_key: str,
    block_index: int,
    records: tuple[tuple[int, str, str], ...],
) -> str:
    fields = ["coverage-block-content", session_key, str(block_index), str(len(records))]
    for sequence, anchor_hash, canonical_json in records:
        fields.extend((str(sequence), validate_hash(anchor_hash), canonical_json))
    return _digest(*fields)


def coverage_leaf_hash(
    session_key: str,
    block_index: int,
    first_sequence: int,
    last_sequence: int,
    item_count: int,
    content_hash: str,
) -> str:
    return _digest(
        "coverage-leaf",
        session_key,
        str(block_index),
        str(first_sequence),
        str(last_sequence),
        str(item_count),
        validate_hash(content_hash),
    )


def coverage_parent_hash(
    session_key: str,
    level: int,
    node_index: int,
    left: tuple[str, int, int, int],
    right: tuple[str, int, int, int],
) -> str:
    return _digest(
        "coverage-parent",
        session_key,
        str(level),
        str(node_index),
        validate_hash(left[0]),
        str(left[1]),
        str(left[2]),
        str(left[3]),
        validate_hash(right[0]),
        str(right[1]),
        str(right[2]),
        str(right[3]),
    )


def authenticated_root_hash(
    session_key: str,
    ledger_hash: str,
    coverage_hash: str,
    transcript_count: int,
    summary_hash: str,
    summary_count: int,
    session_state_hash: str,
    run_projection_hash: str,
    run_count: int,
) -> str:
    return _digest(
        "authenticated-root",
        session_key,
        validate_hash(ledger_hash),
        validate_hash(coverage_hash),
        str(transcript_count),
        validate_hash(summary_hash),
        str(summary_count),
        validate_hash(session_state_hash),
        validate_hash(run_projection_hash),
        str(run_count),
    )


def state_record_hash(
    session_key: str, run_key: str, submission_key: str, canonical_json: str
) -> str:
    return _digest("state-record", session_key, run_key, submission_key, canonical_json)


def run_empty_hash(session_key: str, level: int, node_index: int) -> str:
    return _digest("run-empty", session_key, str(level), str(node_index))


def run_leaf_hash(session_key: str, run_index: int, state_hash: str) -> str:
    return _digest("run-leaf", session_key, str(run_index), validate_hash(state_hash))


def run_parent_hash(
    session_key: str,
    level: int,
    node_index: int,
    left: tuple[str, int],
    right: tuple[str, int],
) -> str:
    return _digest(
        "run-parent", session_key, str(level), str(node_index), validate_hash(left[0]),
        str(left[1]), validate_hash(right[0]), str(right[1])
    )


def summary_empty_hash(session_key: str, level: int, node_index: int) -> str:
    return _digest("summary-empty", session_key, str(level), str(node_index))


def summary_record_hash(session_key: str, summary_index: int, canonical_json: str) -> str:
    return _digest("summary-record", session_key, str(summary_index), canonical_json)


def summary_leaf_hash(session_key: str, summary_index: int, record_digest: str) -> str:
    return _digest(
        "summary-leaf", session_key, str(summary_index), validate_hash(record_digest)
    )


def summary_parent_hash(
    session_key: str,
    level: int,
    node_index: int,
    left: tuple[str, int],
    right: tuple[str, int],
) -> str:
    return _digest(
        "summary-parent",
        session_key,
        str(level),
        str(node_index),
        validate_hash(left[0]),
        str(left[1]),
        validate_hash(right[0]),
        str(right[1]),
    )


def validate_hash(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("SQLite ledger hash is invalid")
    return value
