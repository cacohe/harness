"""Versioned storage-corruption detection, not authentication against a DB administrator.

All frames use UTF-8 fields with unsigned 64-bit big-endian byte lengths. SHA-256
digests are lowercase hexadecimal. Schema/hash versions and domains are explicit.
An administrator able to rewrite every anchor and the root can forge a new chain.
"""

from hashlib import sha256

_DOMAIN = "agent-harness/sqlite-journal/schema-3/hash-1/sha256"


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


def validate_hash(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("SQLite ledger hash is invalid")
    return value
