from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Literal
from uuid import uuid4

from src.adapters.artifacts.filesystem_platform import (
    ArtifactRepositoryError,
    Directory,
    regular_file,
    same_file,
    write_bytes,
)

_INTENT_BYTES = 4096
_DIGEST = re.compile(r"[0-9a-f]{64}")
_NONCE = re.compile(r"[0-9a-f]{32}")
_INTENT = re.compile(r"([0-9a-f]{64})-[0-9a-f]{32}\.json")
_MAX_IDENTITY = (1 << 128) - 1
_MAX_TIMESTAMP = (1 << 127) - 1
_MAX_SIZE = (1 << 63) - 1

DirectoryIdentity = tuple[int, int]
FileIdentity = tuple[int, int, int, int]


def directory_identity(info: os.stat_result) -> DirectoryIdentity:
    return info.st_dev, info.st_ino


def file_identity(info: os.stat_result) -> FileIdentity:
    return *directory_identity(info), info.st_size, info.st_mtime_ns


@dataclass(frozen=True)
class RetirementIntent:
    digest: str
    nonce: str
    directory: DirectoryIdentity
    data: FileIdentity
    descriptor: FileIdentity
    descriptor_sha256: str
    original_mtime_ns: int
    operation: Literal["delete", "gc"]
    cutoff_ns: int | None
    captured_mtime_ns: int | None = None

    @property
    def name(self) -> str:
        return f"{self.digest}-{self.nonce}.json"

    @property
    def tombstone(self) -> str:
        return f".delete-{self.digest}-{self.nonce}"


@dataclass(frozen=True)
class RetirementRecord:
    intent: RetirementIntent
    info: os.stat_result


def intent_digest(name: str) -> str | None:
    match = _INTENT.fullmatch(name)
    return match[1] if match else None


def _integer(value: object, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _timestamp(value: object) -> bool:
    return _integer(value, -_MAX_TIMESTAMP, _MAX_TIMESTAMP)


def _validate(intent: RetirementIntent, name: str) -> None:
    stamps = (intent.directory, intent.data[:2], intent.descriptor[:2])
    if (
        not _DIGEST.fullmatch(intent.digest)
        or not _NONCE.fullmatch(intent.nonce)
        or not _DIGEST.fullmatch(intent.descriptor_sha256)
        or intent.name != name
        or any(
            len(stamp) != 2 or not all(_integer(part, 0, _MAX_IDENTITY) for part in stamp)
            for stamp in stamps
        )
        or any(
            len(stamp) != 4 or not _integer(stamp[2], 0, _MAX_SIZE) or not _timestamp(stamp[3])
            for stamp in (intent.data, intent.descriptor)
        )
        or not _timestamp(intent.original_mtime_ns)
    ):
        raise ValueError

    if intent.operation == "delete":
        if intent.cutoff_ns is not None:
            raise ValueError
    elif (
        intent.operation != "gc"
        or intent.cutoff_ns is None
        or not _timestamp(intent.cutoff_ns)
        or intent.original_mtime_ns >= intent.cutoff_ns
    ):
        raise ValueError

    captured = intent.captured_mtime_ns
    if captured is not None and (
        not _timestamp(captured) or (intent.cutoff_ns is not None and captured >= intent.cutoff_ns)
    ):
        raise ValueError


def _encode(intent: RetirementIntent) -> bytes:
    return json.dumps(asdict(intent), sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode(content: bytes, name: str) -> RetirementIntent:
    try:
        fields = json.loads(content)
        if type(fields) is not dict:
            raise ValueError
        for key in ("directory", "data", "descriptor"):
            if type(fields[key]) is not list:
                raise ValueError
            fields[key] = tuple(fields[key])
        intent = RetirementIntent(**fields)
        _validate(intent, name)
        if _encode(intent) != content:
            raise ValueError
    except (KeyError, TypeError, ValueError, RecursionError):
        raise ArtifactRepositoryError("Invalid artifact retirement intent") from None
    return intent


def read_record(control: Directory, name: str) -> RetirementRecord | None:
    try:
        info = regular_file(control.stat(name))
    except FileNotFoundError:
        return None
    if not 0 < info.st_size <= _INTENT_BYTES:
        raise ArtifactRepositoryError("Invalid artifact retirement intent")
    with control.read(name, info) as fd:
        content = bytearray()
        while len(content) <= _INTENT_BYTES:
            block = os.read(fd, _INTENT_BYTES + 1 - len(content))
            if not block:
                break
            content.extend(block)
        same_file(os.fstat(fd), info)
    if len(content) != info.st_size:
        raise ArtifactRepositoryError("Invalid artifact retirement intent")
    return RetirementRecord(_decode(bytes(content), name), info)


def write_record(
    control: Directory, intent: RetirementIntent, previous: RetirementRecord | None = None
) -> RetirementRecord:
    pending = ".write-" + uuid4().hex
    fd = control.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    info = None
    try:
        try:
            info = os.fstat(fd)
            write_bytes(fd, memoryview(_encode(intent)))
            os.fsync(fd)
        finally:
            os.close(fd)
        if previous is None:
            control.rename(pending, intent.name)
        else:
            same_file(control.stat(intent.name), previous.info)
            control.replace(pending, intent.name)
        control.sync()
    except BaseException:
        if info is not None:
            try:
                current = regular_file(control.stat(pending))
                if directory_identity(current) == directory_identity(info):
                    control.unlink(pending, current)
            except (OSError, ArtifactRepositoryError):
                pass
        raise

    assert info is not None
    current = regular_file(control.stat(intent.name))
    if directory_identity(current) != directory_identity(info):
        raise ArtifactRepositoryError("Artifact retirement intent changed")
    return RetirementRecord(intent, current)


def remove_record(control: Directory, record: RetirementRecord) -> None:
    control.unlink(record.intent.name, record.info)
    control.sync()
