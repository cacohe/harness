from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import stat
from collections.abc import AsyncGenerator, Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from threading import Event
from typing import TypeVar
from uuid import uuid4

from src.adapters.artifacts.filesystem_platform import (
    ArtifactRepositoryError as ArtifactRepositoryError,
)
from src.adapters.artifacts.filesystem_platform import (
    Directory,
    directory_info,
    object_lock,
    pin_root,
    regular_file,
    same_entry,
    same_file,
)
from src.core.artifacts import (
    MAX_ARTIFACT_SIZE_BYTES,
    ArtifactRef,
    ArtifactWrite,
    artifact_id_from_sha256,
    validate_artifact_id,
)

_CHUNK_BYTES = 65_536
_DESCRIPTOR_BYTES = 65_536 + 1024
_DATA = "data"
_DESCRIPTOR = "descriptor.json"
_MEMBERS = {_DATA, _DESCRIPTOR}
_RETIRED = ".retired"
_RETIRED_MEMBERS = _MEMBERS | {_RETIRED}
_LOCKS = ".locks"
_TOMBSTONE = re.compile(r"\.delete-([0-9a-f]{64})-[0-9a-f]{32}")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_T = TypeVar("_T")


@dataclass(frozen=True)
class _Object:
    directory: os.stat_result
    descriptor: os.stat_result
    data: os.stat_result
    ref: ArtifactRef


@dataclass
class _Stage:
    name: str
    directory: os.stat_result | None = None
    fd: int | None = None

    def close(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)


@contextmanager
def _repository_io() -> Iterator[None]:
    try:
        yield
    except OSError:
        raise ArtifactRepositoryError("Artifact filesystem operation failed") from None


async def _finish_io(operation: Callable[[], _T], cancel: Event | None = None) -> _T:
    """Wait for an in-flight I/O operation before its caller closes or cleans up."""
    pending = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        if cancel is not None:
            cancel.set()
        while not pending.done():
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if not pending.cancelled():
            pending.exception()
        raise


def _write_bytes(fd: int, content: memoryview) -> None:
    while content:
        written = os.write(fd, content)
        if not written:
            raise OSError(errno.EIO, "Incomplete artifact write")
        content = content[written:]


def _encode_descriptor(ref: ArtifactRef) -> bytes:
    return json.dumps(
        {
            "artifact_id": ref.artifact_id,
            "media_type": ref.media_type,
            "size_bytes": ref.size_bytes,
            "sha256": ref.sha256,
            "name": ref.name,
            "metadata": ref.metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _decode_descriptor(content: bytes, artifact_id: str) -> ArtifactRef:
    try:
        fields = json.loads(content)
        ref = ArtifactRef(**fields)
        if ref.artifact_id != artifact_id or _encode_descriptor(ref) != content:
            raise ValueError
    except (TypeError, ValueError, RecursionError):
        raise ArtifactRepositoryError("Invalid artifact descriptor") from None
    return ref


def _validated_write(request: ArtifactWrite) -> ArtifactWrite:
    if type(request) is not ArtifactWrite:
        raise ValueError("Expected an ArtifactWrite")
    try:
        return ArtifactWrite(
            request.stream,
            request.media_type,
            request.name,
            request.metadata,
            expected_size_bytes=request.expected_size_bytes,
            expected_sha256=request.expected_sha256,
            max_bytes=request.max_bytes,
        )
    except (AttributeError, TypeError, ValueError, RecursionError):
        raise ValueError("Invalid artifact write request") from None


def _members(directory: Directory, allowed: set[str] = _MEMBERS) -> set[str]:
    with directory.entries() as entries:
        members = {member.name for _, member in zip(range(len(allowed) + 1), entries)}
    if not members <= allowed:
        raise ArtifactRepositoryError("Invalid artifact object members")
    return members


def _descriptor(directory: Directory, info: os.stat_result, artifact_id: str) -> ArtifactRef:
    if info.st_size > _DESCRIPTOR_BYTES:
        raise ArtifactRepositoryError("Invalid artifact descriptor file")
    with directory.read(_DESCRIPTOR, info) as fd:
        content = bytearray()
        while len(content) <= _DESCRIPTOR_BYTES:
            block = os.read(fd, min(_CHUNK_BYTES, _DESCRIPTOR_BYTES + 1 - len(content)))
            if not block:
                break
            content.extend(block)
        same_file(os.fstat(fd), info)
    if len(content) != info.st_size:
        raise ArtifactRepositoryError("Invalid artifact descriptor file")
    return _decode_descriptor(bytes(content), artifact_id)


def _load(root: Directory, name: str, artifact_id: str) -> _Object | None:
    try:
        info = directory_info(root.stat(name))
    except FileNotFoundError:
        return None
    with root.child(name, info) as directory:
        if _members(directory) != _MEMBERS:
            raise ArtifactRepositoryError("Invalid artifact object members")
        descriptor = regular_file(directory.stat(_DESCRIPTOR))
        data = regular_file(directory.stat(_DATA))
        if (descriptor.st_mode | data.st_mode) & stat.S_IWUSR:
            raise ArtifactRepositoryError("Artifact object is writable")
        ref = _descriptor(directory, descriptor, artifact_id)
        if data.st_size != ref.size_bytes:
            raise ArtifactRepositoryError("Invalid artifact data file")
        return _Object(directory.info, descriptor, data, ref)


def _check_object(current: _Object | None, expected: _Object) -> _Object:
    if current is None or current.ref != expected.ref:
        raise ArtifactRepositoryError("Artifact object changed")
    same_entry(current.directory, expected.directory)
    same_file(current.data, expected.data)
    same_file(current.descriptor, expected.descriptor)
    return current


def _restore(root: Directory, captured: str, original: str) -> None:
    try:
        root.stat(original)
    except FileNotFoundError:
        try:
            root.rename(captured, original)
            root.sync()
            return
        except OSError:
            pass
    raise ArtifactRepositoryError("Could not restore captured artifact")


def _remove_directory(
    root: Directory,
    name: str,
    info: os.stat_result,
    *,
    artifact_id: str | None = None,
) -> None:
    with root.child(name, info) as directory:
        members = _members(directory, _RETIRED_MEMBERS if artifact_id is not None else _MEMBERS)
        entries = {member: regular_file(directory.stat(member)) for member in members}
        if artifact_id is not None and members:
            marker = entries.get(_RETIRED)
            if marker is None or marker.st_size != 0 or members == {_DATA, _RETIRED}:
                raise ArtifactRepositoryError("Invalid artifact tombstone")
            if _DESCRIPTOR in members:
                ref = _descriptor(directory, entries[_DESCRIPTOR], artifact_id)
                if _DATA in members and entries[_DATA].st_size != ref.size_bytes:
                    raise ArtifactRepositoryError("Invalid artifact tombstone data")
        for member in (_DATA, _DESCRIPTOR, _RETIRED):
            if member in entries:
                directory.unlink(member, entries[member])
    same_entry(directory_info(root.stat(name)), info)
    root.rmdir(name)
    root.sync()


class FilesystemArtifactRepository:
    """Immutable local artifacts in a root writable exclusively by the adapter.

    Processes using this repository coordinate by digest. Each read chunk belongs
    to the object first observed by the reader. A fully consumed stream verifies
    its SHA-256. Deletion between chunks stops a reader of the deleted object.
    """

    def __init__(self, root: Path, max_artifact_bytes: int) -> None:
        if (
            type(max_artifact_bytes) is not int
            or not 1 <= max_artifact_bytes <= MAX_ARTIFACT_SIZE_BYTES
        ):
            raise ValueError("Maximum artifact size must be a positive bounded integer")
        self._max_artifact_bytes = max_artifact_bytes
        with _repository_io():
            try:
                self._root = root.resolve()
            except RuntimeError:
                raise ArtifactRepositoryError("Invalid artifact repository root") from None
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._root_info = directory_info(self._root.lstat())
            with pin_root(self._root, self._root_info) as directory:
                try:
                    directory.mkdir(_LOCKS)
                except FileExistsError:
                    pass
                self._locks_info = directory_info(directory.stat(_LOCKS))

    @contextmanager
    def _pinned_root(self) -> Iterator[Directory]:
        with pin_root(self._root, self._root_info) as root:
            yield root

    @contextmanager
    def _locked(self, root: Directory, digest: str) -> Iterator[None]:
        with root.child(_LOCKS, self._locks_info) as locks, object_lock(locks, digest):
            yield

    def _start(self, stage: _Stage) -> int:
        with self._pinned_root() as root:
            root.mkdir(stage.name)
            stage.directory = directory_info(root.stat(stage.name))
            with root.child(stage.name, stage.directory) as directory:
                stage.fd = directory.open(_DATA, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                return stage.fd

    def _seal_data(self, stage: _Stage, fd: int) -> os.stat_result:
        assert stage.directory is not None
        with self._pinned_root() as root, root.child(stage.name, stage.directory) as directory:
            same_entry(regular_file(directory.stat(_DATA)), regular_file(os.fstat(fd)))
            directory.set_mode(_DATA, fd, stat.S_IRUSR)
            os.fsync(fd)
            return os.fstat(fd)

    def _describe(self, stage: _Stage, ref: ArtifactRef) -> os.stat_result:
        assert stage.directory is not None
        with self._pinned_root() as root, root.child(stage.name, stage.directory) as directory:
            fd = directory.open(_DESCRIPTOR, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                content = memoryview(_encode_descriptor(ref))
                for offset in range(0, len(content), _CHUNK_BYTES):
                    _write_bytes(fd, content[offset : offset + _CHUNK_BYTES])
                directory.set_mode(_DESCRIPTOR, fd, stat.S_IRUSR)
                os.fsync(fd)
                return os.fstat(fd)
            finally:
                os.close(fd)

    def _publish(self, stage: _Stage, obj: _Object, cancel: Event) -> None:
        digest = obj.ref.sha256
        with self._pinned_root() as root, self._locked(root, digest):
            if cancel.is_set():
                return
            existing = _load(root, digest, obj.ref.artifact_id)
            if existing is None:
                _check_object(_load(root, stage.name, obj.ref.artifact_id), obj)
                with root.child(stage.name, obj.directory, writable=True) as directory:
                    directory.refresh()
                if cancel.is_set():
                    return
                try:
                    root.rename(stage.name, digest)
                except OSError:
                    existing = _load(root, digest, obj.ref.artifact_id)
                    if existing is None:
                        raise
                else:
                    try:
                        _check_object(_load(root, digest, obj.ref.artifact_id), obj)
                    except (OSError, ArtifactRepositoryError):
                        _restore(root, digest, stage.name)
                        raise ArtifactRepositoryError("Published artifact changed") from None
                    root.sync()
                    return
            if existing.ref != obj.ref:
                raise ValueError("Artifact descriptor conflict for content address")
            with root.child(digest, existing.directory, writable=True) as directory:
                directory.refresh()

    def _discard(self, stage: _Stage) -> None:
        stage.close()
        if stage.directory is None:
            return
        with self._pinned_root() as root:
            try:
                root.stat(stage.name)
            except FileNotFoundError:
                return
            _remove_directory(root, stage.name, stage.directory)

    async def put(self, request: ArtifactWrite) -> ArtifactRef:
        await asyncio.sleep(0)
        task = asyncio.current_task()
        baseline = task.cancelling() if task is not None else 0

        def check_cancellation() -> None:
            if task is not None and task.cancelling() > baseline:
                raise asyncio.CancelledError

        request = _validated_write(request)
        limit = min(request.max_bytes, self._max_artifact_bytes)
        expected_size = request.expected_size_bytes
        if expected_size is not None:
            if expected_size > limit:
                raise ValueError("Expected artifact size exceeds the repository write limit")
            limit = expected_size

        stage = _Stage(".put-" + uuid4().hex)
        with _repository_io():
            try:
                fd = await _finish_io(lambda: self._start(stage))
                size = 0
                digest = sha256()
                async for chunk in request.stream:
                    check_cancellation()
                    if type(chunk) is not bytes:
                        raise ValueError("Artifact stream chunks must be exact bytes")
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("Artifact stream exceeds its write limit or expected size")
                    if not chunk:
                        await asyncio.sleep(0)
                    for offset in range(0, len(chunk), _CHUNK_BYTES):
                        view = memoryview(chunk)[offset : offset + _CHUNK_BYTES]
                        digest.update(view)
                        await _finish_io(lambda: _write_bytes(fd, view))

                check_cancellation()
                checksum = digest.hexdigest()
                if expected_size is not None and size != expected_size:
                    raise ValueError("Artifact stream size does not match the expected size")
                if request.expected_sha256 is not None and request.expected_sha256 != checksum:
                    raise ValueError("Artifact stream SHA-256 does not match the expected digest")
                data = await _finish_io(lambda: self._seal_data(stage, fd))
                await _finish_io(stage.close)
                ref = ArtifactRef(
                    artifact_id_from_sha256(checksum),
                    request.media_type,
                    size,
                    checksum,
                    request.name,
                    request.metadata,
                )
                descriptor = await _finish_io(lambda: self._describe(stage, ref))
                check_cancellation()
                assert stage.directory is not None
                obj = _Object(stage.directory, descriptor, data, ref)
                cancel = Event()
                await _finish_io(lambda: self._publish(stage, obj, cancel), cancel)
            except BaseException:
                try:
                    await _finish_io(lambda: self._discard(stage))
                except (OSError, ArtifactRepositoryError):
                    # The original failure takes precedence over cleanup failure.
                    pass
                raise
            await _finish_io(lambda: self._discard(stage))
            return ref

    def _find(self, digest: str, artifact_id: str) -> _Object | None:
        with self._pinned_root() as root, self._locked(root, digest):
            return _load(root, digest, artifact_id)

    async def stat(self, artifact_id: str) -> ArtifactRef | None:
        validate_artifact_id(artifact_id)
        with _repository_io():
            found = await _finish_io(lambda: self._find(artifact_id[7:], artifact_id))
        return found.ref if found is not None else None

    def _read_chunk(self, obj: _Object, offset: int) -> bytes:
        digest = obj.ref.sha256
        with self._pinned_root() as root, self._locked(root, digest):
            try:
                directory_info(root.stat(digest))
            except FileNotFoundError:
                raise KeyError(obj.ref.artifact_id) from None
            with root.child(digest, obj.directory) as directory:
                same_file(directory.stat(_DESCRIPTOR), obj.descriptor)
                with directory.read(_DATA, obj.data) as fd:
                    os.lseek(fd, offset, os.SEEK_SET)
                    block = os.read(fd, min(_CHUNK_BYTES, obj.ref.size_bytes - offset))
                    same_file(os.fstat(fd), obj.data)
            if not block:
                raise ArtifactRepositoryError("Artifact file ended during reading")
            return block

    async def open(self, artifact_id: str) -> AsyncGenerator[bytes, None]:
        validate_artifact_id(artifact_id)
        with _repository_io():
            obj = await _finish_io(lambda: self._find(artifact_id[7:], artifact_id))
            if obj is None:
                raise KeyError(artifact_id)
            offset = 0
            digest = sha256()
            while offset < obj.ref.size_bytes:
                block = await _finish_io(lambda: self._read_chunk(obj, offset))
                offset += len(block)
                digest.update(block)
                yield block
            if offset != obj.ref.size_bytes or digest.hexdigest() != obj.ref.sha256:
                raise ArtifactRepositoryError("Artifact content does not match its size or SHA-256")

    def _delete_object(self, root: Directory, obj: _Object, cutoff_ns: int | None = None) -> bool:
        digest = obj.ref.sha256
        tombstone = f".delete-{digest}-{uuid4().hex}"
        root.rename(digest, tombstone)
        try:
            current = _check_object(_load(root, tombstone, obj.ref.artifact_id), obj)
        except (OSError, ArtifactRepositoryError):
            _restore(root, tombstone, digest)
            raise ArtifactRepositoryError("Captured artifact changed") from None
        if cutoff_ns is not None and current.directory.st_mtime_ns >= cutoff_ns:
            _restore(root, tombstone, digest)
            return False
        # Record approval inside the verified directory. An unapproved capture
        # left by a failed rollback must never be treated as pending cleanup.
        with root.child(tombstone, current.directory) as directory:
            same_file(directory.stat(_DATA), current.data)
            same_file(directory.stat(_DESCRIPTOR), current.descriptor)
            fd = directory.open(_RETIRED, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            directory.sync()
        root.sync()
        _remove_directory(root, tombstone, current.directory, artifact_id=obj.ref.artifact_id)
        return True

    def _recover(self, root: Directory, name: str, artifact_id: str) -> bool:
        try:
            info = directory_info(root.stat(name))
        except FileNotFoundError:
            return False
        _remove_directory(root, name, info, artifact_id=artifact_id)
        return True

    def _delete(self, digest: str, artifact_id: str) -> None:
        with self._pinned_root() as root, self._locked(root, digest):
            obj = _load(root, digest, artifact_id)
            if obj is not None:
                self._delete_object(root, obj)
            with root.entries() as entries:
                for entry in entries:
                    tombstone = _TOMBSTONE.fullmatch(entry.name)
                    if tombstone is not None and tombstone[1] == digest:
                        self._recover(root, entry.name, artifact_id)

    async def delete(self, artifact_id: str) -> None:
        validate_artifact_id(artifact_id)
        with _repository_io():
            await _finish_io(lambda: self._delete(artifact_id[7:], artifact_id))

    async def collect_orphans(self, referenced_ids: Collection[str], older_than: datetime) -> int:
        if (
            type(older_than) is not datetime
            or older_than.tzinfo is not timezone.utc
            or older_than.fold != 0
        ):
            raise ValueError("Artifact orphan cutoff must be an exact UTC datetime")
        references = set()
        for artifact_id in referenced_ids:
            validate_artifact_id(artifact_id)
            references.add(artifact_id)
        age = older_than - _EPOCH
        cutoff_ns = ((age.days * 86_400 + age.seconds) * 1_000_000 + age.microseconds) * 1000
        with _repository_io():
            return await _finish_io(lambda: self._collect(references, cutoff_ns))

    def _collect(self, references: set[str], cutoff_ns: int) -> int:
        removed = 0
        with self._pinned_root() as root, root.entries() as entries:
            for entry in entries:
                tombstone = _TOMBSTONE.fullmatch(entry.name)
                digest = tombstone[1] if tombstone is not None else entry.name
                try:
                    artifact_id = artifact_id_from_sha256(digest)
                except ValueError:
                    continue
                if tombstone is None and artifact_id in references:
                    continue
                try:
                    with self._locked(root, digest):
                        if tombstone is not None:
                            removed += self._recover(root, entry.name, artifact_id)
                            continue
                        obj = _load(root, digest, artifact_id)
                        if obj is not None and obj.directory.st_mtime_ns < cutoff_ns:
                            removed += self._delete_object(root, obj, cutoff_ns)
                except (OSError, ArtifactRepositoryError):
                    # Corrupt, replaced, or busy entries are left for inspection or retry.
                    continue
        return removed
