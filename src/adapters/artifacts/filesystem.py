from __future__ import annotations

import asyncio
import json
import os
import re
import stat
from collections.abc import AsyncGenerator, Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
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
    acquire_object_lock,
    directory_info,
    object_lock,
    pin_root,
    regular_file,
    release_object_lock,
    same_entry,
    same_file,
    write_bytes,
)
from src.adapters.artifacts.filesystem_retirement import (
    RetirementIntent,
    RetirementRecord,
    directory_identity,
    file_identity,
    intent_digest,
    read_record,
    remove_record,
    write_record,
)
from src.core.artifacts import (
    MAX_ARTIFACT_SIZE_BYTES,
    ArtifactRef,
    ArtifactWrite,
    artifact_id_from_descriptor,
    artifact_id_from_sha256,
    validate_artifact_id,
)

_CHUNK_BYTES = 65_536
_DESCRIPTOR_BYTES = 65_536 + 1024
_DATA = "data"
_DESCRIPTOR = "descriptor.json"
_MEMBERS = {_DATA, _DESCRIPTOR}
_LOCKS = ".locks"
_INTENTS = ".retire-intents"
_STAGE_NAME = re.compile(r"\.put-[0-9a-f]{32}")
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
    lock_fd: int | None = None

    def close_data(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            os.close(fd)

    def close_lock(self) -> None:
        fd, self.lock_fd = self.lock_fd, None
        if fd is not None:
            release_object_lock(fd)


def _stage_lock_name(name: str) -> str:
    if _STAGE_NAME.fullmatch(name) is None:
        raise ValueError("Invalid artifact staging name")
    return "stage-" + name[5:7]


@contextmanager
def _repository_io() -> Iterator[None]:
    try:
        yield
    except OSError:
        raise ArtifactRepositoryError("Artifact filesystem operation failed") from None


def _run_io(operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except StopIteration:
        # asyncio futures reject StopIteration and would otherwise remain pending forever.
        raise ArtifactRepositoryError("Artifact filesystem operation failed") from None


async def _finish_io(operation: Callable[[], _T], cancel: Event | None = None) -> _T:
    """Wait for an in-flight I/O operation before its caller closes or cleans up."""
    pending = asyncio.create_task(asyncio.to_thread(_run_io, operation))
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


def _members(directory: Directory) -> set[str]:
    with directory.entries() as entries:
        members = {member.name for _, member in zip(range(len(_MEMBERS) + 1), entries)}
    if not members <= _MEMBERS:
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
) -> None:
    with root.child(name, info) as directory:
        members = _members(directory)
        entries = {member: regular_file(directory.stat(member)) for member in members}
        for member in (_DATA, _DESCRIPTOR):
            if member in entries:
                directory.unlink(member, entries[member])
    _remove_empty_directory(root, name, info)


def _remove_empty_directory(root: Directory, name: str, info: os.stat_result) -> None:
    same_entry(directory_info(root.stat(name)), info)
    root.rmdir(name)
    root.sync()


def _original_matches(obj: _Object | None, intent: RetirementIntent) -> bool:
    return (
        obj is not None
        and directory_identity(obj.directory) == intent.directory
        and file_identity(obj.data) == intent.data
        and file_identity(obj.descriptor) == intent.descriptor
        and sha256(_encode_descriptor(obj.ref)).hexdigest() == intent.descriptor_sha256
    )


def _retired_members(directory: Directory, intent: RetirementIntent) -> dict[str, os.stat_result]:
    if directory_identity(directory.info) != intent.directory:
        raise ArtifactRepositoryError("Captured artifact directory changed")
    members = _members(directory)
    if members == {_DATA} or (intent.captured_mtime_ns is None and members != _MEMBERS):
        raise ArtifactRepositoryError("Invalid artifact tombstone members")
    entries = {member: regular_file(directory.stat(member)) for member in members}
    if _DATA in entries and file_identity(entries[_DATA]) != intent.data:
        raise ArtifactRepositoryError("Captured artifact data changed")
    if _DESCRIPTOR in entries:
        info = entries[_DESCRIPTOR]
        if file_identity(info) != intent.descriptor:
            raise ArtifactRepositoryError("Captured artifact descriptor changed")
        ref = _descriptor(directory, info, artifact_id_from_sha256(intent.digest))
        if sha256(_encode_descriptor(ref)).hexdigest() != intent.descriptor_sha256:
            raise ArtifactRepositoryError("Captured artifact descriptor changed")
    return entries


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
        self._intents_info: os.stat_result | None = None
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

    @contextmanager
    def _intents(self, root: Directory) -> Iterator[Directory]:
        info = self._intents_info
        if info is None:
            try:
                root.mkdir(_INTENTS)
            except FileExistsError:
                pass
            info = directory_info(root.stat(_INTENTS))
            root.sync()
            self._intents_info = info
        with root.child(_INTENTS, info) as control:
            yield control

    def _start(self, stage: _Stage) -> int:
        with self._pinned_root() as root:
            root.mkdir(stage.name)
            stage.directory = directory_info(root.stat(stage.name))
            with root.child(stage.name, stage.directory) as directory:
                stage.fd = directory.open(_DATA, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                return stage.fd

    def _acquire_stage(self, stage: _Stage) -> None:
        with self._pinned_root() as root, root.child(_LOCKS, self._locks_info) as locks:
            stage.lock_fd = acquire_object_lock(locks, _stage_lock_name(stage.name))

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
                    write_bytes(fd, content[offset : offset + _CHUNK_BYTES])
                directory.set_mode(_DESCRIPTOR, fd, stat.S_IRUSR)
                os.fsync(fd)
                return os.fstat(fd)
            finally:
                os.close(fd)

    def _publish(self, stage: _Stage, obj: _Object, cancel: Event) -> None:
        digest = obj.ref.artifact_id[7:]
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
        stage.close_data()
        try:
            if stage.directory is None:
                return
            with self._pinned_root() as root:
                try:
                    root.stat(stage.name)
                except FileNotFoundError:
                    return
                _remove_directory(root, stage.name, stage.directory)
        finally:
            stage.close_lock()

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
                await _finish_io(lambda: self._acquire_stage(stage))
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
                        await _finish_io(lambda: write_bytes(fd, view))

                check_cancellation()
                checksum = digest.hexdigest()
                if expected_size is not None and size != expected_size:
                    raise ValueError("Artifact stream size does not match the expected size")
                if request.expected_sha256 is not None and request.expected_sha256 != checksum:
                    raise ValueError("Artifact stream SHA-256 does not match the expected digest")
                data = await _finish_io(lambda: self._seal_data(stage, fd))
                await _finish_io(stage.close_data)
                descriptor_id = artifact_id_from_descriptor(
                    checksum, request.media_type, request.name, request.metadata
                )
                publication_id = artifact_id_from_sha256(
                    sha256(f"{descriptor_id}:{uuid4().hex}".encode("ascii")).hexdigest()
                )
                ref = ArtifactRef(
                    publication_id,
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
        with self._pinned_root() as root:
            try:
                directory_info(root.stat(digest))
            except FileNotFoundError:
                return None
            with self._locked(root, digest):
                return _load(root, digest, artifact_id)

    async def stat(self, artifact_id: str) -> ArtifactRef | None:
        validate_artifact_id(artifact_id)
        with _repository_io():
            found = await _finish_io(lambda: self._find(artifact_id[7:], artifact_id))
        return found.ref if found is not None else None

    def _read_chunk(self, obj: _Object, offset: int) -> bytes:
        digest = obj.ref.artifact_id[7:]
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

    def _delete_object(
        self,
        root: Directory,
        control: Directory,
        obj: _Object,
        cutoff_ns: int | None = None,
    ) -> bool:
        intent = RetirementIntent(
            digest=obj.ref.artifact_id[7:],
            nonce=uuid4().hex,
            directory=directory_identity(obj.directory),
            data=file_identity(obj.data),
            descriptor=file_identity(obj.descriptor),
            descriptor_sha256=sha256(_encode_descriptor(obj.ref)).hexdigest(),
            original_mtime_ns=obj.directory.st_mtime_ns,
            operation="delete" if cutoff_ns is None else "gc",
            cutoff_ns=cutoff_ns,
        )
        record = write_record(control, intent)
        root.rename(intent.digest, intent.tombstone)
        root.sync()
        return self._resume_retirement(root, control, record, just_captured=True)

    def _resume_retirement(
        self,
        root: Directory,
        control: Directory,
        record: RetirementRecord,
        *,
        referenced: bool = False,
        just_captured: bool = False,
    ) -> bool:
        intent = record.intent
        try:
            info = root.stat(intent.tombstone)
        except FileNotFoundError:
            # Authorization is durable before any member or tombstone is removed.
            # Once its tombstone is gone, only the authorized record needs cleanup.
            if intent.captured_mtime_ns is None:
                final = _load(root, intent.digest, artifact_id_from_sha256(intent.digest))
                if not _original_matches(final, intent):
                    raise ArtifactRepositoryError("Unresolved artifact retirement intent") from None
            remove_record(control, record)
            return False

        verified = False
        try:
            with root.child(intent.tombstone, directory_info(info)) as directory:
                members = _retired_members(directory, intent)
                verified = True
                captured_mtime = directory.info.st_mtime_ns
                restore = (
                    len(members) == 2
                    and intent.cutoff_ns is not None
                    and (referenced or captured_mtime >= intent.cutoff_ns)
                )
                if not restore:
                    if intent.captured_mtime_ns is None:
                        record = write_record(
                            control, replace(intent, captured_mtime_ns=captured_mtime), record
                        )
                    # Check the lease again after persisting permission to start unlinking.
                    restore = (
                        len(members) == 2
                        and intent.cutoff_ns is not None
                        and directory.info.st_mtime_ns >= intent.cutoff_ns
                    )
                    if not restore:
                        for member in (_DATA, _DESCRIPTOR):
                            if member in members:
                                directory.unlink(member, members[member])
                                directory.sync()
        except (ArtifactRepositoryError, FileNotFoundError):
            if just_captured and not verified:
                _restore(root, intent.tombstone, intent.digest)
            raise

        if restore:
            _restore(root, intent.tombstone, intent.digest)
            remove_record(control, record)
            return False
        _remove_empty_directory(root, intent.tombstone, info)
        remove_record(control, record)
        return True

    def _delete(self, digest: str, artifact_id: str) -> None:
        with (
            self._pinned_root() as root,
            self._locked(root, digest),
            self._intents(root) as control,
        ):
            with control.entries() as entries:
                names = [entry.name for entry in entries if intent_digest(entry.name) == digest]
            for name in names:
                record = read_record(control, name)
                if record is not None:
                    self._resume_retirement(root, control, record)
            obj = _load(root, digest, artifact_id)
            if obj is not None:
                self._delete_object(root, control, obj)

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
        with self._pinned_root() as root, self._intents(root) as control:
            with root.entries() as entries:
                stages = [entry.name for entry in entries if _STAGE_NAME.fullmatch(entry.name)]
            for name in stages:
                try:
                    with root.child(_LOCKS, self._locks_info) as locks, object_lock(
                        locks, _stage_lock_name(name)
                    ):
                        removed += self._remove_abandoned_stage(root, name, cutoff_ns)
                except (OSError, ArtifactRepositoryError):
                    continue
            unresolved = set()
            with control.entries() as entries:
                for entry in entries:
                    digest = intent_digest(entry.name)
                    if digest is None:
                        continue
                    try:
                        with self._locked(root, digest):
                            record = read_record(control, entry.name)
                            if record is not None:
                                removed += self._resume_retirement(
                                    root,
                                    control,
                                    record,
                                    referenced=artifact_id_from_sha256(digest) in references,
                                )
                    except (OSError, ArtifactRepositoryError):
                        unresolved.add(digest)

            with root.entries() as entries:
                for entry in entries:
                    digest = entry.name
                    try:
                        artifact_id = artifact_id_from_sha256(digest)
                    except ValueError:
                        continue
                    if artifact_id in references or digest in unresolved:
                        continue
                    try:
                        with self._locked(root, digest):
                            obj = _load(root, digest, artifact_id)
                            if obj is not None and obj.directory.st_mtime_ns < cutoff_ns:
                                removed += self._delete_object(root, control, obj, cutoff_ns)
                    except (OSError, ArtifactRepositoryError):
                        # Corrupt, replaced, or busy entries are left for inspection or retry.
                        continue
        return removed

    @staticmethod
    def _remove_abandoned_stage(root: Directory, name: str, cutoff_ns: int) -> bool:
        try:
            info = directory_info(root.stat(name))
        except FileNotFoundError:
            return False
        if info.st_mtime_ns >= cutoff_ns:
            return False
        with root.child(name, info) as directory:
            members = _members(directory)
            entries = {member: regular_file(directory.stat(member)) for member in members}
            if any(entry.st_mtime_ns >= cutoff_ns for entry in entries.values()):
                return False
        current = directory_info(root.stat(name))
        same_entry(current, info)
        if current.st_mtime_ns >= cutoff_ns:
            return False
        _remove_directory(root, name, info)
        return True
