from __future__ import annotations

import asyncio
import errno
import json
import os
import stat
from collections.abc import AsyncGenerator, Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock
from typing import TypeVar
from uuid import uuid4

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
_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_OBJECT_LOCK = Lock()
_T = TypeVar("_T")


class ArtifactRepositoryError(RuntimeError):
    """A storage failure or an invalid on-disk artifact."""


@dataclass(frozen=True)
class _Object:
    directory: os.stat_result
    descriptor: os.stat_result
    data: os.stat_result
    ref: ArtifactRef


@contextmanager
def _repository_io() -> Iterator[None]:
    try:
        yield
    except OSError:
        raise ArtifactRepositoryError("Artifact filesystem operation failed") from None


async def _finish_io(operation: Callable[[], _T]) -> _T:
    """Wait for an in-flight I/O operation before its caller closes or cleans up."""
    pending = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
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


def _entry(path: Path, expected_type: Callable[[int], bool]) -> os.stat_result:
    info = path.lstat()
    if not expected_type(info.st_mode) or (
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    ):
        raise ArtifactRepositoryError("Invalid artifact filesystem entry")
    return info


def _same_entry(current: os.stat_result, expected: os.stat_result) -> None:
    if not os.path.samestat(current, expected):
        raise ArtifactRepositoryError("Artifact filesystem entry changed")


def _same_file(current: os.stat_result, expected: os.stat_result) -> None:
    _same_entry(current, expected)
    if (
        current.st_nlink != 1
        or current.st_size != expected.st_size
        or current.st_mtime_ns != expected.st_mtime_ns
        or current.st_mode != expected.st_mode
    ):
        raise ArtifactRepositoryError("Artifact file changed")


def _read_fd(path: Path) -> int:
    if os.name != "nt":
        return os.open(path, os.O_RDONLY | _BINARY | _NOFOLLOW)

    import _winapi
    import msvcrt

    generic_read = 0x80000000
    share_read = 0x01
    open_existing = 3
    open_reparse_point = 0x00200000
    handle = _winapi.CreateFile(
        str(path), generic_read, share_read, 0, open_existing, open_reparse_point, 0
    )
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | _BINARY)
    except BaseException:
        _winapi.CloseHandle(handle)
        raise


@contextmanager
def _open_file(path: Path, expected: os.stat_result) -> Iterator[int]:
    _same_file(_entry(path, stat.S_ISREG), expected)
    fd = _read_fd(path)
    try:
        _same_file(os.fstat(fd), expected)
        _same_file(_entry(path, stat.S_ISREG), expected)
        yield fd
    finally:
        os.close(fd)


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


def _sync_directory(directory: Path) -> None:
    # Windows does not support opening directory fds through os.open.
    if os.name != "nt":
        fd = os.open(directory, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _seal_file(path: Path, fd: int) -> None:
    if hasattr(os, "fchmod"):
        os.fchmod(fd, stat.S_IRUSR)
    else:
        _same_entry(_entry(path, stat.S_ISREG), os.fstat(fd))
        os.chmod(path, stat.S_IREAD)
    os.fsync(fd)


class FilesystemArtifactRepository:
    """Store immutable artifacts as directories beneath a fixed local root.

    Each read chunk belongs to the object first observed by the reader. Deletion
    between chunks stops the reader, even if the same address is recreated.
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
            self._root_info = _entry(self._root, stat.S_ISDIR)

    def _check_root(self) -> None:
        _same_entry(_entry(self._root, stat.S_ISDIR), self._root_info)
        if self._root.resolve(strict=True) != self._root:
            raise ArtifactRepositoryError("Artifact repository root changed")

    def _object_path(self, artifact_id: str) -> Path:
        validate_artifact_id(artifact_id)
        return self._root / artifact_id[7:]

    def _check_directory(self, path: Path, expected: os.stat_result) -> None:
        self._check_root()
        if path.parent != self._root:
            raise ArtifactRepositoryError("Invalid artifact object directory")
        _same_entry(_entry(path, stat.S_ISDIR), expected)
        resolved = path.resolve(strict=True)
        if resolved != path or resolved.name != path.name:
            raise ArtifactRepositoryError("Invalid artifact object directory")

    def _check_members(self, path: Path) -> None:
        with os.scandir(path) as entries:
            if {member.name for _, member in zip(range(3), entries)} != {_DATA, _DESCRIPTOR}:
                raise ArtifactRepositoryError("Invalid artifact object members")

    def _load(self, path: Path, artifact_id: str) -> _Object | None:
        self._check_root()
        try:
            directory = _entry(path, stat.S_ISDIR)
        except FileNotFoundError:
            return None
        self._check_directory(path, directory)
        self._check_members(path)
        descriptor_path = path / _DESCRIPTOR
        descriptor = _entry(descriptor_path, stat.S_ISREG)
        if (
            descriptor.st_size > _DESCRIPTOR_BYTES
            or descriptor.st_nlink != 1
            or descriptor.st_mode & stat.S_IWUSR
        ):
            raise ArtifactRepositoryError("Invalid artifact descriptor file")
        with _open_file(descriptor_path, descriptor) as fd:
            self._check_directory(path, directory)
            content = bytearray()
            while len(content) <= _DESCRIPTOR_BYTES:
                block = os.read(fd, min(_CHUNK_BYTES, _DESCRIPTOR_BYTES + 1 - len(content)))
                if not block:
                    break
                content.extend(block)
            _same_file(os.fstat(fd), descriptor)
        if len(content) != descriptor.st_size:
            raise ArtifactRepositoryError("Invalid artifact descriptor file")

        ref = _decode_descriptor(bytes(content), artifact_id)
        data = _entry(path / _DATA, stat.S_ISREG)
        if data.st_size != ref.size_bytes or data.st_nlink != 1 or data.st_mode & stat.S_IWUSR:
            raise ArtifactRepositoryError("Invalid artifact data file")
        self._check_directory(path, directory)
        return _Object(directory, descriptor, data, ref)

    def _write_descriptor(
        self, directory: Path, info: os.stat_result, ref: ArtifactRef
    ) -> os.stat_result:
        self._check_directory(directory, info)
        path = directory / _DESCRIPTOR
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW, 0o600)
        try:
            content = memoryview(_encode_descriptor(ref))
            for offset in range(0, len(content), _CHUNK_BYTES):
                _write_bytes(fd, content[offset : offset + _CHUNK_BYTES])
            _seal_file(path, fd)
            return os.fstat(fd)
        finally:
            os.close(fd)

    def _publish(self, staging: Path, obj: _Object) -> bool:
        artifact_id = obj.ref.artifact_id
        path = self._object_path(artifact_id)
        with _OBJECT_LOCK:
            existing = self._load(path, artifact_id)
            if existing is None:
                self._check_directory(staging, obj.directory)
                self._check_members(staging)
                _same_file(_entry(staging / _DATA, stat.S_ISREG), obj.data)
                _same_file(_entry(staging / _DESCRIPTOR, stat.S_ISREG), obj.descriptor)
                _sync_directory(staging)
                try:
                    os.rename(staging, path)
                except OSError:
                    # Another process can publish a complete directory before rename.
                    existing = self._load(path, artifact_id)
                    if existing is None:
                        raise
                else:
                    _sync_directory(self._root)
                    return True
            if existing.ref != obj.ref:
                raise ValueError("Artifact descriptor conflict for content address")
            return False

    def _remove_directory(self, path: Path, info: os.stat_result) -> None:
        self._check_directory(path, info)
        for name in (_DATA, _DESCRIPTOR):
            member = path / name
            try:
                member_info = _entry(member, stat.S_ISREG)
            except FileNotFoundError:
                continue
            if member_info.st_nlink != 1:
                raise ArtifactRepositoryError("Invalid artifact filesystem entry")
            self._check_directory(path, info)
            if os.name == "nt" and not member_info.st_mode & stat.S_IWUSR:
                os.chmod(member, stat.S_IWRITE)
            member.unlink()
        path.rmdir()

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

        with _repository_io():
            self._check_root()
            staging = Path(mkdtemp(prefix=".put-", dir=self._root))
            info = _entry(staging, stat.S_ISDIR)
            try:
                self._check_directory(staging, info)
                path = staging / _DATA
                fd = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW, 0o600
                )
                try:
                    size = 0
                    digest = sha256()
                    async for chunk in request.stream:
                        check_cancellation()
                        if type(chunk) is not bytes:
                            raise ValueError("Artifact stream chunks must be exact bytes")
                        size += len(chunk)
                        if size > limit:
                            raise ValueError(
                                "Artifact stream exceeds its write limit or expected size"
                            )
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
                        raise ValueError(
                            "Artifact stream SHA-256 does not match the expected digest"
                        )
                    await _finish_io(lambda: _seal_file(path, fd))
                    data = os.fstat(fd)
                finally:
                    os.close(fd)

                ref = ArtifactRef(
                    artifact_id_from_sha256(checksum),
                    request.media_type,
                    size,
                    checksum,
                    request.name,
                    request.metadata,
                )
                descriptor = await _finish_io(lambda: self._write_descriptor(staging, info, ref))
                check_cancellation()
                if self._publish(staging, _Object(info, descriptor, data, ref)):
                    return ref
            except BaseException:
                try:
                    self._remove_directory(staging, info)
                except (OSError, ArtifactRepositoryError):
                    # The original failure takes precedence over cleanup failure.
                    pass
                raise
            self._remove_directory(staging, info)
            return ref

    async def stat(self, artifact_id: str) -> ArtifactRef | None:
        path = self._object_path(artifact_id)
        with _repository_io():
            found = await asyncio.to_thread(self._find, path, artifact_id)
        return found.ref if found is not None else None

    def _find(self, path: Path, artifact_id: str) -> _Object | None:
        with _OBJECT_LOCK:
            return self._load(path, artifact_id)

    def _read_chunk(self, path: Path, obj: _Object, offset: int) -> bytes:
        with _OBJECT_LOCK:
            try:
                self._check_directory(path, obj.directory)
            except FileNotFoundError:
                raise KeyError(obj.ref.artifact_id) from None
            _same_file(_entry(path / _DESCRIPTOR, stat.S_ISREG), obj.descriptor)
            with _open_file(path / _DATA, obj.data) as fd:
                self._check_directory(path, obj.directory)
                os.lseek(fd, offset, os.SEEK_SET)
                block = os.read(fd, min(_CHUNK_BYTES, obj.ref.size_bytes - offset))
                _same_file(os.fstat(fd), obj.data)
            if not block and offset != obj.ref.size_bytes:
                raise ArtifactRepositoryError("Artifact file ended during reading")
            return block

    async def open(self, artifact_id: str) -> AsyncGenerator[bytes, None]:
        path = self._object_path(artifact_id)
        with _repository_io():
            obj = await _finish_io(lambda: self._find(path, artifact_id))
            if obj is None:
                raise KeyError(artifact_id)
            offset = 0
            while offset < obj.ref.size_bytes:
                block = await _finish_io(lambda: self._read_chunk(path, obj, offset))
                offset += len(block)
                yield block

    def _delete_object(self, path: Path, obj: _Object) -> None:
        self._check_directory(path, obj.directory)
        _same_file(_entry(path / _DATA, stat.S_ISREG), obj.data)
        _same_file(_entry(path / _DESCRIPTOR, stat.S_ISREG), obj.descriptor)
        # Unpublish both members together before changing permissions or unlinking.
        tombstone = self._root / (".delete-" + uuid4().hex)
        os.rename(path, tombstone)
        self._remove_directory(tombstone, obj.directory)

    def _delete(self, path: Path, artifact_id: str) -> None:
        with _OBJECT_LOCK:
            obj = self._load(path, artifact_id)
            if obj is not None:
                self._delete_object(path, obj)

    async def delete(self, artifact_id: str) -> None:
        path = self._object_path(artifact_id)
        with _repository_io():
            await _finish_io(lambda: self._delete(path, artifact_id))

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
        self._check_root()
        removed = 0
        for path in self._root.iterdir():
            try:
                artifact_id = artifact_id_from_sha256(path.name)
            except ValueError:
                continue
            if artifact_id in references:
                continue
            try:
                with _OBJECT_LOCK:
                    obj = self._load(path, artifact_id)
                    if obj is not None and obj.directory.st_mtime_ns < cutoff_ns:
                        self._delete_object(path, obj)
                        removed += 1
            except (OSError, ArtifactRepositoryError):
                # A corrupt, replaced, or busy entry is not a collectible object.
                continue
        return removed
