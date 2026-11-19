import asyncio
import errno
import hashlib
import os
import shutil
import stat
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone

import pytest
from src.adapters.artifacts.filesystem import ArtifactRepositoryError, FilesystemArtifactRepository
from src.adapters.artifacts.filesystem_platform import Directory
from src.core.artifacts import ArtifactWrite, artifact_id_from_descriptor


def _address(content, media_type="text/plain", name=None, metadata=None):
    return artifact_id_from_descriptor(
        hashlib.sha256(content).hexdigest(), media_type, name, metadata
    )


def _object(root, artifact_id):
    return root / artifact_id.removeprefix("sha256:")


def _contents(root):
    return {path for path in root.iterdir() if path.name not in {".locks", ".retire-intents"}}


async def _chunks(*chunks):
    for chunk in chunks:
        yield chunk


async def _read(repo, artifact_id):
    return b"".join([chunk async for chunk in repo.open(artifact_id)])


def _link(path, target, *, directory=False):
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        if getattr(error, "winerror", None) != 1314:
            raise
        pytest.skip("Windows symlink privilege is unavailable")


def _junction(path, target):
    if os.name == "nt":
        from _winapi import CreateJunction

        CreateJunction(str(target), str(path))
    else:
        _link(path, target, directory=True)


def test_large_stream_publishes_complete_read_only_object(tmp_path, monkeypatch):
    async def scenario():
        root = tmp_path / "repository"
        repo = FilesystemArtifactRepository(root, max_artifact_bytes=8 * 1024**2)
        payload = b"x" * (256 * 1024)
        digest = hashlib.sha256()
        for _ in range(20):
            digest.update(payload)
        async def source():
            for _ in range(20):
                yield payload

        receipt = await repo.put(
            ArtifactWrite(
                source(),
                "application/octet-stream",
                name="CON",
                metadata={"results": [20]},
                expected_size_bytes=len(payload) * 20,
                expected_sha256=digest.hexdigest(),
            )
        )
        artifact_id = receipt.artifact_id
        assert receipt.name == "CON"
        assert _contents(root) == {_object(root, artifact_id)}
        assert len(list(_object(root, artifact_id).iterdir())) == 2
        for file in _object(root, artifact_id).iterdir():
            assert file.stat().st_mode & stat.S_IWUSR == 0

        reopened = FilesystemArtifactRepository(root, max_artifact_bytes=8 * 1024**2)
        assert await reopened.stat(artifact_id) == receipt
        assert str(root) not in repr(receipt)
        assert receipt.metadata == {"results": [20]}
        read_digest = hashlib.sha256()
        read_size = 0
        async for chunk in reopened.open(artifact_id):
            assert 0 < len(chunk) <= 65_536
            read_size += len(chunk)
            read_digest.update(chunk)
        assert read_size == receipt.size_bytes
        assert read_digest.hexdigest() == digest.hexdigest()

    read = os.read
    write = os.write

    def read_bounded(fd, size):
        if os.fstat(fd).st_size > 128 * 1024:
            assert 0 < size <= 65_536
        return read(fd, size)

    def write_bounded(fd, content):
        assert 0 < len(content) <= 65_536
        return write(fd, content)

    with monkeypatch.context() as patch:
        patch.setattr(os, "read", read_bounded)
        patch.setattr(os, "write", write_bounded)
        asyncio.run(scenario())


def test_concurrent_writers_across_instances_publish_distinct_ids(tmp_path):
    async def scenario():
        repos = [FilesystemArtifactRepository(tmp_path, 3) for _ in range(8)]
        ready = asyncio.Event()
        remaining = len(repos)

        async def source():
            nonlocal remaining
            yield b"a"
            remaining -= 1
            if not remaining:
                ready.set()
            await ready.wait()
            yield b"bc"

        results = await asyncio.gather(
            *(repo.put(ArtifactWrite(source(), "text/plain")) for repo in repos)
        )
        assert len({ref.artifact_id for ref in results}) == len(results)
        assert all(ref.sha256 == hashlib.sha256(b"abc").hexdigest() for ref in results)
        stored = await asyncio.gather(*(repos[0].stat(ref.artifact_id) for ref in results))
        assert all(actual == expected for actual, expected in zip(stored, results))
        assert _contents(tmp_path) == {_object(tmp_path, ref.artifact_id) for ref in results}

    asyncio.run(scenario())


def test_same_bytes_with_different_media_types_publish_distinct_objects(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        text = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        binary = await repo.put(ArtifactWrite(_chunks(b"abc"), "application/octet-stream"))

        assert text.artifact_id != binary.artifact_id
        assert text.sha256 == binary.sha256 == hashlib.sha256(b"abc").hexdigest()
        assert await _read(repo, text.artifact_id) == b"abc"
        assert await _read(repo, binary.artifact_id) == b"abc"
        assert _contents(tmp_path) == {
            _object(tmp_path, text.artifact_id),
            _object(tmp_path, binary.artifact_id),
        }

    asyncio.run(scenario())


def test_missing_stat_does_not_create_permanent_lock_files(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        locks = tmp_path / ".locks"
        for index in range(100):
            artifact_id = "sha256:" + f"{index:064x}"
            assert await repo.stat(artifact_id) is None
        assert list(locks.iterdir()) == []

    asyncio.run(scenario())


def test_distinct_descriptors_publish_independently(tmp_path):
    async def scenario():
        first_repo = FilesystemArtifactRepository(tmp_path, 3)
        second_repo = FilesystemArtifactRepository(tmp_path, 3)
        ready = asyncio.Event()
        proceed = asyncio.Event()

        async def delayed():
            yield b"a"
            ready.set()
            await proceed.wait()
            yield b"bc"

        pending = asyncio.create_task(
            second_repo.put(ArtifactWrite(delayed(), "text/plain", metadata={"value": True}))
        )
        await ready.wait()
        first = await first_repo.put(
            ArtifactWrite(_chunks(b"abc"), "text/plain", metadata={"value": 1})
        )
        proceed.set()
        second = await pending
        assert second.artifact_id != first.artifact_id
        assert await second_repo.stat(first.artifact_id) == first
        assert await second_repo.stat(second.artifact_id) == second
        assert await _read(second_repo, first.artifact_id) == b"abc"
        assert await _read(second_repo, second.artifact_id) == b"abc"
        assert _contents(tmp_path) == {
            _object(tmp_path, first.artifact_id),
            _object(tmp_path, second.artifact_id),
        }

    asyncio.run(scenario())


class _BytesSubclass(bytes):
    pass


@pytest.mark.parametrize(
    ("chunk", "request_kwargs", "repo_limit"),
    [
        (b"bc", {"max_bytes": 2}, 4),
        (b"bc", {"max_bytes": 4}, 2),
        (b"bc", {"expected_size_bytes": 2}, 4),
        (b"bc", {"expected_size_bytes": 4}, 4),
        (b"bc", {"expected_sha256": "0" * 64}, 4),
        (bytearray(b"bc"), {}, 4),
        (_BytesSubclass(b"bc"), {}, 4),
    ],
)
def test_invalid_stream_preserves_other_entries(tmp_path, chunk, request_kwargs, repo_limit):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, repo_limit)
        prior = await repo.put(ArtifactWrite(_chunks(b"ok"), "text/plain"))
        other_temp = tmp_path / ".other-operation"
        other_temp.mkdir()

        with pytest.raises(ValueError):
            await repo.put(ArtifactWrite(_chunks(b"a", chunk), "text/plain", **request_kwargs))

        assert await repo.stat(_address(b"abc")) is None
        assert await _read(repo, prior.artifact_id) == b"ok"
        assert _contents(tmp_path) == {other_temp, _object(tmp_path, prior.artifact_id)}

    asyncio.run(scenario())


def test_invalid_request_is_rejected_before_reading_stream(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)

        async def source():
            pytest.fail("invalid request consumed the stream")
            yield b"a"

        request = ArtifactWrite(source(), "text/plain")
        object.__setattr__(request, "name", "../escaped")
        with pytest.raises(ValueError):
            await repo.put(request)
        assert not _contents(tmp_path)

    asyncio.run(scenario())


def test_iterator_error_removes_own_partial_write(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)

        async def broken():
            yield b"abc"
            raise RuntimeError("error from source")

        with pytest.raises(RuntimeError, match="error from source"):
            await repo.put(ArtifactWrite(broken(), "text/plain"))
        assert not _contents(tmp_path)

    asyncio.run(scenario())


def test_threaded_io_stop_iteration_fails_without_hanging():
    script = """
import asyncio
from src.adapters.artifacts.filesystem import _finish_io
from src.adapters.artifacts.filesystem_platform import ArtifactRepositoryError

def stop():
    raise StopIteration

async def main():
    try:
        await _finish_io(stop)
    except ArtifactRepositoryError:
        return
    raise AssertionError("StopIteration escaped the filesystem I/O boundary")

asyncio.run(main())
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("suppress", [False, True])
def test_cancelled_put_removes_own_partial_write(tmp_path, suppress):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        blocked = asyncio.Event()

        async def source():
            yield b"a"
            blocked.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                if not suppress:
                    raise
                yield b"bc"

        pending = asyncio.create_task(repo.put(ArtifactWrite(source(), "text/plain")))
        await blocked.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not _contents(tmp_path)

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["write", "fsync", "rename"])
def test_filesystem_failure_hides_path_and_removes_only_staging(tmp_path, monkeypatch, operation):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        prior = await repo.put(ArtifactWrite(_chunks(b"ok"), "text/plain"))

        def fail(*_args, **_kwargs):
            raise OSError(errno.EIO, "injected error", str(tmp_path))

        with monkeypatch.context() as patch:
            patch.setattr(os, operation, fail)
            with pytest.raises(ArtifactRepositoryError) as caught:
                await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        assert str(tmp_path) not in str(caught.value)
        assert caught.value.__cause__ is None
        assert _contents(tmp_path) == {_object(tmp_path, prior.artifact_id)}
        assert await _read(repo, prior.artifact_id) == b"ok"

    asyncio.run(scenario())


class _StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "artifact_id",
    ["../abc", "C:\\abc", "sha256:" + "A" * 64, _StringSubclass("sha256:" + "a" * 64)],
)
def test_lookups_and_orphan_filter_reject_invalid_ids(tmp_path, artifact_id):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"ok"), "text/plain"))
        for operation in (
            repo.stat(artifact_id),
            _read(repo, artifact_id),
            repo.delete(artifact_id),
            repo.collect_orphans([artifact_id], datetime.now(timezone.utc) + timedelta(days=1)),
        ):
            with pytest.raises(ValueError):
                await operation
        assert await repo.stat(ref.artifact_id) == ref

    asyncio.run(scenario())


@pytest.mark.parametrize("component", ["directory", "data", "descriptor"])
def test_repository_rejects_symlinks_without_touching_their_targets(tmp_path, component):
    async def scenario():
        root = tmp_path / "repository"
        repo = FilesystemArtifactRepository(root, 3)
        receipt = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        artifact_id = receipt.artifact_id
        obj = _object(root, artifact_id)
        if component == "directory":
            path = obj
        else:
            path = obj / ("data" if component == "data" else "descriptor.json")
        original = tmp_path / "outside"
        path.rename(original)
        if component == "directory":
            _junction(path, original)
        else:
            _link(path, original)

        for operation in (
            repo.stat(artifact_id),
            _read(repo, artifact_id),
            repo.delete(artifact_id),
        ):
            with pytest.raises(ArtifactRepositoryError) as caught:
                await operation
            assert str(tmp_path) not in str(caught.value)
        assert await repo.collect_orphans([], datetime.now(timezone.utc) + timedelta(days=1)) == 0
        assert original.exists()
        assert path.is_symlink() or (
            getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        )

    asyncio.run(scenario())


def test_repository_rejects_hard_linked_data_without_touching_the_target(tmp_path):
    async def scenario():
        root = tmp_path / "repo"
        repo = FilesystemArtifactRepository(root, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        data = _object(root, ref.artifact_id) / "data"
        original = tmp_path / "outside"
        data.rename(original)
        os.link(original, data)

        for operation in (
            repo.stat(ref.artifact_id),
            _read(repo, ref.artifact_id),
            repo.delete(ref.artifact_id),
        ):
            with pytest.raises(ArtifactRepositoryError):
                await operation
        assert await repo.collect_orphans([], datetime.now(timezone.utc) + timedelta(days=1)) == 0
        assert original.read_bytes() == b"abc"
        assert data.samefile(original)

    asyncio.run(scenario())


def test_staged_data_replacement_cannot_publish_a_ref(tmp_path, monkeypatch):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        real_fsync = os.fsync
        replaced = False

        def replace_data_during_descriptor_flush(fd):
            nonlocal replaced
            real_fsync(fd)
            staging = next(tmp_path.glob(".put-*"))
            data = staging / "data"
            if not replaced and (staging / "descriptor.json").exists():
                data.rename(tmp_path / "displaced-data")
                data.write_bytes(b"bad")
                replaced = True

        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", replace_data_during_descriptor_flush)
            with pytest.raises(ArtifactRepositoryError):
                await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        assert replaced
        assert await repo.stat(_address(b"abc")) is None
        assert {path.name for path in _contents(tmp_path)} == {"displaced-data"}

    asyncio.run(scenario())


def test_collect_orphans_skips_invalid_directory_with_extra_members(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"ok"), "text/plain"))
        obj = _object(tmp_path, ref.artifact_id)
        extra = obj / "unexpected"
        extra.write_bytes(b"preserve")
        assert await repo.collect_orphans([], datetime.now(timezone.utc) + timedelta(days=1)) == 0
        assert extra.read_bytes() == b"preserve"

    asyncio.run(scenario())


def test_repository_rejects_root_replacement(tmp_path):
    async def scenario():
        root = tmp_path / "repo"
        repo = FilesystemArtifactRepository(root, 3)
        root.rename(tmp_path / "original")
        root.mkdir()

        with pytest.raises(ArtifactRepositoryError):
            await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        assert not list(root.iterdir())

    asyncio.run(scenario())


def test_open_stops_after_delete_without_reading_a_replacement(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 200_000)
        payload = b"x" * 150_000
        ref = await repo.put(ArtifactWrite(_chunks(payload), "text/plain", name="first"))
        reader = repo.open(ref.artifact_id)
        assert 0 < len(await anext(reader)) < len(payload)
        await repo.delete(ref.artifact_id)
        await repo.delete(ref.artifact_id)
        assert await repo.stat(ref.artifact_id) is None
        await repo.put(ArtifactWrite(_chunks(payload), "text/plain", name="replacement"))
        with pytest.raises(KeyError):
            await anext(reader)

    asyncio.run(scenario())


def test_cancelled_read_and_early_close_allow_deletion(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 200_000)
        payload = b"x" * 150_000
        ref = await repo.put(ArtifactWrite(_chunks(payload), "text/plain"))
        early = repo.open(ref.artifact_id)
        assert await anext(early)
        await early.aclose()

        reader = repo.open(ref.artifact_id)
        assert await anext(reader)
        pending = asyncio.create_task(anext(reader))
        asyncio.get_running_loop().call_soon(pending.cancel)
        with pytest.raises(asyncio.CancelledError):
            await pending
        await repo.delete(ref.artifact_id)
        assert not _contents(tmp_path)

    asyncio.run(scenario())


def test_collect_orphans_observes_journal_failure_and_strict_grace_period(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 10)
        persisted = await repo.put(ArtifactWrite(_chunks(b"saved"), "text/plain"))
        references = {persisted.artifact_id}
        orphan = await repo.put(ArtifactWrite(_chunks(b"orphan"), "text/plain"))
        assert await _read(repo, orphan.artifact_id) == b"orphan"

        def commit_reference(artifact_id):
            assert artifact_id not in references
            raise RuntimeError("journal commit failed")

        with pytest.raises(RuntimeError, match="journal commit failed"):
            commit_reference(orphan.artifact_id)
        assert references == {persisted.artifact_id}

        published = datetime(2026, 11, 1, tzinfo=timezone.utc)
        for artifact_id in (persisted.artifact_id, orphan.artifact_id):
            timestamp = int(published.timestamp())
            os.utime(_object(tmp_path, artifact_id), (timestamp, timestamp))
        incomplete = _object(tmp_path, _address(b"incomplete"))
        incomplete.mkdir()
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        for invalid in (incomplete, unrelated):
            os.utime(invalid, (timestamp, timestamp))

        assert await repo.collect_orphans(references, published - timedelta(microseconds=1)) == 0
        assert await repo.collect_orphans(references, published) == 0
        assert await _read(repo, orphan.artifact_id) == b"orphan"
        assert await repo.collect_orphans(references, published + timedelta(microseconds=1)) == 1
        assert await repo.stat(orphan.artifact_id) is None
        assert await _read(repo, persisted.artifact_id) == b"saved"
        assert incomplete.is_dir() and unrelated.is_dir()
        assert await repo.collect_orphans(references, published + timedelta(days=1)) == 0

    asyncio.run(scenario())


class _DateTimeSubclass(datetime):
    pass


@pytest.mark.parametrize(
    "threshold",
    [
        datetime(2026, 11, 1),
        datetime(2026, 11, 1, tzinfo=timezone(timedelta(hours=1))),
        datetime(2026, 11, 1, tzinfo=timezone.utc, fold=1),
        _DateTimeSubclass(2026, 11, 1, tzinfo=timezone.utc),
    ],
)
def test_orphan_collection_requires_an_exact_utc_cutoff(tmp_path, threshold):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        with pytest.raises(ValueError):
            await repo.collect_orphans([], threshold)

    asyncio.run(scenario())


def test_new_publication_does_not_refresh_an_old_publications_orphan_grace(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        old = int((cutoff - timedelta(days=1)).timestamp())
        os.utime(_object(tmp_path, ref.artifact_id), (old, old))

        replacement = await repo.put(ArtifactWrite(_chunks(b"a", b"bc"), "text/plain"))
        assert replacement.artifact_id != ref.artifact_id
        assert replacement.sha256 == ref.sha256
        assert await repo.collect_orphans([], cutoff) == 1
        assert await repo.stat(ref.artifact_id) is None
        assert await _read(repo, replacement.artifact_id) == b"abc"

    asyncio.run(scenario())


def test_open_reports_same_size_disk_corruption_on_full_consumption(tmp_path):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        data = _object(tmp_path, ref.artifact_id) / "data"
        os.chmod(data, stat.S_IWRITE)
        data.write_bytes(b"bad")
        os.chmod(data, stat.S_IREAD)

        with pytest.raises(ArtifactRepositoryError, match="SHA-256"):
            await _read(repo, ref.artifact_id)

    asyncio.run(scenario())


@pytest.mark.parametrize("recovery", ["delete", "gc"])
def test_tombstone_cleanup_is_retryable(tmp_path, monkeypatch, recovery):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        original_unlink = os.unlink

        def block_descriptor(path, *args, **kwargs):
            if os.fspath(path).endswith("descriptor.json"):
                raise PermissionError(errno.EACCES, "transient failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "unlink", block_descriptor)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        assert await repo.stat(ref.artifact_id) is None
        assert list(tmp_path.glob(".delete-*"))
        assert list((tmp_path / ".retire-intents").glob("*.json"))

        repo = FilesystemArtifactRepository(tmp_path, 3)
        if recovery == "gc":
            assert await repo.collect_orphans([], datetime.now(timezone.utc)) == 1
        else:
            await repo.delete(ref.artifact_id)
        assert not list(tmp_path.glob(".delete-*"))
        assert not list((tmp_path / ".retire-intents").iterdir())

    asyncio.run(scenario())


def test_delete_rolls_back_a_replacement_captured_at_rename(tmp_path, monkeypatch):
    async def scenario():
        root = tmp_path / "repo"
        replacement_root = tmp_path / "replacement"
        repo = FilesystemArtifactRepository(root, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", name="old"))
        replacement_root.mkdir()
        shutil.copytree(
            _object(root, ref.artifact_id), _object(replacement_root, ref.artifact_id)
        )
        replacement = ref
        object_path = _object(root, ref.artifact_id)
        actual_rename = os.rename

        def substitute(source, target, *args, **kwargs):
            if os.path.basename(source) == object_path.name:
                actual_rename(object_path, tmp_path / "displaced")
                actual_rename(_object(replacement_root, ref.artifact_id), object_path)
            return actual_rename(source, target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", substitute)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        assert await repo.stat(ref.artifact_id) == replacement
        assert await _read(repo, ref.artifact_id) == b"abc"

    asyncio.run(scenario())


def test_gc_preserves_an_object_refreshed_immediately_before_capture(tmp_path, monkeypatch):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        final = _object(tmp_path, ref.artifact_id)
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        old = int((cutoff - timedelta(days=1)).timestamp())
        os.utime(final, (old, old))
        actual_rename = os.rename

        def refresh_lease(source, target, *args, **kwargs):
            if os.path.basename(source) == final.name:
                os.utime(final, None)
            return actual_rename(source, target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", refresh_lease)
            assert await repo.collect_orphans([], cutoff) == 0
        assert await repo.stat(ref.artifact_id) == ref
        assert _contents(tmp_path) == {final}

    asyncio.run(scenario())


def test_failed_rollback_does_not_allow_gc_to_remove_the_replacement(tmp_path, monkeypatch):
    async def scenario():
        root = tmp_path / "repo"
        other_root = tmp_path / "replacement"
        repo = FilesystemArtifactRepository(root, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", name="old"))
        other_root.mkdir()
        shutil.copytree(_object(root, ref.artifact_id), _object(other_root, ref.artifact_id))
        replacement = ref
        actual_rename = os.rename

        def substitute(source, target, *args, **kwargs):
            source_name = os.path.basename(source)
            if source_name == ref.artifact_id[7:]:
                final = _object(root, ref.artifact_id)
                actual_rename(final, tmp_path / "displaced")
                actual_rename(_object(other_root, ref.artifact_id), final)
            elif source_name.startswith(".delete-"):
                raise PermissionError(errno.EACCES, "temporarily unable to restore")
            return actual_rename(source, target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", substitute)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        assert await repo.collect_orphans([], datetime.now(timezone.utc) - timedelta(days=1)) == 0
        assert await repo.stat(ref.artifact_id) is None
        (tombstone,) = root.glob(".delete-*")
        assert (tombstone / "data").read_bytes() == b"abc"
        assert replacement.name.encode() in (tombstone / "descriptor.json").read_bytes()

    asyncio.run(scenario())


def test_cancel_during_publication_waits_for_rename_and_leaves_collectible_object(
    tmp_path, monkeypatch
):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        entered = threading.Event()
        release = threading.Event()
        actual_rename = os.rename

        def pause_rename(source, target, *args, **kwargs):
            if os.path.basename(source).startswith(".put-"):
                entered.set()
                assert release.wait(timeout=15)
            return actual_rename(source, target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", pause_rename)
            pending = asyncio.create_task(repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain")))
            try:
                assert await asyncio.to_thread(entered.wait, 10)
                pending.cancel()
                await asyncio.sleep(0)
                assert not pending.done()
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        (published,) = _contents(tmp_path)
        assert await _read(repo, f"sha256:{published.name}") == b"abc"
        assert await repo.collect_orphans([], datetime.now(timezone.utc) + timedelta(days=1)) == 1

    asyncio.run(scenario())


def test_cleanup_does_not_follow_tombstone_directory_substitution(tmp_path, monkeypatch):
    async def scenario():
        root = tmp_path / "repository"
        outside = tmp_path / "outside"
        outside.mkdir()
        victim = outside / "data"
        victim.write_bytes(b"keep")
        repo = FilesystemArtifactRepository(root, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        original_unlink = os.unlink

        def substitute_before_unlink(path, *args, **kwargs):
            tombstone = next(root.glob(".delete-*"), None)
            if tombstone and os.path.basename(path) == "data":
                tombstone.rename(tmp_path / "displaced")
                _junction(tombstone, outside)
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "unlink", substitute_before_unlink)
            try:
                await repo.delete(ref.artifact_id)
            except ArtifactRepositoryError:
                pass
        assert victim.read_bytes() == b"keep"

    asyncio.run(scenario())


def test_publication_and_failure_cleanup_do_not_run_on_event_loop(tmp_path, monkeypatch):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        real_rename = os.rename
        real_unlink = os.unlink
        loop_thread = threading.get_ident()
        rename_threads = []
        unlink_threads = []

        def rename(source, target, *args, **kwargs):
            rename_threads.append(threading.get_ident())
            return real_rename(source, target, *args, **kwargs)

        def unlink(path, *args, **kwargs):
            unlink_threads.append(threading.get_ident())
            return real_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", rename)
            patch.setattr(os, "unlink", unlink)
            await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
            with pytest.raises(ValueError):
                await repo.put(ArtifactWrite(_chunks(b"x"), "text/plain", expected_size_bytes=2))
        assert rename_threads and loop_thread not in rename_threads
        assert unlink_threads and loop_thread not in unlink_threads

    asyncio.run(scenario())


def test_independent_publication_proceeds_during_delete_of_an_old_id(tmp_path, monkeypatch):
    child = """
import asyncio
import sys
from pathlib import Path
from src.adapters.artifacts.filesystem import FilesystemArtifactRepository
from src.core.artifacts import ArtifactWrite

async def chunks():
    yield b"abc"

class SignalingRepository(FilesystemArtifactRepository):
    def _publish(self, *args, **kwargs):
        print("waiting", flush=True)
        return super()._publish(*args, **kwargs)

async def main():
    repo = SignalingRepository(Path(sys.argv[1]), 3)
    await repo.put(ArtifactWrite(chunks(), "text/plain", name="initial"))
    print("published", flush=True)

asyncio.run(main())
"""

    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", name="initial"))
        real_rename = os.rename
        observed = []
        processes = []

        def pause_delete(source, target, *args, **kwargs):
            if os.path.basename(source) == ref.artifact_id[7:]:
                process = subprocess.Popen(
                    [sys._base_executable, "-c", child, str(tmp_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append(process)
                assert process.stdout is not None
                assert process.stdout.readline().strip() == "waiting"
                try:
                    observed.append(process.wait(timeout=0.8))
                except subprocess.TimeoutExpired:
                    observed.append("blocked")
            return real_rename(source, target, *args, **kwargs)

        try:
            with monkeypatch.context() as patch:
                patch.setattr(os, "rename", pause_delete)
                await repo.delete(ref.artifact_id)
            assert observed == [0]
            out, error = processes[0].communicate(timeout=10)
            assert processes[0].returncode == 0, error
            assert out.strip() == "published"
            assert await repo.stat(ref.artifact_id) is None
            (remaining,) = _contents(tmp_path)
            assert (await repo.stat(f"sha256:{remaining.name}")).name == "initial"
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    asyncio.run(scenario())


@pytest.mark.parametrize("recovery", ["delete", "gc"])
def test_forged_retired_member_cannot_authorize_cleanup_after_failed_rollback(
    tmp_path, monkeypatch, recovery
):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        final = _object(tmp_path, ref.artifact_id)
        original_rename = os.rename

        def insert_marker(source, target, *args, **kwargs):
            if os.path.basename(source) == final.name:
                (final / ".retired").touch()
            elif os.path.basename(source).startswith(".delete-"):
                raise PermissionError(errno.EACCES, "injected rollback failure")
            return original_rename(source, target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", insert_marker)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        (tombstone,) = tmp_path.glob(".delete-*")

        restarted = FilesystemArtifactRepository(tmp_path, 3)
        if recovery == "gc":
            assert await restarted.collect_orphans([], datetime.now(timezone.utc)) == 0
        else:
            with pytest.raises(ArtifactRepositoryError):
                await restarted.delete(ref.artifact_id)
        assert (tombstone / ".retired").is_file()
        assert (tombstone / "data").read_bytes() == b"abc"
        assert (tombstone / "descriptor.json").is_file()
        assert list((tmp_path / ".retire-intents").glob("*.json"))

    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["delete", "gc"])
def test_retirement_record_write_failure_leaves_the_live_object_untouched(
    tmp_path, monkeypatch, operation
):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        original_open = Directory.open

        def fail_intent_write(directory, name, flags):
            if flags & os.O_CREAT and (
                directory.path.name == ".retire-intents" or name == ".retired"
            ):
                raise PermissionError(errno.EACCES, "injected retirement record failure")
            return original_open(directory, name, flags)

        cutoff = datetime.now(timezone.utc) + timedelta(days=1)
        with monkeypatch.context() as patch:
            patch.setattr(Directory, "open", fail_intent_write)
            if operation == "delete":
                with pytest.raises(ArtifactRepositoryError):
                    await repo.delete(ref.artifact_id)
            else:
                assert await repo.collect_orphans([], cutoff) == 0
        assert await repo.stat(ref.artifact_id) == ref
        assert await _read(repo, ref.artifact_id) == b"abc"
        assert not list(tmp_path.glob(".delete-*"))

        restarted = FilesystemArtifactRepository(tmp_path, 3)
        if operation == "delete":
            await restarted.delete(ref.artifact_id)
        else:
            assert await restarted.collect_orphans([], cutoff) == 1
        assert await restarted.stat(ref.artifact_id) is None

    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["malformed", "reparse"])
def test_restart_does_not_clean_tombstone_if_its_intent_is_untrusted(tmp_path, monkeypatch, damage):
    async def scenario():
        root = tmp_path / "repository"
        repo = FilesystemArtifactRepository(root, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        original_unlink = os.unlink

        def fail_cleanup(path, *args, **kwargs):
            if os.path.basename(path) == "data":
                raise PermissionError(errno.EACCES, "injected cleanup failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "unlink", fail_cleanup)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        (tombstone,) = root.glob(".delete-*")
        (intent,) = (root / ".retire-intents").glob("*.json")
        if damage == "malformed":
            os.chmod(intent, stat.S_IWRITE)
            intent.write_bytes(b'{"unexpected":1}')
        else:
            outside = tmp_path / "outside"
            outside.mkdir()
            intent.rename(outside / "original")
            _junction(intent, outside)

        restarted = FilesystemArtifactRepository(root, 3)
        assert await restarted.collect_orphans([], datetime.now(timezone.utc)) == 0
        with pytest.raises(ArtifactRepositoryError) as caught:
            await restarted.delete(ref.artifact_id)
        assert str(root) not in str(caught.value)
        assert (tombstone / "data").read_bytes() == b"abc"
        assert (tombstone / "descriptor.json").is_file()

    asyncio.run(scenario())


def test_restart_rechecks_the_captured_gc_lease_before_cleanup(tmp_path, monkeypatch):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        cutoff = datetime.now(timezone.utc) - timedelta(days=1)
        old = int((cutoff - timedelta(days=1)).timestamp())
        final = _object(tmp_path, ref.artifact_id)
        os.utime(final, (old, old))
        original_sync = Directory.sync
        error_injected = False

        def fail_capture_sync(directory):
            nonlocal error_injected
            if directory.path == tmp_path and list(tmp_path.glob(".delete-*")):
                error_injected = True
                raise OSError(errno.EIO, "simulated interrupted capture")
            return original_sync(directory)

        with monkeypatch.context() as patch:
            patch.setattr(Directory, "sync", fail_capture_sync)
            assert await repo.collect_orphans([], cutoff) == 0
        assert error_injected
        (tombstone,) = tmp_path.glob(".delete-*")
        os.utime(tombstone, None)

        restarted = FilesystemArtifactRepository(tmp_path, 3)
        assert await restarted.collect_orphans([], cutoff) == 0
        assert await restarted.stat(ref.artifact_id) == ref
        assert not list(tmp_path.glob(".delete-*"))
        assert not list((tmp_path / ".retire-intents").iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize("final_identity", ["original", "replacement", "missing"])
def test_failed_rename_intent_is_cleared_only_for_the_same_live_object(
    tmp_path, monkeypatch, final_identity
):
    async def scenario():
        root = tmp_path / "repository"
        repo = FilesystemArtifactRepository(root, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", name="first"))
        final = _object(root, ref.artifact_id)
        real_rename = os.rename

        def fail_final_rename(source, target, *args, **kwargs):
            if os.path.basename(source) == final.name:
                raise OSError(errno.EIO, "injected capture failure")
            return real_rename(source, target, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(os, "rename", fail_final_rename)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        assert not list(root.glob(".delete-*"))
        (intent,) = (root / ".retire-intents").glob("*.json")

        expected = ref
        if final_identity != "original":
            final.rename(tmp_path / "original")
            expected = None
            if final_identity == "replacement":
                    shutil.copytree(tmp_path / "original", final)
                    expected = ref

        restarted = FilesystemArtifactRepository(root, 3)
        assert await restarted.collect_orphans({ref.artifact_id}, datetime.now(timezone.utc)) == 0
        assert intent.exists() == (final_identity != "original")
        assert await restarted.stat(ref.artifact_id) == expected
        if final_identity != "original":
            with pytest.raises(ArtifactRepositoryError):
                await restarted.delete(ref.artifact_id)
            assert intent.exists()
            assert await restarted.stat(ref.artifact_id) == expected

    asyncio.run(scenario())


def test_restart_recovers_when_persisting_cleanup_permission_fails(tmp_path, monkeypatch):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        ref = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain"))
        original_replace = Directory.replace
        error_injected = False

        def fail_intent_replace(directory, source, target):
            nonlocal error_injected
            if directory.path.name == ".retire-intents":
                error_injected = True
                raise OSError(errno.EIO, "injected intent update failure")
            return original_replace(directory, source, target)

        with monkeypatch.context() as patch:
            patch.setattr(Directory, "replace", fail_intent_replace)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(ref.artifact_id)
        assert error_injected
        (tombstone,) = tmp_path.glob(".delete-*")
        assert (tombstone / "data").read_bytes() == b"abc"
        assert (tombstone / "descriptor.json").is_file()

        restarted = FilesystemArtifactRepository(tmp_path, 3)
        await restarted.delete(ref.artifact_id)
        assert not tombstone.exists()
        assert not list((tmp_path / ".retire-intents").iterdir())

    asyncio.run(scenario())


@pytest.mark.parametrize("recovery", ["delete", "gc"])
@pytest.mark.parametrize("republish_first", [False, True])
def test_restart_finishes_retirement_after_tombstone_removal(
    tmp_path, monkeypatch, recovery, republish_first
):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        original = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", name="original"))
        control = tmp_path / ".retire-intents"
        original_unlink = Directory.unlink
        failed_once = False

        def fail_first_record_removal(directory, name, info):
            nonlocal failed_once
            if directory.path == control and name.endswith(".json") and not failed_once:
                failed_once = True
                raise PermissionError(errno.EACCES, "injected intent unlink failure")
            return original_unlink(directory, name, info)

        with monkeypatch.context() as patch:
            patch.setattr(Directory, "unlink", fail_first_record_removal)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(original.artifact_id)
        assert failed_once
        assert not list(tmp_path.glob(".delete-*"))
        (old_record,) = control.glob("*.json")
        assert await repo.stat(original.artifact_id) is None

        restarted = FilesystemArtifactRepository(tmp_path, 3)

        async def put_replacement():
            return await restarted.put(
                ArtifactWrite(_chunks(b"abc"), "text/plain", name="replacement")
            )

        replacement = await put_replacement() if republish_first else None
        if recovery == "delete":
            await restarted.delete(original.artifact_id)
            assert await restarted.stat(original.artifact_id) is None
        else:
            references = {replacement.artifact_id} if replacement is not None else set()
            assert await restarted.collect_orphans(references, datetime.now(timezone.utc)) == 0
            assert await restarted.stat(original.artifact_id) is None
            if replacement is not None:
                assert await restarted.stat(replacement.artifact_id) == replacement
        assert not old_record.exists()

        if replacement is None or recovery == "delete":
            replacement = await put_replacement()
        assert await _read(restarted, replacement.artifact_id) == b"abc"
        if recovery == "delete":
            await restarted.delete(replacement.artifact_id)
        else:
            cutoff = datetime.now(timezone.utc) - timedelta(days=1)
            old = int((cutoff - timedelta(days=1)).timestamp())
            os.utime(_object(tmp_path, replacement.artifact_id), (old, old))
            assert await restarted.collect_orphans([], cutoff) == 1
        assert await restarted.stat(original.artifact_id) is None
        assert not list(control.iterdir())

    asyncio.run(scenario())


def test_repeated_record_removal_failure_keeps_descriptor_retirements_isolated(
    tmp_path, monkeypatch
):
    async def scenario():
        repo = FilesystemArtifactRepository(tmp_path, 3)
        original = await repo.put(ArtifactWrite(_chunks(b"abc"), "text/plain", name="original"))
        control = tmp_path / ".retire-intents"
        original_unlink = Directory.unlink

        def fail_record_removal(directory, name, info):
            if directory.path == control and name.endswith(".json"):
                raise PermissionError(errno.EACCES, "persistent failure", str(control))
            return original_unlink(directory, name, info)

        with monkeypatch.context() as patch:
            patch.setattr(Directory, "unlink", fail_record_removal)
            with pytest.raises(ArtifactRepositoryError):
                await repo.delete(original.artifact_id)
            (old_record,) = control.glob("*.json")
            restarted = FilesystemArtifactRepository(tmp_path, 3)
            replacement = await restarted.put(
                ArtifactWrite(_chunks(b"abc"), "text/plain", name="replacement")
            )
            with pytest.raises(ArtifactRepositoryError) as caught:
                await restarted.delete(replacement.artifact_id)
            assert str(control) not in str(caught.value)
            assert await restarted.collect_orphans([], datetime.now(timezone.utc)) == 0
            assert old_record.exists()
            assert not list(tmp_path.glob(".delete-*"))
            assert await restarted.stat(replacement.artifact_id) is None

        await restarted.delete(replacement.artifact_id)
        await restarted.delete(original.artifact_id)
        assert await restarted.stat(replacement.artifact_id) is None
        assert not list(control.iterdir())

    asyncio.run(scenario())
