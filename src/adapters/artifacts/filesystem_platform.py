from __future__ import annotations

import errno
import os
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_REPARSE = stat.FILE_ATTRIBUTE_REPARSE_POINT


class ArtifactRepositoryError(RuntimeError):
    """A storage failure or an invalid on-disk artifact."""


def same_entry(current: os.stat_result, expected: os.stat_result) -> None:
    if not os.path.samestat(current, expected):
        raise ArtifactRepositoryError("Artifact filesystem entry changed")


def directory_info(info: os.stat_result) -> os.stat_result:
    if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & _REPARSE:
        raise ArtifactRepositoryError("Invalid artifact directory")
    return info


def regular_file(info: os.stat_result) -> os.stat_result:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or getattr(info, "st_file_attributes", 0) & _REPARSE
    ):
        raise ArtifactRepositoryError("Invalid artifact file")
    return info


def same_file(current: os.stat_result, expected: os.stat_result) -> None:
    regular_file(current)
    same_entry(current, expected)
    if (
        current.st_size != expected.st_size
        or current.st_mtime_ns != expected.st_mtime_ns
        or current.st_mode != expected.st_mode
    ):
        raise ArtifactRepositoryError("Artifact file changed")


if sys.platform == "win32":

    def _windows_open(path: Path, flags: int, *, directory: bool = False) -> int:
        import _winapi
        import msvcrt

        generic_read = 0x80000000
        generic_write = 0x40000000
        file_write_attributes = 0x100
        share_read_write = 0x03
        open_existing = 3
        open_always = 4
        create_new = 1
        open_reparse_point = 0x00200000
        backup_semantics = 0x02000000

        access = generic_read
        if flags & (os.O_WRONLY | os.O_RDWR):
            access |= generic_write
        if directory:
            access = generic_read | (file_write_attributes if flags & os.O_RDWR else 0)
        disposition = open_existing
        if flags & os.O_CREAT:
            disposition = create_new if flags & os.O_EXCL else open_always
        attributes = open_reparse_point | (backup_semantics if directory else 0)
        # Denying FILE_SHARE_DELETE pins the opened directory or leaf against rename.
        handle = _winapi.CreateFile(
            str(path), access, share_read_write, 0, disposition, attributes, 0
        )
        try:
            return msvcrt.open_osfhandle(
                handle, flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR) | _BINARY
            )
        except BaseException:
            _winapi.CloseHandle(handle)
            raise

    def _windows_touch(fd: int) -> None:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        set_file_time = ctypes.WinDLL("kernel32", use_last_error=True).SetFileTime
        filetime_pointer = ctypes.POINTER(wintypes.FILETIME)
        set_file_time.argtypes = (
            wintypes.HANDLE,
            filetime_pointer,
            filetime_pointer,
            filetime_pointer,
        )
        set_file_time.restype = wintypes.BOOL
        ticks = time.time_ns() // 100 + 116_444_736_000_000_000
        modified = wintypes.FILETIME(ticks & 0xFFFFFFFF, ticks >> 32)
        if not set_file_time(msvcrt.get_osfhandle(fd), None, None, ctypes.byref(modified)):
            raise ctypes.WinError(ctypes.get_last_error())


class Directory:
    """An open directory used as an anchor for operations on immediate children."""

    def __init__(self, path: Path, fd: int) -> None:
        self.path = path
        self.fd = fd
        self._relative = None if sys.platform == "win32" else fd

    @property
    def info(self) -> os.stat_result:
        return os.fstat(self.fd)

    def _leaf(self, name: str) -> str | Path:
        if self._relative is not None:
            return name
        same_entry(directory_info(self.path.lstat()), self.info)
        return self.path / name

    def stat(self, name: str) -> os.stat_result:
        return os.stat(self._leaf(name), dir_fd=self._relative, follow_symlinks=False)

    @contextmanager
    def child(
        self, name: str, expected: os.stat_result, *, writable: bool = False
    ) -> Iterator[Directory]:
        path = self.path / name
        same_entry(directory_info(self.stat(name)), expected)
        if sys.platform == "win32":
            fd = _windows_open(path, os.O_RDWR if writable else os.O_RDONLY, directory=True)
        else:
            fd = os.open(name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW, dir_fd=self.fd)
        try:
            same_entry(directory_info(os.fstat(fd)), expected)
            same_entry(directory_info(self.stat(name)), expected)
            yield Directory(path, fd)
        finally:
            os.close(fd)

    def open(self, name: str, flags: int) -> int:
        leaf = self._leaf(name)
        if sys.platform == "win32":
            return _windows_open(Path(leaf), flags)
        return os.open(leaf, flags | _NOFOLLOW, 0o600, dir_fd=self._relative)

    @contextmanager
    def read(self, name: str, expected: os.stat_result) -> Iterator[int]:
        same_file(self.stat(name), expected)
        fd = self.open(name, os.O_RDONLY)
        try:
            same_file(os.fstat(fd), expected)
            same_file(self.stat(name), expected)
            yield fd
        finally:
            os.close(fd)

    def mkdir(self, name: str) -> None:
        os.mkdir(self._leaf(name), mode=0o700, dir_fd=self._relative)

    def rename(self, source: str, target: str) -> None:
        os.rename(
            self._leaf(source),
            self._leaf(target),
            src_dir_fd=self._relative,
            dst_dir_fd=self._relative,
        )

    def rmdir(self, name: str) -> None:
        os.rmdir(self._leaf(name), dir_fd=self._relative)

    @contextmanager
    def entries(self) -> Iterator[Iterator[os.DirEntry[str]]]:
        if sys.platform == "win32":
            same_entry(directory_info(self.path.lstat()), self.info)
        with os.scandir(self.path if self._relative is None else self.fd) as entries:
            yield entries

    def set_mode(self, name: str, fd: int, mode: int) -> None:
        if sys.platform == "win32":
            same_entry(regular_file(self.stat(name)), regular_file(os.fstat(fd)))
            # This fd denies DELETE sharing, so chmod cannot see a replacement.
            os.chmod(self._leaf(name), mode)
        else:
            os.fchmod(fd, mode)

    def unlink(self, name: str, expected: os.stat_result) -> None:
        same_file(self.stat(name), expected)
        if sys.platform == "win32" and not expected.st_mode & stat.S_IWUSR:
            with self.read(name, expected) as fd:
                self.set_mode(name, fd, stat.S_IWRITE)
        os.unlink(self._leaf(name), dir_fd=self._relative)

    def sync(self) -> None:
        if sys.platform != "win32":
            os.fsync(self.fd)

    def refresh(self) -> None:
        if sys.platform == "win32":
            _windows_touch(self.fd)
        else:
            os.utime(self.fd, None)
        self.sync()


@contextmanager
def pin_root(path: Path, expected: os.stat_result) -> Iterator[Directory]:
    same_entry(directory_info(path.lstat()), expected)
    if path.resolve(strict=True) != path:
        raise ArtifactRepositoryError("Artifact repository root changed")
    if sys.platform == "win32":
        fd = _windows_open(path, os.O_RDONLY, directory=True)
    else:
        fd = os.open(path, os.O_RDONLY | _DIRECTORY | _NOFOLLOW)
    try:
        same_entry(directory_info(os.fstat(fd)), expected)
        same_entry(directory_info(path.lstat()), expected)
        yield Directory(path, fd)
    finally:
        os.close(fd)


def _lock_fd(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                time.sleep(0.01)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


@contextmanager
def object_lock(directory: Directory, digest: str) -> Iterator[None]:
    # Never unlink a lock file. An already waiting process may hold its inode.
    try:
        expected = regular_file(directory.stat(digest))
    except FileNotFoundError:
        expected = None
    fd = directory.open(digest, os.O_RDWR | os.O_CREAT)
    try:
        actual = regular_file(os.fstat(fd))
        if expected is not None:
            same_entry(actual, expected)
        same_entry(regular_file(directory.stat(digest)), actual)
        _lock_fd(fd)
        same_entry(regular_file(directory.stat(digest)), actual)
        yield
    finally:
        # Closing the only descriptor releases both flock and Windows byte locks.
        os.close(fd)
