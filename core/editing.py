import errno
import json
import os
import re
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import config
from core import rebase
from core.ablation import abliteration_direction, effective_coeffs

EDITS_DIR = config.DATA_DIR / "edits"
PRESETS_DIR = config.DATA_DIR / "presets"

DEFAULT_TEMP_STALE_AGE = 24 * 60 * 60
_TEMP_STAGE_RE = re.compile(r"^\..+\.tmp-[0-9a-f]{32}$")
_TEMP_GGUF_RE = re.compile(r"^\..+\.tmp-[0-9a-f]{32}\.gguf$")
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_STAT_SUPPORTS_DIR_FD = os.stat in os.supports_dir_fd
_STAT_SUPPORTS_NOFOLLOW = os.stat in os.supports_follow_symlinks


def _mark_windows_file_for_deletion(file):
    """Mark the exact open Windows file handle for deletion on close."""
    import ctypes
    import msvcrt

    class FILE_DISPOSITION_INFO(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    info = FILE_DISPOSITION_INFO(1)
    handle = msvcrt.get_osfhandle(file.fileno())
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.SetFileInformationByHandle(
        ctypes.c_void_p(handle), 4, ctypes.byref(info), ctypes.sizeof(info)
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _open_windows_lease(path, *, create):
    """Open a lease with DELETE access so disposal stays handle-bound."""
    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = ctypes.c_void_p
    access = 0x80000000 | 0x40000000 | 0x00010000  # read, write, delete
    sharing = 0x1 | 0x2 | 0x4
    flags = 0x00200000  # FILE_FLAG_OPEN_REPARSE_POINT

    def open_handle(disposition):
        handle = kernel32.CreateFileW(
            str(path), access, sharing, None, disposition, flags, None,
        )
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return msvcrt.open_osfhandle(handle, os.O_RDWR)

    if create:
        try:
            return open_handle(1), True  # CREATE_NEW
        except FileExistsError:
            pass
    return open_handle(3), False  # OPEN_EXISTING


def _validate_windows_lease_file(file):
    """Return the structural identity of an ordinary disk-file handle."""
    import ctypes
    import msvcrt

    class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", ctypes.c_uint32),
            ("ftCreationTimeLow", ctypes.c_uint32),
            ("ftCreationTimeHigh", ctypes.c_uint32),
            ("ftLastAccessTimeLow", ctypes.c_uint32),
            ("ftLastAccessTimeHigh", ctypes.c_uint32),
            ("ftLastWriteTimeLow", ctypes.c_uint32),
            ("ftLastWriteTimeHigh", ctypes.c_uint32),
            ("dwVolumeSerialNumber", ctypes.c_uint32),
            ("nFileSizeHigh", ctypes.c_uint32),
            ("nFileSizeLow", ctypes.c_uint32),
            ("nNumberOfLinks", ctypes.c_uint32),
            ("nFileIndexHigh", ctypes.c_uint32),
            ("nFileIndexLow", ctypes.c_uint32),
        ]

    handle = msvcrt.get_osfhandle(file.fileno())
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if kernel32.GetFileType(ctypes.c_void_p(handle)) != 1:  # FILE_TYPE_DISK
        raise RuntimeError("temporary artifact lease is not a normal disk file")
    info = BY_HANDLE_FILE_INFORMATION()
    if not kernel32.GetFileInformationByHandle(
        ctypes.c_void_p(handle), ctypes.byref(info)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    unsafe = 0x10 | 0x400  # FILE_ATTRIBUTE_DIRECTORY | REPARSE_POINT
    if info.dwFileAttributes & unsafe:
        raise RuntimeError("temporary artifact lease is a directory or reparse point")
    return (
        info.dwVolumeSerialNumber,
        (info.nFileIndexHigh << 32) | info.nFileIndexLow,
        info.dwFileAttributes & unsafe,
    )


def _validate_acquired_lease_file(file):
    """Final fallible lease validation, kept separate for deterministic tests."""
    if os.name == "nt":
        return _validate_windows_lease_file(file)
    current = os.fstat(file.fileno())
    if _stat_is_link_or_reparse(current) or not stat.S_ISREG(current.st_mode):
        raise RuntimeError("temporary artifact lease is not a safe regular file")
    return _stat_identity(current)


def internal_temp_leaf_kind(name):
    """Classify exact J-Wash temporary/lease leaf names."""
    artifact_name = name[:-6] if name.endswith(".lease") else name
    if _TEMP_STAGE_RE.fullmatch(artifact_name):
        return "directory-lease" if name.endswith(".lease") else "directory"
    if _TEMP_GGUF_RE.fullmatch(artifact_name):
        return "file-lease" if name.endswith(".lease") else "file"
    return None


class ArtifactLease:
    """Exclusive advisory lease for one temporary export artifact.

    POSIX lease files are deliberately persistent.  Portable POSIX APIs cannot
    condition an unlink on the inode previously inspected, so removing a lease
    during release could delete an unrelated replacement.  Unlocked retained
    leases are safe and reusable.
    """

    def __init__(self, artifact, *, create=True):
        self.artifact = Path(artifact)
        self.path = self.artifact.with_name(self.artifact.name + ".lease")
        self.create = create
        self._file = None
        self._parent_fd = None
        self._identity = None
        self._created = False

    def acquire(self, *, blocking=True):
        created = False
        parent_fd = None
        fd = None
        file = None
        try:
            if os.name == "nt":
                fd, created = _open_windows_lease(self.path, create=self.create)
            else:
                if not hasattr(os, "O_NOFOLLOW"):
                    raise RuntimeError("safe no-follow POSIX lease acquisition is unavailable")
                parent_chain = _snapshot_directory_chain(_absolute_no_follow(self.path.parent))
                parent_fd = _open_verified_directory_chain(parent_chain)
                open_path = self.path.name
                common_flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
                if self.create:
                    try:
                        fd = os.open(
                            open_path, common_flags | os.O_CREAT | os.O_EXCL,
                            0o600, dir_fd=parent_fd,
                        )
                        created = True
                        created_stat = os.fstat(fd)
                        if (
                            _stat_is_link_or_reparse(created_stat)
                            or not stat.S_ISREG(created_stat.st_mode)
                        ):
                            raise RuntimeError(
                                "new temporary artifact lease is not a safe regular file"
                            )
                    except FileExistsError:
                        before = os.stat(
                            open_path, dir_fd=parent_fd, follow_symlinks=False,
                        )
                        if _stat_is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
                            raise RuntimeError("temporary artifact lease is not a safe regular file")
                        fd = os.open(open_path, common_flags, dir_fd=parent_fd)
                        opened = os.fstat(fd)
                        if (
                            _stat_is_link_or_reparse(opened)
                            or not stat.S_ISREG(opened.st_mode)
                            or _candidate_identity(opened) != _candidate_identity(before)
                        ):
                            raise RuntimeError("temporary artifact lease changed during acquisition")
                else:
                    before = os.stat(
                        open_path, dir_fd=parent_fd, follow_symlinks=False,
                    )
                    if _stat_is_link_or_reparse(before) or not stat.S_ISREG(before.st_mode):
                        raise RuntimeError("temporary artifact lease is not a safe regular file")
                    fd = os.open(open_path, common_flags, dir_fd=parent_fd)
                    opened = os.fstat(fd)
                    if (
                        _stat_is_link_or_reparse(opened)
                        or not stat.S_ISREG(opened.st_mode)
                        or _candidate_identity(opened) != _candidate_identity(before)
                    ):
                        raise RuntimeError("temporary artifact lease changed during acquisition")
            file = os.fdopen(fd, "r+b")
            fd = None
            if os.name == "nt":
                _validate_windows_lease_file(file)
            if os.name == "nt":
                file.seek(0, os.SEEK_END)
            if os.name == "nt" and file.tell() == 0:
                file.write(b"0")
                file.flush()
            if os.name == "nt":
                file.seek(0)
            if os.name == "nt":
                import msvcrt
                flag = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                try:
                    msvcrt.locking(file.fileno(), flag, 1)
                except OSError as exc:
                    if not blocking and exc.errno in (
                        errno.EACCES, errno.EAGAIN, errno.EDEADLK,
                    ):
                        file.close()
                        file = None
                        return False
                    raise
            else:
                import fcntl
                flag = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(file.fileno(), flag)
                except BlockingIOError:
                    file.close()
                    file = None
                    return False
            identity = _validate_acquired_lease_file(file)
            # POSIX release never removes the persistent lease, so it does not
            # retain a parent descriptor for the export lifetime.
            if parent_fd is not None:
                os.close(parent_fd)
                parent_fd = None
            self._parent_fd = None
            # Publish object state only after every fallible validation succeeds.
            self._file = file
            self._identity = identity
            self._created = created
            return True
        except Exception:
            if file is not None and not file.closed:
                if created and os.name == "nt":
                    try:
                        _mark_windows_file_for_deletion(file)
                    except OSError:
                        pass
                file.close()
            elif fd is not None:
                os.close(fd)
            raise
        finally:
            if parent_fd is not None:
                os.close(parent_fd)

    def release(self, *, remove=True):
        file, self._file = self._file, None
        parent_fd, self._parent_fd = self._parent_fd, None
        _, self._identity = self._identity, None
        created, self._created = self._created, False
        unlock_error = None
        try:
            if file is not None:
                try:
                    if os.name == "nt":
                        import msvcrt
                        file.seek(0)
                        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
                except Exception as exc:
                    unlock_error = exc
                try:
                    if remove and created and os.name == "nt":
                        # Windows deletion is bound to the owned open handle.
                        _mark_windows_file_for_deletion(file)
                except OSError:
                    pass
                finally:
                    file.close()
        finally:
            if parent_fd is not None:
                try:
                    os.close(parent_fd)
                except OSError:
                    pass
        if unlock_error is not None:
            raise unlock_error

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"temporary artifact lease is already held: {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.release()
        except Exception:
            if exc is None:
                raise
        return False


def artifact_lease(artifact):
    return ArtifactLease(artifact)


def _is_link_or_reparse(path, path_stat=None):
    path_stat = path.lstat() if path_stat is None else path_stat
    return (
        _stat_is_link_or_reparse(path_stat)
        or getattr(path, "is_junction", lambda: False)()
    )


def _stat_is_link_or_reparse(path_stat):
    """Classify links from no-follow stat data without another path lookup."""
    return stat.S_ISLNK(path_stat.st_mode) or bool(
        getattr(path_stat, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _recognized_temp(path):
    """Return the owned temporary kind without following a symlink."""
    try:
        if _is_link_or_reparse(path):
            return None
        if internal_temp_leaf_kind(path.name) == "directory" and path.is_dir():
            return "directory"
        if internal_temp_leaf_kind(path.name) == "file" and path.is_file():
            return "file"
    except OSError:
        raise
    return None


def _stat_identity(value):
    return (
        value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode),
        getattr(value, "st_file_attributes", 0),
        getattr(value, "st_reparse_tag", 0),
    )


def _candidate_identity(value):
    return _stat_identity(value) + (
        getattr(value, "st_ctime_ns", None), getattr(value, "st_mtime_ns", None),
        value.st_size,
    )


def _candidate_structural_identity(value):
    """Return identity fields that ordinary in-place export writes cannot change."""
    return _stat_identity(value)


def _candidate_identity_matches(value, expected, *, full):
    """Compare either structural-only or complete discovery identity."""
    if full:
        return _candidate_identity(value) == expected
    return _candidate_structural_identity(value) == expected[:5]


def _absolute_no_follow(path):
    """Return an absolute normalized spelling without resolving any links."""
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _path_chain(path):
    anchor = Path(path.anchor)
    current = anchor
    chain = [anchor]
    for part in path.parts[1:]:
        current = current / part
        chain.append(current)
    return chain


def _snapshot_directory_chain(path, *, missing_ok=False):
    snapshots = []
    for component in _path_chain(path):
        try:
            component_stat = component.lstat()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        if _is_link_or_reparse(component, component_stat):
            raise ValueError(
                f"cleanup root chain contains a symlink or junction/reparse point: {component}"
            )
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError(f"cleanup path component is not a directory: {component}")
        snapshots.append((component, _stat_identity(component_stat)))
    return tuple(snapshots)


def _revalidate_directory_chain(snapshots):
    for component, expected in snapshots:
        current = component.lstat()
        if _is_link_or_reparse(component, current) or _stat_identity(current) != expected:
            raise ValueError(f"cleanup path component changed: {component}")


def _open_verified_directory_chain(snapshots):
    """Open a POSIX directory chain without following links."""
    if os.name == "nt":
        raise _UnsafeAnchoredInspection(
            "safe handle-relative temporary inspection is unavailable on Windows"
        )
    flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    )
    opened = []
    try:
        fd = os.open(snapshots[0][0], flags)
        opened.append(fd)
        for index, (component, expected) in enumerate(snapshots):
            if index:
                fd = os.open(component.name, flags, dir_fd=fd)
                opened.append(fd)
            current = os.fstat(fd)
            if _stat_identity(current) != expected or not stat.S_ISDIR(current.st_mode):
                raise _UnsafeAnchoredInspection(f"inspection parent changed: {component}")
        keep = opened.pop()
        return keep
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _strict_inspection_capability():
    """Return whether metadata-preserving startup inspection is available."""
    required_flags = ("O_NOATIME", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if os.name != "posix":
        return False
    for name in required_flags:
        value = getattr(os, name, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value == 0:
            return False
    if os.scandir not in getattr(os, "supports_fd", ()):
        return False
    if not _OPEN_SUPPORTS_DIR_FD:
        return False
    if not _STAT_SUPPORTS_DIR_FD or not _STAT_SUPPORTS_NOFOLLOW:
        return False
    # CPython's POSIX scandir accepts an open directory descriptor.  Windows
    # does not, and is rejected above before any tree access.
    return True


def _scandir_inspection_fd(directory_fd):
    """Open an fd-based iterator or fail closed without a pathname fallback."""
    try:
        return os.scandir(directory_fd)
    except (TypeError, NotImplementedError) as exc:
        raise _UnsafeAnchoredInspection(
            "strict metadata-preserving temporary inspection is unavailable: "
            "fd-based os.scandir is not supported at runtime"
        ) from exc


def _inspection_directory_flags():
    if not _strict_inspection_capability():
        raise _UnsafeAnchoredInspection(
            "strict metadata-preserving temporary inspection is unavailable on this platform"
        )
    return (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NOATIME
    )


@dataclass(frozen=True)
class _TempCandidate:
    path: Path
    kind: str
    identity: tuple
    parent_chain: tuple
    anchor_index: int = 0


class _UnsafeAnchoredInspection(RuntimeError):
    """Raised when a no-follow inspection cannot establish the required safety."""


class _AnchoredInspectionParent:
    """Descriptor-relative, no-follow inspector for a temporary candidate."""

    def __init__(self, candidate):
        self.candidate = candidate
        self.parent_fd = None
        self._fds = []

    def __enter__(self):
        flags = _inspection_directory_flags()
        # The complete lexical chain is revalidated separately. Descriptor
        # traversal begins at the configured edits root so O_NOATIME does not
        # require ownership of unrelated ancestors such as ``/``.
        snapshots = self.candidate.parent_chain[self.candidate.anchor_index:]
        try:
            if not snapshots:
                raise _UnsafeAnchoredInspection("candidate has no validated parent chain")
            anchor, expected = snapshots[0]
            fd = os.open(anchor, flags)
            self._fds.append(fd)
            self._verify_fd(fd, expected, anchor)
            for component, expected in snapshots[1:]:
                next_fd = os.open(component.name, flags, dir_fd=fd)
                self._fds.append(next_fd)
                self._verify_fd(next_fd, expected, component)
                fd = next_fd
            self.parent_fd = fd
            self.stat_leaf(self.candidate.path.name, self.candidate.identity)
            return self
        except Exception as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, _UnsafeAnchoredInspection):
                raise
            raise _UnsafeAnchoredInspection(
                f"cannot safely anchor inspection parent: {exc}"
            ) from exc

    @staticmethod
    def _verify_fd(fd, expected, display):
        current = os.fstat(fd)
        if _stat_identity(current) != expected or not stat.S_ISDIR(current.st_mode):
            raise _UnsafeAnchoredInspection(f"inspection parent changed: {display}")

    def stat_leaf(self, leaf, expected_identity=None, *, full=False):
        current = os.stat(leaf, dir_fd=self.parent_fd, follow_symlinks=False)
        if _stat_is_link_or_reparse(current):
            raise _UnsafeAnchoredInspection(
                "temporary artifact changed or became a link/reparse point"
            )
        expected_type = stat.S_ISDIR if self.candidate.kind == "directory" else stat.S_ISREG
        if not expected_type(current.st_mode):
            raise _UnsafeAnchoredInspection("temporary artifact type changed")
        if expected_identity is not None:
            if not _candidate_identity_matches(current, expected_identity, full=full):
                detail = " changed after discovery" if full else " identity changed"
                raise _UnsafeAnchoredInspection("temporary artifact" + detail)
        return current

    def read_completion_marker(self, observer=None):
        """Read ``edit_meta.json`` through the retained candidate-parent fd."""
        if self.candidate.kind != "directory":
            raise _UnsafeAnchoredInspection("completion marker requires a directory artifact")
        flags = _inspection_directory_flags()
        candidate_fd = os.open(self.candidate.path.name, flags, dir_fd=self.parent_fd)
        try:
            candidate_stat = os.fstat(candidate_fd)
            if (
                _stat_is_link_or_reparse(candidate_stat)
                or _candidate_identity(candidate_stat) != self.candidate.identity
                or not stat.S_ISDIR(candidate_stat.st_mode)
            ):
                raise _UnsafeAnchoredInspection(
                    "temporary artifact changed during completion marker inspection"
                )
            if observer:
                observer("before_marker_open", self.candidate.path)
            marker_stat = os.stat(
                "edit_meta.json", dir_fd=candidate_fd, follow_symlinks=False,
            )
            if _stat_is_link_or_reparse(marker_stat) or not stat.S_ISREG(marker_stat.st_mode):
                raise ValueError("possible completion marker is not a safe regular file")
            marker_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NOATIME
            marker_fd = os.open("edit_meta.json", marker_flags, dir_fd=candidate_fd)
            try:
                opened = os.fstat(marker_fd)
                marker_identity = _candidate_identity(marker_stat)
                if _candidate_identity(opened) != marker_identity:
                    raise ValueError("possible completion marker changed during inspection")
                if observer:
                    observer("after_marker_open", self.candidate.path)
                with os.fdopen(marker_fd, "r", encoding="utf-8") as stream:
                    marker_fd = None
                    metadata = json.load(stream)
                    if _candidate_identity(os.fstat(stream.fileno())) != marker_identity:
                        raise ValueError(
                            "possible completion marker changed while being read"
                        )
                    return metadata
            finally:
                if marker_fd is not None:
                    os.close(marker_fd)
        finally:
            os.close(candidate_fd)

    def newest_mtime(self):
        """Return newest lstat mtime using only no-atime descriptor traversal."""
        current = self.stat_leaf(
            self.candidate.path.name, self.candidate.identity, full=True,
        )
        newest = current.st_mtime
        if self.candidate.kind == "file":
            return newest
        candidate_fd = os.open(
            self.candidate.path.name, _inspection_directory_flags(),
            dir_fd=self.parent_fd,
        )
        try:
            return max(newest, _newest_mtime_from_fd(candidate_fd, self.candidate.path))
        finally:
            os.close(candidate_fd)

    def __exit__(self, exc_type, exc, tb):
        for fd in reversed(self._fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self.parent_fd = None
        return False


def _revalidate_candidate_structural(candidate):
    """Revalidate link/type and stable identity before lease state is known."""
    _revalidate_directory_chain(candidate.parent_chain)
    current = candidate.path.lstat()
    if _is_link_or_reparse(candidate.path, current):
        raise ValueError("temporary artifact changed or became a link/reparse point")
    expected = stat.S_ISDIR if candidate.kind == "directory" else stat.S_ISREG
    if not expected(current.st_mode):
        raise ValueError("temporary artifact type changed after discovery")
    if not _candidate_identity_matches(current, candidate.identity, full=False):
        raise ValueError("temporary artifact structural identity changed after discovery")
    return current


def _revalidate_candidate_full(candidate):
    """Require structural and mutable fields to match the discovery snapshot."""
    current = _revalidate_candidate_structural(candidate)
    if not _candidate_identity_matches(current, candidate.identity, full=True):
        raise ValueError("temporary artifact changed after discovery")
    return current


def _newest_mtime_from_fd(directory_fd, display):
    """Walk one opened directory without following names or updating atime."""
    newest = os.fstat(directory_fd).st_mtime
    with _scandir_inspection_fd(directory_fd) as entries:
        for entry in entries:
            entry_stat = os.stat(
                entry.name, dir_fd=directory_fd, follow_symlinks=False,
            )
            newest = max(newest, entry_stat.st_mtime)
            if _stat_is_link_or_reparse(entry_stat) or not stat.S_ISDIR(entry_stat.st_mode):
                continue
            child_fd = os.open(
                entry.name, _inspection_directory_flags(), dir_fd=directory_fd,
            )
            try:
                opened = os.fstat(child_fd)
                if _candidate_identity(opened) != _candidate_identity(entry_stat):
                    raise _UnsafeAnchoredInspection(
                        f"inspection directory changed: {display / entry.name}"
                    )
                newest = max(
                    newest, _newest_mtime_from_fd(child_fd, display / entry.name),
                )
            finally:
                os.close(child_fd)
    return newest


def _discover_temp_candidates(root, root_chain, result):
    """Discover candidates by descriptor-relative, no-atime traversal."""
    candidates = []
    root_fd = _open_inspection_root(root_chain)

    def walk(directory_fd, display, relative_chain):
        with _scandir_inspection_fd(directory_fd) as entries:
            for entry in entries:
                path = display / entry.name
                try:
                    entry_stat = os.stat(
                        entry.name, dir_fd=directory_fd, follow_symlinks=False,
                    )
                    if _stat_is_link_or_reparse(entry_stat):
                        continue
                    kind = None
                    leaf_kind = internal_temp_leaf_kind(entry.name)
                    if leaf_kind == "directory" and stat.S_ISDIR(entry_stat.st_mode):
                        kind = "directory"
                    elif leaf_kind == "file" and stat.S_ISREG(entry_stat.st_mode):
                        kind = "file"
                    if kind:
                        candidates.append(_TempCandidate(
                            path, kind, _candidate_identity(entry_stat),
                            root_chain + relative_chain,
                            len(root_chain) - 1,
                        ))
                        continue
                    if not stat.S_ISDIR(entry_stat.st_mode):
                        continue
                    child_fd = os.open(
                        entry.name, _inspection_directory_flags(), dir_fd=directory_fd,
                    )
                    try:
                        opened = os.fstat(child_fd)
                        if _candidate_identity(opened) != _candidate_identity(entry_stat):
                            raise _UnsafeAnchoredInspection(
                                f"inspection directory changed: {path}"
                            )
                        walk(
                            child_fd, path,
                            relative_chain + ((path, _stat_identity(opened)),),
                        )
                    finally:
                        os.close(child_fd)
                except (OSError, RuntimeError, ValueError) as exc:
                    result["errors"].append({"path": str(path), "error": str(exc)})

    try:
        walk(root_fd, root, ())
    finally:
        os.close(root_fd)
    return candidates


def _open_inspection_root(root_chain):
    """Open the verified root itself without permitting access-time updates."""
    flags = _inspection_directory_flags()
    root, expected = root_chain[-1]
    fd = os.open(root, flags)
    try:
        current = os.fstat(fd)
        if _stat_identity(current) != expected or not stat.S_ISDIR(current.st_mode):
            raise _UnsafeAnchoredInspection(f"inspection root changed: {root}")
        return fd
    except Exception:
        os.close(fd)
        raise


@dataclass
class _LeaseProbe:
    """Descriptor-backed proof returned by a safe existing-lease probe."""

    held: bool
    fd: int
    identity: tuple
    locked_by_us: bool = False

    def close(self):
        if self.fd is None:
            return
        fd, self.fd = self.fd, None
        try:
            if self.locked_by_us:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _inspect_existing_lease(anchored, lease_leaf):
    """Return a retained descriptor proof for an existing POSIX lease."""
    if os.name == "nt":
        raise _UnsafeAnchoredInspection(
            "safe handle-relative lease inspection is unavailable on Windows"
        )
    lease_stat = os.stat(lease_leaf, dir_fd=anchored.parent_fd, follow_symlinks=False)
    if _stat_is_link_or_reparse(lease_stat) or not stat.S_ISREG(lease_stat.st_mode):
        raise _UnsafeAnchoredInspection("temporary artifact lease is not a safe regular file")
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NOATIME
    fd = os.open(lease_leaf, flags, dir_fd=anchored.parent_fd)
    try:
        opened = os.fstat(fd)
        if _stat_identity(opened) != _stat_identity(lease_stat):
            raise _UnsafeAnchoredInspection("temporary artifact lease changed during inspection")
        import fcntl
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            proof = _LeaseProbe(True, fd, _stat_identity(opened))
        else:
            proof = _LeaseProbe(False, fd, _stat_identity(opened), locked_by_us=True)
        fd = None
        return proof
    finally:
        if fd is not None:
            os.close(fd)


def _check_held_lease_advisory(
    *, root_chain, candidate, anchored, lease_leaf, proof, observer=None
):
    """Run ordered, descriptor-backed safety checks for a held lease.

    These checks are deliberately not an atomic filesystem snapshot. A
    successful return means only that every check succeeded when it ran and no
    inconsistency was observed before the result was produced. Entries checked
    earlier may already have changed, so the resulting ``active`` classification
    is advisory and must never authorize a destructive operation.
    """
    _revalidate_directory_chain(root_chain)
    _revalidate_directory_chain(candidate.parent_chain)

    parent_start = os.fstat(anchored.parent_fd)
    expected_parent = candidate.parent_chain[-1][1]
    if (
        _stat_identity(parent_start) != expected_parent
        or not stat.S_ISDIR(parent_start.st_mode)
    ):
        raise _UnsafeAnchoredInspection(
            "temporary artifact parent changed during active checks"
        )

    candidate_first = anchored.stat_leaf(candidate.path.name)
    if not _candidate_identity_matches(candidate_first, candidate.identity, full=False):
        raise _UnsafeAnchoredInspection(
            "temporary artifact structural identity changed during active checks"
        )
    if observer:
        observer("after_active_candidate_check_1", candidate.path)

    lease_first = os.stat(
        lease_leaf, dir_fd=anchored.parent_fd, follow_symlinks=False,
    )
    if (
        _stat_is_link_or_reparse(lease_first)
        or not stat.S_ISREG(lease_first.st_mode)
        or _stat_identity(lease_first) != proof.identity
        or _stat_identity(os.fstat(proof.fd)) != proof.identity
    ):
        raise _UnsafeAnchoredInspection(
            "temporary artifact lease changed during active checks"
        )
    if observer:
        observer("after_active_lease_entry_check_1", candidate.path)

    candidate_second = anchored.stat_leaf(candidate.path.name)
    if not _candidate_identity_matches(candidate_second, candidate.identity, full=False):
        raise _UnsafeAnchoredInspection(
            "temporary artifact structural identity changed during active checks"
        )
    if observer:
        observer("after_active_candidate_check_2", candidate.path)
    lease_second = os.stat(
        lease_leaf, dir_fd=anchored.parent_fd, follow_symlinks=False,
    )
    if (
        _stat_is_link_or_reparse(lease_second)
        or not stat.S_ISREG(lease_second.st_mode)
        or _stat_identity(lease_second) != proof.identity
        or _stat_identity(os.fstat(proof.fd)) != proof.identity
    ):
        raise _UnsafeAnchoredInspection(
            "temporary artifact lease changed during active checks"
        )
    if observer:
        observer("after_active_lease_entry_check_2", candidate.path)

    parent_end = os.fstat(anchored.parent_fd)
    if (
        _stat_identity(parent_end) != expected_parent
    ):
        raise _UnsafeAnchoredInspection(
            "temporary artifact parent changed during active checks"
        )
    _revalidate_directory_chain(root_chain)
    _revalidate_directory_chain(candidate.parent_chain)

    # This is merely the end of the ordered checks, not a filesystem
    # linearization point. A previously checked name may change before this
    # callback or before the caller receives the advisory result.
    if observer:
        observer("before_active_result", candidate.path)


def inspect_abandoned_export_temps(
    *, root=None, now=None, stale_age=DEFAULT_TEMP_STALE_AGE, observer=None
):
    """Non-destructively classify J-Wash temporary export artifacts.

    On supported POSIX systems, all traversal and reads use descriptor-relative
    ``O_NOATIME`` opens; inspection never falls back to an operation that may
    advance access times. Unsupported or denied strict inspection fails closed.
    Startup inspection deliberately never renames, removes, or modifies an
    artifact or lease. Structural inconsistencies detected during inspection
    are ``changed_or_unsafe``. ``active`` means a held lease was observed and
    each ordered no-follow safety check succeeded when performed. The checks are
    not an atomic snapshot: an inter-check change may go undetected and the
    result may already be stale. It must never authorize a destructive action.
    """
    root = _absolute_no_follow(root if root is not None else EDITS_DIR)
    current_time = time.time() if now is None else float(now)
    result = {
        "active": [], "recent": [], "completed": [], "abandoned": [],
        "changed_or_unsafe": [], "errors": [],
        # Deprecated compatibility fields. Startup inspection never deletes.
        "removed": [], "removed_count": 0,
        "skipped_active": [], "skipped_recent": [],
        "skipped_completed": [], "skipped_changed": [],
    }
    if not _strict_inspection_capability():
        result["errors"].append({
            "path": str(root),
            "error": "strict metadata-preserving temporary inspection is unavailable on this platform",
        })
        return result
    try:
        root_chain = _snapshot_directory_chain(root, missing_ok=True)
        if root_chain is None:
            return result
    except (OSError, RuntimeError, ValueError) as exc:
        result["errors"].append({"path": str(root), "error": str(exc)})
        return result

    try:
        candidates = _discover_temp_candidates(root, root_chain, result)
    except (OSError, RuntimeError, ValueError) as exc:
        result["errors"].append({"path": str(root), "error": str(exc)})
        return result

    if observer:
        observer("discovered", tuple(candidate.path for candidate in candidates))

    def classify(key, path):
        result[key].append(str(path))
        legacy = {
            "active": "skipped_active", "recent": "skipped_recent",
            "completed": "skipped_completed", "changed_or_unsafe": "skipped_changed",
        }.get(key)
        if legacy:
            result[legacy].append(str(path))

    for candidate in candidates:
        path, kind = candidate.path, candidate.kind
        try:
            _revalidate_directory_chain(root_chain)
            path.relative_to(root)
            # Before lease state is known, allow only structural continuity.
            # A held exporter may legitimately mutate contents and timestamps.
            _revalidate_candidate_structural(candidate)
            if observer:
                observer("before_inspect", path)
            if os.name == "nt":
                raise _UnsafeAnchoredInspection(
                    "safe handle-relative temporary inspection is unavailable on Windows"
                )
            with _AnchoredInspectionParent(candidate) as anchored:
                lease_leaf = path.name + ".lease"
                try:
                    os.stat(lease_leaf, dir_fd=anchored.parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    has_lease = False
                else:
                    has_lease = True
                    with _inspect_existing_lease(anchored, lease_leaf) as lease_proof:
                        if lease_proof.held:
                            if observer:
                                observer("after_held_lease_probe", path)
                            _check_held_lease_advisory(
                                root_chain=root_chain,
                                candidate=candidate,
                                anchored=anchored,
                                lease_leaf=lease_leaf,
                                proof=lease_proof,
                                observer=observer,
                            )
                            classify("active", path)
                            continue

                # An active writer may legitimately change size/timestamps.
                # Once an existing lease is known to be unlocked (or absent),
                # require the complete discovery identity before classification.
                anchored.stat_leaf(path.name, candidate.identity, full=True)

                if not has_lease and kind == "directory":
                    try:
                        metadata = anchored.read_completion_marker(observer)
                    except FileNotFoundError:
                        metadata = None
                    except Exception as exc:
                        raise _UnsafeAnchoredInspection(
                            f"cannot safely inspect possible completion marker: {exc}"
                        ) from exc
                    if metadata is not None:
                        _revalidate_candidate_full(candidate)
                        relative_name = path.relative_to(root).as_posix()
                        if not isinstance(metadata, dict) or not isinstance(metadata.get("name"), str):
                            raise _UnsafeAnchoredInspection(
                                "possible completion marker has invalid metadata"
                            )
                        if metadata["name"] == relative_name:
                            classify("completed", path)
                            continue

                newest = anchored.newest_mtime()
                if observer:
                    observer("after_activity", path)
                anchored.stat_leaf(path.name, candidate.identity, full=True)
            _revalidate_directory_chain(root_chain)
            if current_time - newest < stale_age:
                classify("recent", path)
            else:
                classify("abandoned", path)
        except Exception as exc:
            classify("changed_or_unsafe", path)
            result["errors"].append({"path": str(path), "error": str(exc)})
    return result


def cleanup_abandoned_export_temps(**kwargs):
    """Deprecated non-destructive alias for startup inspection."""
    return inspect_abandoned_export_temps(**kwargs)


# Residual writes edited by the global abliteration (embed aside)
TARGET_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_presets():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(PRESETS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.append({"name": path.stem, "n_rules": len(data.get("rules", [])), "model_id": data.get("model_id")})
    return out


def save_preset(name, rules, model_id, scale=1.0):
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": model_id, "saved_at": _now(), "scale": scale, "rules": rules}
    (PRESETS_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return payload


def load_preset(name):
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown preset {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_preset(name):
    (PRESETS_DIR / f"{name}.json").unlink(missing_ok=True)


def compute_abliteration(rules, jl, scale=1.0):
    """Global pure-weight edit reproducing the abliteration-mode preview.

    Applies to EVERY residual write (embed_tokens + o_proj/down_proj of every
    layer) the same transform as the abliteration-mode hooks: for each rule,
    ``out += scale·(v̂_A·out)·w`` (applied sequentially, like the hooks). Since
    the residual is the sum of all these writes, the direction is
    removed/redirected across the whole residual — hence the fidelity (~0.97
    cosine on the logits). This is the pure-weights path for architectures the
    rebase does not support (write norms, Gemma style).

    Returns ``(tensors, info)``:
      - ``tensors``: {param_name: W_new (cpu, float32)}
      - ``info``: {tied, embed_key, lm_head_key, path, delta_max, lowrank}
        where ``lowrank`` = {param_name: (B [out, r], A [r, in])} — the SAME edit
        as per-rule rank-1 factors (delta = B·A), exact, for the LoRA export.
        For the embed, delta = (B·A)ᵀ (PEFT lookup convention).
    """
    # layers=[] = disabled rule, in this mode too (consistent with the preview)
    rules = [r for r in rules if r["layers"]]
    if not rules:
        raise ValueError("no active rule (all have 0 layers): nothing to export")
    path = jl.layout.path
    weight_u = jl._lm_head.weight
    # (v_a, w_eff) per rule, with w_eff = alpha·v̂_A + beta·v̂_B: the SAME effective
    # coefficients (saturation included) as the preview hooks
    pairs = []
    for r in rules:
        v_a, v_b = abliteration_direction(weight_u, r)
        alpha, beta = effective_coeffs(r["mode"], r["factor"], scale)
        w_eff = alpha * v_a
        if beta:
            w_eff = w_eff + beta * v_b
        pairs.append((v_a, w_eff))

    # bake on CPU: the float32 matrices (embed ~1.5 GB) don't fit alongside the
    # model on the GPU (OOM measured on 12 GB with a 4B loaded)
    def apply_cols(W):  # [d_model, d_in]: residual output = rows
        cur, us, rows = W, [], []
        for v_a, w in pairs:
            row = v_a @ cur  # composed over the previous rules
            us.append(w)
            rows.append(row)
            cur = cur + torch.outer(w, row)
        return cur, torch.stack(us, dim=1), torch.stack(rows, dim=0)

    def apply_rows(E):  # [vocab, d_model]: each ROW is a residual vector
        cur, us, rows = E, [], []
        for v_a, w in pairs:
            col = cur @ v_a  # [vocab]
            us.append(w)
            rows.append(col)
            cur = cur + torch.outer(col, w)
        return cur, torch.stack(us, dim=1), torch.stack(rows, dim=0)

    tensors = {}
    lowrank = {}
    delta_max = 0.0

    embed_key = f"{path}.{jl.layout.embed}.weight"
    E = jl._embed_tokens.weight.detach().float().cpu()
    E_new, B, A = apply_rows(E)
    delta_max = max(delta_max, (E_new - E).abs().max().item())
    tensors[embed_key] = E_new
    lowrank[embed_key] = (B, A)  # delta_embed = (B·A)ᵀ = summed outer(A_k, B_k)

    skipped_writes = 0
    for i, block in enumerate(jl.layers):
        for suffix in TARGET_SUFFIXES:
            module = block
            for part in suffix.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:  # e.g. linear-attention blocks (no self_attn)
                skipped_writes += 1
                continue
            W = module.weight.detach().float().cpu()
            W_new, B, A = apply_cols(W)
            delta_max = max(delta_max, (W_new - W).abs().max().item())
            name = f"{path}.layers.{i}.{suffix}.weight"
            tensors[name] = W_new
            lowrank[name] = (B, A)

    tied = jl._lm_head.weight.data_ptr() == jl._embed_tokens.weight.data_ptr()
    info = {
        "tied": tied,
        "embed_key": embed_key,
        "lm_head_key": f"{jl.layout.lm_head}.weight",
        "path": path,
        "delta_max": delta_max,
        "lowrank": lowrank,
        "skipped_writes": skipped_writes,
    }
    return tensors, info


def _abliteration_warnings(rules):
    warns = []
    for r in rules:
        if r["mode"] == "scale" and r["factor"] > 1.0:
            warns.append(
                f"\"{(r['token'] or '').strip()}\" ×{r['factor']}: amplifying (factor > 1) "
                "is approximate in pure weights (the hook composes over the layers)"
            )
    return warns


def export_abliteration(rules, jl, model_meta, *, fmt, name, source_dir=None, scale=1.0):
    """Pure-weight export (global abliteration). Formats: ``full`` (full
    checkpoint), ``layers`` (safetensors of only the modified matrices) and
    ``lora`` (exact PEFT adapter, rank = n_rules; embed omitted if embeddings
    are tied). Unties ``lm_head`` (full/layers) if the model has tied embeddings,
    to preserve the original un-embedding."""
    parts = validate_export_name(name)
    name = "/".join(parts)
    out_dir = EDITS_DIR.joinpath(*parts)
    rules = [r for r in rules if r["layers"]]  # layers=[] = disabled rule
    if not rules:
        raise ValueError("no active intervention to export")
    if fmt not in ("full", "layers", "lora"):
        raise ValueError(f"unknown format for abliteration: {fmt}")

    tensors, info = compute_abliteration(rules, jl, scale=scale)
    if info["delta_max"] < 1e-8:
        raise ValueError(
            "the bake changes no weight (neutral factors, scale=0 or null "
            "directions) — the export would be identical to the original model"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if model_meta.get("dtype") == "bf16" else torch.float16
    lm_head_key = info["lm_head_key"]
    summary = [
        {k: r[k] for k in ("token_id", "token", "mode", "factor", "replacement_id", "replacement")}
        for r in rules
    ]
    meta = {
        "name": name,
        "format": fmt,
        "method": "abliteration-global",
        "model_id": model_meta.get("model_id"),
        "model_revision": model_meta.get("revision"),
        "dtype": model_meta.get("dtype"),
        "global_scale": scale,
        "untied_lm_head": info["tied"] and fmt in ("full", "layers"),
        "rules": summary,
        "modified_params_count": len(tensors) + (1 if info["tied"] else 0),
        "warnings": _abliteration_warnings(rules) + (
            [f"{info['skipped_writes']} residual write(s) without o_proj/down_proj "
             "(hybrid architecture) left untouched — the bake is partial there; "
             "prefer read projection when the architecture supports it"]
            if info["skipped_writes"] else []
        ),
        "note": (
            "global abliteration: the token's direction is removed/redirected in "
            "every residual write (embed + o_proj/down_proj of all layers). "
            "Reproduces the abliteration-mode preview (~0.97 cosine on the logits). "
            "Pure weights: a standard safetensors checkpoint."
        ),
        "created_at": _now(),
    }

    if fmt == "layers":
        out = {k: v.to(dtype) for k, v in tensors.items()}
        if info["tied"]:
            # original un-embedding (unedited embed) to write separately
            out[lm_head_key] = jl._embed_tokens.weight.detach().to(dtype).cpu()
        save_file(out, str(out_dir / "modified_layers.safetensors"))

    elif fmt == "lora":
        # The abliteration delta is EXACTLY rank-n_rules per matrix (delta = B·A),
        # so the LoRA is exact — except the embed of a tied-embeddings model: PEFT
        # can't untie lm_head, and editing the embed would corrupt the shared
        # un-embedding → we omit it (reduced fidelity).
        include_embed = not info["tied"]
        if not include_embed:
            meta["warnings"] = meta["warnings"] + [
                "tied embeddings: the embed is not included in the LoRA (PEFT "
                "cannot untie lm_head) — prefer \"full checkpoint\" for maximum "
                "fidelity"
            ]
        out = {}
        target_modules = set()
        for pname, (B, A) in info["lowrank"].items():
            base = pname.removesuffix(".weight")
            if pname == info["embed_key"]:
                if not include_embed:
                    continue
                target_modules.add(base.rsplit(".", 1)[-1])
                # PEFT Embedding convention: delta_lookup = (B·A)ᵀ,
                # A = lora_embedding_A [r, vocab], B = lora_embedding_B [d_model, r]
                out[f"base_model.model.{base}.lora_embedding_A"] = A.contiguous()
                out[f"base_model.model.{base}.lora_embedding_B"] = B.contiguous()
            else:
                target_modules.add(base.rsplit(".", 1)[-1])
                out[f"base_model.model.{base}.lora_A.weight"] = A.contiguous()
                out[f"base_model.model.{base}.lora_B.weight"] = B.contiguous()
        rank = len(rules)
        save_file(out, str(out_dir / "adapter_model.safetensors"))
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": model_meta.get("model_id"),
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "target_modules": sorted(target_modules),
            "bias": "none",
            "fan_in_fan_out": False,
            "task_type": "CAUSAL_LM",
        }
        (out_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=1), encoding="utf-8"
        )

    elif fmt == "full":
        if source_dir is None or not Path(source_dir).is_dir():
            raise ValueError("full checkpoint: model source folder not found")
        source_dir = Path(source_dir)
        shards = sorted(source_dir.glob("*.safetensors"))
        if not shards:
            raise ValueError("full checkpoint: no safetensors in the source")

        lm_head_value = None  # original un-embedding (if tied) = original embed from disk
        embed_shard_name = None
        seen = set()
        for shard in shards:
            ino = shard.stat().st_ino
            if ino in seen:
                continue
            seen.add(ino)
            out = {}
            with safe_open(str(shard), framework="pt") as f:
                keys = list(f.keys())
                for key in keys:
                    original = f.get_tensor(key)
                    if info["tied"] and key == info["embed_key"]:
                        lm_head_value = original.clone()  # BEFORE editing
                        embed_shard_name = shard.name
                    out[key] = tensors[key].to(original.dtype) if key in tensors else original
            # if this shard already carries lm_head (untied model), don't touch it
            save_file(out, str(out_dir / shard.name))

        # untie: add lm_head.weight (= original embed) into the embed's shard
        if info["tied"]:
            if lm_head_value is None:
                raise ValueError("cannot untie: embed not found in the source")
            target_shard = out_dir / embed_shard_name
            with safe_open(str(target_shard), framework="pt") as f:
                merged = {k: f.get_tensor(k) for k in f.keys()}
            merged[lm_head_key] = lm_head_value
            save_file(merged, str(target_shard))

        # config.json: copy, force tie_word_embeddings=False if untied
        cfg_path = source_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if info["tied"]:
                cfg["tie_word_embeddings"] = False
            (out_dir / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        # other tokenizer/config files (json, merges.txt, tokenizer.model…):
        # copy as-is, then fix the index if present
        for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"):
            for extra in source_dir.glob(pattern):
                if extra.name == "config.json":
                    continue
                shutil.copy2(extra, out_dir / extra.name)
        index_path = out_dir / "model.safetensors.index.json"
        if info["tied"] and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            wm = index.setdefault("weight_map", {})
            wm[lm_head_key] = embed_shard_name
            if "metadata" in index and "total_size" in index["metadata"]:
                index["metadata"]["total_size"] += lm_head_value.numel() * lm_head_value.element_size()
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

    (out_dir / "edit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"out_dir": str(out_dir), **meta}


def _disk_mapper(mem_embed_key, disk_keys):
    """Memory keys (instantiated model's layout) → disk checkpoint keys.

    transformers renames on load: e.g. Qwen3.5 is instantiated as ForCausalLM
    ("model.layers.*" in memory) but saved in ConditionalGeneration format
    ("model.language_model.layers.*"). Without this mapping, a "full" export
    would copy the source verbatim without transforming anything. We anchor the
    disk prefix on the embed, whose suffix is unique in the checkpoint."""
    if mem_embed_key in disk_keys:
        return lambda key: key
    suffix = "." + ".".join(mem_embed_key.rsplit(".", 2)[-2:])  # ".embed_tokens.weight"
    candidates = [k for k in disk_keys if k.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError(
            f"checkpoint prefix undecidable: {mem_embed_key} absent from the source "
            f"and {len(candidates)} key(s) end with {suffix}"
        )
    mem_prefix = mem_embed_key.removesuffix(suffix)
    disk_prefix = candidates[0].removesuffix(suffix)

    def to_disk(key):
        if key == mem_prefix or key.startswith(mem_prefix + "."):
            return disk_prefix + key[len(mem_prefix):]
        return key

    return to_disk


_INDEX_RESERVED = {
    "config.json", "model.safetensors.index.json", "edit_meta.json",
    "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "generation_config.json", "special_tokens_map.json",
    "chat_template.json",
}


def _indexed_shards(source_dir):
    """Return a fully schema-validated index and its logical shard paths."""
    path = source_dir / "model.safetensors.index.json"
    if not path.exists():
        return None, None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"full checkpoint: invalid safetensors index: {exc}") from exc
    if not isinstance(index, dict):
        raise ValueError("full checkpoint: safetensors index root must be an object")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("full checkpoint: index metadata must be an object")
    total_size = metadata.get("total_size")
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size < 0:
        raise ValueError("full checkpoint: metadata.total_size must be a nonnegative integer")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("full checkpoint: index weight_map must be a nonempty object")
    filenames = []
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("full checkpoint: every weight_map key must be a nonempty string")
        if not isinstance(filename, str) or not filename:
            raise ValueError("full checkpoint: every shard filename must be a nonempty string")
        try:
            parts = _safe_relative_parts(filename)
        except ValueError as exc:
            raise ValueError(f"full checkpoint: unsafe shard filename {filename!r}: {exc}") from exc
        if len(parts) != 1 or Path(filename).name != filename or not filename.lower().endswith(".safetensors"):
            raise ValueError(f"full checkpoint: shard filename must be a plain .safetensors leaf: {filename!r}")
        filenames.append(filename)
    folded = [name.casefold() for name in dict.fromkeys(filenames)]
    if len(folded) != len(set(folded)):
        raise ValueError("full checkpoint: shard filenames have a case-fold collision")
    auxiliary = {p.name.casefold() for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja")
                 for p in source_dir.glob(pattern)}
    forbidden = {name.casefold() for name in _INDEX_RESERVED} | auxiliary
    collisions = sorted({name for name in dict.fromkeys(filenames) if name.casefold() in forbidden})
    if collisions:
        raise ValueError(f"full checkpoint: shard filename collides with reserved/auxiliary file: {collisions}")
    shards = [source_dir / name for name in dict.fromkeys(filenames)]
    missing = [shard.name for shard in shards if not shard.is_file()]
    if missing:
        raise ValueError(f"full checkpoint: indexed shard(s) missing: {missing}")
    return index, shards


def _validate_index_contents(index, shard_paths):
    """Validate exact key placement, coverage, and logical total size."""
    key_sets = {}
    tensor_sizes = {}
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                   "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                   "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for shard in shard_paths:
        try:
            with safe_open(str(shard), framework="pt") as f:
                keys = set(f.keys()); key_sets[shard.name] = keys
                tensor_sizes[shard.name] = {}
                for key in keys:
                    tensor_slice = f.get_slice(key)
                    size = dtype_bytes.get(tensor_slice.get_dtype())
                    if size is None:
                        raise ValueError(f"unsupported safetensors dtype {tensor_slice.get_dtype()}")
                    elements = 1
                    for dimension in tensor_slice.get_shape(): elements *= dimension
                    tensor_sizes[shard.name][key] = elements * size
        except Exception as exc:
            raise ValueError(f"invalid safetensors shard {shard.name}: {exc}") from exc
    weight_map = index["weight_map"]
    for key, filename in weight_map.items():
        if key not in key_sets.get(filename, set()):
            raise ValueError(f"index maps {key!r} to {filename!r}, but that shard does not contain it")
    staged_keys = set().union(*key_sets.values()) if key_sets else set()
    unindexed = staged_keys - set(weight_map)
    if unindexed:
        raise ValueError(f"indexed checkpoint contains unindexed tensor(s): {sorted(unindexed)[:3]}")
    logical_total = sum(tensor_sizes[filename][key] for key, filename in weight_map.items())
    if index["metadata"]["total_size"] != logical_total:
        raise ValueError(
            f"index metadata.total_size={index['metadata']['total_size']} does not match {logical_total}"
        )
    return key_sets, logical_total


REBASE_EXPORT_ROW_BUDGET = 4096
REBASE_EXPORT_CHUNK_OBSERVER = None

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10)),
                     "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"}


def validate_export_name(name):
    """Lexically validate a portable relative export name before resolution."""
    if not isinstance(name, str) or not name:
        raise ValueError("export name must be a nonempty relative path")
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        raise ValueError("export name must not be absolute, drive-qualified, or UNC")
    if "\\" in name:
        raise ValueError("export name must use portable '/' separators")
    raw = name.split("/")
    if any(part in ("", ".", "..") for part in raw):
        raise ValueError("export name contains an empty, '.', or '..' component")
    for part in raw:
        if internal_temp_leaf_kind(part) is not None:
            raise ValueError(
                f"export name component {part!r} is reserved for internal temporary export use"
            )
        if part.endswith((" ", ".")) or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ValueError(f"export name contains Windows-reserved component {part!r}")
        if any(ord(char) < 32 or char in '<>:"|?*' for char in part):
            raise ValueError(f"export name contains nonportable component {part!r}")
    return tuple(raw)


def _safe_relative_parts(name):
    """Backward-compatible private alias for the public validator."""
    return validate_export_name(name)


def apply_transform_bounded(entry, tensor, *, row_budget=REBASE_EXPORT_ROW_BUDGET,
                            observer=None):
    """Apply a read transform with bounded float32 row temporaries.

    The destination is allocated directly in the source dtype.  Write
    transforms are ordinary dense matrices and retain the established path.
    """
    kind, X, Y = entry
    if kind != "read":
        source = tensor.detach().to("cpu", torch.float32)
        updated, _B, _A = rebase.apply_transform(entry, source)
        return updated.to(tensor.dtype), float((updated - source).abs().max())
    if row_budget < 1:
        raise ValueError("row_budget must be positive")
    source = tensor.detach().cpu().contiguous()
    rows = source.reshape(-1, source.shape[-1])
    destination = torch.empty_like(source)
    dest_rows = destination.reshape_as(rows)
    delta_max = 0.0
    for start in range(0, rows.shape[0], row_budget):
        count = min(row_budget, rows.shape[0] - start)
        if observer is not None:
            observer(count)
        chunk = rows[start:start + count].float()
        B = chunk @ X
        updated = chunk + B @ Y.T
        delta_max = max(delta_max, float((updated - chunk).abs().max()))
        dest_rows[start:start + count].copy_(updated.to(source.dtype))
    return destination, delta_max


def export_rebase(rules, jl, model_meta, *, fmt, name, source_dir=None, scale=1.0, exact=False):
    """Transactionally construct and atomically publish a rebase export.

    Overwrite policy is deliberately conservative: an existing destination is
    rejected and never modified.  All construction occurs in a unique sibling
    staging directory which is removed on every failure.
    """
    parts = validate_export_name(name)
    root = EDITS_DIR.resolve()
    final_dir = root.joinpath(*parts).resolve()
    try:
        relative = final_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("export name must remain beneath the edits directory") from exc
    if not relative.parts:
        raise ValueError("export name must identify a directory beneath the edits directory")
    if final_dir.exists():
        raise ValueError(f"export destination already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = final_dir.parent / f".{final_dir.name}.tmp-{uuid.uuid4().hex}"
    with artifact_lease(stage):
        stage.mkdir()
        try:
            result = _export_rebase_impl(
                rules, jl, model_meta, fmt=fmt, name=name, source_dir=source_dir,
                scale=scale, exact=exact, out_dir=stage,
            )
            stage.replace(final_dir)
            result["out_dir"] = str(final_dir)
            return result
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def _export_rebase_impl(rules, jl, model_meta, *, fmt, name, source_dir=None,
                        scale=1.0, exact=False, out_dir):
    """Pure-weight export by change of basis of the reads (cf. core/rebase).

    ``readthrough`` (exact=False): the downstream read matrices + lm_head.
    ``exact``: adds the counter-transform of the downstream writes.
    Formats: ``full`` (checkpoint), ``layers`` (safetensors of the modified
    matrices) and ``lora`` (PEFT adapter = the exact low-rank diff between the
    baked weights and the originals; the lm_head delta is applied at forward
    time, so tied embeddings need no untying). The bake is done streaming, one
    float32 CPU matrix at a time. Tied-embeddings model (full/layers): the embed
    stays INTACT, it's lm_head (untied) that receives the final read transform."""
    method = "rebase-exact" if exact else "rebase-readthrough"
    if fmt not in ("full", "layers", "lora"):
        raise ValueError(f"unknown format for {method}: {fmt}")
    transforms, info = rebase.build_plan(rules, jl, scale, exact=exact)
    lm_head_key = info["lm_head_key"]

    packed = [key for key, target in info["targets"].items()
              if not target.lora_supported or target.tensor(jl.layers[int(key.split(".layers.", 1)[1].split(".", 1)[0])]).ndim > 2]
    if packed and fmt == "lora":
        raise ValueError(
            "LoRA export is unavailable for packed MoE parameters. "
            "Use full-checkpoint export."
        )
    if packed and fmt == "layers":
        raise ValueError(
            "modified-layers export is unavailable for packed MoE parameters: "
            "bounded-memory sharding is not implemented. Use full-checkpoint export."
        )

    delta_max = 0.0
    applied = set()

    def bake(key, tensor):
        nonlocal delta_max
        W_new, chunk_delta = apply_transform_bounded(
            transforms[key], tensor, row_budget=REBASE_EXPORT_ROW_BUDGET,
            observer=REBASE_EXPORT_CHUNK_OBSERVER,
        )
        delta_max = max(delta_max, chunk_delta)
        applied.add(key)
        return W_new

    dtype = torch.bfloat16 if model_meta.get("dtype") == "bf16" else torch.float16
    warnings = []
    indexed_artifact = None
    if exact and info["regularized_layers"]:
        warnings.append(
            "regularized inverse (full zap ⇒ singular transform) on layers "
            f"{info['regularized_layers']} — the effect there equals readthrough; "
            "prefer readthrough mode for full removals"
        )
    if fmt == "lora" and info["tied"]:
        warnings.append(
            "tied embeddings: use the adapter at runtime (PEFT applies the "
            "lm_head delta at forward time, leaving the shared embed intact); "
            "merging it into the base weights (merge_and_unload) would write "
            "that delta into the embed too — export a full checkpoint if you "
            "need merged weights"
        )

    def source_weight(state, key):
        source = state.get(key)
        if source is None and key == lm_head_key and info["tied"]:
            source = state[info["embed_key"]]  # tied: the un-embedding IS the embed
        if source is None:
            raise ValueError(
                f"parameter {key} not found in the loaded model — "
                "unexpected layout, export cancelled"
            )
        return source

    if fmt == "layers":
        state = jl._hf_model.state_dict()
        to_disk = lambda key: key  # noqa: E731 — refined if the source is available
        if source_dir is not None and Path(source_dir).is_dir():
            disk_keys = set()
            for shard in Path(source_dir).glob("*.safetensors"):
                with safe_open(str(shard), framework="pt") as f:
                    disk_keys.update(f.keys())
            if disk_keys:
                to_disk = _disk_mapper(info["embed_key"], disk_keys)
        tensors = {}
        for key in transforms:
            tensors[to_disk(key)] = bake(key, source_weight(state, key)).to(dtype)
        save_file(tensors, str(out_dir / "modified_layers.safetensors"))

    elif fmt == "lora":
        # The rebase delta is low-rank by construction (delta = B·A exactly, cf.
        # rebase.apply_transform): the adapter is the exact diff between the
        # baked weights and the originals, not an approximation. lm_head: PEFT
        # adds the delta at forward time without writing to the (possibly tied)
        # weight, so the un-embedding is effectively untied while the embed
        # stays intact. Module names follow the model as instantiated by
        # AutoModelForCausalLM (the same loading path as the UI).
        state = jl._hf_model.state_dict()
        factors = {}
        max_rank = 0
        for key in transforms:
            W = source_weight(state, key).detach().to("cpu", torch.float32)
            W_new, B, A = rebase.apply_transform(transforms[key], W)
            delta_max = max(delta_max, (W_new - W).abs().max().item())
            applied.add(key)
            factors[key] = (B, A)
            max_rank = max(max_rank, B.shape[1])
        tensors = {}
        module_paths = []
        for key, (B, A) in factors.items():
            base = key.removesuffix(".weight")
            module_paths.append(base)
            if B.shape[1] < max_rank:  # pad so a single config `r` fits every module
                pad = max_rank - B.shape[1]
                B = torch.cat([B, torch.zeros(B.shape[0], pad)], dim=1)
                A = torch.cat([A, torch.zeros(pad, A.shape[1])], dim=0)
            tensors[f"base_model.model.{base}.lora_A.weight"] = A.contiguous()
            tensors[f"base_model.model.{base}.lora_B.weight"] = B.contiguous()
        save_file(tensors, str(out_dir / "adapter_model.safetensors"))
        # target_modules as an anchored regex over the modules actually edited:
        # a plain suffix list would wrap the same projections in EVERY layer and
        # leave benign but alarming "missing adapter keys" warnings at load time
        target_regex = "(.*\\.)?(" + "|".join(re.escape(p) for p in sorted(module_paths)) + ")"
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": model_meta.get("model_id"),
            "r": max_rank,
            "lora_alpha": max_rank,  # scaling alpha/r = 1: B·A is the raw delta
            "lora_dropout": 0.0,
            "target_modules": target_regex,
            "bias": "none",
            "fan_in_fan_out": False,
            "task_type": "CAUSAL_LM",
        }
        (out_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=1), encoding="utf-8"
        )

    elif fmt == "full":
        if source_dir is None or not Path(source_dir).is_dir():
            raise ValueError("full checkpoint: model source folder not found")
        source_dir = Path(source_dir)
        source_index, indexed_paths = _indexed_shards(source_dir)
        if source_index is not None:
            shards = indexed_paths
            _validate_index_contents(source_index, shards)
        else:
            shards = sorted(source_dir.glob("*.safetensors"))
        if not shards:
            raise ValueError("full checkpoint: no safetensors in the source")

        disk_keys = set()
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                disk_keys.update(f.keys())
        to_disk = _disk_mapper(info["embed_key"], disk_keys)
        transforms = {to_disk(k): fn for k, fn in transforms.items()}
        lm_head_key = to_disk(lm_head_key)
        embed_key = to_disk(info["embed_key"])
        required = set(transforms)
        if info["tied"] and lm_head_key not in disk_keys and embed_key in disk_keys:
            required.remove(lm_head_key)
        missing_source = required - disk_keys
        if missing_source:
            sample = sorted(missing_source)[:3]
            raise ValueError(
                f"{len(missing_source)} parameter(s) to transform absent from the source "
                f"checkpoint (e.g. {sample}) — export cancelled before writing"
            )
        if any(key.startswith("mtp.") for key in disk_keys):
            warnings.append(
                "MTP weights were preserved but not transformed. Ordinary Transformers "
                "generation does not use them, but MTP/speculative decoding fidelity is not "
                "guaranteed for this edited checkpoint."
            )

        lm_head_written = False
        embed_shard_name = None
        for shard in shards:
            out = {}
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    original = f.get_tensor(key)
                    if key in transforms:
                        out[key] = bake(key, original).to(original.dtype)
                        if key == lm_head_key:
                            lm_head_written = True
                    else:
                        out[key] = original
                    if key == embed_key:
                        embed_shard_name = shard.name
            save_file(out, str(out_dir / shard.name))
            del out

        # untie: the transformed un-embedding becomes a separate lm_head, baked
        # from the original embed (which stays intact)
        if info["tied"] and not lm_head_written:
            if embed_shard_name is None:
                raise ValueError("cannot untie: embed not found in the source")
            target_shard = out_dir / embed_shard_name
            with safe_open(str(target_shard), framework="pt") as f:
                merged = {k: f.get_tensor(k) for k in f.keys()}
            embed_original = merged[embed_key]
            lm_head_value = bake(lm_head_key, embed_original).to(embed_original.dtype)
            merged[lm_head_key] = lm_head_value
            save_file(merged, str(out_dir / embed_shard_name))
            del merged

        missing = set(transforms) - applied
        if missing:
            sample = sorted(missing)[:3]
            # A streamed full export may already have written earlier shards.
            # Never leave a checkpoint that looks usable but is only partially
            # edited when the source inventory fails the completeness guard.
            raise ValueError(
                f"{len(missing)} parameter(s) to transform absent from the source "
                f"checkpoint (e.g. {sample}) — unexpected key names, export cancelled "
                "(the written checkpoint would be partially original)"
            )

        cfg_path = source_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if info["tied"]:
                cfg["tie_word_embeddings"] = False
                text_cfg = cfg.get("text_config")
                if isinstance(text_cfg, dict) and "tie_word_embeddings" in text_cfg:
                    text_cfg["tie_word_embeddings"] = False
            (out_dir / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"):
            for extra in source_dir.glob(pattern):
                if extra.name == "config.json":
                    continue
                shutil.copy2(extra, out_dir / extra.name)
        index_path = out_dir / "model.safetensors.index.json"
        if info["tied"] and not lm_head_written and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            wm = index.setdefault("weight_map", {})
            wm[lm_head_key] = embed_shard_name
            if "metadata" in index and "total_size" in index["metadata"]:
                index["metadata"]["total_size"] += (
                    lm_head_value.numel() * lm_head_value.element_size()
                )
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

        if source_index is not None:
            indexed_artifact = (index_path, [out_dir / shard.name for shard in shards], set(transforms))

    if delta_max < 1e-8:
        raise ValueError(
            "the bake changes no weight (null directions?) — the export would be "
            "identical to the original model, folder deleted"
        )

    summary = [
        {k: r[k] for k in ("token_id", "token", "mode", "factor", "replacement_id", "replacement", "layers")}
        for r in rules if r["layers"]
    ]
    meta = {
        "name": name,
        "format": fmt,
        "method": method,
        "model_id": model_meta.get("model_id"),
        "model_revision": model_meta.get("revision"),
        "dtype": model_meta.get("dtype"),
        "global_scale": scale,
        # lora: no physical untying — the lm_head delta lives in the adapter
        "untied_lm_head": info["tied"] and fmt != "lora",
        "rules": summary,
        "layers_span": info["layers_span"],
        "rank": info["rank_final"],
        "modified_params_count": len(transforms),
        "delta_max": delta_max,
        "min_gamma": info["min_gamma"],
        "warnings": warnings,
        "note": (
            "change of basis of the reads: every matrix that READS the residual "
            "downstream of the hooked layers (q/k/v, in_proj*, gate/up + lm_head) sees "
            "the residual transformed by the same J-space directions as the live preview"
            + (" ; downstream writes counter-transformed (exact mode)" if exact else "")
            + ". Pure weights: a standard safetensors checkpoint."
        ),
        "created_at": _now(),
    }
    (out_dir / "edit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if indexed_artifact is not None:
        index_path, staged_shards, transformed_keys = indexed_artifact
        staged_index, _ = _indexed_shards(out_dir)
        key_sets, _ = _validate_index_contents(staged_index, staged_shards)
        for key in transformed_keys:
            filename = staged_index["weight_map"].get(key)
            if filename is None or key not in key_sets.get(filename, set()):
                raise ValueError(f"transformed key {key} is not validly placed by the final index")
    return {"out_dir": str(out_dir), **meta}
