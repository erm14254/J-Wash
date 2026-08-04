import multiprocessing
import errno
import os
import shutil
import socket
import stat
from pathlib import Path

import pytest

from core import editing

HEX = "a" * 32
OLD = 1_000.0
NOW = OLD + editing.DEFAULT_TEMP_STALE_AGE + 10


def _mtime(path, value=OLD):
    os.utime(path, (value, value))


def _stage(root, name=f".model.tmp-{HEX}", *, recent=False):
    path = root / name
    path.mkdir(parents=True)
    (path / "payload").write_bytes(b"directory-bytes")
    _mtime(path / "payload", NOW - 1 if recent else OLD)
    _mtime(path, NOW - 1 if recent else OLD)
    return path


def _gguf(root, name=f".model.tmp-{HEX}.gguf", *, recent=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"file-bytes")
    _mtime(path, NOW - 1 if recent else OLD)
    return path


def _inspect(root, **kwargs):
    return editing.inspect_abandoned_export_temps(root=root, now=NOW, **kwargs)


def _set_old_atime(path, *, mtime_ns=None):
    current = path.stat()
    old_atime_ns = 946_684_800_000_000_000
    os.utime(path, ns=(old_atime_ns, current.st_mtime_ns if mtime_ns is None else mtime_ns))
    return path.stat()


def _hold(path, conn):
    lease = editing.ArtifactLease(path)
    lease.acquire()
    conn.send("ready")
    conn.recv()
    lease.release(remove=False)


def _inspect_while_child_holds(root, path, observer):
    parent, child = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_hold, args=(path, child))
    proc.start()
    assert parent.recv() == "ready"
    try:
        return _inspect(root, observer=observer)
    finally:
        parent.send("release")
        proc.join(10)
        assert proc.exitcode == 0


@pytest.mark.parametrize("maker", [_stage, _gguf])
def test_old_unlocked_temporary_is_reported_not_removed(tmp_path, maker):
    path = maker(tmp_path)
    before = (path / "payload").read_bytes() if path.is_dir() else path.read_bytes()
    result = _inspect(tmp_path)
    if os.name == "nt":
        assert result["errors"] and not result["abandoned"]
    else:
        assert result["abandoned"] == [str(path)]
    assert result["removed"] == [] and result["removed_count"] == 0
    assert path.exists()
    after = (path / "payload").read_bytes() if path.is_dir() else path.read_bytes()
    assert after == before


@pytest.mark.parametrize("maker", [_stage, _gguf])
def test_recent_temporary_is_preserved(tmp_path, maker):
    path = maker(tmp_path, recent=True)
    result = _inspect(tmp_path)
    if os.name == "nt":
        assert result["errors"] and not result["recent"]
    else:
        assert result["recent"] == [str(path)]
    assert path.exists()


def test_completed_collision_and_hf_cache_are_preserved(tmp_path):
    path = _stage(tmp_path, ".completed.tmp-" + HEX)
    (path / "hf").mkdir()
    (path / "hf" / "config.json").write_bytes(b"config")
    (path / "edit_meta.json").write_text(
        '{"name":".completed.tmp-' + HEX + '"}', encoding="utf-8"
    )
    result = _inspect(tmp_path)
    if os.name == "nt":
        assert result["errors"] and not result["completed"]
    else:
        assert result["completed"] == [str(path)]
    assert (path / "hf" / "config.json").read_bytes() == b"config"


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_completion_marker_and_directories_preserve_atime(tmp_path):
    path = _stage(tmp_path, ".completed.tmp-" + HEX)
    marker = path / "edit_meta.json"
    marker.write_text('{"name":".completed.tmp-' + HEX + '"}', encoding="utf-8")
    before_marker = _set_old_atime(marker)
    before_stage = _set_old_atime(path)
    before_root = _set_old_atime(tmp_path)

    result = _inspect(tmp_path)

    assert result["completed"] == [str(path)] and result["errors"] == []
    after_marker, after_stage, after_root = marker.stat(), path.stat(), tmp_path.stat()
    assert after_marker.st_atime_ns == before_marker.st_atime_ns
    assert after_marker.st_mtime_ns == before_marker.st_mtime_ns
    assert after_marker.st_ctime_ns == before_marker.st_ctime_ns
    assert after_marker.st_size == before_marker.st_size
    assert after_stage.st_atime_ns == before_stage.st_atime_ns
    assert after_root.st_atime_ns == before_root.st_atime_ns


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_abandoned_nested_stage_preserves_all_atimes(tmp_path):
    path = _stage(tmp_path)
    nested = path / "nested"
    nested.mkdir()
    payload = nested / "payload"
    payload.write_bytes(b"nested")
    _mtime(payload)
    _mtime(nested)
    _mtime(path)
    watched = [tmp_path, path, path / "payload", nested, payload]
    before = {item: _set_old_atime(item) for item in watched}

    result = _inspect(tmp_path)

    assert result["abandoned"] == [str(path)] and result["errors"] == []
    for item in watched:
        after = item.stat()
        assert after.st_atime_ns == before[item].st_atime_ns
        assert after.st_mtime_ns == before[item].st_mtime_ns
        assert after.st_ctime_ns == before[item].st_ctime_ns
        assert after.st_size == before[item].st_size
    assert payload.read_bytes() == b"nested"


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_candidate_free_hierarchy_preserves_directory_atimes(tmp_path):
    hf = tmp_path / "job" / "hf"
    hf.mkdir(parents=True)
    (hf / "config.json").write_bytes(b"cache")
    watched = [tmp_path, tmp_path / "job", hf]
    before = {item: _set_old_atime(item) for item in watched}

    result = _inspect(tmp_path)

    assert not any(result[key] for key in (
        "active", "recent", "completed", "abandoned", "changed_or_unsafe",
    ))
    assert result["errors"] == []
    assert all(item.stat().st_atime_ns == before[item].st_atime_ns for item in watched)


def test_unavailable_strict_inspection_does_not_traverse(tmp_path, monkeypatch):
    _gguf(tmp_path)
    before = _set_old_atime(tmp_path)
    monkeypatch.setattr(editing, "_strict_inspection_capability", lambda: False)
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(os, "scandir", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(os, "walk", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path, "iterdir", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    result = _inspect(tmp_path)

    assert result["errors"] and "unavailable" in result["errors"][0]["error"]
    assert tmp_path.stat().st_atime_ns == before.st_atime_ns


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_zero_noatime_capability_fails_before_traversal(tmp_path, monkeypatch):
    child = tmp_path / "job" / "hf"
    child.mkdir(parents=True)
    (child / "config.json").write_bytes(b"cache")
    watched = [tmp_path, tmp_path / "job", child]
    before = {path: _set_old_atime(path).st_atime_ns for path in watched}
    monkeypatch.setattr(os, "O_NOATIME", 0)
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(os, "scandir", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(os, "walk", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path, "iterdir", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path, "rglob", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    result = _inspect(tmp_path)

    assert editing._strict_inspection_capability() is False
    assert result["errors"] and "unavailable" in result["errors"][0]["error"]
    assert not any(result[key] for key in (
        "active", "recent", "completed", "abandoned", "changed_or_unsafe",
    ))
    assert all(path.stat().st_atime_ns == before[path] for path in watched)


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_missing_fd_scandir_capability_fails_before_traversal(tmp_path, monkeypatch):
    child = tmp_path / "job"
    child.mkdir()
    before = {path: _set_old_atime(path).st_atime_ns for path in (tmp_path, child)}
    monkeypatch.setattr(os, "supports_fd", set(os.supports_fd) - {os.scandir})
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(os, "scandir", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))

    result = _inspect(tmp_path)

    assert editing._strict_inspection_capability() is False
    assert result["errors"] and "unavailable" in result["errors"][0]["error"]
    assert not any(result[key] for key in (
        "active", "recent", "completed", "abandoned", "changed_or_unsafe",
    ))
    assert all(path.stat().st_atime_ns == before[path] for path in (tmp_path, child))


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_advertised_fd_scandir_runtime_failure_is_controlled(tmp_path, monkeypatch):
    child = tmp_path / "job"
    child.mkdir()
    before = {path: _set_old_atime(path).st_atime_ns for path in (tmp_path, child)}
    real_open = os.open
    opened = []

    def tracking_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def reject_fd(path):
        if isinstance(path, int):
            raise TypeError("fd scandir unavailable")
        raise AssertionError("pathname scandir fallback attempted")

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "scandir", reject_fd)
    monkeypatch.setattr(os, "supports_fd", set(os.supports_fd) | {reject_fd})

    result = _inspect(tmp_path)

    assert result["errors"] and "fd-based os.scandir" in result["errors"][0]["error"]
    assert not any(result[key] for key in (
        "active", "recent", "completed", "abandoned", "changed_or_unsafe",
    ))
    assert opened
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert all(path.stat().st_atime_ns == before[path] for path in (tmp_path, child))


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_root_noatime_denial_has_no_fallback(tmp_path, monkeypatch):
    _gguf(tmp_path)
    before = _set_old_atime(tmp_path)
    real_open = os.open
    calls = []

    def deny_root(path, flags, *args, **kwargs):
        calls.append(flags)
        if Path(path) == tmp_path and flags & os.O_NOATIME:
            raise PermissionError(errno.EPERM, "no-atime denied", str(path))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_root)
    result = _inspect(tmp_path)
    assert result["errors"] and "denied" in result["errors"][0]["error"]
    assert calls and all(flags & os.O_NOATIME for flags in calls)
    assert tmp_path.stat().st_atime_ns == before.st_atime_ns


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
@pytest.mark.parametrize("held", [False, True])
def test_lease_probe_preserves_exact_atime(tmp_path, held):
    path = _gguf(tmp_path)
    lease = editing.ArtifactLease(path)
    lease.acquire()
    lease_path = lease.path
    parent = child = proc = None
    if held:
        lease.release()
        parent, child = multiprocessing.Pipe()
        proc = multiprocessing.Process(target=_hold, args=(path, child))
        proc.start()
        assert parent.recv() == "ready"
    else:
        lease.release()
    before = _set_old_atime(lease_path)
    try:
        result = _inspect(tmp_path)
    finally:
        if held:
            parent.send("release")
            proc.join(10)
            assert proc.exitcode == 0
    assert result["active" if held else "abandoned"] == [str(path)]
    assert lease_path.stat().st_atime_ns == before.st_atime_ns


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_nested_noatime_denial_has_no_normal_open_fallback(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    _gguf(blocked)
    safe = _gguf(tmp_path / "safe", ".safe.tmp-" + HEX + ".gguf")
    blocked_before = _set_old_atime(blocked)
    real_open = os.open

    def deny_nested(path, flags, *args, **kwargs):
        if path == "blocked" and flags & os.O_NOATIME:
            raise PermissionError(errno.EPERM, "nested no-atime denied", path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_nested)
    result = _inspect(tmp_path)
    assert result["abandoned"] == [str(safe)]
    assert any("nested no-atime denied" in item["error"] for item in result["errors"])
    assert blocked.stat().st_atime_ns == blocked_before.st_atime_ns


@pytest.mark.skipif(os.name != "posix", reason="strict O_NOATIME inspection is POSIX-only")
def test_marker_noatime_denial_preserves_marker_and_candidate(tmp_path, monkeypatch):
    path = _stage(tmp_path, ".completed.tmp-" + HEX)
    marker = path / "edit_meta.json"
    marker.write_text('{"name":".completed.tmp-' + HEX + '"}', encoding="utf-8")
    before = _set_old_atime(marker)
    original = marker.read_bytes()
    before = _set_old_atime(marker)  # reset after the test's own byte read
    real_open = os.open

    def deny_marker(name, flags, *args, **kwargs):
        if name == "edit_meta.json" and flags & os.O_NOATIME:
            raise PermissionError(errno.EPERM, "marker no-atime denied", name)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", deny_marker)
    result = _inspect(tmp_path)
    assert result["changed_or_unsafe"] == [str(path)]
    assert any("marker no-atime denied" in item["error"] for item in result["errors"])
    assert marker.stat().st_atime_ns == before.st_atime_ns
    assert marker.read_bytes() == original


def test_inspection_never_calls_destructive_operations(tmp_path, monkeypatch):
    path = _gguf(tmp_path)
    forbidden = lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    monkeypatch.setattr(Path, "replace", forbidden)
    monkeypatch.setattr(Path, "unlink", forbidden)
    monkeypatch.setattr(Path, "touch", forbidden)
    for name in (
        "replace", "rename", "unlink", "remove", "rmdir", "utime", "chmod",
        "chown", "fchmod", "fchown", "truncate", "ftruncate",
    ):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(shutil, "rmtree", forbidden)
    result = _inspect(tmp_path)
    if os.name == "nt":
        assert result["errors"] and not result["abandoned"]
    else:
        assert result["abandoned"] == [str(path)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory-lock integration")
def test_cross_process_active_lease_is_preserved_then_reported_abandoned(tmp_path):
    path = _gguf(tmp_path)
    parent, child = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_hold, args=(path, child))
    proc.start()
    assert parent.recv() == "ready"
    first = _inspect(tmp_path)
    assert first["active"] == [str(path)]
    assert path.exists() and path.with_name(path.name + ".lease").exists()
    parent.send("release")
    proc.join(10)
    assert proc.exitcode == 0
    second = _inspect(tmp_path)
    assert second["abandoned"] == [str(path)]
    assert path.exists() and path.with_name(path.name + ".lease").exists()


@pytest.mark.skipif(os.name == "nt", reason="Windows fails closed before candidate hooks")
def test_same_inode_modification_after_discovery_is_unsafe(tmp_path):
    path = _gguf(tmp_path)
    def observer(phase, _value):
        if phase == "before_inspect":
            path.write_bytes(b"modified-same-inode")
    result = _inspect(tmp_path, observer=observer)
    assert result["changed_or_unsafe"] == [str(path)]
    assert path.read_bytes() == b"modified-same-inode"


@pytest.mark.skipif(os.name == "nt", reason="Windows fails closed before candidate hooks")
def test_candidate_replacement_is_unsafe_and_target_survives(tmp_path):
    path = _gguf(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"target")
    def observer(phase, _value):
        if phase == "before_inspect":
            path.rename(tmp_path / "original")
            path.symlink_to(target)
    result = _inspect(tmp_path, observer=observer)
    assert result["changed_or_unsafe"] == [str(path)]
    assert path.is_symlink() and target.read_bytes() == b"target"


@pytest.mark.skipif(os.name == "nt", reason="Windows fails closed before marker inspection")
def test_parent_change_during_marker_inspection_is_unsafe(tmp_path):
    parent = tmp_path / "nested"
    path = _stage(parent)
    (path / "edit_meta.json").write_text('{"name":"normal"}', encoding="utf-8")
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    def observer(phase, _value):
        if phase == "before_marker_open":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)
    result = _inspect(tmp_path, observer=observer)
    assert str(path) in result["changed_or_unsafe"]
    assert (moved / path.name / "payload").read_bytes() == b"directory-bytes"
    assert parent.is_symlink()


def test_root_symlink_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    path = _gguf(outside)
    root = tmp_path / "root"
    root.symlink_to(outside, target_is_directory=True)
    result = _inspect(root)
    assert result["errors"] and path.exists()


def test_reserved_names_rejected():
    with pytest.raises(ValueError, match="reserved"):
        editing.validate_export_name("nested/.x.tmp-" + HEX)
    with pytest.raises(ValueError, match="reserved"):
        editing.validate_export_name(".x.tmp-" + HEX + ".gguf")


@pytest.mark.skipif(os.name == "nt", reason="POSIX persistent lease policy")
def test_owned_lease_release_retains_reusable_lease(tmp_path):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    lease = editing.ArtifactLease(artifact)
    lease.acquire()
    assert lease.path.exists()
    lease.release()
    assert lease.path.is_file()
    second = editing.ArtifactLease(artifact)
    assert second.acquire(blocking=False)
    second.release()
    assert lease.path.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX persistent lease policy")
def test_lease_parent_change_does_not_unlink_unrelated_lease(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    artifact = parent / (".x.tmp-" + HEX + ".gguf")
    lease = editing.ArtifactLease(artifact)
    lease.acquire()
    moved = tmp_path / "moved"
    parent.rename(moved)
    parent.mkdir()
    unrelated = parent / lease.path.name
    unrelated.write_bytes(b"unrelated")
    lease.release()
    assert unrelated.read_bytes() == b"unrelated"
    assert (moved / lease.path.name).is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow lease acquisition")
def test_preexisting_lease_symlink_is_rejected_without_touching_target(tmp_path):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    target = tmp_path / "target"
    target.write_bytes(b"")
    lease_path = artifact.with_name(artifact.name + ".lease")
    lease_path.symlink_to(target)
    with pytest.raises((OSError, RuntimeError)):
        editing.ArtifactLease(artifact).acquire()
    assert lease_path.is_symlink()
    assert target.read_bytes() == b""


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow lease acquisition")
@pytest.mark.parametrize("kind", ["directory", "fifo", "socket"])
def test_preexisting_nonregular_lease_is_rejected(tmp_path, kind):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    lease_path = artifact.with_name(artifact.name + ".lease")
    sock = None
    if kind == "directory":
        lease_path.mkdir()
    elif kind == "fifo":
        os.mkfifo(lease_path)
    else:
        sock = socket.socket(socket.AF_UNIX)
        short_socket = tmp_path / "socket"
        sock.bind(str(short_socket))
        short_socket.rename(lease_path)
    try:
        with pytest.raises((OSError, RuntimeError)):
            editing.ArtifactLease(artifact).acquire()
    finally:
        if sock is not None:
            sock.close()
    assert lease_path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX persistent lease policy")
def test_posix_release_never_uses_path_unlink(tmp_path, monkeypatch):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    lease = editing.ArtifactLease(artifact)
    lease.acquire()
    monkeypatch.setattr(os, "unlink", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    lease.release()
    assert lease.path.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock behavior")
def test_failed_initial_lock_leaves_closed_reusable_regular_lease(tmp_path, monkeypatch):
    import fcntl

    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    real_flock = fcntl.flock
    monkeypatch.setattr(fcntl, "flock", lambda *_: (_ for _ in ()).throw(OSError("lock failed")))
    with pytest.raises(OSError, match="lock failed"):
        editing.ArtifactLease(artifact).acquire()
    lease_path = artifact.with_name(artifact.name + ".lease")
    assert lease_path.is_file()
    monkeypatch.setattr(fcntl, "flock", real_flock)
    retry = editing.ArtifactLease(artifact)
    assert retry.acquire(blocking=False)
    retry.release()


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock behavior")
def test_unlock_failure_closes_descriptor_and_preserves_primary_error(tmp_path, monkeypatch):
    import fcntl

    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    lease = editing.ArtifactLease(artifact)
    lease.acquire()
    fd = lease._file.fileno()
    real_flock = fcntl.flock

    def fail_unlock(target_fd, operation):
        if operation == fcntl.LOCK_UN:
            raise OSError("unlock failed")
        return real_flock(target_fd, operation)

    monkeypatch.setattr(fcntl, "flock", fail_unlock)
    with pytest.raises(OSError, match="unlock failed"):
        lease.release()
    assert lease._file is None and lease._parent_fd is None
    with pytest.raises(OSError):
        os.fstat(fd)
    lease.release()  # idempotent

    other = editing.ArtifactLease(artifact)
    with pytest.raises(RuntimeError, match="primary export failure"):
        with other:
            raise RuntimeError("primary export failure")
    assert other._file is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX completion-marker inspection")
def test_completion_marker_same_inode_rewrite_is_unsafe(tmp_path):
    path = _stage(tmp_path, ".completed.tmp-" + HEX)
    marker = path / "edit_meta.json"
    marker.write_text('{"name":".completed.tmp-' + HEX + '"}', encoding="utf-8")

    def observer(phase, _value):
        if phase == "after_marker_open":
            marker.write_text('{"name":"changed","padding":"different"}', encoding="utf-8")

    result = _inspect(tmp_path, observer=observer)
    assert result["completed"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert marker.read_text(encoding="utf-8").startswith('{"name":"changed"')


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease classification")
def test_active_lease_wins_over_mutable_candidate_metadata(tmp_path):
    path = _gguf(tmp_path)
    lease = editing.ArtifactLease(path)
    lease.acquire()

    def observer(phase, _value):
        if phase == "before_inspect":
            path.write_bytes(b"actively-updated")

    try:
        result = _inspect(tmp_path, observer=observer)
    finally:
        lease.release()
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert path.read_bytes() == b"actively-updated"


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease proof")
def test_held_lease_candidate_leaf_replacement_is_unsafe(tmp_path):
    path = _gguf(tmp_path)
    original = tmp_path / "original"
    target = tmp_path / "target"
    target.write_bytes(b"target")

    def observer(phase, _value):
        if phase == "after_held_lease_probe":
            path.rename(original)
            path.symlink_to(target)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed" in result["errors"][0]["error"]
    assert original.read_bytes() == b"file-bytes"
    assert path.is_symlink() and target.read_bytes() == b"target"


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease proof")
def test_held_lease_parent_replacement_is_unsafe(tmp_path):
    parent = tmp_path / "nested"
    path = _gguf(parent)
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()

    def observer(phase, _value):
        if phase == "after_held_lease_probe":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed" in result["errors"][0]["error"]
    assert (moved / path.name).read_bytes() == b"file-bytes"
    assert (moved / (path.name + ".lease")).is_file()
    assert parent.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease proof")
def test_held_lease_leaf_replacement_is_unsafe(tmp_path):
    path = _gguf(tmp_path)
    lease_path = path.with_name(path.name + ".lease")
    original_lease = tmp_path / "original.lease"
    replacement = b"replacement-lease"

    def observer(phase, _value):
        if phase == "after_held_lease_probe":
            lease_path.rename(original_lease)
            lease_path.write_bytes(replacement)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "lease changed" in result["errors"][0]["error"]
    assert original_lease.is_file()
    assert lease_path.read_bytes() == replacement


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease proof")
def test_child_held_lease_allows_mutable_candidate_metadata(tmp_path):
    path = _gguf(tmp_path)

    def observer(phase, _value):
        if phase == "after_held_lease_probe":
            path.write_bytes(b"updated-with-same-inode-and-new-size")

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert path.read_bytes() == b"updated-with-same-inode-and-new-size"


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease classification")
def test_held_gguf_mutation_at_discovery_is_active(tmp_path):
    path = _gguf(tmp_path)
    discovered = path.lstat()

    def observer(phase, _value):
        if phase == "discovered":
            path.write_bytes(b"active-write-after-discovery-with-new-size")

    result = _inspect_while_child_holds(tmp_path, path, observer)
    current = path.lstat()
    assert (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode)) == (
        discovered.st_dev, discovered.st_ino, stat.S_IFMT(discovered.st_mode),
    )
    assert current.st_size != discovered.st_size
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert not [error for error in result["errors"] if error["path"] == str(path)]
    assert path.read_bytes() == b"active-write-after-discovery-with-new-size"
    assert path.with_name(path.name + ".lease").is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX active lease classification")
def test_held_stage_content_mutation_at_discovery_is_active(tmp_path):
    path = _stage(tmp_path)
    discovered = path.lstat()

    def observer(phase, _value):
        if phase == "discovered":
            (path / "new-child").write_bytes(b"active-child")

    result = _inspect_while_child_holds(tmp_path, path, observer)
    current = path.lstat()
    assert (current.st_dev, current.st_ino, stat.S_IFMT(current.st_mode)) == (
        discovered.st_dev, discovered.st_ino, stat.S_IFMT(discovered.st_mode),
    )
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert (path / "new-child").read_bytes() == b"active-child"
    assert path.with_name(path.name + ".lease").is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX inactive identity classification")
def test_unleased_gguf_mutation_at_discovery_is_unsafe(tmp_path):
    path = _gguf(tmp_path)

    def observer(phase, _value):
        if phase == "discovered":
            path.write_bytes(b"inactive-write-after-discovery")

    result = _inspect(tmp_path, observer=observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed after discovery" in result["errors"][0]["error"]
    assert path.read_bytes() == b"inactive-write-after-discovery"


@pytest.mark.skipif(os.name == "nt", reason="POSIX inactive identity classification")
def test_unlocked_lease_gguf_mutation_at_discovery_is_unsafe(tmp_path):
    path = _gguf(tmp_path)
    lease = editing.ArtifactLease(path)
    lease.acquire()
    lease.release()
    lease_path = path.with_name(path.name + ".lease")

    def observer(phase, _value):
        if phase == "discovered":
            path.write_bytes(b"unlocked-write-after-discovery")

    result = _inspect(tmp_path, observer=observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed after discovery" in result["errors"][0]["error"]
    assert path.read_bytes() == b"unlocked-write-after-discovery"
    assert lease_path.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX inactive identity classification")
def test_unleased_stage_content_mutation_at_discovery_is_unsafe(tmp_path):
    path = _stage(tmp_path)

    def observer(phase, _value):
        if phase == "discovered":
            (path / "new-child").write_bytes(b"inactive-child")
            forced = int((NOW + 100) * 1_000_000_000)
            os.utime(path, ns=(forced, forced))

    result = _inspect(tmp_path, observer=observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed after discovery" in result["errors"][0]["error"]
    assert (path / "new-child").read_bytes() == b"inactive-child"


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
def test_active_checks_detect_candidate_replacement_before_last_check(tmp_path):
    path = _gguf(tmp_path)
    original = tmp_path / "candidate-original"
    target = tmp_path / "candidate-target"
    target.write_bytes(b"target")

    def observer(phase, _value):
        if phase == "after_active_candidate_check_1":
            path.rename(original)
            path.symlink_to(target)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed" in result["errors"][0]["error"]
    assert original.read_bytes() == b"file-bytes"
    assert path.is_symlink() and target.read_bytes() == b"target"


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
def test_active_checks_detect_lease_replacement_before_last_check(tmp_path):
    path = _gguf(tmp_path)
    lease_path = path.with_name(path.name + ".lease")
    original = tmp_path / "lease-original"
    target = tmp_path / "lease-target"
    target.write_bytes(b"target")

    def observer(phase, _value):
        if phase == "after_active_lease_entry_check_1":
            lease_path.rename(original)
            lease_path.symlink_to(target)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "lease changed" in result["errors"][0]["error"]
    assert original.is_file()
    assert lease_path.is_symlink() and target.read_bytes() == b"target"


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
def test_active_checks_detect_parent_replacement_before_last_chain_check(tmp_path):
    parent = tmp_path / "nested"
    path = _gguf(parent)
    moved = tmp_path / "nested-original"
    replacement = tmp_path / "nested-target"
    replacement.mkdir()

    def observer(phase, _value):
        if phase == "after_active_lease_entry_check_2":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == []
    assert result["changed_or_unsafe"] == [str(path)]
    assert "changed" in result["errors"][0]["error"]
    assert (moved / path.name).read_bytes() == b"file-bytes"
    assert (moved / (path.name + ".lease")).is_file()
    assert parent.is_symlink()


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
def test_parent_timestamps_are_not_required_as_namespace_generations(tmp_path):
    path = _gguf(tmp_path)

    def observer(phase, _value):
        if phase == "after_active_lease_entry_check_2":
            transient = tmp_path / "transient"
            transient.write_bytes(b"transient")
            transient.unlink()

    result = _inspect_while_child_holds(tmp_path, path, observer)
    # A rapid create/remove need not advance directory timestamps on every
    # filesystem. Ordered structural checks still preserve every object, and an
    # advisory active result is permitted when no inconsistency is observed.
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert path.read_bytes() == b"file-bytes"


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
def test_active_result_is_advisory_after_ordered_checks(tmp_path):
    path = _gguf(tmp_path)

    def observer(phase, _value):
        if phase == "before_active_result":
            # This mutation is after this entry's last ordered check. There is no
            # atomic snapshot, so advisory active is permitted and remains
            # strictly non-destructive.
            path.write_bytes(b"changed-after-observation")

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert path.read_bytes() == b"changed-after-observation"


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
@pytest.mark.parametrize("entry", ["candidate", "lease"])
def test_replacement_after_entry_last_check_may_return_advisory_active(tmp_path, entry):
    path = _gguf(tmp_path)
    lease_path = path.with_name(path.name + ".lease")
    changed = tmp_path / f"{entry}-original"
    target = tmp_path / f"{entry}-target"
    target.write_bytes(b"target")
    phase_name = (
        "after_active_candidate_check_2"
        if entry == "candidate"
        else "after_active_lease_entry_check_2"
    )

    def observer(phase, _value):
        if phase == phase_name:
            selected = path if entry == "candidate" else lease_path
            selected.rename(changed)
            selected.symlink_to(target)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert changed.exists()
    selected = path if entry == "candidate" else lease_path
    assert selected.is_symlink() and target.read_bytes() == b"target"


@pytest.mark.skipif(os.name == "nt", reason="POSIX advisory active checks")
def test_parent_replacement_after_last_chain_check_may_return_advisory_active(tmp_path):
    parent = tmp_path / "nested"
    path = _gguf(parent)
    moved = tmp_path / "nested-original"
    replacement = tmp_path / "nested-target"
    replacement.mkdir()

    def observer(phase, _value):
        if phase == "before_active_result":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)

    result = _inspect_while_child_holds(tmp_path, path, observer)
    assert result["active"] == [str(path)]
    assert result["changed_or_unsafe"] == []
    assert (moved / path.name).read_bytes() == b"file-bytes"
    assert parent.is_symlink() and replacement.is_dir()


@pytest.mark.skipif(os.name == "nt", reason="POSIX validation hook")
def test_late_lease_validation_failure_clears_acquisition_state(tmp_path, monkeypatch):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    lease = editing.ArtifactLease(artifact)
    real_validator = editing._validate_acquired_lease_file
    monkeypatch.setattr(
        editing,
        "_validate_acquired_lease_file",
        lambda _file: (_ for _ in ()).throw(RuntimeError("late validation failed")),
    )
    with pytest.raises(RuntimeError, match="late validation failed"):
        lease.acquire()
    assert lease._file is None
    assert lease._parent_fd is None
    assert lease._identity is None
    assert lease._created is False
    lease.release()
    monkeypatch.setattr(editing, "_validate_acquired_lease_file", real_validator)
    retry = editing.ArtifactLease(artifact)
    assert retry.acquire(blocking=False)
    retry.release()


@pytest.mark.skipif(os.name != "nt", reason="Windows fails closed without safe handles")
def test_windows_inspection_preserves_candidates(tmp_path):
    path = _gguf(tmp_path)
    result = _inspect(tmp_path)
    assert result["changed_or_unsafe"] == []
    assert result["errors"] and "unavailable" in result["errors"][0]["error"]
    assert path.read_bytes() == b"file-bytes"


@pytest.mark.skipif(os.name != "nt", reason="native Windows reparse validation")
def test_windows_lease_symlink_is_rejected_without_touching_target(tmp_path):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    target = tmp_path / "lease-target"
    target.write_bytes(b"unchanged")
    lease_path = artifact.with_name(artifact.name + ".lease")
    try:
        lease_path.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"Windows runner cannot create symlink fixture: {exc}")
    with pytest.raises((OSError, RuntimeError), match="reparse|safe|normal"):
        editing.ArtifactLease(artifact).acquire()
    assert lease_path.is_symlink()
    assert target.read_bytes() == b"unchanged"
