import multiprocessing
import os
import shutil
import socket
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
    category = "changed_or_unsafe" if os.name == "nt" else "abandoned"
    assert result[category] == [str(path)]
    assert result["removed"] == [] and result["removed_count"] == 0
    assert path.exists()
    after = (path / "payload").read_bytes() if path.is_dir() else path.read_bytes()
    assert after == before


@pytest.mark.parametrize("maker", [_stage, _gguf])
def test_recent_temporary_is_preserved(tmp_path, maker):
    path = maker(tmp_path, recent=True)
    category = "changed_or_unsafe" if os.name == "nt" else "recent"
    assert _inspect(tmp_path)[category] == [str(path)]
    assert path.exists()


def test_completed_collision_and_hf_cache_are_preserved(tmp_path):
    path = _stage(tmp_path, ".completed.tmp-" + HEX)
    (path / "hf").mkdir()
    (path / "hf" / "config.json").write_bytes(b"config")
    (path / "edit_meta.json").write_text(
        '{"name":".completed.tmp-' + HEX + '"}', encoding="utf-8"
    )
    result = _inspect(tmp_path)
    category = "changed_or_unsafe" if os.name == "nt" else "completed"
    assert result[category] == [str(path)]
    assert (path / "hf" / "config.json").read_bytes() == b"config"


def test_inspection_never_calls_destructive_operations(tmp_path, monkeypatch):
    path = _gguf(tmp_path)
    monkeypatch.setattr(Path, "replace", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(Path, "unlink", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(os, "unlink", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    category = "changed_or_unsafe" if os.name == "nt" else "abandoned"
    assert _inspect(tmp_path)[category] == [str(path)]


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
    assert result["changed_or_unsafe"] == [str(path)]
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
