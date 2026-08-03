import multiprocessing
import os
import shutil
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-relative lease removal")
def test_owned_lease_release_removes_unchanged_lease(tmp_path):
    artifact = tmp_path / (".x.tmp-" + HEX + ".gguf")
    lease = editing.ArtifactLease(artifact)
    lease.acquire()
    assert lease.path.exists()
    lease.release()
    assert not lease.path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor-relative lease removal")
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


@pytest.mark.skipif(os.name != "nt", reason="Windows fails closed without safe handles")
def test_windows_inspection_preserves_candidates(tmp_path):
    path = _gguf(tmp_path)
    result = _inspect(tmp_path)
    assert result["changed_or_unsafe"] == [str(path)]
    assert path.read_bytes() == b"file-bytes"
