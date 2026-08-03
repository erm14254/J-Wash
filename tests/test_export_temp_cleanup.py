import os
import json
import multiprocessing
import subprocess
from pathlib import Path

import pytest

from core import editing


HEX = "a" * 32
OLD = 1_000.0
NOW = OLD + editing.DEFAULT_TEMP_STALE_AGE + 10


def _mtime(path, value=OLD):
    os.utime(path, (value, value))


def _stage(root, name=f".model.tmp-{HEX}", *, mtime=OLD):
    path = root / name
    path.mkdir(parents=True)
    _mtime(path, mtime)
    return path


def _gguf(root, name=f".model-bf16.tmp-{HEX}.gguf", *, mtime=OLD):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"partial")
    _mtime(path, mtime)
    return path


def _cleanup(root, **kwargs):
    return editing.cleanup_abandoned_export_temps(root=root, now=NOW, **kwargs)


def _hold_artifact_lease(path, connection):
    lease = editing.ArtifactLease(path)
    lease.acquire()
    connection.send("ready")
    connection.recv()
    lease.release(remove=False)
    connection.close()


def test_removes_old_recognized_legacy_staging_directory(tmp_path):
    path = _stage(tmp_path)
    result = _cleanup(tmp_path)
    assert not path.exists()
    assert result["removed"] == [str(path)]
    assert result["removed_count"] == 1


def test_removes_old_recognized_legacy_gguf_file(tmp_path):
    path = _gguf(tmp_path)
    assert _cleanup(tmp_path)["removed"] == [str(path)]
    assert not path.exists()


@pytest.mark.parametrize("maker", [_stage, _gguf])
def test_preserves_recent_legacy_artifact(tmp_path, maker):
    path = maker(tmp_path, mtime=NOW - 10)
    result = _cleanup(tmp_path)
    assert path.exists()
    assert result["skipped_recent"] == [str(path)]


@pytest.mark.parametrize("name", [
    ".x.tmp-short", ".x.tmp-" + "A" * 32, ".x.tmp-" + "a" * 31,
    ".x.tmp-" + "a" * 33, ".x.tmp-" + "g" * 32,
    ".x.tmp-" + "a" * 32 + ".bin", "x.tmp-" + "a" * 32,
])
def test_preserves_malformed_temporary_names(tmp_path, name):
    path = tmp_path / name
    if name.endswith(".bin"):
        path.write_bytes(b"x")
    else:
        path.mkdir()
    _cleanup(tmp_path)
    assert path.exists()


def test_preserves_completed_outputs_and_hf_cache(tmp_path):
    export = tmp_path / "job"
    hf = export / "hf"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text("{}")
    final = export / "job-bf16.gguf"
    final.write_bytes(b"valid")
    hidden = tmp_path / ".notes"
    hidden.write_text("keep")
    result = _cleanup(tmp_path)
    assert export.is_dir() and hf.is_dir() and final.is_file() and hidden.is_file()
    assert result["removed_count"] == 0


def test_nested_export_temporary_is_cleaned(tmp_path):
    path = _stage(tmp_path / "outer" / "inner")
    assert _cleanup(tmp_path)["removed"] == [str(path)]


def test_matching_symlinks_are_never_followed_or_deleted(tmp_path):
    outside = tmp_path.parent / f"outside-{tmp_path.name}"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep")
    link = tmp_path / f".escape.tmp-{HEX}"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    _cleanup(tmp_path)
    assert link.is_symlink() and sentinel.read_text() == "keep"


def test_candidate_resolving_outside_root_is_refused(tmp_path, monkeypatch):
    path = _stage(tmp_path)
    original = Path.resolve
    outside = tmp_path.parent / "outside"

    def resolve(candidate, *args, **kwargs):
        if candidate == path:
            return outside
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    result = _cleanup(tmp_path)
    assert path.exists()
    assert "escapes" in result["errors"][0]["error"]


def test_one_deletion_error_does_not_stop_other_cleanup(tmp_path, monkeypatch):
    bad = _stage(tmp_path, f".bad.tmp-{HEX}")
    good = _stage(tmp_path, f".good.tmp-{'b' * 32}")
    original = editing.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path) == bad:
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(editing.shutil, "rmtree", rmtree)
    result = _cleanup(tmp_path)
    assert bad.exists() and not good.exists()
    assert result["removed_count"] == 1
    assert result["errors"] == [{"path": str(bad), "error": "denied"}]


def test_newest_contained_activity_controls_legacy_staleness(tmp_path):
    path = _stage(tmp_path)
    child = path / "recent"
    child.write_text("active")
    _mtime(path, OLD)
    _mtime(child, NOW - 1)
    result = _cleanup(tmp_path)
    assert path.exists() and result["skipped_recent"] == [str(path)]


def test_cleanup_skips_held_lease_then_removes_released_artifact(tmp_path):
    path = _stage(tmp_path, mtime=NOW)
    lease = editing.ArtifactLease(path)
    assert lease.acquire()
    try:
        result = _cleanup(tmp_path)
        assert path.exists()
        assert result["skipped_active"] == [str(path)]
    finally:
        lease.release(remove=False)
    result = _cleanup(tmp_path)
    assert not path.exists() and not lease.path.exists()
    assert result["removed"] == [str(path)]


def test_export_rebase_lease_removed_after_success(tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path)
    monkeypatch.setattr(editing, "_export_rebase_impl", lambda *a, out_dir, **k: {"ok": True})
    result = editing.export_rebase([], object(), {}, fmt="full", name="nested/model")
    final = tmp_path / "nested" / "model"
    assert result["out_dir"] == str(final.resolve())
    assert final.is_dir()
    assert not list(final.parent.glob("*.lease"))
    assert not list(final.rglob("*.lease"))


def test_export_rebase_lease_removed_after_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("export failed")

    monkeypatch.setattr(editing, "_export_rebase_impl", fail)
    with pytest.raises(RuntimeError, match="export failed"):
        editing.export_rebase([], object(), {}, fmt="full", name="model")
    assert not (tmp_path / "model").exists()
    assert not list(tmp_path.iterdir())


def test_lease_acquisition_failure_prevents_staging_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path)

    def fail(self, *, blocking=True):
        raise OSError("lock unavailable")

    monkeypatch.setattr(editing.ArtifactLease, "acquire", fail)
    with pytest.raises(OSError, match="lock unavailable"):
        editing.export_rebase([], object(), {}, fmt="full", name="model")
    assert not (tmp_path / "model").exists()
    assert not any(path.name.startswith(".model.tmp-") for path in tmp_path.iterdir())


def test_cleanup_lock_error_preserves_artifact(tmp_path, monkeypatch):
    path = _stage(tmp_path)
    lease_path = path.with_name(path.name + ".lease")
    lease_path.write_bytes(b"0")

    def fail(self, *, blocking=True):
        raise OSError("lock inspection failed")

    monkeypatch.setattr(editing.ArtifactLease, "acquire", fail)
    result = _cleanup(tmp_path)
    assert path.exists() and lease_path.exists()
    assert result["removed_count"] == 0
    assert result["errors"] == [{"path": str(path), "error": "lock inspection failed"}]


def test_symlink_cleanup_root_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = _stage(outside)
    root = tmp_path / "root"
    try:
        root.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    result = _cleanup(root)
    assert artifact.exists()
    assert "symlink or junction" in result["errors"][0]["error"]


def test_self_referential_cleanup_root_returns_safely(tmp_path):
    root = tmp_path / "root"
    try:
        root.symlink_to(root, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    result = _cleanup(root)
    assert result["removed_count"] == 0
    assert "symlink or junction" in result["errors"][0]["error"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_windows_junction_cleanup_root_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    artifact = _stage(outside)
    root = tmp_path / "root"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(outside)],
        capture_output=True, text=True,
    )
    if result.returncode:
        pytest.skip(f"junction creation unavailable: {result.stderr}")
    cleanup = _cleanup(root)
    assert artifact.exists()
    assert "symlink or junction" in cleanup["errors"][0]["error"]


@pytest.mark.parametrize("name", [
    f".completed.tmp-{HEX}", f"nested/.completed.tmp-{HEX}",
    f".model.tmp-{HEX}.gguf", f".model.tmp-{HEX}.lease",
    f".model.tmp-{HEX}.gguf.lease",
])
def test_export_names_reserve_internal_temporary_namespace(name):
    with pytest.raises(ValueError, match="reserved for internal temporary export"):
        editing.validate_export_name(name)


def test_export_rebase_rejects_reserved_final_name(tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path)
    with pytest.raises(ValueError, match="reserved for internal temporary export"):
        editing.export_rebase([], object(), {}, fmt="full", name=f".done.tmp-{HEX}")
    assert not list(tmp_path.iterdir())


def test_legacy_completed_colliding_export_is_preserved(tmp_path):
    path = _stage(tmp_path, f".completed.tmp-{HEX}")
    hf = path / "hf"
    hf.mkdir()
    sentinel = hf / "config.json"
    sentinel.write_text("{}")
    (path / "edit_meta.json").write_text(json.dumps({"name": path.name}))
    _mtime(path / "edit_meta.json", OLD)
    _mtime(sentinel, OLD)
    _mtime(hf, OLD)
    _mtime(path, OLD)
    result = _cleanup(tmp_path)
    assert result["skipped_completed"] == [str(path)]
    assert sentinel.read_text() == "{}"


def test_abandoned_stage_with_final_name_marker_is_removed(tmp_path):
    path = _stage(tmp_path)
    marker = path / "edit_meta.json"
    marker.write_text(json.dumps({"name": "model"}))
    _mtime(marker, OLD)
    _mtime(path, OLD)
    assert _cleanup(tmp_path)["removed"] == [str(path)]


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock regression")
def test_new_lease_is_removed_when_flock_fails(tmp_path, monkeypatch):
    import fcntl
    artifact = tmp_path / f".model.tmp-{HEX}"
    lease = editing.ArtifactLease(artifact)
    def fail(*args):
        raise OSError("flock exploded")
    monkeypatch.setattr(fcntl, "flock", fail)
    with pytest.raises(OSError, match="flock exploded"):
        lease.acquire()
    assert not lease.path.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock regression")
def test_preexisting_lease_survives_failed_flock(tmp_path, monkeypatch):
    import fcntl
    artifact = tmp_path / f".model.tmp-{HEX}"
    lease = editing.ArtifactLease(artifact)
    lease.path.write_bytes(b"0")
    monkeypatch.setattr(fcntl, "flock", lambda *args: (_ for _ in ()).throw(OSError("busy")))
    with pytest.raises(OSError, match="busy"):
        lease.acquire()
    assert lease.path.read_bytes() == b"0"


@pytest.mark.skipif(os.name != "nt", reason="Windows locking regression")
def test_new_lease_is_removed_when_windows_locking_fails(tmp_path, monkeypatch):
    import msvcrt
    artifact = tmp_path / f".model.tmp-{HEX}"
    lease = editing.ArtifactLease(artifact)
    def fail(*args):
        raise OSError("locking exploded")
    monkeypatch.setattr(msvcrt, "locking", fail)
    with pytest.raises(OSError, match="locking exploded"):
        lease.acquire()
    assert not lease.path.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows locking regression")
def test_preexisting_lease_survives_failed_windows_locking(tmp_path, monkeypatch):
    import msvcrt
    artifact = tmp_path / f".model.tmp-{HEX}"
    lease = editing.ArtifactLease(artifact)
    lease.path.write_bytes(b"0")
    monkeypatch.setattr(
        msvcrt, "locking", lambda *args: (_ for _ in ()).throw(OSError("busy")),
    )
    with pytest.raises(OSError, match="busy"):
        lease.acquire()
    assert lease.path.read_bytes() == b"0"


def test_cross_process_lease_protects_active_artifact(tmp_path):
    artifact = _stage(tmp_path, mtime=OLD)
    context = multiprocessing.get_context("spawn")
    parent, child_connection = context.Pipe()
    child = context.Process(
        target=_hold_artifact_lease, args=(artifact, child_connection),
    )
    child.start()
    try:
        assert parent.recv() == "ready"
        first = _cleanup(tmp_path)
        assert first["skipped_active"] == [str(artifact)] and artifact.exists()
        parent.send("release")
        child.join(timeout=10)
        assert child.exitcode == 0
        second = _cleanup(tmp_path)
        assert second["removed"] == [str(artifact)] and not artifact.exists()
    finally:
        parent.close()
        child_connection.close()
        if child.is_alive():
            child.kill()
            child.join()
