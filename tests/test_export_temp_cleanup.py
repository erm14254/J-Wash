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
requires_anchored_deletion = pytest.mark.skipif(
    os.name == "nt",
    reason="Windows cleanup intentionally fails closed without handle-relative deletion",
)


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


@requires_anchored_deletion
def test_removes_old_recognized_legacy_staging_directory(tmp_path):
    path = _stage(tmp_path)
    result = _cleanup(tmp_path)
    assert not path.exists()
    assert result["removed"] == [str(path)]
    assert result["removed_count"] == 1


@requires_anchored_deletion
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


@requires_anchored_deletion
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


@requires_anchored_deletion
def test_candidate_cleanup_never_resolves_leaf(tmp_path, monkeypatch):
    path = _stage(tmp_path)
    original = Path.resolve
    def resolve(candidate, *args, **kwargs):
        raise AssertionError(f"unexpected resolve of {candidate}")

    monkeypatch.setattr(Path, "resolve", resolve)
    result = _cleanup(tmp_path)
    assert not path.exists()
    assert result["removed"] == [str(path)]


@requires_anchored_deletion
def test_one_deletion_error_does_not_stop_other_cleanup(tmp_path, monkeypatch):
    bad = _stage(tmp_path, f".bad.tmp-{HEX}")
    good = _stage(tmp_path, f".good.tmp-{'b' * 32}")
    original = editing.shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if Path(path).name.startswith(".bad.tmp-"):
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


@requires_anchored_deletion
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


def test_symlinked_cleanup_root_ancestor_is_refused(tmp_path):
    outside = tmp_path / "outside"
    edits = outside / "edits"
    edits.mkdir(parents=True)
    artifact = _stage(edits)
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    result = _cleanup(link / "edits")
    assert artifact.exists()
    assert result["removed_count"] == 0
    assert "symlink or junction/reparse" in result["errors"][0]["error"]


@requires_anchored_deletion
def test_real_cleanup_root_ancestor_chain_is_accepted(tmp_path):
    root = tmp_path / "one" / "two" / "edits"
    root.mkdir(parents=True)
    artifact = _stage(root)
    assert _cleanup(root)["removed"] == [str(artifact)]


def test_validated_root_ancestor_change_aborts_cleanup(tmp_path):
    parent = tmp_path / "parent"
    root = parent / "edits"
    root.mkdir(parents=True)
    artifact = _stage(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_artifact = _stage(outside, f".outside.tmp-{'b' * 32}")
    moved = tmp_path / "original-parent"

    def observer(phase, _value):
        if phase != "discovered":
            return
        parent.rename(moved)
        parent.symlink_to(outside, target_is_directory=True)

    try:
        result = editing.cleanup_abandoned_export_temps(
            root=root, now=NOW, observer=observer,
        )
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    assert (moved / "edits" / artifact.name).exists()
    assert outside_artifact.exists()
    assert result["removed_count"] == 0
    assert result["skipped_changed"] == [str(root / artifact.name)]


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_candidate_replaced_by_symlink_after_discovery_is_preserved(tmp_path, kind):
    candidate = _gguf(tmp_path) if kind == "file" else _stage(tmp_path)
    target = tmp_path / ("target-file" if kind == "file" else "target-dir")
    if kind == "file":
        target.write_bytes(b"safe")
    else:
        target.mkdir()
        (target / "sentinel").write_text("safe")

    def observer(phase, _value):
        if phase != "discovered":
            return
        if kind == "file":
            candidate.unlink()
        else:
            candidate.rmdir()
        candidate.symlink_to(target, target_is_directory=kind == "directory")

    try:
        result = editing.cleanup_abandoned_export_temps(
            root=tmp_path, now=NOW, observer=observer,
        )
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    assert candidate.is_symlink()
    assert target.exists()
    assert str(candidate) not in result["removed"]
    assert result["skipped_changed"] == [str(candidate)]


def test_candidate_replaced_by_ordinary_inode_after_discovery_is_preserved(tmp_path):
    candidate = _gguf(tmp_path)

    def observer(phase, _value):
        if phase == "discovered":
            candidate.unlink()
            candidate.write_bytes(b"replacement")

    result = editing.cleanup_abandoned_export_temps(
        root=tmp_path, now=NOW, observer=observer,
    )
    assert candidate.read_bytes() == b"replacement"
    assert str(candidate) not in result["removed"]
    assert result["skipped_changed"] == [str(candidate)]


def test_candidate_swap_immediately_before_claim_is_preserved(tmp_path):
    candidate = _gguf(tmp_path)

    def observer(phase, _value):
        if phase == "before_claim":
            candidate.unlink()
            candidate.write_bytes(b"last-moment replacement")

    result = editing.cleanup_abandoned_export_temps(
        root=tmp_path, now=NOW, observer=observer,
    )
    assert candidate.read_bytes() == b"last-moment replacement"
    assert str(candidate) not in result["removed"]


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_parent_replacement_before_anchored_claim_is_fail_closed(tmp_path, kind):
    parent = tmp_path / "nested"
    candidate = _gguf(parent) if kind == "file" else _stage(parent)
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    sentinel = replacement / "sentinel"
    sentinel.write_text("keep")

    def observer(phase, _value):
        if phase == "before_anchored_claim":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)

    try:
        result = editing.cleanup_abandoned_export_temps(
            root=tmp_path, now=NOW, observer=observer,
        )
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    assert (moved / candidate.name).exists()
    assert parent.is_symlink() and sentinel.read_text() == "keep"
    assert str(candidate) not in result["removed"]
    assert str(candidate) in result["skipped_changed"]


@requires_anchored_deletion
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_claim_replacement_before_delete_is_preserved(tmp_path, kind):
    candidate = _gguf(tmp_path) if kind == "file" else _stage(tmp_path)
    target = tmp_path / "safe-target"
    if kind == "file":
        target.write_bytes(b"safe")
    else:
        target.mkdir()
        (target / "sentinel").write_text("safe")
    moved_claim = tmp_path / "original-claim"
    replacement = {}

    def observer(phase, value):
        if phase != "before_delete":
            return
        _, claim = value
        claim.rename(moved_claim)
        claim.symlink_to(target, target_is_directory=kind == "directory")
        replacement["path"] = claim

    try:
        result = editing.cleanup_abandoned_export_temps(
            root=tmp_path, now=NOW, observer=observer,
        )
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")
    assert moved_claim.exists()
    assert replacement["path"].is_symlink()
    assert target.exists()
    assert str(candidate) not in result["removed"]
    assert str(candidate) in result["skipped_changed"]


@requires_anchored_deletion
def test_claim_changed_to_different_inode_is_not_deleted(tmp_path):
    candidate = _gguf(tmp_path)
    moved_claim = tmp_path / "original-claim"
    replacement = {}

    def observer(phase, value):
        if phase == "before_delete":
            _, claim = value
            claim.rename(moved_claim)
            claim.write_bytes(b"replacement")
            replacement["path"] = claim

    result = editing.cleanup_abandoned_export_temps(
        root=tmp_path, now=NOW, observer=observer,
    )
    assert moved_claim.read_bytes() == b"partial"
    assert replacement["path"].read_bytes() == b"replacement"
    assert str(candidate) not in result["removed"]


@requires_anchored_deletion
def test_claim_metadata_change_is_not_deleted(tmp_path):
    candidate = _gguf(tmp_path)
    claim_path = {}

    def observer(phase, value):
        if phase == "before_delete":
            _, claim = value
            claim.write_bytes(b"changed during claim")
            claim_path["path"] = claim

    result = editing.cleanup_abandoned_export_temps(
        root=tmp_path, now=NOW, observer=observer,
    )
    assert claim_path["path"].read_bytes() == b"changed during claim"
    assert str(candidate) not in result["removed"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor anchoring regression")
def test_parent_move_after_anchor_stays_attached_to_verified_directory(tmp_path):
    parent = tmp_path / "nested"
    candidate = _gguf(parent)
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    sentinel = replacement / "sentinel"
    sentinel.write_text("keep")

    def observer(phase, _value):
        if phase == "before_delete":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)

    try:
        result = editing.cleanup_abandoned_export_temps(
            root=tmp_path, now=NOW, observer=observer,
        )
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    assert result["removed"] == [str(candidate)]
    assert parent.is_symlink() and sentinel.read_text() == "keep"
    assert not (moved / candidate.name).exists()
    assert not any(editing.internal_temp_leaf_kind(path.name) for path in moved.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="POSIX dir_fd regression")
def test_final_cleanup_uses_descriptor_relative_operations(tmp_path, monkeypatch):
    directory = _stage(tmp_path, f".dir.tmp-{'b' * 32}")
    file = _gguf(tmp_path)

    monkeypatch.setattr(
        Path, "replace",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Path.replace used")),
    )
    monkeypatch.setattr(
        Path, "unlink",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Path.unlink used")),
    )
    original_rmtree = editing.shutil.rmtree

    def anchored_rmtree(path, *args, **kwargs):
        assert kwargs.get("dir_fd") is not None
        assert not os.path.isabs(os.fspath(path))
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(editing.shutil, "rmtree", anchored_rmtree)
    result = _cleanup(tmp_path)
    assert result["removed_count"] == 2
    assert not directory.exists() and not file.exists()


def test_completion_marker_replaced_by_symlink_is_fail_closed(tmp_path):
    candidate = _stage(tmp_path)
    marker = candidate / "edit_meta.json"
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"name": "other"}))
    try:
        marker.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks are unavailable")
    result = _cleanup(tmp_path)
    assert candidate.exists() and marker.is_symlink()
    assert result["removed_count"] == 0
    if os.name == "nt":
        assert str(candidate) in result["skipped_changed"]
        assert "handle-relative completion marker inspection is unavailable" in (
            result["errors"][0]["error"]
        )
    else:
        assert "safe regular file" in result["errors"][0]["error"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor marker inspection")
def test_parent_change_during_completion_marker_inspection_is_fail_closed(tmp_path):
    parent = tmp_path / "nested"
    candidate = _stage(parent)
    marker = candidate / "edit_meta.json"
    marker.write_text(json.dumps({"name": "model"}))
    _mtime(marker, OLD)
    _mtime(candidate, OLD)

    moved = tmp_path / "moved"
    replacement_parent = tmp_path / "replacement"
    replacement_candidate = replacement_parent / candidate.name
    replacement_candidate.mkdir(parents=True)
    replacement_marker = replacement_candidate / "edit_meta.json"
    replacement_marker.write_text(json.dumps({"name": f"nested/{candidate.name}"}))
    sentinel = replacement_candidate / "sentinel"
    sentinel.write_text("keep")

    def observer(phase, _value):
        if phase == "before_marker_open":
            parent.rename(moved)
            parent.symlink_to(replacement_parent, target_is_directory=True)

    try:
        result = editing.cleanup_abandoned_export_temps(
            root=tmp_path, now=NOW, observer=observer,
        )
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    assert (moved / candidate.name / "edit_meta.json").is_file()
    assert parent.is_symlink()
    assert sentinel.read_text() == "keep"
    assert result["removed_count"] == 0
    assert str(candidate) in result["skipped_changed"]


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


@pytest.mark.parametrize("name", [
    f".done.tmp-{HEX}", f"nested/.done.tmp-{HEX}",
    f".done.tmp-{HEX}.gguf", f".done.tmp-{HEX}.lease",
])
def test_export_abliteration_rejects_reserved_name_before_compute(
    tmp_path, monkeypatch, name,
):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path)
    called = []
    monkeypatch.setattr(
        editing, "compute_abliteration",
        lambda *args, **kwargs: called.append(True),
    )
    with pytest.raises(ValueError, match="reserved for internal temporary export"):
        editing.export_abliteration(
            [{"layers": [0]}], object(), {}, fmt="layers", name=name,
        )
    assert called == []
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


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
    if os.name == "nt":
        assert result["skipped_completed"] == []
        assert result["skipped_changed"] == [str(path)]
        assert "handle-relative completion marker inspection is unavailable" in (
            result["errors"][0]["error"]
        )
    else:
        assert result["skipped_completed"] == [str(path)]
    assert sentinel.read_text() == "{}"


@requires_anchored_deletion
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


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock regression")
@pytest.mark.parametrize("preexisting", [False, True])
def test_nonblocking_flock_contention_removes_only_new_lease(
    tmp_path, monkeypatch, preexisting,
):
    import fcntl
    artifact = tmp_path / f".model.tmp-{HEX}"
    lease = editing.ArtifactLease(artifact)
    if preexisting:
        lease.path.write_bytes(b"existing")
    monkeypatch.setattr(
        fcntl, "flock",
        lambda *args: (_ for _ in ()).throw(BlockingIOError("busy")),
    )
    assert lease.acquire(blocking=False) is False
    if preexisting:
        assert lease.path.read_bytes() == b"existing"
    else:
        assert not lease.path.exists()


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
        if os.name == "nt":
            assert artifact.exists()
            assert second["removed"] == []
            assert "handle-relative" in second["errors"][0]["error"]
        else:
            assert second["removed"] == [str(artifact)] and not artifact.exists()
    finally:
        parent.close()
        child_connection.close()
        if child.is_alive():
            child.kill()
            child.join()


@pytest.mark.skipif(os.name != "nt", reason="Windows fail-closed regression")
@pytest.mark.parametrize("kind", ["file", "directory"])
def test_windows_cleanup_without_handle_relative_deletion_fails_closed(tmp_path, kind):
    artifact = _gguf(tmp_path) if kind == "file" else _stage(tmp_path)
    sentinel = artifact.parent / "unrelated"
    if kind == "directory":
        hf = artifact / "hf"
        hf.mkdir()
        sentinel = hf / "config.json"
    sentinel.write_text("keep")
    if kind == "directory":
        _mtime(sentinel, OLD)
        _mtime(hf, OLD)
        _mtime(artifact, OLD)
    observed = []
    result = _cleanup(tmp_path, observer=lambda phase, value: observed.append(phase))
    assert artifact.exists()
    assert sentinel.read_text() == "keep"
    assert result["removed"] == []
    assert result["removed_count"] == 0
    assert result["skipped_changed"] == [str(artifact)]
    assert "handle-relative" in result["errors"][0]["error"]
    assert "before_delete" not in observed
