import os
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
