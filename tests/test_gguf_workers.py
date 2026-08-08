import copy
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from core import capabilities, editing, rebase
from core.ablation import Interventions
from core.model_manager import _rebase_capability_meta
from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType
from helpers import *
from helpers import _deferred_threads, _gguf_test_tools, _track_test_worker
from helpers import GGUFTestWorkerRegistry


def _set_gguf_state(app, state):
    with app._gguf_lock:
        assert app._gguf_owner is None and app._gguf_delete_claim is None
        app._gguf_state.clear()
        app._gguf_state.update(state)


def _set_idle_gguf_state(app):
    _set_gguf_state(app, {
        "state": "idle", "name": None, "step": None,
        "error": None, "result": None,
    })


def test_gguf_worker_registry_is_per_instance_and_clears_references():
    first, second = GGUFTestWorkerRegistry(), GGUFTestWorkerRegistry()
    marker = object(); first.releases.append(marker)
    assert second.real == [] and second.deferred == [] and second.releases == []
    first.clear()
    assert first.real == [] and first.deferred == [] and first.releases == []


def _install_loaded_bundle(app, monkeypatch, *, jl=None, profile="__default__", meta=None):
    supported = capabilities.decision(True, "supported")
    if profile == "__default__":
        profile = {
            "declared_quantization": None,
            "has_packed_read_parameters": False,
            "modes": {
                "standard": dict(supported),
                "readthrough": dict(supported),
                "exact": dict(supported),
                "abliteration": capabilities.decision(False, "global_projection_unvalidated"),
            },
        }
    meta = dict({"model_id": "local", "quant": None}, **(meta or {}))
    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    hf_model = object()
    tokenizer = object()
    jl = jl if jl is not None else object()
    token, snap = coordinator.acquire(OperationType.LOAD)
    bundle = LoadedModelBundle.from_parts(hf_model, tokenizer, jl, meta, profile)
    coordinator.publish_loaded(token, bundle, expected_unloaded_session=snap.model_session_id)
    coordinator.release(token)
    monkeypatch.setattr(app.manager, "hf_model", hf_model)
    monkeypatch.setattr(app.manager, "tokenizer", tokenizer)
    monkeypatch.setattr(app.manager, "jl", jl)
    monkeypatch.setattr(app.manager, "meta", meta)
    monkeypatch.setattr(app.manager, "capability_profile", profile)
    return bundle


def _mock_loaded_readthrough(app, monkeypatch, *, meta=None):
    rules = [{"id": 1, "layers": [0], "enabled": True, "mode": "scale", "factor": 0.0}]
    _install_loaded_bundle(app, monkeypatch, meta=meta)
    app.manager.coordinator.bootstrap_interventions_for_test({
        "revision": 1,
        "model_session_id": app.manager.coordinator.status_snapshot().model_session_id,
        "lens_binding_id": None,
        "scale": 1.0,
        "mode": "readthrough",
        "rules": rules,
        "active_rules": rules,
        "summary": rules,
        "active_summary": rules,
    })
    monkeypatch.setattr(app.interventions, "active_rules_full", lambda: rules)
    monkeypatch.setattr(app.interventions, "snapshot", lambda: {
        "revision": 1,
        "scale": 1.0,
        "mode": "readthrough",
        "rules": rules,
        "active_rules": rules,
        "summary": rules,
        "active_summary": rules,
    })
    monkeypatch.setattr(app.interventions, "_mode", "readthrough")
    monkeypatch.setattr(app.interventions, "_scale", 1.0)


def test_api_gguf_preparation_uses_nested_hf_name(tmp_path, monkeypatch, gguf_test_workers):
    import api.app as app
    captured = {}
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    revisions = []
    monkeypatch.setattr(
        app, "resolve_local_dir",
        lambda _model, *, revision=None: revisions.append(revision) or tmp_path / "source",
    )
    _mock_loaded_readthrough(app, monkeypatch,
                             meta={"revision": "recorded-revision"})
    def fake_export(*_args, **kwargs):
        captured["name"] = kwargs["name"]
        target = app.editing.EDITS_DIR / kwargs["name"]
        stage = target.with_name(".hf.test-stage")
        stage.mkdir(parents=True); (stage / "config.json").write_text("{}")
        kwargs["publication_guard"].publish(lambda: stage.replace(target))
        return {"out_dir": str(target)}
    monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    pending = _deferred_threads(monkeypatch, app, gguf_test_workers)
    _set_idle_gguf_state(app)
    result = asyncio.run(app.api_edit_export_gguf(SimpleNamespace(name="job", gguf_type="bf16")))
    assert captured["name"] == "job/hf" and result["checkpoint"] == "baked"
    assert revisions == ["recorded-revision"]
    assert (app.editing.EDITS_DIR / "job" / "hf" / "config.json").exists()
    pending[0].run()
    assert app._gguf_owner is None


def test_api_gguf_maps_generic_export_error_to_500(tmp_path, monkeypatch):
    import api.app as app
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app, "resolve_local_dir",
                        lambda _model, *, revision=None: tmp_path / "source")
    _mock_loaded_readthrough(app, monkeypatch)
    monkeypatch.setattr(app.editing, "export_rebase", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk failed")))
    _set_idle_gguf_state(app)
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.api_edit_export_gguf(SimpleNamespace(name="job", gguf_type="bf16")))
    assert exc.value.status_code == 500 and exc.value.detail == "disk failed"


def test_production_fresh_gguf_builder_caller_cancel_rejects_late_cache_publish(
        tmp_path, monkeypatch):
    import api.app as app

    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app, "resolve_local_dir",
                        lambda _model, *, revision=None: str(tmp_path / "source"))
    _mock_loaded_readthrough(app, monkeypatch)
    ready = __import__("threading").Event(); release = __import__("threading").Event()
    stage = edits / "job" / ".hf.test-stage"; final = edits / "job" / "hf"
    received = []
    def fake_export(*_args, publication_guard=None, **_kwargs):
        received.append(publication_guard)
        stage.mkdir(parents=True); (stage / "config.json").write_text("{}")
        ready.set()
        try:
            assert release.wait(2)
            publication_guard.publish(lambda: stage.replace(final))
        finally:
            import shutil
            shutil.rmtree(stage, ignore_errors=True)
    monkeypatch.setattr(app.editing, "export_rebase", fake_export)

    async def scenario():
        task = asyncio.create_task(app.api_edit_export_gguf(
            SimpleNamespace(name="job", gguf_type="bf16")
        ))
        assert await asyncio.to_thread(ready.wait, 2)
        try:
            task.cancel()
            while received[0].state == "pending":
                await asyncio.sleep(0)
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert received and received[0].state == "cancelled-before-publication"
    assert not final.exists() and not stage.exists()
    assert app.manager.coordinator.status_snapshot().operation is None
    assert app._gguf_owner is None


def test_fresh_gguf_bake_without_capability_profile_reaches_guarded_exporter(
        tmp_path, monkeypatch, gguf_test_workers):
    import api.app as app
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app, "resolve_local_dir",
                        lambda _model, *, revision=None: tmp_path / "source")
    _install_loaded_bundle(app, monkeypatch, profile=None)
    rules = [{"id": 1, "layers": [0], "enabled": True, "mode": "scale", "factor": 0.0}]
    app.manager.coordinator.bootstrap_interventions_for_test({
        "revision": 1,
        "model_session_id": app.manager.coordinator.status_snapshot().model_session_id,
        "lens_binding_id": None,
        "scale": 1.0,
        "mode": "readthrough",
        "rules": rules,
        "active_rules": rules,
        "summary": rules,
        "active_summary": rules,
    })
    monkeypatch.setattr(app.interventions, "active_rules_full", lambda: rules)
    monkeypatch.setattr(app.interventions, "snapshot", lambda: {
        "revision": 1, "scale": 1.0, "mode": "readthrough",
        "rules": rules, "active_rules": rules, "summary": rules,
        "active_summary": rules,
    })
    monkeypatch.setattr(app.interventions, "_mode", "readthrough")
    calls = []
    def fake_export(*_args, publication_guard=None, **kwargs):
        calls.append(kwargs["name"])
        target = app.editing.EDITS_DIR / kwargs["name"]
        stage = target.with_name(".hf.advisory-stage")
        stage.mkdir(parents=True); (stage / "config.json").write_text("{}")
        publication_guard.publish(lambda: stage.replace(target))
        return {"out_dir": str(target)}
    monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    pending = _deferred_threads(monkeypatch, app, gguf_test_workers)
    _set_idle_gguf_state(app)
    result = asyncio.run(app.api_edit_export_gguf(
        SimpleNamespace(name="job", gguf_type="bf16")
    ))
    assert calls == ["job/hf"]
    assert result["checkpoint"] == "baked"
    assert app.manager.coordinator.status_snapshot().operation is None
    pending[0].run()
    assert app._gguf_owner is None


def test_fresh_gguf_predispatch_failure_releases_bake_owner(
        tmp_path, monkeypatch, gguf_test_workers):
    import api.app as app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    _install_loaded_bundle(app, monkeypatch)
    monkeypatch.setattr(app.interventions, "active_rules_full", lambda: [])
    monkeypatch.setattr(app.interventions, "_mode", "readthrough")
    monkeypatch.setattr(app.interventions, "_scale", 1.0)
    monkeypatch.setattr(app.editing, "export_rebase",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("pre-dispatch validation should reject first")))
    _set_idle_gguf_state(app)

    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.api_edit_export_gguf(SimpleNamespace(name="job", gguf_type="bf16")))

    assert exc.value.status_code == 422
    assert app.manager.coordinator.snapshot().operation is None

    cached = app.editing.EDITS_DIR / "job" / "hf"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    pending = _deferred_threads(monkeypatch, app, gguf_test_workers)
    response = TestClient(app.app).post(
        "/api/edit/export-gguf", json={"name": "job", "gguf_type": "bf16"})

    assert response.status_code == 200
    assert response.json()["checkpoint"] == "reused"
    assert pending
    pending[0].run()
    assert app._gguf_owner is None
    assert app.manager.coordinator.snapshot().operation is None


def test_fresh_gguf_abliteration_rejected_after_terminal_reservation(tmp_path, monkeypatch):
    import api.app as app
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths",
                        lambda: (tmp_path / "convert.py", None, None))
    _mock_loaded_readthrough(app, monkeypatch)
    monkeypatch.setattr(app.manager, "jl", dense_lens())
    monkeypatch.setattr(app.interventions, "_mode", "abliteration")
    def forbidden(*_args, **_kwargs):
        raise AssertionError("global export or worker dispatch must not run")
    monkeypatch.setattr(app.editing, "export_abliteration", forbidden)
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=forbidden))
    original = {"state": "idle", "name": None, "step": None,
                "error": None, "result": None}
    _set_gguf_state(app, original)
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.api_edit_export_gguf(
            SimpleNamespace(name="global", gguf_type="bf16")
        ))
    assert exc.value.status_code == 422
    assert app._gguf_state["state"] == "error"
    assert app._gguf_state["name"] == "global"
    assert app._gguf_state["result"] is None
    assert app._gguf_owner is None


@pytest.mark.parametrize("name", ["COM¹", "com²", "COM³.txt", "LPT¹", "lpt².json", "folder/LPT³"])
def test_gguf_reserved_cache_hit_rejected_before_side_effects(name, tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    cached = app.editing.EDITS_DIR / name / "hf"; cached.mkdir(parents=True)
    sentinel = cached / "config.json"; sentinel.write_text('{"old": true}')
    def forbidden(*_args, **_kwargs): raise AssertionError("validation ordering violated")
    monkeypatch.setattr(app, "_llamacpp_paths", forbidden)
    monkeypatch.setattr(app.editing, "export_rebase", forbidden)
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=forbidden))
    original_state = {"state": "idle", "name": None, "step": None, "error": None, "result": None}
    _set_gguf_state(app, original_state)
    response = TestClient(app.app).post("/api/edit/export-gguf", json={"name": name, "gguf_type": "bf16"})
    assert response.status_code == 422 and "reserved" in response.json()["detail"]
    assert sentinel.read_text() == '{"old": true}' and app._gguf_state == original_state


@pytest.mark.parametrize("name", ["COM¹", "com²", "COM³.txt", "LPT¹", "lpt².json", "folder/LPT³"])
def test_gguf_reserved_cache_delete_preserves_sentinel(name, tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    cached = app.editing.EDITS_DIR / name / "hf"; cached.mkdir(parents=True)
    sentinel = cached / "keep.bin"; sentinel.write_bytes(b"keep")
    original_state = {"state": "idle", "name": None, "step": None, "error": None, "result": None}
    _set_gguf_state(app, original_state)
    response = TestClient(app.app).post("/api/edit/gguf-cache/delete", json={"name": name})
    assert response.status_code == 422 and "reserved" in response.json()["detail"]
    assert sentinel.read_bytes() == b"keep" and app._gguf_state == original_state


def test_gguf_valid_cached_reuse_and_delete(tmp_path, monkeypatch, gguf_test_workers):
    import api.app as app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    cached = app.editing.EDITS_DIR / "job" / "hf"; cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app.editing, "export_rebase",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cache was not reused")))
    pending = _deferred_threads(monkeypatch, app, gguf_test_workers)
    _set_idle_gguf_state(app)
    client = TestClient(app.app)
    response = client.post("/api/edit/export-gguf", json={"name": "job", "gguf_type": "bf16"})
    assert response.status_code == 200 and response.json()["checkpoint"] == "reused" and pending
    pending[0].run()
    assert app._gguf_owner is None
    response = client.post("/api/edit/gguf-cache/delete", json={"name": "job"})
    assert response.status_code == 200 and not cached.exists()


@pytest.mark.parametrize(
    ("name", "cached", "gguf_type", "expected_checkpoint", "expected_files"),
    [
        ("job/hf", True, "bf16", "reused", ["hf-bf16.gguf"]),
        ("job/hf", False, "bf16", "baked", ["hf-bf16.gguf"]),
        ("job/hf", True, "q4_k_m", "reused", ["hf-bf16.gguf", "hf-q4_k_m.gguf"]),
        ("job", True, "bf16", "reused", ["job-bf16.gguf"]),
    ],
)
def test_gguf_nested_worker_outputs_use_leaf_stem(
        name, cached, gguf_type, expected_checkpoint, expected_files, tmp_path,
        monkeypatch, gguf_test_workers):
    import api.app as app
    from fastapi.testclient import TestClient
    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    name_parts = tuple(name.split("/")); job_dir = edits.joinpath(*name_parts); hf_dir = job_dir / "hf"
    if cached:
        hf_dir.mkdir(parents=True); (hf_dir / "config.json").write_text("{}")
    convert, quantize = _gguf_test_tools(tmp_path)
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (convert, quantize, None))
    if not cached:
        _mock_loaded_readthrough(app, monkeypatch)
        monkeypatch.setattr(app, "resolve_local_dir",
                            lambda _model, *, revision=None: tmp_path / "source")
        def fake_export(*_args, **kwargs):
            assert kwargs["name"] == name + "/hf"
            stage = hf_dir.with_name(".hf.test-stage")
            stage.mkdir(parents=True); (stage / "config.json").write_text("{}")
            kwargs["publication_guard"].publish(lambda: stage.replace(hf_dir))
        monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    pending = _deferred_threads(monkeypatch, app, gguf_test_workers)
    _set_idle_gguf_state(app)
    response = TestClient(app.app).post(
        "/api/edit/export-gguf", json={"name": name, "gguf_type": gguf_type})
    assert response.status_code == 200
    assert response.json()["checkpoint"] == expected_checkpoint
    assert response.json()["state"]["state"] == "running"
    assert response.json()["state"]["name"] == name
    assert len(pending) == 1
    pending[0].run()
    assert app._gguf_state["state"] == "done" and app._gguf_state["name"] == name
    for filename in expected_files:
        assert (job_dir / filename).is_file()
    assert Path(app._gguf_state["result"]["gguf"]) == job_dir / expected_files[-1]
    assert not (job_dir / "job").exists()


@pytest.mark.parametrize(
    ("name", "cached", "gguf_type"),
    [
        ("job/hf", True, "bf16"),
        ("job/hf", False, "bf16"),
        ("job/hf", True, "q4_k_m"),
        ("job", True, "bf16"),
    ],
)
def test_gguf_atomic_partial_failure_then_retry(
        name, cached, gguf_type, tmp_path, monkeypatch, gguf_test_workers):
    import api.app as app
    from fastapi.testclient import TestClient
    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    parts = tuple(name.split("/")); job_dir = edits.joinpath(*parts); hf_dir = job_dir / "hf"
    if cached:
        hf_dir.mkdir(parents=True)
    sentinel = hf_dir / "config.json"
    if cached:
        sentinel.write_text('{"cache": true}')
    invocation = tmp_path / "converter-outfile.txt"
    convert = tmp_path / "convert.py"
    convert.write_text(
        "import pathlib, sys\n"
        f"marker = pathlib.Path({str(invocation)!r})\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        "marker.write_text(str(out))\n"
        "out.write_bytes(b'CORRUPT-PARTIAL')\n"
        "sys.stderr.write('deliberate partial conversion failure')\n"
        "sys.exit(3)\n"
    )
    quant_marker = tmp_path / "quantizer-invoked.json"
    quantize = make_python_cli_launcher(
        tmp_path,
        "llama-quantize",
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(quant_marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-quant')\n"
    )
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (convert, quantize, None))
    baked = []
    if not cached:
        _mock_loaded_readthrough(app, monkeypatch)
        monkeypatch.setattr(app, "resolve_local_dir",
                            lambda _model, *, revision=None: tmp_path / "source")
        def fake_export(*_args, **kwargs):
            baked.append(True)
            stage = hf_dir.with_name(".hf.test-stage")
            stage.mkdir(parents=True)
            (stage / "config.json").write_text('{"baked": true}')
            kwargs["publication_guard"].publish(lambda: stage.replace(hf_dir))
        monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    pending = _deferred_threads(monkeypatch, app, gguf_test_workers)
    _set_idle_gguf_state(app)
    client = TestClient(app.app)
    response = client.post("/api/edit/export-gguf", json={"name": name, "gguf_type": gguf_type})
    assert response.status_code == 200 and response.json()["state"]["state"] == "running"
    assert response.json()["checkpoint"] == ("reused" if cached else "baked")
    pending.pop(0).run()
    stem = parts[-1]; base = job_dir / f"{stem}-bf16.gguf"
    quantized = job_dir / f"{stem}-{gguf_type}.gguf"
    assert app._gguf_state["state"] == "error" and app._gguf_state["name"] == name
    assert "deliberate partial conversion failure" in app._gguf_state["error"]
    assert not base.exists() and not list(job_dir.glob(".*.tmp-*.gguf"))
    assert sentinel.is_file() and invocation.read_text() != str(base)
    assert not quant_marker.exists()
    if gguf_type not in app.GGUF_BASE_TYPES:
        assert not quantized.exists()
    if "/" in name:
        assert not (job_dir / "job").exists()
    invocation.unlink()
    success_marker = tmp_path / "successful-converter"
    convert.write_text(
        "import pathlib, sys\n"
        f"pathlib.Path({str(success_marker)!r}).write_text('invoked')\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        "out.write_bytes(b'VALID-GGUF')\n"
    )
    response = client.post("/api/edit/export-gguf", json={"name": name, "gguf_type": gguf_type})
    assert response.status_code == 200 and response.json()["checkpoint"] == "reused"
    pending.pop(0).run()
    assert app._gguf_state["state"] == "done" and success_marker.read_text() == "invoked"
    assert base.read_bytes() == b"VALID-GGUF" and not list(job_dir.glob(".*.tmp-*.gguf"))
    expected_result = quantized if gguf_type not in app.GGUF_BASE_TYPES else base
    assert Path(app._gguf_state["result"]["gguf"]) == expected_result
    assert expected_result.is_file() and sentinel.is_file()
    assert quant_marker.exists() == (gguf_type not in app.GGUF_BASE_TYPES)
    if gguf_type not in app.GGUF_BASE_TYPES:
        quant_args = json.loads(quant_marker.read_text())
        assert quant_args[0] == str(base) and quant_args[2] == gguf_type
        assert Path(quant_args[1]).parent == quantized.parent
        assert Path(quant_args[1]).match(f".{quantized.stem}.tmp-*.gguf")
        assert quant_args[1] != str(quantized)
        assert quantized.read_bytes() == b"VALID-GGUF-quant"
    assert len(baked) == (0 if cached else 1)


@pytest.mark.parametrize("existing", [False, True])
def test_quantized_gguf_is_published_atomically_and_retries(
        existing, tmp_path, monkeypatch):
    import api.app as app
    job_dir = tmp_path / "edits" / "job" / "hf"
    hf_dir = job_dir / "hf"
    hf_dir.mkdir(parents=True)
    cache = hf_dir / "config.json"
    cache.write_text('{"cached": true}')
    base = job_dir / "hf-bf16.gguf"
    base.write_bytes(b"BASE-GGUF")
    final = job_dir / "hf-q4_k_m.gguf"
    if existing:
        final.write_bytes(b"PREVIOUS-VALID")

    failure_marker = tmp_path / "failed-quantizer.json"
    quantize = make_python_cli_launcher(
        tmp_path,
        "llama-quantize",
        "import json, pathlib, sys\n"
        f"final = pathlib.Path({str(final)!r})\n"
        f"marker = pathlib.Path({str(failure_marker)!r})\n"
        "marker.write_text(json.dumps({'args': sys.argv[1:], 'final_exists': final.exists()}))\n"
        "pathlib.Path(sys.argv[2]).write_bytes(b'PARTIAL-QUANTIZED')\n"
        "sys.stderr.write('deliberate quantizer failure')\n"
        "sys.exit(3)\n",
    )
    token = app._reserve_gguf_job("job/hf")
    app._gguf_worker(token, "job/hf", job_dir, "hf", "q4_k_m", hf_dir,
                     tmp_path / "unused-converter.py", quantize, None)

    failure = json.loads(failure_marker.read_text())
    failed_output = Path(failure["args"][1])
    assert failure["args"][0] == str(base) and failure["args"][2] == "q4_k_m"
    assert failed_output.parent == final.parent
    assert failed_output.match(f".{final.stem}.tmp-*.gguf") and failed_output != final
    assert failure["final_exists"] is existing
    assert app._gguf_state["state"] == "error"
    assert "deliberate quantizer failure" in app._gguf_state["error"]
    assert not failed_output.exists() and not list(job_dir.glob(".*.tmp-*.gguf"))
    if existing:
        assert final.read_bytes() == b"PREVIOUS-VALID"
    else:
        assert not final.exists()
    assert base.read_bytes() == b"BASE-GGUF" and cache.read_text() == '{"cached": true}'

    success_marker = tmp_path / "successful-quantizer.json"
    quantize = make_python_cli_launcher(
        tmp_path,
        "llama-quantize",
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(success_marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "pathlib.Path(sys.argv[2]).write_bytes(b'VALID-QUANTIZED')\n",
    )
    token = app._reserve_gguf_job("job/hf")
    app._gguf_worker(token, "job/hf", job_dir, "hf", "q4_k_m", hf_dir,
                     tmp_path / "unused-converter.py", quantize, None)

    success_args = json.loads(success_marker.read_text())
    assert success_args[0] == str(base) and success_args[2] == "q4_k_m"
    assert success_args[1] != str(final)
    assert Path(success_args[1]).match(f".{final.stem}.tmp-*.gguf")
    assert app._gguf_state["state"] == "done"
    assert Path(app._gguf_state["result"]["gguf"]) == final
    assert final.read_bytes() == b"VALID-QUANTIZED"
    assert not list(job_dir.glob(".*.tmp-*.gguf"))
    if os.name != "nt":
        leases = list(job_dir.glob(".*.lease"))
        assert leases and all(path.is_file() for path in leases)
    assert base.read_bytes() == b"BASE-GGUF" and cache.read_text() == '{"cached": true}'


def test_gguf_atomic_reuses_existing_published_base(tmp_path, monkeypatch):
    import api.app as app
    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    job_dir = edits / "job"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}")
    base = job_dir / "job-bf16.gguf"; base.write_bytes(b"PUBLISHED-GGUF")
    marker = tmp_path / "converter-invoked"
    convert = tmp_path / "convert.py"
    convert.write_text(f"import pathlib\npathlib.Path({str(marker)!r}).write_text('bad')\n")
    token = app._reserve_gguf_job("job")
    app._gguf_worker(token, "job", job_dir, "job", "bf16", hf_dir, convert, None, None)
    assert app._gguf_state["state"] == "done" and not marker.exists()
    assert base.read_bytes() == b"PUBLISHED-GGUF"


@pytest.mark.parametrize(
    ("body", "message"),
    [("pass\n", "without producing"), ("import pathlib, sys\npathlib.Path(sys.argv[sys.argv.index('--outfile') + 1]).write_bytes(b'')\n", "empty output")],
)
def test_gguf_atomic_rejects_missing_or_empty_converter_output(body, message, tmp_path):
    import api.app as app
    job_dir = tmp_path / "job"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    convert = tmp_path / "convert.py"; convert.write_text(body)
    token = app._reserve_gguf_job("job")
    app._gguf_worker(token, "job", job_dir, "job", "bf16", hf_dir, convert, None, None)
    assert app._gguf_state["state"] == "error" and message in app._gguf_state["error"]
    assert not (job_dir / "job-bf16.gguf").exists()
    assert not list(job_dir.glob(".*.tmp-*.gguf"))
    if os.name != "nt":
        leases = list(job_dir.glob(".*.lease"))
        assert leases and all(path.is_file() for path in leases)


def test_stale_gguf_updates_and_worker_entry_cannot_touch_successor(tmp_path):
    import api.app as app

    old = app._reserve_gguf_job("old")
    assert app._finish_gguf_job(old, error="finished")
    successor = app._reserve_gguf_job("successor")
    before = app._gguf_state_snapshot()
    job_dir = tmp_path / "must-not-exist"
    app._gguf_worker(
        old, "old", job_dir, "old", "bf16", tmp_path / "hf",
        tmp_path / "convert.py", None, None,
    )
    assert not job_dir.exists()
    assert not app._gguf_job_progress(old, "stale")
    assert not app._finish_gguf_job(old, result={"bad": True})
    assert not app._finish_gguf_job(old, error="stale")
    assert app._gguf_state_snapshot() == before
    assert app._gguf_job_is_current(successor)
    assert app._finish_gguf_job(successor, error="test complete")


@pytest.mark.parametrize("same_name", [True, False])
def test_simultaneous_gguf_reservation_has_exactly_one_owner(
        same_name, gguf_test_workers):
    import threading
    import api.app as app

    barrier = threading.Barrier(3)
    results = []
    def reserve(name):
        barrier.wait(timeout=2)
        try:
            results.append(("won", app._reserve_gguf_job(name)))
        except Exception as exc:
            results.append(("lost", exc))
    names = ("job", "job" if same_name else "other")
    threads = [_track_test_worker(
        gguf_test_workers, threading.Thread(target=reserve, args=(name,)),
        barrier.abort, label=f"reservation-{name}")
               for name in names]
    try:
        for thread in threads: thread.start()
        barrier.wait(timeout=2)
    finally:
        for thread in threads: thread.join(2)
        barrier.abort()
    assert all(not thread.is_alive() for thread in threads)
    winners = [value for status, value in results if status == "won"]
    losers = [value for status, value in results if status == "lost"]
    assert len(winners) == len(losers) == 1
    assert app._gguf_job_is_current(winners[0])
    assert app._finish_gguf_job(winners[0], error="test complete")


@pytest.mark.parametrize("phase", ["construct", "start"])
def test_gguf_thread_setup_failure_releases_exact_owner(phase, tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient

    edits = tmp_path / "edits"
    cached = edits / "job" / "hf"
    cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))

    class BrokenThread:
        def __init__(self, **_kwargs):
            if phase == "construct":
                raise RuntimeError("constructor failed")
        def start(self):
            raise RuntimeError("start failed")
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=BrokenThread))
    response = TestClient(app.app).post(
        "/api/edit/export-gguf", json={"name": "job", "gguf_type": "bf16"}
    )
    assert response.status_code == 500
    assert app._gguf_owner is None
    assert app._gguf_state["state"] == "error"
    assert "failed" in app._gguf_state["error"]


@pytest.mark.parametrize("kind", ["empty", "directory", "symlink"])
def test_invalid_reused_gguf_base_is_preserved(kind, tmp_path, monkeypatch):
    import api.app as app

    job_dir = tmp_path / "job"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    base = job_dir / "job-bf16.gguf"
    if kind == "empty": base.write_bytes(b"")
    elif kind == "directory": base.mkdir()
    else:
        target = tmp_path / "target.gguf"; target.write_bytes(b"valid-target")
        try: base.symlink_to(target)
        except OSError: pytest.skip("symlinks unavailable")
    token = app._reserve_gguf_job("job")
    app._gguf_worker(token, "job", job_dir, "job", "bf16", hf_dir,
                     tmp_path / "converter.py", None, None)
    assert app._gguf_owner is None and app._gguf_state["state"] == "error"
    assert base.exists() or base.is_symlink()


def test_cache_delete_claim_releases_on_404_and_nested_rejection(tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    client = TestClient(app.app)
    response = client.post("/api/edit/gguf-cache/delete", json={"name": "missing"})
    assert response.status_code == 404 and app._gguf_delete_claim is None
    cache = app.editing.EDITS_DIR / "job" / "hf"
    nested = cache / "nested"; nested.mkdir(parents=True)
    (cache / "config.json").write_text("{}")
    (nested / "config.json").write_text("{}")
    response = client.post("/api/edit/gguf-cache/delete", json={"name": "job"})
    assert response.status_code == 409 and app._gguf_delete_claim is None
    assert cache.exists()


def test_stale_converter_completion_cannot_publish_or_mutate_successor(
        tmp_path, monkeypatch, gguf_test_workers):
    import subprocess
    import threading
    import api.app as app

    entered = threading.Event(); release = threading.Event()
    original_run = subprocess.run
    def blocked_run(args, **_kwargs):
        outfile = Path(args[args.index("--outfile") + 1])
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"stale-temp")
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(subprocess, "run", blocked_run)
    job_dir = tmp_path / "old"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    old = app._reserve_gguf_job("old")
    worker = _track_test_worker(gguf_test_workers, threading.Thread(
        target=app._gguf_worker,
        args=(old, "old", job_dir, "old", "bf16", hf_dir,
              tmp_path / "convert.py", None, None),
    ), release.set, label="stale-converter")
    successor = None
    try:
        worker.start()
        assert entered.wait(2)
        assert app._finish_gguf_job(old, error="superseded")
        successor = app._reserve_gguf_job("successor")
        before = app._gguf_state_snapshot()
    finally:
        release.set(); worker.join(2)
    assert not worker.is_alive()
    assert not (job_dir / "old-bf16.gguf").exists()
    assert not list(job_dir.glob(".*.tmp-*.gguf"))
    assert app._gguf_state_snapshot() == before
    assert app._gguf_job_is_current(successor)
    assert app._finish_gguf_job(successor, error="test complete")
    monkeypatch.setattr(subprocess, "run", original_run)


def test_cache_delete_serializes_with_jobs_and_releases_after_success(tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient

    edits = tmp_path / "edits"; cache = edits / "job" / "hf"
    cache.mkdir(parents=True); (cache / "config.json").write_text("{}")
    monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    client = TestClient(app.app)
    token = app._reserve_gguf_job("active")
    response = client.post("/api/edit/gguf-cache/delete", json={"name": "job"})
    assert response.status_code == 409 and cache.exists()
    assert app._finish_gguf_job(token, error="test complete")
    response = client.post("/api/edit/gguf-cache/delete", json={"name": "job"})
    assert response.status_code == 200 and not cache.exists()
    assert app._gguf_delete_claim is None


@pytest.mark.parametrize("evidence", ["config", "gguf"])
def test_cache_delete_blocks_nested_export_evidence(evidence, tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient

    cache = tmp_path / "edits" / "job" / "hf"; nested = cache / "nested"
    nested.mkdir(parents=True); (cache / "config.json").write_text("{}")
    if evidence == "config": (nested / "config.json").write_text("{}")
    else: (nested / "published.gguf").write_bytes(b"valid")
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    response = TestClient(app.app).post(
        "/api/edit/gguf-cache/delete", json={"name": "job"}
    )
    assert response.status_code == 409 and cache.exists()
    assert app._gguf_delete_claim is None


def test_held_delete_claim_blocks_job_and_second_delete(
        tmp_path, monkeypatch, gguf_test_workers):
    import shutil
    import threading
    import api.app as app

    cache = tmp_path / "edits" / "job" / "hf"
    cache.mkdir(parents=True); (cache / "config.json").write_text("{}")
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    entered = threading.Event(); release = threading.Event(); results = []
    original = shutil.rmtree
    def blocked(path):
        entered.set()
        assert release.wait(2)
        return original(path)
    monkeypatch.setattr(shutil, "rmtree", blocked)
    worker = _track_test_worker(
        gguf_test_workers,
        threading.Thread(
            target=lambda: results.append(app.api_gguf_cache_delete(
                app.GGUFCacheRequest(name="job")
            )),
        ),
        release.set, label="held-delete",
    )
    try:
        worker.start(); assert entered.wait(2)
        claim = app._gguf_delete_claim
        with pytest.raises(Exception, match="already in progress"):
            app._reserve_gguf_job("blocked")
        with pytest.raises(app.HTTPException) as raised:
            app.api_gguf_cache_delete(app.GGUFCacheRequest(name="job"))
        assert raised.value.status_code == 409
        assert app._gguf_delete_claim == claim
    finally:
        release.set(); worker.join(2)
    assert not worker.is_alive()
    assert app._gguf_delete_claim is None and results


@pytest.mark.parametrize("body", ["pass\n", "import pathlib, sys\npathlib.Path(sys.argv[2]).write_bytes(b'')\n"])
def test_quantizer_success_rejects_missing_or_empty_output(body, tmp_path):
    import api.app as app

    job_dir = tmp_path / "job"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    base = job_dir / "job-bf16.gguf"; base.write_bytes(b"VALID-BASE")
    quantize = make_python_cli_launcher(tmp_path, "llama-quantize", body)
    token = app._reserve_gguf_job("job")
    app._gguf_worker(token, "job", job_dir, "job", "q4_k_m", hf_dir,
                     tmp_path / "unused.py", quantize, None)
    assert app._gguf_owner is None and app._gguf_state["state"] == "error"
    assert base.read_bytes() == b"VALID-BASE"
    assert not (job_dir / "job-q4_k_m.gguf").exists()
    assert not list(job_dir.glob(".*.tmp-*.gguf"))


@pytest.mark.parametrize("kind", ["empty", "directory", "symlink"])
def test_invalid_reused_base_quantized_skips_converter_and_quantizer(
        kind, tmp_path):
    import api.app as app

    job_dir = tmp_path / "job"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    base = job_dir / "job-bf16.gguf"
    if kind == "empty": base.write_bytes(b"")
    elif kind == "directory": base.mkdir()
    else:
        target = tmp_path / "target.gguf"; target.write_bytes(b"target")
        try: base.symlink_to(target)
        except OSError: pytest.skip("symlinks unavailable")
    convert_marker = tmp_path / "converter-called"
    convert = tmp_path / "convert.py"
    convert.write_text(f"import pathlib\npathlib.Path({str(convert_marker)!r}).write_text('called')\n")
    quant_marker = tmp_path / "quantizer-called"
    quantize = make_python_cli_launcher(
        tmp_path, "llama-quantize",
        f"import pathlib\npathlib.Path({str(quant_marker)!r}).write_text('called')\n",
    )
    token = app._reserve_gguf_job("job")
    app._gguf_worker(token, "job", job_dir, "job", "q4_k_m", hf_dir,
                     convert, quantize, None)
    assert app._gguf_owner is None and app._gguf_state["state"] == "error"
    assert base.exists() or base.is_symlink()
    assert not convert_marker.exists() and not quant_marker.exists()
    assert not (job_dir / "job-q4_k_m.gguf").exists()
