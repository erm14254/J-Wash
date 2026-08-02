import copy
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn

from core import editing, rebase
from core.ablation import Interventions
from core.model_manager import _rebase_capability_meta
from helpers import *
from helpers import _deferred_threads, _gguf_test_tools

def test_api_gguf_preparation_uses_nested_hf_name(tmp_path, monkeypatch):
    import api.app as app
    captured = {}
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app, "resolve_local_dir", lambda _model: tmp_path / "source")
    monkeypatch.setattr(app.manager, "hf_model", object())
    monkeypatch.setattr(app.manager, "jl", object())
    monkeypatch.setattr(app.manager, "meta", {"model_id": "local"})
    monkeypatch.setattr(app.interventions, "active_rules_full", lambda: [{"layers": [0]}])
    monkeypatch.setattr(app.interventions, "_mode", "readthrough")
    monkeypatch.setattr(app.interventions, "_scale", 1.0)
    def fake_export(*_args, **kwargs):
        captured["name"] = kwargs["name"]
        target = app.editing.EDITS_DIR / kwargs["name"]
        target.mkdir(parents=True); (target / "config.json").write_text("{}")
        return {"out_dir": str(target)}
    monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    monkeypatch.setattr(app, "_gguf_worker", lambda *_args: None)
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
    result = asyncio.run(app.api_edit_export_gguf(SimpleNamespace(name="job", gguf_type="bf16")))
    assert captured["name"] == "job/hf" and result["checkpoint"] == "baked"
    assert (app.editing.EDITS_DIR / "job" / "hf" / "config.json").exists()


def test_api_gguf_maps_generic_export_error_to_500(tmp_path, monkeypatch):
    import api.app as app
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app, "resolve_local_dir", lambda _model: tmp_path / "source")
    monkeypatch.setattr(app.manager, "hf_model", object()); monkeypatch.setattr(app.manager, "jl", object())
    monkeypatch.setattr(app.manager, "meta", {"model_id": "local"})
    monkeypatch.setattr(app.interventions, "active_rules_full", lambda: [{"layers": [0]}])
    monkeypatch.setattr(app.interventions, "_mode", "readthrough")
    monkeypatch.setattr(app.editing, "export_rebase", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk failed")))
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.api_edit_export_gguf(SimpleNamespace(name="job", gguf_type="bf16")))
    assert exc.value.status_code == 500 and exc.value.detail == "disk failed"


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
    app._gguf_state.update(original_state)
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
    app._gguf_state.update(original_state)
    response = TestClient(app.app).post("/api/edit/gguf-cache/delete", json={"name": name})
    assert response.status_code == 422 and "reserved" in response.json()["detail"]
    assert sentinel.read_bytes() == b"keep" and app._gguf_state == original_state


def test_gguf_valid_cached_reuse_and_delete(tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(app.editing, "EDITS_DIR", tmp_path / "edits")
    cached = app.editing.EDITS_DIR / "job" / "hf"; cached.mkdir(parents=True)
    (cached / "config.json").write_text("{}")
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (tmp_path / "convert.py", None, None))
    monkeypatch.setattr(app.editing, "export_rebase",
                        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("cache was not reused")))
    started = []
    class FakeThread:
        def __init__(self, *args, **kwargs): started.append((args, kwargs))
        def start(self): pass
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=FakeThread))
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
    client = TestClient(app.app)
    response = client.post("/api/edit/export-gguf", json={"name": "job", "gguf_type": "bf16"})
    assert response.status_code == 200 and response.json()["checkpoint"] == "reused" and started
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
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
        name, cached, gguf_type, expected_checkpoint, expected_files, tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient
    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    name_parts = tuple(name.split("/")); job_dir = edits.joinpath(*name_parts); hf_dir = job_dir / "hf"
    if cached:
        hf_dir.mkdir(parents=True); (hf_dir / "config.json").write_text("{}")
    convert, quantize = _gguf_test_tools(tmp_path)
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (convert, quantize, None))
    if not cached:
        monkeypatch.setattr(app.manager, "hf_model", object())
        monkeypatch.setattr(app.manager, "jl", object())
        monkeypatch.setattr(app.manager, "meta", {"model_id": "local"})
        monkeypatch.setattr(app.interventions, "active_rules_full", lambda: [{"layers": [0]}])
        monkeypatch.setattr(app.interventions, "_mode", "readthrough")
        monkeypatch.setattr(app.interventions, "_scale", 1.0)
        monkeypatch.setattr(app, "resolve_local_dir", lambda _model: tmp_path / "source")
        def fake_export(*_args, **kwargs):
            assert kwargs["name"] == name + "/hf"
            hf_dir.mkdir(parents=True); (hf_dir / "config.json").write_text("{}")
        monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    pending = _deferred_threads(monkeypatch, app)
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
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
def test_gguf_atomic_partial_failure_then_retry(name, cached, gguf_type, tmp_path, monkeypatch):
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
    quant_marker = tmp_path / "quantizer-invoked"
    quantize = tmp_path / "llama-quantize"
    quantize.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(quant_marker)!r}).write_text('yes')\n"
        "pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-quant')\n"
    )
    quantize.chmod(0o755)
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (convert, quantize, None))
    baked = []
    if not cached:
        monkeypatch.setattr(app.manager, "hf_model", object())
        monkeypatch.setattr(app.manager, "jl", object())
        monkeypatch.setattr(app.manager, "meta", {"model_id": "local"})
        monkeypatch.setattr(app.interventions, "active_rules_full", lambda: [{"layers": [0]}])
        monkeypatch.setattr(app.interventions, "_mode", "readthrough")
        monkeypatch.setattr(app.interventions, "_scale", 1.0)
        monkeypatch.setattr(app, "resolve_local_dir", lambda _model: tmp_path / "source")
        def fake_export(*_args, **_kwargs):
            baked.append(True); hf_dir.mkdir(parents=True); sentinel.write_text('{"baked": true}')
        monkeypatch.setattr(app.editing, "export_rebase", fake_export)
    pending = _deferred_threads(monkeypatch, app)
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
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
    assert len(baked) == (0 if cached else 1)


def test_gguf_atomic_reuses_existing_published_base(tmp_path, monkeypatch):
    import api.app as app
    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    job_dir = edits / "job"; hf_dir = job_dir / "hf"; hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}")
    base = job_dir / "job-bf16.gguf"; base.write_bytes(b"PUBLISHED-GGUF")
    marker = tmp_path / "converter-invoked"
    convert = tmp_path / "convert.py"
    convert.write_text(f"import pathlib\npathlib.Path({str(marker)!r}).write_text('bad')\n")
    app._gguf_state.update(state="running", name="job", step="starting", error=None, result=None)
    app._gguf_worker("job", job_dir, "job", "bf16", hf_dir, convert, None, None)
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
    app._gguf_state.update(state="running", name="job", step="starting", error=None, result=None)
    app._gguf_worker("job", job_dir, "job", "bf16", hf_dir, convert, None, None)
    assert app._gguf_state["state"] == "error" and message in app._gguf_state["error"]
    assert not (job_dir / "job-bf16.gguf").exists()
    assert not list(job_dir.glob(".*.tmp-*.gguf"))
