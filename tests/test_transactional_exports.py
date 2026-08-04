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

@pytest.mark.parametrize("shape", [(9, 8), (3, 7, 8)])
@pytest.mark.parametrize("budget", [1, 2, 64])
def test_bounded_read_transform_matches_math(shape, budget):
    torch.manual_seed(4); source = torch.randn(*shape, dtype=torch.bfloat16)
    U, V = torch.randn(8, 3), torch.randn(8, 3); seen = []
    got, delta = editing.apply_transform_bounded(
        ("read", U, V), source, row_budget=budget, observer=seen.append)
    expected = rebase.apply_read(source.float(), U, V)[0].to(source.dtype)
    assert torch.equal(got, expected)
    assert got.shape == source.shape and got.dtype == source.dtype
    assert seen and max(seen) <= budget and delta > 0


def test_full_export_observer_covers_every_bounded_row(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source)
    disk = load_file(str(next(source.glob("*.safetensors"))))
    transforms, info = rebase.build_plan(rules_for(tiny), lens(tiny), 1.0)
    mapper = editing._disk_mapper(info["embed_key"], set(disk))
    row_counts = [disk[mapper(key)].numel() // disk[mapper(key)].shape[-1]
                  for key, entry in transforms.items() if entry[0] == "read"]
    observed = []
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(editing, "REBASE_EXPORT_ROW_BUDGET", 3)
    monkeypatch.setattr(editing, "REBASE_EXPORT_CHUNK_OBSERVER", observed.append)
    editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                          fmt="full", name="observed", source_dir=source)
    assert sum(observed) == sum(row_counts)
    assert len(observed) == sum((rows + 2) // 3 for rows in row_counts)
    assert max(observed) == 3 and all(0 < count <= 3 for count in observed)
    assert any(count < 3 for count in observed)


def test_public_export_builds_one_semantic_inventory(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    calls = 0
    original = rebase.model_inventory
    def counted(jl):
        nonlocal calls
        calls += 1
        return original(jl)
    monkeypatch.setattr(rebase, "model_inventory", counted)
    editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                          fmt="full", name="single-pass", source_dir=source)
    assert calls == 1


def test_packed_export_rejections_are_early_and_clean(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path); jl = lens(tiny); rules = rules_for(tiny)
    def forbidden(): raise AssertionError("full state traversal occurred")
    monkeypatch.setattr(jl._hf_model, "state_dict", forbidden)
    for fmt, phrase in (("layers", "layers export"), ("lora", "lora export")):
        with pytest.raises(ValueError, match=phrase):
            editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt=fmt, name=fmt)
        assert not (tmp_path / fmt).exists()
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt="full", name="exact", exact=True)
    assert not (tmp_path / "exact").exists()


def test_full_checkpoint_export_reload_mapping_mtp_and_missing(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source)
    shard = next(source.glob("*.safetensors")); state = load_file(str(shard))
    sentinel = torch.arange(12, dtype=torch.int16).reshape(3, 4)
    state["mtp.sentinel"] = sentinel; save_file(state, str(shard))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    rules = rules_for(tiny); result = editing.export_rebase(
        rules, lens(tiny), {"dtype": "fp16", "model_id": "local"}, fmt="full",
        name="full", source_dir=source)
    out = Path(result["out_dir"]); out_state = load_file(str(out / shard.name))
    packed_key = "model.language_model.layers.1.mlp.experts.gate_up_proj"
    assert packed_key in out_state and packed_key + ".weight" not in out_state
    assert out_state[packed_key].shape == state[packed_key].shape
    assert out_state[packed_key].dtype == state[packed_key].dtype
    assert torch.equal(out_state["mtp.sentinel"], sentinel)
    meta = json.loads((out / "edit_meta.json").read_text())
    assert any("MTP" in warning for warning in meta["warnings"])
    # Reload the artifact exactly as emitted: no prefix rewrite and no MTP deletion.
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).float().eval()
    baked_model = bake(tiny, rules); ids = torch.tensor([[1, 8, 4]])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, baked_model(ids).logits, rtol=2e-5, atol=2e-6)
    copytree = __import__("shutil").copytree
    broken = tmp_path / "broken"; copytree(source, broken)
    broken_shard = next(broken.glob("*.safetensors")); broken_state = load_file(str(broken_shard))
    del broken_state[packed_key]; save_file(broken_state, str(broken_shard))
    with pytest.raises(ValueError, match="absent from the source"):
        editing.export_rebase(rules, lens(tiny), {"dtype": "fp16"}, fmt="full",
                              name="broken-output", source_dir=broken)
    assert not (editing.EDITS_DIR / "broken-output").exists()


def test_existing_export_is_never_overwritten(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path)
    existing = tmp_path / "same"; existing.mkdir(); marker = existing / "valid"
    marker.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="already exists"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp16"},
                              fmt="full", name="same", source_dir=tmp_path / "missing")
    assert marker.read_bytes() == b"preserve me"
    assert not list(tmp_path.glob(".same.tmp-*"))


def test_nested_gguf_style_export_is_transactional(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source)
    edits = tmp_path / "edits"; monkeypatch.setattr(editing, "EDITS_DIR", edits)
    result = editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                                   fmt="full", name="job/hf", source_dir=source)
    final = edits / "job" / "hf"
    assert Path(result["out_dir"]) == final.resolve() and (final / "edit_meta.json").exists()
    assert not [p for p in final.parent.glob(".hf.tmp-*") if not p.name.endswith(".lease")]
    with pytest.raises(ValueError, match="already exists"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="job/hf", source_dir=source)
    assert (final / "edit_meta.json").exists()
    inside_absolute = str((edits / "inside").resolve())
    for unsafe in ("../escape", "/tmp/escape", inside_absolute, "job/../other",
                   r"..\escape", r"C:\drive\leaf", r"\\server\share", "job/./hf",
                   "CON/file", "job/leaf."):
        with pytest.raises(ValueError):
            editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                                  fmt="full", name=unsafe, source_dir=source)
    outside = tmp_path / "outside"; outside.mkdir()
    (edits / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="beneath"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="link/hf", source_dir=source)


@pytest.mark.parametrize("failure", ["shard", "metadata"])
def test_nested_gguf_style_failures_leave_no_partial(tiny, tmp_path, monkeypatch, failure):
    source = tmp_path / "source"; make_source(tiny, source)
    edits = tmp_path / "edits"; monkeypatch.setattr(editing, "EDITS_DIR", edits)
    if failure == "shard":
        monkeypatch.setattr(editing, "save_file", lambda *_a, **_k: (_ for _ in ()).throw(OSError("shard")))
    else:
        original = Path.write_text
        def fail_meta(self, *args, **kwargs):
            if self.name == "edit_meta.json": raise OSError("metadata")
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", fail_meta)
    with pytest.raises(OSError, match=failure):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="job/hf", source_dir=source)
    parent = edits / "job"
    assert not (parent / "hf").exists()
    assert not [p for p in parent.glob(".hf.tmp-*") if not p.name.endswith(".lease")]


@pytest.mark.parametrize("failure", ["second_shard", "config", "auxiliary", "metadata"])
def test_full_export_staging_rolls_back_every_failure(tiny, tmp_path, monkeypatch, failure):
    source = tmp_path / "source"; make_source(tiny, source, two_shards=True)
    (source / "tokenizer.txt").write_text("sentinel")
    edits = tmp_path / "edits"; monkeypatch.setattr(editing, "EDITS_DIR", edits)
    if failure == "second_shard":
        original = editing.save_file; calls = 0
        def failing_save(*args, **kwargs):
            nonlocal calls; calls += 1
            if calls == 2: raise OSError("second shard")
            return original(*args, **kwargs)
        monkeypatch.setattr(editing, "save_file", failing_save)
    elif failure == "auxiliary":
        monkeypatch.setattr(editing.shutil, "copy2", lambda *_a, **_k: (_ for _ in ()).throw(OSError("auxiliary")))
    else:
        original = Path.write_text
        target = "config.json" if failure == "config" else "edit_meta.json"
        def failing_write(self, *args, **kwargs):
            if self.name == target: raise OSError(failure)
            return original(self, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", failing_write)
    with pytest.raises(OSError, match=failure.replace("_", " ") if failure == "second_shard" else failure):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="failed", source_dir=source)
    assert not (edits / "failed").exists()
    assert not [p for p in edits.glob(".failed.tmp-*") if not p.name.endswith(".lease")]


def test_bf16_live_bake_and_direct_reload(tiny, tmp_path, monkeypatch):
    model = copy.deepcopy(tiny).bfloat16().eval(); rules = rules_for(model)
    ids = torch.tensor([[1, 8, 4]])
    oracle, _ = run_variant(model, ids, rules, "oracle")
    production, _ = run_variant(model, ids, rules, "production")
    baked_model = bake(model, rules); baked, _ = run_variant(baked_model, ids)
    torch.testing.assert_close(oracle["logits"], production["logits"], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(oracle["logits"], baked["logits"], rtol=2e-2, atol=2e-2)
    source = tmp_path / "bf16-source"; make_source(model, source, dtype=torch.bfloat16)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    result = editing.export_rebase(rules, lens(model), {"dtype": "bf16"}, fmt="full",
                                   name="bf16", source_dir=source)
    out = Path(result["out_dir"])
    packed = load_file(str(next(out.glob("*.safetensors"))))[
        "model.language_model.layers.1.mlp.experts.gate_up_proj"]
    assert packed.dtype == torch.bfloat16
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, baked_model(ids).logits,
                                   rtol=2e-2, atol=2e-2)


def test_shared_dense_full_layers_and_lora_exports(tmp_path, monkeypatch):
    jl = dense_lens(); torch.manual_seed(2); direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": 0.5,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    source = tmp_path / "dense-source"; source.mkdir()
    save_file({k: v.detach() for k, v in jl._hf_model.state_dict().items()},
              str(source / "model.safetensors"))
    (source / "config.json").write_text("{}")
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    full = editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt="full",
                                 name="dense-full", source_dir=source)
    assert (Path(full["out_dir"]) / "model.safetensors").exists()
    layers = editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt="layers",
                                   name="dense-layers", source_dir=source)
    assert load_file(str(Path(layers["out_dir"]) / "modified_layers.safetensors"))
    lora = editing.export_rebase(rules, jl, {"dtype": "fp16", "model_id": "local"},
                                 fmt="lora", name="dense-lora")
    lora_dir = Path(lora["out_dir"])
    assert load_file(str(lora_dir / "adapter_model.safetensors"))
    config = json.loads((lora_dir / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["bias"] == "none"
