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

def test_hardlinked_indexed_aliases_are_all_emitted_and_reload(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; names = make_hardlinked_indexed_source(tiny, source)
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained(source, local_files_only=True)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    result = editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                                   fmt="full", name="aliases", source_dir=source)
    out = Path(result["out_dir"])
    assert all((out / name).is_file() for name in names)
    reloaded = AutoModelForCausalLM.from_pretrained(out, local_files_only=True).float().eval()
    baked_model = bake(tiny, rules_for(tiny)); ids = torch.tensor([[1, 8, 4]])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, baked_model(ids).logits,
                                   rtol=2e-5, atol=2e-6)


def test_noninformative_inode_does_not_affect_indexed_aliases(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; names = make_hardlinked_indexed_source(tiny, source)
    original_stat = Path.stat
    def zero_inode(self, *args, **kwargs):
        value = original_stat(self, *args, **kwargs)
        if self.name in names:
            return SimpleNamespace(st_mode=value.st_mode, st_ino=0)
        return value
    monkeypatch.setattr(Path, "stat", zero_inode)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    result = editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                                   fmt="full", name="inode-zero", source_dir=source)
    assert all((Path(result["out_dir"]) / name).is_file() for name in names)


def test_equal_mocked_inodes_do_not_collapse_distinct_shards(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source, two_shards=True)
    names = [path.name for path in source.glob("*.safetensors")]
    original_stat = Path.stat
    def same_inode(self, *args, **kwargs):
        value = original_stat(self, *args, **kwargs)
        if self.name in names:
            return SimpleNamespace(st_mode=value.st_mode, st_ino=123)
        return value
    monkeypatch.setattr(Path, "stat", same_inode)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    result = editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                                   fmt="full", name="same-inode", source_dir=source)
    assert all((Path(result["out_dir"]) / name).is_file() for name in names)


def test_indexed_missing_alias_is_rejected_without_publication(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; names = make_hardlinked_indexed_source(tiny, source)
    (source / names[1]).unlink()
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    with pytest.raises(ValueError, match="indexed shard.*missing"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="missing-alias", source_dir=source)
    assert not (editing.EDITS_DIR / "missing-alias").exists()


@pytest.mark.parametrize("case", [
    "root", "metadata", "total_missing", "total_negative", "total_float", "total_bool",
    "empty_map", "empty_key", "filename_none", "filename_list", "tokenizer_name",
    "edit_meta_name", "casefold", "non_safetensors", "traversal", "windows",
    "bogus_key", "wrong_shard",
])
def test_malformed_index_is_controlled_and_never_published(tiny, tmp_path, monkeypatch, case):
    source = tmp_path / "source"; make_source(tiny, source, two_shards=True)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()); names = list(dict.fromkeys(index["weight_map"].values()))
    if case == "root": payload = []
    else:
        payload = index
        if case == "metadata": payload.pop("metadata")
        elif case == "total_missing": payload["metadata"].pop("total_size")
        elif case == "total_negative": payload["metadata"]["total_size"] = -1
        elif case == "total_float": payload["metadata"]["total_size"] = 1.5
        elif case == "total_bool": payload["metadata"]["total_size"] = True
        elif case == "empty_map": payload["weight_map"] = {}
        elif case == "empty_key": payload["weight_map"][""] = names[0]
        elif case == "filename_none": next_key = next(iter(payload["weight_map"])); payload["weight_map"][next_key] = None
        elif case == "filename_list": next_key = next(iter(payload["weight_map"])); payload["weight_map"][next_key] = [names[0]]
        elif case in ("tokenizer_name", "edit_meta_name", "non_safetensors", "traversal", "windows"):
            value = {"tokenizer_name": "tokenizer.json", "edit_meta_name": "edit_meta.json",
                     "non_safetensors": "weights.bin", "traversal": "../evil.safetensors",
                     "windows": r"C:\evil.safetensors"}[case]
            payload["weight_map"][next(iter(payload["weight_map"]))] = value
        elif case == "casefold":
            keys = list(payload["weight_map"]); payload["weight_map"][keys[0]] = "Alias.safetensors"
            payload["weight_map"][keys[1]] = "alias.safetensors"
        elif case == "bogus_key": payload["weight_map"]["model.language_model.bogus"] = names[0]
        elif case == "wrong_shard":
            key = next(key for key, filename in payload["weight_map"].items() if filename == names[0])
            payload["weight_map"][key] = names[1]
    index_path.write_text(json.dumps(payload))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    with pytest.raises(ValueError):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="malformed", source_dir=source)
    assert not (editing.EDITS_DIR / "malformed").exists()
    assert not [
        path for path in editing.EDITS_DIR.glob(".malformed.tmp-*")
        if not path.name.endswith(".lease")
    ]


def test_nonstring_index_key_is_controlled(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source, two_shards=True)
    original_loads = editing.json.loads
    valid = original_loads((source / "model.safetensors.index.json").read_text())
    valid["weight_map"] = {1: next(iter(valid["weight_map"].values()))}
    monkeypatch.setattr(editing.json, "loads", lambda *_a, **_k: valid)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    with pytest.raises(ValueError, match="weight_map key"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="bad-key", source_dir=source)


def test_transaction_rolls_back_index_write_failure(tmp_path, monkeypatch):
    jl = dense_lens(); jl._lm_head.weight = jl._embed_tokens.weight
    jl._hf_model.config = SimpleNamespace(tie_word_embeddings=True)
    direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": 0.5,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    source = tmp_path / "source"; source.mkdir()
    state = {k: v.detach() for k, v in jl._hf_model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, str(source / "model.safetensors"))
    (source / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    total = sum(value.numel() * value.element_size() for value in state.values())
    (source / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": {key: "model.safetensors" for key in state},
    }))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    original = Path.write_text
    def fail_index(self, *args, **kwargs):
        if self.name == "model.safetensors.index.json": raise OSError("index")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", fail_index)
    with pytest.raises(OSError, match="index"):
        editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt="full",
                              name="index-failure", source_dir=source)
    assert not (editing.EDITS_DIR / "index-failure").exists()
    assert not [
        path for path in editing.EDITS_DIR.glob(".index-failure.tmp-*")
        if not path.name.endswith(".lease")
    ]


@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("corruption", ["rank", "shape", "integer"])
def test_tied_embedding_source_contract_fails_before_first_staged_save(
        indexed, corruption, tmp_path, monkeypatch):
    jl = dense_lens(); jl._lm_head.weight = jl._embed_tokens.weight
    jl._hf_model.config = SimpleNamespace(tie_word_embeddings=True)
    direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": 0.5,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    source = tmp_path / "source"; source.mkdir()
    state = {k: v.detach().clone() for k, v in jl._hf_model.state_dict().items()
             if k != "lm_head.weight"}
    embed_key = jl.layout.path + ".embed_tokens.weight"
    embed = state[embed_key]
    if corruption == "rank":
        state[embed_key] = embed.reshape(-1)
    elif corruption == "shape":
        state[embed_key] = embed.reshape(2, -1)
    else:
        state[embed_key] = embed.to(torch.int32)
    shard = source / "model.safetensors"; save_file(state, str(shard))
    (source / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    if indexed:
        total = sum(value.numel() * value.element_size() for value in state.values())
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total},
            "weight_map": {key: shard.name for key in state},
        }))
    source_bytes = shard.read_bytes()
    writes = []
    original_save = editing.save_file
    monkeypatch.setattr(editing, "save_file",
                        lambda *args, **kwargs: writes.append(args[1]) or original_save(*args, **kwargs))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    with pytest.raises(ValueError, match="source shape|dtype"):
        editing.export_rebase(rules, jl, {"dtype": "bf16"}, fmt="full",
                              name="tied-corrupt", source_dir=source)
    assert writes == []
    assert shard.read_bytes() == source_bytes
    assert not (editing.EDITS_DIR / "tied-corrupt").exists()
    assert not [p for p in editing.EDITS_DIR.glob(".tied-corrupt.tmp-*")
                if not p.name.endswith(".lease")]


def test_multishard_indexed_tied_export_uses_authoritative_embedding_owner(
        tmp_path, monkeypatch):
    jl = dense_lens(); jl._lm_head.weight = jl._embed_tokens.weight
    jl._hf_model.config = SimpleNamespace(tie_word_embeddings=True)
    direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": 0.5,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    source = tmp_path / "source"; source.mkdir()
    state = {k: v.detach().clone() for k, v in jl._hf_model.state_dict().items()
             if k != "lm_head.weight"}
    embed_key = "model.embed_tokens.weight"; lm_head_key = "lm_head.weight"
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    groups = [{}, {}]; weight_map = {}
    for index, (key, value) in enumerate(state.items()):
        owner = 1 if key == embed_key else index % 2
        groups[owner][key] = value; weight_map[key] = names[owner]
    for name, group in zip(names, groups): save_file(group, str(source / name))
    total = sum(value.numel() * value.element_size() for value in state.values())
    index_path = source / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {"total_size": total},
                                      "weight_map": weight_map}))
    (source / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
    source_bytes = {path.name: path.read_bytes() for path in source.glob("*.safetensors")}
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    result = editing.export_rebase(rules, jl, {"dtype": "fp32"}, fmt="full",
                                   name="tied-success", source_dir=source)
    out = Path(result["out_dir"]); final_index = json.loads(
        (out / "model.safetensors.index.json").read_text()
    )
    owner = weight_map[embed_key]
    assert final_index["weight_map"][lm_head_key] == owner
    occurrences = []
    for name in names:
        with safe_open(str(out / name), framework="pt") as handle:
            if lm_head_key in handle.keys():
                occurrences.append(name)
                head = handle.get_tensor(lm_head_key)
    assert occurrences == [owner]
    assert head.shape == state[embed_key].shape and head.dtype == state[embed_key].dtype
    staged_index, staged_paths = editing._indexed_shards(out)
    _, logical_total = editing._validate_index_contents(staged_index, staged_paths)
    assert final_index["metadata"]["total_size"] == logical_total
    assert source_bytes == {path.name: path.read_bytes()
                            for path in source.glob("*.safetensors")}
