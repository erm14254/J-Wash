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

from core import capabilities, editing, rebase
from core.ablation import Interventions
from core.model_manager import _rebase_capability_meta
from helpers import *

def test_meta_norm_without_accelerate_hook_fails_closed():
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    jl = dense_lens(); bad = LlamaRMSNorm(8).to(device="meta")
    jl.layers[0].input_layernorm = bad
    meta = _rebase_capability_meta(jl)
    assert not meta["readthrough_supported"] and not meta["exact_supported"]
    assert meta["readthrough_reason"] == capabilities.PUBLIC_REASONS["architecture_unsupported"]


def test_actual_auto_disk_offload_norm_inspection(tmp_path):
    from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=1024, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=4, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=32)
    source = tmp_path / "source"; LlamaForCausalLM(config).save_pretrained(source, safe_serialization=True)
    loaded = AutoModelForCausalLM.from_pretrained(
        source, device_map="auto", max_memory={"cpu": "2MB"},
        offload_folder=tmp_path / "offload", dtype=torch.float32, local_files_only=True).eval()
    norms = [loaded.model.norm] + [n for layer in loaded.model.layers
                                  for n in (layer.input_layernorm, layer.post_attention_layernorm)]
    offloaded = next((norm for norm in norms if norm.weight.device.type == "meta"), None)
    assert offloaded is not None and getattr(offloaded, "_hf_hook", None) is not None
    before = offloaded.weight.device
    gain = rebase.materialize_parameter_for_inspection(offloaded, "weight")
    assert gain.device.type == "cpu" and gain._base is None
    gamma = rebase.effective_gamma(offloaded)
    assert gamma.device.type == "cpu" and torch.isfinite(gamma).all()
    U, V = torch.randn(128, 1), torch.randn(128, 1)
    Ug, Vg = rebase.gamma_pair(offloaded, U, V)
    assert torch.isfinite(Ug).all() and torch.isfinite(Vg).all()
    jl = lens(loaded); rebase.model_inventory(jl)
    first = _rebase_capability_meta(jl); second = _rebase_capability_meta(jl)
    assert first["readthrough_supported"] and second == first
    assert offloaded.weight.device == before
    with torch.no_grad(): output = loaded(torch.tensor([[1, 2, 3]]))
    assert torch.isfinite(output.logits).all() and offloaded.weight.device == before


def test_profile_construction_inspects_each_unique_norm_once(monkeypatch):
    jl = dense_lens()
    before = [parameter.device for parameter in jl._hf_model.parameters()]
    calls = []
    original = rebase.validate_rms_norm
    def counted(norm, hidden, name="RMSNorm", expected=None):
        calls.append(id(norm))
        return original(norm, hidden, name, expected)
    monkeypatch.setattr(rebase, "validate_rms_norm", counted)
    profile = capabilities.build_profile(jl, None)
    assert capabilities.normalize_profile(profile) is not None
    assert len(calls) == len(set(calls)) == 5
    assert [parameter.device for parameter in jl._hf_model.parameters()] == before
    assert all(not module._forward_hooks for module in jl._hf_model.modules())

    import api.app as app
    monkeypatch.setattr(app.manager, "hf_model", jl._hf_model)
    monkeypatch.setattr(app.manager, "meta", {"model_id": "synthetic"})
    monkeypatch.setattr(app.manager, "capability_profile", profile)
    monkeypatch.setattr(app, "gpu_stats", lambda: [])
    app.api_status()
    assert len(calls) == 5


def test_forward_provenance_accepts_only_authentic_accelerate_wrapper():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    expected = rebase.class_contract(nn.Linear)
    module = nn.Linear(4, 4, False)
    add_hook_to_module(module, AlignDevicesHook(execution_device="cpu"))
    rebase.validate_forward_provenance(module, expected, "linear")

    replaced = module
    replaced.forward = lambda x: x
    with pytest.raises(ValueError, match="wrapper shape"):
        rebase.validate_forward_provenance(replaced, expected, "linear")

    fabricated = nn.Linear(4, 4, False)
    fabricated._hf_hook = SimpleNamespace()
    fabricated._old_forward = fabricated.forward
    fabricated.forward = lambda x: x
    with pytest.raises(ValueError, match="Accelerate wrapper"):
        rebase.validate_forward_provenance(fabricated, expected, "linear")

    wrong_old = nn.Linear(4, 4, False)
    other = nn.Linear(4, 4, False)
    add_hook_to_module(wrong_old, AlignDevicesHook(execution_device="cpu"))
    wrong_old._old_forward = other.forward
    with pytest.raises(ValueError, match="Accelerate wrapper"):
        rebase.validate_forward_provenance(wrong_old, expected, "linear")


def test_frozen_class_and_hook_method_identities_reject_spoofs(monkeypatch):
    expected = rebase.class_contract(nn.Linear)
    lookalike = type("Linear", (nn.Module,), {
        "__module__": nn.Linear.__module__,
        "forward": lambda self, value: value,
    })()
    with pytest.raises(ValueError, match="class is not audited"):
        rebase.validate_forward_provenance(lookalike, expected, "linear")
    original = nn.Linear.forward
    monkeypatch.setattr(nn.Linear, "forward", lambda self, value: original(self, value))
    with pytest.raises(ValueError, match="class forward was modified"):
        rebase.validate_forward_provenance(nn.Linear(4, 4), expected, "linear")


def test_accelerate_hook_method_override_fails_closed():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    module = nn.Linear(4, 4, False)
    hook = AlignDevicesHook(execution_device="cpu")
    add_hook_to_module(module, hook)
    hook.pre_forward = lambda _module, *args, **kwargs: (args, kwargs)
    with pytest.raises(ValueError, match="Accelerate wrapper"):
        rebase.validate_forward_provenance(
            module, rebase.class_contract(nn.Linear), "linear"
        )


def test_meta_parameter_direct_lazy_lookup_never_iterates_keys():
    module = nn.Linear(4, 4, False, device="meta")
    value = torch.randn(4, 4)
    class LazyMap:
        def __getitem__(self, key):
            if key == "weight": return value
            raise KeyError(key)
        def keys(self): raise AssertionError("lazy keyspace must not be enumerated")
    module._hf_hook = SimpleNamespace(weights_map=LazyMap())
    got = rebase.materialize_parameter_for_inspection(module, "weight")
    assert got.device.type == "cpu" and got._base is None
    assert torch.equal(got, value) and got.data_ptr() != value.data_ptr()
