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
    jl = lens(loaded); first = _rebase_capability_meta(jl); second = _rebase_capability_meta(jl)
    assert first["readthrough_supported"] and second == first
    assert offloaded.weight.device == before
    with torch.no_grad(): output = loaded(torch.tensor([[1, 2, 3]]))
    assert torch.isfinite(output.logits).all() and offloaded.weight.device == before
