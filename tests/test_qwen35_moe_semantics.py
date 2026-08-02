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

@pytest.mark.parametrize("mode,factor", [("replace", 1.0), ("scale", 0.65)])
def test_tiny_semantic_prefill_and_cached_decode(tiny, mode, factor):
    rules = rules_for(tiny, mode, factor); ids = torch.tensor([[1, 8, 11, 4]])
    base, _ = run_variant(tiny, ids)
    oracle, oracle_past = run_variant(tiny, ids, rules, "oracle")
    production, production_past = run_variant(tiny, ids, rules, "production")
    baked_model = bake(tiny, rules); baked, baked_past = run_variant(baked_model, ids)
    assert set(oracle) == set(production) == set(baked)
    for key in oracle:
        if key.endswith("ids"):
            assert torch.equal(oracle[key], production[key]) and torch.equal(oracle[key], baked[key])
        else:
            torch.testing.assert_close(oracle[key], production[key], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(oracle[key], baked[key], rtol=2e-5, atol=2e-6)
    assert not torch.allclose(base["logits"], oracle["logits"])
    next_id = torch.tensor([[9]])
    assert oracle_past is not None and production_past is not None and baked_past is not None
    before = oracle_past.get_seq_length()
    oracle_d, oracle_next = run_variant(tiny, next_id, rules, "oracle", oracle_past)
    production_d, production_next = run_variant(tiny, next_id, rules, "production", production_past)
    baked_d, baked_next = run_variant(baked_model, next_id, past=baked_past)
    assert oracle_next.get_seq_length() == before + 1
    assert production_next.get_seq_length() == before + 1
    assert baked_next.get_seq_length() == before + 1
    torch.testing.assert_close(oracle_d["logits"], production_d["logits"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(oracle_d["logits"], baked_d["logits"], rtol=2e-5, atol=2e-6)
