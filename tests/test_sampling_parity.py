"""Sampler parity with mainstream runtimes: min_p, repetition penalty,
author-shipped generation_config adoption."""

from types import SimpleNamespace

import torch

import config
from core.model_manager import _sample, _shipped_sampling


def test_defaults_match_mainstream_runtimes():
    assert config.DEFAULT_SAMPLING["repetition_penalty"] == 1.1
    assert config.DEFAULT_SAMPLING["min_p"] == 0.05
    assert config.DEFAULT_SAMPLING["thinking"] is False


def test_min_p_filters_the_tail():
    # top token has prob ~0.5; a 1e-4 tail token must never be sampled
    logits = torch.full((100,), -10.0)
    logits[0] = 5.0
    logits[1] = 4.5
    logits[2] = -1.0  # tail
    g = torch.Generator().manual_seed(0)
    picks = {
        _sample(logits.clone(), 1.0, 0.0, 0, g, min_p=0.1) for _ in range(200)
    }
    assert picks <= {0, 1}
    # min_p off: the tail is reachable in principle (probability mass kept)
    probs = torch.softmax(logits, -1)
    assert probs[2] > 0


def test_min_p_zero_is_off():
    logits = torch.zeros(5)  # uniform
    g = torch.Generator().manual_seed(1)
    picks = {_sample(logits.clone(), 1.0, 0.0, 0, g, min_p=0.0) for _ in range(100)}
    assert len(picks) == 5  # nothing filtered


def test_repetition_penalty_discourages_context_tokens():
    logits = torch.tensor([2.0, 2.0, 0.0])
    penalty_ids = torch.tensor([0])
    out = logits.clone()
    _sample(out, 0.0, 0.0, 0, None, 1.5, penalty_ids)  # greedy path mutates logits
    assert out[0] == 2.0 / 1.5 and out[1] == 2.0  # positive score divided
    neg = torch.tensor([-2.0, 1.0])
    _sample(neg, 0.0, 0.0, 0, None, 1.5, torch.tensor([0]))
    assert neg[0] == -3.0  # negative score multiplied (pushed further down)


def test_shipped_sampling_uses_diff_only():
    class Cfg:
        def to_diff_dict(self):
            return {"temperature": 0.6, "top_k": 20, "do_sample": True}

    model = SimpleNamespace(generation_config=Cfg())
    assert _shipped_sampling(model) == {"temperature": 0.6, "top_k": 20}
    assert _shipped_sampling(SimpleNamespace(generation_config=None)) == {}
