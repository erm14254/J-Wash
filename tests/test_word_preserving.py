"""Word-preserving controls: partial replace (keep), preserve_lm_head plan,
and the analytic vocabulary-impact report."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core import editing, rebase
from core.ablation import Interventions, effective_coeffs
from core.impact import impact_report

VOCAB, DIM = 10, 8


class RMS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def unit(weight, index):
    row = weight[index].float()
    return row / row.norm()


def surface_rule(weight, a, b, layers, keep=0.0, factor=1.0):
    """Replace rule whose directions are the raw unembedding rows (the
    'surface' worst case: exactly what kills the word at the output)."""
    return {
        "id": 1, "token_id": a, "token_ids": [a], "token": f"<{a}>",
        "mode": "replace", "factor": factor, "keep": keep,
        "replacement_id": b, "replacement_ids": [b], "replacement": f"<{b}>",
        "layers": layers,
        "dirs_a": {l: unit(weight, a) for l in layers},
        "dirs_b": {l: unit(weight, b) for l in layers},
        "enabled": True,
    }


def test_effective_coeffs_keep():
    assert effective_coeffs("replace", 1.0, 1.0) == (-1.0, 1.0)
    alpha, beta = effective_coeffs("replace", 0.8, 1.0, keep=0.3)
    assert alpha == pytest.approx(-0.7) and beta == pytest.approx(0.8)
    # clamped to [0, 1]; scale mode ignores keep
    assert effective_coeffs("replace", 1.0, 1.0, keep=2.0)[0] == 0.0
    assert effective_coeffs("scale", 0.5, 1.0, keep=0.9) == effective_coeffs("scale", 0.5, 1.0)


def test_rule_factors_apply_keep():
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, DIM)
    rule = surface_rule(weight, 3, 5, [0], keep=0.4)
    ((w, v_a),) = rebase.rule_factors([rule], 1.0)[0]
    expected = -0.6 * unit(weight, 3) + 1.0 * unit(weight, 5)
    torch.testing.assert_close(w, expected)
    torch.testing.assert_close(v_a, unit(weight, 3))


def make_jl(n_layers=2):
    torch.manual_seed(1)
    weight = torch.randn(VOCAB, DIM)
    lm_head = nn.Linear(DIM, VOCAB, False)
    lm_head.weight.data = weight.clone()
    return SimpleNamespace(
        layers=[object()] * n_layers,
        _lm_head=lm_head,
        _embed_tokens=nn.Embedding(VOCAB, DIM),
        _final_norm=RMS(DIM),
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"),
    ), weight


def test_build_plan_preserve_skips_lm_head():
    jl, weight = make_jl()
    rule = surface_rule(weight, 3, 5, [0])

    def plan(preserve):
        # blocks are dummies: collect only which KEYS the plan wants, using a
        # fake iter_reads via monkeypatching? — not needed: layer 0 hooks make
        # reads for layer 1 → iter_reads needs a real block. Use layers=[1]
        # (last layer): no downstream layer, only lm_head remains.
        return rebase.build_plan([surface_rule(weight, 3, 5, [1])], jl, 1.0,
                                 preserve_lm_head=preserve)

    transforms, info = plan(False)
    assert set(transforms) == {"lm_head.weight"}
    assert info["preserve_lm_head"] is False
    transforms, info = plan(True)
    assert transforms == {} and info["preserve_lm_head"] is True


def test_export_rebase_preserve_nothing_to_bake():
    jl, weight = make_jl()
    rule = surface_rule(weight, 3, 5, [1])  # last layer only
    with pytest.raises(ValueError, match="nothing to bake"):
        editing.export_rebase([rule], jl, {"dtype": "bf16"}, fmt="layers",
                              name="never-written", preserve_lm_head=True)


class DecodeTok:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def test_impact_report_flags_suppressed_word_and_redirect():
    jl, weight = make_jl()
    rule = surface_rule(weight, 3, 5, [0])
    report = impact_report([rule], jl, 1.0, mode="readthrough",
                           tokenizer=DecodeTok())
    assert report["approx"] is False
    (src,) = report["sources"]
    assert src["token_id"] == 3 and src["token"] == "<3>"
    assert src["retained"] < 0.5  # the word is (nearly) gone at the output
    assert src["redirect"] and src["redirect"][0]["token_id"] == 5

    # partial replace keeps most of the word
    soft = surface_rule(weight, 3, 5, [0], keep=0.8)
    report = impact_report([soft], jl, 1.0, mode="readthrough", tokenizer=DecodeTok())
    assert report["sources"][0]["retained"] > 0.7

    # protected output head: vocabulary intact by construction
    report = impact_report([rule], jl, 1.0, mode="readthrough",
                           preserve_lm_head=True, tokenizer=DecodeTok())
    assert report["preserve_lm_head"] is True
    assert report["sources"] == [] and report["collateral"] == []


def test_interventions_keep_roundtrip():
    torch.manual_seed(2)
    weight = torch.randn(VOCAB, DIM)
    jl = SimpleNamespace(layers=[object()] * 3,
                         _lm_head=SimpleNamespace(weight=weight),
                         tokenizer=DecodeTok())
    lm = SimpleNamespace(lens=SimpleNamespace(jacobians={}))
    iv = Interventions()
    iv.add(lm, jl, token_id=3, mode="replace", factor=1.0, keep=0.25,
           replacement_id=5, layers=[0])
    assert iv.summary()[0]["keep"] == 0.25
    rid = iv.summary()[0]["id"]
    iv.update(rid, keep=0.6)  # keep alone needs no lens/model
    assert iv.summary()[0]["keep"] == 0.6
    assert iv.preserve_lm_head is False
    assert iv.set_preserve_lm_head(True) is True
