from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core import editing, rebase


FULL_READERS = {
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "mlp.gate.weight", "mlp.experts.gate_up_proj",
    "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert_gate.weight",
}
LINEAR_READERS = {
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
    "mlp.gate.weight", "mlp.experts.gate_up_proj",
    "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert_gate.weight",
}


class RMS(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * self.weight


def block(linear=False, hidden=8):
    mixer = nn.Module()
    names = ({"in_proj_qkv": 12, "in_proj_z": 8, "in_proj_b": 2, "in_proj_a": 2,
              "out_proj": hidden} if linear else
             {"q_proj": 16, "k_proj": 4, "v_proj": 4, "o_proj": hidden})
    for name, out in names.items():
        setattr(mixer, name, nn.Linear(hidden, out, False))
    mlp = nn.Module()
    experts = nn.Module()
    experts.gate_up_proj = nn.Parameter(torch.randn(4, 6, hidden))
    mlp.experts = experts
    mlp.gate = nn.Linear(hidden, 4, False)
    shared = nn.Module()
    shared.gate_proj = nn.Linear(hidden, 3, False)
    shared.up_proj = nn.Linear(hidden, 3, False)
    mlp.shared_expert = shared
    mlp.shared_expert_gate = nn.Linear(hidden, 1, False)
    result = nn.Module()
    setattr(result, "linear_attn" if linear else "self_attn", mixer)
    result.mlp = mlp
    result.input_layernorm = RMS(hidden)
    result.post_attention_layernorm = RMS(hidden)
    return result


def test_literal_reader_inventory_and_raw_key():
    assert {target.state_suffix for target, _, _ in rebase.iter_reads(block())} == FULL_READERS
    assert {target.state_suffix for target, _, _ in rebase.iter_reads(block(True))} == LINEAR_READERS
    assert "mlp.experts.gate_up_proj.weight" not in FULL_READERS | LINEAR_READERS


def test_rank3_read_matches_explicit_expert_loop_and_validates_shape():
    torch.manual_seed(1)
    weight = torch.randn(4, 6, 8)
    u = torch.randn(8, 2)
    v = torch.randn(8, 2)
    got, b, a = rebase.apply_read(weight, u, v)
    expected = torch.stack([rebase.apply_read(expert, u, v)[0] for expert in weight])
    assert got.shape == weight.shape
    assert torch.allclose(got, expected)
    assert torch.allclose(got - weight, b @ a)
    with pytest.raises(ValueError, match="at least two"):
        rebase.apply_read(torch.ones(8), u, v)
    with pytest.raises(ValueError, match="hidden axis"):
        rebase.apply_read(torch.ones(2, 7), u, v)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("budget", [1, 3, 100])
def test_bounded_read_transform_matches_math(dtype, budget):
    torch.manual_seed(2)
    source = torch.randn(3, 5, 8).to(dtype)
    u, v = torch.randn(8, 2), torch.randn(8, 2)
    observed = []
    got, delta = editing.apply_read_transform_bounded(
        ("read", u, v), source, row_budget=budget, observer=observed.append
    )
    expected = rebase.apply_read(source.float(), u, v)[0].to(dtype)
    assert got.shape == source.shape and got.dtype == source.dtype
    assert torch.equal(got, expected)
    assert max(observed) <= budget and sum(observed) == 15
    expected_delta = (rebase.apply_read(source.float(), u, v)[0] - source.float()).abs().max()
    assert delta == pytest.approx(float(expected_delta))


def test_build_plan_raw_key_and_exact_noop_precedence():
    layer = block()
    jl = SimpleNamespace(
        layers=[layer, block()], _final_norm=RMS(8),
        _lm_head=nn.Linear(8, 10, False), _embed_tokens=nn.Embedding(10, 8),
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"),
    )
    direction = torch.randn(8)
    direction /= direction.norm()
    rule = {"mode": "scale", "factor": 0.5, "layers": [0],
            "dirs_a": {0: direction}, "dirs_b": None}
    transforms, info = rebase.build_plan([rule], jl, 1.0)
    assert info["packed_moe"] is True
    assert "model.layers.1.mlp.experts.gate_up_proj" in transforms
    assert "model.layers.1.mlp.experts.gate_up_proj.weight" not in transforms
    with pytest.raises(ValueError, match="no active rule"):
        rebase.build_plan([{**rule, "layers": []}], jl, 1.0, exact=True)
    with pytest.raises(ValueError, match="all coefficients neutral"):
        rebase.build_plan([{**rule, "factor": 1.0}], jl, 1.0, exact=True)
    with pytest.raises(ValueError, match="Exact is not implemented for packed MoE; use Readthrough"):
        rebase.build_plan([rule], jl, 1.0, exact=True)
