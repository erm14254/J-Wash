from types import SimpleNamespace

import pytest
import torch
from torch import nn

from core import rebase


class RMS(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n))
    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.weight)


def block(linear=False, hidden=8):
    mixer = SimpleNamespace(**({
        "in_proj_qkv": nn.Linear(hidden, 12, False), "in_proj_z": nn.Linear(hidden, 8, False),
        "in_proj_b": nn.Linear(hidden, 2, False), "in_proj_a": nn.Linear(hidden, 2, False),
    } if linear else {
        "q_proj": nn.Linear(hidden, 16, False), "k_proj": nn.Linear(hidden, 4, False),
        "v_proj": nn.Linear(hidden, 4, False),
    }))
    experts = nn.Module(); experts.gate_up_proj = nn.Parameter(torch.randn(4, 6, hidden)); experts.down_proj = nn.Parameter(torch.randn(4, hidden, 3))
    mlp = SimpleNamespace(gate=nn.Linear(hidden, 4, False), experts=experts,
        shared_expert=SimpleNamespace(gate_proj=nn.Linear(hidden, 3, False), up_proj=nn.Linear(hidden, 3, False), down_proj=nn.Linear(3, hidden, False)),
        shared_expert_gate=nn.Linear(hidden, 1, False))
    kwargs = {"linear_attn" if linear else "self_attn": mixer}
    return SimpleNamespace(**kwargs, mlp=mlp, input_layernorm=RMS(hidden), post_attention_layernorm=RMS(hidden))


def test_literal_reader_inventory_and_raw_key():
    expected_full = {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "mlp.gate.weight", "mlp.experts.gate_up_proj", "mlp.shared_expert.gate_proj.weight",
        "mlp.shared_expert.up_proj.weight", "mlp.shared_expert_gate.weight"}
    expected_linear = {"linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight"} | {k for k in expected_full if k.startswith("mlp.")}
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(block())} == expected_full
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(block(True))} == expected_linear
    assert "mlp.experts.gate_up_proj.weight" not in expected_full


def test_rank3_read_matches_explicit_expert_loop_and_validates_shape():
    torch.manual_seed(1); W = torch.randn(4, 6, 8); U = torch.randn(8, 2); V = torch.randn(8, 2)
    got, B, A = rebase.apply_read(W, U, V)
    expected = torch.stack([rebase.apply_read(w, U, V)[0] for w in W])
    assert got.shape == W.shape and torch.allclose(got, expected)
    assert torch.allclose(got - W, B @ A)
    with pytest.raises(ValueError, match="at least two"): rebase.apply_read(torch.ones(8), U, V)
    with pytest.raises(ValueError, match="hidden axis"): rebase.apply_read(torch.ones(2, 7), U, V)


def test_capabilities_fail_closed_and_exact_is_rejected():
    b = block(); cap = rebase.block_capabilities(b)
    assert cap.readthrough_supported and not cap.exact_supported and "aggregate-MoE" in cap.exact_reason
    del b.mlp.shared_expert_gate
    assert not rebase.block_capabilities(b).readthrough_supported
    b = block(); b.linear_attn = SimpleNamespace()
    assert not rebase.block_capabilities(b).readthrough_supported
    b = block(); b.input_layernorm = nn.LayerNorm(8)
    assert not rebase.block_capabilities(b).readthrough_supported


def test_official_packed_memory_dimensions():
    sizes = {"35B": (256, 1024, 2048), "122B": (256, 2048, 3072)}
    assert {name: 2 * a * b * c for name, (a, b, c) in sizes.items()} == {"35B": 1 << 30, "122B": 3 << 30}


def test_transformers_55_tiny_hybrid_inventory():
    """No tokenizer/download: validate against the actual supported classes."""
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.0":
        pytest.skip("support contract is developed against Transformers 5.5.0")
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(vocab_size=67, hidden_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        linear_conv_kernel_dim=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, moe_intermediate_size=16,
        shared_expert_intermediate_size=12, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    cfg._attn_implementation = "eager"; cfg._experts_implementation = "eager"
    model = Qwen3_5MoeForCausalLM(cfg).float().eval()
    assert [rebase.block_capabilities(layer).readthrough_supported for layer in model.model.layers] == [True] * 3
    packed = model.model.layers[1].mlp.experts.gate_up_proj
    assert packed.ndim == 3
    assert "model.layers.1.mlp.experts.gate_up_proj" in model.state_dict()
