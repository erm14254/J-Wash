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
    def __init__(self, n):
        super().__init__(); self.weight = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.weight)


class OrdinaryRMS(nn.Module):
    def __init__(self, n, dtype=torch.float32):
        super().__init__(); self.weight = nn.Parameter(torch.ones(n, dtype=dtype))
    def forward(self, x):
        return (self.weight * x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class BadNorm(nn.Module):
    def __init__(self, n, kind, dtype=torch.float32):
        super().__init__(); self.weight = nn.Parameter(torch.ones(n, dtype=dtype)); self.kind = kind
    def forward(self, x):
        xf = x.float(); rms = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        if self.kind == "offset": out = rms + 0.2
        elif self.kind == "rotated": out = rms.roll(1, -1)
        elif self.kind == "cubic": out = xf ** 3
        elif self.kind == "tanh": out = xf.tanh()
        elif self.kind == "mean": out = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(xf.var(-1, unbiased=False, keepdim=True) + 1e-6)
        elif self.kind == "nonfinite":
            out = rms
            if bool((xf[..., 0] < -1).any()): out = out * torch.tensor(float("nan"), device=x.device)
        return (self.weight.float() * out).to(x.dtype)


class CallableNotHookable:
    def __init__(self, n): self.weight = torch.ones(n)
    def __call__(self, x): return x


def block(linear=False, hidden=8, sparse=True):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    mixer = nn.Module()
    names = ({"in_proj_qkv": 12, "in_proj_z": 8, "in_proj_b": 2, "in_proj_a": 2,
              "out_proj": hidden} if linear else
             {"q_proj": 16, "k_proj": 4, "v_proj": 4, "o_proj": hidden})
    for name, out in names.items(): setattr(mixer, name, nn.Linear(hidden, out, False))
    mlp = nn.Module()
    if sparse:
        experts = nn.Module(); experts.gate_up_proj = nn.Parameter(torch.randn(4, 6, hidden))
        experts.down_proj = nn.Parameter(torch.randn(4, hidden, 3)); mlp.experts = experts
        mlp.gate = nn.Linear(hidden, 4, False)
        shared = nn.Module(); shared.gate_proj = nn.Linear(hidden, 3, False)
        shared.up_proj = nn.Linear(hidden, 3, False); shared.down_proj = nn.Linear(3, hidden, False)
        mlp.shared_expert = shared; mlp.shared_expert_gate = nn.Linear(hidden, 1, False)
    else:
        mlp.gate_proj = nn.Linear(hidden, 3, False); mlp.up_proj = nn.Linear(hidden, 3, False)
        mlp.down_proj = nn.Linear(3, hidden, False)
    result = nn.Module(); setattr(result, "linear_attn" if linear else "self_attn", mixer)
    result.mlp = mlp; result.input_layernorm = LlamaRMSNorm(hidden)
    result.post_attention_layernorm = LlamaRMSNorm(hidden)
    return result


def test_literal_reader_inventory_and_raw_key():
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(block())} == FULL_READERS
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(block(True))} == LINEAR_READERS
    assert "mlp.experts.gate_up_proj.weight" not in FULL_READERS | LINEAR_READERS


def test_rank3_read_matches_explicit_expert_loop_and_validates_shape():
    torch.manual_seed(1); W = torch.randn(4, 6, 8); U = torch.randn(8, 2); V = torch.randn(8, 2)
    got, B, A = rebase.apply_read(W, U, V)
    expected = torch.stack([rebase.apply_read(w, U, V)[0] for w in W])
    assert got.shape == W.shape and torch.allclose(got, expected)
    assert torch.allclose(got - W, B @ A)
    with pytest.raises(ValueError, match="at least two"): rebase.apply_read(torch.ones(8), U, V)
    with pytest.raises(ValueError, match="hidden axis"): rebase.apply_read(torch.ones(2, 7), U, V)


def test_capabilities_fail_closed_and_dense_exact_is_positive():
    cap = rebase.block_capabilities(block())
    assert cap.readthrough_supported and not cap.exact_supported and "aggregate-MoE" in cap.exact_reason
    for linear in (False, True):
        dense = block(linear, sparse=False)
        assert rebase.block_capabilities(dense).exact_supported
        writer = dense.linear_attn.out_proj if linear else dense.self_attn.o_proj
        writer.bias = nn.Parameter(torch.zeros(8))
        cap = rebase.block_capabilities(dense)
        assert cap.readthrough_supported and not cap.exact_supported and "bias" in cap.exact_reason
        writer.bias = None; writer.weight = nn.Parameter(torch.zeros(7, 8))
        assert "output axis" in rebase.block_capabilities(dense).exact_reason
    b = block(); del b.mlp.shared_expert_gate
    assert not rebase.block_capabilities(b).readthrough_supported
    b = block(); b.linear_attn = nn.Module()
    assert not rebase.block_capabilities(b).readthrough_supported
    b = block(); b.input_layernorm = nn.LayerNorm(8)
    assert not rebase.block_capabilities(b).readthrough_supported
    b = block(); b.pre_feedforward_layernorm = RMS(8)
    assert "unsupported residual branch" in rebase.block_capabilities(b).readthrough_reason


def test_official_packed_memory_dimensions():
    sizes = {"35B": (256, 1024, 2048), "122B": (256, 2048, 3072)}
    assert {name: 2 * a * b * c for name, (a, b, c) in sizes.items()} == {"35B": 1 << 30, "122B": 3 << 30}


@pytest.fixture(scope="module")
def tiny():
    import transformers
    from packaging.version import Version
    assert Version("5.5") <= Version(transformers.__version__) < Version("5.15"), (
        "Qwen3.5-MoE integration requires transformers>=5.5,<5.15; installed " + transformers.__version__)
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=67, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        linear_conv_kernel_dim=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, moe_intermediate_size=16,
        shared_expert_intermediate_size=12, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True})
    cfg._attn_implementation = "eager"; cfg._experts_implementation = "eager"
    torch.manual_seed(7)
    model = Qwen3_5MoeForCausalLM(cfg).float().eval()
    return model


def lens(model):
    return SimpleNamespace(
        layers=model.model.layers, _final_norm=model.model.norm,
        _lm_head=model.lm_head, _embed_tokens=model.model.embed_tokens,
        _hf_model=model,
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"),
    )


def rules_for(model, mode="replace", factor=1.0):
    W = model.lm_head.weight.detach().float()
    unit = lambda i: W[i] / W[i].norm().clamp_min(1e-8)
    return [{"id": 1, "token_id": 3, "token": "<3>", "mode": mode, "factor": factor,
             "replacement_id": 5 if mode == "replace" else None,
             "replacement": "<5>" if mode == "replace" else None, "layers": [0],
             "dirs_a": {0: unit(3)}, "dirs_b": {0: unit(5)} if mode == "replace" else None}]


def rule_at(model, layer):
    rule = rules_for(model)[0]
    rule["layers"] = [layer]
    rule["dirs_a"] = {layer: rule["dirs_a"][0]}
    rule["dirs_b"] = {layer: rule["dirs_b"][0]}
    return [rule]


def bake(model, rules, scale=1.0):
    clone = copy.deepcopy(model); transforms, _ = rebase.build_plan(rules, lens(model), scale)
    state = clone.state_dict()
    for key, transform in transforms.items():
        state[key] = rebase.apply_transform(transform, state[key].float())[0]
    clone.load_state_dict(state); return clone.eval()


def oracle_handles(model, rules, scale=1.0):
    """Independent live oracle: literal norm sites, never production iter_reads()."""
    cums = rebase.cumulative(rules, scale, len(model.model.layers)); handles = []
    def attach(norm, U, V):
        Ug, Vg = rebase.gamma_pair(norm, U, V)
        def hook(_m, _i, out): return out + (out @ Vg.to(out)) @ Ug.to(out).T
        handles.append(norm.register_forward_hook(hook))
    for index in (1, 2):
        U, V = cums[index]
        attach(model.model.layers[index].input_layernorm, U, V)
        attach(model.model.layers[index].post_attention_layernorm, U, V)
    attach(model.model.norm, *cums[3])
    return handles


def capture_moe(model):
    captured, handles = {}, []
    for i in (1, 2):
        mlp = model.model.layers[i].mlp
        def gate_hook(_m, inp, out, i=i, mlp=mlp):
            logits, weights, ids = out
            x = inp[0].reshape(-1, inp[0].shape[-1]); packed = mlp.experts.gate_up_proj
            routed = [torch.nn.functional.linear(x, packed[e]).chunk(2, -1)
                      for e in range(packed.shape[0])]
            captured[f"{i}.router"] = logits.detach(); captured[f"{i}.weights"] = weights.detach()
            captured[f"{i}.ids"] = ids.detach()
            captured[f"{i}.routed_gate"] = torch.stack([pair[0] for pair in routed], 1).detach()
            captured[f"{i}.routed_up"] = torch.stack([pair[1] for pair in routed], 1).detach()
        handles.append(mlp.gate.register_forward_hook(gate_hook))
        handles.append(mlp.shared_expert_gate.register_forward_hook(
            lambda _m, _i, out, i=i: captured.__setitem__(f"{i}.shared_gate", out.sigmoid().detach())))
        handles.append(mlp.register_forward_hook(
            lambda _m, _i, out, i=i: captured.__setitem__(f"{i}.aggregate", out.detach())))
    return captured, handles


def run_variant(model, ids, rules=None, kind="base", past=None):
    captures, capture_handles = capture_moe(model); handles = []
    iv = None
    if rules and kind == "oracle": handles = oracle_handles(model, rules)
    if rules and kind == "production":
        iv = Interventions(); iv._rules = rules; iv.set_mode("readthrough"); iv.attach(lens(model))
    try:
        with torch.no_grad(): out = model(ids, past_key_values=past, use_cache=True)
        captures["logits"] = out.logits.detach()
        return captures, out.past_key_values
    finally:
        for h in handles + capture_handles: h.remove()
        if iv: iv.detach()


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


def test_tiny_supported_inventory_and_version_contract(tiny):
    assert [rebase.block_capabilities(x).readthrough_supported for x in tiny.model.layers] == [True] * 3
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(tiny.model.layers[1])} == FULL_READERS
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(tiny.model.layers[2])} == LINEAR_READERS
    assert tiny.model.layers[1].mlp.experts.gate_up_proj.ndim == 3
    assert "model.layers.1.mlp.experts.gate_up_proj" in tiny.state_dict()


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_packed_exact_is_model_wide_and_live_rejection_is_clean(tiny, layer):
    jl = lens(tiny); rules = rule_at(tiny, layer)
    with pytest.raises(ValueError, match="exact mode is unavailable for this model"):
        rebase.build_plan(rules, jl, 1.0, exact=True)
    before = [len(module._forward_hooks) for module in tiny.modules()]
    iv = Interventions(); iv._rules = rules; iv.set_mode("exact")
    with pytest.raises(ValueError, match="exact mode is unavailable for this model"):
        iv.attach(jl)
    assert before == [len(module._forward_hooks) for module in tiny.modules()]
    assert iv._handles == []


def test_api_and_legacy_capability_fields(tiny):
    meta = _rebase_capability_meta(lens(tiny))
    assert meta["readthrough_supported"] and meta["rebase_supported"]
    assert not meta["exact_supported"] and "aggregate-MoE" in meta["exact_reason"]
    bad = lens(tiny); bad.layers[0].pre_feedforward_layernorm = RMS(32)
    try:
        rejected = _rebase_capability_meta(bad)
        assert not rejected["readthrough_supported"] and not rejected["rebase_supported"]
        assert "unsupported residual branch" in rejected["readthrough_reason"]
    finally:
        del bad.layers[0].pre_feedforward_layernorm


def test_live_registration_failure_rolls_back(tiny, monkeypatch):
    jl = lens(tiny); iv = Interventions(); iv._rules = rules_for(tiny)
    iv.set_mode("readthrough")
    before = [len(module._forward_hooks) for module in tiny.modules()]
    def fail(_hook): raise RuntimeError("injected final registration failure")
    monkeypatch.setattr(jl._final_norm, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="injected"):
        iv.attach(jl)
    assert before == [len(module._forward_hooks) for module in tiny.modules()]
    assert iv._handles == []


def test_norm_semantics_and_final_norm_preflight(tiny):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    # Positive regressions: Qwen3.5 zero-centered RMSNorm and audited dense RMS.
    rebase.validate_rms_norm(tiny.model.norm, 32, "qwen final")
    rebase.validate_rms_norm(LlamaRMSNorm(8), 8, "Llama RMS")
    for where in ("input", "post"):
        dense = block(sparse=False)
        setattr(dense, "input_layernorm" if where == "input" else "post_attention_layernorm",
                nn.LayerNorm(8, bias=False))
        cap = rebase.block_capabilities(dense)
        assert not cap.readthrough_supported and "LayerNorm" in cap.readthrough_reason
    bad = lens(tiny); bad._final_norm = nn.LayerNorm(32, bias=False)
    with pytest.raises(ValueError, match="LayerNorm"):
        rebase.model_preflight(bad)
    for value, phrase in ((SimpleNamespace(weight=torch.ones(32)), "noncallable"),
                          (lambda x: x, "hookable")):
        bad = lens(tiny); bad._final_norm = value
        with pytest.raises(ValueError, match=phrase): rebase.model_preflight(bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_audited_hf_rmsnorm_classes_and_dtypes(dtype):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    for cls in (LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, Qwen3RMSNorm,
                Qwen3_5RMSNorm, Qwen3_5MoeRMSNorm):
        rebase.validate_rms_norm(cls(8).to(dtype=dtype), 8, cls.__name__)


@pytest.mark.parametrize("hidden", [2048, 3072])
def test_production_width_bf16_allowlisted_rmsnorms(hidden):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from transformers.models.mistral.modeling_mistral import MistralRMSNorm
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    for cls in (LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm, Qwen3RMSNorm,
                Qwen3_5RMSNorm, Qwen3_5MoeRMSNorm):
        rebase.validate_rms_norm(cls(hidden).to(torch.bfloat16), hidden, cls.__name__)


@pytest.mark.parametrize("hidden", [2048, 3072])
def test_production_width_qwen35_moe_block_capability(hidden):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    b = nn.Module(); b.self_attn = nn.Module()
    b.self_attn.q_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.self_attn.k_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.self_attn.v_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.mlp = nn.Module(); b.mlp.gate = nn.Linear(hidden, 4, False, dtype=torch.bfloat16)
    b.mlp.experts = nn.Module(); b.mlp.experts.gate_up_proj = nn.Parameter(torch.randn(2, 8, hidden, dtype=torch.bfloat16))
    b.mlp.shared_expert = nn.Module()
    b.mlp.shared_expert.gate_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.mlp.shared_expert.up_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.mlp.shared_expert_gate = nn.Linear(hidden, 1, False, dtype=torch.bfloat16)
    b.input_layernorm = Qwen3_5MoeRMSNorm(hidden).to(torch.bfloat16)
    b.post_attention_layernorm = Qwen3_5MoeRMSNorm(hidden).to(torch.bfloat16)
    cap = rebase.block_capabilities(b)
    assert cap.readthrough_supported and not cap.exact_supported


def test_meta_norm_without_accelerate_hook_fails_closed():
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    jl = dense_lens(); bad = LlamaRMSNorm(8).to(device="meta")
    jl.layers[0].input_layernorm = bad
    meta = _rebase_capability_meta(jl)
    assert not meta["readthrough_supported"] and not meta["exact_supported"]
    assert "no supported Accelerate weights_map" in meta["readthrough_reason"]


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


def test_norm_contract_revalidates_mutation_and_rejects_unknowns():
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    norm = LlamaRMSNorm(8); rebase.validate_rms_norm(norm, 8)
    norm.weight = nn.Parameter(torch.tensor([0., 1., 1., 1., 1., 1., 1., 1.]))
    with pytest.raises(ValueError, match="singular"): rebase.validate_rms_norm(norm, 8)
    norm = LlamaRMSNorm(8); rebase.validate_rms_norm(norm, 8)
    norm.weight.data[0] = 0
    with pytest.raises(ValueError, match="singular"): rebase.validate_rms_norm(norm, 8)
    norm = LlamaRMSNorm(8); norm.variance_epsilon = -1
    with pytest.raises(ValueError, match="epsilon"): rebase.validate_rms_norm(norm, 8)
    with pytest.raises(ValueError, match="not audited"):
        rebase.validate_rms_norm(OrdinaryRMS(8), 8)


class ContainerWriter(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.weight = nn.Parameter(torch.eye(8)); self.bias = None; self.kind = kind
    def forward(self, x):
        value = nn.functional.linear(x, self.weight)
        if self.kind == "tuple": return (value,)
        if self.kind == "namedtuple":
            from collections import namedtuple
            return namedtuple("Output", "hidden")(value)
        if self.kind == "list": return [value]
        if self.kind == "mapping": return {"hidden": value}
        return value


@pytest.mark.parametrize("kind", ["tensor", "tuple", "namedtuple", "list", "mapping"])
def test_exact_writer_contract_is_linear_tensor_only(kind):
    jl = dense_lens(); dense = jl.layers[1]
    if kind != "tensor": dense.self_attn.o_proj = ContainerWriter(kind)
    cap = rebase.block_capabilities(dense)
    assert cap.exact_supported is (kind == "tensor")
    if kind != "tensor":
        assert "audited tensor-returning Linear" in cap.exact_reason
        direction = torch.randn(8); direction /= direction.norm()
        iv = Interventions(); iv._rules = [{"id": 1, "token_id": 1, "token": "x",
            "mode": "scale", "factor": .8, "replacement_id": None, "replacement": None,
            "layers": [0], "dirs_a": {0: direction}, "dirs_b": None}]
        iv.set_mode("exact"); before = [len(module._forward_hooks) for module in jl._hf_model.modules()]
        with pytest.raises(ValueError, match="audited tensor-returning Linear"): iv.attach(jl)
        assert before == [len(module._forward_hooks) for module in jl._hf_model.modules()]
        assert iv._handles == []


@pytest.mark.parametrize("site", ["input", "post", "final"])
@pytest.mark.parametrize("kind", ["layernorm", "mean", "offset", "rotated", "cubic", "tanh", "nonfinite", "nonhookable"])
def test_adversarial_norms_fail_closed(tiny, site, kind):
    candidate = (nn.LayerNorm(8, bias=False) if kind == "layernorm" else
                 CallableNotHookable(8) if kind == "nonhookable" else
                 BadNorm(8, kind, torch.bfloat16 if kind == "mean" else torch.float32))
    if site == "final":
        jl = dense_lens(); jl._final_norm = candidate
        meta = _rebase_capability_meta(jl)
        assert not meta["readthrough_supported"] and not meta["exact_supported"]
    else:
        dense = block(sparse=False)
        attr = "input_layernorm" if site == "input" else "post_attention_layernorm"
        if kind == "nonhookable":
            dense._modules.pop(attr); object.__setattr__(dense, attr, candidate)
        else:
            setattr(dense, attr, candidate)
        cap = rebase.block_capabilities(dense)
        assert not cap.readthrough_supported and not cap.exact_supported


def test_live_read_hook_uses_bf16_output_not_fp32_gain(tiny):
    import types
    def mixed_model():
        model = copy.deepcopy(tiny).bfloat16().eval()
        norms = [model.model.norm]
        norms += [n for layer in model.model.layers
                  for n in (layer.input_layernorm, layer.post_attention_layernorm)]
        for norm in norms:
            norm.weight.data = norm.weight.data.float()
            norm.forward = types.MethodType(
                lambda self, x: type(self).forward(self, x).to(x.dtype), norm)
        return model
    model, oracle_model = mixed_model(), mixed_model()
    rules = rules_for(model); ids = torch.tensor([[1, 8, 4]])
    oracle_rules = rules_for(oracle_model); oracle_sites = oracle_handles(oracle_model, oracle_rules)
    with torch.no_grad(): oracle_logits = oracle_model(ids).logits
    for handle in oracle_sites: handle.remove()
    iv = Interventions(); iv._rules = rules; iv.set_mode("readthrough"); iv.attach(lens(model))
    try:
        with torch.no_grad(): got = model(ids).logits
        assert got.dtype == torch.bfloat16 and torch.isfinite(got).all()
        torch.testing.assert_close(got, oracle_logits, rtol=4e-3, atol=4e-3)
    finally:
        iv.detach()
    assert iv._handles == [] and all(len(module._forward_hooks) == 0 for module in model.modules())


def test_exact_writer_hook_uses_runtime_output_dtype():
    jl = dense_lens(); direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": 0.8,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    writer = jl.layers[1].self_attn.o_proj
    original_forward = writer.forward
    writer.forward = lambda x: original_forward(x.float()).to(torch.bfloat16)
    iv = Interventions(); iv._rules = rules; iv.set_mode("exact"); iv.attach(jl)
    try:
        x = torch.randn(2, 8)
        raw = original_forward(x).to(torch.bfloat16)
        U, V = rebase.cumulative(rules, 1.0, 2)[1]
        U_inv, Vw, _ = rebase.inverse_uv(U, V)
        expected = raw - (raw @ Vw.to(raw)) @ U_inv.to(raw).T
        got = writer(x)
        assert got.dtype == torch.bfloat16
        torch.testing.assert_close(got, expected, rtol=4e-3, atol=4e-3)
    finally:
        iv.detach()
    assert iv._handles == [] and len(writer._forward_hooks) == 0


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


def test_packed_export_rejections_are_early_and_clean(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path); jl = lens(tiny); rules = rules_for(tiny)
    def forbidden(): raise AssertionError("full state traversal occurred")
    monkeypatch.setattr(jl._hf_model, "state_dict", forbidden)
    for fmt, phrase in (("layers", "bounded-memory"), ("lora", "LoRA export")):
        with pytest.raises(ValueError, match=phrase):
            editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt=fmt, name=fmt)
        assert not (tmp_path / fmt).exists()
    with pytest.raises(ValueError, match="aggregate-MoE"):
        editing.export_rebase(rules, jl, {"dtype": "fp16"}, fmt="full", name="exact", exact=True)
    assert not (tmp_path / "exact").exists()


def _rewrite_prefix(source: Path):
    """Exercise the official disk prefix while retaining a locally reloadable alias."""
    for shard in source.glob("*.safetensors"):
        tensors = load_file(str(shard)); rewritten = {}
        for key, value in tensors.items():
            rewritten[("model.language_model." + key.removeprefix("model."))
                      if key.startswith("model.") else key] = value
        save_file(rewritten, str(shard))
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index["weight_map"] = {
            (("model.language_model." + key.removeprefix("model."))
             if key.startswith("model.") else key): value
            for key, value in index["weight_map"].items()
        }
        index_path.write_text(json.dumps(index))


def make_source(model, path, *, dtype=None, two_shards=False):
    model.save_pretrained(path, safe_serialization=True)
    for old_shard in path.glob("*.safetensors"): old_shard.unlink()
    state = {k: v.detach().cpu().to(dtype or v.dtype) for k, v in model.state_dict().items()}
    if two_shards:
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        groups = [{}, {}]; weight_map = {}
        for i, (key, value) in enumerate(state.items()):
            groups[i % 2][key] = value; weight_map[key] = names[i % 2]
        for name, group in zip(names, groups): save_file(group, str(path / name))
        total = sum(value.numel() * value.element_size() for value in state.values())
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total}, "weight_map": weight_map}))
    else:
        save_file(state, str(path / "model.safetensors"))
    _rewrite_prefix(path)


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
    assert not list(final.parent.glob(".hf.tmp-*"))
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
    assert not (parent / "hf").exists() and not list(parent.glob(".hf.tmp-*"))


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


@pytest.mark.parametrize("base", ["COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"])
@pytest.mark.parametrize("style", ["bare", "mixed", "extension", "nested"])
def test_superscript_reserved_names_fail_core_and_api(base, style, tmp_path):
    value = {"bare": base, "mixed": base.lower(), "extension": base + ".txt",
             "nested": "folder/" + base}[style]
    with pytest.raises(ValueError, match="reserved"):
        editing._safe_relative_parts(value)
    import api.app as app
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
    with pytest.raises(app.HTTPException) as exc:
        asyncio.run(app.api_edit_export_gguf(SimpleNamespace(name=value, gguf_type="bf16")))
    assert exc.value.status_code == 422 and "reserved" in exc.value.detail


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


def _gguf_test_tools(tmp_path, *, converter_fails=False):
    convert = tmp_path / "convert.py"
    if converter_fails:
        convert.write_text("import sys\nsys.stderr.write('converter exploded')\nsys.exit(3)\n")
    else:
        convert.write_text(
            "import pathlib, sys\n"
            "out = pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
            "out.parent.mkdir(parents=True, exist_ok=True)\n"
            "out.write_bytes(b'base-gguf')\n"
        )
    quantize = tmp_path / "llama-quantize"
    quantize.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-quant')\n"
    )
    quantize.chmod(0o755)
    return convert, quantize


def _deferred_threads(monkeypatch, app):
    pending = []
    class DeferredThread:
        def __init__(self, *, target, args, daemon):
            self.target, self.args, self.daemon = target, args, daemon
            pending.append(self)
        def start(self):
            pass
        def run(self):
            self.target(*self.args)
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=DeferredThread))
    return pending


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


def test_gguf_nested_worker_converter_error_preserves_cache_and_allows_retry(tmp_path, monkeypatch):
    import api.app as app
    from fastapi.testclient import TestClient
    edits = tmp_path / "edits"; monkeypatch.setattr(app.editing, "EDITS_DIR", edits)
    job_dir = edits / "job" / "hf"; hf_dir = job_dir / "hf"
    hf_dir.mkdir(parents=True); sentinel = hf_dir / "config.json"; sentinel.write_text("{}")
    convert, quantize = _gguf_test_tools(tmp_path, converter_fails=True)
    monkeypatch.setattr(app, "_llamacpp_paths", lambda: (convert, quantize, None))
    pending = _deferred_threads(monkeypatch, app)
    app._gguf_state.update(state="idle", name=None, step=None, error=None, result=None)
    client = TestClient(app.app)
    response = client.post("/api/edit/export-gguf", json={"name": "job/hf", "gguf_type": "bf16"})
    assert response.status_code == 200 and response.json()["state"]["state"] == "running"
    pending.pop(0).run()
    assert app._gguf_state["state"] == "error"
    assert "converter exploded" in app._gguf_state["error"]
    assert sentinel.is_file() and not (job_dir / "job").exists()
    convert, _ = _gguf_test_tools(tmp_path)
    response = client.post("/api/edit/export-gguf", json={"name": "job/hf", "gguf_type": "bf16"})
    assert response.status_code == 200 and response.json()["checkpoint"] == "reused"
    pending.pop(0).run()
    assert app._gguf_state["state"] == "done"
    assert (job_dir / "hf-bf16.gguf").is_file()


def test_superscript_reserved_index_shard_is_rejected(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source, two_shards=True)
    index_path = source / "model.safetensors.index.json"; index = json.loads(index_path.read_text())
    index["weight_map"][next(iter(index["weight_map"]))] = "COM¹.safetensors"
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    with pytest.raises(ValueError, match="reserved"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="bad-index", source_dir=source)


def make_hardlinked_indexed_source(model, path):
    model.save_pretrained(path, safe_serialization=True)
    for shard in path.glob("*.safetensors"): shard.unlink()
    disk_state = {("model.language_model." + k.removeprefix("model."))
                  if k.startswith("model.") else k: v.detach().cpu()
                  for k, v in model.state_dict().items()}
    first = path / "model-00001-of-00002.safetensors"
    second = path / "model-00002-of-00002.safetensors"
    save_file(disk_state, str(first)); __import__("os").link(first, second)
    names = [first.name, second.name]
    weight_map = {key: names[i % 2] for i, key in enumerate(disk_state)}
    total = sum(value.numel() * value.element_size() for value in disk_state.values())
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total}, "weight_map": weight_map}))
    return names


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
    assert not list(editing.EDITS_DIR.glob(".malformed.tmp-*"))


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
    assert not list(edits.glob(".failed.tmp-*"))


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


def dense_lens():
    class DenseHF(nn.Module):
        def __init__(self):
            super().__init__(); self.model = nn.Module()
            self.model.layers = nn.ModuleList([block(sparse=False), block(sparse=False)])
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            self.model.embed_tokens = nn.Embedding(17, 8); self.model.norm = LlamaRMSNorm(8)
            self.lm_head = nn.Linear(8, 17, False)
    model = DenseHF()
    return SimpleNamespace(layers=model.model.layers, _final_norm=model.model.norm,
        _lm_head=model.lm_head, _embed_tokens=model.model.embed_tokens, _hf_model=model,
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"))


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


def test_transaction_rolls_back_index_write_failure(tmp_path, monkeypatch):
    jl = dense_lens(); jl._lm_head.weight = jl._embed_tokens.weight
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
    assert not list(editing.EDITS_DIR.glob(".index-failure.tmp-*"))
