import copy
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


def block(linear=False, hidden=8, sparse=True):
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
    result.mlp = mlp; result.input_layernorm = RMS(hidden); result.post_attention_layernorm = RMS(hidden)
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


def test_official_packed_memory_dimensions():
    sizes = {"35B": (256, 1024, 2048), "122B": (256, 2048, 3072)}
    assert {name: 2 * a * b * c for name, (a, b, c) in sizes.items()} == {"35B": 1 << 30, "122B": 3 << 30}


@pytest.fixture(scope="module")
def tiny():
    import transformers
    from packaging.version import Version
    assert Version(transformers.__version__) >= Version("5.5"), (
        "Qwen3.5-MoE integration requires transformers>=5.5; installed " + transformers.__version__)
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
    oracle_d, _ = run_variant(tiny, next_id, rules, "oracle", oracle_past)
    production_d, _ = run_variant(tiny, next_id, rules, "production", production_past)
    baked_d, _ = run_variant(baked_model, next_id, past=baked_past)
    torch.testing.assert_close(oracle_d["logits"], production_d["logits"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(oracle_d["logits"], baked_d["logits"], rtol=2e-5, atol=2e-6)


def test_tiny_supported_inventory_and_version_contract(tiny):
    assert [rebase.block_capabilities(x).readthrough_supported for x in tiny.model.layers] == [True] * 3
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(tiny.model.layers[1])} == FULL_READERS
    assert {t.state_suffix for t, _, _ in rebase.iter_reads(tiny.model.layers[2])} == LINEAR_READERS
    assert tiny.model.layers[1].mlp.experts.gate_up_proj.ndim == 3
    assert "model.layers.1.mlp.experts.gate_up_proj" in tiny.state_dict()


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


def test_full_checkpoint_export_reload_mapping_mtp_and_missing(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; tiny.save_pretrained(source, safe_serialization=True)
    # save_pretrained in newer Transformers may serialize a compatibility
    # unpacked view; the production source format under test is the raw packed
    # safetensors state used by Qwen3.5-MoE checkpoints.
    for old_shard in source.glob("*.safetensors"): old_shard.unlink()
    save_file({k: v.detach().cpu() for k, v in tiny.state_dict().items()},
              str(source / "model.safetensors"))
    _rewrite_prefix(source)
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
    # Transformers' in-memory prefix is restored only for the local reload.
    reload_dir = tmp_path / "reload"; copytree = __import__("shutil").copytree
    copytree(out, reload_dir)
    reload_shard = reload_dir / shard.name; reload_state = load_file(str(reload_shard))
    reload_state = {("model." + k.removeprefix("model.language_model."))
                    if k.startswith("model.language_model.") else k: v
                    for k, v in reload_state.items() if not k.startswith("mtp.")}
    save_file(reload_state, str(reload_shard))
    reloaded = type(tiny).from_pretrained(reload_dir, local_files_only=True).float().eval()
    baked_model = bake(tiny, rules); ids = torch.tensor([[1, 8, 4]])
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, baked_model(ids).logits, rtol=2e-5, atol=2e-6)
    broken = tmp_path / "broken"; copytree(source, broken)
    broken_shard = next(broken.glob("*.safetensors")); broken_state = load_file(str(broken_shard))
    del broken_state[packed_key]; save_file(broken_state, str(broken_shard))
    with pytest.raises(ValueError, match="absent from the source"):
        editing.export_rebase(rules, lens(tiny), {"dtype": "fp16"}, fmt="full",
                              name="broken-output", source_dir=broken)
    assert not (editing.EDITS_DIR / "broken-output").exists()
