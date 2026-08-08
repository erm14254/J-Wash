import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

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
    if dtype == torch.float32:
        # Chunking changes the GEMM batch shape, so float32 accumulation need
        # not be bitwise identical to the single unbounded matrix multiply.
        # Keep this tight: only ordinary float32 roundoff is accepted.
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
    else:
        assert torch.equal(got, expected)
    assert max(observed) <= budget and sum(observed) == 15
    expected_delta = (rebase.apply_read(source.float(), u, v)[0] - source.float()).abs().max()
    assert delta == pytest.approx(float(expected_delta), rel=1e-6, abs=1e-6)


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


@pytest.fixture(scope="module")
def tiny():
    """A genuine mixed full/linear-attention Qwen3.5 MoE model."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    config = Qwen3_5MoeTextConfig(
        vocab_size=67, hidden_size=32, num_hidden_layers=3, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        linear_conv_kernel_dim=2, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4, moe_intermediate_size=16,
        shared_expert_intermediate_size=12, num_experts=4, num_experts_per_tok=2,
        max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 1, 1],
                         "mrope_interleaved": True},
    )
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    torch.manual_seed(7)
    return Qwen3_5MoeForCausalLM(config).float().eval()


def lens(model):
    return SimpleNamespace(
        layers=model.model.layers, _final_norm=model.model.norm,
        _lm_head=model.lm_head, _embed_tokens=model.model.embed_tokens,
        _hf_model=model,
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"),
    )


def rules_for(model, mode="replace", factor=1.0):
    weight = model.lm_head.weight.detach().float()

    def unit(index):
        row = weight[index]
        return row / row.norm().clamp_min(1e-8)

    return [{
        "id": 1, "token_id": 3, "token": "<3>", "mode": mode, "factor": factor,
        "replacement_id": 5 if mode == "replace" else None,
        "replacement": "<5>" if mode == "replace" else None,
        "layers": [0], "dirs_a": {0: unit(3)},
        "dirs_b": {0: unit(5)} if mode == "replace" else None,
    }]


def bake(model, rules, scale=1.0):
    clone = copy.deepcopy(model)
    transforms, _ = rebase.build_plan(rules, lens(model), scale)
    state = clone.state_dict()
    for key, transform in transforms.items():
        state[key] = rebase.apply_transform(transform, state[key].float())[0].to(state[key])
    clone.load_state_dict(state)
    return clone.eval()


def oracle_handles(model, rules, scale=1.0):
    """Independent oracle using literal norm sites, not production discovery."""
    cums = rebase.cumulative(rules, scale, len(model.model.layers))
    handles = []

    def attach(norm, u, v):
        ug, vg = rebase.gamma_pair(norm, u, v)

        def hook(_module, _inputs, output):
            return output + (output @ vg.to(output)) @ ug.to(output).T

        handles.append(norm.register_forward_hook(hook))

    for index in (1, 2):
        attach(model.model.layers[index].input_layernorm, *cums[index])
        attach(model.model.layers[index].post_attention_layernorm, *cums[index])
    attach(model.model.norm, *cums[3])
    return handles


def capture_moe(model):
    captured, handles = {}, []
    for index in (1, 2):
        mlp = model.model.layers[index].mlp

        def gate_hook(_module, inputs, output, index=index, mlp=mlp):
            logits, weights, expert_ids = output
            value = inputs[0].reshape(-1, inputs[0].shape[-1])
            routed = [torch.nn.functional.linear(value, packed).chunk(2, -1)
                      for packed in mlp.experts.gate_up_proj]
            captured[f"{index}.router"] = logits.detach()
            captured[f"{index}.weights"] = weights.detach()
            captured[f"{index}.ids"] = expert_ids.detach()
            captured[f"{index}.routed_gate"] = torch.stack([pair[0] for pair in routed], 1).detach()
            captured[f"{index}.routed_up"] = torch.stack([pair[1] for pair in routed], 1).detach()

        handles.append(mlp.gate.register_forward_hook(gate_hook))
        handles.append(mlp.shared_expert_gate.register_forward_hook(
            lambda _module, _inputs, output, index=index:
            captured.__setitem__(f"{index}.shared_gate", output.sigmoid().detach())
        ))
        handles.append(mlp.register_forward_hook(
            lambda _module, _inputs, output, index=index:
            captured.__setitem__(f"{index}.aggregate", output.detach())
        ))
    return captured, handles


def run_variant(model, input_ids, rules=None, kind="base", past=None):
    captures, capture_handles = capture_moe(model)
    oracle = []
    interventions = None
    if rules and kind == "oracle":
        oracle = oracle_handles(model, rules)
    if rules and kind == "production":
        interventions = Interventions()
        interventions._rules = rules
        interventions.set_mode("readthrough")
        interventions.attach(lens(model))
    try:
        with torch.no_grad():
            output = model(input_ids, past_key_values=past, use_cache=True)
        captures["logits"] = output.logits.detach()
        return captures, output.past_key_values
    finally:
        for handle in oracle + capture_handles:
            handle.remove()
        if interventions:
            interventions.detach()


@pytest.mark.parametrize("mode,factor", [("scale", 0.65), ("replace", 1.0)])
def test_tiny_semantic_prefill_and_cached_decode(tiny, mode, factor):
    rules = rules_for(tiny, mode, factor)
    input_ids = torch.tensor([[1, 8, 11, 4]])
    base, _ = run_variant(tiny, input_ids)
    oracle, oracle_past = run_variant(tiny, input_ids, rules, "oracle")
    production, production_past = run_variant(tiny, input_ids, rules, "production")
    baked_model = bake(tiny, rules)
    baked, baked_past = run_variant(baked_model, input_ids)
    assert set(oracle) == set(production) == set(baked)
    for key in oracle:
        if key.endswith("ids"):
            assert torch.equal(oracle[key], production[key])
            assert torch.equal(oracle[key], baked[key])
        else:
            torch.testing.assert_close(oracle[key], production[key], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(oracle[key], baked[key], rtol=2e-5, atol=2e-6)
    assert not torch.allclose(base["logits"], oracle["logits"])

    next_id = torch.tensor([[9]])
    before = oracle_past.get_seq_length()
    oracle_decode, oracle_next = run_variant(tiny, next_id, rules, "oracle", oracle_past)
    production_decode, production_next = run_variant(
        tiny, next_id, rules, "production", production_past
    )
    baked_decode, baked_next = run_variant(baked_model, next_id, past=baked_past)
    assert oracle_next.get_seq_length() == before + 1
    assert production_next.get_seq_length() == before + 1
    assert baked_next.get_seq_length() == before + 1
    torch.testing.assert_close(
        oracle_decode["logits"], production_decode["logits"], rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        oracle_decode["logits"], baked_decode["logits"], rtol=2e-5, atol=2e-6
    )


def _hook_counts(model):
    return {id(module): len(module._forward_hooks) for module in model.modules()}


def test_live_readthrough_hooks_exact_rejection_and_noop_precedence(tiny):
    jl = lens(tiny)
    rules = rules_for(tiny, "scale", 0.65)
    interventions = Interventions()
    interventions._rules = rules
    interventions.set_mode("readthrough")
    interventions.attach(jl)
    assert len(interventions._handles) == 5
    for index in (1, 2):
        assert len(jl.layers[index].input_layernorm._forward_hooks) == 1
        assert len(jl.layers[index].post_attention_layernorm._forward_hooks) == 1
    assert len(jl._final_norm._forward_hooks) == 1
    interventions.detach()
    assert interventions._handles == []
    assert all(count == 0 for count in _hook_counts(tiny).values())

    before = _hook_counts(tiny)
    interventions.set_mode("exact")
    with pytest.raises(ValueError, match="^Exact is not implemented for packed MoE; use Readthrough$"):
        interventions.attach(jl)
    assert _hook_counts(tiny) == before
    assert interventions._handles == []

    empty = Interventions()
    empty.set_mode("exact")
    empty.attach(jl)
    assert empty._handles == [] and _hook_counts(tiny) == before
    neutral = Interventions()
    neutral._rules = rules_for(tiny, "scale", 1.0)
    neutral.set_mode("exact")
    neutral.attach(jl)
    assert neutral._handles == [] and _hook_counts(tiny) == before


def _rewrite_official_prefix(source):
    for shard in source.glob("*.safetensors"):
        rewritten = {}
        for key, value in load_file(str(shard)).items():
            disk_key = "model.language_model." + key.removeprefix("model.") \
                if key.startswith("model.") else key
            rewritten[disk_key] = value
        save_file(rewritten, str(shard))
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index["weight_map"] = {
            ("model.language_model." + key.removeprefix("model.")
             if key.startswith("model.") else key): shard
            for key, shard in index["weight_map"].items()
        }
        index_path.write_text(json.dumps(index))


def make_source(model, path, *, dtype=None):
    model.save_pretrained(path, safe_serialization=True)
    for shard in path.glob("*.safetensors"):
        shard.unlink()
    state = {key: value.detach().cpu().to(dtype or value.dtype)
             for key, value in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"))
    _rewrite_official_prefix(path)


def _add_mtp_sentinel(source):
    shard = next(source.glob("*.safetensors"))
    state = load_file(str(shard))
    sentinel = torch.arange(12, dtype=torch.int16).reshape(3, 4)
    state["mtp.sentinel"] = sentinel
    save_file(state, str(shard))
    return shard, sentinel


def test_full_export_reload_raw_packed_key_and_mtp(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(tiny, source)
    shard, sentinel = _add_mtp_sentinel(source)
    source_state = load_file(str(shard))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    rules = rules_for(tiny)
    result = editing.export_rebase(
        rules, lens(tiny), {"dtype": "fp32", "model_id": "local"},
        fmt="full", name="full", source_dir=source,
    )
    output = Path(result["out_dir"])
    output_state = load_file(str(output / shard.name))
    packed_key = "model.language_model.layers.1.mlp.experts.gate_up_proj"
    assert packed_key in output_state
    assert packed_key + ".weight" not in output_state
    assert output_state[packed_key].ndim == 3
    assert output_state[packed_key].shape == source_state[packed_key].shape
    assert output_state[packed_key].dtype == source_state[packed_key].dtype
    assert not torch.equal(output_state[packed_key], source_state[packed_key])
    assert torch.equal(output_state["mtp.sentinel"], sentinel)
    embed_key = "model.language_model.embed_tokens.weight"
    assert torch.equal(output_state[embed_key], source_state[embed_key])

    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(
        output, local_files_only=True
    ).float().eval()
    baked_model = bake(tiny, rules)
    input_ids = torch.tensor([[1, 8, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(input_ids).logits, baked_model(input_ids).logits,
            rtol=2e-5, atol=2e-6,
        )


def test_full_export_uses_bounded_path_only_for_rank3(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(tiny, source)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    monkeypatch.setattr(editing, "REBASE_EXPORT_ROW_BUDGET", 3)
    bounded_shapes, chunk_sizes, vanilla_dims = [], [], []
    original_bounded = editing.apply_read_transform_bounded
    original_vanilla = rebase.apply_transform

    def observed_bounded(entry, tensor, *, row_budget):
        bounded_shapes.append(tensor.shape)
        return original_bounded(entry, tensor, row_budget=row_budget, observer=chunk_sizes.append)

    def observed_vanilla(entry, tensor):
        vanilla_dims.append(tensor.ndim)
        return original_vanilla(entry, tensor)

    monkeypatch.setattr(editing, "apply_read_transform_bounded", observed_bounded)
    monkeypatch.setattr(rebase, "apply_transform", observed_vanilla)
    editing.export_rebase(
        rules_for(tiny), lens(tiny), {"dtype": "fp32"}, fmt="full",
        name="bounded", source_dir=source,
    )
    assert bounded_shapes and all(len(shape) == 3 for shape in bounded_shapes)
    assert chunk_sizes and max(chunk_sizes) <= editing.REBASE_EXPORT_ROW_BUDGET
    assert vanilla_dims and set(vanilla_dims) == {2}


def test_packed_format_scope_is_early(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"
    make_source(tiny, source)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    jl, rules = lens(tiny), rules_for(tiny)
    allowed = editing.export_rebase(
        rules, jl, {"dtype": "fp32"}, fmt="full", name="allowed", source_dir=source
    )
    assert Path(allowed["out_dir"]).is_dir()
    with pytest.raises(ValueError, match="^Exact is not implemented for packed MoE; use Readthrough$"):
        editing.export_rebase(rules, jl, {"dtype": "fp32"}, fmt="full",
                              name="exact", source_dir=source, exact=True)

    def forbidden_state_dict():
        raise AssertionError("serializer traversed model state")

    monkeypatch.setattr(jl._hf_model, "state_dict", forbidden_state_dict)
    with pytest.raises(ValueError, match="^LoRA is not implemented for packed MoE"):
        editing.export_rebase(rules, jl, {"dtype": "fp32"}, fmt="lora", name="lora")
    with pytest.raises(ValueError, match="^Layers is not implemented for packed MoE"):
        editing.export_rebase(rules, jl, {"dtype": "fp32"}, fmt="layers", name="layers")


def test_bf16_live_bake_export_and_reload(tiny, tmp_path, monkeypatch):
    model = copy.deepcopy(tiny).bfloat16().eval()
    rules = rules_for(model)
    input_ids = torch.tensor([[1, 8, 4]])
    oracle, _ = run_variant(model, input_ids, rules, "oracle")
    production, _ = run_variant(model, input_ids, rules, "production")
    baked_model = bake(model, rules)
    baked, _ = run_variant(baked_model, input_ids)
    assert torch.isfinite(production["logits"]).all()
    torch.testing.assert_close(oracle["logits"], production["logits"], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(oracle["logits"], baked["logits"], rtol=2e-2, atol=2e-2)

    source = tmp_path / "source"
    make_source(model, source, dtype=torch.bfloat16)
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    result = editing.export_rebase(
        rules, lens(model), {"dtype": "bf16"}, fmt="full", name="bf16", source_dir=source
    )
    output = Path(result["out_dir"])
    packed = load_file(str(next(output.glob("*.safetensors"))))[
        "model.language_model.layers.1.mlp.experts.gate_up_proj"
    ]
    assert packed.dtype == torch.bfloat16
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(output, local_files_only=True).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            reloaded(input_ids).logits, baked_model(input_ids).logits,
            rtol=2e-2, atol=2e-2,
        )


def test_dense_reader_target_keeps_vanilla_plan_and_rank2_math():
    first, second = block(), block()
    for dense in (first, second):
        del dense.mlp.experts
        del dense.mlp.gate
        del dense.mlp.shared_expert
        del dense.mlp.shared_expert_gate
        dense.mlp.gate_proj = nn.Linear(8, 12, False)
        dense.mlp.up_proj = nn.Linear(8, 12, False)
        dense.mlp.down_proj = nn.Linear(12, 8, False)
    jl = SimpleNamespace(
        layers=[first, second], _final_norm=RMS(8),
        _lm_head=nn.Linear(8, 17, False), _embed_tokens=nn.Embedding(17, 8),
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"),
    )
    direction = torch.randn(8)
    direction /= direction.norm()
    rules = [{"mode": "scale", "factor": 0.5, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    transforms, info = rebase.build_plan(rules, jl, 1.0)
    assert info["packed_moe"] is False
    gate_key = "model.layers.1.mlp.gate_proj.weight"
    assert gate_key in transforms
    original = second.mlp.gate_proj.weight.detach().float()
    updated = rebase.apply_transform(transforms[gate_key], original)[0]
    assert updated.ndim == 2 and updated.shape == original.shape
