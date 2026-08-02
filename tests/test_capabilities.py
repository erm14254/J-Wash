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


def test_superscript_reserved_index_shard_is_rejected(tiny, tmp_path, monkeypatch):
    source = tmp_path / "source"; make_source(tiny, source, two_shards=True)
    index_path = source / "model.safetensors.index.json"; index = json.loads(index_path.read_text())
    index["weight_map"][next(iter(index["weight_map"]))] = "COM¹.safetensors"
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    with pytest.raises(ValueError, match="reserved"):
        editing.export_rebase(rules_for(tiny), lens(tiny), {"dtype": "fp32"},
                              fmt="full", name="bad-index", source_dir=source)
