import copy
import asyncio
import json
from dataclasses import replace
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
    assert cap.readthrough_supported and not cap.exact_supported and "packed" in cap.exact_reason
    for linear in (False, True):
        dense = block(linear, sparse=False)
        assert rebase.block_capabilities(dense).exact_supported
        writer = dense.linear_attn.out_proj if linear else dense.self_attn.o_proj
        writer.bias = nn.Parameter(torch.zeros(8))
        cap = rebase.block_capabilities(dense)
        assert not cap.readthrough_supported
        writer.bias = None; writer.weight = nn.Parameter(torch.zeros(7, 8))
        assert not rebase.block_capabilities(dense).exact_supported
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


def test_audited_decoder_class_and_alias_contracts_fail_closed():
    unknown = block(sparse=False)
    rebase.AUDITED_DECODER_SPECS.pop(type(unknown))
    try:
        with pytest.raises(ValueError, match="not audited"):
            rebase.model_preflight(SimpleNamespace(
                layers=[unknown], _lm_head=nn.Linear(8, 17, False),
                _final_norm=unknown.post_attention_layernorm,
                _embed_tokens=nn.Embedding(17, 8),
                layout=SimpleNamespace(lm_head="lm_head")))
    finally:
        rebase.AUDITED_DECODER_SPECS[type(unknown)] = (
            SYNTHETIC_DENSE_SPEC)

    cases = []
    jl = dense_lens(); jl.layers = nn.ModuleList([jl.layers[0], jl.layers[0]]); cases.append(jl)
    jl = dense_lens(); jl.layers[1].input_layernorm = jl.layers[0].input_layernorm; cases.append(jl)
    jl = dense_lens(); jl.layers[1].self_attn.q_proj = jl.layers[0].self_attn.q_proj; cases.append(jl)
    jl = dense_lens(); jl.layers[1].self_attn.q_proj.weight = jl.layers[0].self_attn.q_proj.weight; cases.append(jl)
    jl = dense_lens(); jl.layers[1].self_attn.o_proj = jl.layers[0].self_attn.o_proj; cases.append(jl)
    jl = dense_lens(); jl.layers[1].self_attn.o_proj.weight = jl.layers[0].self_attn.o_proj.weight; cases.append(jl)
    jl = dense_lens(); jl._final_norm = jl.layers[0].input_layernorm; cases.append(jl)
    for candidate in cases:
        with pytest.raises(ValueError, match="shared|alias"):
            rebase.model_preflight(candidate)


def test_modified_and_noncanonical_dense_topologies_fail_closed():
    base = block(sparse=False)
    subclass = type("SyntheticSubclass", (type(base),), {})()
    subclass.__dict__.update(base.__dict__)
    jl = dense_lens(); jl.layers[0] = subclass
    with pytest.raises(ValueError, match="not audited"):
        rebase.model_preflight(jl)
    jl = dense_lens()
    jl.layers[0].adapter = nn.Linear(8, 8, False)
    with pytest.raises(ValueError, match="inventory is not audited"):
        rebase.model_preflight(jl)
    jl = dense_lens()
    jl.layers[0].forward = lambda hidden: hidden
    with pytest.raises(ValueError, match="not audited"):
        rebase.model_preflight(jl)
    jl = dense_lens()
    jl.layers[0].mlp.gate_proj.weight = nn.Parameter(torch.randn(2, 3, 8))
    with pytest.raises(ValueError, match="audited Linear"):
        rebase.model_preflight(jl)


def test_final_head_contract_and_explicit_embedding_tie():
    assert rebase.model_preflight(dense_lens()).final_head[0] == "lm_head.weight"
    tied = dense_lens(); tied._lm_head.weight = tied._embed_tokens.weight
    tied._hf_model.config = SimpleNamespace(tie_word_embeddings=True)
    assert rebase.model_preflight(tied).final_head[2] is tied._embed_tokens.weight

    bad = []
    jl = dense_lens(); jl._lm_head.weight = jl.layers[0].self_attn.q_proj.weight; bad.append(jl)
    jl = dense_lens(); jl._lm_head.forward = lambda x: x; bad.append(jl)
    jl = dense_lens(); jl._lm_head = nn.Sequential(nn.Linear(8, 17, False)); bad.append(jl)
    jl = dense_lens(); jl._lm_head.bias = nn.Parameter(torch.zeros(17)); bad.append(jl)
    jl = dense_lens(); jl._lm_head.weight = nn.Parameter(torch.ones(17, 2, 4)); bad.append(jl)
    jl = dense_lens(); jl._lm_head.weight = nn.Parameter(torch.ones(17, 7)); bad.append(jl)
    jl = dense_lens(); jl._lm_head.weight = nn.Parameter(torch.ones(17, 8, dtype=torch.int8),
                                                         requires_grad=False); bad.append(jl)
    for candidate in bad:
        with pytest.raises((ValueError, AttributeError)):
            rebase.model_preflight(candidate)


def test_head_embedding_storage_aliases_and_execution_state_fail_closed():
    jl = dense_lens()
    jl._lm_head.weight = nn.Parameter(jl._embed_tokens.weight.detach())
    with pytest.raises(ValueError, match="share storage|storage"):
        rebase.model_preflight(jl)
    jl = dense_lens(); storage = torch.randn(18, 8)
    jl._embed_tokens.weight = nn.Parameter(storage[:17])
    jl._lm_head.weight = nn.Parameter(storage[1:18])
    with pytest.raises(ValueError, match="share storage|storage"):
        rebase.model_preflight(jl)
    jl = dense_lens()
    handle = jl.layers[0].self_attn.q_proj.register_forward_pre_hook(
        lambda _module, inputs: inputs)
    try:
        with pytest.raises(ValueError, match="execution hooks"):
            rebase.model_preflight(jl)
    finally:
        handle.remove()
    jl = dense_lens()
    torch.nn.utils.parametrize.register_parametrization(
        jl.layers[0].self_attn.q_proj, "weight", nn.Identity())
    with pytest.raises(ValueError, match="parameterization|child inventory|class is not audited"):
        rebase.model_preflight(jl)

def test_forged_bound_method_metadata_is_rejected():
    jl = dense_lens()
    class ForgedCallable:
        __self__ = jl.layers[0]
        __func__ = type(jl.layers[0]).forward
        def __call__(self, *args, **kwargs):
            return args[0] if args else None
    jl.layers[0].forward = ForgedCallable()
    with pytest.raises(ValueError, match="forward provenance"):
        rebase.model_preflight(jl)


def test_registered_attribute_shadows_fail_closed():
    cases = []
    jl = dense_lens(); jl.layers[0].self_attn.__dict__["q_proj"] = nn.Linear(8, 16, False); cases.append(jl)
    jl = dense_lens(); jl.layers[0].self_attn.q_proj.__dict__["weight"] = nn.Parameter(torch.randn(16, 8)); cases.append(jl)
    jl = dense_lens(); jl.layers[0].self_attn.o_proj.__dict__["weight"] = nn.Parameter(torch.randn(8, 16)); cases.append(jl)
    jl = dense_lens(); jl.layers[0].input_layernorm.__dict__["weight"] = nn.Parameter(torch.zeros(8)); cases.append(jl)
    jl = dense_lens(); jl._embed_tokens.__dict__["weight"] = nn.Parameter(torch.randn(17, 8)); cases.append(jl)
    jl = dense_lens(); jl._lm_head.__dict__["weight"] = nn.Parameter(torch.randn(17, 8)); cases.append(jl)
    for candidate in cases:
        with pytest.raises(ValueError, match="registered"):
            rebase.model_preflight(candidate)


def test_tie_declaration_and_physical_state_must_agree():
    declared_untied = dense_lens()
    declared_untied._hf_model.config = SimpleNamespace(tie_word_embeddings=True)
    with pytest.raises(ValueError, match="tie declaration"):
        rebase.model_preflight(declared_untied)
    physical_undeclared = dense_lens()
    physical_undeclared._lm_head.weight = physical_undeclared._embed_tokens.weight
    physical_undeclared._hf_model.config = SimpleNamespace(tie_word_embeddings=False)
    with pytest.raises(ValueError, match="tie declaration"):
        rebase.model_preflight(physical_undeclared)


def test_supported_mode_with_bad_scale_is_side_effect_free(monkeypatch):
    import api.app as app
    iv = Interventions(); iv.set_scale_and_mode(scale=2.0, mode="standard")
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.manager, "meta", {"model_id": "local"})
    monkeypatch.setattr(app.manager, "capability_profile", {
        "declared_quantization": None, "has_packed_read_parameters": False,
        "modes": {
            "standard": capabilities.decision(True, "supported"),
            "readthrough": capabilities.decision(True, "supported"),
            "exact": capabilities.decision(True, "supported"),
            "abliteration": capabilities.decision(False, "global_projection_unvalidated"),
        }})
    with pytest.raises(app.HTTPException):
        app.api_interventions_scale(SimpleNamespace(scale="not-a-number", mode="readthrough"))
    assert iv.state_snapshot() == (2.0, "standard")


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_packed_exact_is_model_wide_and_live_rejection_is_clean(tiny, layer):
    jl = lens(tiny); rules = rule_at(tiny, layer)
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        rebase.build_plan(rules, jl, 1.0, exact=True)
    before = [len(module._forward_hooks) for module in tiny.modules()]
    iv = Interventions(); iv._rules = rules; iv.set_mode("exact")
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        iv.attach(jl)
    assert before == [len(module._forward_hooks) for module in tiny.modules()]
    assert iv._handles == []


def test_public_plan_revalidates_and_exposes_no_inventory_injection(tiny):
    packed_jl = lens(tiny)
    with pytest.raises(TypeError):
        rebase.build_plan(rules_for(tiny), packed_jl, 1.0, inventory=rebase.model_preflight(packed_jl))
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        rebase.build_plan(rules_for(tiny), packed_jl, 1.0, exact=True)

    first = dense_lens()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": .8,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: torch.nn.functional.normalize(torch.randn(8), dim=0)},
              "dirs_b": None}]
    assert rebase.build_plan(rules, first, 1.0)[0]
    first.layers[0] = block(sparse=False)
    assert rebase.build_plan(rules, first, 1.0)[0]


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_packed_exact_denies_before_norm_materialization(tiny, layer, monkeypatch):
    jl, rules = lens(tiny), rule_at(tiny, layer)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("packed exact reached semantic materialization")
    monkeypatch.setattr(rebase, "validate_rms_norm", forbidden)
    monkeypatch.setattr(rebase, "materialize_parameter_for_inspection", forbidden)
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        rebase.model_preflight(jl, exact=True)
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        rebase.build_plan(rules, jl, 1.0, exact=True)
    iv = Interventions(); iv._rules = rules; iv.set_mode("exact")
    with pytest.raises(ValueError, match="exact mode is unavailable for packed"):
        iv.attach(jl)
    assert iv._handles == []


def test_qwen_nested_storage_and_hooks_fail_closed(tiny):
    model = copy.deepcopy(tiny); jl = lens(model)
    linear = model.model.layers[0].linear_attn
    linear.dt_bias = nn.Parameter(linear.dt_bias.reshape(1, -1))
    with pytest.raises(ValueError, match="raw parameters"):
        rebase.model_preflight(jl)

    model = copy.deepcopy(tiny); jl = lens(model)
    experts = model.model.layers[0].mlp.experts
    experts.down_proj = nn.Parameter(experts.down_proj[..., :-1])
    with pytest.raises(ValueError, match="packed expert storage"):
        rebase.model_preflight(jl)

    model = copy.deepcopy(tiny); jl = lens(model)
    handle = model.model.layers[0].mlp.register_forward_hook(
        lambda _module, _inputs, output: output)
    try:
        with pytest.raises(ValueError, match="execution hooks"):
            rebase.model_preflight(jl)
    finally:
        handle.remove()


def test_api_and_legacy_capability_fields(tiny):
    meta = _rebase_capability_meta(lens(tiny))
    assert meta["readthrough_supported"] and meta["rebase_supported"]
    assert not meta["exact_supported"]
    assert meta["exact_reason"] == capabilities.PUBLIC_REASONS["packed_moe_exact_unsupported"]
    bad = lens(tiny); bad.layers[0].pre_feedforward_layernorm = RMS(32)
    try:
        rejected = _rebase_capability_meta(bad)
        assert not rejected["readthrough_supported"] and not rejected["rebase_supported"]
        assert rejected["readthrough_reason"] == capabilities.PUBLIC_REASONS["architecture_unsupported"]
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
        assert not cap.readthrough_supported and "class is not audited" in cap.readthrough_reason
    bad = lens(tiny); bad._final_norm = nn.LayerNorm(32, bias=False)
    with pytest.raises(ValueError, match="class is not audited"):
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
def test_production_width_qwen35_moe_block_capability(hidden, monkeypatch):
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm
    b = nn.Module(); b.self_attn = nn.Module()
    b.self_attn.q_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.self_attn.k_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.self_attn.v_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.self_attn.o_proj = nn.Linear(8, hidden, False, dtype=torch.bfloat16)
    b.mlp = nn.Module(); b.mlp.gate = nn.Linear(hidden, 4, False, dtype=torch.bfloat16)
    b.mlp.experts = nn.Module(); b.mlp.experts.gate_up_proj = nn.Parameter(torch.randn(2, 8, hidden, dtype=torch.bfloat16))
    b.mlp.experts.down_proj = nn.Parameter(torch.randn(2, hidden, 4,
                                                       dtype=torch.bfloat16))
    b.mlp.shared_expert = nn.Module()
    b.mlp.shared_expert.gate_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.mlp.shared_expert.up_proj = nn.Linear(hidden, 8, False, dtype=torch.bfloat16)
    b.mlp.shared_expert.down_proj = nn.Linear(8, hidden, False,
                                              dtype=torch.bfloat16)
    b.mlp.shared_expert_gate = nn.Linear(hidden, 1, False, dtype=torch.bfloat16)
    b.input_layernorm = Qwen3_5MoeRMSNorm(hidden).to(torch.bfloat16)
    b.post_attention_layernorm = Qwen3_5MoeRMSNorm(hidden).to(torch.bfloat16)
    monkeypatch.setitem(rebase.AUDITED_DECODER_SPECS, type(b),
                        replace(SYNTHETIC_PACKED_SPEC,
                                decoder=rebase.class_contract(type(b)),
                                norm_class=rebase.class_contract(Qwen3_5MoeRMSNorm)))
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
        assert "not audited" in cap.exact_reason
        direction = torch.randn(8); direction /= direction.norm()
        iv = Interventions(); iv._rules = [{"id": 1, "token_id": 1, "token": "x",
            "mode": "scale", "factor": .8, "replacement_id": None, "replacement": None,
            "layers": [0], "dirs_a": {0: direction}, "dirs_b": None}]
        iv.set_mode("exact"); before = [len(module._forward_hooks) for module in jl._hf_model.modules()]
        with pytest.raises(ValueError, match="not audited"): iv.attach(jl)
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


def test_live_read_rejects_instance_overridden_norm(tiny):
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
    model = mixed_model(); rules = rules_for(model)
    iv = Interventions(); iv._rules = rules; iv.set_mode("readthrough")
    with pytest.raises(ValueError, match="forward provenance"):
        iv.attach(lens(model))
    assert iv._handles == [] and all(len(module._forward_hooks) == 0 for module in model.modules())


def test_exact_writer_forward_override_is_rejected():
    jl = dense_lens(); direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale", "factor": 0.8,
              "replacement_id": None, "replacement": None, "layers": [0],
              "dirs_a": {0: direction}, "dirs_b": None}]
    writer = jl.layers[1].self_attn.o_proj
    original_forward = writer.forward
    writer.forward = lambda x: original_forward(x.float()).to(torch.bfloat16)
    iv = Interventions(); iv._rules = rules; iv.set_mode("exact")
    with pytest.raises(ValueError, match="forward provenance"):
        iv.attach(jl)
    assert iv._handles == [] and len(writer._forward_hooks) == 0


@pytest.mark.parametrize("exact", [False, True])
def test_dense_live_hooks_match_baked_reader_and_writer_transforms(exact):
    jl = dense_lens()
    direction = torch.randn(8); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale",
              "factor": 0.8, "replacement_id": None, "replacement": None,
              "layers": [0], "dirs_a": {0: direction}, "dirs_b": None}]
    transforms, _ = rebase.build_plan(rules, jl, 1.0, exact=exact)
    reader = jl.layers[1].self_attn.q_proj
    reader_weight = reader.weight.detach().clone()
    reader_baked = rebase.apply_transform(
        transforms["model.layers.1.self_attn.q_proj.weight"], reader_weight
    )[0]
    writer = jl.layers[1].self_attn.o_proj
    writer_weight = writer.weight.detach().clone()
    writer_baked = (rebase.apply_transform(
        transforms["model.layers.1.self_attn.o_proj.weight"], writer_weight
    )[0] if exact else None)
    iv = Interventions(); iv._rules = rules
    iv.set_mode("exact" if exact else "readthrough"); iv.attach(jl)
    h = torch.randn(2, 3, 8)
    try:
        live_reader = reader(jl.layers[1].input_layernorm(h))
        live_writer = writer(h)
    finally:
        iv.detach()
    baked_reader = torch.nn.functional.linear(
        jl.layers[1].input_layernorm(h), reader_baked
    )
    torch.testing.assert_close(live_reader, baked_reader, rtol=2e-5, atol=2e-6)
    if exact:
        baked_writer = torch.nn.functional.linear(h, writer_baked)
        torch.testing.assert_close(live_writer, baked_writer, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("exact", [False, True])
def test_real_tiny_llama_live_logits_match_baked_plan(exact):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=32)
    torch.manual_seed(17)
    model = LlamaForCausalLM(config).float().eval()
    baked = copy.deepcopy(model).eval()
    jl = lens(model)
    direction = torch.randn(16); direction /= direction.norm()
    rules = [{"id": 1, "token_id": 1, "token": "x", "mode": "scale",
              "factor": 0.9, "replacement_id": None, "replacement": None,
              "layers": [0], "dirs_a": {0: direction}, "dirs_b": None}]
    transforms, _ = rebase.build_plan(rules, jl, 1.0, exact=exact)
    state = baked.state_dict()
    for key, transform in transforms.items():
        state[key] = rebase.apply_transform(transform, state[key].float())[0]
    baked.load_state_dict(state)
    iv = Interventions(); iv._rules = rules
    iv.set_mode("exact" if exact else "readthrough")
    ids = torch.tensor([[1, 7, 4, 9]])
    iv.attach(jl)
    try:
        with torch.no_grad():
            live_logits = model(ids).logits
    finally:
        iv.detach()
    with torch.no_grad():
        baked_logits = baked(ids).logits
    torch.testing.assert_close(live_logits, baked_logits, rtol=3e-3, atol=3e-4)


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


def test_qwen_full_attention_head_arithmetic_fail_closed(tiny):
    assert rebase.model_preflight(lens(tiny)).packed

    mutations = []
    model = copy.deepcopy(tiny); model.config.num_attention_heads = 3; mutations.append((model, "grouping"))
    model = copy.deepcopy(tiny); layer = model.model.layers[1]
    layer.self_attn.q_proj.weight = nn.Parameter(layer.self_attn.q_proj.weight[:-1].clone()); mutations.append((model, "attention projection"))
    model = copy.deepcopy(tiny); layer = model.model.layers[1]
    layer.self_attn.k_proj.weight = nn.Parameter(layer.self_attn.k_proj.weight[:-1].clone()); mutations.append((model, "attention projection"))
    model = copy.deepcopy(tiny); model.config.head_dim = 0; mutations.append((model, "head_dim"))
    model = copy.deepcopy(tiny); layer = model.model.layers[1]
    layer.self_attn.q_norm.weight = nn.Parameter(torch.ones(layer.self_attn.q_norm.weight.numel() + 1)); mutations.append((model, "q_norm"))
    for model, phrase in mutations:
        with pytest.raises(ValueError, match=phrase):
            rebase.model_preflight(lens(model))
        assert not capabilities.build_profile(lens(model), None)["modes"]["readthrough"]["supported"]


def test_qwen_linear_attention_arithmetic_fail_closed(tiny):
    assert rebase.model_preflight(lens(tiny)).packed
    cases = []
    model = copy.deepcopy(tiny); model.config.linear_num_key_heads = 0; cases.append((model, "key-head"))
    model = copy.deepcopy(tiny); model.config.linear_num_value_heads = 0; cases.append((model, "value-head"))
    model = copy.deepcopy(tiny); model.config.linear_num_key_heads = 3; cases.append((model, "grouping"))
    model = copy.deepcopy(tiny); la = model.model.layers[0].linear_attn
    la.in_proj_qkv.weight = nn.Parameter(la.in_proj_qkv.weight[:-1].clone()); cases.append((model, "raw parameters"))
    model = copy.deepcopy(tiny); la = model.model.layers[0].linear_attn
    la.in_proj_z.weight = nn.Parameter(la.in_proj_z.weight[:-1].clone()); cases.append((model, "raw parameters"))
    model = copy.deepcopy(tiny); la = model.model.layers[0].linear_attn
    la.in_proj_b.weight = nn.Parameter(la.in_proj_b.weight.repeat(2, 1)); cases.append((model, "raw parameters"))
    model = copy.deepcopy(tiny); la = model.model.layers[0].linear_attn
    la.dt_bias = nn.Parameter(torch.ones(la.dt_bias.numel() + 1)); cases.append((model, "raw parameters"))
    model = copy.deepcopy(tiny); la = model.model.layers[0].linear_attn
    la.conv1d.groups = max(1, la.conv1d.groups - 1); cases.append((model, "raw parameters"))
    model = copy.deepcopy(tiny); la = model.model.layers[0].linear_attn
    la.out_proj.weight = nn.Parameter(la.out_proj.weight[:, :-1].clone()); cases.append((model, "raw parameters"))
    for model, phrase in cases:
        with pytest.raises(ValueError, match=phrase):
            rebase.model_preflight(lens(model))


def test_qwen_router_expert_count_and_topk_fail_closed(tiny):
    cases = []
    model = copy.deepcopy(tiny); model.config.num_experts = 0; cases.append((model, "expert count"))
    model = copy.deepcopy(tiny); model.config.num_experts_per_tok = 0; cases.append((model, "top-k"))
    model = copy.deepcopy(tiny); model.config.num_experts_per_tok = -1; cases.append((model, "top-k"))
    model = copy.deepcopy(tiny); model.config.num_experts_per_tok = model.config.num_experts + 1; cases.append((model, "top-k"))
    model = copy.deepcopy(tiny); mlp = model.model.layers[0].mlp
    mlp.gate.weight = nn.Parameter(mlp.gate.weight[:-1].clone()); cases.append((model, "expert storage"))
    model = copy.deepcopy(tiny); experts = model.model.layers[0].mlp.experts
    experts.gate_up_proj = nn.Parameter(experts.gate_up_proj[:-1].clone()); cases.append((model, "expert storage"))
    model = copy.deepcopy(tiny); experts = model.model.layers[0].mlp.experts
    experts.down_proj = nn.Parameter(experts.down_proj[:-1].clone()); cases.append((model, "expert storage"))
    for model, phrase in cases:
        with pytest.raises(ValueError, match=phrase):
            rebase.model_preflight(lens(model))

    lower = copy.deepcopy(tiny); lower.config.num_experts_per_tok = 1
    assert rebase.model_preflight(lens(lower)).packed
    upper = copy.deepcopy(tiny); upper.config.num_experts_per_tok = upper.config.num_experts
    assert rebase.model_preflight(lens(upper)).packed


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_qwen_gated_norm_nonfinite_gain_fails_closed(tiny, value):
    model = copy.deepcopy(tiny)
    norm = model.model.layers[0].linear_attn.norm
    norm.weight.data[0] = value
    with pytest.raises(ValueError, match="nonfinite"):
        rebase.model_preflight(lens(model))
    assert not capabilities.build_profile(lens(model), None)["modes"]["readthrough"]["supported"]


def test_final_norm_storage_overlap_is_authoritative():
    jl = dense_lens()
    source = jl.layers[0].self_attn.q_proj.weight
    jl._final_norm.weight = nn.Parameter(source.reshape(-1)[:8])
    with pytest.raises(ValueError, match="final norm storage aliases"):
        rebase.model_preflight(jl)

    jl = dense_lens()
    storage = torch.randn(18, 8)
    jl._embed_tokens.weight = nn.Parameter(storage[:17])
    jl._final_norm.weight = nn.Parameter(storage.reshape(-1)[:8])
    with pytest.raises(ValueError, match="embedding or head storage aliases|final norm storage"):
        rebase.model_preflight(jl)

    jl = dense_lens()
    storage = torch.randn(18, 8)
    jl._lm_head.weight = nn.Parameter(storage[:17])
    jl._final_norm.weight = nn.Parameter(storage.reshape(-1)[:8])
    with pytest.raises(ValueError, match="embedding or head storage aliases|final norm storage"):
        rebase.model_preflight(jl)
