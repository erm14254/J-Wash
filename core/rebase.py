"""Change of basis of the residual: faithful pure-weight bake of the steering.

The standard hook applies ``h ← M_l·h`` at the output of each hooked layer, with
``M_l = Π_rules (I + w·v̂ᵀ)`` (rank-1 per rule, the layer's J-space directions).
This transformed residual is then READ by everything downstream through matrices:
each sub-block reads ``W·(γ ⊙ h/rms(h))`` via its RMSNorm, and lm_head reads via
the final norm. So we realize the transform in the downstream READS instead of the
writes (the "skip" escapes no one in reading, whereas no matrix carries it in
writing — the cause of the ~1.5 % of the per-layer bake):

  read of layer m:  W ← W·Γ·C_m·Γ⁻¹      (Γ = diag(γ) of the read RMSNorm)
  lm_head:          W ← W·Γ_f·C_fin·Γ_f⁻¹
  write of layer m: W ← C_m⁻¹·W          ("exact" mode only)

where ``C_m = M_{m-1}···M_{l0}`` composes the hooks strictly upstream of m.
``C = I + U·Vᵀ`` stays low-rank end to end (one column per rule and per hooked
layer), so each matrix receives a rank-r update.

Two variants:
  - "readthrough": reads only. For saturated zaps/replaces (M idempotent), this
    equals the hook applied over a range extended to the last layer, with a slight
    bias toward MORE effect (the range's intermediate writes are projected too).
    No inversion: robust in bf16 and to GGUF quantization.
  - "exact": adds the counter-transform of the writes to reproduce a hook applied
    ONCE at the chosen point. C⁻¹ blows up near a full zap (1 + v̂ᵀw → 0):
    reserved for soft factors, regularized inverse.

Assumed approximation (the only one): the rms in the RMSNorm denominator stays
that of the untransformed residual — a per-position scalar error, second-order
when the modified component is small compared to ‖h‖. Same assumption as all of
the weight-orthogonalization literature.

The live preview mode (core/ablation) applies the SAME transform via hooks on the
RMSNorm output: the preview and the exported checkpoint differ only by rounding.
"""

from dataclasses import dataclass
import functools
import types

import torch
import transformers
from packaging.version import Version
from torch.nn.utils import parametrize
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_module
from transformers.activations import SiLUActivation
from transformers.models.llama import modeling_llama as llama_modeling
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen_moe_modeling

from core.ablation import effective_coeffs

# division by γ: channels with γ=0 are dead (never read via this norm), the clamp
# is exact there; between 0 and EPS the error is bounded and negligible
GAMMA_EPS = 1e-6

# regularization threshold of the inverse (exact mode): below it, the
# counter-transform amplifies the downstream writes (×1/σ), which makes the RMS
# error first-order and destroys bf16 precision then GGUF quantization. 0.2 bounds
# the amplification to ×5; a saturated replace (α = −1 as soon as scale ≥ 1) is
# ALWAYS in this regime → prefer readthrough.
INV_COND_EPS = 0.2

# Residual reads per sub-block: {module suffix: suffix of the read RMSNorm}.
# Covers Llama/Qwen/Mistral (self_attn+mlp) and Qwen3.5/Qwen3-Next
# (linear_attn GatedDeltaNet). conv1d/q_norm/k_norm operate AFTER these
# projections: they see the transformed residual without us touching them.
READS = {
    "self_attn.q_proj": "input_layernorm",
    "self_attn.k_proj": "input_layernorm",
    "self_attn.v_proj": "input_layernorm",
    "linear_attn.in_proj_qkv": "input_layernorm",
    "linear_attn.in_proj_z": "input_layernorm",
    "linear_attn.in_proj_b": "input_layernorm",
    "linear_attn.in_proj_a": "input_layernorm",
    "mlp.gate_proj": "input_layernorm",  # replaced if post_attention is present
    "mlp.up_proj": "input_layernorm",
}
# most archs read the MLP via post_attention_layernorm
READS_POST = {"mlp.gate_proj", "mlp.up_proj"}

# Writes into the residual (exact mode only)
WRITES = ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj")

# archs where post_attention_layernorm normalizes the attention WRITE (not the
# MLP read): the read transform would be wrong there
_UNSUPPORTED_MARKERS = ("pre_feedforward_layernorm", "post_feedforward_layernorm")


def _submodule(block, dotted):
    module = block
    for part in dotted.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


_LINEAR_FORWARD = torch.nn.Linear.forward
_EMBEDDING_FORWARD = torch.nn.Embedding.forward
_ALIGN_PRE_FORWARD = AlignDevicesHook.pre_forward
_ALIGN_POST_FORWARD = AlignDevicesHook.post_forward


def _capture_accelerate_wrapper_code():
    module = torch.nn.Linear(1, 1, bias=False)
    add_hook_to_module(module, AlignDevicesHook(execution_device="cpu"))
    code = module.forward.func.__code__
    remove_hook_from_module(module)
    return code


_ACCELERATE_WRAPPER_CODE = _capture_accelerate_wrapper_code()


@dataclass(frozen=True, slots=True)
class ClassContract:
    cls: type
    forward: object


def class_contract(cls):
    return ClassContract(cls, cls.forward)


def _bound_matches(method, module, function):
    return (type(method) is types.MethodType
            and method.__self__ is module
            and method.__func__ is function)


def _supported_accelerate_hook(hook):
    # SequentialHook is deliberately fail-closed: parameter inspection cannot
    # yet prove which member owns the offload map.
    return (type(hook) is AlignDevicesHook
            and _bound_matches(hook.pre_forward, hook, _ALIGN_PRE_FORWARD)
            and _bound_matches(hook.post_forward, hook, _ALIGN_POST_FORWARD))


def validate_forward_provenance(module, expected_class, label):
    """Require the class forward or Accelerate 1.14's exact wrapper shape."""
    if not isinstance(expected_class, ClassContract):
        raise ValueError(f"{label} has no frozen class contract")
    if type(module) is not expected_class.cls:
        raise ValueError(f"{label} class is not audited")
    expected = expected_class.forward
    if type(module).forward is not expected:
        raise ValueError(f"{label} class forward was modified")
    current = getattr(module, "forward", None)
    old = getattr(module, "_old_forward", None)
    hook = getattr(module, "_hf_hook", None)
    if old is None and hook is None:
        if not _bound_matches(current, module, expected):
            raise ValueError(f"{label} forward provenance is not audited")
        return
    if (not _supported_accelerate_hook(hook) or type(old) is not types.MethodType
            or not _bound_matches(old, module, expected)):
        raise ValueError(f"{label} Accelerate wrapper is not audited")
    if (type(current) is not functools.partial or len(current.args) != 1
            or current.args[0] is not module or current.keywords):
        raise ValueError(f"{label} Accelerate wrapper shape is not audited")
    wrapper = current.func
    if (type(wrapper) is not types.FunctionType
            or wrapper.__code__ is not _ACCELERATE_WRAPPER_CODE
            or getattr(current, "__wrapped__", None) is not old
            or wrapper.__globals__.get("AlignDevicesHook") is not AlignDevicesHook
            or wrapper.__defaults__ is not None
            or wrapper.__kwdefaults__ is not None
            or wrapper.__closure__ is not None):
        raise ValueError(f"{label} Accelerate wrapper provenance is not audited")


def _validate_execution_state(module, label):
    if (module._forward_pre_hooks or module._forward_hooks or module._backward_hooks):
        raise ValueError(f"{label} has pre-existing execution hooks")
    if parametrize.is_parametrized(module):
        raise ValueError(f"{label} uses unsupported parameterization")


def _validate_direct_inventory(module, modules, parameters, label, buffers=()):
    _validate_execution_state(module, label)
    if (set(module._modules) != set(modules)
            or set(module._parameters) != set(parameters)
            or set(module._buffers) != set(buffers)):
        raise ValueError(
            f"{label} child inventory is not audited: modules={tuple(module._modules)}, "
            f"parameters={tuple(module._parameters)}, buffers={tuple(module._buffers)}"
        )
    for name in modules:
        if getattr(module, name) is not module._modules[name]:
            raise ValueError(f"{label}.{name} is not the registered child module")
    for name in parameters:
        if getattr(module, name) is not module._parameters[name]:
            raise ValueError(f"{label}.{name} is not the registered parameter")
    for name in buffers:
        if getattr(module, name) is not module._buffers[name]:
            raise ValueError(f"{label}.{name} is not the registered buffer")


@dataclass(frozen=True)
class TransformTarget:
    """A residual-coordinate tensor, without assuming ``nn.Linear.weight``.

    ``state_suffix`` is the exact checkpoint suffix.  ``accessor`` names the
    module/parameter on the instantiated block, ``axis`` is the residual input
    axis (negative indexing), and ``lora_supported`` distinguishes ordinary
    module weights from packed functional parameters.
    """

    state_suffix: str
    accessor: str
    norm_name: str
    axis: int = -1
    activation_accessor: str | None = None
    lora_supported: bool = True

    def tensor(self, block):
        obj = _submodule(block, self.accessor)
        if obj is None:
            return None
        return obj.weight if self.state_suffix.endswith(".weight") else obj


@dataclass(frozen=True)
class RebaseCapabilities:
    readthrough_supported: bool
    readthrough_reason: str
    exact_supported: bool
    exact_reason: str


@dataclass(frozen=True, slots=True)
class TopologySpec:
    decoder: ClassContract
    mixers: tuple[tuple[str, ClassContract, tuple[str, ...], tuple[str, ...], str | None], ...]
    mlp_class: ClassContract
    mlp_modules: tuple[str, ...]
    mlp_parameters: tuple[str, ...] = ()
    packed: bool = False
    router_class: ClassContract | None = None
    experts_class: ClassContract | None = None
    shared_expert_class: ClassContract | None = None
    norm_class: ClassContract | None = None


@dataclass(frozen=True, slots=True)
class BlockInventory:
    index: int
    kind: str
    norms: tuple[tuple[str, object], ...]
    reads: tuple[tuple[TransformTarget, object, object], ...]
    writes: tuple[tuple[str, object], ...]
    packed: bool
    exact_supported: bool
    exact_reason: str



@dataclass(frozen=True, slots=True)
class ModelInventory:
    blocks: tuple[BlockInventory, ...]
    final_norm: object
    hidden: int
    packed: bool
    exact_supported: bool
    exact_reason: str
    final_head: tuple[str, object, object]
    embedding: tuple[str, object, object]
    tied: bool
    owner: object
    layer_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class StorageSpan:
    storage: int
    start: int
    end: int


def storage_span(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    if tensor.device.type == "meta":
        return None
    if not tensor.is_contiguous():
        raise ValueError("editing-relevant tensor storage is non-contiguous")
    storage = tensor.untyped_storage()
    start = tensor.storage_offset() * tensor.element_size()
    return StorageSpan(storage._cdata, start, start + tensor.numel() * tensor.element_size())


def _offloaded_registered_span(module, parameter_name, tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "meta":
        return None
    materialized = materialize_parameter_for_inspection(module, parameter_name)
    if materialized.device.type != "cpu" or not materialized.is_contiguous():
        raise ValueError("offloaded tensor storage cannot be classified safely")
    return StorageSpan(-id(tensor), 0, materialized.numel() * materialized.element_size())


def _overlaps(left, right):
    return (left is not None and right is not None and left.storage == right.storage
            and max(left.start, right.start) < min(left.end, right.end))


_LINEAR = class_contract(torch.nn.Linear)
_EMBEDDING = class_contract(torch.nn.Embedding)
_CONV1D = class_contract(torch.nn.Conv1d)
_SILU = class_contract(SiLUActivation)
_LLAMA_NORM = class_contract(llama_modeling.LlamaRMSNorm)
_QWEN_NORM = class_contract(qwen_moe_modeling.Qwen3_5MoeRMSNorm)
_QWEN_GATED_NORM = class_contract(qwen_moe_modeling.Qwen3_5MoeRMSNormGated)
AUDITED_DECODER_SPECS = {
    llama_modeling.LlamaDecoderLayer: TopologySpec(
        class_contract(llama_modeling.LlamaDecoderLayer),
        (("self_attn", class_contract(llama_modeling.LlamaAttention),
          ("q_proj", "k_proj", "v_proj", "o_proj"), (), None),),
        class_contract(llama_modeling.LlamaMLP),
        ("gate_proj", "up_proj", "down_proj", "act_fn"),
        norm_class=_LLAMA_NORM,
    ),
    qwen_moe_modeling.Qwen3_5MoeDecoderLayer: TopologySpec(
        class_contract(qwen_moe_modeling.Qwen3_5MoeDecoderLayer),
        (("self_attn", class_contract(qwen_moe_modeling.Qwen3_5MoeAttention),
          ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"),
          (), "full_attention"),
         ("linear_attn", class_contract(qwen_moe_modeling.Qwen3_5MoeGatedDeltaNet),
          ("act", "conv1d", "norm", "out_proj", "in_proj_qkv", "in_proj_z",
           "in_proj_b", "in_proj_a"), ("dt_bias", "A_log"), "linear_attention")),
        class_contract(qwen_moe_modeling.Qwen3_5MoeSparseMoeBlock),
        ("gate", "experts", "shared_expert", "shared_expert_gate"), packed=True,
        router_class=class_contract(qwen_moe_modeling.Qwen3_5MoeTopKRouter),
        experts_class=class_contract(qwen_moe_modeling.Qwen3_5MoeExperts),
        shared_expert_class=class_contract(qwen_moe_modeling.Qwen3_5MoeMLP),
        norm_class=_QWEN_NORM,
    ),
}


_MOE_READS = (
    TransformTarget("mlp.gate.weight", "mlp.gate", "post_attention_layernorm"),
    TransformTarget("mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj", "post_attention_layernorm", lora_supported=False),
    TransformTarget("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.gate_proj", "post_attention_layernorm"),
    TransformTarget("mlp.shared_expert.up_proj.weight", "mlp.shared_expert.up_proj", "post_attention_layernorm"),
    TransformTarget("mlp.shared_expert_gate.weight", "mlp.shared_expert_gate", "post_attention_layernorm"),
)


_AUDITED_RMS_NORMS = {
    ("transformers.models.llama.modeling_llama", "LlamaRMSNorm"): "ordinary",
    ("transformers.models.mistral.modeling_mistral", "MistralRMSNorm"): "ordinary",
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2RMSNorm"): "ordinary",
    ("transformers.models.qwen3.modeling_qwen3", "Qwen3RMSNorm"): "ordinary",
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5RMSNorm"): "zero-centered",
    ("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe", "Qwen3_5MoeRMSNorm"): "zero-centered",
}


class ParameterInspectionError(ValueError):
    """A small parameter could not be inspected without disturbing dispatch."""


def materialize_parameter_for_inspection(module, parameter_name):
    """Return an owned CPU copy of a resident or Accelerate-offloaded parameter."""
    parameter = getattr(module, parameter_name, None)
    if parameter is None:
        raise ParameterInspectionError(f"parameter {parameter_name!r} is absent")
    if parameter.device.type != "meta":
        return parameter.detach().cpu().clone()
    hook = getattr(module, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if hook is None or weights_map is None:
        raise ParameterInspectionError(
            f"meta parameter {parameter_name!r} has no supported Accelerate weights_map"
        )
    try:
        value = weights_map[parameter_name]
        if isinstance(value, torch.Tensor) and value.device.type != "meta":
            return value.detach().cpu().clone()
    except (KeyError, OSError, RuntimeError, ValueError, TypeError):
        pass
    prefix = getattr(weights_map, "prefix", None)
    candidates = ([f"{prefix}{parameter_name}"] if isinstance(prefix, str) else [])
    errors = []
    for key in candidates:
        try:
            value = weights_map[key]
            if isinstance(value, torch.Tensor) and value.device.type != "meta":
                return value.detach().cpu().clone()
        except (KeyError, OSError, RuntimeError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    detail = f": {'; '.join(errors[:2])}" if errors else ""
    raise ParameterInspectionError(
        f"Accelerate weights_map cannot materialize {parameter_name!r}{detail}"
    )


def _norm_probe_device(norm, placeholder):
    if placeholder.device.type != "meta":
        return placeholder.device
    device = getattr(getattr(norm, "_hf_hook", None), "execution_device", None)
    if device is None or torch.device(device).type == "meta":
        raise ParameterInspectionError("offloaded norm has no usable execution_device")
    return torch.device(device)


def validate_rms_norm(norm, hidden, name="RMSNorm", expected=None):
    """Validate the structural and functional contract used by read hooks.

    Phase 1 accepts only exact audited Transformers classes.  Rank-3 functional
    probes are defense in depth, not generic RMS recognition.
    """
    if norm is None or not callable(norm):
        raise ValueError(f"{name} is absent or noncallable")
    if not callable(getattr(norm, "register_forward_hook", None)):
        raise ValueError(f"{name} is not hookable")
    weight_parameter = getattr(norm, "weight", None)
    if weight_parameter is None or tuple(weight_parameter.shape) != (hidden,):
        raise ValueError(f"{name} has wrong hidden width")
    cls = type(norm)
    if expected is not None:
        validate_forward_provenance(norm, expected, name)
        _validate_direct_inventory(norm, (), ("weight",), name)
    gain_kind = _AUDITED_RMS_NORMS.get((cls.__module__, cls.__name__))
    if gain_kind is None:
        raise ValueError(f"{name} class {cls.__module__}.{cls.__name__} is not audited")
    weight = materialize_parameter_for_inspection(norm, "weight")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"{name} gain is nonfinite")
    epsilon = getattr(norm, "variance_epsilon", getattr(norm, "eps", None))
    if not isinstance(epsilon, (float, int)) or not (0 < float(epsilon) < 1):
        raise ValueError(f"{name} has invalid RMS epsilon")
    device = _norm_probe_device(norm, weight_parameter)
    probe_dtype = weight.dtype if weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    rtol, atol = ((3e-2, 3e-3) if probe_dtype == torch.bfloat16 else
                  (5e-3, 5e-4) if probe_dtype == torch.float16 else
                  (3e-4, 3e-5))
    try:
        with torch.no_grad():
            zero = torch.zeros(2, 3, hidden, device=device, dtype=probe_dtype)
            ones = torch.ones_like(zero)
            base = torch.linspace(-1.37, 2.11, hidden, device=device, dtype=probe_dtype)
            probes = (torch.stack([base.roll(i) * (1 + i / 7) for i in range(6)]).reshape(2, 3, hidden),
                      torch.stack([base.flip(0).roll(2 * i) - i / 9 for i in range(6)]).reshape(2, 3, hidden))
            outputs = [norm(zero), norm(ones)] + [norm(x) for x in probes]
        if any(out.shape != zero.shape or not torch.isfinite(out).all() for out in outputs):
            raise ValueError(f"{name} produced nonfinite or wrong-shaped probe output")
        if not torch.allclose(outputs[0].float(), torch.zeros_like(outputs[0].float()),
                              rtol=0, atol=atol):
            raise ValueError(f"{name} has an additive offset")
        gamma = outputs[1].float()[0, 0]
        expected_gain = weight.float().to(gamma.device) + (1 if gain_kind == "zero-centered" else 0)
        if not torch.allclose(gamma, expected_gain, rtol=rtol, atol=atol):
            raise ValueError(f"{name} effective gain does not match audited {gain_kind} semantics")
        if (gamma.abs() <= max(atol, 1e-6)).any():
            raise ValueError(f"{name} has an effectively singular gain channel")
        for x, output in zip(probes, outputs[2:]):
            z = output.float() / gamma.float()
            x32 = x.float()
            tiny = torch.finfo(torch.float32).tiny
            alpha = (z * x32).sum(-1, keepdim=True) / (x32.square().sum(-1, keepdim=True) + tiny)
            expected = alpha * x32
            if not torch.isfinite(z).all() or not torch.allclose(
                    z, expected, rtol=rtol, atol=atol):
                raise ValueError(f"{name} is not rank-3 direction-preserving RMS normalization")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{name} failed the RMS semantic probe: {exc}") from exc
    return norm


def _block_inventory(block, index, hidden, validated_norms):
    spec = AUDITED_DECODER_SPECS.get(type(block))
    if spec is None:
        raise ValueError(f"layer {index} decoder class or forward is not audited")
    validate_forward_provenance(block, spec.decoder, f"layer {index} decoder")
    for marker in _UNSUPPORTED_MARKERS:
        if getattr(block, marker, None) is not None:
            raise ValueError(f"layer {index} has unsupported residual branch {marker}")
    present_mixers = tuple(name for name in ("self_attn", "linear_attn")
                           if _submodule(block, name) is not None)
    contracts = {item[0]: item for item in spec.mixers}
    if len(present_mixers) != 1 or present_mixers[0] not in contracts:
        raise ValueError(f"layer {index} token-mixer topology is not audited")
    kind = present_mixers[0]
    _kind, mixer_class, mixer_modules, mixer_parameters, discriminator = contracts[kind]
    mixer_module = getattr(block, kind)
    validate_forward_provenance(mixer_module, mixer_class, f"layer {index} {kind}")
    _validate_direct_inventory(mixer_module, mixer_modules, mixer_parameters,
                               f"layer {index} {kind}")
    for child_name in mixer_modules:
        child = getattr(mixer_module, child_name)
        if child_name in ("q_norm", "k_norm"):
            child_class = _QWEN_NORM
        elif child_name == "norm":
            child_class = _QWEN_GATED_NORM
        elif child_name == "conv1d":
            child_class = _CONV1D
        elif child_name == "act":
            child_class = _SILU
        else:
            child_class = _LINEAR
        validate_forward_provenance(
            child, child_class, f"layer {index} {kind}.{child_name}"
        )
        child_label = f"layer {index} {kind}.{child_name}"
        if child_class == _LINEAR:
            _validate_direct_inventory(child, (), ("weight", "bias"), child_label)
            if child.bias is not None or child.weight.ndim != 2 or not child.weight.dtype.is_floating_point:
                raise ValueError(f"{child_label} storage is not audited")
        elif child_class == _SILU:
            _validate_direct_inventory(child, (), (), child_label)
        elif child_class.cls is torch.nn.Conv1d:
            _validate_direct_inventory(child, (), ("weight", "bias"), child_label)
        elif child_name in ("q_norm", "k_norm"):
            validate_rms_norm(child, child.weight.numel(), child_label, _QWEN_NORM)
        else:
            _validate_direct_inventory(child, (), ("weight",), child_label)
    if discriminator is not None:
        layer_type = getattr(block, "layer_type", None)
        block_type = getattr(block, "block_type", None)
        mixer_discriminator = getattr(mixer_module, "layer_type", None)
        version = Version(transformers.__version__)
        if version < Version("5.14"):
            valid = layer_type == discriminator and block_type is None
        else:
            valid = block_type == discriminator and layer_type is None
            if kind == "linear_attn":
                valid = valid and mixer_discriminator in (None, discriminator)
            else:
                valid = valid and mixer_discriminator is None
        if not valid:
            raise ValueError(
                f"layer {index} mixer discriminator is inconsistent: "
                f"layer_type={layer_type!r}, block_type={block_type!r}, "
                f"mixer={mixer_discriminator!r}, expected={discriminator!r}"
            )
    if kind == "self_attn" and spec.decoder.cls in (
            llama_modeling.LlamaDecoderLayer, qwen_moe_modeling.Qwen3_5MoeDecoderLayer):
        q, k, v, o = (mixer_module.q_proj.weight, mixer_module.k_proj.weight,
                      mixer_module.v_proj.weight, mixer_module.o_proj.weight)
        num_heads = getattr(mixer_module, "num_heads", None)
        num_kv_heads = getattr(mixer_module, "num_key_value_heads", None)
        head_dim = getattr(mixer_module, "head_dim", None)
        scaling = getattr(mixer_module, "scaling", None)
        if isinstance(num_heads, int) and isinstance(head_dim, int):
            q_out = num_heads * head_dim
            if q.shape[0] != q_out or o.shape[1] != q_out:
                raise ValueError(f"layer {index} attention projection dimensions are modified")
        if isinstance(num_kv_heads, int) and isinstance(head_dim, int):
            kv_out = num_kv_heads * head_dim
            if k.shape[0] != kv_out or v.shape[0] != kv_out:
                raise ValueError(f"layer {index} attention projection dimensions are modified")
        if (k.shape != v.shape or o.shape[0] != hidden
                or any(weight.shape[1] != hidden for weight in (q, k, v))
                or not all(weight.dtype.is_floating_point for weight in (q, k, v, o))):
            raise ValueError(f"layer {index} attention projection dimensions are modified")
        if scaling is not None and not isinstance(scaling, (float, int)):
            raise ValueError(f"layer {index} attention configuration is modified")
    if (kind == "linear_attn" and spec.packed
            and mixer_class.cls is qwen_moe_modeling.Qwen3_5MoeGatedDeltaNet):
        dt_bias, a_log = mixer_module.dt_bias, mixer_module.A_log
        conv = mixer_module.conv1d
        qkv, z, b, a, out_proj = (mixer_module.in_proj_qkv.weight,
                                  mixer_module.in_proj_z.weight,
                                  mixer_module.in_proj_b.weight,
                                  mixer_module.in_proj_a.weight,
                                  mixer_module.out_proj.weight)
        if (dt_bias.ndim != 1 or a_log.ndim != 1 or dt_bias.shape != a_log.shape
                or not dt_bias.dtype.is_floating_point or not a_log.dtype.is_floating_point
                or qkv.shape[1] != hidden or any(weight.shape[1] != hidden for weight in (z, b, a))
                or out_proj.shape[0] != hidden or out_proj.shape[1] != z.shape[0]
                or conv.weight.ndim != 3 or conv.bias is not None
                or conv.groups != conv.in_channels or conv.weight.shape[0] != conv.in_channels
                or conv.weight.shape[1] != 1):
            raise ValueError(f"layer {index} linear attention raw parameters are modified")
    expected_children = {kind, "mlp", "input_layernorm", "post_attention_layernorm"}
    _validate_direct_inventory(block, expected_children, (), f"layer {index} decoder")
    mixer = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
             if kind == "self_attn" else
             ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
              "linear_attn.in_proj_b", "linear_attn.in_proj_a"))
    sparse_present = _submodule(block, "mlp.experts.gate_up_proj") is not None
    if sparse_present != spec.packed:
        raise ValueError(f"layer {index} MLP topology is not audited")
    mlp = block.mlp
    validate_forward_provenance(mlp, spec.mlp_class, f"layer {index} mlp")
    _validate_direct_inventory(mlp, spec.mlp_modules, spec.mlp_parameters,
                               f"layer {index} mlp")
    if not spec.packed and "act_fn" in spec.mlp_modules:
        validate_forward_provenance(
            mlp.act_fn, _SILU,
            f"layer {index} mlp.act_fn"
        )
        _validate_direct_inventory(mlp.act_fn, (), (), f"layer {index} mlp.act_fn")
        if (mlp.gate_proj.weight.shape != mlp.up_proj.weight.shape
                or mlp.down_proj.weight.shape != mlp.gate_proj.weight.T.shape):
            raise ValueError(f"layer {index} dense MLP dimensions are modified")
    if spec.packed:
        router, experts, shared = mlp.gate, mlp.experts, mlp.shared_expert
        validate_forward_provenance(router, spec.router_class, f"layer {index} router")
        validate_forward_provenance(experts, spec.experts_class, f"layer {index} experts")
        validate_forward_provenance(shared, spec.shared_expert_class,
                                    f"layer {index} shared expert")
        router_parameters = (("weight", "bias") if spec.router_class == _LINEAR
                             else ("weight",))
        _validate_direct_inventory(router, (), router_parameters, f"layer {index} router")
        experts_modules = (("act_fn",) if spec.experts_class.cls is
                           qwen_moe_modeling.Qwen3_5MoeExperts else ())
        _validate_direct_inventory(experts, experts_modules,
                                   ("gate_up_proj", "down_proj"),
                                   f"layer {index} experts")
        gate_up, down = experts.gate_up_proj, experts.down_proj
        if (gate_up.ndim != 3 or down.ndim != 3
                or not gate_up.dtype.is_floating_point or not down.dtype.is_floating_point
                or gate_up.shape[0] != down.shape[0]
                or gate_up.shape[1] != 2 * down.shape[2]
                or gate_up.shape[2] != hidden or down.shape[1] != hidden):
            raise ValueError(f"layer {index} packed expert storage is modified")
        if "act_fn" in experts_modules:
            validate_forward_provenance(
                experts.act_fn, _SILU, f"layer {index} experts.act_fn"
            )
            _validate_direct_inventory(experts.act_fn, (), (),
                                       f"layer {index} experts.act_fn")
        shared_modules = (("gate_proj", "up_proj", "down_proj", "act_fn")
                          if spec.shared_expert_class.cls is qwen_moe_modeling.Qwen3_5MoeMLP
                          else ("gate_proj", "up_proj", "down_proj"))
        _validate_direct_inventory(shared, shared_modules, (),
                                   f"layer {index} shared expert")
        if (shared.gate_proj.weight.shape != shared.up_proj.weight.shape
                or shared.down_proj.weight.shape != shared.gate_proj.weight.T.shape):
            raise ValueError(f"layer {index} shared expert dimensions are modified")
        for child_name in ("gate_proj", "up_proj", "down_proj"):
            validate_forward_provenance(
                getattr(shared, child_name),
                _LINEAR,
                f"layer {index} shared expert.{child_name}"
            )
        if "act_fn" in shared_modules:
            validate_forward_provenance(
                shared.act_fn, _SILU,
                f"layer {index} shared expert.act_fn"
            )
            _validate_direct_inventory(shared.act_fn, (), (),
                                       f"layer {index} shared expert.act_fn")
        validate_forward_provenance(mlp.shared_expert_gate,
                                    _LINEAR,
                                    f"layer {index} shared expert gate")
    targets = [TransformTarget(path + ".weight", path, "input_layernorm")
               for path in mixer]
    if spec.packed:
        targets.extend(_MOE_READS)
    else:
        norm_name = ("post_attention_layernorm"
                     if getattr(block, "post_attention_layernorm", None) is not None
                     else "input_layernorm")
        targets.extend(TransformTarget(path + ".weight", path, norm_name)
                       for path in ("mlp.gate_proj", "mlp.up_proj"))
    norm_names = tuple(dict.fromkeys(target.norm_name for target in targets))
    norms = []
    for name in norm_names:
        norm = getattr(block, name, None)
        if id(norm) not in validated_norms:
            validate_rms_norm(norm, hidden, f"layer {index} {name}", spec.norm_class)
            validated_norms.add(id(norm))
        norms.append((name, norm))
    reads = []
    for target in targets:
        tensor = target.tensor(block)
        module = _submodule(block, target.accessor)
        if tensor is None or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"layer {index} reader {target.state_suffix} is absent")
        raw_packed = spec.packed and not target.state_suffix.endswith(".weight")
        if raw_packed:
            if not isinstance(module, torch.nn.Parameter) or tensor.ndim != 3:
                raise ValueError(f"layer {index} packed reader {target.state_suffix} is modified")
        elif target.accessor == "mlp.gate" and spec.packed:
            if tensor.ndim != 2:
                raise ValueError(f"layer {index} packed router storage is modified")
        elif (type(module) is not torch.nn.Linear or tensor.ndim != 2):
            raise ValueError(f"layer {index} reader {target.state_suffix} is not an audited Linear")
        if isinstance(module, torch.nn.Module):
            expected = spec.router_class if target.accessor == "mlp.gate" else _LINEAR
            validate_forward_provenance(module, expected,
                                        f"layer {index} reader {target.state_suffix}")
            reader_parameters = (("weight",) if target.accessor == "mlp.gate"
                                 and spec.packed and spec.router_class != _LINEAR
                                 else ("weight", "bias"))
            _validate_direct_inventory(module, (), reader_parameters,
                                       f"layer {index} reader {target.state_suffix}")
            if getattr(module, "bias", None) is not None:
                raise ValueError(f"layer {index} reader {target.state_suffix} has bias")
        if not tensor.dtype.is_floating_point or tensor.shape[target.axis] != hidden:
            raise ValueError(f"layer {index} reader {target.state_suffix} has invalid storage")
        reads.append((target, tensor, getattr(block, target.norm_name)))
    expected_writes = ((f"{kind}.o_proj",) if kind == "self_attn" else
                       ("linear_attn.out_proj",)) + ("mlp.down_proj",)
    writes, exact_ok, exact_reason = [], not spec.packed, "supported"
    if spec.packed:
        exact_reason = "packed residual writers are unsupported"
    else:
        present = {name for name in WRITES if _submodule(block, name) is not None}
        if present != set(expected_writes):
            exact_ok, exact_reason = False, "residual writer inventory is incomplete"
        for name in expected_writes:
            writer, weight = _submodule(block, name), None
            if writer is not None:
                weight = getattr(writer, "weight", None)
            if (type(writer) is not torch.nn.Linear or weight is None or weight.ndim != 2
                    or not weight.dtype.is_floating_point or weight.shape[0] != hidden
                    or getattr(writer, "bias", None) is not None
                    or not callable(getattr(writer, "register_forward_hook", None))):
                exact_ok, exact_reason = False, f"residual writer {name} is unsupported"
            else:
                validate_forward_provenance(
                    writer, _LINEAR,
                    f"layer {index} writer {name}"
                )
                _validate_direct_inventory(writer, (), ("weight", "bias"),
                                           f"layer {index} writer {name}")
            writes.append((name, writer))
    return BlockInventory(index, kind, tuple(norms), tuple(reads), tuple(writes),
                          spec.packed, exact_ok, exact_reason)


def model_inventory(jl):
    layers = getattr(jl, "layers", None)
    if not layers:
        raise ValueError("model has no decoder blocks")
    embed = getattr(jl, "_embed_tokens", None)
    embed_weight = getattr(embed, "weight", None)
    if embed_weight is None or embed_weight.ndim != 2:
        raise ValueError("embedding storage is unsupported")
    validate_forward_provenance(embed, _EMBEDDING, "embedding")
    _validate_direct_inventory(embed, (), ("weight",), "embedding")
    hidden = embed_weight.shape[-1]
    blocks, validated_norms = [], set()
    block_ids, norm_layers, module_sites, tensor_sites, names = set(), {}, {}, {}, set()
    family = None
    decoder_spans = []
    decoder_tensor_ids = set()
    for index, block in enumerate(layers):
        if id(block) in block_ids:
            raise ValueError("decoder block object is shared across layers")
        block_ids.add(id(block))
        if family is None:
            family = type(block)
        elif type(block) is not family:
            raise ValueError("mixed decoder families are not audited")
        inventory = _block_inventory(block, index, hidden, validated_norms)
        for norm_name, norm in inventory.norms:
            previous = norm_layers.setdefault(id(norm), index)
            if previous != index:
                raise ValueError("normalization module is shared across decoder layers")
        for role, entries in (("reader", inventory.reads), ("writer", inventory.writes)):
            for entry in entries:
                if role == "reader":
                    target, tensor, _norm = entry
                    path, module = target.state_suffix, _submodule(block, target.accessor)
                else:
                    path, module = entry
                    tensor = getattr(module, "weight", None)
                logical = f"layers.{index}.{path}"
                if logical in names:
                    raise ValueError("duplicate logical checkpoint target")
                names.add(logical)
                for table, identity in ((module_sites, id(module)), (tensor_sites, id(tensor))):
                    previous = table.setdefault(identity, (role, logical))
                    if previous != (role, logical):
                        raise ValueError(f"unsafe {role} module or parameter alias")
                span = storage_span(tensor)
                if any(_overlaps(span, prior) for prior in decoder_spans):
                    raise ValueError("decoder reader/writer storage aliases another site")
                if span is not None:
                    decoder_spans.append(span)
                    decoder_tensor_ids.add(id(tensor))
        for _name, tensor in tuple(block.named_parameters(recurse=True)) + tuple(block.named_buffers(recurse=True)):
            if id(tensor) in decoder_tensor_ids:
                continue
            span = storage_span(tensor)
            if any(_overlaps(span, prior) for prior in decoder_spans):
                raise ValueError("decoder storage aliases another audited site")
            if span is not None:
                decoder_spans.append(span)
                decoder_tensor_ids.add(id(tensor))
        blocks.append(inventory)
    final_norm = getattr(jl, "_final_norm", None)
    if id(final_norm) in norm_layers:
        raise ValueError("final norm aliases a decoder block norm")
    final_spec = AUDITED_DECODER_SPECS[family]
    validate_rms_norm(final_norm, hidden, "final norm", final_spec.norm_class)
    head = getattr(jl, "_lm_head", None)
    head_weight = getattr(head, "weight", None)
    validate_forward_provenance(
        head, _LINEAR, "final lm_head"
    )
    _validate_direct_inventory(head, (), ("weight", "bias"), "final lm_head")
    if (head_weight is None or head_weight.ndim != 2
            or not head_weight.dtype.is_floating_point or head_weight.shape[1] != hidden
            or getattr(head, "bias", None) is not None
            or not callable(getattr(head, "register_forward_hook", None))):
        raise ValueError("final lm_head storage or execution contract is unsupported")
    head_name = f"{jl.layout.lm_head}.weight"
    if head_name in names:
        raise ValueError("duplicate final checkpoint target")
    if id(head) in module_sites or id(head_weight) in tensor_sites:
        raise ValueError("final lm_head aliases a decoder reader or writer")
    head_span, embed_span = storage_span(head_weight), storage_span(embed_weight)
    if head_span is None and head_weight is not embed_weight:
        head_span = _offloaded_registered_span(head, "weight", head_weight)
    if embed_span is None and embed_weight is not head_weight:
        embed_span = _offloaded_registered_span(embed, "weight", embed_weight)
    config = getattr(getattr(jl, "_hf_model", None), "config", None)
    declared_tie = getattr(config, "tie_word_embeddings", False) is True
    tied = head_weight is embed_weight
    if declared_tie != tied:
        raise ValueError("embedding/head tie declaration does not match physical parameters")
    if head_span is None or embed_span is None:
        if not tied:
            raise ValueError("embedding/head storage cannot be classified safely")
    elif not tied and _overlaps(head_span, embed_span):
        raise ValueError("distinct embedding and head parameters share storage")
    if any(_overlaps(span, head_span) or _overlaps(span, embed_span)
           for span in decoder_spans):
        raise ValueError("embedding or head storage aliases decoder storage")
    if embed_weight.ndim != 2 or embed_weight.shape[1] != hidden or embed_weight.shape[0] != head_weight.shape[0] or not embed_weight.dtype.is_floating_point:
        raise ValueError("embedding/head contract is incomplete")
    packed = any(block.packed for block in blocks)
    exact_ok = not packed and all(block.exact_supported for block in blocks)
    reason = ("packed residual writers are unsupported" if packed else
              next((block.exact_reason for block in blocks
                    if not block.exact_supported), "supported"))
    embed_name = f"{jl.layout.path}.{jl.layout.embed}.weight"
    return ModelInventory(tuple(blocks), final_norm, hidden, packed, exact_ok, reason,
                          (head_name, head, head_weight),
                          (embed_name, embed, embed_weight), tied, jl,
                          tuple(id(block) for block in layers))


def block_capabilities(block, _validated_norms=None):
    try:
        first = next((m for name in ("self_attn.q_proj", "linear_attn.in_proj_qkv")
                      if (m := _submodule(block, name)) is not None), None)
        hidden = first.weight.shape[-1]
        inv = _block_inventory(block, 0, hidden,
                               _validated_norms if _validated_norms is not None else set())
        return RebaseCapabilities(True, "supported", inv.exact_supported, inv.exact_reason)
    except (AttributeError, StopIteration, TypeError, ValueError) as exc:
        return RebaseCapabilities(False, str(exc), False, f"readthrough unsupported: {exc}")


PACKED_EXACT_REASON = "exact mode is unavailable for packed MoE models"


def _deny_known_packed_exact(jl):
    """Deny only an entirely canonical packed decoder class; never grants support."""
    layers = getattr(jl, "layers", None)
    packed_spec = AUDITED_DECODER_SPECS[qwen_moe_modeling.Qwen3_5MoeDecoderLayer]
    if layers and all(type(block) is packed_spec.decoder.cls for block in layers):
        try:
            for index, block in enumerate(layers):
                validate_forward_provenance(block, packed_spec.decoder,
                                            f"layer {index} decoder")
        except ValueError:
            return
        raise ValueError(PACKED_EXACT_REASON)


def _validate_inventory_policy(inventory, jl, *, exact=False):
    """Authorize an operation-private inventory for the exact requested mode."""
    if type(inventory) is not ModelInventory or inventory.owner is not jl:
        raise ValueError("model inventory does not belong to this model operation")
    layers = getattr(jl, "layers", None)
    if (layers is None or len(layers) != len(inventory.layer_ids)
            or tuple(id(block) for block in layers) != inventory.layer_ids
            or inventory.final_norm is not getattr(jl, "_final_norm", None)
            or inventory.final_head[1] is not getattr(jl, "_lm_head", None)
            or inventory.embedding[1] is not getattr(jl, "_embed_tokens", None)):
        raise ValueError("model changed after inventory validation")
    for index, block in enumerate(layers):
        spec = AUDITED_DECODER_SPECS.get(type(block))
        if spec is None:
            raise ValueError("model topology is no longer audited")
        validate_forward_provenance(block, spec.decoder, f"layer {index} decoder")
        block_inventory = inventory.blocks[index]
        for target, tensor, norm in block_inventory.reads:
            if target.tensor(block) is not tensor:
                raise ValueError("model reader changed after inventory validation")
            if getattr(block, target.norm_name, None) is not norm:
                raise ValueError("model norm changed after inventory validation")
        for suffix, module in block_inventory.writes:
            if _submodule(block, suffix) is not module:
                raise ValueError("model writer changed after inventory validation")
    head_name, head, head_weight = inventory.final_head
    embed_name, embed, embed_weight = inventory.embedding
    if getattr(head, "weight", None) is not head_weight or getattr(embed, "weight", None) is not embed_weight:
        raise ValueError("embedding/head storage changed after inventory validation")
    declared_tie = getattr(getattr(getattr(jl, "_hf_model", None), "config", None), "tie_word_embeddings", False) is True
    if declared_tie != inventory.tied or (head_weight is embed_weight) != inventory.tied:
        raise ValueError("embedding/head tie changed after inventory validation")
    if exact and inventory.packed:
        raise ValueError(PACKED_EXACT_REASON)
    if exact and not inventory.exact_supported:
        raise ValueError("exact mode is unavailable for this model: " + inventory.exact_reason)
    return inventory


def model_preflight(jl, *, exact=False):
    if exact:
        _deny_known_packed_exact(jl)
    inventory = model_inventory(jl)
    return _validate_inventory_policy(inventory, jl, exact=exact)


def has_packed_read_parameters(jl):
    return model_inventory(jl).packed


def iter_reads(block):
    inv = _block_inventory(
        block, 0, next(iter(_submodule(block, p).weight.shape[-1]
                            for p in ("self_attn.q_proj", "linear_attn.in_proj_qkv")
                            if _submodule(block, p) is not None)), set())
    yield from inv.reads


def iter_writes(block):
    inv = _block_inventory(
        block, 0, next(iter(_submodule(block, p).weight.shape[-1]
                            for p in ("self_attn.q_proj", "linear_attn.in_proj_qkv")
                            if _submodule(block, p) is not None)), set())
    yield from inv.writes


def rule_factors(rules, scale):
    """Rank-1 factors ``{layer: [(w, v̂), ...]}`` float32 CPU, in the standard
    hook's application order (increasing layers, rules in order).
    ``w = α·v̂_A + β·v̂_B`` with the effective coefficients (saturation included).
    Returns an empty dict if all coefficients are neutral."""
    by_layer = {}
    for rule in rules:
        alpha, beta = effective_coeffs(rule["mode"], rule["factor"], scale)
        if alpha == 0.0 and not beta:
            continue
        for layer in rule["layers"]:
            v_a = rule["dirs_a"][layer].detach().float().cpu()
            w = alpha * v_a
            if beta:
                w = w + beta * rule["dirs_b"][layer].detach().float().cpu()
            if w.norm() < 1e-8:
                continue  # null W_U row → empty direction, nothing to apply
            by_layer.setdefault(int(layer), []).append((w, v_a))
    return by_layer


def _compose_left(U, V, w, v):
    """``(I + w·vᵀ)·(I + U·Vᵀ)`` → new ``(U, V)`` (one more column)."""
    if U is None:
        return w.unsqueeze(1), v.unsqueeze(1)
    v_new = v + V @ (U.T @ v)
    return torch.cat([U, w.unsqueeze(1)], dim=1), torch.cat([V, v_new.unsqueeze(1)], dim=1)


def compress_uv(U, V, tol=1e-5):
    """Recompacts ``C − I = U·Vᵀ`` via QR + truncated SVD.

    Essential, not cosmetic: a token's directions across layers are nearly
    collinear, so naive composition inflates the columns (multiplicative cross
    terms) and the result only holds through cancellation between large numbers —
    invisible in float32, destructive in bf16 (live preview → random tokens,
    measured). After compression V is orthonormal and U carries the true singular
    values (~O(1)): stable in bf16 and rank reduced to the effective rank."""
    Qu, Ru = torch.linalg.qr(U)
    Qv, Rv = torch.linalg.qr(V)
    Us, S, Vh = torch.linalg.svd(Ru @ Rv.T)
    keep = S > tol * S.max().clamp_min(1e-12)
    return Qu @ (Us[:, keep] * S[keep]), Qv @ Vh.T[:, keep]


def cumulative(rules, scale, n_layers):
    """Cumulative transforms ``{m: (U, V)}`` for each read point:
    m = layer (its reads see ``C_m`` = hooks of layers < m);
    the ``n_layers`` key is the final norm / lm_head point.
    Returns ``{}`` if no factor is active. The (U, V) of consecutive layers with
    no intermediate hook share their tensors (never mutated)."""
    factors = rule_factors(rules, scale)
    if not factors:
        return {}
    l_min = min(factors)
    out = {}
    U = V = None
    for layer in range(l_min, n_layers):
        if factors.get(layer):
            for w, v in factors[layer]:
                U, V = _compose_left(U, V, w, v)
            U, V = compress_uv(U, V)
        if U is not None:
            out[layer + 1] = (U, V)
    return out


def effective_gamma(norm):
    """MEASURED effective γ: ``norm(1⃗) = γ_eff`` since rms(1⃗) = 1.

    Do NOT read ``norm.weight`` directly: Qwen3.5 (like Gemma) uses a
    zero-centered RMSNorm where γ = 1 + weight — dividing by ``weight`` (~0, of
    arbitrary sign) made the transform chaotic (live preview → random tokens,
    measured). The functional measurement covers both styles."""
    placeholder = norm.weight
    weight = materialize_parameter_for_inspection(norm, "weight").float()
    cls = type(norm)
    gain_kind = _AUDITED_RMS_NORMS.get((cls.__module__, cls.__name__))
    if gain_kind is None:
        raise ValueError(f"norm class {cls.__module__}.{cls.__name__} is not audited")
    epsilon = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 0.0)))
    gamma = weight + (1 if gain_kind == "zero-centered" else 0)
    gamma = gamma / (1.0 + epsilon) ** 0.5
    if placeholder.device.type == "meta" and not torch.isfinite(gamma).all():
        raise ParameterInspectionError("offloaded norm gain is nonfinite")
    return gamma.detach().clone()


def gamma_pair(norm, U, V):
    """``(γ⊙U, V/γ)`` float32 CPU for the read via this RMSNorm:
    ``W·Γ·C·Γ⁻¹ = W + (W·(γ⊙U))·(V/γ)ᵀ``."""
    gamma = effective_gamma(norm)
    safe = torch.where(gamma.abs() < GAMMA_EPS, torch.full_like(gamma, GAMMA_EPS), gamma)
    return gamma.unsqueeze(1) * U, V / safe.unsqueeze(1)


def apply_read(W, Ug, Vg):
    """Right-transform ``[..., output_features, hidden_size]`` tensors."""
    if W.ndim < 2:
        raise ValueError("read tensor must have at least two dimensions")
    if W.shape[-1] != Ug.shape[0] or Ug.shape[0] != Vg.shape[0]:
        raise ValueError("read tensor residual hidden axis does not match transform")
    B = W @ Ug  # [out, r]
    result = W + B @ Vg.T
    if result.shape != W.shape:
        raise ValueError("read transform changed tensor shape")
    return result, B, Vg.T.contiguous()


def inverse_uv(U, V):
    """``C⁻¹ = I − U_inv·Vᵀ`` (Woodbury: ``U_inv = U·(I_r + VᵀU)⁻¹``).
    Returns ``(U_inv, V, regularized)``; near a full zap the small matrix is
    singular → thresholded pseudo-inverse (the local effect ≈ readthrough)."""
    r = U.shape[1]
    small = torch.eye(r) + V.T @ U
    svals = torch.linalg.svdvals(small)
    regularized = bool(svals.min() < INV_COND_EPS * max(1.0, float(svals.max())))
    if regularized:
        inv = torch.linalg.pinv(small, rtol=INV_COND_EPS)
    else:
        inv = torch.linalg.inv(small)
    return U @ inv, V, regularized


def apply_write(W, U_inv, V):
    """``W ← (I − U_inv·Vᵀ)·W``; returns ``(W_new, B, A)`` with delta = B·A."""
    A = V.T @ W  # [r, in]
    return W - U_inv @ A, (-U_inv).contiguous(), A


def apply_transform(entry, W):
    """Applies a plan entry to a float32 weight.

    Returns ``(W_new, B, A)`` where ``B·A`` is the EXACT delta ``W_new − W``:
    the rebase update is low-rank by construction, which is what makes the LoRA
    export exact rather than an approximation."""
    kind, X, Y = entry
    if kind == "read":
        return apply_read(W, X, Y)
    return apply_write(W, X, Y)


def build_plan(rules, jl, scale, exact=False):
    """Bake plan: ``{param_name: entry}`` with ``entry = ("read", Ug, Vg)`` or
    ``("write", U_inv, V)`` — apply with :func:`apply_transform` — plus the
    diagnostic metadata.

    The names follow the model's layout (``{path}.layers.{m}.{suffix}.weight``,
    ``{lm_head}.weight``); the guard matching them against the checkpoint keys is
    done by the export."""
    from core.capabilities import ensure_unquantized
    ensure_unquantized(jl)
    inventory = model_preflight(jl, exact=exact)
    return _build_plan_from_inventory(rules, jl, scale, exact=exact, inventory=inventory)


def _build_plan_from_inventory(rules, jl, scale, *, exact=False, inventory):
    _validate_inventory_policy(inventory, jl, exact=exact)
    active = [r for r in rules if r["layers"]]
    if not active:
        raise ValueError("no active rule (all have 0 layers): nothing to export")
    n_layers = len(jl.layers)
    cums = cumulative(active, scale, n_layers)
    if not cums:
        raise ValueError(
            "all coefficients neutral (factors at 1 and/or scale=0): "
            "the bake would change no weight"
        )
    path = jl.layout.path
    transforms = {}
    regularized_layers = []
    min_gamma = None

    for m in sorted(k for k in cums if k < n_layers):
        U, V = cums[m]
        block = jl.layers[m]
        block_inventory = inventory.blocks[m]
        for target, _tensor, norm in block_inventory.reads:
            Ug, Vg = gamma_pair(norm, U, V)
            g_min = effective_gamma(norm).abs().min().item()
            min_gamma = g_min if min_gamma is None else min(min_gamma, g_min)
            transforms[f"{path}.layers.{m}.{target.state_suffix}"] = ("read", Ug, Vg)
        if exact:
            U_inv, Vw, regularized = inverse_uv(U, V)
            if regularized:
                regularized_layers.append(m)
            for suffix, _module in block_inventory.writes:
                transforms[f"{path}.layers.{m}.{suffix}.weight"] = ("write", U_inv, Vw)

    U, V = cums[n_layers]
    Ug, Vg = gamma_pair(inventory.final_norm, U, V)
    lm_head_key, _head, _head_weight = inventory.final_head
    transforms[lm_head_key] = ("read", Ug, Vg)

    info = {
        "tied": inventory.tied,
        "lm_head_key": lm_head_key,
        "embed_key": inventory.embedding[0],
        "path": path,
        "rank_final": cums[n_layers][0].shape[1],
        "layers_span": [min(cums), n_layers - 1],
        "regularized_layers": regularized_layers,
        "min_gamma": min_gamma,
        "targets": {f"{path}.layers.{m}.{target.state_suffix}": target
                    for m in sorted(k for k in cums if k < n_layers)
                    for target, _tensor, _norm in inventory.blocks[m].reads},
    }
    return transforms, info
