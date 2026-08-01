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
import weakref

import torch

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


_MOE_READS = (
    TransformTarget("mlp.gate.weight", "mlp.gate", "post_attention_layernorm"),
    TransformTarget("mlp.experts.gate_up_proj", "mlp.experts.gate_up_proj", "post_attention_layernorm", lora_supported=False),
    TransformTarget("mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.gate_proj", "post_attention_layernorm"),
    TransformTarget("mlp.shared_expert.up_proj.weight", "mlp.shared_expert.up_proj", "post_attention_layernorm"),
    TransformTarget("mlp.shared_expert_gate.weight", "mlp.shared_expert_gate", "post_attention_layernorm"),
)


_VALIDATED_RMS_NORMS = weakref.WeakKeyDictionary()


def validate_rms_norm(norm, hidden, name="RMSNorm"):
    """Validate the structural and functional contract used by read hooks.

    The read conjugation requires ``N(x) / gamma`` to preserve the direction of
    each token vector.  Class names alone are insufficient, so several probes
    enforce that contract for ordinary- and zero-centered-gain RMSNorms.
    """
    if norm is None or not callable(norm):
        raise ValueError(f"{name} is absent or noncallable")
    if not callable(getattr(norm, "register_forward_hook", None)):
        raise ValueError(f"{name} is not hookable")
    weight = getattr(norm, "weight", None)
    if weight is None or tuple(weight.shape) != (hidden,):
        raise ValueError(f"{name} has wrong hidden width")
    if isinstance(norm, torch.nn.LayerNorm):
        raise ValueError(f"{name} is mean-subtracting LayerNorm, not RMSNorm")
    cache_key = (hidden, weight.dtype, weight.device, getattr(weight, "_version", None))
    if _VALIDATED_RMS_NORMS.get(norm) == cache_key:
        return norm
    device = weight.device
    probe_dtype = weight.dtype if weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    rtol, atol = ((3e-2, 3e-3) if probe_dtype == torch.bfloat16 else
                  (5e-3, 5e-4) if probe_dtype == torch.float16 else
                  (3e-4, 3e-5))
    try:
        with torch.no_grad():
            zero = torch.zeros(1, hidden, device=device, dtype=probe_dtype)
            ones = torch.ones_like(zero)
            probes = (
                torch.linspace(-1.37, 2.11, hidden, device=device, dtype=probe_dtype).unsqueeze(0),
                torch.linspace(0.23, 1.91, hidden, device=device, dtype=probe_dtype).flip(-1).unsqueeze(0),
                torch.where(torch.arange(hidden, device=device) % 2 == 0,
                            torch.tensor(0.61, device=device),
                            torch.tensor(-1.43, device=device)).to(probe_dtype).unsqueeze(0),
            )
            outputs = [norm(zero), norm(ones)] + [norm(x) for x in probes]
        if any(out.shape != zero.shape or not torch.isfinite(out).all() for out in outputs):
            raise ValueError(f"{name} produced nonfinite or wrong-shaped probe output")
        if not torch.allclose(outputs[0].float(), torch.zeros_like(outputs[0].float()),
                              rtol=0, atol=atol):
            raise ValueError(f"{name} has an additive offset")
        gamma = outputs[1].float().flatten()
        valid = gamma.abs() > max(atol, 1e-6)
        if int(valid.sum()) < min(hidden, 2):
            raise ValueError(f"{name} has insufficient nonzero effective gain channels")
        for x, output in zip(probes, outputs[2:]):
            xv = x.float().flatten()[valid]
            normalized = output.float().flatten()[valid] / gamma[valid]
            channel_scale = normalized / xv
            scalar = channel_scale.mean()
            if not torch.isfinite(channel_scale).all() or not torch.allclose(
                    channel_scale, scalar.expand_as(channel_scale), rtol=rtol, atol=atol):
                raise ValueError(f"{name} is not direction-preserving RMS-style normalization")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{name} failed the RMS semantic probe: {exc}") from exc
    _VALIDATED_RMS_NORMS[norm] = cache_key
    return norm


def block_capabilities(block):
    """Positively validate a complete supported topology; unknowns fail closed."""
    try:
        for marker in _UNSUPPORTED_MARKERS:
            if getattr(block, marker, None) is not None:
                raise ValueError(f"layer has unsupported residual branch {marker}")
        full = _submodule(block, "self_attn") is not None
        linear = _submodule(block, "linear_attn") is not None
        if full == linear:
            raise ValueError("token-mixer topology is unknown or ambiguous")
        mixer = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj") if full else
                 ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a"))
        first = _submodule(block, mixer[0])
        hidden = first.weight.shape[-1] if first is not None and hasattr(first, "weight") else None
        if hidden is None:
            raise ValueError(f"required reader {mixer[0]}.weight is absent")
        validate_rms_norm(getattr(block, "input_layernorm", None), hidden, "input_layernorm")
        targets = [TransformTarget(s + ".weight", s, "input_layernorm") for s in mixer]
        sparse = _submodule(block, "mlp.experts.gate_up_proj") is not None
        if sparse:
            targets.extend(_MOE_READS)
            validate_rms_norm(getattr(block, "post_attention_layernorm", None), hidden,
                              "post_attention_layernorm")
        else:
            for s in ("mlp.gate_proj", "mlp.up_proj"):
                if _submodule(block, s) is None:
                    raise ValueError(f"required reader {s}.weight is absent")
                targets.append(TransformTarget(s + ".weight", s, "post_attention_layernorm" if getattr(block, "post_attention_layernorm", None) is not None else "input_layernorm"))
        for target in targets:
            tensor = target.tensor(block)
            if tensor is None:
                raise ValueError(f"required reader {target.state_suffix} is absent")
            if tensor.ndim < 2 or tensor.shape[target.axis] != hidden:
                raise ValueError(f"reader {target.state_suffix} has wrong residual hidden axis")
            validate_rms_norm(getattr(block, target.norm_name, None), hidden, target.norm_name)
        exact_ok = False
        if sparse:
            exact_reason = (
                "packed routed and shared-expert residual writers require a separate "
                "implementation and aggregate-MoE live oracle"
            )
        else:
            # Exact preview hooks, and the bake left-multiplies, every residual
            # writer.  Claim support only when the complete inventory for the
            # positively identified mixer exists and has the contract used by
            # both paths.  In particular a bias would be transformed by the
            # live output hook but is not represented in the current bake.
            expected_writes = (("self_attn.o_proj",) if full else
                               ("linear_attn.out_proj",)) + ("mlp.down_proj",)
            exact_reason = "supported"
            for name in expected_writes:
                writer = _submodule(block, name)
                weight = getattr(writer, "weight", None)
                if writer is None or weight is None:
                    exact_reason = f"required residual writer {name}.weight is absent"
                    break
                if weight.ndim != 2 or weight.shape[0] != hidden:
                    exact_reason = f"residual writer {name}.weight has wrong residual output axis"
                    break
                if getattr(writer, "bias", None) is not None:
                    exact_reason = f"residual writer {name} has an untransformed bias"
                    break
                if not callable(writer) or not callable(getattr(writer, "register_forward_hook", None)):
                    exact_reason = f"residual writer {name} is not hookable by the live exact path"
                    break
            else:
                # No additional writers may silently escape the inventory.
                present = {name for name in WRITES if _submodule(block, name) is not None}
                if present != set(expected_writes):
                    exact_reason = "residual writer inventory is ambiguous or incomplete"
                else:
                    exact_ok = True
        return RebaseCapabilities(True, "supported", exact_ok, exact_reason)
    except ValueError as exc:
        return RebaseCapabilities(False, str(exc), False, f"readthrough unsupported: {exc}")


def model_preflight(jl, *, exact=False):
    """Model-wide capability contract, including the final normalization."""
    if not getattr(jl, "layers", None):
        raise ValueError("model has no decoder blocks")
    hidden = jl._lm_head.weight.shape[-1]
    for index, block in enumerate(jl.layers):
        cap = block_capabilities(block)
        if not cap.readthrough_supported:
            raise ValueError(f"layer {index} readthrough unsupported: {cap.readthrough_reason}")
        if exact and not cap.exact_supported:
            raise ValueError(f"exact mode is unavailable for this model: layer {index}: {cap.exact_reason}")
    validate_rms_norm(getattr(jl, "_final_norm", None), hidden, "final norm")
    return True


def check_block_supported(block):
    capability = block_capabilities(block)
    if not capability.readthrough_supported:
        raise ValueError("architecture not supported by readthrough: " + capability.readthrough_reason)


def iter_reads(block):
    """Yields ``(suffix, module, norm)`` for each residual read."""
    check_block_supported(block)
    if _submodule(block, "mlp.experts.gate_up_proj") is not None:
        mixer = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj") if _submodule(block, "self_attn") is not None else ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a")
        targets = [TransformTarget(s + ".weight", s, "input_layernorm") for s in mixer] + list(_MOE_READS)
        for target in targets:
            yield target, target.tensor(block), getattr(block, target.norm_name)
        return
    for suffix, norm_name in READS.items():
        module = _submodule(block, suffix)
        if module is None:
            continue
        if suffix in READS_POST and getattr(block, "post_attention_layernorm", None) is not None:
            norm_name = "post_attention_layernorm"
        norm = getattr(block, norm_name, None)
        if norm is None or not hasattr(norm, "weight"):
            raise ValueError(f"RMSNorm {norm_name} not found for {suffix}")
        yield TransformTarget(suffix + ".weight", suffix, norm_name), module.weight, norm


def iter_writes(block):
    for suffix in WRITES:
        module = _submodule(block, suffix)
        if module is not None:
            yield suffix, module


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
    weight = norm.weight
    with torch.no_grad():
        ones = torch.ones(1, weight.shape[-1], device=weight.device, dtype=torch.float32)
        return norm(ones).detach().flatten().float().cpu()


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
    model_preflight(jl, exact=exact)
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
        caps = block_capabilities(block)
        if exact and not caps.exact_supported:
            raise ValueError("exact mode is unavailable: " + caps.exact_reason)
        for target, _tensor, norm in iter_reads(block):
            Ug, Vg = gamma_pair(norm, U, V)
            g_min = effective_gamma(norm).abs().min().item()
            min_gamma = g_min if min_gamma is None else min(min_gamma, g_min)
            transforms[f"{path}.layers.{m}.{target.state_suffix}"] = ("read", Ug, Vg)
        if exact:
            U_inv, Vw, regularized = inverse_uv(U, V)
            if regularized:
                regularized_layers.append(m)
            for suffix, _module in iter_writes(block):
                transforms[f"{path}.layers.{m}.{suffix}.weight"] = ("write", U_inv, Vw)

    U, V = cums[n_layers]
    Ug, Vg = gamma_pair(jl._final_norm, U, V)
    lm_head_key = f"{jl.layout.lm_head}.weight"
    transforms[lm_head_key] = ("read", Ug, Vg)

    tied = jl._lm_head.weight.data_ptr() == jl._embed_tokens.weight.data_ptr()
    info = {
        "tied": tied,
        "lm_head_key": lm_head_key,
        "embed_key": f"{path}.{jl.layout.embed}.weight",
        "path": path,
        "rank_final": cums[n_layers][0].shape[1],
        "layers_span": [min(cums), n_layers - 1],
        "regularized_layers": regularized_layers,
        "min_gamma": min_gamma,
        "targets": {f"{path}.layers.{m}.{target.state_suffix}": target for m in sorted(k for k in cums if k < n_layers) for target, _tensor, _norm in iter_reads(jl.layers[m])},
    }
    return transforms, info
