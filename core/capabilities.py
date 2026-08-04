"""Small, fail-closed capability contract for the currently loaded model."""

from uuid import uuid4


PUBLIC_REASONS = {
    "supported": "Supported for this model.",
    "no_model_loaded": "Load a model to use this operation.",
    "quantized_model_unsupported": "Editing quantized model loads is not supported.",
    "architecture_unsupported": "This model architecture has not been validated for this operation.",
    "packed_moe_exact_unsupported": "Exact mode is not supported for packed MoE models.",
    "packed_moe_format_unsupported": "This export format is not supported for packed MoE models.",
    "standard_not_exportable": "Standard interventions cannot be exported faithfully.",
    "mode_unsupported": "The selected intervention mode is not supported for this model.",
}


def decision(supported, code):
    return {"supported": bool(supported), "reason_code": code,
            "reason": PUBLIC_REASONS[code]}


def _safe_preflight(fn, unsupported_code="architecture_unsupported"):
    try:
        fn()
        return decision(True, "supported")
    except Exception:
        # Internal topology/probe details are useful in logs and deep execution
        # errors, but are intentionally not part of this stable public contract.
        return decision(False, unsupported_code)


def build_profile(jl, quant):
    """Compute intrinsic facts once per successful model load."""
    from core import rebase

    quantized = quant in ("int8", "nf4")
    packed = rebase.has_packed_read_parameters(jl)
    if quantized:
        blocked = decision(False, "quantized_model_unsupported")
        readthrough = exact = global_projection = blocked
    else:
        readthrough = _safe_preflight(lambda: rebase.model_preflight(jl, exact=False))
        exact_code = "packed_moe_exact_unsupported" if packed else "architecture_unsupported"
        exact = _safe_preflight(lambda: rebase.model_preflight(jl, exact=True), exact_code)
        global_projection = _safe_preflight(lambda: rebase.global_projection_preflight(jl))
    standard = (decision(False, "quantized_model_unsupported") if quantized
                else decision(True, "supported"))
    return {
        "model_session_id": uuid4().hex,
        "declared_quantization": quant,
        "has_packed_read_parameters": packed,
        "modes": {
            "standard": standard,
            "readthrough": readthrough,
            "exact": exact,
            "abliteration": global_projection,
        },
    }


def snapshot(profile, mode="standard"):
    if profile is None:
        modes = {name: decision(False, "no_model_loaded") for name in
                 ("standard", "readthrough", "exact", "abliteration")}
        return {
            "model_session_id": None,
            "modes": modes,
            "exports": {"mode": mode, **{fmt: decision(False, "no_model_loaded")
                        for fmt in ("full", "layers", "lora", "gguf")}},
        }

    modes = profile["modes"]
    mode_decision = modes.get(mode, decision(False, "mode_unsupported"))
    if mode == "standard":
        formats = {fmt: decision(False, "standard_not_exportable")
                   for fmt in ("full", "layers", "lora", "gguf")}
    elif not mode_decision["supported"]:
        formats = {fmt: dict(mode_decision) for fmt in ("full", "layers", "lora", "gguf")}
    else:
        formats = {fmt: decision(True, "supported")
                   for fmt in ("full", "layers", "lora", "gguf")}
        if profile["has_packed_read_parameters"]:
            formats["layers"] = decision(False, "packed_moe_format_unsupported")
            formats["lora"] = decision(False, "packed_moe_format_unsupported")
    return {"model_session_id": profile["model_session_id"], "modes": modes,
            "exports": {"mode": mode, **formats}}


def require(profile, kind, name, mode="standard"):
    contract = snapshot(profile, mode)
    items = contract["modes"] if kind == "modes" else contract["exports"]
    item = items.get(name, decision(False, "mode_unsupported"))
    if not item["supported"]:
        raise ValueError(item["reason"])
