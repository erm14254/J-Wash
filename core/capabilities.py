"""Advisory validation diagnostics for the loaded model.

``supported`` means positively validated by the current audit.  A false value
is a warning, not authorization to deny an operation; concrete implementations
remain responsible for deciding whether an attempted operation can proceed.
"""

PUBLIC_REASONS = {
    "supported": "Supported for this model.",
    "no_model_loaded": "Load a model to use this operation.",
    "capability_data_unavailable": "Capability data is unavailable for this model.",
    "quantized_model_unsupported": "Editing quantized model loads is not supported.",
    "architecture_unsupported": "This model architecture has not been validated for this operation.",
    "packed_moe_exact_unsupported": "Exact mode is not supported for packed MoE models.",
    "packed_moe_format_unsupported": "This export format is not supported for packed MoE models.",
    "standard_not_exportable": "Standard interventions cannot be exported faithfully.",
    "mode_unsupported": "The selected intervention mode is not supported for this model.",
    "global_projection_unvalidated": (
        "Global projection is temporarily unavailable because no topology and "
        "export contract has been positively validated."
    ),
}

MODES = ("standard", "readthrough", "exact", "abliteration")
FORMATS = ("full", "layers", "lora", "gguf")
QUANTS = ("int8", "nf4")


def decision(supported, code):
    """Create a decision only from exact, known values."""
    if type(supported) is not bool or code not in PUBLIC_REASONS:
        raise ValueError("invalid capability decision")
    return {"supported": supported, "reason_code": code, "reason": PUBLIC_REASONS[code]}


def _blocked_modes(code):
    return {name: decision(False, code) for name in MODES}


def _normalize_decision(value):
    if not isinstance(value, dict) or set(value) != {"supported", "reason_code", "reason"}:
        raise ValueError("malformed capability decision")
    supported, code = value["supported"], value["reason_code"]
    if type(supported) is not bool or code not in PUBLIC_REASONS:
        raise ValueError("malformed capability decision")
    if value["reason"] != PUBLIC_REASONS[code]:
        raise ValueError("noncanonical capability reason")
    if supported != (code == "supported"):
        raise ValueError("contradictory capability decision")
    return decision(supported, code)


def normalize_profile(profile):
    """Return an isolated valid profile, or ``None`` for any malformed input."""
    try:
        if not isinstance(profile, dict):
            raise ValueError
        if set(profile) != {"declared_quantization", "has_packed_read_parameters", "modes"}:
            raise ValueError
        quant = profile["declared_quantization"]
        packed = profile["has_packed_read_parameters"]
        modes = profile["modes"]
        if quant not in (None, *QUANTS) or type(packed) is not bool:
            raise ValueError
        if not isinstance(modes, dict) or set(modes) != set(MODES):
            raise ValueError
        normalized = {name: _normalize_decision(modes[name]) for name in MODES}
        if normalized["abliteration"] != decision(False, "global_projection_unvalidated"):
            raise ValueError
        if quant in QUANTS and any(
                item != decision(False, "quantized_model_unsupported")
                for name, item in normalized.items() if name != "abliteration"):
            raise ValueError
        if quant is None and normalized["standard"] != decision(True, "supported"):
            raise ValueError
        if quant is None:
            readthrough, exact = normalized["readthrough"], normalized["exact"]
            if readthrough not in (decision(True, "supported"),
                                    decision(False, "architecture_unsupported")):
                raise ValueError
            valid_exact = (decision(True, "supported"),
                           decision(False, "architecture_unsupported"),
                           decision(False, "packed_moe_exact_unsupported"))
            if exact not in valid_exact:
                raise ValueError
            if not readthrough["supported"] and exact != decision(
                    False, "architecture_unsupported"):
                raise ValueError
            if packed and (not readthrough["supported"] or exact != decision(
                    False, "packed_moe_exact_unsupported")):
                raise ValueError
            if not packed and exact == decision(False, "packed_moe_exact_unsupported"):
                raise ValueError
            if exact["supported"] and (not readthrough["supported"] or packed):
                raise ValueError
        return {"declared_quantization": quant,
                "has_packed_read_parameters": packed, "modes": normalized}
    except (KeyError, TypeError, ValueError):
        return None


def _safe_preflight(fn, unsupported_code="architecture_unsupported"):
    try:
        fn()
        return decision(True, "supported")
    except Exception:
        return decision(False, unsupported_code)


def build_profile(jl, quant):
    """Compute intrinsic facts once per successful model load."""
    from core import rebase

    jl_quant = getattr(jl, "_jwash_declared_quant", None)
    if quant not in (None, *QUANTS) or jl_quant not in (None, *QUANTS):
        return None
    if jl_quant != quant and not (jl_quant is None and quant is None):
        return None
    quantized = quant in QUANTS
    if quantized:
        packed = False
        modes = {name: decision(False, "quantized_model_unsupported")
                 for name in ("standard", "readthrough", "exact")}
    else:
        try:
            inventory = rebase.model_inventory(jl)
        except Exception:
            inventory = None
        readthrough = decision(inventory is not None,
                               "supported" if inventory is not None else
                               "architecture_unsupported")
        packed = inventory.packed if inventory is not None else False
        if packed:
            exact = decision(False, "packed_moe_exact_unsupported")
        elif inventory is not None and inventory.exact_supported:
            exact = decision(True, "supported")
        else:
            exact = decision(False, "architecture_unsupported")
        modes = {"standard": decision(True, "supported"),
                 "readthrough": readthrough, "exact": exact}
    modes["abliteration"] = decision(False, "global_projection_unvalidated")
    return {"declared_quantization": quant,
            "has_packed_read_parameters": packed, "modes": modes}


def snapshot(profile, mode="standard", *, loaded=False):
    normalized = normalize_profile(profile)
    if normalized is None:
        code = "capability_data_unavailable" if loaded else "no_model_loaded"
        modes = _blocked_modes(code)
        formats = {fmt: decision(False, code) for fmt in FORMATS}
        return {"modes": modes, "exports": {"mode": mode, **formats}}
    modes = normalized["modes"]
    selected = modes.get(mode, decision(False, "mode_unsupported"))
    if mode == "standard":
        formats = {fmt: decision(False, "standard_not_exportable") for fmt in FORMATS}
    elif not selected["supported"]:
        formats = {fmt: decision(False, selected["reason_code"]) for fmt in FORMATS}
    else:
        formats = {fmt: decision(True, "supported") for fmt in FORMATS}
        if normalized["has_packed_read_parameters"]:
            formats["layers"] = decision(False, "packed_moe_format_unsupported")
            formats["lora"] = decision(False, "packed_moe_format_unsupported")
    return {"modes": modes, "exports": {"mode": mode, **formats}}


def legacy(profile, *, loaded=False):
    modes = snapshot(profile, loaded=loaded)["modes"]
    readthrough, exact = modes["readthrough"], modes["exact"]
    return {"rebase_supported": readthrough["supported"],
            "readthrough_supported": readthrough["supported"],
            "readthrough_reason": readthrough["reason"],
            "exact_supported": exact["supported"], "exact_reason": exact["reason"]}
