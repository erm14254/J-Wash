import copy
import itertools
import threading
from collections.abc import Mapping

import torch

# Default layer slice for a new rule, as fractions of the model's layer count:
# e.g. 56 layers -> from int(56*3/5)=33 to int(56*4/5)=44.
DEFAULT_LAYERS_FRAC_LO = 3 / 5
DEFAULT_LAYERS_FRAC_HI = 4 / 5


def default_layers(n_layers):
    lo = int(n_layers * DEFAULT_LAYERS_FRAC_LO)
    hi = min(int(n_layers * DEFAULT_LAYERS_FRAC_HI), n_layers - 1)
    return list(range(lo, hi + 1))


def effective_coeffs(mode, factor, g):
    """Effective coefficients ``(alpha, beta)`` of a rule's effect under the
    global multiplier ``g``: ``delta = alpha·(v̂_A·h)·v̂_A + beta·(v̂_A·h)·v̂_B``
    (``beta = 0`` in scale mode).

    Saturates the over-correction: at g=1 the effect is exactly that of the
    factor; beyond it, it converges to full removal of the component (or to the
    explicitly requested inversion if factor < 0) WITHOUT overshooting it.
    Without this bound, g·(factor-1) < -1 makes the component negative — a
    chaotic anti-direction (measured: "zap Paris" at scale 4 → "Paris Paris
    Paris..." in a loop).
    """
    if mode == "scale":
        alpha = g * (factor - 1.0)
        if factor < 1.0:
            # final component 1+alpha bounded to min(factor, 0)
            alpha = max(alpha, min(factor, 0.0) - 1.0)
        return alpha, 0.0
    # replace: saturated removal of A (never anti-A), addition of B linear in g
    return -min(g, 1.0), g * factor


def abliteration_direction(weight_u, rule):
    """Residual directions of a rule for the abliteration mode (global
    pure-weight edit).

    ``weight_u``: the un-embedding matrix W_U (lm_head), [vocab, d_model]. The
    directions live in the residual space (the basis W_U reads). Returns
    ``(v_a, v_b)`` (float, CPU, normalized); ``v_b`` is None in scale mode. The
    effect applied to each residual write ``h`` is
    ``h += alpha·(v̂_A·h)·v̂_A + beta·(v̂_A·h)·v̂_B`` with ``(alpha, beta)`` given
    by :func:`effective_coeffs` (which folds in the global scale).
    """
    v_a = weight_u[rule["token_id"]].detach().float().cpu()
    v_a = v_a / v_a.norm().clamp_min(1e-8)
    v_b = None
    if rule["mode"] != "scale":
        v_b = weight_u[rule["replacement_id"]].detach().float().cpu()
        v_b = v_b / v_b.norm().clamp_min(1e-8)
    return v_a, v_b


def clone_intervention_snapshot(snapshot):
    """Clone a coordinated intervention record for one operation.

    This intentionally avoids generic deepcopy of MappingProxyType records and
    preserves tensor identities in direction maps.
    """
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        snapshot = dict(snapshot)

    def clone_rule(rule):
        cloned = dict(rule)
        if cloned.get("layers") is not None:
            cloned["layers"] = list(cloned["layers"])
        if cloned.get("dirs_a") is not None:
            cloned["dirs_a"] = dict(cloned["dirs_a"])
        if cloned.get("dirs_b") is not None:
            cloned["dirs_b"] = dict(cloned["dirs_b"])
        return cloned

    def clone_summary_item(item):
        cloned = dict(item)
        if cloned.get("layers") is not None:
            cloned["layers"] = list(cloned["layers"])
        return cloned

    return {
        "revision": snapshot.get("revision"),
        "model_session_id": snapshot.get("model_session_id"),
        "lens_binding_id": snapshot.get("lens_binding_id"),
        "scale": snapshot.get("scale", 1.0),
        "mode": snapshot.get("mode", "standard"),
        "rules": [clone_rule(rule) for rule in snapshot.get("rules") or ()],
        "active_rules": [clone_rule(rule) for rule in snapshot.get("active_rules") or ()],
        "summary": [clone_summary_item(item) for item in snapshot.get("summary") or ()],
        "active_summary": [clone_summary_item(item) for item in snapshot.get("active_summary") or ()],
    }


# Rule application modes:
#   standard    — layer-by-layer residual steering (hook on the output of the
#                 chosen layers). The most expressive live, but no layer write
#                 carries the "skip": not faithfully exportable.
#   readthrough — change of basis of the downstream READS (cf. core/rebase):
#                 the preview hooks the RMSNorm output with the same transform
#                 as the bake → preview = exported checkpoint.
#   exact       — readthrough + counter-transform of the downstream writes
#                 (reproduces a hook applied exactly once; regularized inverse
#                 near a full zap → reserved for soft factors).
#   abliteration — global W_U projection on every residual write (embed + all
#                 block outputs); bake = the same projections on the writes.
#                 The pure-weights path for architectures the rebase does not
#                 support (write norms, Gemma style). Faithful for full
#                 zaps/replaces; a rule's layers are ignored (global).
MODES = ("standard", "readthrough", "exact", "abliteration")


class UnknownInterventionRule(ValueError):
    """A requested intervention rule does not exist."""


class HookAttachment:
    def __init__(self, handles):
        self._handles = list(handles)
        self._lock = threading.Lock()
        self._closed = False

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            handles, self._handles = self._handles, []
        first = None
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first


class Interventions:
    def __init__(self):
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._rules = []
        self._handles = []  # non-authoritative legacy inspection only
        self._scale = 1.0
        self._mode = "standard"
        self._revision = 0

    @property
    def active(self):
        with self._lock:
            return bool(self._rules)

    @property
    def global_scale(self):
        with self._lock:
            return self._scale

    @property
    def mode(self):
        with self._lock:
            return self._mode

    def _clone_rule_locked(self, rule):
        cloned = dict(rule)
        if "layers" in cloned:
            cloned["layers"] = list(cloned["layers"])
        if "dirs_a" in cloned and cloned["dirs_a"] is not None:
            cloned["dirs_a"] = dict(cloned["dirs_a"])
        if "dirs_b" in cloned and cloned["dirs_b"] is not None:
            cloned["dirs_b"] = dict(cloned["dirs_b"])
        return cloned

    def _active_rules_locked(self):
        return [self._clone_rule_locked(r) for r in self._rules if r["layers"] and r.get("enabled", True)]

    def _summary_locked(self):
        return [
            {
                "id": rule.get("id"),
                "token_id": rule.get("token_id"),
                "token": rule.get("token"),
                "mode": rule.get("mode"),
                "factor": rule.get("factor"),
                "replacement_id": rule.get("replacement_id"),
                "replacement": rule.get("replacement"),
                "layers": list(rule.get("layers") or []),
                "enabled": rule.get("enabled", True),
            }
            for rule in self._rules
        ]

    def set_scale(self, scale):
        with self._lock:
            self._scale = float(scale)
            self._revision += 1
            return self._scale

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"unknown intervention mode: {mode}")
        with self._lock:
            self._mode = mode
            self._revision += 1
            return self._mode

    def set_scale_and_mode(self, *, scale=None, mode=None):
        """Validate the complete patch before atomically changing either field."""
        if mode is not None and mode not in MODES:
            raise ValueError(f"unknown intervention mode: {mode}")
        new_scale = None if scale is None else float(scale)
        with self._lock:
            if new_scale is not None:
                self._scale = new_scale
            if mode is not None:
                self._mode = mode
            if new_scale is not None or mode is not None:
                self._revision += 1
            return self._scale, self._mode

    def state_snapshot(self):
        with self._lock:
            return self._scale, self._mode

    def state_record(self):
        with self._lock:
            return {
                "rules": [self._clone_rule_locked(r) for r in self._rules],
                "scale": self._scale,
                "mode": self._mode,
                "revision": self._revision,
            }

    def restore_state_record(self, record):
        with self._lock:
            self._rules = [self._clone_rule_locked(r) for r in record["rules"]]
            self._scale = record["scale"]
            self._mode = record["mode"]
            self._revision = record["revision"]

    def rules_full(self):
        with self._lock:
            return [self._clone_rule_locked(r) for r in self._rules]

    def clear_for_model_transition(self):
        with self._lock:
            self._rules = []
            self._mode = "standard"
            self._revision += 1
            return self._summary_locked()

    def active_rules_full(self):
        """Full rules (with directions) actually applied — for export: a disabled
        rule or one without layers must not be baked."""
        with self._lock:
            return self._active_rules_locked()

    def _active_rules(self):
        """Rules actually applied: non-empty layers AND not disabled. The
        `enabled` flag lets you switch a rule off without losing its layer
        selection (the "layers=[]" gesture stays possible but clears the selection)."""
        return [r for r in self._rules if r["layers"] and r.get("enabled", True)]

    def summary(self):
        with self._lock:
            return self._summary_locked()

    def _direction_source(self, lens_manager, jl, token_ids, layers):
        if lens_manager is None or lens_manager.lens is None:
            raise ValueError("no lens loaded")
        if jl is None:
            raise ValueError("model and lens required to construct directions")
        layers_obj = getattr(jl, "layers", None)
        if layers_obj is None:
            raise ValueError("model layers are unavailable for direction construction")
        try:
            n_layers = len(layers_obj)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("model layers are not a readable sequence") from exc
        head_module = getattr(jl, "_lm_head", None)
        head = getattr(head_module, "weight", None)
        if not isinstance(head, torch.Tensor) or head.ndim != 2:
            raise ValueError("lm_head weight must be a readable 2-D tensor")
        if not head.is_floating_point() or head.is_complex():
            raise ValueError("lm_head weight must support floating-point direction arithmetic")
        if head.layout != torch.strided:
            raise ValueError("lm_head weight must use a dense strided layout")
        if head.device.type == "meta":
            from core.rebase import materialize_parameter_for_inspection
            try:
                head = materialize_parameter_for_inspection(head_module, "weight")
            except ValueError as exc:
                raise ValueError(f"lm_head weight is not materializable: {exc}") from exc
            if not isinstance(head, torch.Tensor) or head.device.type == "meta":
                raise ValueError("lm_head weight could not be materialized")
            if head.ndim != 2 or not head.is_floating_point() or head.is_complex() \
                    or head.layout != torch.strided:
                raise ValueError("materialized lm_head weight is not a readable 2-D floating tensor")

        embed = getattr(getattr(jl, "_embed_tokens", None), "weight", None)
        if not isinstance(embed, torch.Tensor) or embed.ndim != 2:
            raise ValueError("model embedding weight is unavailable for residual-width validation")
        residual_width = int(embed.shape[1])
        if residual_width <= 0:
            raise ValueError("model residual width must be positive")
        if int(head.shape[1]) != residual_width:
            raise ValueError(
                f"lm_head input width {head.shape[1]} does not match model residual width "
                f"{residual_width}"
            )
        try:
            resolved = sorted({int(layer) for layer in layers})
        except (TypeError, ValueError) as exc:
            raise ValueError("requested layers must be an iterable of integers") from exc
        if any(layer < 0 or layer >= n_layers for layer in resolved):
            raise ValueError(f"requested layer must be between 0 and {max(n_layers - 1, 0)}")
        for token_id in token_ids:
            if token_id is None or int(token_id) < 0 or int(token_id) >= head.shape[0]:
                raise ValueError(f"token id {token_id} is outside lm_head vocabulary")
        lens = lens_manager.lens
        if not isinstance(getattr(lens, "jacobians", None), Mapping):
            raise ValueError("loaded lens has no readable Jacobian mapping")
        tokenizer = getattr(jl, "tokenizer", None)
        if not callable(getattr(tokenizer, "decode", None)):
            raise ValueError("model tokenizer is unavailable for intervention labels")
        return lens, head, resolved, residual_width

    def _direction(self, lens, weight, token_id, layers, residual_width):
        try:
            row = weight[token_id].detach().float()
        except (IndexError, RuntimeError, TypeError) as exc:
            raise ValueError(f"lm_head row for token {token_id} is unreadable: {exc}") from exc
        if row.layout != torch.strided or row.device.type == "meta":
            raise ValueError(f"lm_head row for token {token_id} is not readable")
        if row.ndim != 1 or row.shape[0] != residual_width:
            raise ValueError("lm_head row does not match model residual width")
        if not torch.isfinite(row).all():
            raise ValueError(f"lm_head row for token {token_id} contains non-finite values")
        dirs = {}
        for layer in layers:
            J = lens.jacobians.get(layer)
            if J is None:
                # layer not fitted by the lens: direct logit lens (J = I),
                # a good approximation near the output
                v = row
            else:
                if not isinstance(J, torch.Tensor):
                    raise ValueError(f"Jacobian at layer {layer} must be a tensor")
                if J.ndim != 2:
                    raise ValueError(f"Jacobian at layer {layer} must be 2-D")
                if J.layout != torch.strided or J.device.type == "meta":
                    raise ValueError(f"Jacobian at layer {layer} is not readable")
                if J.shape[0] != row.shape[0]:
                    raise ValueError(
                        f"Jacobian input width at layer {layer} does not match lm_head width"
                    )
                if J.shape[1] != residual_width:
                    raise ValueError(
                        f"Jacobian output width at layer {layer} does not match model residual width"
                    )
                if not (J.is_floating_point() and not J.is_complex()):
                    raise ValueError(f"Jacobian at layer {layer} must use real floating-point arithmetic")
                if not torch.isfinite(J).all():
                    raise ValueError(f"Jacobian at layer {layer} contains non-finite values")
                try:
                    v = row @ J.to(device=row.device, dtype=row.dtype)
                except (RuntimeError, TypeError) as exc:
                    raise ValueError(f"Jacobian at layer {layer} cannot construct a direction: {exc}") from exc
            if v.ndim != 1 or v.shape[0] != residual_width:
                raise ValueError(f"direction width at layer {layer} does not match model residual width")
            if not torch.isfinite(v).all():
                raise ValueError(f"direction at layer {layer} contains non-finite values")
            norm = v.norm()
            if not torch.isfinite(norm) or norm.item() <= 1e-8:
                raise ValueError(f"direction at layer {layer} cannot be normalized")
            dirs[layer] = v / norm
        return dirs

    def add(self, lens_manager, jl, *, token_id, mode="scale", factor=0.0,
            replacement_id=None, layers=None, enabled=True):
        with self._lock:
            if mode not in ("scale", "replace"):
                raise ValueError(f"invalid mode: {mode}")
            if mode == "replace" and replacement_id is None:
                raise ValueError("replacement_id required in replace mode")
            if layers is None:
                layers_obj = getattr(jl, "layers", None)
                try:
                    layers = default_layers(len(layers_obj))
                except (TypeError, AttributeError, RuntimeError) as exc:
                    raise ValueError("model layers are unavailable for direction construction") from exc
            # layers=[] is valid: rule recorded but inactive
            token_ids = [token_id] + ([replacement_id] if replacement_id is not None else [])
            lens, weight, layers, residual_width = self._direction_source(
                lens_manager, jl, token_ids, layers
            )
            tokenizer = jl.tokenizer
            dirs_a = self._direction(lens, weight, int(token_id), layers, residual_width)
            dirs_b = (
                self._direction(lens, weight, int(replacement_id), layers, residual_width)
                if replacement_id is not None else None
            )
            rule = {
                "id": next(self._counter),
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "mode": mode,
                "factor": float(factor),
                "replacement_id": int(replacement_id) if replacement_id is not None else None,
                "replacement": tokenizer.decode([int(replacement_id)]) if replacement_id is not None else None,
                "layers": [int(l) for l in layers],
                "enabled": bool(enabled),
                "dirs_a": dirs_a,
                "dirs_b": dirs_b,
            }
            self._rules.append(rule)
            self._revision += 1
            return self._summary_locked()

    def update(self, rule_id, *, factor=None, layers=None, enabled=None,
               token_id=None, replacement_id=None, mode=None,
               lens_manager=None, jl=None):
        needs_dirs = any(x is not None for x in (layers, token_id, replacement_id, mode))
        with self._lock:
            for index, rule in enumerate(self._rules):
                if rule["id"] != rule_id:
                    continue
                candidate = self._clone_rule_locked(rule)
                if factor is not None:
                    candidate["factor"] = float(factor)
                if enabled is not None:
                    candidate["enabled"] = bool(enabled)
                # token / replacement / mode / layers change the directions →
                # the lens and model are required to re-resolve them
                if not needs_dirs:
                    self._rules[index] = candidate
                    self._revision += 1
                    return self._summary_locked()
                # Establish existence under the same lock before concrete
                # direction prerequisites can mask an unknown-rule response.
                # Validation and commit remain one locked transaction, so no
                # concurrent mutation can invalidate the cloned candidate.
                if lens_manager is None or jl is None:
                    raise ValueError("model and lens required to edit the rule")
                lens = lens_manager.lens
                if lens is None:
                    raise ValueError("no lens loaded")
                tokenizer = jl.tokenizer
                if mode is not None:
                    if mode not in ("scale", "replace"):
                        raise ValueError(f"invalid mode: {mode}")
                    candidate["mode"] = mode
                if token_id is not None:
                    candidate["token_id"] = int(token_id)
                    candidate["token"] = tokenizer.decode([int(token_id)])
                if replacement_id is not None:
                    candidate["replacement_id"] = int(replacement_id)
                    candidate["replacement"] = tokenizer.decode([int(replacement_id)])
                if candidate["mode"] == "scale":
                    candidate["replacement_id"] = None
                    candidate["replacement"] = None
                elif candidate["replacement_id"] is None:
                    raise ValueError("replacement_id required in replace mode")
                if layers is not None:
                    candidate["layers"] = list(layers)
                token_ids = [candidate["token_id"]]
                if candidate["replacement_id"] is not None:
                    token_ids.append(candidate["replacement_id"])
                lens, weight, candidate["layers"], residual_width = self._direction_source(
                    lens_manager, jl, token_ids, candidate["layers"]
                )
                dirs_a = self._direction(
                    lens, weight, candidate["token_id"], candidate["layers"], residual_width
                )
                dirs_b = (
                    self._direction(
                        lens, weight, candidate["replacement_id"], candidate["layers"],
                        residual_width,
                    )
                    if candidate["replacement_id"] is not None
                    else None
                )
                candidate["dirs_a"] = dirs_a
                candidate["dirs_b"] = dirs_b
                self._rules[index] = candidate
                self._revision += 1
                return self._summary_locked()
            raise UnknownInterventionRule(f"unknown rule {rule_id}")

    def remove(self, rule_id=None):
        with self._lock:
            if rule_id is None:
                self._rules = []
            else:
                self._rules = [r for r in self._rules if r["id"] != rule_id]
            self._revision += 1
            return self._summary_locked()

    def snapshot(self):
        with self._lock:
            return {
                "revision": self._revision,
                "scale": self._scale,
                "mode": self._mode,
                "rules": [self._clone_rule_locked(r) for r in self._rules],
                "active_rules": self._active_rules_locked(),
                "summary": self._summary_locked(),
                "active_summary": [
                    {
                        "id": rule.get("id"),
                        "token_id": rule.get("token_id"),
                        "token": rule.get("token"),
                        "mode": rule.get("mode"),
                        "factor": rule.get("factor"),
                        "replacement_id": rule.get("replacement_id"),
                        "replacement": rule.get("replacement"),
                        "layers": list(rule.get("layers") or []),
                        "enabled": rule.get("enabled", True),
                    }
                    for rule in self._rules
                    if rule["layers"] and rule.get("enabled", True)
                ],
            }

    def attach(self, jl, *, snapshot=None):
        snap = clone_intervention_snapshot(snapshot) if snapshot is not None else self.snapshot()
        mode = snap["mode"]
        active = [r for r in snap["rules"] if r["layers"] and r.get("enabled", True)]
        if not active:
            return HookAttachment([])
        if mode == "abliteration":
            return self._attach_abliteration(jl)
        if mode in ("readthrough", "exact"):
            return self._attach_rebase(jl, exact=mode == "exact", active=active, scale=snap["scale"])
        by_layer = {}
        for rule in active:
            for layer in rule["layers"]:
                by_layer.setdefault(layer, []).append(dict(rule))
        scale = snap["scale"]

        def make_hook(layer, rules):
            rules = [dict(r) for r in rules]
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                for rule in rules:
                    alpha, beta = effective_coeffs(rule["mode"], rule["factor"], scale)
                    vA = rule["dirs_a"][layer].to(h.device, h.dtype)
                    coef = (h * vA).sum(-1, keepdim=True)
                    h = h + alpha * coef * vA
                    if beta:
                        vB = rule["dirs_b"][layer].to(h.device, h.dtype)
                        h = h + beta * coef * vB
                if isinstance(output, tuple):
                    return (h,) + tuple(output[1:])
                return h
            return hook

        handles = []
        try:
            for layer, rules in by_layer.items():
                handles.append(jl.layers[layer].register_forward_hook(make_hook(layer, rules)))
        except Exception:
            HookAttachment(handles).close()
            raise
        return HookAttachment(handles)

    def _attach_abliteration(self, jl):
        raise ValueError("Global Projection live attachment is not implemented")

    def _attach_rebase(self, jl, exact, active=None, scale=None):
        # readthrough/exact preview: the SAME transform as the bake (core/rebase),
        # applied by hooks on the OUTPUT of the reading RMSNorms (and, in exact
        # mode, on the downstream writes) — the preview and the exported
        # checkpoint differ only by rounding.
        from core import rebase  # local import (rebase imports effective_coeffs from here)

        active = active if active is not None else self._active_rules()
        scale = self._scale if scale is None else scale
        if not active:
            return HookAttachment([])
        n_layers = len(jl.layers)
        # Model-wide Phase-1 contract: capability does not depend on which rule
        # happens to be last.  This also validates the final norm before any
        # hook registration is attempted.
        inventory = rebase.model_preflight(jl, exact=exact)
        cums = rebase.cumulative(active, scale, n_layers)
        if not cums:
            return HookAttachment([])

        def read_hook_for(norm, U, V):
            Ug, Vg = rebase.gamma_pair(norm, U, V)

            def hook(module, inputs, output):
                residual = output[0] if isinstance(output, tuple) else output
                ug = Ug.to(device=residual.device, dtype=residual.dtype)
                vg = Vg.to(device=residual.device, dtype=residual.dtype)
                changed = residual + (residual @ vg) @ ug.T
                return ((changed,) + tuple(output[1:])
                        if isinstance(output, tuple) else changed)

            return hook

        def write_hook_for(module, U_inv, V):
            def hook(module, inputs, output):
                if not isinstance(output, torch.Tensor):
                    raise TypeError("exact residual writers must return torch.Tensor")
                u_inv = U_inv.to(device=output.device, dtype=output.dtype)
                v = V.to(device=output.device, dtype=output.dtype)
                return output - (output @ v) @ u_inv.T

            return hook

        # Resolve the entire hook plan transactionally before registering any
        # hook.  Unknown topology, missing writers, or an invalid final norm can
        # therefore never leave a partially attached intervention.
        sites = []
        for m in sorted(k for k in cums if k < n_layers):
            U, V = cums[m]
            block = jl.layers[m]
            block_inventory = inventory.blocks[m]
            norms = {}
            for _suffix, _module, norm in block_inventory.reads:
                norms[id(norm)] = norm
            for norm in norms.values():
                sites.append((norm, read_hook_for(norm, U, V)))
            if exact:
                U_inv, Vw, _regularized = rebase.inverse_uv(U, V)
                for _suffix, module in block_inventory.writes:
                    sites.append((module, write_hook_for(module, U_inv, Vw)))
        U, V = cums[n_layers]
        sites.append((inventory.final_norm,
                      read_hook_for(inventory.final_norm, U, V)))
        handles = []
        try:
            for module, hook in sites:
                handles.append(module.register_forward_hook(hook))
        except Exception:
            HookAttachment(handles).close()
            raise
        return HookAttachment(handles)

    def detach(self):
        # Compatibility shim: operation-local attachments are authoritative.
        return None
