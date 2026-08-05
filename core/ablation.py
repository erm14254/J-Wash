import copy
import itertools
import threading

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
                "id": rule["id"],
                "token_id": rule["token_id"],
                "token": rule["token"],
                "mode": rule["mode"],
                "factor": rule["factor"],
                "replacement_id": rule["replacement_id"],
                "replacement": rule["replacement"],
                "layers": list(rule["layers"]),
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

    def _direction(self, lens, weight, token_id, layers):
        row = weight[token_id].float()
        dirs = {}
        for layer in layers:
            J = lens.jacobians.get(layer)
            if J is None:
                # layer not fitted by the lens: direct logit lens (J = I),
                # a good approximation near the output
                v = row
            else:
                v = row @ J.float().to(weight.device)
            dirs[layer] = v / v.norm().clamp_min(1e-8)
        return dirs

    def add(self, lens_manager, jl, *, token_id, mode="scale", factor=0.0,
            replacement_id=None, layers=None, enabled=True):
        from core.capabilities import ensure_unquantized
        ensure_unquantized(jl)
        with self._lock:
            lens = lens_manager.lens
            if lens is None:
                raise ValueError("no lens loaded")
            if mode not in ("scale", "replace"):
                raise ValueError(f"invalid mode: {mode}")
            if mode == "replace" and replacement_id is None:
                raise ValueError("replacement_id required in replace mode")
            n_layers = len(jl.layers)
            if layers is None:
                layers = default_layers(n_layers)
            # layers=[] is valid: rule recorded but inactive
            layers = sorted({int(l) for l in layers if 0 <= int(l) < n_layers})
            weight = jl._lm_head.weight
            if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("interventions unavailable on a quantized model")
            tokenizer = jl.tokenizer
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
                "dirs_a": self._direction(lens, weight, int(token_id), layers),
                "dirs_b": self._direction(lens, weight, int(replacement_id), layers)
                if replacement_id is not None
                else None,
            }
            self._rules.append(rule)
            self._revision += 1
            return self._summary_locked()

    def update(self, rule_id, *, factor=None, layers=None, enabled=None,
               token_id=None, replacement_id=None, mode=None,
               lens_manager=None, jl=None):
        needs_dirs = any(x is not None for x in (layers, token_id, replacement_id, mode))
        if needs_dirs:
            from core.capabilities import ensure_unquantized
            ensure_unquantized(jl)
        with self._lock:
            for rule in self._rules:
                if rule["id"] != rule_id:
                    continue
                if factor is not None:
                    rule["factor"] = float(factor)
                if enabled is not None:
                    rule["enabled"] = bool(enabled)
                # token / replacement / mode / layers change the directions →
                # the lens and model are required to re-resolve them
                if not needs_dirs:
                    self._revision += 1
                    return self._summary_locked()
                if lens_manager is None or jl is None:
                    raise ValueError("model and lens required to edit the rule")
                lens = lens_manager.lens
                if lens is None:
                    raise ValueError("no lens loaded")
                tokenizer = jl.tokenizer
                if mode is not None:
                    if mode not in ("scale", "replace"):
                        raise ValueError(f"invalid mode: {mode}")
                    rule["mode"] = mode
                if token_id is not None:
                    rule["token_id"] = int(token_id)
                    rule["token"] = tokenizer.decode([int(token_id)])
                if replacement_id is not None:
                    rule["replacement_id"] = int(replacement_id)
                    rule["replacement"] = tokenizer.decode([int(replacement_id)])
                if rule["mode"] == "scale":
                    rule["replacement_id"] = None
                    rule["replacement"] = None
                elif rule["replacement_id"] is None:
                    raise ValueError("replacement_id required in replace mode")
                if layers is not None:
                    n_layers = len(jl.layers)
                    # new_layers=[] is valid: rule kept but inactive
                    rule["layers"] = sorted({int(l) for l in layers if 0 <= int(l) < n_layers})
                weight = jl._lm_head.weight
                rule["dirs_a"] = self._direction(lens, weight, rule["token_id"], rule["layers"])
                rule["dirs_b"] = (
                    self._direction(lens, weight, rule["replacement_id"], rule["layers"])
                    if rule["replacement_id"] is not None
                    else None
                )
                self._revision += 1
                return self._summary_locked()
            raise ValueError(f"unknown rule {rule_id}")

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
                        "id": rule["id"],
                        "token_id": rule["token_id"],
                        "token": rule["token"],
                        "mode": rule["mode"],
                        "factor": rule["factor"],
                        "replacement_id": rule["replacement_id"],
                        "replacement": rule["replacement"],
                        "layers": list(rule["layers"]),
                        "enabled": rule.get("enabled", True),
                    }
                    for rule in self._rules
                    if rule["layers"] and rule.get("enabled", True)
                ],
            }

    def attach(self, jl, *, snapshot=None):
        from core.capabilities import ensure_unquantized, PUBLIC_REASONS
        ensure_unquantized(jl)
        snap = copy.deepcopy(snapshot) if snapshot is not None else self.snapshot()
        mode = snap["mode"]
        if mode == "abliteration":
            raise ValueError(PUBLIC_REASONS["global_projection_unvalidated"])
        active = [r for r in snap["rules"] if r["layers"] and r.get("enabled", True)]
        if not active:
            return HookAttachment([])
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
        from core.capabilities import PUBLIC_REASONS
        raise ValueError(PUBLIC_REASONS["global_projection_unvalidated"])

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
