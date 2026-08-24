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


def effective_coeffs(mode, factor, g, keep=0.0):
    """Effective coefficients ``(alpha, beta)`` of a rule's effect under the
    global multiplier ``g``: ``delta = alpha·(v̂_A·h)·v̂_A + beta·(v̂_A·h)·v̂_B``
    (``beta = 0`` in scale mode).

    Saturates the over-correction: at g=1 the effect is exactly that of the
    factor; beyond it, it converges to full removal of the component (or to the
    explicitly requested inversion if factor < 0) WITHOUT overshooting it.
    Without this bound, g·(factor-1) < -1 makes the component negative — a
    chaotic anti-direction (measured: "zap Paris" at scale 4 → "Paris Paris
    Paris..." in a loop).

    ``keep`` (replace mode only, 0..1): fraction of the ORIGINAL component that
    survives the replacement. The historical behavior (keep=0) removes A
    entirely; keep=0.3 leaves 30 % of A in place while still adding B — a
    partial replace that preserves the word's normal usability.
    """
    if mode == "scale":
        alpha = g * (factor - 1.0)
        if factor < 1.0:
            # final component 1+alpha bounded to min(factor, 0)
            alpha = max(alpha, min(factor, 0.0) - 1.0)
        return alpha, 0.0
    # replace: saturated removal of A (never anti-A), addition of B linear in g
    keep = min(max(float(keep), 0.0), 1.0)
    return -min(g, 1.0) * (1.0 - keep), g * factor


def rule_token_ids(rule, side="a"):
    """A rule's token ids as a list — ``token_ids`` when present (multi-token
    composite), else the legacy single ``token_id`` (old presets)."""
    if side == "a":
        return rule.get("token_ids") or [rule["token_id"]]
    return rule.get("replacement_ids") or [rule["replacement_id"]]


def _composite_unembed(weight_u, token_ids, weights=None):
    """Normalized (weighted) mean direction of several W_U rows (each row
    normalized first, so a high-norm piece does not dominate). One id → the
    plain normalized row, identical to the single-token behavior."""
    rows = weight_u[list(token_ids)].detach().float().cpu()
    rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if weights is not None:
        rows = rows * torch.tensor([float(x) for x in weights]).unsqueeze(1)
    v = rows.sum(0)
    return v / v.norm().clamp_min(1e-8)


# Replacement-side piece weighting: geometric decay of the pieces after the
# first. The FIRST piece must dominate — it is the token that has to win the
# next-token race for the word to start appearing at all — while the later
# pieces steer the continuation (measured live: a uniform mean buries the
# first piece and the model dodges the word entirely).
ANCHOR_DECAY_DEFAULT = 0.5


def piece_weights(n, decay):
    """[1, decay, decay², …] — uniform when decay=1."""
    return [decay ** i for i in range(n)]


def composite_directions(lens, weight, token_ids, layers, weights=None):
    """Per-layer J-space direction of a (possibly multi-token) word:
    normalized weighted mean of the pieces' directions, each piece normalized
    first (default uniform). One id → identical to the single-token behavior."""
    rows = weight[list(token_ids)].float()
    w = None
    if weights is not None:
        w = torch.tensor([float(x) for x in weights],
                         device=weight.device).unsqueeze(1)
    dirs = {}
    for layer in layers:
        J = lens.jacobians.get(layer)
        if J is None:
            # layer not fitted by the lens: direct logit lens (J = I),
            # a good approximation near the output
            v = rows
        else:
            v = rows @ J.float().to(weight.device)
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v = (v * w).sum(0) if w is not None else v.sum(0)
        dirs[layer] = v / v.norm().clamp_min(1e-8)
    return dirs


def abliteration_direction(weight_u, rule):
    """Residual directions of a rule for the abliteration mode (global
    pure-weight edit).

    ``weight_u``: the un-embedding matrix W_U (lm_head), [vocab, d_model]. The
    directions live in the residual space (the basis W_U reads). Returns
    ``(v_a, v_b)`` (float, CPU, normalized); ``v_b`` is None in scale mode. The
    effect applied to each residual write ``h`` is
    ``h += alpha·(v̂_A·h)·v̂_A + beta·(v̂_A·h)·v̂_B`` with ``(alpha, beta)`` given
    by :func:`effective_coeffs` (which folds in the global scale). Multi-token
    rules use the composite (normalized-mean) direction of their pieces.
    """
    v_a = _composite_unembed(weight_u, rule_token_ids(rule, "a"))
    v_b = None
    if rule["mode"] != "scale":
        ids_b = rule_token_ids(rule, "b")
        decay = rule.get("anchor_decay", ANCHOR_DECAY_DEFAULT)
        v_b = _composite_unembed(weight_u, ids_b, piece_weights(len(ids_b), decay))
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


class Interventions:
    def __init__(self):
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._rules = []
        self._handles = []
        self._scale = 1.0
        self._mode = "standard"
        self._preserve_lm_head = False

    @property
    def active(self):
        return bool(self._rules)

    @property
    def global_scale(self):
        return self._scale

    @property
    def mode(self):
        return self._mode

    @property
    def preserve_lm_head(self):
        return self._preserve_lm_head

    def set_preserve_lm_head(self, value):
        # readthrough/exact only: leaves the final read (lm_head) untransformed,
        # so the output vocabulary is untouched — the edit acts only through the
        # downstream layers' reads. Ignored by standard steering and abliteration.
        with self._lock:
            self._preserve_lm_head = bool(value)
            return self._preserve_lm_head

    def set_scale(self, scale):
        with self._lock:
            self._scale = float(scale)
            return self._scale

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError(f"unknown intervention mode: {mode}")
        with self._lock:
            self._mode = mode
            return self._mode

    def rules_full(self):
        return list(self._rules)

    def active_rules_full(self):
        """Full rules (with directions) actually applied — for export: a disabled
        rule or one without layers must not be baked."""
        return list(self._active_rules())

    def _active_rules(self):
        """Rules actually applied: non-empty layers AND not disabled. The
        `enabled` flag lets you switch a rule off without losing its layer
        selection (the "layers=[]" gesture stays possible but clears the selection)."""
        return [r for r in self._rules if r["layers"] and r.get("enabled", True)]

    def summary(self):
        return [
            {
                "id": rule["id"],
                "token_id": rule["token_id"],
                "token_ids": rule["token_ids"],
                "token": rule["token"],
                "token_pieces": rule["token_pieces"],
                "mode": rule["mode"],
                "factor": rule["factor"],
                "keep": rule.get("keep", 0.0),
                "replacement_id": rule["replacement_id"],
                "replacement_ids": rule["replacement_ids"],
                "replacement": rule["replacement"],
                "replacement_pieces": rule["replacement_pieces"],
                "layers": rule["layers"],
                "enabled": rule.get("enabled", True),
                "anchor_decay": rule.get("anchor_decay", ANCHOR_DECAY_DEFAULT),
            }
            for rule in self._rules
        ]

    @staticmethod
    def _clean_ids(weight, token_ids, token_id, what):
        """Canonical id list from either parameter (list wins); None = absent."""
        ids = token_ids if token_ids is not None else (
            [token_id] if token_id is not None else None
        )
        if ids is None:
            return None
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError(f"empty token sequence for {what}")
        vocab = weight.shape[0]
        if any(t < 0 or t >= vocab for t in ids):
            raise ValueError(f"{what}: token id out of range (vocab {vocab})")
        return ids

    def _direction(self, lens, weight, token_ids, layers):
        return composite_directions(lens, weight, token_ids, layers)

    def add(self, lens_manager, jl, *, token_id=None, token_ids=None,
            mode="scale", factor=0.0, keep=0.0, replacement_id=None,
            replacement_ids=None, layers=None, enabled=True, anchor_decay=None):
        with self._lock:
            lens = lens_manager.lens
            if lens is None:
                raise ValueError("no lens loaded")
            if mode not in ("scale", "replace"):
                raise ValueError(f"invalid mode: {mode}")
            n_layers = len(jl.layers)
            if layers is None:
                layers = default_layers(n_layers)
            # layers=[] is valid: rule recorded but inactive
            layers = sorted({int(l) for l in layers if 0 <= int(l) < n_layers})
            weight = jl._lm_head.weight
            if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError("interventions unavailable on a quantized model")
            ids_a = self._clean_ids(weight, token_ids, token_id, "token")
            if ids_a is None:
                raise ValueError("token_id or token_ids required")
            ids_b = self._clean_ids(weight, replacement_ids, replacement_id, "replacement")
            if mode == "replace" and ids_b is None:
                raise ValueError("replacement_id required in replace mode")
            if mode == "scale":
                ids_b = None
            decay = ANCHOR_DECAY_DEFAULT if anchor_decay is None else float(anchor_decay)
            decay = min(max(decay, 0.0), 1.0)
            tokenizer = jl.tokenizer
            rule = {
                "id": next(self._counter),
                "token_id": ids_a[0],
                "token_ids": ids_a,
                "token": tokenizer.decode(ids_a),
                "token_pieces": [tokenizer.decode([t]) for t in ids_a],
                "mode": mode,
                "factor": float(factor),
                "keep": min(max(float(keep), 0.0), 1.0),
                "replacement_id": ids_b[0] if ids_b else None,
                "replacement_ids": ids_b,
                "replacement": tokenizer.decode(ids_b) if ids_b else None,
                "replacement_pieces": [tokenizer.decode([t]) for t in ids_b] if ids_b else None,
                "layers": [int(l) for l in layers],
                "enabled": bool(enabled),
                "anchor_decay": decay,
                "dirs_a": self._direction(lens, weight, ids_a, layers),
                "dirs_b": composite_directions(
                    lens, weight, ids_b, layers, piece_weights(len(ids_b), decay))
                if ids_b else None,
            }
            self._rules.append(rule)
            return self.summary()

    def update(self, rule_id, *, factor=None, keep=None, layers=None,
               enabled=None, token_id=None, token_ids=None, replacement_id=None,
               replacement_ids=None, mode=None, anchor_decay=None,
               lens_manager=None, jl=None):
        with self._lock:
            for rule in self._rules:
                if rule["id"] != rule_id:
                    continue
                if factor is not None:
                    rule["factor"] = float(factor)
                if keep is not None:
                    rule["keep"] = min(max(float(keep), 0.0), 1.0)
                if enabled is not None:
                    rule["enabled"] = bool(enabled)
                # token / replacement / mode / layers change the directions →
                # the lens and model are required to re-resolve them
                needs_dirs = any(x is not None for x in (
                    layers, token_id, token_ids, replacement_id, replacement_ids,
                    mode, anchor_decay
                ))
                if not needs_dirs:
                    return self.summary()
                if lens_manager is None or jl is None:
                    raise ValueError("model and lens required to edit the rule")
                lens = lens_manager.lens
                if lens is None:
                    raise ValueError("no lens loaded")
                tokenizer = jl.tokenizer
                weight = jl._lm_head.weight
                if mode is not None:
                    if mode not in ("scale", "replace"):
                        raise ValueError(f"invalid mode: {mode}")
                    rule["mode"] = mode
                ids_a = self._clean_ids(weight, token_ids, token_id, "token")
                if ids_a is not None:
                    rule["token_id"] = ids_a[0]
                    rule["token_ids"] = ids_a
                    rule["token"] = tokenizer.decode(ids_a)
                    rule["token_pieces"] = [tokenizer.decode([t]) for t in ids_a]
                ids_b = self._clean_ids(weight, replacement_ids, replacement_id, "replacement")
                if ids_b is not None:
                    rule["replacement_id"] = ids_b[0]
                    rule["replacement_ids"] = ids_b
                    rule["replacement"] = tokenizer.decode(ids_b)
                    rule["replacement_pieces"] = [tokenizer.decode([t]) for t in ids_b]
                if rule["mode"] == "scale":
                    rule["replacement_id"] = None
                    rule["replacement_ids"] = None
                    rule["replacement"] = None
                    rule["replacement_pieces"] = None
                elif rule["replacement_ids"] is None:
                    raise ValueError("replacement_id required in replace mode")
                if layers is not None:
                    n_layers = len(jl.layers)
                    # new_layers=[] is valid: rule kept but inactive
                    rule["layers"] = sorted({int(l) for l in layers if 0 <= int(l) < n_layers})
                if anchor_decay is not None:
                    rule["anchor_decay"] = min(max(float(anchor_decay), 0.0), 1.0)
                decay = rule.get("anchor_decay", ANCHOR_DECAY_DEFAULT)
                rule["dirs_a"] = self._direction(lens, weight, rule["token_ids"], rule["layers"])
                rule["dirs_b"] = (
                    composite_directions(
                        lens, weight, rule["replacement_ids"], rule["layers"],
                        piece_weights(len(rule["replacement_ids"]), decay))
                    if rule["replacement_ids"] is not None
                    else None
                )
                return self.summary()
            raise ValueError(f"unknown rule {rule_id}")

    def remove(self, rule_id=None):
        with self._lock:
            self.detach()
            if rule_id is None:
                self._rules = []
            else:
                self._rules = [r for r in self._rules if r["id"] != rule_id]
            return self.summary()

    def attach(self, jl):
        if not self._rules:
            return
        if self._mode == "abliteration":
            self._attach_abliteration(jl)
            return
        if self._mode in ("readthrough", "exact"):
            self._attach_rebase(jl, exact=self._mode == "exact")
            return
        by_layer = {}
        for rule in self._active_rules():
            for layer in rule["layers"]:
                by_layer.setdefault(layer, []).append(rule)

        def make_hook(layer, rules):
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                g = self._scale
                for rule in rules:
                    alpha, beta = effective_coeffs(
                        rule["mode"], rule["factor"], g, rule.get("keep", 0.0))
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

        self._handles = [
            jl.layers[layer].register_forward_hook(make_hook(layer, rules))
            for layer, rules in by_layer.items()
        ]

    def _attach_abliteration(self, jl):
        # Abliteration-mode preview: the SAME projection on every residual write
        # (embed + each block's output), mirroring the pure-weight bake. A rule's
        # layers make no sense here (global projection), but layers=[] stays THE
        # "rule disabled" gesture: we honor it too.
        active = self._active_rules()
        if not active:
            return
        weight_u = jl._lm_head.weight
        dirs = [(abliteration_direction(weight_u, r), r) for r in active]

        def apply(h):
            g = self._scale
            for (v_a, v_b), rule in dirs:
                alpha, beta = effective_coeffs(
                    rule["mode"], rule["factor"], g, rule.get("keep", 0.0))
                va = v_a.to(h.device, h.dtype)
                coef = (h * va).sum(-1, keepdim=True)
                h = h + alpha * coef * va
                if beta:
                    h = h + beta * coef * v_b.to(h.device, h.dtype)
            return h

        def emb_hook(module, inputs, output):
            return apply(output)

        def blk_hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = apply(h)
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

        self._handles = [jl._embed_tokens.register_forward_hook(emb_hook)]
        self._handles += [blk.register_forward_hook(blk_hook) for blk in jl.layers]

    def _attach_rebase(self, jl, exact):
        # readthrough/exact preview: the SAME transform as the bake (core/rebase),
        # applied by hooks on the OUTPUT of the reading RMSNorms (and, in exact
        # mode, on the downstream writes) — the preview and the exported
        # checkpoint differ only by rounding.
        from core import rebase  # local import (rebase imports effective_coeffs from here)

        active = self._active_rules()
        if not active:
            return
        n_layers = len(jl.layers)
        cums = rebase.cumulative(active, self._scale, n_layers)
        if not cums:
            return
        if exact and rebase.has_packed_moe(jl):
            raise ValueError("Exact is not implemented for packed MoE; use Readthrough")

        def read_hook_for(norm, U, V):
            Ug, Vg = rebase.gamma_pair(norm, U, V)
            weight = norm.weight
            Ug = Ug.to(weight.device, weight.dtype)
            Vg = Vg.to(weight.device, weight.dtype)

            def hook(module, inputs, output):
                return output + (output @ Vg) @ Ug.T

            return hook

        def write_hook_for(module, U_inv, V):
            weight = module.weight
            U_inv = U_inv.to(weight.device, weight.dtype)
            V = V.to(weight.device, weight.dtype)

            def hook(module, inputs, output):
                return output - (output @ V) @ U_inv.T

            return hook

        handles = []
        for m in sorted(k for k in cums if k < n_layers):
            U, V = cums[m]
            block = jl.layers[m]
            norms = {}
            for _suffix, _module, norm in rebase.iter_reads(block):
                norms[id(norm)] = norm
            for norm in norms.values():
                handles.append(norm.register_forward_hook(read_hook_for(norm, U, V)))
            if exact:
                U_inv, Vw, _regularized = rebase.inverse_uv(U, V)
                for _suffix, module in rebase.iter_writes(block):
                    handles.append(module.register_forward_hook(write_hook_for(module, U_inv, Vw)))
        if not self._preserve_lm_head:
            U, V = cums[n_layers]
            handles.append(
                jl._final_norm.register_forward_hook(read_hook_for(jl._final_norm, U, V))
            )
        self._handles = handles

    def detach(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []
