"""Analytic impact of the active rules on the OUTPUT vocabulary.

Answers, before any export: "after this edit, can the model still say X?".

The composed final-read transform is ``C = I + U·Vᵀ`` (core/rebase.cumulative
at the lm_head point). For each vocab token t we probe the residual direction
that maximally expresses t (``x̂_t ∝ γ ⊙ u_t``) and measure how much of t's own
logit survives the transform:

    retained_t = (u_t·Γ·C·x̂_t) / (u_t·Γ·x̂_t) = 1 + (G_t·U)(G_t·V) / ‖G_t‖²

with ``G = W_U·Γ`` (each row = the γ-scaled unembedding). retained ≈ 1: the
word is unaffected; retained ≈ 0: the model has effectively lost the word;
retained < 0: the direction was inverted. The same probe's logit DELTA over the
whole vocabulary (``G·U·(Vᵀx̂_t)``) names the tokens that gain — i.e. what the
model now says instead ("name" → "Anna").

Exact for the readthrough/exact bakes (it IS their final-read transform);
labeled approximate for standard steering and abliteration previews, whose
final effect differs slightly. With ``preserve_lm_head`` the final read is
untouched: retained ≡ 1 by construction and the report says so.
"""

import torch

from core import rebase

# tokens with |1 − retained| below this are considered untouched
NEUTRAL_EPS = 5e-3
CHUNK = 8192


def _chunked_matmul(weight, right):
    """``weight.float() @ right`` [vocab, r] computed in row chunks (the full
    float32 copy of a 150k×5k lm_head would be ~3 GB)."""
    out = torch.empty(weight.shape[0], right.shape[1], dtype=torch.float32,
                      device=weight.device)
    for start in range(0, weight.shape[0], CHUNK):
        out[start:start + CHUNK] = weight[start:start + CHUNK].float() @ right
    return out


def _row_stats(weight, gamma2):
    """``‖G_t‖² = Σ_d u_td²·γ_d²`` per vocab row, chunked."""
    out = torch.empty(weight.shape[0], dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], CHUNK):
        chunk = weight[start:start + CHUNK].float()
        out[start:start + CHUNK] = (chunk * chunk * gamma2).sum(1)
    return out


def impact_report(rules, jl, scale, *, mode="readthrough", preserve_lm_head=False,
                  tokenizer=None, top_collateral=12, top_redirect=3):
    """``{sources, collateral, ...}`` for the active rules (see module doc)."""
    active = [r for r in rules if r["layers"] and r.get("enabled", True)]
    base = {
        "mode": mode,
        "approx": mode not in ("readthrough", "exact"),
        "preserve_lm_head": bool(preserve_lm_head),
        "sources": [],
        "collateral": [],
    }
    if not active:
        return base
    if preserve_lm_head and mode in ("readthrough", "exact"):
        # final read untouched: the output vocabulary is intact by construction
        return base

    n_layers = len(jl.layers)
    cums = rebase.cumulative(active, scale, n_layers)
    if not cums:
        return base
    U, V = cums[n_layers]

    weight = jl._lm_head.weight.detach()
    device = weight.device
    gamma = rebase.effective_gamma(jl._final_norm).to(device)  # [d]
    Ug = (gamma.unsqueeze(1) * U.to(device)).contiguous()      # G·U = W·(γ⊙U)
    Vg = (gamma.unsqueeze(1) * V.to(device)).contiguous()
    with torch.no_grad():
        P = _chunked_matmul(weight, Ug)            # [vocab, r]
        Q = _chunked_matmul(weight, Vg)            # [vocab, r]
        norm2 = _row_stats(weight, gamma * gamma).clamp_min(1e-12)
        retained = 1.0 + (P * Q).sum(1) / norm2    # [vocab]

    decode = (lambda tid: tokenizer.decode([tid])) if tokenizer is not None else str

    source_ids = []
    sources = []
    for r in active:
        for tid in (r.get("token_ids") or [r["token_id"]]):
            if tid in source_ids:
                continue
            source_ids.append(tid)
            with torch.no_grad():
                # probe aimed at this piece: what gains when the model "wants" it
                g_row = weight[tid].float() * gamma
                x_hat = g_row / g_row.norm().clamp_min(1e-12)
                delta = P @ (V.to(device).T @ x_hat)  # [vocab] logit change
                delta[tid] = float("-inf")
                top = torch.topk(delta, min(top_redirect, delta.shape[0]))
            sources.append({
                "token_id": int(tid),
                "token": decode(int(tid)),
                "rule_token": r["token"],
                "retained": round(float(retained[tid]), 3),
                "redirect": [
                    {"token_id": int(i), "token": decode(int(i)), "gain": round(float(v), 3)}
                    for v, i in zip(top.values.tolist(), top.indices.tolist())
                    if v > 0.05
                ],
            })

    with torch.no_grad():
        damage = (1.0 - retained).clone()
        damage[torch.tensor(source_ids, device=device)] = float("-inf")
        worst = torch.topk(damage, min(top_collateral * 4, damage.shape[0]))
    collateral = []
    for value, tid in zip(worst.values.tolist(), worst.indices.tolist()):
        if value < NEUTRAL_EPS or len(collateral) >= top_collateral:
            break
        collateral.append({
            "token_id": int(tid),
            "token": decode(int(tid)),
            "retained": round(float(retained[tid]), 3),
        })

    base["sources"] = sources
    base["collateral"] = collateral
    return base
