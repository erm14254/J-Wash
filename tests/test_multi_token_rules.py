"""Multi-token (composite) rules: lookup splits, composite directions,
schema round-trips, and back-compat with single-token presets."""

from types import SimpleNamespace

import pytest
import torch

from core import editing, rebase
from core.ablation import (
    ANCHOR_DECAY_DEFAULT, Interventions, abliteration_direction, piece_weights,
)
from core.tokens import token_candidates

VOCAB, DIM, N_LAYERS = 12, 8, 4


class DecodeTok:
    """decode-only tokenizer: id → ``<id>``."""

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def composite(weight, ids, weights=None):
    rows = weight[list(ids)].float()
    rows = rows / rows.norm(dim=-1, keepdim=True)
    if weights is not None:
        rows = rows * torch.tensor(weights).unsqueeze(1)
    v = rows.sum(0)
    return v / v.norm()


def make_env(jacobians=None):
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, DIM)
    jl = SimpleNamespace(
        layers=[object()] * N_LAYERS,
        _lm_head=SimpleNamespace(weight=weight),
        tokenizer=DecodeTok(),
    )
    lens_manager = SimpleNamespace(lens=SimpleNamespace(jacobians=jacobians or {}))
    return jl, lens_manager, weight


def test_add_composite_rule_directions_and_summary():
    jl, lm, weight = make_env()
    iv = Interventions()
    rules = iv.add(lm, jl, token_ids=[3, 5], mode="replace",
                   factor=0.8, replacement_ids=[4, 6, 7], layers=[0, 2])
    (r,) = rules
    assert r["token_ids"] == [3, 5] and r["token_id"] == 3
    assert r["token"] == "<3><5>" and r["token_pieces"] == ["<3>", "<5>"]
    assert r["replacement_ids"] == [4, 6, 7] and r["replacement_id"] == 4
    assert r["replacement_pieces"] == ["<4>", "<6>", "<7>"]
    full = iv.rules_full()[0]
    # source = uniform composite; replacement = first-piece-anchored (geometric
    # decay, engine default) so the word can actually win the next-token race
    repl_w = piece_weights(3, ANCHOR_DECAY_DEFAULT)
    for layer in (0, 2):
        torch.testing.assert_close(full["dirs_a"][layer], composite(weight, [3, 5]))
        torch.testing.assert_close(full["dirs_b"][layer], composite(weight, [4, 6, 7], repl_w))
    # the schema feeds the rebase engine unchanged
    by_layer = rebase.rule_factors([full], scale=1.0)
    assert set(by_layer) == {0, 2}
    w, v_a = by_layer[0][0]
    assert w.shape == (DIM,) and v_a.shape == (DIM,)


def test_single_token_unchanged_and_jacobian_path():
    J = torch.randn(DIM, DIM)
    jl, lm, weight = make_env(jacobians={1: J})
    iv = Interventions()
    iv.add(lm, jl, token_id=3, mode="scale", factor=0.0, layers=[0, 1])
    full = iv.rules_full()[0]
    assert full["token_ids"] == [3] and full["token"] == "<3>"
    row = weight[3].float()
    torch.testing.assert_close(full["dirs_a"][0], row / row.norm())
    proj = row @ J
    torch.testing.assert_close(full["dirs_a"][1], proj / proj.norm())


def test_update_switches_to_composite_and_scale_clears_replacement():
    jl, lm, weight = make_env()
    iv = Interventions()
    iv.add(lm, jl, token_id=2, mode="replace", factor=1.0,
           replacement_id=4, layers=[1])
    rid = iv.summary()[0]["id"]
    iv.update(rid, token_ids=[3, 5], lens_manager=lm, jl=jl)
    full = iv.rules_full()[0]
    assert full["token_ids"] == [3, 5] and full["token"] == "<3><5>"
    torch.testing.assert_close(full["dirs_a"][1], composite(weight, [3, 5]))
    iv.update(rid, mode="scale", lens_manager=lm, jl=jl)
    full = iv.rules_full()[0]
    assert full["replacement_ids"] is None and full["replacement_pieces"] is None
    assert full["dirs_b"] is None


def test_validation_errors():
    jl, lm, _ = make_env()
    iv = Interventions()
    with pytest.raises(ValueError, match="token_id or token_ids required"):
        iv.add(lm, jl, mode="scale", layers=[0])
    with pytest.raises(ValueError, match="replacement_id required"):
        iv.add(lm, jl, token_ids=[3], mode="replace", layers=[0])
    with pytest.raises(ValueError, match="out of range"):
        iv.add(lm, jl, token_ids=[3, VOCAB], mode="scale", layers=[0])
    with pytest.raises(ValueError, match="empty token sequence"):
        iv.add(lm, jl, token_ids=[], mode="scale", layers=[0])


def test_anchor_decay_explicit():
    jl, lm, weight = make_env()
    iv = Interventions()
    iv.add(lm, jl, token_ids=[3], mode="replace", factor=1.0,
           replacement_ids=[4, 6], anchor_decay=0.25, layers=[0])
    full = iv.rules_full()[0]
    assert full["anchor_decay"] == 0.25
    torch.testing.assert_close(full["dirs_b"][0], composite(weight, [4, 6], [1.0, 0.25]))
    rid = iv.summary()[0]["id"]
    iv.update(rid, anchor_decay=1.0, lens_manager=lm, jl=jl)
    torch.testing.assert_close(iv.rules_full()[0]["dirs_b"][0], composite(weight, [4, 6]))


def test_abliteration_direction_composite_and_legacy():
    torch.manual_seed(1)
    weight = torch.randn(VOCAB, DIM)
    v_a, v_b = abliteration_direction(
        weight, {"token_ids": [3, 5], "mode": "replace", "replacement_ids": [2]}
    )
    torch.testing.assert_close(v_a, composite(weight, [3, 5]))
    torch.testing.assert_close(v_b, composite(weight, [2]))
    legacy_a, legacy_b = abliteration_direction(weight, {"token_id": 3, "mode": "scale"})
    torch.testing.assert_close(legacy_a, composite(weight, [3]))
    assert legacy_b is None


def test_preset_roundtrip_and_legacy_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(editing, "PRESETS_DIR", tmp_path)
    jl, lm, weight = make_env()
    iv = Interventions()
    iv.add(lm, jl, token_ids=[3, 5], mode="replace",
           factor=1.0, replacement_ids=[4, 6], layers=[0])
    editing.save_preset("composite", iv.summary(), "test/model",
                        mode="readthrough", preserve_lm_head=True)
    loaded = editing.load_preset("composite")
    (rule,) = loaded["rules"]
    assert rule["token_ids"] == [3, 5] and rule["replacement_ids"] == [4, 6]
    # the mode and head-protection the preset was built with travel with it
    assert loaded["mode"] == "readthrough" and loaded["preserve_lm_head"] is True
    # old-style save (no mode/preserve): fields absent, not defaulted
    editing.save_preset("legacy-style", iv.summary(), "test/model")
    legacy_loaded = editing.load_preset("legacy-style")
    assert "mode" not in legacy_loaded and "preserve_lm_head" not in legacy_loaded

    # re-apply the way the API does: token_ids preferred, token_id fallback
    fresh = Interventions()
    fresh.add(lm, jl, token_id=rule.get("token_id"), token_ids=rule.get("token_ids"),
              mode=rule["mode"], factor=rule["factor"],
              replacement_id=rule.get("replacement_id"),
              replacement_ids=rule.get("replacement_ids"), layers=rule.get("layers"))
    torch.testing.assert_close(fresh.rules_full()[0]["dirs_a"][0], composite(weight, [3, 5]))

    # an OLD preset (single ids only, e.g. nikusui-27b-start.json) still applies
    legacy = Interventions()
    legacy.add(lm, jl, token_id=7, token_ids=None, mode="replace",
               factor=1.0, replacement_id=2, replacement_ids=None, layers=[0])
    assert legacy.summary()[0]["token_ids"] == [7]


class LookupTok:
    SINGLE = {" Anna": 7, "Anna": 8}
    MULTI = {" Ametista": [1, 3], "Ametista": [2, 3]}

    def encode(self, text, add_special_tokens=False):
        if text in self.SINGLE:
            return [self.SINGLE[text]]
        if text in self.MULTI:
            return list(self.MULTI[text])
        return [100 + i for i, _c in enumerate(text)]  # generic multi-token fallback

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def test_token_candidates_singles_and_splits():
    singles, splits = token_candidates(LookupTok(), "Anna")
    assert {c["id"] for c in singles} == {7, 8}
    singles, splits = token_candidates(LookupTok(), "Ametista")
    assert singles == []
    assert splits[0]["ids"] == [2, 3]  # original spelling first
    assert [2, 3] in [s["ids"] for s in splits] and [1, 3] in [s["ids"] for s in splits]
    assert splits[0]["pieces"] == ["<2>", "<3>"]
    assert len(splits) <= 6
    # very long sequences are not offered as composites
    singles, splits = token_candidates(LookupTok(), "x" * 20)
    assert singles == [] and splits == []
