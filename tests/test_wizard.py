"""Wizard pure functions: scoring gate, loop detector, band math, resolution,
free-text goal extraction."""

import pytest

from core.wizard import (
    band, candidates_for, degenerate, extract_goals_json, resolve_word,
    style_candidates, total_score,
)


def test_extract_goals_json_tolerates_fences_and_prose():
    reply = ('Sure! Here is the JSON:\n```json\n{"goals": [{"type": "rename", '
             '"sources": ["Qwen", "Qwen3.5"], "target": "Ametista", '
             '"battery": "identity"}], "notes": "backstory is beyond token edits"}\n```')
    out = extract_goals_json(reply)
    assert out["goals"] == [{"type": "rename", "sources": ["Qwen", "Qwen3.5"],
                             "target": "Ametista", "battery": "identity"}]
    assert "backstory" in out["notes"]
    # invalid battery falls back to identity; empty goals rejected
    out = extract_goals_json('{"goals": [{"sources": ["User"], "target": "Master", '
                             '"battery": "weird"}]}')
    assert out["goals"][0]["battery"] == "identity"
    with pytest.raises(ValueError):
        extract_goals_json("no json here at all")
    with pytest.raises(ValueError):
        extract_goals_json('{"goals": [{"sources": [], "target": ""}]}')


def test_extract_goals_json_repairs_truncation_and_dedupes():
    # a reply cut mid-list at max_tokens (observed live): the complete first
    # goal survives, the truncated garbage goal is dropped
    truncated = ('{"goals": [{"type": "rename", "sources": ["Qwen", "Qwen3.5"], '
                 '"target": "Ametista", "battery": "identity"}, '
                 '{"type": "rename", "sources": ["my lord", "my lady", "my lo')
    out = extract_goals_json(truncated)
    assert len(out["goals"]) == 1
    assert out["goals"][0]["target"] == "Ametista"
    # duplicated sources are deduped and capped at 4
    many = ('{"goals": [{"sources": ["a", "b", "a", "c", "b", "d", "e"], '
            '"target": "X", "battery": "address"}]}')
    assert extract_goals_json(many)["goals"][0]["sources"] == ["a", "b", "c", "d"]


def test_total_score_gates_on_identity():
    # a no-op candidate (perfect vocab/control, zero identity) must not win
    assert total_score(0.0, 1.0, 1.0, 0) == 0.0
    achieved = total_score(0.3, 0.25, 1.0, 1)
    assert achieved > total_score(0.0, 1.0, 1.0, 0)
    # loops are punished
    assert total_score(0.5, 1.0, 1.0, 2) < total_score(0.5, 1.0, 1.0, 0)


def test_degenerate_detector():
    assert degenerate("Lun " * 20)
    assert degenerate("Lunaria Lunaria Lunaria Lunaria Lunaria Lunaria "
                      "Lunaria Lunaria Lunaria Lunaria Lunaria Lunaria")
    assert not degenerate("I am Lunaria, a large language model developed by "
                          "the Lunaria team, designed to assist with tasks.")
    assert not degenerate("short reply")


def test_band_and_candidates():
    layers = band(64, 0.78, 0.98)
    assert layers[0] == 49 and layers[-1] == 62  # last layer excluded
    cands = candidates_for(64)
    names = [c["name"] for c in cands]
    assert len(names) == len(set(names))
    assert any(c["preserve"] for c in cands) and any(not c["preserve"] for c in cands)
    for c in cands:
        assert c["layers"] and c["layers"][-1] < 63


def test_candidates_for_write_norm_architectures():
    # Gemma-style branch: global projection — no head protection, layers only
    # mark the rules active, knobs are factor/keep/decay/companions
    cands = candidates_for(48, rebase_supported=False)
    names = [c["name"] for c in cands]
    assert len(names) == len(set(names)) and all(n.startswith("abl-") for n in names)
    for c in cands:
        assert c["preserve"] is False
        assert c["layers"] == list(range(48))
        assert "keep" in c
    assert any(c["keep"] > 0 for c in cands)
    assert any(c["companions"] for c in cands)
    # the read-projection grid is unchanged by the new parameter's default
    assert candidates_for(64) == candidates_for(64, rebase_supported=True)


def test_extract_style_goal():
    reply = ('{"goals": [{"type": "rename", "sources": ["Qwen"], "target": '
             '"Ametista", "battery": "identity"}, {"type": "style", "boost": '
             '["gentle", "softly", "gentle"], "suppress": ["assist", "certainly"]}], '
             '"notes": ""}')
    out = extract_goals_json(reply)
    assert [g["type"] for g in out["goals"]] == ["rename", "style"]
    style = out["goals"][1]
    assert style["boost"] == ["gentle", "softly"]  # deduped
    assert style["suppress"] == ["assist", "certainly"]
    # a style goal with no words is dropped, not kept empty
    only_empty = '{"goals": [{"type": "style", "boost": [], "suppress": []}]}'
    with pytest.raises(ValueError):
        extract_goals_json(only_empty)


def test_style_candidates():
    read = style_candidates(64, rebase_supported=True)
    assert [c["name"] for c in read] == ["gentle", "medium", "strong"]
    # concept band, mid-depth, never the surface/output layers
    assert read[0]["layers"][0] >= 20 and read[0]["layers"][-1] <= 40
    assert all(c["boost"] >= 1.0 and c["suppress"] <= 1.0 for c in read)
    # Gemma path: global (all layers), same knobs
    gemma = style_candidates(48, rebase_supported=False)
    assert gemma[0]["layers"] == list(range(48))


class LookupTok:
    SINGLE = {" Anna": 7}
    MULTI = {" Ametista": [1, 3, 4]}
    STRINGS = {7: " Anna", 1: " Am", 3: "et", 4: "ista"}

    def encode(self, text, add_special_tokens=False):
        if text in self.SINGLE:
            return [self.SINGLE[text]]
        if text in self.MULTI:
            return list(self.MULTI[text])
        return [100 + i for i, _ in enumerate(text)]

    def decode(self, ids):
        return "".join(self.STRINGS.get(i, f"<{i}>") for i in ids)


def test_resolve_word():
    ids, s = resolve_word(LookupTok(), "Anna")
    assert ids == [7] and s == " Anna"
    # the leading-space split is preferred over the no-space one
    ids, s = resolve_word(LookupTok(), "Ametista")
    assert ids == [1, 3, 4] and s == " Ametista"
    with pytest.raises(ValueError):
        resolve_word(LookupTok(), "x" * 40)
