"""Integration of the wizard run-loop (style + rename dispatch) against a
stubbed model — exercises the code paths a live GPU run would, minus the GPU."""

from types import SimpleNamespace

import torch

from core.ablation import Interventions
from core.wizard import WizardRun

VOCAB, DIM = 40, 8


class DecodeTok:
    # a handful of real words → stable single ids; everything else is 1 id too
    WORDS = {" gentle": 10, " softly": 11, " assist": 12, " certainly": 13,
             " Anna": 14, " Qwen": 15, " user": 16, " Master": 17}

    def decode(self, ids):
        rev = {v: k for k, v in self.WORDS.items()}
        return "".join(rev.get(i, f"<{i}>") for i in ids)

    def encode(self, text, add_special_tokens=False):
        t = text if text.startswith(" ") else " " + text
        return [self.WORDS.get(t, 20)]


def make_run(reply_fn):
    torch.manual_seed(0)
    weight = torch.randn(VOCAB, DIM)
    jl = SimpleNamespace(
        layers=[object()] * 12,
        _lm_head=SimpleNamespace(weight=weight),
        tokenizer=DecodeTok(),
    )
    manager = SimpleNamespace(
        jl=jl, tokenizer=DecodeTok(),
        meta={"model_id": "test/qwen", "rebase_supported": True},
    )
    lens_manager = SimpleNamespace(lens=SimpleNamespace(jacobians={}))
    interventions = Interventions()
    run = WizardRun(manager, lens_manager, interventions)
    # replace live generation with a deterministic stub driven by the rules
    run.generate = lambda prompt, max_tokens=45, ablated=True, repetition_penalty=1.0: (
        reply_fn(prompt, interventions.active_rules_full()) if ablated else reply_fn(prompt, [])
    )
    return run


def test_style_goal_picks_a_boosting_candidate():
    # stub: when ANY scale rule with factor>1 is active, boosted words appear;
    # controls always answer correctly, nothing loops
    def reply(prompt, rules):
        boosting = any(r["mode"] == "scale" and r["factor"] > 1.0 for r in rules)
        if "7 times 8" in prompt:
            return "56"
        if "capital of France" in prompt:
            return "Paris"
        return "gentle softly light" if boosting else "the room was plain"

    run = make_run(reply)
    state = {}
    run.state = state
    result = run._run_style_goal(
        {"type": "style", "boost": ["gentle", "softly"], "suppress": []},
        run.manager.tokenizer, 12, True,
    )
    # baseline has no boosted words; the winner lifts them and stays coherent
    assert result["baseline"]["boost"] == 0.0
    assert result["winner"]["identity"] > 0        # boost gain > 0
    assert result["winner"]["control"] == 1.0
    # the winning scale rules are left active
    active = run.interventions.active_rules_full()
    assert active and all(r["mode"] == "scale" for r in active)
    assert {r["token_id"] for r in active} == {10, 11}  # gentle, softly


def test_run_dispatches_rename_and_style_and_applies_both(tmp_path, monkeypatch):
    from core import editing
    monkeypatch.setattr(editing, "PRESETS_DIR", tmp_path)

    def reply(prompt, rules):
        renamed = any(r["mode"] == "replace" for r in rules)
        boosting = any(r["mode"] == "scale" and r["factor"] > 1.0 for r in rules)
        if "7 times 8" in prompt:
            return "56"
        if "capital of France" in prompt:
            return "Paris"
        if "your name" in prompt.lower() or "who are you" in prompt.lower():
            return "I am Anna" if renamed else "I am Qwen"
        if "Spell" in prompt or "What is" in prompt:
            return "Qwen is a model"  # source word stays usable
        return "gentle softly" if boosting else "plain words"

    run = make_run(reply)
    state = {}
    results = run.run(
        [
            {"type": "rename", "sources": ["Qwen"], "target": "Anna", "battery": "identity"},
            {"type": "style", "boost": ["gentle"], "suppress": []},
        ],
        state,
    )
    assert state["state"] == "done"
    assert [r["goal"]["type"] for r in results] == ["rename", "style"]
    # both goals' winners are applied cumulatively: a replace AND a scale rule
    modes = {r["mode"] for r in run.interventions.active_rules_full()}
    assert "replace" in modes and "scale" in modes
    # the run persisted a preset named for the first goal's target
    assert state.get("preset") == "wizard-Anna" or state.get("preset") is None
