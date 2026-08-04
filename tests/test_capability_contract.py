import copy
import threading
from types import SimpleNamespace

import pytest
import torch

from core import capabilities, editing, rebase
from core.ablation import Interventions
from helpers import dense_lens


def _profile():
    supported = capabilities.decision(True, "supported")
    return {
        "declared_quantization": None,
        "has_packed_read_parameters": False,
        "modes": {
            "standard": dict(supported),
            "readthrough": dict(supported),
            "exact": dict(supported),
            "abliteration": capabilities.decision(
                False, "global_projection_unvalidated"
            ),
        },
    }


def test_no_model_snapshot_is_complete_and_fail_closed():
    snap = capabilities.snapshot(None)
    assert set(snap["modes"]) == set(capabilities.MODES)
    assert set(snap["exports"]) == {"mode", *capabilities.FORMATS}
    assert all(not item["supported"] for item in snap["modes"].values())
    assert {item["reason_code"] for item in snap["modes"].values()} == {
        "no_model_loaded"
    }


@pytest.mark.parametrize("bad", [
    {},
    {"declared_quantization": None, "has_packed_read_parameters": False, "modes": {}},
])
def test_missing_capability_data_fails_closed_for_loaded_model(bad):
    snap = capabilities.snapshot(bad, loaded=True)
    assert all(item["reason_code"] == "capability_data_unavailable"
               for item in snap["modes"].values())
    with pytest.raises(ValueError, match="Capability data is unavailable"):
        capabilities.require(bad, "modes", "standard", loaded=True)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_non_boolean_supported_is_never_authorized(value):
    profile = _profile()
    profile["modes"]["readthrough"]["supported"] = value
    assert capabilities.snapshot(profile, loaded=True)["modes"]["readthrough"][
        "reason_code"] == "capability_data_unavailable"


def test_unknown_or_injected_reasons_fail_closed():
    for mutation in (lambda d: d.update(reason_code="injected"),
                     lambda d: d.update(reason="free form support")):
        profile = _profile()
        mutation(profile["modes"]["readthrough"])
        snap = capabilities.snapshot(profile, loaded=True)
        assert snap["modes"]["readthrough"]["reason_code"] == "capability_data_unavailable"


def test_snapshot_and_decision_identity_are_isolated():
    profile = _profile()
    first = capabilities.snapshot(profile, "readthrough", loaded=True)
    first["modes"]["standard"]["supported"] = False
    first["exports"]["full"]["supported"] = False
    second = capabilities.snapshot(profile, "readthrough", loaded=True)
    assert second["modes"]["standard"]["supported"]
    assert second["exports"]["full"]["supported"]
    ids = [id(item) for item in second["modes"].values()]
    assert len(ids) == len(set(ids))


def test_global_projection_is_always_disabled_in_contract_and_deep_paths(
        tmp_path, monkeypatch):
    jl = dense_lens()
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    profile = capabilities.build_profile(jl, None)
    assert profile["modes"]["abliteration"] == capabilities.decision(
        False, "global_projection_unvalidated"
    )
    rules = [{"layers": [0]}]
    with pytest.raises(ValueError, match="temporarily unavailable"):
        editing.compute_abliteration(rules, jl)
    iv = Interventions()
    iv.set_mode("abliteration")
    with pytest.raises(ValueError, match="temporarily unavailable"):
        iv.attach(jl)
    with pytest.raises(ValueError, match="temporarily unavailable"):
        editing.export_abliteration(rules, jl, {"quant": None}, fmt="full",
                                    name="must-not-exist", source_dir=tmp_path)
    assert not (editing.EDITS_DIR / "must-not-exist").exists()


@pytest.mark.parametrize("quant", ["int8", "nf4"])
def test_declared_quantization_blocks_floating_head_mutations(quant):
    jl = dense_lens()
    jl._jwash_declared_quant = quant
    jl.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    lens_manager = SimpleNamespace(lens=SimpleNamespace(jacobians={}))
    iv = Interventions()
    with pytest.raises(ValueError, match="quantized"):
        iv.add(lens_manager, jl, token_id=1, layers=[0])
    assert iv.summary() == []
    iv._rules = [{"id": 1, "factor": 1.0}]
    before = copy.deepcopy(iv._rules)
    with pytest.raises(ValueError, match="quantized"):
        iv.update(1, layers=[0], lens_manager=lens_manager, jl=jl)
    assert iv._rules == before
    iv.set_mode("readthrough")
    with pytest.raises(ValueError, match="quantized"):
        iv.attach(jl)


def test_quantization_declarations_disagree_fail_closed():
    jl = dense_lens()
    jl._jwash_declared_quant = "int8"
    with pytest.raises(ValueError, match="quantized"):
        capabilities.ensure_unquantized(jl, {"quant": None})
    jl._jwash_declared_quant = None
    with pytest.raises(ValueError, match="quantized"):
        capabilities.ensure_unquantized(jl, {"quant": "nf4"})


def test_model_wide_packed_fact_uses_validated_reader_rank():
    jl = dense_lens()
    jl.layers[0].mlp.gate_proj.weight = torch.nn.Parameter(torch.randn(2, 3, 8))
    assert rebase.has_packed_read_parameters(jl)


def test_legacy_fields_are_derived_from_validated_profile():
    profile = _profile()
    legacy = capabilities.legacy(profile, loaded=True)
    snap = capabilities.snapshot(profile, loaded=True)
    assert legacy["rebase_supported"] is snap["modes"]["readthrough"]["supported"]
    assert legacy["readthrough_reason"] == snap["modes"]["readthrough"]["reason"]
    assert legacy["exact_supported"] is snap["modes"]["exact"]["supported"]
    assert legacy["exact_reason"] == snap["modes"]["exact"]["reason"]


def test_status_malformed_loaded_profile_is_controlled_and_legacy_matches(monkeypatch):
    import api.app as app
    monkeypatch.setattr(app.manager, "hf_model", object())
    monkeypatch.setattr(app.manager, "meta", {"model_id": "broken"})
    monkeypatch.setattr(app.manager, "capability_profile", {"modes": {}})
    monkeypatch.setattr(app, "gpu_stats", lambda: [])
    status = app.api_status()
    assert status["capabilities"]["modes"]["readthrough"]["reason_code"] == (
        "capability_data_unavailable"
    )
    assert status["loaded"]["rebase_supported"] is False
    assert status["loaded"]["readthrough_reason"] == status["capabilities"][
        "modes"]["readthrough"]["reason"]


def test_global_mode_is_rejected_by_mode_and_export_apis(tmp_path, monkeypatch):
    import api.app as app
    iv = Interventions()
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.manager, "capability_profile", _profile())
    monkeypatch.setattr(app.manager, "meta", {"model_id": "local", "quant": None})
    monkeypatch.setattr(app.manager, "hf_model", object())
    monkeypatch.setattr(app.manager, "jl", dense_lens())
    with pytest.raises(app.HTTPException) as exc:
        app.api_interventions_scale(SimpleNamespace(scale=2, mode="abliteration"))
    assert exc.value.status_code == 422
    assert iv.state_snapshot() == (1.0, "standard")

    iv._mode = "abliteration"
    iv._rules = [{"layers": [0], "enabled": True}]
    monkeypatch.setattr(app, "resolve_local_dir", lambda _name: tmp_path)
    with pytest.raises(app.HTTPException) as exc:
        __import__("asyncio").run(app.api_edit_export(
            SimpleNamespace(name="global", format="full")
        ))
    assert exc.value.status_code == 422


@pytest.mark.parametrize("quant", ["int8", "nf4"])
def test_quantized_preset_rejection_is_transactional(quant, monkeypatch):
    import api.app as app
    jl = dense_lens()
    jl._jwash_declared_quant = quant
    iv = Interventions()
    iv.set_scale(2.0)
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.manager, "hf_model", object())
    monkeypatch.setattr(app.manager, "jl", jl)
    monkeypatch.setattr(app.manager, "meta", {"model_id": "local", "quant": quant})
    monkeypatch.setattr(app.lens_manager, "lens", SimpleNamespace(jacobians={}))
    monkeypatch.setattr(app.editing, "load_preset", lambda _name: {
        "scale": 9.0, "rules": [{"token_id": 1, "mode": "scale", "factor": 0.0}]
    })
    with pytest.raises(app.HTTPException) as exc:
        app.api_presets_apply("quantized")
    assert exc.value.status_code == 422
    assert iv.summary() == [] and iv.global_scale == 2.0


def test_mode_only_does_not_overwrite_concurrent_scale():
    iv = Interventions()
    iv._lock.acquire()
    thread = threading.Thread(target=lambda: iv.set_scale_and_mode(mode="readthrough"))
    thread.start()
    iv._scale = 4.0  # represents the scale update committed ahead of the waiter
    iv._lock.release()
    thread.join()
    assert iv.state_snapshot() == (4.0, "readthrough")


def test_scale_only_does_not_overwrite_concurrent_mode():
    iv = Interventions()
    iv._lock.acquire()
    thread = threading.Thread(target=lambda: iv.set_scale_and_mode(scale=3.0))
    thread.start()
    iv._mode = "exact"  # represents the mode update committed ahead of the waiter
    iv._lock.release()
    thread.join()
    assert iv.state_snapshot() == (3.0, "exact")


def test_intervention_patch_returns_the_committed_pair(monkeypatch):
    import api.app as app
    iv = Interventions()
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.manager, "meta", {"model_id": "local"})
    monkeypatch.setattr(app.manager, "capability_profile", _profile())
    response = app.api_interventions_scale(
        SimpleNamespace(scale=2.25, mode="readthrough")
    )
    assert response == {"scale": 2.25, "mode": "readthrough"}
    assert iv.state_snapshot() == (2.25, "readthrough")
