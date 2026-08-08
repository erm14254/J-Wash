import copy
import threading
from types import SimpleNamespace

import pytest
import torch

from core import capabilities, editing, rebase
from core.ablation import Interventions
from core.model_session import GenerationContext, LoadedModelBundle, ModelSessionCoordinator, OperationType
from helpers import dense_lens


def _install_loaded_bundle(app, monkeypatch, *, jl=None, profile=None, meta=None):
    meta = dict({"model_id": "local", "quant": None}, **(meta or {}))
    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    hf_model = object()
    tokenizer = object()
    jl = jl if jl is not None else object()
    token, snap = coordinator.acquire(OperationType.LOAD)
    bundle = LoadedModelBundle.from_parts(hf_model, tokenizer, jl, meta, profile)
    coordinator.publish_loaded(token, bundle, expected_unloaded_session=snap.model_session_id)
    coordinator.release(token)
    monkeypatch.setattr(app.manager, "hf_model", hf_model)
    monkeypatch.setattr(app.manager, "tokenizer", tokenizer)
    monkeypatch.setattr(app.manager, "jl", jl)
    monkeypatch.setattr(app.manager, "meta", meta)
    monkeypatch.setattr(app.manager, "capability_profile", profile)
    return bundle


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
def test_missing_capability_data_produces_advisory_for_loaded_model(bad):
    snap = capabilities.snapshot(bad, loaded=True)
    assert all(item["reason_code"] == "capability_data_unavailable"
               for item in snap["modes"].values())


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


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(extra=True),
    lambda p: p.pop("declared_quantization"),
    lambda p: p["modes"].update(readthrough=capabilities.decision(
        False, "architecture_unsupported")),
    lambda p: p.update(has_packed_read_parameters=True),
    lambda p: p["modes"].update(exact=capabilities.decision(
        False, "packed_moe_exact_unsupported")),
    lambda p: (p.update(has_packed_read_parameters=True),
               p["modes"].update(exact=capabilities.decision(True, "supported"))),
    lambda p: p["modes"].update(exact=capabilities.decision(
        False, "mode_unsupported")),
])
def test_contradictory_intrinsic_profiles_never_authorize_exact(mutate):
    profile = _profile()
    mutate(profile)
    assert capabilities.normalize_profile(profile) is None
    snap = capabilities.snapshot(profile, "exact", loaded=True)
    assert not snap["modes"]["exact"]["supported"]
    assert not snap["exports"]["full"]["supported"]
    assert capabilities.legacy(profile, loaded=True)["exact_supported"] is False


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


def test_global_projection_is_advisory_and_fails_at_implementation_seams(
        tmp_path, monkeypatch):
    jl = dense_lens()
    monkeypatch.setattr(editing, "EDITS_DIR", tmp_path / "edits")
    profile = capabilities.build_profile(jl, None)
    assert profile["modes"]["abliteration"] == capabilities.decision(
        False, "global_projection_unvalidated"
    )
    rules = [{"layers": [0]}]
    with pytest.raises(ValueError, match="not safely implemented"):
        editing.compute_abliteration(rules, jl)
    iv = Interventions()
    iv.set_mode("abliteration")
    iv._rules = [{"layers": [0], "enabled": True}]
    with pytest.raises(ValueError, match="live attachment is not implemented"):
        attachment = iv.attach(jl)
        attachment.close()
    with pytest.raises(ValueError, match="not safely implemented"):
        editing.export_abliteration(rules, jl, {"quant": None}, fmt="full",
                                    name="must-not-exist", source_dir=tmp_path)
    assert not (editing.EDITS_DIR / "must-not-exist").exists()


@pytest.mark.parametrize("quant", ["int8", "nf4"])
def test_declared_quantization_does_not_block_usable_floating_head(quant):
    jl = dense_lens()
    jl._jwash_declared_quant = quant
    jl.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    lens_manager = SimpleNamespace(lens=SimpleNamespace(jacobians={}))
    iv = Interventions()
    assert iv.add(lens_manager, jl, token_id=1, layers=[0])[0]["token_id"] == 1
    assert iv.update(1, layers=[0], lens_manager=lens_manager, jl=jl)[0]["layers"] == [0]


def test_unusable_direction_tensor_fails_concretely_and_transactionally():
    jl = dense_lens()
    jl._lm_head.weight = torch.nn.Parameter(torch.ones(8, dtype=torch.int64), requires_grad=False)
    jl.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    iv = Interventions()
    with pytest.raises(ValueError, match="2-D tensor"):
        iv.add(SimpleNamespace(lens=SimpleNamespace(jacobians={})), jl, token_id=1, layers=[0])
    assert iv.summary() == []


@pytest.mark.parametrize("quant", [None, "int8", "nf4"])
def test_compatibility_metadata_uses_jl_quantization(quant):
    from core.model_manager import _rebase_capability_meta
    jl = dense_lens()
    if quant is not None:
        jl._jwash_declared_quant = quant
    profile = capabilities.build_profile(jl, quant)
    assert _rebase_capability_meta(jl) == capabilities.legacy(profile, loaded=True)


def test_profile_construction_conflicting_quantization_is_unavailable():
    jl = dense_lens()
    jl._jwash_declared_quant = "int8"
    profile = capabilities.build_profile(jl, "nf4")
    assert profile is None
    assert capabilities.snapshot(profile, loaded=True)["modes"]["standard"][
        "reason_code"] == "capability_data_unavailable"


def test_noncanonical_rank3_dense_reader_is_not_reclassified_as_packed():
    jl = dense_lens()
    jl.layers[0].mlp.gate_proj.weight = torch.nn.Parameter(torch.randn(2, 3, 8))
    with pytest.raises(ValueError, match="audited Linear"):
        rebase.has_packed_read_parameters(jl)


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
    _install_loaded_bundle(app, monkeypatch, profile={"modes": {}}, meta={"model_id": "broken"})
    monkeypatch.setattr(app, "gpu_stats", lambda: [])
    status = app.api_status()
    assert status["capabilities"]["modes"]["readthrough"]["reason_code"] == (
        "capability_data_unavailable"
    )
    assert status["loaded"]["rebase_supported"] is False
    assert status["loaded"]["readthrough_reason"] == status["capabilities"][
        "modes"]["readthrough"]["reason"]


def test_global_mode_commits_and_export_reaches_safe_stub(tmp_path, monkeypatch):
    import api.app as app
    iv = Interventions()
    monkeypatch.setattr(app, "interventions", iv)
    _install_loaded_bundle(app, monkeypatch, jl=dense_lens(), profile=_profile())
    assert app.api_interventions_scale(SimpleNamespace(scale=2, mode="abliteration")) == {
        "scale": 2.0, "mode": "abliteration",
    }

    iv._mode = "abliteration"
    iv._rules = [{"layers": [0], "enabled": True}]
    monkeypatch.setattr(app, "resolve_local_dir", lambda _name: tmp_path)
    with pytest.raises(app.HTTPException) as exc:
        __import__("asyncio").run(app.api_edit_export(
            SimpleNamespace(name="global", format="full")
        ))
    assert exc.value.status_code == 422


@pytest.mark.parametrize("quant", ["int8", "nf4"])
def test_quantized_declaration_does_not_reject_usable_preset(quant, monkeypatch):
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
    result = app.api_presets_apply("quantized")
    assert result["scale"] == 9.0
    assert len(result["rules"]) == 1


def test_mode_only_does_not_overwrite_concurrent_scale():
    iv = Interventions()
    attempted = threading.Event()
    original = iv._lock
    class SignalingLock:
        def __enter__(self):
            attempted.set(); original.acquire(); return self
        def __exit__(self, *_args): original.release()
    original.acquire()
    iv._lock = SignalingLock()
    thread = threading.Thread(target=lambda: iv.set_scale_and_mode(mode="readthrough"))
    thread.start()
    assert attempted.wait(2), "mode worker never attempted the locked commit"
    iv._scale = 4.0  # represents the scale update committed ahead of the waiter
    original.release()
    thread.join(2)
    assert not thread.is_alive(), "mode worker did not finish"
    iv._lock = original
    assert iv.state_snapshot() == (4.0, "readthrough")


def test_scale_only_does_not_overwrite_concurrent_mode():
    iv = Interventions()
    attempted = threading.Event()
    original = iv._lock
    class SignalingLock:
        def __enter__(self):
            attempted.set(); original.acquire(); return self
        def __exit__(self, *_args): original.release()
    original.acquire()
    iv._lock = SignalingLock()
    thread = threading.Thread(target=lambda: iv.set_scale_and_mode(scale=3.0))
    thread.start()
    assert attempted.wait(2), "scale worker never attempted the locked commit"
    iv._mode = "exact"  # represents the mode update committed ahead of the waiter
    original.release()
    thread.join(2)
    assert not thread.is_alive(), "scale worker did not finish"
    iv._lock = original
    assert iv.state_snapshot() == (3.0, "exact")


def test_intervention_patch_returns_the_committed_pair(monkeypatch):
    import api.app as app
    iv = Interventions()
    monkeypatch.setattr(app, "interventions", iv)
    _install_loaded_bundle(app, monkeypatch, profile=_profile())
    response = app.api_interventions_scale(
        SimpleNamespace(scale=2.25, mode="readthrough")
    )
    assert response == {"scale": 2.25, "mode": "readthrough"}
    assert iv.state_snapshot() == (2.25, "readthrough")


def _existing_rule():
    return {
        "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
        "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
        "dirs_a": {0: object()}, "dirs_b": None,
    }


def test_direction_patch_unavailable_profile_reaches_concrete_validation(monkeypatch):
    import api.app as app
    jl = dense_lens()
    iv = Interventions()
    iv._rules = [_existing_rule()]
    before = iv.state_record()
    monkeypatch.setattr(app, "interventions", iv)
    _install_loaded_bundle(app, monkeypatch, jl=jl, profile={"modes": {}})
    with pytest.raises(app.HTTPException) as exc:
        app.api_interventions_patch(
            1, SimpleNamespace(factor=None, layers=[1], enabled=None, token_id=None,
                               replacement_id=None, mode=None)
        )
    assert exc.value.status_code == 422
    assert "lens" in exc.value.detail
    assert iv.state_record() == before


def test_direction_patch_unknown_rule_with_valid_profile_remains_404(monkeypatch):
    import api.app as app
    jl = dense_lens()
    jl.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    iv = Interventions()
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.lens_manager, "lens", SimpleNamespace(jacobians={}))
    _install_loaded_bundle(app, monkeypatch, jl=jl, profile=_profile())
    with pytest.raises(app.HTTPException) as exc:
        app.api_interventions_patch(
            999, SimpleNamespace(factor=None, layers=[1], enabled=None, token_id=None,
                                 replacement_id=None, mode=None)
        )
    assert exc.value.status_code == 404
    assert exc.value.detail == "unknown rule 999"


def test_preset_unavailable_profile_does_not_preempt_mutation(monkeypatch):
    import api.app as app
    iv = Interventions()
    iv._rules = [_existing_rule()]
    iv._scale = 2.5
    before = iv.state_record()
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.lens_manager, "lens", SimpleNamespace(jacobians={}))
    _install_loaded_bundle(app, monkeypatch, jl=dense_lens(), profile={"modes": {}})
    monkeypatch.setattr(app.editing, "load_preset", lambda _name: {
        "scale": 9.0, "rules": [{"token_id": 2, "mode": "scale", "factor": 0.0}],
    })
    result = app.api_presets_apply("unavailable")
    assert result["scale"] == 9.0
    assert all("Capability data" not in warning for warning in result["warnings"])


def test_cleanup_disable_delete_and_clear_survive_unavailable_profile(monkeypatch):
    import api.app as app
    iv = Interventions()
    iv._rules = [_existing_rule(), {**_existing_rule(), "id": 2}]
    monkeypatch.setattr(app, "interventions", iv)
    _install_loaded_bundle(app, monkeypatch, profile={"modes": {}})

    disabled = app.api_interventions_patch(
        1, SimpleNamespace(factor=None, layers=None, enabled=False, token_id=None,
                           replacement_id=None, mode=None)
    )
    assert next(rule for rule in disabled["rules"] if rule["id"] == 1)["enabled"] is False
    removed = app.api_interventions_remove(1)
    assert [rule["id"] for rule in removed["rules"]] == [2]
    cleared = app.api_interventions_clear()
    assert cleared == {"rules": []}


@pytest.mark.parametrize("patch,field,expected", [
    ({"factor": 0.75}, "factor", 0.0),
    ({"enabled": True}, "enabled", False),
    ({"enabled": False, "factor": 0.75}, "factor", 0.0),
])
def test_effectful_factor_enable_and_combined_patches_ignore_capability(
        monkeypatch, patch, field, expected):
    import api.app as app
    rule = _existing_rule()
    if "enabled" in patch and patch["enabled"] is True:
        rule["enabled"] = False
    iv = Interventions()
    iv._rules = [rule]
    monkeypatch.setattr(app, "interventions", iv)
    _install_loaded_bundle(app, monkeypatch, profile={"modes": {}})
    request = dict(factor=None, layers=None, enabled=None, token_id=None,
                   replacement_id=None, mode=None)
    request.update(patch)
    response = app.api_interventions_patch(1, SimpleNamespace(**request))
    assert response["rules"][0][field] == patch[field]


def test_standard_editing_does_not_require_readthrough_support(monkeypatch):
    import api.app as app
    profile = _profile()
    profile["modes"]["readthrough"] = capabilities.decision(False, "architecture_unsupported")
    profile["modes"]["exact"] = capabilities.decision(False, "architecture_unsupported")
    jl = dense_lens()
    jl.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))
    iv = Interventions()
    iv._rules = [_existing_rule()]
    monkeypatch.setattr(app, "interventions", iv)
    monkeypatch.setattr(app.lens_manager, "lens", SimpleNamespace(jacobians={0: torch.eye(8)}))
    _install_loaded_bundle(app, monkeypatch, jl=jl, profile=profile)
    token, snap = app.manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE)
    try:
        response = app._interventions_patch_resource(
            token, snap, 1,
            SimpleNamespace(factor=None, layers=[0], enabled=None, token_id=None,
                            replacement_id=None, mode=None), True,
        )
    finally:
        app.manager.coordinator.release(token)
    assert response.status is None
    assert response.value["rules"][0]["layers"] == [0]


@pytest.mark.parametrize("active", [True, False])
def test_generation_unavailable_profile_only_blocks_actual_active_rules(monkeypatch, active):
    import api.app as app

    class NormalGenerationSeam(RuntimeError):
        pass

    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            raise NormalGenerationSeam("normal generation reached")

    class Ablator:
        def __init__(self):
            self.attach_calls = 0

        def attach(self, *_args, **_kwargs):
            self.attach_calls += 1
            return SimpleNamespace(close=lambda: None)

    rule = _existing_rule()
    intervention_snapshot = {
        "mode": "standard", "rules": [rule],
        "active_rules": [rule] if active else [],
    }
    jl = SimpleNamespace()
    bundle = LoadedModelBundle.from_parts(
        object(), Tokenizer(), jl, {"model_id": "m", "quant": None}, {"modes": {}},
    )
    context = GenerationContext(
        token=object(), model_session_id=7, bundle=bundle, hf_model=object(),
        tokenizer=bundle.tokenizer, jl=jl, meta=bundle.meta,
        capability_profile=bundle.capability_profile, lens=None,
        intervention_snapshot=intervention_snapshot, stop_event=threading.Event(),
    )
    ablator = Ablator()
    with pytest.raises(NormalGenerationSeam, match="normal generation reached"):
        app.manager.generate([], {}, threading.Event(), lambda _event: None,
                             lens=context, ablator=ablator)
    assert ablator.attach_calls == 1


@pytest.mark.parametrize("mode,profile,allowed", [
    ("readthrough", {
        "declared_quantization": None, "has_packed_read_parameters": False,
        "modes": {
            "standard": capabilities.decision(True, "supported"),
            "readthrough": capabilities.decision(False, "architecture_unsupported"),
            "exact": capabilities.decision(False, "architecture_unsupported"),
            "abliteration": capabilities.decision(False, "global_projection_unvalidated"),
        },
    }, False),
    ("standard", {
        "declared_quantization": None, "has_packed_read_parameters": False,
        "modes": {
            "standard": capabilities.decision(True, "supported"),
            "readthrough": capabilities.decision(False, "architecture_unsupported"),
            "exact": capabilities.decision(False, "architecture_unsupported"),
            "abliteration": capabilities.decision(False, "global_projection_unvalidated"),
        },
    }, True),
])
def test_generation_guard_uses_captured_selected_mode(monkeypatch, mode, profile, allowed):
    import api.app as app
    class Seam(RuntimeError): pass
    class Tokenizer:
        def apply_chat_template(self, *_args, **_kwargs): raise Seam("normal seam")
    class Ablator:
        def __init__(self): self.calls = 0
        def attach(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(close=lambda: None)
    snapshot = {"mode": mode, "active_rules": [_existing_rule()], "rules": [_existing_rule()]}
    bundle = LoadedModelBundle.from_parts(
        object(), Tokenizer(), SimpleNamespace(), {"model_id": "m", "quant": None}, profile,
    )
    context = GenerationContext(
        token=object(), model_session_id=9, bundle=bundle, hf_model=object(),
        tokenizer=bundle.tokenizer, jl=bundle.jl, meta=bundle.meta,
        capability_profile=bundle.capability_profile, lens=None,
        intervention_snapshot=snapshot, stop_event=threading.Event(),
    )
    ablator = Ablator()
    with pytest.raises(Seam, match="normal seam"):
        app.manager.generate([], {}, threading.Event(), lambda _event: None,
                             lens=context, ablator=ablator)
    assert ablator.calls == 1
