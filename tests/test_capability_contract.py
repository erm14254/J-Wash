from types import SimpleNamespace

import pytest

from core import capabilities, rebase
from core.ablation import Interventions
from helpers import dense_lens


def test_no_model_snapshot_is_complete_and_fail_closed():
    snap = capabilities.snapshot(None)
    assert snap["model_session_id"] is None
    assert set(snap["modes"]) == {"standard", "readthrough", "exact", "abliteration"}
    assert set(snap["exports"]) == {"mode", "full", "layers", "lora", "gguf"}
    assert all(not item["supported"] for item in snap["modes"].values())
    assert all(not snap["exports"][fmt]["supported"]
               for fmt in ("full", "layers", "lora", "gguf"))


def test_dense_profile_positively_proves_global_projection():
    profile = capabilities.build_profile(dense_lens(), None)
    assert all(profile["modes"][mode]["supported"]
               for mode in ("standard", "readthrough", "exact", "abliteration"))
    assert len(rebase.global_projection_inventory(dense_lens())) == 4
    snap = capabilities.snapshot(profile, "abliteration")
    assert all(snap["exports"][fmt]["supported"]
               for fmt in ("full", "layers", "lora", "gguf"))


def test_unknown_and_quantized_profiles_fail_closed(monkeypatch):
    unknown = dense_lens()
    del unknown.layers[0].self_attn.o_proj
    profile = capabilities.build_profile(unknown, None)
    assert not profile["modes"]["abliteration"]["supported"]

    floating_head = dense_lens()
    quantized = capabilities.build_profile(floating_head, "nf4")
    assert all(not item["supported"] for item in quantized["modes"].values())
    assert {item["reason_code"] for item in quantized["modes"].values()} == {
        "quantized_model_unsupported"
    }


def test_packed_policy_is_model_wide_for_every_rule_position():
    profile = capabilities.build_profile(dense_lens(), None)
    profile["has_packed_read_parameters"] = True
    profile["modes"]["readthrough"] = capabilities.decision(True, "supported")
    profile["modes"]["exact"] = capabilities.decision(
        False, "packed_moe_exact_unsupported"
    )
    for layer in (0, 1, 2):
        snap = capabilities.snapshot(profile, "readthrough")
        assert snap["exports"]["full"]["supported"]
        assert not snap["exports"]["layers"]["supported"]
        assert not snap["exports"]["lora"]["supported"]


def test_scale_and_mode_validation_is_transactional_and_reset_discards_state():
    iv = Interventions()
    iv.set_scale_and_mode(scale=2.5, mode="standard")
    with pytest.raises(ValueError, match="unknown intervention mode"):
        iv.set_scale_and_mode(scale=9, mode="future")
    assert iv.global_scale == 2.5 and iv.mode == "standard"
    iv._rules = [{"dirs_a": {0: object()}, "dirs_b": None}]
    iv._handles = [SimpleNamespace(remove=lambda: None)]
    iv.set_scale_and_mode(scale=3, mode="readthrough")
    iv.reset()
    assert iv.summary() == [] and iv.global_scale == 1.0 and iv.mode == "standard"
    assert iv._handles == []


def test_successful_profiles_always_get_new_session_identity():
    assert (capabilities.build_profile(dense_lens(), None)["model_session_id"] !=
            capabilities.build_profile(dense_lens(), None)["model_session_id"])
