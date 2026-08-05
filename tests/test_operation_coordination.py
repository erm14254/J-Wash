import threading

import pytest

from core.ablation import HookAttachment
from core.editing import PublicationCancelled, PublicationGate
from core.model_session import ModelSessionCoordinator, OperationConflict, OperationType


class Handle:
    def __init__(self, seen, name, fail=False):
        self.seen = seen
        self.name = name
        self.fail = fail
        self.removed = 0

    def remove(self):
        self.removed += 1
        self.seen.append(self.name)
        if self.fail:
            raise RuntimeError(self.name)


def test_generation_versus_unload_conflict_event_driven():
    c = ModelSessionCoordinator()
    load, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(load, __import__("core.model_session", fromlist=["LoadedModelBundle"]).LoadedModelBundle.from_parts(object(), object(), object(), {}, None), expected_unloaded_session=snap.model_session_id)
    c.release(load)
    acquired = threading.Event()
    release = threading.Event()
    errors = []

    def worker():
        token, _ = c.acquire(OperationType.GENERATE, requires_loaded=True)
        acquired.set()
        assert release.wait(2), "generation release phase blocked"
        c.release(token)

    thread = threading.Thread(target=worker)
    thread.start()
    assert acquired.wait(2), "generation acquisition phase blocked"
    try:
        c.acquire(OperationType.UNLOAD)
    except OperationConflict as exc:
        errors.append(str(exc))
    release.set()
    thread.join(2)
    assert not thread.is_alive(), "generation worker did not terminate after release phase"
    assert errors and "generate" in errors[0]


def test_cancellation_before_and_after_claim():
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    assert c.release(token)
    assert not c.request_cancel(token)

    token, _ = c.acquire(OperationType.GENERATE)
    assert c.request_cancel(token)
    assert c.snapshot().operation.cancellation_requested is True
    assert c.release(token)


def test_hook_attachment_owns_only_its_handles_and_is_idempotent():
    seen = []
    a1 = Handle(seen, "a1")
    a2 = Handle(seen, "a2")
    b = Handle(seen, "b")
    attach_a = HookAttachment([a1, a2])
    attach_b = HookAttachment([b])
    attach_a.close()
    attach_a.close()
    assert seen == ["a1", "a2"]
    assert b.removed == 0
    attach_b.close()
    assert seen == ["a1", "a2", "b"]


def test_hook_attachment_attempts_every_removal():
    seen = []
    attachment = HookAttachment([Handle(seen, "first", fail=True), Handle(seen, "second")])
    try:
        attachment.close()
    except RuntimeError as exc:
        assert str(exc) == "first"
    assert seen == ["first", "second"]


def test_generation_cleanup_preserves_coordinator_and_successor_owner():
    c = ModelSessionCoordinator()
    before = c
    token, _ = c.acquire(OperationType.GENERATE)
    assert c.release(token)
    successor, _ = c.acquire(OperationType.UNLOAD)
    # A stale generation finalizer must not clear a successor or replace the coordinator.
    assert c.release(token) is False
    assert before is c
    assert c.snapshot().operation.id == successor.id
    assert c.release(successor)


def test_dispatch_cancellation_before_claim_releases_without_worker_ownership():
    from core.model_session import WorkerDispatch
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    dispatch = WorkerDispatch(c, token)
    assert dispatch.cancel_from_awaiter() is True
    assert dispatch.claim() is False
    assert c.snapshot().operation is None


def test_dispatch_cancellation_after_claim_leaves_release_to_worker():
    from core.model_session import WorkerDispatch
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    dispatch = WorkerDispatch(c, token)
    assert dispatch.claim() is True
    assert dispatch.cancel_from_awaiter() is False
    assert c.snapshot().operation.id == token.id
    assert c.snapshot().operation.cancellation_requested is True
    assert dispatch.release_from_worker() is True


def test_publication_gate_cancellation_wins_before_rename():
    published = []
    gate = PublicationGate(lambda: True)
    assert gate.cancel() is True
    try:
        gate.publish(lambda: published.append("renamed"))
    except PublicationCancelled:
        pass
    assert published == []
    assert gate.state == "cancelled-before-publication"


def test_publication_gate_rename_wins_before_cancellation():
    published = []
    gate = PublicationGate(lambda: True)
    gate.publish(lambda: published.append("renamed"))
    assert gate.cancel() is False
    assert published == ["renamed"]
    assert gate.state == "published"


def test_dispatch_payload_cleared_on_cancellation_before_claim():
    import weakref
    from core.model_session import WorkerDispatch

    class Payload:
        pass

    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.GENERATE)
    payload = Payload()
    ref = weakref.ref(payload)
    dispatch = WorkerDispatch(c, token, payload=payload)
    del payload
    assert dispatch.cancel_from_awaiter() is True
    import gc
    gc.collect()
    assert ref() is None
    assert c.snapshot().operation is None


def test_intervention_update_invalid_replace_is_transactional():
    import pytest
    from core.ablation import Interventions

    iv = Interventions()
    iv._rules = [{
        "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
        "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
        "dirs_a": {0: object()}, "dirs_b": None,
    }]
    iv._revision = 1
    before = iv.state_record()
    with pytest.raises(ValueError, match="replacement_id required"):
        iv.update(1, mode="replace", lens_manager=type("LM", (), {"lens": object()})(), jl=type("JL", (), {"tokenizer": object(), "layers": [object()], "_lm_head": None})())
    assert iv.state_record() == before


def test_lens_cleanup_clears_real_withdrawn_container_before_cuda_cleanup(monkeypatch):
    import weakref
    import gc
    import core.lens_manager as lens_module
    from core.lens_manager import LensManager

    class TensorLike:
        pass

    lens = LensManager()
    live = TensorLike()
    lens.lens = TensorLike()
    lens.mask = live
    lens._J = TensorLike()
    old = lens.withdraw()
    mask_ref = weakref.ref(live)
    seen = {}

    def fake_empty_cache():
        seen["during_cuda_keys"] = list(old.keys())

    monkeypatch.setattr(lens_module.torch.cuda, "empty_cache", fake_empty_cache)
    err = lens.cleanup_withdrawn(old)
    del live
    gc.collect()
    assert err is None
    assert old == {}
    assert seen["during_cuda_keys"] == []
    assert mask_ref() is None


def test_model_transition_cleanup_does_not_restore_rules_after_publication_failure(monkeypatch):
    from api import app

    previous = app.interventions.state_record()
    try:
        app.interventions._rules = [{
            "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
            "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
            "dirs_a": {0: object()}, "dirs_b": None,
        }]
        app.interventions._revision = 1
        app.interventions._scale = 2.5
        app.interventions._mode = "readthrough"

        prepared = app._prepare_model_transition_interventions(42)
        assert prepared["rules"] == []
        app._cleanup_model_bound_state(42, token=object())
        snap = app.interventions.snapshot()
        assert snap["rules"] == []
        assert snap["active_rules"] == []
        assert snap["scale"] == 2.5
        assert snap["mode"] == "standard"
    finally:
        app.interventions.restore_state_record(previous)


def test_http_generation_with_coordinated_mappingproxy_intervention(monkeypatch):
    import asyncio
    from types import MappingProxyType, SimpleNamespace
    from api import app
    from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType

    class Layer:
        def __init__(self):
            self.handles = []

        def register_forward_hook(self, hook):
            handle = SimpleNamespace(remove=lambda: self.handles.remove(handle))
            self.handles.append(handle)
            return handle

    class FakeJL:
        def __init__(self):
            self.layers = [Layer()]
            self._jwash_declared_quant = None

    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    jl = FakeJL()
    token, snap = coordinator.acquire(OperationType.LOAD)
    coordinator.publish_loaded(
        token,
        LoadedModelBundle.from_parts(object(), object(), jl, {"model_id": "m", "quant": None}, {"ok": True}),
        expected_unloaded_session=snap.model_session_id,
    )
    coordinator.release(token)
    direction = object()
    rule = {
        "id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0,
        "replacement_id": None, "replacement": None, "layers": [0], "enabled": True,
        "dirs_a": {0: direction}, "dirs_b": None,
    }
    coordinator.bootstrap_interventions_for_test({
        "revision": 7,
        "model_session_id": coordinator.status_snapshot().model_session_id,
        "lens_binding_id": None,
        "scale": 1.0,
        "mode": "standard",
        "rules": [rule],
        "active_rules": [rule],
        "summary": [rule],
        "active_summary": [rule],
    })

    captured = {}

    def fake_generate(_messages, _sampling, _stop_event, emit, lens=None, ablator=None):
        attachment = ablator.attach(lens.jl, snapshot=lens.intervention_snapshot)
        captured["snapshot_type"] = type(lens.intervention_snapshot)
        captured["direction_identity"] = lens.intervention_snapshot["rules"][0]["dirs_a"][0]
        attachment.close()
        emit({"type": "done", "text": "ok", "stats": {"tokens": 0}})

    monkeypatch.setattr(app.manager, "generate", fake_generate)
    result = asyncio.run(app.api_generate_sync(SimpleNamespace(messages=[{"role": "user", "content": "hi"}], sampling={})))
    assert result["text"] == "ok"
    assert captured["snapshot_type"] is MappingProxyType
    assert captured["direction_identity"] is direction
    assert jl.layers[0].handles == []


def test_http_generation_metadata_preserves_captured_intervention_provenance(monkeypatch):
    import asyncio
    import torch
    from types import SimpleNamespace
    from api import app
    from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType

    class FakeTokenizer:
        chat_template = ""
        unk_token_id = -1

        def apply_chat_template(self, *args, **kwargs):
            return torch.tensor([[1]])

        def decode(self, ids, skip_special_tokens=True):
            return "ok" if ids else ""

        def encode(self, *args, **kwargs):
            return [1]

        def convert_tokens_to_ids(self, _token):
            return -1

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=2)

        def __init__(self):
            self.config = SimpleNamespace(get_text_config=lambda: SimpleNamespace(num_hidden_layers=1, hidden_size=4))
            self.emb = SimpleNamespace(weight=torch.zeros(4, 4))

        def get_input_embeddings(self):
            return self.emb

        def __call__(self, **kwargs):
            logits = torch.zeros(1, 1, 5)
            logits[0, 0, 3] = 10
            return SimpleNamespace(logits=logits, past_key_values=None)

    class Layer:
        def register_forward_hook(self, hook):
            return SimpleNamespace(remove=lambda: None)

    class FakeJL:
        _jwash_declared_quant = None
        layers = [Layer()]

    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    monkeypatch.setattr(app.lens_manager, "snapshot_for_generation", lambda: None)
    token, snap = coordinator.acquire(OperationType.LOAD)
    coordinator.publish_loaded(
        token,
        LoadedModelBundle.from_parts(
            FakeModel(), FakeTokenizer(), FakeJL(), {"model_id": "fake", "chat_template_fallback": False}, {"ok": True}
        ),
        expected_unloaded_session=snap.model_session_id,
    )
    coordinator.release(token)
    session_id = coordinator.status_snapshot().model_session_id
    coordinator.bootstrap_interventions_for_test({
        "revision": 9,
        "model_session_id": session_id,
        "lens_binding_id": 3,
        "scale": 2.0,
        "mode": "readthrough",
        "rules": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0, "layers": [], "enabled": False}],
        "active_rules": [],
        "summary": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.0, "layers": [], "enabled": False}],
        "active_summary": [],
    })

    result = asyncio.run(app.api_generate_sync(SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}],
        sampling={"max_tokens": 1, "temperature": 0, "top_p": 1, "top_k": 0, "seed": -1},
    )))
    provenance = result["meta"]["intervention_provenance"]
    assert provenance["revision"] == 9
    assert provenance["mode"] == "readthrough"
    assert provenance["scale"] == 2.0
    assert provenance["model_session_id"] == session_id
    assert provenance["lens_binding_id"] == 3
    assert provenance["summary"][0]["id"] == 1
    assert provenance["active_summary"] == []
    assert result["last_generation"]["meta"]["intervention_provenance"] == provenance


def test_persisted_continue_records_segment_provenance(monkeypatch):
    import json
    import threading

    from types import SimpleNamespace
    from api import app

    class FakeStore:
        def __init__(self):
            self.meta = {
                "intervention_provenance": {"revision": 1, "mode": "standard", "scale": 1.0},
                "model_session_id": 1,
            }
            self.updated = None

        def get_message(self, message_id):
            return {
                "id": message_id,
                "conversation_id": 99,
                "parent_id": 1,
                "role": "assistant",
                "content": "old",
                "meta": json.dumps(self.meta),
                "frames_file": None,
            }

        def path_to_root(self, message_id):
            return [{"role": "user", "content": "u"}, {"role": "assistant", "content": "old"}]

        def update_message_and_frames_if_unchanged(
            self, message_id, expected_version, content, meta=None, **kwargs
        ):
            assert expected_version == 0
            assert kwargs.get("frames") is None
            self.updated = (message_id, content, meta)
            return True

        def save_frames(self, *args, **kwargs):
            raise AssertionError("no lens frames expected")

    continuation_meta = {
        "intervention_provenance": {
            "revision": 2,
            "mode": "readthrough",
            "scale": 1.5,
            "model_session_id": 4,
            "lens_binding_id": 7,
            "summary": [{"id": 1}],
            "active_summary": [{"id": 1}],
        },
        "model_session_id": 4,
        "lens": None,
    }

    def fake_generate(**kwargs):
        kwargs["emit"]({"type": "done", "text": " new", "stats": {"tokens": 2}, "stopped": False, "meta": continuation_meta})

    fake_store = FakeStore()
    monkeypatch.setattr(app, "store", fake_store)
    monkeypatch.setattr(app.manager, "generate", fake_generate)
    emitted = []
    context = SimpleNamespace(lens=None, intervention_snapshot={"rules": []})

    app._persisted_continue({}, 10, threading.Event(), emitted.append, context)

    assert fake_store.updated[1] == "old new"
    saved_meta = fake_store.updated[2]
    assert saved_meta["intervention_provenance"]["revision"] == 2
    assert saved_meta["intervention_provenance"]["mode"] == "readthrough"
    assert saved_meta["intervention_provenance"]["scale"] == 1.5
    assert saved_meta["continuations"][0]["start_offset"] == 0
    assert saved_meta["continuations"][0]["end_offset"] == 3
    assert saved_meta["continuations"][0]["meta"]["intervention_provenance"]["revision"] == 1
    assert saved_meta["continuations"][-1]["start_offset"] == 3
    assert saved_meta["continuations"][-1]["end_offset"] == 7
    assert saved_meta["continuations"][-1]["meta"]["intervention_provenance"]["revision"] == 2
    assert emitted[-1]["meta"]["intervention_provenance"]["revision"] == 2


def test_zero_length_continuation_does_not_reattribute_existing_text(monkeypatch):
    import json
    import threading

    from types import SimpleNamespace
    from api import app

    class FakeStore:
        def __init__(self):
            self.meta = {"intervention_provenance": {"revision": 1}}
            self.updated = None
        def get_message(self, message_id):
            return {"id": message_id, "conversation_id": 1, "parent_id": None, "role": "assistant", "content": "old", "meta": json.dumps(self.meta), "frames_file": None, "version": 0}
        def path_to_root(self, message_id):
            return [{"role": "assistant", "content": "old"}]
        def update_message_and_frames_if_unchanged(self, message_id, expected_version, content, meta=None, **kwargs):
            assert expected_version == 0
            self.updated = (content, meta)
            return True

    def fake_generate(**kwargs):
        kwargs["emit"]({"type": "done", "text": "", "stats": {"tokens": 0}, "stopped": True, "meta": {"intervention_provenance": {"revision": 2}}})

    fake_store = FakeStore()
    monkeypatch.setattr(app, "store", fake_store)
    monkeypatch.setattr(app.manager, "generate", fake_generate)
    app._persisted_continue({}, 1, threading.Event(), lambda frame: None, SimpleNamespace(lens=None, intervention_snapshot={"rules": []}))
    assert fake_store.updated[0] == "old"
    meta = fake_store.updated[1]
    assert meta["intervention_provenance"]["revision"] == 1
    assert meta["continuations"][0]["meta"]["intervention_provenance"]["revision"] == 1
    assert meta["continuation_attempts"][-1]["meta"]["intervention_provenance"]["revision"] == 2


def test_preset_save_persists_coordinated_provenance(monkeypatch):
    from api import app
    from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType

    coordinator = ModelSessionCoordinator()
    monkeypatch.setattr(app.manager, "coordinator", coordinator)
    token, snap = coordinator.acquire(OperationType.LOAD)
    coordinator.publish_loaded(
        token,
        LoadedModelBundle.from_parts(object(), object(), object(), {"model_id": "m", "revision": "r1"}, {"ok": True}),
        expected_unloaded_session=snap.model_session_id,
    )
    coordinator.release(token)
    session_id = coordinator.status_snapshot().model_session_id
    coordinator.bootstrap_interventions_for_test({
        "revision": 12,
        "model_session_id": session_id,
        "lens_binding_id": 44,
        "scale": 1.75,
        "mode": "readthrough",
        "rules": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.1, "layers": [0], "enabled": True}],
        "active_rules": [],
        "summary": [{"id": 1, "token_id": 1, "token": "a", "mode": "scale", "factor": 0.1, "layers": [0], "enabled": True}],
        "active_summary": [],
    })
    captured = {}

    def save_preset(name, rules, model_id, scale=1.0, **provenance):
        captured.update(name=name, rules=rules, model_id=model_id, scale=scale, **provenance)
        return dict(captured, schema_version=2)

    monkeypatch.setattr(app.editing, "save_preset", save_preset)
    result = app.api_presets_save("preset")
    assert result["schema_version"] == 2
    assert captured["model_id"] == "m"
    assert captured["model_revision"] == "r1"
    assert captured["intervention_mode"] == "readthrough"
    assert captured["intervention_revision"] == 12
    assert captured["model_session_id"] == session_id
    assert captured["lens_binding_id"] == 44
    assert captured["rules"][0]["id"] == 1


def _frame(pos=0):
    return {
        "type": "frame",
        "phase": "gen",
        "pos": pos,
        "token_id": 1,
        "tok": "a",
        "layers": {
            0: {
                "ids": [1], "strs": ["a"], "p": [1.0],
                "m_ids": [2], "m_strs": ["b"], "m_p": [0.5], "m_rank": [1],
            }
        },
    }


def test_store_versioned_cas_rejects_stale_continuation_after_patch(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid = s.add_message(cid, None, "assistant", "old", meta={"intervention_provenance": {"revision": 1}})
    old_frame = s.save_frames(mid, [_frame(0)], [0], 1)
    captured = s.get_message(mid)
    assert captured["version"] == 1
    assert (store_mod.FRAMES_DIR / old_frame).exists()

    s.update_message(mid, "edited", meta={"edited": True}, clear_frames=True)
    assert not (store_mod.FRAMES_DIR / old_frame).exists()

    ok = s.update_message_and_frames_if_unchanged(
        mid,
        captured["version"],
        "old suffix",
        {"intervention_provenance": {"revision": 2}},
        frames=[_frame(1)],
        layers=[0],
        k=1,
    )
    assert ok is False
    current = s.get_message(mid)
    assert current["content"] == "edited"
    assert current["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.tmp-*"))
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_store_versioned_frames_commit_uses_unique_file_and_cleans_old_after_commit(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid = s.add_message(cid, None, "assistant", "old", meta={"m": 1})
    old_frame = s.save_frames(mid, [_frame(0)], [0], 1)
    captured = s.get_message(mid)
    assert old_frame.startswith(f"{mid}-v1-")

    ok = s.update_message_and_frames_if_unchanged(
        mid,
        captured["version"],
        "old new",
        {"m": 2},
        frames=[_frame(0), _frame(1)],
        layers=[0],
        k=1,
    )
    assert ok is True
    updated = s.get_message(mid)
    assert updated["version"] == captured["version"] + 1
    assert updated["frames_file"] != old_frame
    assert updated["frames_file"].startswith(f"{mid}-v{updated['version']}-")
    assert not (store_mod.FRAMES_DIR / old_frame).exists()
    assert (store_mod.FRAMES_DIR / updated["frames_file"]).exists()
    assert len(s.load_frames(mid)["frames"]) == 2


def test_legacy_continued_message_gets_unknown_base_provenance():
    from api import app

    meta = app._continuation_metadata(
        {"continued": True, "intervention_provenance": {"revision": 9}},
        12,
        15,
        {"intervention_provenance": {"revision": 10}},
        {"tokens": 1},
        False,
    )
    assert meta["continuations"][0] == {
        "start_offset": 0,
        "end_offset": 12,
        "provenance": "legacy_mixed_unknown",
        "meta": None,
    }
    assert meta["continuations"][1]["meta"]["intervention_provenance"]["revision"] == 10


def test_preset_lens_descriptor_detects_reused_runtime_binding_id():
    from api import app

    saved = app._stable_lens_descriptor({
        "lens_binding_id": 7,
        "repo_id": "lens-a",
        "filename": "lens.pt",
        "revision": "ra",
        "tapped_layers": [0, 1],
        "k": 8,
    })
    current = app._stable_lens_descriptor({
        "lens_binding_id": 7,
        "repo_id": "lens-b",
        "filename": "lens.pt",
        "revision": "rb",
        "tapped_layers": [0, 1],
        "k": 8,
    })
    same_stable_new_runtime = dict(saved, runtime_lens_binding_id=99, runtime_model_session_id=123)
    assert app._lens_descriptor_mismatch(saved, current)
    assert not app._lens_descriptor_mismatch(saved, same_stable_new_runtime)


def test_preset_lens_descriptor_compares_missing_revision_symmetrically():
    from api import app

    saved = app._stable_lens_descriptor({
        "repo_id": "lens-a",
        "filename": "lens.pt",
        "revision": None,
        "tapped_layers": [0],
        "k": 4,
    })
    current = app._stable_lens_descriptor({
        "repo_id": "lens-a",
        "filename": "lens.pt",
        "revision": "rb",
        "tapped_layers": [0],
        "k": 4,
    })
    assert saved["revision"] is None
    assert app._lens_descriptor_mismatch(saved, current)


def test_store_load_frames_retries_when_old_pointer_is_retired(tmp_path, monkeypatch):
    from pathlib import Path
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid = s.add_message(cid, None, "assistant", "old", meta={"m": 1})
    old_frame = s.save_frames(mid, [_frame(0)], [0], 1)
    captured = s.get_message(mid)
    original_read_bytes = Path.read_bytes
    retired = {"done": False}

    def racing_read_bytes(path):
        if path.name == old_frame and not retired["done"]:
            retired["done"] = True
            assert s.update_message_and_frames_if_unchanged(
                mid, captured["version"], "old new", {"m": 2},
                frames=[_frame(0), _frame(1)], layers=[0], k=1,
            )
            raise FileNotFoundError(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", racing_read_bytes)
    loaded = s.load_frames(mid)
    assert retired["done"]
    assert len(loaded["frames"]) == 2


def test_store_save_frames_detects_stale_initial_attachment(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid = s.add_message(cid, None, "assistant", "old", meta={"m": 1})
    original = s._write_unique_frame_candidate

    def racing_candidate(message_id, expected_version, blob):
        s.update_message(mid, "edited", meta={"edited": True}, clear_frames=True)
        return original(message_id, expected_version, blob)

    monkeypatch.setattr(s, "_write_unique_frame_candidate", racing_candidate)
    with pytest.raises(store_mod.StaleMessageUpdate):
        s.save_frames(mid, [_frame(0)], [0], 1)
    current = s.get_message(mid)
    assert current["content"] == "edited"
    assert current["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_persisted_continue_fails_closed_when_base_frames_cannot_load(monkeypatch):
    from api import app

    unchanged = {"content": "old", "version": 3, "updated": False}
    emitted = []

    class FakeStore:
        def get_message(self, message_id):
            return {
                "id": message_id, "conversation_id": 1, "parent_id": None,
                "role": "assistant", "content": unchanged["content"],
                "meta": "{}", "frames_file": "base.msgpack", "version": unchanged["version"],
            }
        def path_to_root(self, message_id):
            return [{"role": "assistant", "content": unchanged["content"]}]
        def load_frames(self, message_id):
            raise app.FrameStorageError("base corrupt")
        def update_message_and_frames_if_unchanged(self, *args, **kwargs):
            unchanged["updated"] = True
            return True

    def generate(**kwargs):
        kwargs["emit"]({"type": "frame", **_frame(1)})
        kwargs["emit"]({"type": "done", "text": " new", "meta": {"m": 2}, "stats": {}, "stopped": False})

    monkeypatch.setattr(app, "store", FakeStore())
    monkeypatch.setattr(app.manager, "generate", generate)
    context = type("Ctx", (), {"lens": None, "intervention_snapshot": {"rules": []}})()
    app._persisted_continue({}, 1, threading.Event(), emitted.append, context)
    assert not unchanged["updated"]
    assert emitted[-1]["type"] == "error"


def test_initial_save_frames_uses_inserted_version_not_later_edit(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid, version = s.add_message(cid, None, "assistant", "T", meta={"publication_state": "frames_pending"}, return_version=True)
    s.update_message(mid, "E", meta={"edited": True}, clear_frames=True)
    with pytest.raises(store_mod.StaleMessageUpdate):
        s.save_frames(mid, [_frame(0)], [0], 1, expected_version=version, complete_meta={"publication_state": "complete"})
    current = s.get_message(mid)
    assert current["content"] == "E"
    assert current["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_store_save_frames_rolls_back_failed_commit_before_connection_reuse(tmp_path, monkeypatch):
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid, version = s.add_message(cid, None, "assistant", "T", meta={"publication_state": "frames_pending"}, return_version=True)
    conn = s._conn()
    fail_once = {"yes": True}

    class CommitFailProxy:
        def __getattr__(self, name):
            return getattr(conn, name)
        @property
        def in_transaction(self):
            return conn.in_transaction
        def execute(self, *args, **kwargs):
            return conn.execute(*args, **kwargs)
        def rollback(self):
            return conn.rollback()
        def commit(self):
            if fail_once["yes"]:
                fail_once["yes"] = False
                raise RuntimeError("commit failed")
            return conn.commit()

    monkeypatch.setattr(s, "_conn", lambda: CommitFailProxy())
    with pytest.raises(RuntimeError):
        s.save_frames(mid, [_frame(0)], [0], 1, expected_version=version, complete_meta={"publication_state": "complete"})
    assert not conn.in_transaction
    s.update_message(mid, "after", meta={"ok": True}, clear_frames=True)
    current = s.get_message(mid)
    assert current["content"] == "after"
    assert current["frames_file"] is None


def test_store_candidate_file_fsync_failure_is_actionable(tmp_path, monkeypatch):
    import errno
    import os
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid, version = s.add_message(cid, None, "assistant", "T", return_version=True)

    def fail_fsync(fd):
        raise OSError(errno.ENOSPC, "full")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(store_mod.FrameStorageError):
        s.save_frames(mid, [_frame(0)], [0], 1, expected_version=version)
    assert s.get_message(mid)["frames_file"] is None
    assert not list(store_mod.FRAMES_DIR.glob("*.tmp-*"))
    assert not list(store_mod.FRAMES_DIR.glob("*.msgpack"))


def test_delete_conversation_treats_frame_unlink_as_garbage_cleanup(tmp_path, monkeypatch):
    from pathlib import Path
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid = s.add_message(cid, None, "assistant", "T")
    frame_file = s.save_frames(mid, [_frame(0)], [0], 1)
    original_unlink = Path.unlink

    def failing_unlink(path, *args, **kwargs):
        if path.name == frame_file:
            raise PermissionError("open reader")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    s.delete_conversation(cid)
    with pytest.raises(ValueError):
        s.get_conversation(cid)
    assert (store_mod.FRAMES_DIR / frame_file).exists()


def test_store_load_frames_corrupt_schema_maps_to_frame_storage_error(tmp_path, monkeypatch):
    import msgpack
    from core import store as store_mod

    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "db.sqlite3")
    monkeypatch.setattr(store_mod, "FRAMES_DIR", tmp_path / "frames")
    s = store_mod.Store()
    cid = s.create_conversation("c")
    mid, version = s.add_message(cid, None, "assistant", "T", return_version=True)
    filename, _ = s._write_unique_frame_candidate(mid, version, msgpack.packb({"version": 1, "frames": [{}]}))
    conn = s._conn()
    conn.execute("UPDATE messages SET frames_file = ?, version = version + 1 WHERE id = ?", (filename, mid))
    conn.commit()
    with pytest.raises(store_mod.FrameStorageError):
        s.load_frames(mid)
