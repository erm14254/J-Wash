import threading

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

        def fail_update(_token, _snapshot):
            raise RuntimeError("publication failed")

        monkeypatch.setattr(app.manager.coordinator, "update_interventions", fail_update)
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
