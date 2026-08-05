import pytest
import weakref
import threading

from core.model_session import (
    LoadedModelBundle,
    ModelSessionCoordinator,
    ModelStateError,
    OperationConflict,
    OperationType,
)


def bundle(name="m"):
    return LoadedModelBundle.from_parts(object(), object(), object(), {"model_id": name}, {"ok": True})


class WeakModel:
    pass


def weak_bundle(name="m"):
    return LoadedModelBundle.from_parts(WeakModel(), object(), object(), {"model_id": name}, {"ok": True})


def test_initial_load_success_and_failure_session_rules():
    c = ModelSessionCoordinator()
    token, snap = c.acquire(OperationType.LOAD)
    assert snap.model_session_id == 0
    c.release(token)
    assert c.snapshot().model_session_id == 0

    token, snap = c.acquire(OperationType.LOAD)
    assert c.publish_loaded(token, bundle("a"), expected_unloaded_session=snap.model_session_id) == 1
    c.release(token)
    snap = c.snapshot()
    assert snap.loaded is True
    assert snap.model_session_id == 1


def test_unload_redundant_and_loaded_transition():
    c = ModelSessionCoordinator()
    token, _ = c.acquire(OperationType.UNLOAD)
    old, sid, changed = c.withdraw_loaded(token)
    assert old is None and sid == 0 and changed is False
    c.release(token)

    token, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(token, bundle(), expected_unloaded_session=snap.model_session_id)
    c.release(token)
    token, _ = c.acquire(OperationType.UNLOAD)
    old, sid, changed = c.withdraw_loaded(token)
    assert old is not None and sid == 2 and changed is True
    c.release(token)
    assert c.snapshot().model_state == "unloaded"


def test_replacement_success_failure_and_same_id_reload():
    c = ModelSessionCoordinator()
    t, s = c.acquire(OperationType.LOAD)
    c.publish_loaded(t, bundle("same"), expected_unloaded_session=s.model_session_id)
    c.release(t)

    t, s = c.acquire(OperationType.LOAD)
    old, unloaded_sid, changed = c.withdraw_loaded(t)
    assert changed and old.meta["model_id"] == "same" and unloaded_sid == 2
    c.release(t)
    assert c.snapshot().model_session_id == 2
    assert not c.snapshot().loaded

    t, s = c.acquire(OperationType.LOAD)
    c.publish_loaded(t, bundle("same"), expected_unloaded_session=s.model_session_id)
    c.release(t)
    assert c.snapshot().model_session_id == 3


def test_exclusive_operation_conflict_and_state_validation():
    c = ModelSessionCoordinator()
    t, _ = c.acquire(OperationType.LOAD)
    with pytest.raises(OperationConflict):
        c.acquire(OperationType.GENERATE)
    assert c.release(t) is True
    with pytest.raises(ModelStateError):
        c.acquire(OperationType.GENERATE, requires_loaded=True)


def test_stale_publish_release_and_double_release():
    c = ModelSessionCoordinator()
    t, s = c.acquire(OperationType.LOAD)
    c.release(t)
    with pytest.raises(OperationConflict):
        c.publish_loaded(t, bundle(), expected_unloaded_session=s.model_session_id)
    assert c.release(t) is False


def test_last_generation_stale_rejected():
    c = ModelSessionCoordinator()
    t, s = c.acquire(OperationType.LOAD)
    c.publish_loaded(t, bundle(), expected_unloaded_session=s.model_session_id)
    c.release(t)
    t, s = c.acquire(OperationType.GENERATE, requires_loaded=True)
    old_session = s.model_session_id
    c.release(t)
    assert not c.publish_last_generation(t, old_session, {"text": "stale"})
    assert c.snapshot().last_generation is None


def test_metadata_snapshots_do_not_retain_model_bundle():
    c = ModelSessionCoordinator()
    t, s = c.acquire(OperationType.LOAD)
    b = weak_bundle("a")
    model_ref = weakref.ref(b.hf_model)
    c.publish_loaded(t, b, expected_unloaded_session=s.model_session_id)
    c.release(t)
    status = c.status_snapshot()
    admission = c.snapshot(include_bundle=False)
    assert status.loaded and admission.loaded
    assert not hasattr(status, "bundle")
    assert admission.bundle is None
    del status, admission, b
    t, _ = c.acquire(OperationType.UNLOAD, include_bundle=False)
    old, _, changed = c.withdraw_loaded(t)
    assert changed
    del old
    c.release(t)
    import gc
    gc.collect()
    assert model_ref() is None


def test_acquire_rolls_back_when_snapshot_copy_fails():
    c = ModelSessionCoordinator()
    token, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(token, bundle("copy-fail"), expected_unloaded_session=snap.model_session_id)
    c.release(token)

    def fail_copy(_value):
        raise RuntimeError("copy failed")

    c._copy_hook = fail_copy
    with pytest.raises(RuntimeError, match="copy failed"):
        c.acquire(OperationType.GENERATE, requires_loaded=True)
    c._copy_hook = None
    assert c.status_snapshot().operation is None
    successor, _ = c.acquire(OperationType.UNLOAD, include_bundle=False)
    assert c.release(successor)


def test_snapshot_copy_hook_runs_after_lock_release_and_does_not_block_release():
    c = ModelSessionCoordinator()
    token, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(token, bundle("lock-boundary"), expected_unloaded_session=snap.model_session_id)
    c.release(token)
    active, _ = c.acquire(OperationType.GENERATE, requires_loaded=True)
    entered = threading.Event()
    unblock = threading.Event()
    first_call = threading.Event()
    hook_assertions = []

    def blocking_hook(_value):
        hook_assertions.append(not c._lock.locked())
        entered.set()
        assert unblock.wait(2), "snapshot outward-copy hook was not unblocked"

    c._copy_hook = blocking_hook
    errors = []

    def snapshot_worker():
        try:
            c.snapshot()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=snapshot_worker)
    thread.start()
    assert entered.wait(2), "snapshot outward construction did not reach hook"
    assert c.release(active) is True
    unblock.set()
    thread.join(2)
    assert not thread.is_alive(), "snapshot outward construction did not finish"
    c._copy_hook = None
    assert hook_assertions == [True]
    assert errors == []


def test_status_snapshot_does_not_retain_bundle_during_outward_construction():
    c = ModelSessionCoordinator()
    token, snap = c.acquire(OperationType.LOAD)
    b = weak_bundle("status")
    model_ref = weakref.ref(b.hf_model)
    c.publish_loaded(token, b, expected_unloaded_session=snap.model_session_id)
    c.release(token)
    entered = threading.Event()
    unblock = threading.Event()

    def blocking_hook(_value):
        if first_call.is_set():
            return
        first_call.set()
        entered.set()
        assert unblock.wait(2), "status outward construction was not unblocked"

    c._copy_hook = blocking_hook
    errors = []

    def status_worker():
        try:
            c.status_snapshot()
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=status_worker)
    thread.start()
    assert entered.wait(2), "status outward construction did not reach hook"
    c._copy_hook = None
    unload, _ = c.acquire(OperationType.UNLOAD, include_bundle=False)
    old, _, changed = c.withdraw_loaded(unload)
    assert changed
    del b, old
    c.release(unload)
    import gc
    gc.collect()
    assert model_ref() is None
    unblock.set()
    thread.join(2)
    assert not thread.is_alive(), "status outward construction did not finish"
    assert errors == []


class NoDeepcopyTensor:
    def __deepcopy__(self, memo):
        raise AssertionError("direction tensor was deep-copied")


def test_intervention_publication_does_not_deepcopy_direction_tensors():
    c = ModelSessionCoordinator()
    token, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(token, bundle("directions"), expected_unloaded_session=snap.model_session_id)
    c.release(token)
    token, _ = c.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
    record = {
        "revision": 1,
        "model_session_id": 1,
        "lens_binding_id": 7,
        "scale": 1.0,
        "mode": "standard",
        "rules": [{"id": 1, "layers": [0], "enabled": True, "dirs_a": {0: NoDeepcopyTensor()}}],
        "active_rules": [{"id": 1, "layers": [0], "enabled": True, "dirs_a": {0: NoDeepcopyTensor()}}],
        "summary": [{"id": 1, "layers": [0], "enabled": True}],
        "active_summary": [{"id": 1, "layers": [0], "enabled": True}],
    }
    c.update_interventions(token, record)
    snap = c.snapshot()
    assert snap.interventions["rules"][0]["dirs_a"][0].__class__ is NoDeepcopyTensor
    assert c.release(token)


def test_intervention_publication_failure_preserves_previous_revision():
    c = ModelSessionCoordinator()
    token, snap = c.acquire(OperationType.LOAD)
    c.publish_loaded(token, bundle("revision"), expected_unloaded_session=snap.model_session_id)
    c.release(token)
    first = {
        "revision": 1, "scale": 1.0, "mode": "standard",
        "rules": [], "active_rules": [], "summary": [], "active_summary": [],
    }
    token, _ = c.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
    c.update_interventions(token, first)
    c.release(token)

    class BadSummary:
        def __deepcopy__(self, memo):
            raise RuntimeError("summary copy failed")

    token, _ = c.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
    with pytest.raises(RuntimeError, match="summary copy failed"):
        c.update_interventions(token, dict(first, revision=2, summary=[BadSummary()]))
    c.release(token)
    assert c.status_snapshot().interventions["revision"] == 1
