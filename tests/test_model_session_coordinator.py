import pytest
import weakref

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
