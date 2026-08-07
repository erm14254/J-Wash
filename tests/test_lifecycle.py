import asyncio
import inspect
import logging
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from api import app as app_module
from core import capabilities
from core.ablation import Interventions
from core.model_session import LoadedModelBundle, ModelSessionCoordinator, OperationType


def _result(errors=None):
    return {
        "active": [], "recent": [], "completed": [], "abandoned": [],
        "changed_or_unsafe": [], "errors": errors or [],
    }


def test_quiet_polling_filter_only_drops_status_get():
    quiet = app_module._QuietPolling()
    status = logging.LogRecord("x", logging.INFO, "", 0, 'GET /api/status HTTP/1.1', (), None)
    other = logging.LogRecord("x", logging.INFO, "", 0, 'POST /api/status HTTP/1.1', (), None)
    assert quiet.filter(status) is False
    assert quiet.filter(other) is True


def test_lifespan_installs_loop_filter_and_cleans_matching_state(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module.editing, "inspect_abandoned_export_temps", lambda: calls.append(1) or _result())
    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    with TestClient(app_module.app):
        loop = app_module._loop_holder["loop"]
        added = [item for item in logger.filters if item not in before]
        assert len(added) == 1 and isinstance(added[0], app_module._QuietPolling)
        assert loop.is_running()
    assert calls == [1]
    assert logger.filters == before
    assert "loop" not in app_module._loop_holder


def test_shutdown_clears_only_matching_event_loop(monkeypatch):
    monkeypatch.setattr(app_module.editing, "inspect_abandoned_export_temps", lambda: _result())
    replacement = object()
    with TestClient(app_module.app):
        app_module._loop_holder["loop"] = replacement
    assert app_module._loop_holder["loop"] is replacement
    app_module._loop_holder.clear()


def test_sequential_lifespans_do_not_accumulate_filters(monkeypatch):
    monkeypatch.setattr(app_module.editing, "inspect_abandoned_export_temps", lambda: _result())
    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    for _ in range(2):
        with TestClient(app_module.app):
            assert len(logger.filters) == len(before) + 1
        assert logger.filters == before


def test_cleanup_error_is_logged_without_preventing_startup(monkeypatch, caplog):
    monkeypatch.setattr(
        app_module.editing, "inspect_abandoned_export_temps",
        lambda: _result([{"path": "/tmp/artifact", "error": "denied"}]),
    )
    with caplog.at_level(logging.WARNING), TestClient(app_module.app):
        assert "loop" in app_module._loop_holder
    assert "denied" in caplog.text


def test_unexpected_cleanup_exception_is_nonfatal(monkeypatch, caplog):
    def fail():
        raise PermissionError("root denied")
    monkeypatch.setattr(app_module.editing, "inspect_abandoned_export_temps", fail)
    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    with caplog.at_level(logging.ERROR), TestClient(app_module.app):
        assert "loop" in app_module._loop_holder
        assert len(logger.filters) == len(before) + 1
    assert logger.filters == before
    assert "root denied" in caplog.text
    assert "loop" not in app_module._loop_holder


def test_cleanup_cancellation_propagates_without_lifecycle_leaks(monkeypatch):
    async def cancelled_to_thread(*args, **kwargs):
        raise asyncio.CancelledError()
    monkeypatch.setattr(app_module.asyncio, "to_thread", cancelled_to_thread)
    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    async def exercise():
        with pytest.raises(asyncio.CancelledError):
            async with app_module.lifespan(None):
                raise AssertionError("lifespan must not yield")
    import pytest
    asyncio.run(exercise())
    assert logger.filters == before
    assert "loop" not in app_module._loop_holder


def test_fit_progress_after_shutdown_is_a_noop(monkeypatch):
    monkeypatch.setattr(app_module.editing, "inspect_abandoned_export_temps", lambda: _result())
    with TestClient(app_module.app):
        pass
    app_module._broadcast_fit({"state": "fitting"})
    assert "loop" not in app_module._loop_holder


def test_no_deprecated_event_handlers_remain():
    assert app_module.app.router.on_startup == []
    assert app_module.app.router.on_shutdown == []
    assert "on_event" not in inspect.getsource(app_module)


def test_cleanup_runs_off_the_event_loop_thread(monkeypatch):
    cleanup_thread = []

    def cleanup():
        cleanup_thread.append(threading.get_ident())
        return _result()

    monkeypatch.setattr(app_module.editing, "inspect_abandoned_export_temps", cleanup)

    async def exercise():
        loop_thread = threading.get_ident()
        async with app_module.lifespan(None):
            assert cleanup_thread == [cleanup_thread[0]]
        assert cleanup_thread[0] != loop_thread

    asyncio.run(exercise())


def _loaded_coordinator(profile=None):
    coordinator = ModelSessionCoordinator()
    token, snap = coordinator.acquire(OperationType.LOAD)
    bundle = LoadedModelBundle.from_parts(
        object(), object(), object(), {"model_id": "m", "quant": None},
        profile if profile is not None else _supported_profile(),
    )
    coordinator.publish_loaded(token, bundle, expected_unloaded_session=snap.model_session_id)
    coordinator.release(token)
    return coordinator


def _supported_profile():
    yes = capabilities.decision(True, "supported")
    return {
        "declared_quantization": None, "has_packed_read_parameters": False,
        "modes": {"standard": dict(yes), "readthrough": dict(yes),
                  "exact": dict(yes), "abliteration": capabilities.decision(
                      False, "global_projection_unvalidated")},
    }


def test_mode_notification_is_exact_and_scale_or_rejection_emit_none(monkeypatch):
    coordinator = _loaded_coordinator(_supported_profile())
    iv = Interventions()
    monkeypatch.setattr(app_module.manager, "coordinator", coordinator)
    monkeypatch.setattr(app_module, "interventions", iv)
    events = []
    monkeypatch.setattr(app_module, "_broadcast_capabilities_changed", events.append)

    result = app_module.api_interventions_scale(SimpleNamespace(scale=None, mode="readthrough"))
    assert result == {"scale": 1.0, "mode": "readthrough"}
    assert events == [coordinator.status_snapshot().model_session_id]

    assert app_module.api_interventions_scale(SimpleNamespace(scale=2.0, mode=None)) == {
        "scale": 2.0, "mode": "readthrough",
    }
    assert events == [coordinator.status_snapshot().model_session_id]

    with pytest.raises(app_module.HTTPException) as exc:
        app_module.api_interventions_scale(SimpleNamespace(scale=None, mode="abliteration"))
    assert exc.value.status_code == 422
    assert events == [coordinator.status_snapshot().model_session_id]


class _SessionSequence:
    def __init__(self, *session_ids):
        self._ids = iter(session_ids)

    def status_snapshot(self):
        return SimpleNamespace(model_session_id=next(self._ids))


@pytest.mark.parametrize("case", ["initial-load", "replacement"])
def test_successful_load_and_replacement_session_changes_notify(monkeypatch, case):
    monkeypatch.setattr(app_module.manager, "coordinator", _SessionSequence(4, 5))
    monkeypatch.setattr(app_module.manager, "load", lambda *_args: {"model_id": case})
    monkeypatch.setattr(app_module, "_valid_devices", lambda: {"cpu"})
    events = []
    monkeypatch.setattr(app_module, "_broadcast_capabilities_changed", events.append)
    result = asyncio.run(app_module.api_load(SimpleNamespace(
        model_id="m", dtype="bf16", quant=None, device="cpu",
    )))
    assert result == {"model_id": case}
    assert events == [5]


def test_successful_unload_session_change_notifies(monkeypatch):
    monkeypatch.setattr(app_module.manager, "coordinator", _SessionSequence(8, 9))
    monkeypatch.setattr(app_module.manager, "unload", lambda: {"unloaded": True})
    events = []
    monkeypatch.setattr(app_module, "_broadcast_capabilities_changed", events.append)
    assert asyncio.run(app_module.api_unload()) == {"unloaded": True}
    assert events == [9]


def test_failed_replacement_with_changed_session_notifies_resulting_session(monkeypatch):
    monkeypatch.setattr(app_module.manager, "coordinator", _SessionSequence(11, 12))
    monkeypatch.setattr(app_module.manager, "load", lambda *_args: (_ for _ in ()).throw(RuntimeError("replacement failed")))
    monkeypatch.setattr(app_module, "_valid_devices", lambda: {"cpu"})
    events = []
    monkeypatch.setattr(app_module, "_broadcast_capabilities_changed", events.append)
    with pytest.raises(app_module.HTTPException) as exc:
        asyncio.run(app_module.api_load(SimpleNamespace(
            model_id="replacement", dtype="bf16", quant=None, device="cpu",
        )))
    assert exc.value.status_code == 500
    assert events == [12]


def test_failed_load_without_session_change_does_not_notify(monkeypatch):
    monkeypatch.setattr(app_module.manager, "coordinator", _SessionSequence(3, 3))
    monkeypatch.setattr(app_module.manager, "load", lambda *_args: (_ for _ in ()).throw(RuntimeError("load failed")))
    monkeypatch.setattr(app_module, "_valid_devices", lambda: {"cpu"})
    events = []
    monkeypatch.setattr(app_module, "_broadcast_capabilities_changed", events.append)
    with pytest.raises(app_module.HTTPException):
        asyncio.run(app_module.api_load(SimpleNamespace(
            model_id="m", dtype="bf16", quant=None, device="cpu",
        )))
    assert events == []


def test_notification_scheduling_failure_cannot_fail_committed_mode(monkeypatch):
    coordinator = _loaded_coordinator(_supported_profile())
    iv = Interventions()
    monkeypatch.setattr(app_module.manager, "coordinator", coordinator)
    monkeypatch.setattr(app_module, "interventions", iv)
    app_module._loop_holder["loop"] = object()
    app_module._ws_locks[object()] = object()
    monkeypatch.setattr(
        app_module.asyncio, "run_coroutine_threadsafe",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("schedule failed")),
    )
    try:
        assert app_module.api_interventions_scale(
            SimpleNamespace(scale=None, mode="readthrough")
        ) == {"scale": 1.0, "mode": "readthrough"}
        assert iv.mode == "readthrough"
    finally:
        app_module._loop_holder.clear()
        app_module._ws_locks.clear()
