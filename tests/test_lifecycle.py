import asyncio
import inspect
import logging
import threading

from fastapi.testclient import TestClient

from api import app as app_module


def _result(errors=None):
    return {
        "removed": [], "removed_count": 0, "skipped_active": [],
        "skipped_recent": [], "errors": errors or [],
    }


def test_lifespan_installs_loop_filter_and_cleans_matching_state(monkeypatch):
    calls = []
    monkeypatch.setattr(app_module.editing, "cleanup_abandoned_export_temps", lambda: calls.append(1) or _result())
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
    monkeypatch.setattr(app_module.editing, "cleanup_abandoned_export_temps", lambda: _result())
    replacement = object()
    with TestClient(app_module.app):
        app_module._loop_holder["loop"] = replacement
    assert app_module._loop_holder["loop"] is replacement
    app_module._loop_holder.clear()


def test_sequential_lifespans_do_not_accumulate_filters(monkeypatch):
    monkeypatch.setattr(app_module.editing, "cleanup_abandoned_export_temps", lambda: _result())
    logger = logging.getLogger("uvicorn.access")
    before = list(logger.filters)
    for _ in range(2):
        with TestClient(app_module.app):
            assert len(logger.filters) == len(before) + 1
        assert logger.filters == before


def test_cleanup_error_is_logged_without_preventing_startup(monkeypatch, caplog):
    monkeypatch.setattr(
        app_module.editing, "cleanup_abandoned_export_temps",
        lambda: _result([{"path": "/tmp/artifact", "error": "denied"}]),
    )
    with caplog.at_level(logging.WARNING), TestClient(app_module.app):
        assert "loop" in app_module._loop_holder
    assert "denied" in caplog.text


def test_fit_progress_after_shutdown_is_a_noop(monkeypatch):
    monkeypatch.setattr(app_module.editing, "cleanup_abandoned_export_temps", lambda: _result())
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

    monkeypatch.setattr(app_module.editing, "cleanup_abandoned_export_temps", cleanup)

    async def exercise():
        loop_thread = threading.get_ident()
        async with app_module.lifespan(None):
            assert cleanup_thread == [cleanup_thread[0]]
        assert cleanup_thread[0] != loop_thread

    asyncio.run(exercise())
