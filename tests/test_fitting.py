import io
import threading

import pytest

from core import fitting


@pytest.mark.parametrize(
    ("vram", "expected"),
    [
        (15 * 2**30 - 1, 2),
        (15 * 2**30, 4),
        (15 * 2**30 + 1, 4),
    ],
)
def test_dim_batch_for_vram_boundaries(vram, expected):
    assert fitting.dim_batch_for_vram(vram) == expected


def test_default_dim_batch_falls_back_when_gpu_detection_fails(monkeypatch):
    monkeypatch.setattr(fitting, "gpu_stats", lambda: (_ for _ in ()).throw(RuntimeError("no GPU")))
    assert fitting._default_dim_batch("cuda:0") == 1


def _worker_state():
    return {
        "device": "cuda:0",
        "done": 0,
        "total": 3,
        "dim_batch": 2,
        "state": "loading",
        "elapsed": 0.0,
        "hist": [],
    }


def _manager(clock=lambda: 12.0):
    manager = fitting.FitManager(clock=clock)
    manager.state = {
        "state": "running",
        "phase": "loading",
        "done": 0,
        "total": 3,
        "workers": [_worker_state()],
    }
    manager._emit = lambda: None
    manager._refresh_totals = lambda started: manager.state.update(
        done=sum(worker["done"] for worker in manager.state["workers"])
    )
    return manager


def test_worker_fitting_event_transitions_without_completing_sequence():
    manager = _manager()
    worker = manager.state["workers"][0]
    manager._handle_worker_event({"event": "fitting", "total": 3}, worker, 10.0)
    assert manager.state["phase"] == "fitting"
    assert worker["state"] == "fitting"
    assert manager.state["done"] == worker["done"] == 0


def test_heartbeat_updates_elapsed_without_changing_done():
    manager = _manager(clock=lambda: 17.5)
    worker = manager.state["workers"][0]
    manager._heartbeat_once(10.0)
    assert manager.state["elapsed"] == 7.5
    assert worker["elapsed"] == 7.5
    assert worker["done"] == manager.state["done"] == 0


@pytest.mark.parametrize(
    "line",
    ["not json\n", "[]\n", "{}\n", '{"event":"other"}\n', '{"event":"progress","done":"one"}\n'],
)
def test_worker_output_parser_ignores_malformed_or_unrelated_lines(line):
    manager = _manager()
    worker = manager.state["workers"][0]

    class Proc:
        stdout = io.StringIO(line)

    manager._read_worker(Proc(), worker, 10.0)
    assert manager.state["phase"] == "loading"
    assert worker["state"] == "loading"
    assert worker["done"] == 0


@pytest.mark.parametrize("terminal", ["stopped", "error"])
def test_heartbeat_exits_after_worker_stops_or_fails(terminal):
    manager = _manager()

    class Proc:
        def poll(self):
            return 1

    manager._procs = [Proc()]
    manager.state["state"] = terminal
    stop = threading.Event()
    thread = threading.Thread(target=manager._heartbeat, args=(stop, 10.0))
    thread.start()
    thread.join(timeout=0.2)
    assert not thread.is_alive()


def test_corpus_loader_uses_installed_datasets_without_network(monkeypatch):
    import datasets

    sentinel = object()
    monkeypatch.setattr(datasets, "load_dataset", lambda dataset, split=None: sentinel)
    assert fitting._load_split("local/test-corpus") is sentinel


def test_fit_worker_emits_fitting_event_before_library_fit():
    source = fitting.WORKER.read_text(encoding="utf-8")
    event = 'json.dumps({"event": "fitting", "total": len(prompts)})'
    assert source.index(event) < source.index("lens = jlens.fit(")
