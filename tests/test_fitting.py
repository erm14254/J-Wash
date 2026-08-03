import io
import json
import threading
from types import SimpleNamespace

import pytest

from core import fitting


@pytest.mark.parametrize(
    ("vram", "expected"),
    [(15 * 2**30 - 1, 2), (15 * 2**30, 4), (15 * 2**30 + 1, 4)],
)
def test_dim_batch_for_vram_boundaries(vram, expected):
    assert fitting.dim_batch_for_vram(vram) == expected


def test_default_dim_batch_falls_back_when_gpu_detection_fails(monkeypatch):
    monkeypatch.setattr(
        fitting, "gpu_stats", lambda: (_ for _ in ()).throw(RuntimeError("no GPU"))
    )
    assert fitting._default_dim_batch("cuda:0") == 1


def _start(manager, name="fit"):
    return manager.start(
        model_id="local/model",
        source="local/model",
        n_prompts=1,
        devices=("cuda:0",),
        name=name,
        dim_batch=2,
        datasets=("local/corpus",),
    )


def _wait(event, message="timed out"):
    assert event.wait(2), message


class _Process:
    def __init__(self, *, release=None, returncode=0, stdout=""):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO("")
        self._release = release or threading.Event()
        if release is None:
            self._release.set()
        self.returncode = None
        self.terminated = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self):
        self._release.wait()
        self.waited = True
        self.returncode = 1 if self.terminated else 0
        return self.returncode


class _BlockingStdout:
    def __init__(self, release, text):
        self.release = release
        self.text = text

    def __iter__(self):
        _wait(self.release, "stdout was not released")
        yield from io.StringIO(self.text)


class _ExplodingStdout:
    def __iter__(self):
        raise RuntimeError("reader exploded")
        yield  # pragma: no cover - makes this an iterator


class _Lens:
    d_model = 8
    source_layers = [0, 1]
    n_prompts = 1

    def __init__(self, saved=None):
        self.saved = saved

    def save(self, path):
        if self.saved is not None:
            self.saved.append(path)


@pytest.fixture
def fit_env(tmp_path, monkeypatch):
    fits = tmp_path / "fits"
    lenses = tmp_path / "lenses"
    monkeypatch.setattr(fitting, "FITS_DIR", fits)
    monkeypatch.setattr(fitting.config, "LENSES_DIR", lenses)
    monkeypatch.setattr(fitting, "gpu_stats", lambda: [])
    return fits, lenses


def test_stop_during_corpus_loading_prevents_worker_start(fit_env, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    popen_calls = []

    def load(*args):
        entered.set()
        _wait(release)
        return ["prompt"]

    monkeypatch.setattr(fitting, "_load_corpus", load)
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: popen_calls.append(a))
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    _wait(entered)

    assert manager.stop()["state"] == "stopping"
    release.set()
    _wait(run.done)

    assert manager.state["state"] == "stopped"
    assert not popen_calls
    assert run.heartbeat_thread is None
    assert not (fit_env[1] / "fit").exists()


def test_stop_during_worker_startup_is_not_missed(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    popen_entered = threading.Event()
    allow_popen = threading.Event()
    proc = _Process()

    def popen(*args, **kwargs):
        popen_entered.set()
        _wait(allow_popen)
        return proc

    monkeypatch.setattr(fitting.subprocess, "Popen", popen)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    _wait(popen_entered)
    stopped = threading.Event()
    stopper = threading.Thread(target=lambda: (manager.stop(), stopped.set()))
    stopper.start()
    assert run.cancel.wait(1)
    allow_popen.set()
    _wait(stopped)
    stopper.join()
    _wait(run.done)

    assert proc.terminated and proc.waited
    assert manager.state["state"] == "stopped"
    assert run.heartbeat_thread is None or not run.heartbeat_thread.is_alive()


def test_restart_rejected_until_old_run_cleanup(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release = threading.Event()
    proc = _Process(release=release)
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager, "first")
    run = manager._active_run
    while not run.processes:
        assert not run.done.wait(0.01)

    manager.stop()
    with pytest.raises(ValueError, match="already in progress"):
        _start(manager, "second")
    release.set()
    _wait(run.done)

    # The old identity is gone before its done event is published.
    assert manager._active_run is None
    entered = threading.Event()
    hold = threading.Event()
    monkeypatch.setattr(
        fitting,
        "_load_corpus",
        lambda *a: (entered.set(), _wait(hold), ["prompt"])[2],
    )
    _start(manager, "second")
    second = manager._active_run
    _wait(entered)
    manager.stop()
    hold.set()
    _wait(second.done)


def test_old_run_cannot_mutate_new_run():
    manager = fitting.FitManager()
    old = fitting._FitRun()
    new = fitting._FitRun()
    attempt = threading.Event()
    finished = threading.Event()

    def late_old_update():
        _wait(attempt)
        assert not manager._update(old, state="error", name="old")
        manager._heartbeat_once(old, 0)
        manager._handle_worker_event(
            old,
            {"event": "progress", "done": 1, "total": 1},
            _worker(total=1),
            0,
        )
        finished.set()

    thread = threading.Thread(target=late_old_update)
    thread.start()
    with manager._lock:
        manager._active_run = new
        manager.state = {"state": "running", "name": "new"}

    attempt.set()
    _wait(finished)
    thread.join()
    assert manager.state == {"state": "running", "name": "new"}


def test_cancelled_run_ignores_late_worker_and_heartbeat_updates():
    manager, run = _eta_manager(lambda: 20.0)
    worker = manager.state["workers"][0]
    manager.state["state"] = "stopping"
    run.cancel.set()

    manager._handle_worker_event(
        run, {"event": "progress", "done": 1, "total": 3}, worker, 0
    )
    manager._heartbeat_once(run, 0)

    assert manager.state["state"] == "stopping"
    assert manager.state["done"] == worker["done"] == 0
    assert worker["hist"] == []
    assert "elapsed" not in manager.state


@pytest.mark.parametrize("outcome", ["success", "error", "stop"])
def test_heartbeat_cleanup_after_terminal_outcome(
    outcome, fit_env, monkeypatch
):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release = threading.Event()
    proc = _Process(
        release=release,
        stdout='{"event":"fitting","total":1}\n{"event":"done"}\n',
    )
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    saved = []
    if outcome == "error":
        monkeypatch.setattr(
            fitting.JacobianLens, "load", lambda *a: (_ for _ in ()).throw(RuntimeError("merge failed"))
        )
    else:
        monkeypatch.setattr(fitting.JacobianLens, "load", lambda *a: _Lens(saved))
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    while run.heartbeat_thread is None:
        assert not run.done.wait(0.01)
    assert run.heartbeat_thread.is_alive()

    if outcome == "stop":
        manager.stop()
    release.set()
    _wait(run.done)

    assert run.heartbeat_stop.is_set()
    assert not run.heartbeat_thread.is_alive()
    assert all(not thread.is_alive() for thread in run.helper_threads)
    assert manager.state["state"] == {"success": "done", "error": "error", "stop": "stopped"}[outcome]


def test_merge_waits_for_stdout_done_event(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release_stdout = threading.Event()
    merge_started = threading.Event()
    proc = _Process()
    proc.stdout = _BlockingStdout(
        release_stdout,
        '{"event":"fitting"}\n'
        '{"event":"progress","done":1,"total":1}\n'
        '{"event":"done"}\n',
    )
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)

    def load(*args):
        merge_started.set()
        return _Lens()

    monkeypatch.setattr(fitting.JacobianLens, "load", load)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    assert not merge_started.wait(0.1)
    assert manager.state["phase"] == "loading"

    release_stdout.set()
    _wait(run.done)
    assert merge_started.is_set()
    assert manager.state["state"] == "done"
    assert manager.state["done"] == manager.state["total"] == 1
    assert manager.state["workers"][0]["state"] == "done"


def test_worker_states_are_published_before_reader_events(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["a", "b"])
    second_spawn_entered = threading.Event()
    release_second_spawn = threading.Event()
    first_done = threading.Event()
    calls = [0]

    def popen(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return _Process(stdout='{"event":"progress","done":1,"total":1}\n{"event":"done"}\n')
        second_spawn_entered.set()
        _wait(release_second_spawn)
        return _Process(stdout='{"event":"progress","done":1,"total":1}\n{"event":"done"}\n')

    monkeypatch.setattr(fitting.subprocess, "Popen", popen)
    monkeypatch.setattr(fitting.JacobianLens, "load", lambda *a: _Lens())
    monkeypatch.setattr(fitting.JacobianLens, "merge", lambda parts: _Lens())
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = lambda state: (
        first_done.set()
        if state.get("workers") and state["workers"][0].get("state") == "done"
        else None
    )
    manager.start(
        model_id="local/model", source="local/model", n_prompts=2,
        devices=("cuda:0", "cuda:1"), name="fit", dim_batch=2,
        datasets=("local/corpus",),
    )
    run = manager._active_run
    _wait(second_spawn_entered)
    _wait(first_done)
    assert len(manager.state["workers"]) == 2
    assert manager.state["done"] == manager.state["workers"][0]["done"] == 1

    release_second_spawn.set()
    _wait(run.done)
    assert manager.state["state"] == "done"
    assert manager.state["done"] == manager.state["total"] == 2
    assert all(worker["state"] == "done" for worker in manager.state["workers"])
    assert manager.state["eta_seconds"] == 0


@pytest.mark.parametrize("cancel", [False, True])
def test_stdout_reader_exception_blocks_merge(fit_env, monkeypatch, cancel):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    proc = _Process()
    proc.stdout = _ExplodingStdout()
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    merge_started = threading.Event()
    monkeypatch.setattr(
        fitting.JacobianLens,
        "load",
        lambda *a: (merge_started.set(), _Lens())[1],
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    if cancel:
        manager.stop()
    _wait(run.done)
    assert manager.state["state"] == ("stopped" if cancel else "error")
    assert not merge_started.is_set()
    assert all(not thread.is_alive() for thread in run.reader_threads)


def test_successful_process_without_terminal_event_fails_closed(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    proc = _Process(stdout='{"event":"fitting"}\n')
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    merge_started = threading.Event()
    monkeypatch.setattr(
        fitting.JacobianLens,
        "load",
        lambda *a: (merge_started.set(), _Lens())[1],
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    _wait(run.done)
    assert manager.state["state"] == "error"
    assert "terminal done event" in manager.state["error"]
    assert not merge_started.is_set()


@pytest.mark.parametrize("terminal", ["done", "error", "stopped"])
def test_terminal_worker_events_are_ignored(terminal):
    manager, run = _eta_manager(lambda: 10.0)
    worker = manager.state["workers"][0]
    manager.state.update(
        state=terminal,
        phase=terminal,
        done=0,
        total=3,
        eta_seconds=17,
    )
    before = dict(manager.state)
    worker_before = dict(worker)
    manager._handle_worker_event(
        run, {"event": "progress", "done": 2, "total": 3}, worker, 0
    )
    assert manager.state == before
    assert worker == worker_before


def test_unpublished_worker_state_cannot_mutate_authoritative_aggregate():
    manager, run = _eta_manager(lambda: 10.0)
    authoritative = manager.state["workers"][0]
    unpublished = _worker(total=9)
    before = dict(manager.state)

    manager._handle_worker_event(
        run, {"event": "progress", "done": 9, "total": 9}, unpublished, 0
    )

    assert manager.state == before
    assert authoritative["done"] == 0
    assert unpublished["done"] == 0


def test_error_wins_then_stop():
    manager = fitting.FitManager()
    manager._emit = lambda: None
    run = fitting._FitRun()
    manager._active_run = run
    manager.state = {"state": "running"}
    manager._publish_terminal(run, error=RuntimeError("failed"))
    assert manager.stop()["state"] == "error"
    assert not run.cancel.is_set()


@pytest.mark.parametrize("winner", ["stop", "success"])
def test_stop_success_true_concurrent_lock_order(winner):
    manager = fitting.FitManager()
    manager._emit = lambda: None
    run = fitting._FitRun()
    manager._active_run = run
    manager.state = {"state": "running", "phase": "fitting"}
    ready = threading.Barrier(3)
    first_published = threading.Event()

    def publish_success():
        ready.wait()
        if winner == "stop":
            _wait(first_published)
        manager._publish_terminal(
            run, success={"state": "done", "phase": "done", "eta_seconds": 0}
        )
        if winner == "success":
            first_published.set()

    def publish_stop():
        ready.wait()
        if winner == "success":
            _wait(first_published)
        manager.stop()
        if winner == "stop":
            first_published.set()

    success_thread = threading.Thread(target=publish_success)
    stop_thread = threading.Thread(target=publish_stop)
    success_thread.start()
    stop_thread.start()
    ready.wait()
    success_thread.join()
    stop_thread.join()

    assert manager.state["state"] == ("stopped" if winner == "stop" else "done")


def _worker(done=0, total=3):
    return {
        "device": "cuda:0",
        "done": done,
        "total": total,
        "dim_batch": 2,
        "state": "fitting",
        "elapsed": 0.0,
        "hist": [],
    }


def _eta_manager(clock):
    manager = fitting.FitManager(clock=clock)
    run = fitting._FitRun()
    manager._active_run = run
    manager.state = {
        "state": "running",
        "phase": "fitting",
        "done": 0,
        "total": 3,
        "workers": [_worker()],
        "eta_seconds": None,
    }
    manager._emit = lambda: None
    return manager, run


def test_heartbeat_does_not_change_eta_or_completion_history():
    now = [10.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    manager._handle_worker_event(run, {"event": "progress", "done": 1, "total": 3}, worker, 0)
    assert manager.state["eta_seconds"] == 20
    before = list(worker["hist"])

    for now[0] in (20.0, 30.0, 40.0):
        manager._heartbeat_once(run, 0)

    assert manager.state["done"] == 1
    assert worker["hist"] == before
    assert manager.state["eta_seconds"] == 20


def test_eta_recalculates_on_next_completion():
    now = [10.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    manager._handle_worker_event(run, {"event": "progress", "done": 1, "total": 3}, worker, 0)
    now[0] = 30.0
    manager._handle_worker_event(run, {"event": "progress", "done": 2, "total": 3}, worker, 0)
    assert manager.state["eta_seconds"] == 20


def test_resume_updates_done_without_creating_completion_sample():
    manager, run = _eta_manager(lambda: 10.0)
    worker = manager.state["workers"][0]
    manager._handle_worker_event(
        run, {"event": "resume", "done": 1, "total": 3}, worker, 0
    )
    assert manager.state["done"] == 1
    assert worker["hist"] == []
    assert manager.state["eta_seconds"] is None


def test_resume_worker_then_other_worker_progress_does_not_crash():
    now = [0.0]
    manager, run = _eta_manager(lambda: now[0])
    first = manager.state["workers"][0]
    second = _worker(total=3)
    manager.state["workers"].append(second)
    manager._handle_worker_event(run, {"event": "resume", "done": 1, "total": 3}, first, 0)

    now[0] = 10.0
    manager._handle_worker_event(run, {"event": "progress", "done": 1, "total": 3}, second, 0)
    assert manager.state["done"] == 2
    assert manager.state["eta_seconds"] is None
    assert first["hist"] == []

    now[0] = 20.0
    manager._handle_worker_event(run, {"event": "progress", "done": 2, "total": 3}, second, 0)
    assert second["done"] == 2


def test_reader_continues_with_unestimated_resumed_worker():
    now = [0.0]
    manager, run = _eta_manager(lambda: now[0])
    resumed = manager.state["workers"][0]
    active = _worker(total=3)
    manager.state["workers"].append(active)
    manager._handle_worker_event(
        run, {"event": "resume", "done": 1, "total": 3}, resumed, 0
    )

    now[0] = 10.0
    proc = SimpleNamespace(
        stdout=io.StringIO(
            '{"event":"progress","done":1,"total":3}\n'
            'not-json\n'
            '{"event":"progress","done":2,"total":3}\n'
        )
    )
    manager._read_worker(run, proc, active, 0)

    assert active["done"] == 2
    assert len(active["hist"]) == 2
    # The resumed worker is unfinished but has no post-resume observation, so
    # reporting the other worker's estimate would understate total fit time.
    assert manager.state["eta_seconds"] is None


def test_malformed_completion_history_fails_eta_closed_without_stopping_reader():
    now = [10.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    worker["hist"] = [[1], None, [float("nan"), 2.0]]
    proc = SimpleNamespace(
        stdout=io.StringIO(
            '{"event":"progress","done":1,"total":3}\n'
            '{"event":"progress","done":2,"total":3}\n'
        )
    )

    manager._read_worker(run, proc, worker, 0)

    assert worker["done"] == 2
    assert manager.state["done"] == 2
    assert manager.state["eta_seconds"] == 5


def test_first_post_resume_completion_uses_delta_rate():
    now = [0.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    manager._handle_worker_event(run, {"event": "resume", "done": 50, "total": 100}, worker, 0)
    now[0] = 10.0
    manager._handle_worker_event(run, {"event": "progress", "done": 51, "total": 100}, worker, 0)
    assert manager.state["eta_seconds"] == 490


def test_resumed_worker_eta_uses_current_process_recent_window():
    now = [0.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    manager._handle_worker_event(run, {"event": "resume", "done": 50, "total": 100}, worker, 0)
    now[0] = 10.0
    manager._handle_worker_event(run, {"event": "progress", "done": 51, "total": 100}, worker, 0)
    now[0] = 30.0
    manager._handle_worker_event(run, {"event": "progress", "done": 52, "total": 100}, worker, 0)
    assert manager.state["eta_seconds"] == 960


def test_done_event_clears_stale_eta():
    now = [10.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    manager._handle_worker_event(run, {"event": "progress", "done": 1, "total": 3}, worker, 0)
    assert manager.state["eta_seconds"] == 20
    manager._handle_worker_event(run, {"event": "done"}, worker, 0)
    assert manager.state["done"] == 3
    assert manager.state["eta_seconds"] is None


def test_stop_and_success_publication_are_linearized():
    manager = fitting.FitManager()
    run = fitting._FitRun()
    manager._active_run = run
    manager.state = {"state": "running", "eta_seconds": 12}
    manager._emit = lambda: None

    manager.stop()
    manager._publish_terminal(run, success={"state": "done", "eta_seconds": 0})
    assert manager.state["state"] == "stopped"

    second = fitting._FitRun()
    manager._active_run = second
    manager.state = {"state": "running"}
    manager._publish_terminal(second, success={"state": "done", "eta_seconds": 0})
    assert manager.stop()["state"] == "done"
    assert not second.cancel.is_set()


@pytest.mark.parametrize("cancel", [False, True])
def test_popen_failure_respects_concurrent_cancellation(fit_env, monkeypatch, cancel):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    entered = threading.Event()
    release = threading.Event()

    def popen(*args, **kwargs):
        entered.set()
        _wait(release)
        raise OSError("spawn failed")

    monkeypatch.setattr(fitting.subprocess, "Popen", popen)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    _wait(entered)
    if cancel:
        stopper = threading.Thread(target=manager.stop)
        stopper.start()
        assert run.cancel.wait(1)
    release.set()
    if cancel:
        stopper.join()
    _wait(run.done)
    assert manager.state["state"] == ("stopped" if cancel else "error")
    assert run.processes == []
    assert run.heartbeat_thread is None
    assert len(manager.state["workers"]) == 1
    assert manager.state["workers"][0]["state"] == "loading"


def test_eta_first_sample_uses_completion_timestamp_not_heartbeat_elapsed():
    now = [10.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    worker["elapsed"] = 99.0
    manager._handle_worker_event(run, {"event": "progress", "done": 1, "total": 3}, worker, 0)
    assert worker["hist"] == [[1, 10.0]]
    assert manager.state["eta_seconds"] == 20


def test_eta_recent_sliding_window_uses_only_last_ten_completions():
    now = [0.0]
    manager, run = _eta_manager(lambda: now[0])
    worker = manager.state["workers"][0]
    worker["total"] = manager.state["total"] = 20
    for done in range(1, 13):
        now[0] = float(done * done)
        manager._handle_worker_event(
            run, {"event": "progress", "done": done, "total": 20}, worker, 0
        )
    assert len(worker["hist"]) == 10
    first_done, first_time = worker["hist"][0]
    last_done, last_time = worker["hist"][-1]
    expected = round((20 - last_done) / ((last_done - first_done) / (last_time - first_time)), 0)
    assert manager.state["eta_seconds"] == expected


def test_eta_multiple_workers_uses_slowest_remaining_worker():
    manager = fitting.FitManager()
    manager._emit = lambda: None
    manager.state = {
        "workers": [
            {**_worker(1, 3), "hist": [[1, 10.0]], "elapsed": 10.0},
            {**_worker(1, 3), "hist": [[1, 20.0]], "elapsed": 20.0},
        ]
    }
    manager._refresh_totals()
    assert manager.state["done"] == 2
    assert manager.state["eta_seconds"] == 40


def test_aggregate_total_is_recomputed_from_authoritative_workers():
    manager = fitting.FitManager()
    manager._emit = lambda: None
    manager.state = {
        "total": 999,
        "workers": [
            {**_worker(1, 2), "hist": [[1, 10.0]]},
            {**_worker(2, 3), "hist": [[2, 20.0]]},
        ],
    }

    manager._refresh_totals()

    assert manager.state["done"] == 3
    assert manager.state["total"] == 5


@pytest.mark.parametrize(
    "line",
    [
        "not json\n",
        "[]\n",
        "{}\n",
        '{"event":"other"}\n',
        '{"event":"progress","done":"one"}\n',
        '{"event":"progress","done":true,"total":3}\n',
        '{"event":"progress","done":4,"total":3}\n',
    ],
)
def test_worker_output_parser_ignores_malformed_or_unrelated_lines(line):
    manager, run = _eta_manager(lambda: 12.0)
    worker = manager.state["workers"][0]
    proc = SimpleNamespace(stdout=io.StringIO(line))
    manager._read_worker(run, proc, worker, 10.0)
    assert worker["done"] == 0
    assert worker["hist"] == []


def test_corpus_loader_uses_installed_datasets_without_network(monkeypatch):
    import datasets

    sentinel = object()
    monkeypatch.setattr(datasets, "load_dataset", lambda dataset, split=None: sentinel)
    assert fitting._load_split("local/test-corpus") is sentinel


def test_runtime_fitting_event_precedes_jlens_fit(tmp_path, monkeypatch, capsys):
    from scripts import fit_worker

    prompts = tmp_path / "prompts.json"
    prompts.write_text('["prompt"]', encoding="utf-8")
    order = []

    class Adapter:
        n_layers = 2
        d_model = 8

    class Lens:
        n_prompts = 1

        def save(self, path):
            order.append("save")

    monkeypatch.setattr(
        fit_worker.transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(
        fit_worker.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(fit_worker.jlens, "from_hf", lambda *a: Adapter())

    def fit(*args, **kwargs):
        output = capsys.readouterr().out.splitlines()
        assert json.loads(output[-1])["event"] == "fitting"
        order.append("fit")
        return Lens()

    monkeypatch.setattr(fit_worker.jlens, "fit", fit)
    fit_worker.main(
        [
            "--model", "local/model", "--device", "cpu", "--prompts", str(prompts),
            "--checkpoint", str(tmp_path / "checkpoint.pt"), "--out", str(tmp_path / "lens.pt"),
            "--dim-batch", "2",
        ]
    )
    assert order == ["fit", "save"]


def test_ui_auto_dim_batch_help_has_no_numeric_policy():
    source = (fitting.config.ROOT / "ui" / "src" / "App.jsx").read_text(encoding="utf-8")
    tooltip = source.split('title="auto is selected', 1)[1].split('"', 1)[0]
    assert "15" not in tooltip and "4" not in tooltip and "2" not in tooltip


class _HelperBlockedProcess(_Process):
    """A worker that cannot exit until manager-driven termination occurs."""

    def __init__(self, *, stdout="", stderr=None):
        super().__init__(release=threading.Event(), stdout=stdout)
        if stderr is not None:
            self.stderr = stderr
        self.killed = False

    def terminate(self):
        super().terminate()
        self._release.set()

    def kill(self):
        self.killed = True
        self.terminated = True
        self._release.set()


class _RaiseOnceOnFitting:
    def __init__(self):
        self.raised = False

    def __call__(self, state):
        if state.get("phase") == "fitting" and not self.raised:
            self.raised = True
            raise RuntimeError("progress callback failed")


def test_stdout_reader_failure_terminates_live_worker_and_surfaces_error(
    fit_env, monkeypatch
):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release_failure = threading.Event()
    proc = _HelperBlockedProcess()
    proc.stdout = _BlockingStdout(release_failure, '{"event":"fitting"}\n')
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    merge_started = threading.Event()
    monkeypatch.setattr(
        fitting.JacobianLens,
        "load",
        lambda *a: (merge_started.set(), _Lens())[1],
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    _start(manager)
    run = manager._active_run
    assert run is not None
    release_failure.set()

    _wait(run.helper_failure, "stdout helper failure was not signalled")
    _wait(run.done, "helper failure did not unblock the fit")

    assert proc.terminated and proc.waited
    assert manager.state["state"] == "error"
    assert "stdout reader failed" in manager.state["error"]
    assert "progress callback failed" in manager.state["error"]
    assert not merge_started.is_set()
    assert run.heartbeat_stop.is_set()
    assert all(not thread.is_alive() for thread in run.helper_threads)
    assert manager._active_run is None


def test_stdout_reader_failure_user_cancel_has_precedence(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release_termination = threading.Event()

    class DelayedTermination(_HelperBlockedProcess):
        def terminate(self):
            self.terminated = True
            _wait(release_termination)

        def kill(self):
            self.killed = True
            self._release.set()

    release_failure = threading.Event()
    proc = DelayedTermination()
    proc.stdout = _BlockingStdout(release_failure, '{"event":"fitting"}\n')
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    _start(manager)
    run = manager._active_run
    assert run is not None
    release_failure.set()
    _wait(run.helper_failure)
    stopper = threading.Thread(target=manager.stop)
    stopper.start()
    assert run.cancel.wait(1)
    release_termination.set()
    stopper.join()
    _wait(run.done)
    assert manager.state["state"] == "stopped"


class _ExplodingStderr:
    def __init__(self, release=None):
        self.release = release

    def read(self):
        if self.release is not None:
            _wait(self.release)
        raise OSError("stderr pipe failed")


def test_stderr_drainer_failure_terminates_live_worker_and_surfaces_error(
    fit_env, monkeypatch
):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release_failure = threading.Event()
    proc = _HelperBlockedProcess(
        stdout='{"event":"fitting"}\n', stderr=_ExplodingStderr(release_failure)
    )
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    merge_started = threading.Event()
    monkeypatch.setattr(
        fitting.JacobianLens,
        "load",
        lambda *a: (merge_started.set(), _Lens())[1],
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    assert run is not None
    release_failure.set()
    _wait(run.done)
    assert proc.terminated and proc.waited
    assert manager.state["state"] == "error"
    assert "stderr drainer failed" in manager.state["error"]
    assert "stderr pipe failed" in manager.state["error"]
    assert not merge_started.is_set()


def test_first_helper_failure_wins_concurrently():
    manager = fitting.FitManager()
    run = fitting._FitRun()
    barrier = threading.Barrier(3)
    results = []
    results_lock = threading.Lock()

    def record(source, message):
        barrier.wait()
        won = manager._record_helper_failure(
            run, source, RuntimeError(message), worker=0
        )
        with results_lock:
            results.append((source, message, won))

    threads = [
        threading.Thread(target=record, args=("stdout reader", "stdout failure")),
        threading.Thread(target=record, args=("stderr drainer", "stderr failure")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    winners = [result for result in results if result[2]]
    assert len(winners) == 1
    source, message, _ = winners[0]
    with pytest.raises(RuntimeError, match=f"{source}.*{message}"):
        manager._raise_helper_failure(run)


def test_real_subprocess_pipe_backpressure_recovery(fit_env, monkeypatch, tmp_path):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    worker = tmp_path / "stream_worker.py"
    release_file = tmp_path / "release"
    worker.write_text(
        "import json, pathlib, sys\n"
        f"gate = pathlib.Path({str(release_file)!r})\n"
        "while not gate.exists(): pass\n"
        "while True:\n"
        " print(json.dumps({'event': 'fitting'}), flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(fitting, "WORKER", worker)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    _start(manager)
    run = manager._active_run
    assert run is not None
    release_file.touch()
    _wait(run.done, "real streaming child was not terminated")
    assert manager.state["state"] == "error"
    assert "stdout reader failed" in manager.state["error"]
    assert "progress callback failed" in manager.state["error"]
    assert run.processes[0].poll() is not None


def test_helper_failure_during_later_worker_spawn(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["one", "two"])
    release_stdout = threading.Event()
    second_spawned = threading.Event()
    release_second = threading.Event()
    first = _HelperBlockedProcess()
    first.stdout = _BlockingStdout(release_stdout, '{"event":"fitting"}\n')
    second = _HelperBlockedProcess(stdout='{"event":"done"}\n')
    calls = [0]

    def popen(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return first
        second_spawned.set()
        _wait(release_second)
        return second

    monkeypatch.setattr(fitting.subprocess, "Popen", popen)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    manager.start(
        model_id="local/model", source="local/model", n_prompts=2,
        devices=("cuda:0", "cuda:1"), name="fit", dim_batch=2,
        datasets=("local/corpus",),
    )
    run = manager._active_run
    _wait(second_spawned)
    release_stdout.set()
    _wait(run.helper_failure)
    release_second.set()
    _wait(run.done)

    assert first.terminated and first.waited
    assert second.terminated and second.waited
    assert manager.state["state"] == "error"
    assert "stdout reader failed" in manager.state["error"]
    assert all(not thread.is_alive() for thread in run.helper_threads)


def test_manager_lock_available_during_helper_failure_cleanup(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    terminate_entered = threading.Event()
    release_terminate = threading.Event()

    class BlockingTerminate(_HelperBlockedProcess):
        def terminate(self):
            self.terminated = True
            terminate_entered.set()
            _wait(release_terminate)
            self._release.set()

    release_failure = threading.Event()
    proc = BlockingTerminate()
    proc.stdout = _BlockingStdout(release_failure, '{"event":"fitting"}\n')
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    _start(manager)
    run = manager._active_run
    assert run is not None
    release_failure.set()
    _wait(terminate_entered)

    acquired = manager._lock.acquire(timeout=1)
    assert acquired, "helper cleanup held the FitManager lock"
    manager._lock.release()
    release_terminate.set()
    _wait(run.done)
    assert manager.state["state"] == "error"


@pytest.mark.parametrize("cancel", [False, True])
def test_helper_failure_precedes_later_popen_error(
    fit_env, monkeypatch, cancel
):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["one", "two"])
    release_first = threading.Event()
    second_entered = threading.Event()
    release_second = threading.Event()
    first = _HelperBlockedProcess()
    first.stdout = _BlockingStdout(release_first, '{"event":"fitting"}\n')
    calls = [0]

    def popen(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return first
        second_entered.set()
        _wait(release_second)
        raise OSError("second spawn failed")

    monkeypatch.setattr(fitting.subprocess, "Popen", popen)
    merge_started = threading.Event()
    monkeypatch.setattr(
        fitting.JacobianLens,
        "load",
        lambda *a: (merge_started.set(), _Lens())[1],
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    manager.start(
        model_id="local/model", source="local/model", n_prompts=2,
        devices=("cuda:0", "cuda:1"), name="fit", dim_batch=2,
        datasets=("local/corpus",),
    )
    run = manager._active_run
    assert run is not None
    _wait(second_entered)
    release_first.set()
    _wait(run.helper_failure)
    stopper = None
    if cancel:
        stopper = threading.Thread(target=manager.stop)
        stopper.start()
        assert run.cancel.wait(1)
    release_second.set()
    if stopper is not None:
        stopper.join()
    _wait(run.done)

    assert first.terminated and first.waited
    assert not merge_started.is_set()
    assert manager._active_run is None
    if cancel:
        assert manager.state["state"] == "stopped"
    else:
        assert manager.state["state"] == "error"
        assert "stdout reader failed" in manager.state["error"]
        assert "progress callback failed" in manager.state["error"]
        assert not manager.state["error"].startswith("second spawn failed")


def test_generic_exception_uses_existing_helper_failure():
    manager = fitting.FitManager()
    manager._emit = lambda: None
    run = fitting._FitRun()
    manager._active_run = run
    manager.state = {"state": "running", "phase": "loading"}
    manager._record_helper_failure(
        run, "stdout reader", RuntimeError("primary helper failure"), worker=0
    )

    manager._publish_preferred_failure(run, OSError("later orchestration failure"))

    assert manager.state["state"] == "error"
    assert "stdout reader failed" in manager.state["error"]
    assert "primary helper failure" in manager.state["error"]
    assert "later orchestration failure" not in manager.state["error"]


def test_spawn_failure_without_helper_remains_spawn_error(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    spawn_entered = threading.Event()
    release_spawn = threading.Event()

    def popen(*args, **kwargs):
        spawn_entered.set()
        _wait(release_spawn)
        raise OSError("plain spawn failure")

    monkeypatch.setattr(fitting.subprocess, "Popen", popen)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    _start(manager)
    run = manager._active_run
    assert run is not None
    _wait(spawn_entered)
    release_spawn.set()
    _wait(run.done)
    assert manager.state["state"] == "error"
    assert "plain spawn failure" in manager.state["error"]


def test_restart_after_helper_cleanup(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release_failure = threading.Event()
    proc = _HelperBlockedProcess()
    proc.stdout = _BlockingStdout(release_failure, '{"event":"fitting"}\n')
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = _RaiseOnceOnFitting()
    _start(manager, "first")
    first_run = manager._active_run
    assert first_run is not None
    release_failure.set()
    _wait(first_run.done)

    corpus_entered = threading.Event()
    release_corpus = threading.Event()
    monkeypatch.setattr(
        fitting,
        "_load_corpus",
        lambda *a: (corpus_entered.set(), _wait(release_corpus), ["prompt"])[2],
    )
    manager.on_progress = None
    _start(manager, "second")
    second_run = manager._active_run
    assert second_run is not None and second_run is not first_run
    _wait(corpus_entered)
    manager.stop()
    release_corpus.set()
    _wait(second_run.done)


def test_heartbeat_callback_failure_terminates_worker_and_allows_restart(
    fit_env, monkeypatch
):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    release_stdout = threading.Event()
    proc = _HelperBlockedProcess()
    proc.stdout = _BlockingStdout(release_stdout, "")
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    armed = threading.Event()

    def callback(state):
        if armed.is_set() and "elapsed" in state:
            raise RuntimeError("heartbeat callback failed")

    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = callback
    _start(manager)
    run = manager._active_run
    assert run is not None
    armed.set()
    _wait(run.helper_failure, "heartbeat failure was not recorded")
    release_stdout.set()
    _wait(run.done)

    assert proc.terminated and proc.waited
    assert manager.state["state"] == "error"
    assert "heartbeat failed" in manager.state["error"]
    assert "heartbeat callback failed" in manager.state["error"]
    assert manager._active_run is None


def _late_heartbeat_setup(fit_env, monkeypatch, callback):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    proc = _HelperBlockedProcess(
        stdout='{"event":"progress","done":1,"total":1}\n{"event":"done"}\n'
    )
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: proc)
    load_started = threading.Event()
    monkeypatch.setattr(
        fitting.JacobianLens,
        "load",
        lambda *a: (load_started.set(), _Lens())[1],
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    manager.on_progress = callback
    _start(manager)
    run = manager._active_run
    assert run is not None
    return manager, run, proc, load_started


def test_late_heartbeat_failure_blocks_merge_and_success(fit_env, monkeypatch):
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()

    def callback(state):
        if "elapsed" in state and not heartbeat_entered.is_set():
            heartbeat_entered.set()
            _wait(release_heartbeat)
            raise RuntimeError("late heartbeat callback failure")

    manager, run, proc, load_started = _late_heartbeat_setup(
        fit_env, monkeypatch, callback
    )
    _wait(heartbeat_entered)
    proc._release.set()
    assert not load_started.wait(0.1), "merge began before heartbeat quiescence"
    release_heartbeat.set()
    _wait(run.done)

    assert manager.state["state"] == "error"
    assert "heartbeat failed" in manager.state["error"]
    assert "late heartbeat callback failure" in manager.state["error"]
    assert not load_started.is_set()
    assert run.heartbeat_thread is not None
    assert not run.heartbeat_thread.is_alive()
    assert manager._active_run is None


def test_merge_waits_for_heartbeat_thread_to_finish(fit_env, monkeypatch):
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()

    def callback(state):
        if "elapsed" in state and not heartbeat_entered.is_set():
            heartbeat_entered.set()
            _wait(release_heartbeat)

    manager, run, proc, load_started = _late_heartbeat_setup(
        fit_env, monkeypatch, callback
    )
    _wait(heartbeat_entered)
    proc._release.set()
    assert not load_started.wait(0.1)

    acquired = manager._lock.acquire(timeout=1)
    assert acquired, "heartbeat join held the FitManager lock"
    manager._lock.release()
    release_heartbeat.set()
    _wait(run.done)

    assert load_started.is_set()
    assert manager.state["state"] == "done"
    assert not run.heartbeat_thread.is_alive()


def test_late_heartbeat_failure_with_cancel(fit_env, monkeypatch):
    heartbeat_entered = threading.Event()
    release_heartbeat = threading.Event()

    def callback(state):
        if "elapsed" in state and not heartbeat_entered.is_set():
            heartbeat_entered.set()
            _wait(release_heartbeat)
            raise RuntimeError("late heartbeat callback failure")

    manager, run, proc, load_started = _late_heartbeat_setup(
        fit_env, monkeypatch, callback
    )
    _wait(heartbeat_entered)
    proc._release.set()
    manager.stop()
    release_heartbeat.set()
    _wait(run.done)

    assert manager.state["state"] == "stopped"
    assert not load_started.is_set()


def test_no_heartbeat_after_workers_exit(monkeypatch):
    manager = fitting.FitManager(heartbeat_interval=1)
    run = fitting._FitRun()
    proc = _Process(release=threading.Event())
    run.processes.append(proc)
    wait_entered = threading.Event()
    release_wait = threading.Event()
    heartbeat_calls = []

    class ControlledStop:
        def is_set(self):
            return False

        def set(self):
            pass

        def wait(self, timeout):
            wait_entered.set()
            _wait(release_wait)
            return False

    run.heartbeat_stop = ControlledStop()
    monkeypatch.setattr(
        manager, "_heartbeat_once", lambda *a: heartbeat_calls.append(True)
    )
    thread = threading.Thread(target=manager._heartbeat, args=(run, 0))
    thread.start()
    _wait(wait_entered)
    proc.returncode = 0
    release_wait.set()
    thread.join()
    assert heartbeat_calls == []


def test_helper_failure_between_partial_loads_blocks_merge(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["one", "two"])
    processes = iter([
        _Process(stdout='{"event":"done"}\n'),
        _Process(stdout='{"event":"done"}\n'),
    ])
    monkeypatch.setattr(fitting.subprocess, "Popen", lambda *a, **k: next(processes))
    manager = fitting.FitManager(heartbeat_interval=0.01)
    loads = []
    merge_called = threading.Event()

    def load(*args):
        loads.append(args)
        if len(loads) == 1:
            manager._record_helper_failure(
                manager._active_run,
                "stdout reader",
                RuntimeError("failure between partial loads"),
            )
        return _Lens()

    monkeypatch.setattr(fitting.JacobianLens, "load", load)
    monkeypatch.setattr(
        fitting.JacobianLens,
        "merge",
        lambda *a: (merge_called.set(), _Lens())[1],
    )
    manager.start(
        model_id="local/model", source="local/model", n_prompts=2,
        devices=("cuda:0", "cuda:1"), name="fit", dim_batch=2,
        datasets=("local/corpus",),
    )
    run = manager._active_run
    _wait(run.done)
    assert manager.state["state"] == "error"
    assert len(loads) == 1
    assert not merge_called.is_set()


def test_helper_failure_before_lens_save_blocks_publication(fit_env, monkeypatch):
    monkeypatch.setattr(fitting, "_load_corpus", lambda *a: ["prompt"])
    monkeypatch.setattr(
        fitting.subprocess,
        "Popen",
        lambda *a, **k: _Process(stdout='{"event":"done"}\n'),
    )
    manager = fitting.FitManager(heartbeat_interval=0.01)
    saved = []

    def load(*args):
        manager._record_helper_failure(
            manager._active_run,
            "stdout reader",
            RuntimeError("failure before save"),
        )
        return _Lens(saved)

    monkeypatch.setattr(fitting.JacobianLens, "load", load)
    _start(manager)
    run = manager._active_run
    _wait(run.done)
    assert manager.state["state"] == "error"
    assert saved == []
    assert not (fit_env[1] / "fit" / "meta.json").exists()
