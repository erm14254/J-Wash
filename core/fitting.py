import hashlib
import json
import math
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from jlens.lens import JacobianLens

import config
from core.gpus import gpu_stats

FITS_DIR = config.DATA_DIR / "fits"
WORKER = config.ROOT / "scripts" / "fit_worker.py"

# Fit corpus. Any HuggingFace dataset id works as-is: wikitext is the default
# and keeps a dedicated streamed path, every other id goes through the generic
# loader below. Several ids = an equal-parts mix.
DATASET_WIKITEXT = "Salesforce/wikitext-103-raw-v1"

# Seed shared by the row sampling and the mix shuffle: a continued fit that asks
# for skip+n rows deterministically extends the sequence it drew the first n
# from (sample(skip+n) then drop the head).
_SAMPLE_SEED = 1729


def _slug(dataset):
    """Short, filename-safe tag from a dataset id's last path segment, e.g.
    ``heretic-org/Semantic-Harmless`` -> ``semantic-harmless``."""
    tail = dataset.rstrip("/").split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tail).strip("-")[:24] or "dataset"


def _text_column(features):
    """Column to fit on: prefer ``text``, else the first string-valued column."""
    from datasets import Value

    if "text" in features:
        return "text"
    for name, feat in features.items():
        if isinstance(feat, Value) and feat.dtype == "string":
            return name
    raise ValueError(
        "dataset exposes no text column to fit on "
        f"(columns: {', '.join(features) or 'none'})"
    )


def _pack(texts, count, target=350):
    """Pack ``texts`` into ~``target``-char sequences, stopping as soon as
    ``count`` sequences are ready. jlens skips the first 16 positions of every
    sequence as attention sinks, so short unpacked prompts would almost all be
    dropped as too short. Returns fewer than ``count`` only if ``texts`` runs
    out (the caller decides whether that is an error)."""
    packs, cur = [], ""
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        cur = f"{cur}\n\n{text}" if cur else text
        if len(cur) >= target:
            packs.append(cur)
            cur = ""
            if len(packs) >= count:
                return packs
    if cur and len(packs) < count:
        # trailing remainder: keep it so a just-large-enough dataset still fills
        # its quota (jlens tolerates a slightly-short final sequence)
        packs.append(cur)
    return packs


def _load_split(dataset):
    """``dataset``'s ``train`` split, or its first split if it has no ``train``."""
    from datasets import load_dataset

    try:
        return load_dataset(dataset, split="train")
    except ValueError:
        dd = load_dataset(dataset)
        return dd[next(iter(dd))]


def _load_one(dataset, n, skip):
    """``n`` training SEQUENCES from a single ``dataset`` id, skipping the first
    ``skip`` (continue-from: the new sequences must not overlap the base lens's).

    wikitext keeps its historical path (first records >=600 chars, streamed —
    one record already is one sequence). Any other HF dataset is loaded whole,
    its text column shuffled with a fixed seed, then PACKED into ~350-char
    sequences until skip+n are ready (median instruct prompt ~10 tokens, so
    several rows per sequence). ``n``/``skip`` count OUTPUT sequences, so the
    number the user asks for is exactly what the fit iterates over — not source
    rows, whose count varies per dataset."""
    if n <= 0:
        return []
    if dataset == DATASET_WIKITEXT:
        from jlens.examples import load_wikitext_prompts

        # load skip + n then keep the tail: the new sequences don't overlap the
        # base lens's
        return load_wikitext_prompts(skip + n)[skip:]
    import random

    ds = _load_split(dataset)
    col = _text_column(ds.features)
    texts = [r[col] for r in ds]
    random.Random(_SAMPLE_SEED).shuffle(texts)
    packs = _pack(texts, skip + n)
    if len(packs) < skip + n:
        raise ValueError(
            f"{dataset}: {len(texts)} rows pack into only {len(packs)} sequences, "
            f"{skip + n} requested — lower n_prompts"
        )
    return packs[skip:skip + n]


def _load_corpus(datasets, n, skip=0):
    """``n`` training SEQUENCES drawn from ``datasets`` (a list of HF dataset
    ids). A single id loads that dataset; several are mixed in EQUAL parts — n
    and skip are each split across them (the first datasets take the rounding
    remainder) and the union is shuffled so multi-GPU slices stay mixed. Because
    the count is in sequences, ``n`` is exactly what the fit iterates over."""
    datasets = list(datasets)
    if len(datasets) == 1:
        return _load_one(datasets[0], n, skip)
    import random

    k = len(datasets)
    prompts = []
    for i, ds in enumerate(datasets):
        prompts += _load_one(ds, n // k + int(i < n % k), skip // k + int(i < skip % k))
    random.Random(_SAMPLE_SEED).shuffle(prompts)
    return prompts

def dim_batch_for_vram(vram_bytes):
    """Return the conservative automatic dimension batch for VRAM bytes."""
    return 4 if vram_bytes >= 15 * 2**30 else 2


def _default_dim_batch(device):
    """Resolve automatic ``dim_batch`` from the selected device's VRAM."""
    try:
        total = gpu_stats()[int(device.split(":")[1])]["vram_total"]
        return dim_batch_for_vram(total)
    except Exception:
        return 1


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


class _FitCancelled(Exception):
    pass


@dataclass
class _FitRun:
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    heartbeat_stop: threading.Event = field(default_factory=threading.Event)
    process_lock: threading.Lock = field(default_factory=threading.Lock)
    processes: list = field(default_factory=list)
    helper_threads: list = field(default_factory=list)
    reader_threads: list = field(default_factory=list)
    reader_errors: list = field(default_factory=list)
    reader_error_lock: threading.Lock = field(default_factory=threading.Lock)
    heartbeat_thread: threading.Thread | None = None


class FitManager:
    def __init__(self, *, clock=time.perf_counter, heartbeat_interval=2.0):
        self._lock = threading.Lock()
        self._clock = clock
        self._heartbeat_interval = heartbeat_interval
        self._active_run = None
        self.state = {"state": "idle"}
        self.on_progress = None

    def _emit(self):
        if self.on_progress:
            self.on_progress(dict(self.state))

    def start(self, *, model_id, source, n_prompts=100, dtype="bf16", quant=None,
              devices=("cuda:0",), name=None, dim_batch=None,
              max_seq_len=128, source_layers=None, model_revision=None,
              continue_from=None, datasets=(DATASET_WIKITEXT,)):
        with self._lock:
            if self._active_run is not None:
                raise ValueError("a fitting is already in progress")
            if not devices:
                raise ValueError("at least one device required")
            datasets = [d.strip() for d in datasets if d and d.strip()]
            if not datasets:
                raise ValueError("at least one dataset required")
            skip_prompts = 0
            base_lens = None
            if continue_from:
                base_lens = JacobianLens.load(continue_from)
                # new prompts: skip those already seen by the base lens
                skip_prompts = base_lens.n_prompts
                if source_layers is None:
                    source_layers = list(base_lens.source_layers)
            if name is None:
                base = model_id.split("/")[-1]
                if len(datasets) > 1:
                    base += "_mixed"
                elif datasets[0] != DATASET_WIKITEXT:
                    base += "_" + _slug(datasets[0])
                total = n_prompts + skip_prompts
                name = f"{base}_n{total}" if continue_from else f"{base}_n{n_prompts}"
            params = {
                "model_id": model_id,
                "source": source,
                "model_revision": model_revision,
                "dtype": dtype,
                "quant": quant,
                "n_prompts": n_prompts,
                "datasets": datasets,
                "devices": list(devices),
                "dim_batch": dim_batch,
                "max_seq_len": max_seq_len,
                "source_layers": source_layers,
                "continue_from": continue_from,
                "skip_prompts": skip_prompts,
            }
            self.state = {
                "state": "running",
                "name": name,
                "phase": "corpus",
                "total": n_prompts,
                "done": 0,
                "workers": [],
                "eta_seconds": None,
                "started_at": _now(),
                "params": params,
                "error": None,
            }
            run = _FitRun()
            self._active_run = run
        threading.Thread(target=self._run, args=(run, name, params), daemon=True).start()
        return dict(self.state)

    def stop(self):
        with self._lock:
            run = self._active_run
            if run is None:
                return dict(self.state)
            if self.state.get("state") in ("done", "error", "stopped"):
                return dict(self.state)
            run.cancel.set()
            run.heartbeat_stop.set()
            if self._active_run is run and self.state.get("state") == "running":
                self.state["state"] = "stopping"
        with run.process_lock:
            for proc in run.processes:
                if proc.poll() is None:
                    proc.terminate()
        self._emit()
        return dict(self.state)

    def _owns(self, run):
        with self._lock:
            return self._active_run is run

    def _check_cancelled(self, run):
        if run.cancel.is_set() or not self._owns(run):
            raise _FitCancelled()

    def _update(self, run, **changes):
        with self._lock:
            if self._active_run is not run:
                return False
            self.state.update(changes)
        self._emit()
        return True

    def _publish_terminal(self, run, *, success=None, error=None):
        """Linearize cancellation against terminal state publication."""
        with self._lock:
            if self._active_run is not run:
                return False
            if run.cancel.is_set():
                self.state.update(state="stopped", phase="stopped", eta_seconds=None)
            elif success is not None:
                self.state.update(success)
            else:
                self.state.update(state="error", error=str(error), eta_seconds=None)
        self._emit()
        return True

    def _run(self, run, name, params):
        try:
            self._check_cancelled(run)
            job_dir = FITS_DIR / name
            job_dir.mkdir(parents=True, exist_ok=True)
            corpus_path = job_dir / "corpus.json"
            if corpus_path.exists():
                prompts = json.loads(corpus_path.read_text(encoding="utf-8"))
            else:
                prompts = _load_corpus(
                    params.get("datasets", [DATASET_WIKITEXT]),
                    params["n_prompts"],
                    params.get("skip_prompts", 0),
                )
                self._check_cancelled(run)
                corpus_path.write_text(
                    json.dumps(prompts, ensure_ascii=False), encoding="utf-8"
                )
            self._check_cancelled(run)
            devices = params["devices"]
            n = len(prompts)
            if len(devices) == 2:
                cut = int(n * 0.65)
                slices = [prompts[:cut], prompts[cut:]]
            else:
                slices = [prompts]

            started = self._clock()
            launch_plans = []
            for i, (device, chunk) in enumerate(zip(devices, slices)):
                self._check_cancelled(run)
                slice_path = job_dir / f"slice{i}.json"
                if not slice_path.exists():
                    slice_path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
                dim_batch = params["dim_batch"] or _default_dim_batch(device)
                cmd = [
                    sys.executable, "-X", "utf8", str(WORKER),
                    "--model", params["source"],
                    "--device", device,
                    "--dtype", params["dtype"],
                    "--prompts", str(slice_path),
                    "--checkpoint", str(job_dir / f"ckpt{i}.pt"),
                    "--out", str(job_dir / f"lens{i}.pt"),
                    "--dim-batch", str(dim_batch),
                    "--max-seq-len", str(params["max_seq_len"]),
                ]
                if params["quant"]:
                    cmd += ["--quant", params["quant"]]
                if params["source_layers"]:
                    cmd += ["--source-layers", json.dumps(params["source_layers"])]
                worker_state = {
                    "device": device,
                    "done": 0,
                    "total": len(chunk),
                    "dim_batch": dim_batch,
                    "state": "loading",
                    "elapsed": 0.0,
                    # [done, elapsed] of the last 10 updates: the ETA follows the
                    # RECENT pace (throughput can degrade mid-fit, e.g. VRAM
                    # saturated — a global average would then freeze the ETA)
                    "hist": [],
                    "baseline_done": 0,
                    "baseline_elapsed": 0.0,
                }
                launch_plans.append((cmd, worker_state))

            workers = [worker for _, worker in launch_plans]
            if not self._update(
                run,
                phase="loading",
                workers=workers,
                total=sum(worker["total"] for worker in workers),
                done=0,
                eta_seconds=None,
            ):
                raise _FitCancelled()

            for cmd, worker_state in launch_plans:
                self._check_cancelled(run)
                with run.process_lock:
                    self._check_cancelled(run)
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding="utf-8",
                        cwd=str(config.ROOT),
                    )
                    run.processes.append(proc)
                    if run.cancel.is_set():
                        proc.terminate()
                        raise _FitCancelled()
                reader = threading.Thread(
                    target=self._read_worker,
                    args=(run, proc, worker_state, started),
                    daemon=True,
                )
                run.helper_threads.append(reader)
                run.reader_threads.append(reader)
                reader.start()
            self._check_cancelled(run)

            self._check_cancelled(run)
            heartbeat = threading.Thread(
                target=self._heartbeat,
                args=(run, started),
                daemon=True,
            )
            run.heartbeat_thread = heartbeat
            heartbeat.start()
            self._check_cancelled(run)

            stderr_tails = [""] * len(run.processes)

            def drain_err(index, proc):
                data = proc.stderr.read()
                stderr_tails[index] = (data or "")[-2000:]

            drainers = [
                threading.Thread(target=drain_err, args=(i, p), daemon=True)
                for i, p in enumerate(run.processes)
            ]
            run.helper_threads.extend(drainers)
            for t in drainers:
                t.start()
            for proc in run.processes:
                proc.wait()
            # Popen.wait() does not guarantee that Python reader threads have
            # consumed all bytes buffered in stdout.  Drain stdout to EOF before
            # evaluating failures or allowing merge/publication to begin.
            for reader in run.reader_threads:
                reader.join()
            for t in drainers:
                t.join()
            self._check_cancelled(run)
            with run.reader_error_lock:
                reader_errors = list(run.reader_errors)
            if reader_errors:
                index, exc = reader_errors[0]
                raise RuntimeError(f"worker {index} stdout reader failed: {exc}") from exc
            failed = [i for i, p in enumerate(run.processes) if p.returncode != 0]
            if failed:
                detail = " | ".join(stderr_tails[i].strip().splitlines()[-1] if stderr_tails[i].strip() else "?" for i in failed)
                raise RuntimeError(f"worker(s) {failed} failed: {detail}")

            self._validate_workers_after_drain(run)
            partials = []
            for i in range(len(slices)):
                self._check_cancelled(run)
                partials.append(JacobianLens.load(str(job_dir / f"lens{i}.pt")))
            self._check_cancelled(run)
            merged = JacobianLens.merge(partials) if len(partials) > 1 else partials[0]
            if params.get("continue_from"):
                self._check_cancelled(run)
                base_lens = JacobianLens.load(params["continue_from"])
                if base_lens.source_layers != merged.source_layers:
                    raise RuntimeError(
                        "cannot continue: the source layers differ from the base lens "
                        f"({base_lens.source_layers[0]}..{base_lens.source_layers[-1]} vs "
                        f"{merged.source_layers[0]}..{merged.source_layers[-1]})"
                    )
                # weighted average by n_prompts = equivalent to a fit over the union
                self._check_cancelled(run)
                merged = JacobianLens.merge([base_lens, merged])
            self._check_cancelled(run)
            out_dir = config.LENSES_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            lens_path = out_dir / "lens.pt"
            self._check_cancelled(run)
            merged.save(str(lens_path))
            self._check_cancelled(run)
            meta = {
                "name": name,
                "model_id": params["model_id"],
                "model_revision": params["model_revision"],
                "model_source": params["source"],
                "d_model": merged.d_model,
                "source_layers": [merged.source_layers[0], merged.source_layers[-1]],
                "dtype": params["dtype"],
                "quant": params["quant"],
                "n_prompts": merged.n_prompts,
                "corpus": (
                    "mixed: " + " + ".join(params["datasets"]) + " (equal parts)"
                    if len(params["datasets"]) > 1
                    else params["datasets"][0]
                ),
                "max_seq_len": params["max_seq_len"],
                "devices": params["devices"],
                "continued_from": params.get("continue_from"),
                "config_hash": hashlib.sha1(
                    json.dumps(params, sort_keys=True).encode()
                ).hexdigest()[:16],
                "created_at": _now(),
                "fit_seconds": round(self._clock() - started, 1),
            }
            self._check_cancelled(run)
            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            self._publish_terminal(run, success={
                "state": "done",
                "phase": "done",
                "lens_path": str(lens_path),
                "meta": meta,
                "eta_seconds": 0,
            })
        except _FitCancelled:
            self._publish_terminal(run)
        except Exception as exc:
            self._publish_terminal(run, error=exc)
        finally:
            run.heartbeat_stop.set()
            with run.process_lock:
                for proc in run.processes:
                    if proc.poll() is None:
                        proc.terminate()
            for proc in run.processes:
                # ``wait`` is required even when terminate made ``poll`` turn
                # non-None immediately: the child still needs to be reaped.
                proc.wait()
            heartbeat = run.heartbeat_thread
            if heartbeat is not None and heartbeat is not threading.current_thread():
                heartbeat.join()
            for thread in run.helper_threads:
                if thread is not threading.current_thread():
                    thread.join()
            with self._lock:
                if self._active_run is run:
                    self._active_run = None
            run.done.set()

    def _heartbeat(self, run, started):
        """Emit elapsed-time updates while at least one fit worker is alive."""
        while not run.heartbeat_stop.is_set():
            if run.processes and not any(proc.poll() is None for proc in run.processes):
                return
            if run.heartbeat_stop.wait(self._heartbeat_interval):
                return
            self._heartbeat_once(run, started)

    def _heartbeat_once(self, run, started):
        with self._lock:
            if self._active_run is not run or run.cancel.is_set():
                return
            elapsed = round(self._clock() - started, 1)
            self.state["elapsed"] = elapsed
            for worker in self.state.get("workers", []):
                if worker.get("state") in ("loading", "fitting"):
                    worker["elapsed"] = elapsed
        self._emit()

    def _read_worker(self, run, proc, worker_state, started):
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    continue
                self._handle_worker_event(run, event, worker_state, started)
        except Exception as exc:
            try:
                index = run.processes.index(proc)
            except ValueError:
                index = -1
            with run.reader_error_lock:
                run.reader_errors.append((index, exc))

    def _handle_worker_event(self, run, event, worker_state, started):
        with self._lock:
            if (
                self._active_run is not run
                or run.cancel.is_set()
                or self.state.get("state") in ("done", "error", "stopped")
            ):
                return
            if event["event"] == "loading":
                worker_state["state"] = "loading"
            elif event["event"] == "fitting":
                worker_state["state"] = "fitting"
                self.state["phase"] = "fitting"
            elif event["event"] in ("progress", "resume"):
                done = event.get("done")
                total = event.get("total")
                if (
                    not isinstance(done, int)
                    or isinstance(done, bool)
                    or not isinstance(total, int)
                    or isinstance(total, bool)
                    or done < 0
                    or total < 0
                    or done > total
                ):
                    return
                worker_state["state"] = "fitting"
                self.state["phase"] = "fitting"
                worker_state["done"] = done
                worker_state["total"] = total
                worker_state["elapsed"] = round(self._clock() - started, 1)
                if event["event"] == "resume":
                    worker_state["baseline_done"] = done
                    worker_state["baseline_elapsed"] = worker_state["elapsed"]
                    worker_state["hist"] = []
                else:
                    hist = worker_state.setdefault("hist", [])
                    hist.append([done, worker_state["elapsed"]])
                    del hist[:-10]
            elif event["event"] == "done":
                worker_state["state"] = "done"
                worker_state["done"] = worker_state["total"]
            else:
                return
            self._refresh_totals(
                recalculate_eta=event["event"] in ("progress", "done")
            )
        self._emit()

    def _validate_workers_after_drain(self, run):
        """Validate authoritative worker state before entering merge."""
        with self._lock:
            if self._active_run is not run or run.cancel.is_set():
                raise _FitCancelled()
            workers = self.state.get("workers", [])
            if len(workers) != len(run.processes):
                raise RuntimeError("worker process/state inventory is incomplete")
            self._refresh_totals(recalculate_eta=True)
            incomplete = [
                index for index, worker in enumerate(workers)
                if worker.get("state") != "done"
            ]
            if incomplete:
                raise RuntimeError(
                    "worker(s) did not emit a terminal done event: "
                    + ", ".join(map(str, incomplete))
                )
            self.state["eta_seconds"] = None
            self.state["phase"] = "merge"
        self._emit()

    def _refresh_totals(self, *, recalculate_eta=True):
        workers = self.state.get("workers", [])
        self.state["done"] = sum(w["done"] for w in workers)
        if not recalculate_eta:
            return
        etas = []
        unknown_unfinished = False
        for w in workers:
            if w["done"] >= w["total"]:
                continue
            hist = [
                sample for sample in (w.get("hist") or [])
                if (
                    isinstance(sample, (list, tuple))
                    and len(sample) == 2
                    and _finite_number(sample[0])
                    and _finite_number(sample[1])
                )
            ]
            if len(hist) >= 2 and hist[-1][1] > hist[0][1] and hist[-1][0] > hist[0][0]:
                # pace over the last 10 updates (sliding window)
                rate = (hist[-1][0] - hist[0][0]) / (hist[-1][1] - hist[0][1])
            elif hist:
                progress_delta = hist[-1][0] - w.get("baseline_done", 0)
                elapsed_delta = hist[-1][1] - w.get("baseline_elapsed", 0.0)
                rate = progress_delta / elapsed_delta if progress_delta > 0 and elapsed_delta > 0 else 0
            else:
                rate = 0
            if not rate or not math.isfinite(rate):
                unknown_unfinished = True
                continue
            etas.append((w["total"] - w["done"]) / rate)
        # multi-GPU: the fit ETA = the slowest worker
        self.state["eta_seconds"] = (
            None if unknown_unfinished or not etas else round(max(etas), 0)
        )
        try:
            self.state["vram"] = [
                {"index": g["index"], "used_gb": round(g["vram_used"] / 2**30, 1)}
                for g in gpu_stats()
            ]
        except Exception:
            # Progress reporting must survive transient NVML failures. Device
            # telemetry is informative and does not affect fitting.
            pass
