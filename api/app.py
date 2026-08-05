import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import mimetypes
import os
import re
import threading
from threading import Event as ThreadingEvent
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core import editing, registry
from core import capabilities
from core.ablation import Interventions
from core.neighbors import TokenNeighbors
from core import fitting
from core.fitting import FitManager
from core.gpus import gpu_stats
from core.lens_manager import LensManager
from core.model_session import GenerationContext, LoadedModelBundle, OperationConflict, OperationType, ModelStateError, WorkerDispatch, WorkerOutcome
from core.model_manager import (
    ModelManager,
    _resolve_revision,
    resolve_local_dir,
    resolve_source,
)
from core.store import Store, FrameStorageError, StaleMessageUpdate, FramesNotAttached, FramePointerTurnover, FrameFileMissing

manager = ModelManager()
lens_manager = LensManager()
store = Store()
fit_manager = FitManager()
interventions = Interventions()
neighbors = TokenNeighbors()
_transition_intervention_rollback = None


def _prepare_model_transition_interventions(target_session_id):
    global _transition_intervention_rollback
    previous = interventions.state_record()
    try:
        interventions.clear_for_model_transition()
        snapshot = interventions.snapshot()
        snapshot["model_session_id"] = target_session_id
        snapshot["lens_binding_id"] = None
        _transition_intervention_rollback = previous
        return snapshot
    except Exception:
        interventions.restore_state_record(previous)
        raise


def _rollback_model_transition_interventions():
    global _transition_intervention_rollback
    previous, _transition_intervention_rollback = _transition_intervention_rollback, None
    if previous is not None:
        interventions.restore_state_record(previous)


def _cleanup_model_bound_state(_session_id=None, token=None):
    global _transition_intervention_rollback
    _transition_intervention_rollback = None
    errors = []
    old_lens = None
    try:
        old_lens = lens_manager.withdraw()
        manager.coordinator.clear_lens_binding(token)
    except Exception as exc:
        errors.append(exc)
    try:
        neighbors.reset()
    except Exception as exc:
        errors.append(exc)
    # Intervention clearing is prepared and installed atomically with the
    # authoritative coordinator withdrawal.  This cleanup callback only owns
    # resource/cache cleanup after that transition has linearized.
    cleanup_error = lens_manager.cleanup_withdrawn(old_lens)
    if cleanup_error is not None:
        errors.append(cleanup_error)
    if errors:
        first = errors[0]
        logging.getLogger(__name__).warning(
            "model-bound cleanup completed with %d non-fatal error(s)", len(errors),
            exc_info=(type(first), first, first.__traceback__),
        )


manager.on_model_withdraw = _cleanup_model_bound_state
manager.on_model_transition_prepare = _prepare_model_transition_interventions
manager.on_model_transition_rollback = _rollback_model_transition_interventions

_ws_locks = {}
_loop_holder = {}
_deferred_worker_tasks = set()




@dataclass(frozen=True)
class RouteOutcome:
    value: object | None = None
    status: int | None = None
    message: str | None = None


def _route_value(value=None):
    return RouteOutcome(value=value)


def _route_http(status, message):
    return RouteOutcome(status=status, message=str(message))


def _raise_route_outcome(outcome: RouteOutcome):
    if outcome.status is not None:
        raise HTTPException(outcome.status, outcome.message or "request failed") from None
    return outcome.value


class PreparedWorkerPayload:
    def __init__(self, payload):
        self._payload = payload

    def take(self):
        payload, self._payload = self._payload, None
        if payload is None:
            raise RuntimeError("prepared payload already transferred")
        return payload


def _route_exception(status, exc):
    message = str(exc)
    try:
        exc.__traceback__ = None
        exc.__context__ = None
        exc.__cause__ = None
    except Exception:
        pass
    return _route_http(status, message)

def _worker_success(value=None):
    return WorkerOutcome(value=value)


def _worker_failure(exc, *, default_status=500, value_status=422):
    if isinstance(exc, HTTPException):
        return WorkerOutcome(
            failure_kind="http",
            failure_message=str(exc.detail),
            http_status=exc.status_code,
        )
    status = value_status if isinstance(exc, ValueError) else default_status
    return WorkerOutcome(
        failure_kind=exc.__class__.__name__,
        failure_message=str(exc),
        http_status=status,
    )


def _raise_worker_outcome(outcome: WorkerOutcome):
    if outcome is None:
        return None
    if not isinstance(outcome, WorkerOutcome):
        return outcome
    if outcome.ok:
        return outcome.value
    raise HTTPException(outcome.http_status or 500, outcome.failure_message or "model worker failed") from None


def _coordinated_interventions(snap):
    if snap.interventions is None:
        raise HTTPException(500, "coordinated intervention state unavailable")
    return snap.interventions


async def _retain_worker_task_or_cancel(task, dispatch):
    try:
        _retain_worker_task(task)
    except Exception as exc:
        dispatch.cancel_from_awaiter()
        original = exc
        try:
            await _drain_worker_uninterruptibly(task)
        except asyncio.CancelledError:
            raise original from None
        raise original from None
    return task


def _retain_worker_task(task):
    _deferred_worker_tasks.add(task)

    def _done(done_task):
        try:
            if not done_task.cancelled():
                done_task.exception()
        except Exception:
            logging.getLogger(__name__).exception("background model worker failed after awaiter cancellation")
        finally:
            _deferred_worker_tasks.discard(done_task)

    try:
        task.add_done_callback(_done)
    except Exception:
        _deferred_worker_tasks.discard(task)
        raise




async def _drain_worker_uninterruptibly(task):
    async def _drain():
        return await asyncio.gather(asyncio.shield(task), return_exceptions=True)
    drain = asyncio.create_task(_drain())
    cancelled = False
    try:
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                cancelled = True
        result = drain.result()
    finally:
        if not drain.done():
            drain.cancel()
            await asyncio.gather(drain, return_exceptions=True)
    if cancelled:
        raise asyncio.CancelledError()
    return result

def _valid_devices():
    """Accepted devices = "auto" + one cuda:N per GPU actually present.
    Adaptive: no longer assumes the personal 2-GPU (cuda:0/cuda:1) setup."""
    try:
        n = len(gpu_stats())
    except Exception:
        n = 0
    return {"auto"} | {f"cuda:{i}" for i in range(n)}


class _QuietPolling(logging.Filter):
    """Drops the access lines from the UI polling (GET /api/status every 2 s)."""

    def filter(self, record):
        return "GET /api/status " not in record.getMessage()


@asynccontextmanager
async def lifespan(_app):
    loop = asyncio.get_running_loop()
    access_logger = logging.getLogger("uvicorn.access")
    quiet_filter = _QuietPolling()
    _loop_holder["loop"] = loop
    access_logger.addFilter(quiet_filter)
    try:
        try:
            inspection = await asyncio.to_thread(editing.inspect_abandoned_export_temps)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.getLogger(__name__).exception(
                "abandoned export temporary inspection failed; startup will continue"
            )
        else:
            logging.getLogger(__name__).info(
                "export temporary inspection: abandoned=%d active=%d recent=%d completed=%d unsafe=%d errors=%d",
                len(inspection["abandoned"]), len(inspection["active"]),
                len(inspection["recent"]), len(inspection["completed"]),
                len(inspection["changed_or_unsafe"]), len(inspection["errors"]),
            )
            for failure in inspection["errors"]:
                logging.getLogger(__name__).warning(
                    "could not safely inspect export temporary artifact %s: %s",
                    failure["path"], failure["error"],
                )
        yield
    finally:
        access_logger.removeFilter(quiet_filter)
        if _loop_holder.get("loop") is loop:
            _loop_holder.pop("loop", None)


app = FastAPI(title="J-Wash", lifespan=lifespan)


async def _ws_send(ws, text):
    lock = _ws_locks.get(ws)
    if lock is None:
        return
    async with lock:
        await ws.send_text(text)


def _broadcast_fit(state):
    loop = _loop_holder.get("loop")
    if loop is None:
        return
    payload = json.dumps({"type": "fit_progress", "fit": state})
    for ws in list(_ws_locks):
        asyncio.run_coroutine_threadsafe(_ws_send(ws, payload), loop)


fit_manager.on_progress = _broadcast_fit

# concurrent HF downloads: one state per repo_id
_downloads = {}
_downloads_lock = threading.Lock()
# last synchronous /api/generate exchange (CLI/MCP): surfaced in /api/status so
# the UI can show what an external client generates, without persisting it
_generation_counter = 0


class LoadRequest(BaseModel):
    model_id: str
    dtype: str = config.DEFAULT_DTYPE
    quant: str | None = None
    device: str = config.DEFAULT_DEVICE


class DownloadRequest(BaseModel):
    repo_id: str


class DeleteModelRequest(BaseModel):
    model_id: str


class LensLoadRequest(BaseModel):
    repo_id: str | None = None
    filename: str = "lens.pt"
    revision: str | None = None
    path: str | None = None
    layers: list[int] | None = None
    k: int = 8


class LensLayersRequest(BaseModel):
    layers: list[int]
    k: int | None = None


class PinRequest(BaseModel):
    gen_id: int
    token_ids: list[int]


class ConversationPatch(BaseModel):
    title: str | None = None
    tags: list[str] | None = None


class InterventionRequest(BaseModel):
    token_id: int
    mode: str = "scale"
    factor: float = 0.0
    replacement_id: int | None = None
    layers: list[int] | None = None


_NAME_RE = re.compile(r"[\w][\w.\- ]*", re.UNICODE)


def _safe_name(name):
    """Validate a user-supplied file/folder name (presets, exports): plain
    names only — no separators, no traversal."""
    name = (name or "").strip()
    if not name or ".." in name or not _NAME_RE.fullmatch(name):
        raise HTTPException(
            422, f"invalid name {name!r}: letters, digits, spaces, . - _ only"
        )
    return name


class InterventionPatch(BaseModel):
    factor: float | None = None
    layers: list[int] | None = None
    enabled: bool | None = None
    token_id: int | None = None
    replacement_id: int | None = None
    mode: str | None = None  # scale | replace


class InterventionsScale(BaseModel):
    scale: float | None = None
    mode: str | None = None


class ExportRequest(BaseModel):
    format: str = "layers"
    name: str


class FitRequest(BaseModel):
    model_id: str
    dtype: str = "bf16"
    quant: str | None = None
    n_prompts: int = 100
    datasets: list[str] = [fitting.DATASET_WIKITEXT]  # any HF ids; several = equal-parts mix
    devices: list[str] = ["cuda:0"]
    name: str | None = None
    dim_batch: int | None = None
    max_seq_len: int = 128
    source_layers: list[int] | None = None
    continue_from: str | None = None


@app.get("/api/models")
def api_models():
    return {"models": manager.list_models()}


@app.get("/api/status")
def api_status():
    coord = manager.coordinator.status_snapshot()
    iv = coord.interventions or {}
    scale = iv.get("scale", 1.0)
    mode = iv.get("mode", "standard")
    loaded = coord.loaded
    profile = coord.capability_profile
    meta = dict(coord.model_meta or {})
    capability_snapshot = capabilities.snapshot(profile, mode, loaded=loaded)
    loaded_meta = ({**meta, **capabilities.legacy(profile, loaded=True)} if loaded else None)
    return {
        "loaded": loaded_meta,
        "capabilities": capability_snapshot,
        "busy": None if coord.operation is None else coord.operation.type,
        "model_session_id": coord.model_session_id,
        "model_state": coord.model_state,
        "operation": None if coord.operation is None else {
            "id": coord.operation.id,
            "type": coord.operation.type,
            "acquired_session": coord.operation.acquired_session,
            "current_session": coord.operation.current_session,
            "start_timestamp": coord.operation.start_timestamp,
            "cancellation_requested": coord.operation.cancellation_requested,
        },
        "lens": dict(coord.lens) if hasattr(coord.lens, "items") else coord.lens,
        "gpus": gpu_stats(),
        "downloads": list(_downloads.values()),
        "convert": _convert_state,
        "fit": fit_manager.state,
        "gguf": dict(_gguf_state),
        "interventions": iv.get("summary", []),
        "interventions_scale": scale,
        "interventions_mode": mode,
        "last_generation": coord.last_generation,
    }


def _conflict(exc):
    return HTTPException(409, str(exc))


def _publish_intervention_snapshot(token, snap):
    data = interventions.snapshot()
    data["model_session_id"] = snap.model_session_id
    lens = manager.coordinator.status_snapshot().lens
    data["lens_binding_id"] = (lens or {}).get("lens_binding_id") if hasattr(lens or {}, "get") else None
    manager.coordinator.update_interventions(token, data)
    return data


def _run_intervention_transaction(token, snap, mutate):
    previous = interventions.state_record()
    try:
        result = mutate()
        published = _publish_intervention_snapshot(token, snap)
        return result, published
    except Exception:
        interventions.restore_state_record(previous)
        raise

@app.post("/api/load")
async def api_load(req: LoadRequest):
    if req.dtype not in config.DTYPES:
        raise HTTPException(422, f"invalid dtype: {req.dtype}")
    if req.quant is not None and req.quant not in config.QUANTS:
        raise HTTPException(422, f"invalid quant: {req.quant}")
    if req.device not in _valid_devices():
        raise HTTPException(422, f"invalid device: {req.device}")
    try:
        return await asyncio.to_thread(
            manager.load, req.model_id, req.dtype, req.quant, req.device
        )
    except OperationConflict as exc:
        raise _conflict(exc)
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/models/delete")
async def api_delete_model(req: DeleteModelRequest):
    from core.model_manager import delete_model
    stop_event = ThreadingEvent()
    token = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_DELETE, include_bundle=False)
        status = manager.coordinator.status_snapshot()
        loaded_model_id = (status.model_meta or {}).get("model_id") if status.model_meta is not None else None
        snap = status = None
        if loaded_model_id == req.model_id:
            manager.coordinator.release(token)
            token = None
            raise HTTPException(409, "unload this model before deleting it")
    except OperationConflict as exc:
        raise _conflict(exc)
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    try:
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event)
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise

    model_id = req.model_id

    def worker():
        if not dispatch.claim():
            return _worker_success()
        try:
            try:
                return _worker_success(delete_model(model_id))
            except Exception as exc:
                return _worker_failure(exc, default_status=500, value_status=400)
        finally:
            dispatch.release_from_worker()

    try:
        task = asyncio.create_task(asyncio.to_thread(worker))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(task, dispatch)
    try:
        return _raise_worker_outcome(await asyncio.shield(task))
    except asyncio.CancelledError:
        dispatch.cancel_from_awaiter()
        raise
    except HTTPException:
        raise


# --- user settings (Options tab) --------------------------------------------
SETTINGS_PATH = config.DATA_DIR / "settings.json"
SETTINGS_DEFAULTS = {
    "default_quant": "",       # '', 'int8' or 'nf4' — preselected in the Model tab
    "auto_layer_radius": 2,    # editor: peak ± radius when auto-selecting layers
    "chat_markdown": True,     # render assistant replies as markdown
    "hf_cache": "",            # HF cache dir — applied at startup (--hf-cache wins)
    "llamacpp_dir": "",        # llama.cpp folder → enables the direct GGUF export
}


def read_settings():
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    return {**SETTINGS_DEFAULTS, **{k: v for k, v in data.items() if k in SETTINGS_DEFAULTS}}


class SettingsPatch(BaseModel):
    default_quant: str | None = None
    auto_layer_radius: int | None = None
    chat_markdown: bool | None = None
    hf_cache: str | None = None
    llamacpp_dir: str | None = None


@app.get("/api/settings")
def api_settings():
    return read_settings()


@app.patch("/api/settings")
def api_settings_patch(req: SettingsPatch):
    if req.default_quant is not None and req.default_quant not in ("", "int8", "nf4"):
        raise HTTPException(422, f"invalid quant: {req.default_quant}")
    current = read_settings()
    for key, value in req.model_dump(exclude_none=True).items():
        if key == "auto_layer_radius":
            value = max(0, min(8, int(value)))
        current[key] = value
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return current


class RegisterModelRequest(BaseModel):
    path: str


@app.post("/api/models/register")
def api_models_register(req: RegisterModelRequest):
    """Add a model folder to the available list (no copy — just remembered)."""
    from core.model_manager import register_model_dir
    try:
        return register_model_dir(req.path)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/models/unregister")
def api_models_unregister(req: RegisterModelRequest):
    """Forget a registered entry; the model files are left untouched."""
    from core.model_manager import unregister_model_dir
    try:
        return unregister_model_dir(req.path)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.post("/api/unload")
async def api_unload():
    try:
        return await asyncio.to_thread(manager.unload)
    except OperationConflict as exc:
        raise _conflict(exc)


@app.post("/api/lens/load")
async def api_lens_load(req: LensLoadRequest):
    if not req.repo_id and not req.path:
        raise HTTPException(422, "repo_id or path required")
    token = None
    stop_event = ThreadingEvent()
    try:
        token, snap = manager.coordinator.acquire(OperationType.LENS_UPDATE, requires_loaded=True, include_bundle=False)
        snap = None
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    try:
        payload = {
            "repo_id": req.repo_id,
            "filename": req.filename,
            "revision": req.revision,
            "path": req.path,
            "layers": req.layers,
            "k": req.k,
        }
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    payload = None

    def worker():
        if not dispatch.claim():
            return _worker_success()
        payload = dispatch.take_payload()
        old_state = None
        try:
            try:
                old_state = lens_manager.state_record()
                result = lens_manager.load(
                    manager,
                    repo_id=payload["repo_id"],
                    filename=payload["filename"],
                    revision=payload["revision"],
                    path=payload["path"],
                    layers=payload["layers"],
                    k=payload["k"],
                )
                manager.coordinator.update_lens_binding(token, result)
                return _worker_success(result)
            except Exception as exc:
                if old_state is not None:
                    try:
                        lens_manager.restore_state_record(old_state)
                    except Exception:
                        logging.getLogger(__name__).exception("failed to restore lens state after worker failure")
                return _worker_failure(exc)
        finally:
            payload = old_state = result = None
            dispatch.release_from_worker()

    try:
        task = asyncio.create_task(asyncio.to_thread(worker))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(task, dispatch)
    try:
        return _raise_worker_outcome(await asyncio.shield(task))
    except asyncio.CancelledError:
        dispatch.cancel_from_awaiter()
        raise
    except HTTPException:
        raise


@app.post("/api/lens/unload")
def api_lens_unload():
    token = None
    old = None
    try:
        token, _snap = manager.coordinator.acquire(OperationType.LENS_UPDATE, include_bundle=False)
        old = lens_manager.withdraw()
        try:
            manager.coordinator.clear_lens_binding(token)
        except Exception:
            lens_manager.restore_withdrawn(old)
            raise
        cleanup_error = lens_manager.cleanup_withdrawn(old)
        old = None
        if cleanup_error is not None:
            logging.getLogger(__name__).warning(
                "lens cleanup failed after authoritative unload",
                exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
            )
        return {"unloaded": True}
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        if token is not None:
            manager.coordinator.release(token)


@app.post("/api/lens/layers")
def api_lens_layers(req: LensLayersRequest):
    token = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.LENS_UPDATE, requires_loaded=True, include_bundle=False)
        outcome = _lens_layers_resource(token, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _lens_layers_resource(token, req):
    old_state = result = None
    try:
        old_state = lens_manager.state_record()
        result = lens_manager.set_layers(manager, req.layers, k=req.k)
        try:
            manager.coordinator.update_lens_binding(token, result)
        except Exception:
            lens_manager.restore_state_record(old_state)
            raise
        return _route_value(dict(result))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return _route_http(exc.status_code, exc.detail)
        return _route_exception(500, exc)
    finally:
        old_state = result = None


@app.get("/api/interventions")
def api_interventions():
    snap = manager.coordinator.status_snapshot()
    iv = snap.interventions or {}
    return {"rules": list(iv.get("summary") or [])}


@app.post("/api/interventions")
def api_interventions_add(req: InterventionRequest):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
        outcome = _interventions_add_resource(token, snap, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_add_resource(token, snap, req):
    bundle = rules = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap)
        if lens_manager.lens is None:
            return _route_http(422, "model and lens required")
        capabilities.require(bundle.capability_profile, "modes", "standard", loaded=True)
        def mutate():
            return interventions.add(
                lens_manager,
                bundle.jl,
                token_id=req.token_id,
                mode=req.mode,
                factor=req.factor,
                replacement_id=req.replacement_id,
                layers=req.layers,
            )
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = rules = mutate = None


@app.patch("/api/interventions/{rule_id}")
def api_interventions_patch(rule_id: int, req: InterventionPatch):
    needs_dirs = any(
        x is not None for x in (req.layers, req.token_id, req.replacement_id, req.mode)
    )
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(
            OperationType.INTERVENTION_UPDATE, requires_loaded=True if needs_dirs else None
        )
        outcome = _interventions_patch_resource(token, snap, rule_id, req, needs_dirs)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_patch_resource(token, snap, rule_id, req, needs_dirs):
    bundle = rules = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap) if needs_dirs else None
        def mutate():
            return interventions.update(
                rule_id,
                factor=req.factor,
                layers=req.layers,
                enabled=req.enabled,
                token_id=req.token_id,
                replacement_id=req.replacement_id,
                mode=req.mode,
                lens_manager=lens_manager if needs_dirs else None,
                jl=bundle.jl if needs_dirs else None,
            )
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except ValueError as exc:
        return _route_http(404, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = rules = mutate = None


@app.patch("/api/interventions")
def api_interventions_scale(req: InterventionsScale):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE)
        outcome = _interventions_scale_resource(token, snap, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_scale_resource(token, snap, req):
    try:
        if req.mode is not None:
            status = manager.coordinator.status_snapshot()
            capabilities.require(status.capability_profile, "modes", req.mode,
                                 loaded=status.loaded)
        def mutate():
            return interventions.set_scale_and_mode(scale=req.scale, mode=req.mode)
        (scale, mode), _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"scale": scale, "mode": mode})
    except ValueError as exc:
        return _route_http(422, str(exc))
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        mutate = None


@app.delete("/api/interventions/{rule_id}")
def api_interventions_remove(rule_id: int):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE)
        outcome = _interventions_remove_resource(token, snap, rule_id)
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_remove_resource(token, snap, rule_id):
    rules = None
    try:
        def mutate():
            return interventions.remove(rule_id)
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        rules = mutate = None


@app.delete("/api/interventions")
def api_interventions_clear():
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE)
        outcome = _interventions_clear_resource(token, snap)
    except OperationConflict as exc:
        raise _conflict(exc)
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _interventions_clear_resource(token, snap):
    rules = None
    try:
        def mutate():
            return interventions.remove()
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({"rules": list(rules or [])})
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        rules = mutate = None


def _stable_lens_descriptor(lens_binding):
    if lens_binding is None:
        return {
            "schema_version": 1,
            "source_kind": "none",
            "repo_id": None,
            "path": None,
            "filename": None,
            "revision": None,
            "model_id": None,
            "model_revision": None,
            "fitted_layers": [],
            "tapped_layers": [],
            "k": None,
            "runtime_lens_binding_id": None,
            "runtime_model_session_id": None,
        }
    lens = dict(lens_binding)
    path = lens.get("path")
    repo_id = lens.get("repo_id")
    source_kind = "local" if path else ("hub" if repo_id else "none")

    def int_list(value):
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return []
        try:
            return [int(v) for v in value]
        except TypeError:
            return []

    fitted = int_list(lens.get("fitted_layers_all"))
    if not fitted:
        fitted = int_list(lens.get("fitted_layers"))
    return {
        "schema_version": 1,
        "source_kind": source_kind,
        "repo_id": repo_id,
        "path": os.path.normpath(str(path)) if path else None,
        "filename": lens.get("filename") or ("lens.pt" if repo_id else None),
        "revision": lens.get("revision"),
        "model_id": lens.get("model_id"),
        "model_revision": lens.get("model_revision"),
        "fitted_layers": fitted,
        "tapped_layers": int_list(lens.get("tapped_layers")),
        "k": int(lens["k"]) if lens.get("k") is not None else None,
        "runtime_lens_binding_id": lens.get("lens_binding_id"),
        "runtime_model_session_id": lens.get("model_session_id"),
    }


def _lens_descriptor_mismatch(saved, current):
    if saved is None:
        return False
    current = current or _stable_lens_descriptor(None)
    if not isinstance(saved, dict) or saved.get("schema_version") != 1:
        return True
    if not isinstance(current, dict) or current.get("schema_version") != 1:
        return True
    stable_keys = (
        "source_kind", "repo_id", "path", "filename", "revision",
        "model_id", "model_revision", "fitted_layers", "tapped_layers", "k",
    )
    return any(saved.get(key) != current.get(key) for key in stable_keys)


def _lens_descriptor_is_legacy(value):
    return value is not None and (not isinstance(value, dict) or value.get("schema_version") != 1)


@app.get("/api/presets")
def api_presets():
    return {"presets": editing.list_presets()}


@app.post("/api/presets/{name}")
def api_presets_save(name: str):
    name = _safe_name(name)
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_READ, requires_loaded=True, include_bundle=True)
        bundle = snap.bundle
        iv = _coordinated_interventions(snap)
        rules = [dict(rule) for rule in (iv.get("summary") or [])]
        if not rules:
            raise HTTPException(422, "no active intervention to save")
        model_id = bundle.meta.get("model_id") if bundle is not None else None
        scale = iv.get("scale")
        lens_descriptor = _stable_lens_descriptor(snap.lens_binding)
        result = editing.save_preset(
            name, rules, model_id, scale=scale,
            model_revision=bundle.meta.get("revision") if bundle is not None else None,
            intervention_mode=iv.get("mode"),
            intervention_scale=scale,
            intervention_revision=iv.get("revision"),
            model_session_id=iv.get("model_session_id"),
            lens_binding_id=iv.get("lens_binding_id"),
            lens_descriptor=lens_descriptor,
        )
        return result
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = bundle = iv = rules = result = None
        if token is not None:
            manager.coordinator.release(token)


@app.post("/api/presets/{name}/apply")
def api_presets_apply(name: str):
    name = _safe_name(name)
    try:
        preset = editing.load_preset(name)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.INTERVENTION_UPDATE, requires_loaded=True)
        outcome = _presets_apply_resource(token, snap, preset)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _presets_apply_resource(token, snap, preset):
    bundle = current_iv = rules = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap)
        if lens_manager.lens is None:
            return _route_http(422, "model and lens required")
        capabilities.ensure_unquantized(bundle.jl, dict(bundle.meta))
        warnings = []
        if preset.get("model_id") and preset["model_id"] != bundle.meta["model_id"]:
            warnings.append(
                f"preset saved for {preset['model_id']}, loaded model: {bundle.meta['model_id']}"
            )
        if preset.get("model_revision") and preset.get("model_revision") != bundle.meta.get("revision"):
            warnings.append(
                f"preset saved for revision {preset['model_revision']}, loaded revision: {bundle.meta.get('revision')}"
            )
        current_iv = _coordinated_interventions(snap)
        if preset.get("intervention_mode") and preset.get("intervention_mode") != current_iv.get("mode"):
            warnings.append(
                f"preset saved under intervention mode {preset['intervention_mode']}; current mode: {current_iv.get('mode')}"
            )
        current_lens_descriptor = _stable_lens_descriptor(snap.lens_binding)
        if preset.get("lens_descriptor") is not None:
            if _lens_descriptor_is_legacy(preset.get("lens_descriptor")):
                warnings.append("preset has sparse lens provenance; stable lens compatibility cannot be verified")
            elif _lens_descriptor_mismatch(preset.get("lens_descriptor"), current_lens_descriptor):
                warnings.append("preset lens descriptor differs from the current coordinated lens")
        elif preset.get("lens_binding_id") is not None:
            warnings.append("preset has only process-local lens provenance; current stable lens identity is unknown")
        if preset.get("schema_version") is None:
            warnings.append("legacy preset: provenance is unknown")
        def mutate():
            rules = None
            for rule in preset.get("rules", []):
                try:
                    rules = interventions.add(
                        lens_manager,
                        bundle.jl,
                        token_id=rule["token_id"],
                        mode=rule["mode"],
                        factor=rule["factor"],
                        replacement_id=rule.get("replacement_id"),
                        layers=rule.get("layers"),
                        enabled=rule.get("enabled", True),
                    )
                except ValueError as exc:
                    warnings.append(f"rule {rule.get('token')!r} skipped: {exc}")
            if preset.get("intervention_scale", preset.get("scale")) is not None:
                interventions.set_scale(preset.get("intervention_scale", preset.get("scale")))
            return rules
        rules, _ = _run_intervention_transaction(token, snap, mutate)
        return _route_value({
            "rules": rules or interventions.summary(),
            "scale": interventions.global_scale,
            "warnings": warnings,
        })
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = current_iv = rules = mutate = warnings = None


@app.delete("/api/presets/{name}")
def api_presets_delete(name: str):
    editing.delete_preset(_safe_name(name))
    return {"ok": True}


@app.post("/api/edit/export")
async def api_edit_export(req: ExportRequest):
    req.name = _safe_name(req.name)
    if req.format not in ("layers", "lora", "full"):
        raise HTTPException(422, f"unknown format: {req.format}")
    token = snap = None
    stop_event = ThreadingEvent()
    try:
        token, snap = manager.coordinator.acquire(OperationType.EXPORT, requires_loaded=True)
        outcome = _export_preflight_resource(snap, req)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    finally:
        snap = None

    if outcome.status is not None:
        if token is not None:
            manager.coordinator.release(token)
            token = None
        return _raise_route_outcome(outcome)
    prepared = outcome.value
    outcome = None
    payload_seed = prepared.take()
    prepared = None

    try:
        publication_gate = editing.PublicationGate(
            lambda: not manager.coordinator.is_cancelled(token) and manager.coordinator.is_current(token)
        )
        payload = dict(payload_seed)
        payload.update(publication_gate=publication_gate, format=req.format, name=req.name)
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
    except Exception:
        payload_seed = payload = prepared = None
        if token is not None:
            manager.coordinator.release(token)
        raise
    payload_seed = payload = prepared = None

    def worker():
        if not dispatch.claim():
            return _worker_success()
        payload = dispatch.take_payload()
        try:
            try:
                call_kwargs = dict(payload["kwargs"])
                if payload["export_fn"] is editing.export_rebase:
                    call_kwargs["publication_guard"] = payload["publication_gate"]
                result = payload["export_fn"](
                    payload["rules"], payload["bundle"].jl, payload["meta"],
                    fmt=payload["format"], name=payload["name"],
                    source_dir=payload["source_dir"], scale=payload["scale"], **call_kwargs,
                )
                return _worker_success(result)
            except Exception as exc:
                return _worker_failure(exc)
        finally:
            try:
                import gc
                payload = call_kwargs = result = None
                gc.collect()
            finally:
                dispatch.release_from_worker()

    try:
        task = asyncio.create_task(asyncio.to_thread(worker))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(task, dispatch)
    try:
        return _raise_worker_outcome(await asyncio.shield(task))
    except asyncio.CancelledError:
        publication_gate.cancel()
        dispatch.cancel_from_awaiter()
        raise
    except HTTPException:
        raise


def _export_preflight_resource(snap, req):
    bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None
    try:
        bundle = snap.bundle
        meta = dict(bundle.meta)
        intervention_snapshot = _coordinated_interventions(snap)
        rules = intervention_snapshot["active_rules"]
        if not rules:
            return _route_http(422, "no active intervention to export (rules disabled or without layers?)")
        source_dir = resolve_local_dir(meta["model_id"])
        scale, mode = intervention_snapshot["scale"], intervention_snapshot["mode"]
        capabilities.ensure_unquantized(bundle.jl, meta)
        capabilities.require(bundle.capability_profile, "exports", req.format, mode, loaded=True)
        kwargs = {}
        if mode in ("readthrough", "exact"):
            export_fn = editing.export_rebase
            kwargs["exact"] = mode == "exact"
        elif mode == "abliteration":
            export_fn = editing.export_abliteration
        else:
            return _route_http(
                422,
                "export requires a pure-weights mode: switch to \"read projection\" "
                "(or \"global projection\" on write-norm architectures) — per-layer "
                "steering does not bake faithfully",
            )
        return _route_value(PreparedWorkerPayload({
            "bundle": bundle, "meta": meta, "rules": rules, "source_dir": source_dir,
            "scale": scale, "kwargs": kwargs, "export_fn": export_fn,
        }))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None




def _gguf_preflight_resource(snap):
    bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None
    try:
        bundle = _bundle_from_snapshot_or_legacy(snap)
        meta = dict(bundle.meta)
        intervention_snapshot = _coordinated_interventions(snap)
        rules = intervention_snapshot["active_rules"]
        if not rules:
            return _route_http(422, "no active intervention to export")
        scale, mode = intervention_snapshot["scale"], intervention_snapshot["mode"]
        capabilities.ensure_unquantized(bundle.jl, meta)
        capabilities.require(bundle.capability_profile, "exports", "gguf", mode, loaded=True)
        if mode in ("readthrough", "exact"):
            export_fn, kwargs = editing.export_rebase, {"exact": mode == "exact"}
        elif mode == "abliteration":
            export_fn, kwargs = editing.export_abliteration, {}
        else:
            return _route_http(422, "export requires a pure-weights mode")
        source_dir = resolve_local_dir(meta["model_id"])
        return _route_value(PreparedWorkerPayload({
            "bundle": bundle, "meta": meta, "rules": rules, "source_dir": source_dir,
            "scale": scale, "kwargs": kwargs, "export_fn": export_fn,
        }))
    except ValueError as exc:
        return _route_http(422, str(exc))
    except HTTPException as exc:
        return _route_http(exc.status_code, exc.detail)
    except Exception as exc:
        return _route_exception(500, exc)
    finally:
        bundle = meta = intervention_snapshot = rules = source_dir = kwargs = export_fn = None


# --- direct GGUF export (via a user-provided llama.cpp folder) ---------------
# Two stages: (1) bake the full HF checkpoint into data/edits/<name>/hf — kept
# as a CACHE so several GGUF types can be exported without re-baking — then
# (2) convert_hf_to_gguf.py (+ llama-quantize for quantized types) in a
# background thread, progress polled through /api/status.
_gguf_state = {"state": "idle", "name": None, "step": None, "error": None, "result": None}

GGUF_BASE_TYPES = ("bf16", "f16")
GGUF_QUANT_TYPES = ("q8_0", "q6_k", "q5_k_m", "q4_k_m", "q3_k_m")


class GGUFExportRequest(BaseModel):
    name: str
    gguf_type: str = "q4_k_m"


def _llamacpp_paths():
    """(convert_py, quantize_exe, gguf_py) from the configured llama.cpp dir."""
    root = read_settings().get("llamacpp_dir") or ""
    root = Path(root).expanduser() if root else None
    if not root or not root.is_dir():
        raise ValueError(
            "llama.cpp folder not set — configure it in the Options tab to "
            "enable the direct GGUF export"
        )
    convert = root / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise ValueError(f"convert_hf_to_gguf.py not found in {root}")
    quantize = None
    for cand in ("llama-quantize", "llama-quantize.exe"):
        for sub in (".", "bin", "build/bin"):
            p = root / sub / cand
            if p.exists():
                quantize = p
                break
        if quantize:
            break
    gguf_py = root / "gguf-py"
    return convert, quantize, (gguf_py if gguf_py.is_dir() else None)


def _gguf_worker(job_name, job_dir, filename_stem, gguf_type,
                 hf_dir, convert, quantize, gguf_py):
    import subprocess
    import sys
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        base_type = gguf_type if gguf_type in GGUF_BASE_TYPES else "bf16"
        base_gguf = job_dir / f"{filename_stem}-{base_type}.gguf"
        env = dict(os.environ)
        if gguf_py is not None:  # vendored gguf package inside the llama.cpp repo
            env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
        if not base_gguf.exists():
            _gguf_state.update(step=f"converting to {base_type}")
            temp_gguf = base_gguf.with_name(
                f".{base_gguf.stem}.tmp-{uuid.uuid4().hex}.gguf"
            )
            temp_gguf.unlink(missing_ok=True)
            with editing.artifact_lease(temp_gguf):
                try:
                    proc = subprocess.run(
                        [sys.executable, "-X", "utf8", str(convert), str(hf_dir),
                         "--outfile", str(temp_gguf), "--outtype", base_type],
                        capture_output=True, text=True, env=env,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"convert_hf_to_gguf failed: {proc.stderr[-2000:]}"
                        )
                    if not temp_gguf.is_file():
                        raise RuntimeError(
                            "convert_hf_to_gguf succeeded without producing an output"
                        )
                    if temp_gguf.stat().st_size <= 0:
                        raise RuntimeError("convert_hf_to_gguf produced an empty output")
                    temp_gguf.replace(base_gguf)
                finally:
                    temp_gguf.unlink(missing_ok=True)
        result_path = base_gguf
        if gguf_type not in GGUF_BASE_TYPES:
            if quantize is None:
                raise RuntimeError(
                    "llama-quantize not found in the llama.cpp folder — only "
                    "bf16/f16 exports are possible"
                )
            _gguf_state.update(step=f"quantizing to {gguf_type}")
            result_path = job_dir / f"{filename_stem}-{gguf_type}.gguf"
            temp_quantized = result_path.with_name(
                f".{result_path.stem}.tmp-{uuid.uuid4().hex}.gguf"
            )
            temp_quantized.unlink(missing_ok=True)
            with editing.artifact_lease(temp_quantized):
                try:
                    proc = subprocess.run(
                        [str(quantize), str(base_gguf), str(temp_quantized), gguf_type],
                        capture_output=True, text=True,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(f"llama-quantize failed: {proc.stderr[-2000:]}")
                    if not temp_quantized.is_file():
                        raise RuntimeError(
                            "llama-quantize succeeded without producing an output"
                        )
                    if temp_quantized.stat().st_size <= 0:
                        raise RuntimeError("llama-quantize produced an empty output")
                    temp_quantized.replace(result_path)
                finally:
                    temp_quantized.unlink(missing_ok=True)
        _gguf_state.update(
            state="done", step=None, error=None,
            result={
                "gguf": str(result_path),
                "size_bytes": result_path.stat().st_size,
                "hf_cache": str(hf_dir),
            },
        )
    except Exception as exc:
        _gguf_state.update(state="error", step=None, error=str(exc))


@app.post("/api/edit/export-gguf")
async def api_edit_export_gguf(req: GGUFExportRequest):
    try:
        name_parts = editing.validate_export_name(req.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    validated_name = "/".join(name_parts)
    job_dir = editing.EDITS_DIR.joinpath(*name_parts)
    filename_stem = name_parts[-1]
    if _gguf_state["state"] == "running":
        raise HTTPException(409, "a GGUF export is already in progress")
    if req.gguf_type not in GGUF_BASE_TYPES + GGUF_QUANT_TYPES:
        raise HTTPException(422, f"unknown GGUF type: {req.gguf_type}")
    try:
        convert, quantize, gguf_py = _llamacpp_paths()
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    if req.gguf_type not in GGUF_BASE_TYPES and quantize is None:
        raise HTTPException(422, "llama-quantize not found — pick bf16 or f16")

    hf_dir = job_dir / "hf"
    baked = "reused"
    if not (hf_dir / "config.json").exists():
        token = snap = None
        stop_event = ThreadingEvent()
        try:
            token, snap = manager.coordinator.acquire(OperationType.GGUF_BAKE, requires_loaded=True)
            outcome = _gguf_preflight_resource(snap)
        except OperationConflict as exc:
            raise _conflict(exc)
        except ModelStateError as exc:
            raise HTTPException(422, "no model loaded (and no cached checkpoint for this name)") from exc
        except Exception:
            if token is not None:
                manager.coordinator.release(token)
            raise
        finally:
            snap = None
        if outcome.status is not None:
            if token is not None:
                manager.coordinator.release(token)
                token = None
            return _raise_route_outcome(outcome)
        prepared = outcome.value
        outcome = None
        payload_seed = prepared.take()
        prepared = None
        try:
            publication_gate = editing.PublicationGate(
                lambda: not manager.coordinator.is_cancelled(token) and manager.coordinator.is_current(token)
            )
            payload = dict(payload_seed)
            payload.update(name="/".join((*name_parts, "hf")), publication_gate=publication_gate)
            dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
        except Exception:
            payload_seed = payload = prepared = None
            if token is not None:
                manager.coordinator.release(token)
            raise
        payload_seed = payload = prepared = None

        def bake_worker():
            if not dispatch.claim():
                return _worker_success()
            payload = dispatch.take_payload()
            try:
                try:
                    call_kwargs = dict(payload["kwargs"])
                    if payload["export_fn"] is editing.export_rebase:
                        call_kwargs["publication_guard"] = payload["publication_gate"]
                    result = payload["export_fn"](
                        payload["rules"], payload["bundle"].jl, payload["meta"],
                        fmt="full", name=payload["name"],
                        source_dir=payload["source_dir"], scale=payload["scale"], **call_kwargs,
                    )
                    return _worker_success(result)
                except Exception as exc:
                    return _worker_failure(exc)
            finally:
                payload = call_kwargs = result = None
                dispatch.release_from_worker()
        try:
            task = asyncio.create_task(asyncio.to_thread(bake_worker))
        except Exception:
            dispatch.cancel_from_awaiter()
            raise
        await _retain_worker_task_or_cancel(task, dispatch)
        try:
            _raise_worker_outcome(await asyncio.shield(task))
        except asyncio.CancelledError:
            publication_gate.cancel()
            dispatch.cancel_from_awaiter()
            raise
        except HTTPException:
            raise
        baked = "baked"

    _gguf_state.update(state="running", name=validated_name, step="starting", error=None, result=None)
    threading.Thread(
        target=_gguf_worker,
        args=(validated_name, job_dir, filename_stem, req.gguf_type,
              hf_dir, convert, quantize, gguf_py),
        daemon=True,
    ).start()
    return {"started": True, "checkpoint": baked, "state": dict(_gguf_state)}


class GGUFCacheRequest(BaseModel):
    name: str


@app.post("/api/edit/gguf-cache/delete")
def api_gguf_cache_delete(req: GGUFCacheRequest):
    """Drop the cached HF checkpoint of a GGUF export (the .gguf files stay)."""
    import shutil
    try:
        name_parts = editing.validate_export_name(req.name)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    validated_name = "/".join(name_parts)
    hf_dir = editing.EDITS_DIR.joinpath(*name_parts) / "hf"
    if not hf_dir.is_dir():
        raise HTTPException(404, f"no cached checkpoint for {req.name}")
    if _gguf_state["state"] == "running" and _gguf_state["name"] == validated_name:
        raise HTTPException(409, "a GGUF export is using this cache")
    freed = sum(f.stat().st_size for f in hf_dir.rglob("*") if f.is_file())
    shutil.rmtree(hf_dir)
    return {"deleted": str(hf_dir), "freed_bytes": freed}


class GenerateSyncRequest(BaseModel):
    messages: list[dict]
    sampling: dict = {}


def _bundle_from_snapshot_or_legacy(snap):
    if snap.bundle is None:
        raise ModelStateError("no model loaded")
    return snap.bundle


def _captured_generation_context(token, snap, stop_event):
    bundle = _bundle_from_snapshot_or_legacy(snap)
    return GenerationContext(
        token=token,
        model_session_id=snap.model_session_id,
        bundle=bundle,
        hf_model=bundle.hf_model,
        tokenizer=bundle.tokenizer,
        jl=bundle.jl,
        meta=bundle.meta,
        capability_profile=bundle.capability_profile,
        lens=lens_manager.snapshot_for_generation(),
        intervention_snapshot=_coordinated_interventions(snap),
        stop_event=stop_event,
    )


@app.post("/api/generate")
async def api_generate_sync(req: GenerateSyncRequest):
    global _generation_counter
    token = None
    stop_event = ThreadingEvent()
    try:
        token, snap = manager.coordinator.acquire(OperationType.GENERATE, requires_loaded=True)
        context = _captured_generation_context(token, snap, stop_event)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    try:
        payload = {"context": context, "messages": req.messages, "sampling": req.sampling}
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    snap = context = payload = None
    done = {}

    def emit(frame):
        if frame["type"] == "done":
            done.update(frame)
        elif frame["type"] == "error":
            done["error"] = frame.get("message")

    def worker():
        global _generation_counter
        if not dispatch.claim():
            return _worker_success()
        payload = dispatch.take_payload()
        try:
            try:
                context = payload["context"]
                manager.generate(
                    payload["messages"], payload["sampling"], stop_event, emit,
                    lens=context, ablator=interventions if context.intervention_snapshot.get("rules") else None,
                )
                if done.get("error"):
                    return _worker_failure(RuntimeError(done["error"]))
                _generation_counter += 1
                last = {
                    "n": _generation_counter,
                    "prompt": next((m.get("content", "") for m in reversed(payload["messages"]) if m.get("role") == "user"), ""),
                    "text": done.get("text", ""),
                    "stats": done.get("stats"),
                    "meta": done.get("meta"),
                }
                manager.coordinator.publish_last_generation(token, context.model_session_id, last)
                done["last_generation"] = last
                return _worker_success({"text": done.get("text", ""), "stats": done.get("stats"), "meta": done.get("meta"), "last_generation": last})
            except Exception as exc:
                return _worker_failure(exc)
        finally:
            payload = context = None
            dispatch.release_from_worker()

    try:
        task = asyncio.create_task(asyncio.to_thread(worker))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(task, dispatch)
    try:
        outcome_value = _raise_worker_outcome(await asyncio.shield(task))
    except asyncio.CancelledError:
        dispatch.cancel_from_awaiter()
        raise
    except HTTPException:
        raise
    for ws in list(_ws_locks):
        try:
            await _ws_send(ws, json.dumps({"type": "api_generation"}))
        except Exception:
            pass
    return {
        "text": (outcome_value or {}).get("text", ""),
        "stats": (outcome_value or {}).get("stats"),
        "meta": (outcome_value or {}).get("meta"),
        "last_generation": (outcome_value or {}).get("last_generation"),
    }


class NeighborsRequest(BaseModel):
    token_ids: list[int]
    k: int = 3


@app.post("/api/token-neighbors")
async def api_token_neighbors(req: NeighborsRequest):
    token = None
    stop_event = ThreadingEvent()
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_READ, requires_loaded=True)
        bundle = snap.bundle
        key = (snap.model_session_id, dict(bundle.meta).get("model_id"), dict(bundle.meta).get("revision"))
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    try:
        payload = {"bundle": bundle, "key": key, "token_ids": req.token_ids[:64], "k": req.k}
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    snap = bundle = payload = None

    def worker():
        if not dispatch.claim():
            return _worker_success()
        payload = dispatch.take_payload()
        try:
            try:
                result = neighbors.lookup(
                    payload["bundle"].jl, payload["bundle"].tokenizer,
                    payload["key"], payload["token_ids"], payload["k"],
                )
                return _worker_success(result)
            except Exception as exc:
                return _worker_failure(exc)
        finally:
            payload = result = None
            dispatch.release_from_worker()

    try:
        task = asyncio.create_task(asyncio.to_thread(worker))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(task, dispatch)
    try:
        result = _raise_worker_outcome(await asyncio.shield(task))
    except asyncio.CancelledError:
        dispatch.cancel_from_awaiter()
        raise
    except HTTPException:
        raise
    return {"neighbors": {str(tid): entries for tid, entries in result.items()}}


@app.get("/api/token-lookup")
def api_token_lookup(q: str):
    token = snap = None
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_READ, requires_loaded=True)
        outcome = _token_lookup_resource(snap, q)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    finally:
        snap = None
        if token is not None:
            manager.coordinator.release(token)
    return _raise_route_outcome(outcome)


def _token_lookup_resource(snap, q: str):
    tokenizer = candidates = None
    try:
        tokenizer = snap.bundle.tokenizer
        candidates = {}
        for variant in (q, " " + q, q.lower(), " " + q.lower(),
                        q.capitalize(), " " + q.capitalize(), q.upper(), " " + q.upper()):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in candidates:
                candidates[ids[0]] = tokenizer.decode([ids[0]])
        return _route_value({"candidates": [{"id": tid, "str": s} for tid, s in candidates.items()]})
    except Exception as exc:
        if isinstance(exc, HTTPException):
            return _route_http(exc.status_code, exc.detail)
        return _route_exception(500, exc)
    finally:
        tokenizer = candidates = None


@app.get("/api/registry/local")
def api_registry_local():
    return {"lenses": registry.local_lenses()}


@app.get("/api/registry/for-model")
async def api_registry_for_model(model_id: str, revision: str | None = None):
    return await asyncio.to_thread(registry.lenses_for_model, model_id, revision)


@app.get("/api/registry/resolve")
def api_registry_resolve(path: str | None = None, repo_id: str | None = None, filename: str | None = None):
    return registry.resolve_lens(path=path, repo_id=repo_id, filename=filename)


@app.post("/api/fit")
def api_fit(req: FitRequest):
    token = None
    try:
        token, _ = manager.coordinator.acquire(OperationType.FIT, requires_loaded=False)
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError:
        raise HTTPException(409, "unload the model first: fitting needs all the VRAM")
    valid = _valid_devices()
    bad = [d for d in req.devices if d not in valid]
    if bad:
        manager.coordinator.release(token)
        raise HTTPException(422, f"invalid device(s): {', '.join(bad)}")
    source = resolve_source(req.model_id)
    transferred = False

    def release_fit():
        manager.coordinator.release(token)

    try:
        result = fit_manager.start(
            model_id=req.model_id,
            source=source,
            model_revision=_resolve_revision(source),
            n_prompts=req.n_prompts,
            dtype=req.dtype,
            quant=req.quant,
            datasets=req.datasets,
            devices=req.devices,
            name=req.name,
            dim_batch=req.dim_batch,
            max_seq_len=req.max_seq_len,
            source_layers=req.source_layers,
            continue_from=req.continue_from,
            reservation_release=release_fit,
        )
        transferred = True
        return result
    except ValueError as exc:
        if not transferred:
            manager.coordinator.release(token)
        raise HTTPException(409, str(exc))
    except Exception:
        if not transferred:
            manager.coordinator.release(token)
        raise


@app.get("/api/fit/status")
def api_fit_status():
    return fit_manager.state


@app.post("/api/fit/stop")
def api_fit_stop():
    return fit_manager.stop()


@app.post("/api/lens/pin")
async def api_lens_pin(req: PinRequest):
    token = None
    stop_event = ThreadingEvent()
    try:
        token, snap = manager.coordinator.acquire(OperationType.MODEL_READ, requires_loaded=True)
        jl = snap.bundle.jl
    except OperationConflict as exc:
        raise _conflict(exc)
    except ModelStateError as exc:
        raise HTTPException(422, str(exc))
    try:
        payload = {"jl": jl, "gen_id": req.gen_id, "token_ids": req.token_ids}
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
    except Exception:
        if token is not None:
            manager.coordinator.release(token)
        raise
    snap = jl = payload = None

    def worker():
        if not dispatch.claim():
            return _worker_success()
        payload = dispatch.take_payload()
        try:
            try:
                result = lens_manager.pin_ranks(payload["gen_id"], payload["token_ids"], payload["jl"])
                return _worker_success(result)
            except Exception as exc:
                return _worker_failure(exc)
        finally:
            payload = result = None
            dispatch.release_from_worker()

    try:
        task = asyncio.create_task(asyncio.to_thread(worker))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(task, dispatch)
    try:
        return _raise_worker_outcome(await asyncio.shield(task))
    except asyncio.CancelledError:
        dispatch.cancel_from_awaiter()
        raise
    except HTTPException:
        raise


# Alternative/duplicate weight folders, never needed for transformers inference
# (GPT-OSS-20B ships original/ + metal/ = 2 × ~14 GB of waste).
DOWNLOAD_IGNORE_DIRS = [
    "original/*", "metal/*", "onnx/*", "openvino/*", "coreml/*", "gguf/*",
]
# Auxiliary files that are always useful (configs, tokenizer, custom code) — light.
DOWNLOAD_EXTRAS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "*.py", "*.md"]
# "large" fp32 model: past 1 GB of weights, convert to bf16 automatically
AUTO_BF16_MIN_BYTES = 1_000_000_000


def _plan_download(api, repo_id, token):
    """Pick the strict minimum: ONE weight set (the lightest if several variants)
    + the auxiliary files. Returns (allow_patterns, ignore_patterns, plan) —
    allow_patterns None = "take everything" fallback."""
    import re

    files = {}  # path -> size
    for entry in api.list_repo_tree(repo_id, recursive=True, token=token or None):
        size = getattr(entry, "size", None)
        if size is not None:
            files[entry.path] = size

    def in_ignored_dir(path):
        return any(path.startswith(d.split("/*")[0] + "/") for d in DOWNLOAD_IGNORE_DIRS)

    # root-level safetensors sets, grouped by variant:
    # "model(-00001-of-00002)?.safetensors" -> group "model";
    # "model.fp32(-...)?.safetensors" -> group "model.fp32", etc.
    st_groups = {}
    for path, size in files.items():
        if "/" in path or not path.endswith(".safetensors"):
            continue
        stem = re.sub(r"-\d{5}-of-\d{5}", "", path.removesuffix(".safetensors"))
        st_groups.setdefault(stem, []).append(path)

    if st_groups:
        stem, chosen = min(
            st_groups.items(), key=lambda kv: sum(files[p] for p in kv[1])
        )
        index = f"{stem}.safetensors.index.json"
        patterns = sorted(chosen) + ([index] if index in files else []) + DOWNLOAD_EXTRAS
        return patterns, None, {
            "kind": "safetensors",
            "variant": stem,
            "size_bytes": sum(files[p] for p in chosen),
        }

    # no safetensors (legacy .bin/.h5 repos, or GGUF-only ones we can't load):
    # take everything except the alternative folders and obvious format
    # duplicates. GGUF weights are ignored — J-Wash only loads transformers
    # (safetensors) models.
    ignore = DOWNLOAD_IGNORE_DIRS + ["*.gguf", "*.msgpack", "*.h5", "*.tflite", "*.onnx"]
    return None, ignore, {
        "kind": "fallback",
        "size_bytes": sum(
            s for p, s in files.items()
            if not in_ignored_dir(p) and not p.endswith(".gguf")
        ),
    }


def _maybe_autoconvert_bf16(repo_id, state):
    """After download: if the safetensors weights are float32 and heavy, convert
    to bf16 automatically into a local folder (halves the space in use; the HF
    cache source stays intact)."""
    from pathlib import Path

    from safetensors import safe_open

    from core.model_manager import convert_to_bf16, resolve_local_dir

    src = resolve_local_dir(repo_id)
    if not src:
        return
    src = Path(src)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        return
    total = sum(s.stat().st_size for s in shards)
    if total < AUTO_BF16_MIN_BYTES:
        return
    import math

    with safe_open(str(shards[0]), framework="pt") as f:
        keys = list(f.keys())
        if not keys:
            return
        # the shard's biggest tensor is representative of the "large layers"
        biggest = max(keys, key=lambda k: math.prod(f.get_slice(k).get_shape()))
        dtype = str(f.get_slice(biggest).get_dtype())
    if dtype not in ("F32", "F64"):
        return
    state.update(state="converting")
    base = repo_id.split("/")[-1]
    result = convert_to_bf16(str(src), out_dir=str(config.LOCAL_MODELS_ROOT / f"{base}-bf16"))
    state.update(converted=result["id"])


def _download_worker(repo_id):
    import os

    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    state = _downloads[repo_id]
    try:
        allow, ignore, plan = None, None, None
        try:
            allow, ignore, plan = _plan_download(HfApi(), repo_id, token)
            state.update(plan=plan)
        except Exception:
            # planning failed (network, permissions): cautious fallback
            ignore = DOWNLOAD_IGNORE_DIRS + ["*.gguf", "*.msgpack", "*.h5", "*.tflite", "*.onnx"]
        # progress: poll the cache size (robust — the tqdm hook misses some files
        # depending on the download mechanism). We sum the repo's blobs (including
        # .incomplete files) and compare to the planned total.
        total_bytes = (plan or {}).get("size_bytes") or 0
        _blobs = config.HF_CACHE / "hub" / f"models--{repo_id.replace('/', '--')}" / "blobs"
        stop_poll = ThreadingEvent()

        def _poll_progress():
            while not stop_poll.is_set():
                done = 0
                if _blobs.exists():
                    for f in _blobs.iterdir():
                        try:
                            done += f.stat().st_size
                        except OSError:
                            pass
                if total_bytes:
                    state["progress"] = {"done": min(done, total_bytes), "total": total_bytes}
                stop_poll.wait(1.0)

        poller = threading.Thread(target=_poll_progress, daemon=True)
        poller.start()
        try:
            snapshot_download(
                repo_id, token=token or None,
                allow_patterns=allow, ignore_patterns=ignore,
            )
        finally:
            stop_poll.set()
            state.pop("progress", None)
        if plan and plan["kind"] == "safetensors":
            _maybe_autoconvert_bf16(repo_id, state)
        state.update(state="done", error=None)
    except GatedRepoError:
        msg = (
            f'gated repo "{repo_id}": accept the terms on huggingface.co and make '
            "sure a valid HF_TOKEN is set in the environment."
            + ("" if token else " (no HF_TOKEN detected)")
        )
        state.update(state="error", error=msg)
    except RepositoryNotFoundError:
        state.update(
            state="error",
            error=f'repo "{repo_id}" not found (or private without access using the current token)',
        )
    except Exception as exc:
        state.update(state="error", error=str(exc))


@app.post("/api/download")
def api_download(req: DownloadRequest):
    repo_id = req.repo_id.strip()
    with _downloads_lock:
        current = _downloads.get(repo_id)
        if current and current["state"] == "running":
            raise HTTPException(409, f"download already in progress: {repo_id}")
        # several downloads in parallel: one state per repo
        _downloads[repo_id] = {"repo_id": repo_id, "state": "running", "error": None}
    threading.Thread(target=_download_worker, args=(repo_id,), daemon=True).start()
    return _downloads[repo_id]


@app.delete("/api/download/{repo_id:path}")
def api_download_dismiss(repo_id: str):
    """Remove a finished (done/error) entry from the displayed list."""
    with _downloads_lock:
        state = _downloads.get(repo_id)
        if state and state["state"] != "running":
            del _downloads[repo_id]
    return {"downloads": list(_downloads.values())}


@app.get("/api/browse")
def api_browse(path: str | None = None):
    from core.model_manager import browse_dir

    try:
        return browse_dir(path)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


class PickPathRequest(BaseModel):
    kind: str = "dir"  # kept for API compatibility; only directory picking is used


_pick_lock = threading.Lock()


@app.post("/api/pick-path")
async def api_pick_path(req: PickPathRequest):
    """Open Windows' NATIVE file picker (the server runs on the user's own
    machine) and return the chosen path — this notably lets you paste a path,
    which the built-in browser cannot do."""

    def pick():
        if not _pick_lock.acquire(blocking=False):
            raise ValueError("a file picker is already open")
        try:
            # tkinter ships with CPython on every platform, but headless
            # Linux installs may lack it (or a display): fail with a hint
            # instead of a stack trace — the built-in Browse still works.
            try:
                import tkinter as tk
                from tkinter import filedialog
            except ImportError:
                raise ValueError(
                    "no native folder picker available (tkinter missing) — "
                    "use the built-in Browse, or paste the path directly"
                )

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                path = filedialog.askdirectory(
                    parent=root, title="Choose a model folder (HF)"
                )
            finally:
                root.destroy()
            return {"path": path or None}
        finally:
            _pick_lock.release()

    try:
        return await asyncio.to_thread(pick)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


class ConvertRequest(BaseModel):
    path: str


_convert_state = {"path": None, "state": "idle", "error": None, "result": None}


def _convert_worker(path):
    from core.model_manager import convert_to_bf16

    try:
        result = convert_to_bf16(path)
        _convert_state.update(state="done", error=None, result=result)
    except Exception as exc:
        _convert_state.update(state="error", error=str(exc))


@app.post("/api/convert-bf16")
def api_convert_bf16(req: ConvertRequest):
    if _convert_state["state"] == "running":
        raise HTTPException(409, "a conversion is already in progress")
    _convert_state.update(path=req.path, state="running", error=None, result=None)
    threading.Thread(target=_convert_worker, args=(req.path,), daemon=True).start()
    return _convert_state


@app.get("/api/conversations")
def api_conversations(query: str | None = None):
    return {"conversations": store.list_conversations(query)}


@app.get("/api/conversations/{cid}")
def api_conversation(cid: int):
    try:
        return store.get_conversation(cid)
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.patch("/api/conversations/{cid}")
def api_conversation_patch(cid: int, req: ConversationPatch):
    store.update_conversation(cid, title=req.title, tags=req.tags)
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
def api_conversation_delete(cid: int):
    store.delete_conversation(cid)
    return {"ok": True}


@app.get("/api/messages/{mid}/frames")
def api_message_frames(mid: int):
    try:
        return store.load_frames(mid)
    except FramesNotAttached as exc:
        raise HTTPException(404, str(exc)) from None
    except FramePointerTurnover as exc:
        raise HTTPException(409, str(exc)) from None
    except FrameFileMissing as exc:
        raise HTTPException(422, str(exc)) from None
    except FrameStorageError as exc:
        raise HTTPException(422, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None


class MessagePatch(BaseModel):
    content: str


@app.patch("/api/messages/{mid}")
def api_message_patch(mid: int, req: MessagePatch):
    """Edit a message's content (e.g. rewrite an assistant reply). Later turns
    are generated from the stored path, so the edit takes effect immediately."""
    try:
        store.update_message(
            mid,
            req.content,
            meta={
                "provenance_invalidated": True,
                "edited": True,
                "provenance": "user_edited_unknown",
                "continuations": [],
            },
            clear_frames=True,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return {"ok": True, "id": mid}


@app.get("/api/conversations/{cid}/export")
def api_conversation_export(cid: int, format: str = "json", frames: int = 0):
    try:
        body, media_type = store.export(cid, fmt=format, include_frames=bool(frames))
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    ext = "json" if format == "json" else "md"
    return Response(
        content=body,
        media_type=f"{media_type}; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="conversation-{cid}.{ext}"'},
    )


def _generate_safely(messages, sampling, stop_event, emit, context):
    try:
        manager.generate(
            messages, sampling, stop_event, emit, lens=context,
            ablator=interventions if context.intervention_snapshot.get("rules") else None,
        )
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


def _persisted_generate(req, stop_event, emit, gen_context):
    try:
        continue_id = req.get("continue_message_id")
        if continue_id is not None:
            _persisted_continue(req, continue_id, stop_event, emit, gen_context)
            return
        conversation_id = req.get("conversation_id")
        parent_id = req.get("parent_id")
        system = (req.get("system") or "").strip()
        content = req.get("content")
        if conversation_id is None and not content:
            emit({"type": "error", "message": "content required for a new conversation"})
            return
        if conversation_id is None:
            conversation_id = store.create_conversation(content[:60])
            if system:
                parent_id = store.add_message(conversation_id, None, "system", system)
        if content:
            parent_id = store.add_message(conversation_id, parent_id, "user", content)
        if parent_id is None:
            emit({"type": "error", "message": "parent_id or content required"})
            return
        emit({
            "type": "persisted",
            "conversation_id": conversation_id,
            "user_message_id": parent_id,
        })
        messages = store.path_to_root(parent_id)
        lens = gen_context.lens
        layers_used = list(lens.layers) if lens is not None else []
        k_used = lens.k if lens is not None else 0
        frames_acc = []
        done_holder = {}

        def emit_inner(frame):
            if frame["type"] == "done":
                done_holder.update(frame)
            else:
                if frame["type"] == "frame":
                    frames_acc.append(frame)
                emit(frame)

        manager.generate(
            messages=messages, sampling=req.get("sampling", {}), stop_event=stop_event, emit=emit_inner, lens=gen_context,
            ablator=interventions if gen_context.intervention_snapshot.get("rules") else None,
        )
        meta = dict(
            done_holder.get("meta") or {},
            stats=done_holder.get("stats"),
            stopped=done_holder.get("stopped"),
        )
        insert_meta = dict(meta)
        if frames_acc:
            insert_meta.update(publication_state="frames_pending", frames_expected=True)
        else:
            insert_meta.update(publication_state="complete", frames_expected=False)
        message_id, inserted_version = store.add_message(
            conversation_id, parent_id, "assistant", done_holder.get("text", ""), meta=insert_meta,
            return_version=True,
        )
        if frames_acc:
            complete_meta = dict(meta, publication_state="complete", frames_expected=True)
            try:
                store.save_frames(
                    message_id, frames_acc, layers_used, k_used,
                    expected_version=inserted_version, complete_meta=complete_meta,
                )
            except StaleMessageUpdate:
                emit({"type": "error", "message": "message changed before frames were attached; retry"})
                return
            except Exception as exc:
                failure_meta = dict(
                    meta, publication_state="frames_publication_failed", frames_expected=True,
                    frames_error_kind=exc.__class__.__name__, frames_error=str(exc),
                )
                try:
                    marked = store.mark_frame_publication_failed(
                        message_id, expected_version=inserted_version, failure_meta=failure_meta
                    )
                except Exception as mark_exc:
                    emit({"type": "error", "message": f"frames could not be published; terminalization failed: {mark_exc}"})
                    return
                if not marked:
                    emit({"type": "error", "message": "message changed before frame failure could be recorded; retry"})
                    return
                emit({"type": "error", "message": "frames could not be published; retry", "message_id": message_id})
                return
        emit(dict(done_holder, conversation_id=conversation_id, message_id=message_id))
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


def _continuation_metadata(previous_meta, start_offset, end_offset, done_meta, stats, stopped):
    previous_meta = dict(previous_meta or {})
    latest_meta = dict(done_meta or {})
    latest_meta.update(stats=stats, stopped=stopped, continued=True)
    continuations = list(previous_meta.get("continuations") or [])
    if not continuations and start_offset > 0:
        base_meta = dict(previous_meta)
        base_meta.pop("continuations", None)
        base_meta.pop("continuation_attempts", None)
        if previous_meta.get("continued"):
            continuations.append({
                "start_offset": 0,
                "end_offset": start_offset,
                "provenance": "legacy_mixed_unknown",
                "meta": None,
            })
        else:
            continuations.append({"start_offset": 0, "end_offset": start_offset, "meta": base_meta})
    merged = dict(previous_meta)
    if end_offset > start_offset:
        continuations.append({
            "start_offset": start_offset,
            "end_offset": end_offset,
            "meta": dict(latest_meta),
        })
        merged.update(latest_meta)
        merged["continuations"] = continuations
    else:
        attempts = list(previous_meta.get("continuation_attempts") or [])
        attempts.append({"offset": start_offset, "meta": latest_meta})
        if continuations:
            merged["continuations"] = continuations
        merged["continuation_attempts"] = attempts
    return merged


def _persisted_continue(req, message_id, stop_event, emit, gen_context):
    """Extend an existing assistant reply: generate with the turn left open,
    append the text to the message, and merge the new lens frames into its
    stored blob (positions keep increasing, so both parts stay coherent)."""
    msg = store.get_message(message_id)
    expected_content = msg["content"]
    expected_meta = msg.get("meta")
    expected_version = msg.get("version", 0)
    if msg["role"] != "assistant":
        emit({"type": "error", "message": "only an assistant reply can be continued"})
        return
    messages = store.path_to_root(message_id)
    lens = gen_context.lens
    layers_used = list(lens.layers) if lens is not None else []
    k_used = lens.k if lens is not None else 0
    frames_acc = []
    done_holder = {}

    def emit_inner(frame):
        if frame["type"] == "done":
            done_holder.update(frame)
        else:
            if frame["type"] == "frame":
                frames_acc.append(frame)
            emit(frame)

    manager.generate(
        messages=messages, sampling=req.get("sampling", {}), stop_event=stop_event, emit=emit_inner, lens=gen_context,
        ablator=interventions if gen_context.intervention_snapshot.get("rules") else None,
        continue_final=True,
    )
    old_content = expected_content
    appended = done_holder.get("text", "")
    new_content = old_content + appended
    previous_meta = json.loads(expected_meta) if expected_meta else {}
    meta = _continuation_metadata(
        previous_meta, len(old_content), len(new_content),
        done_holder.get("meta"), done_holder.get("stats"), done_holder.get("stopped"),
    )
    merged = None
    if frames_acc:
        merged = frames_acc
        if msg.get("frames_file"):
            try:
                merged = store.load_frames(message_id)["frames"] + frames_acc
            except Exception:
                emit({"type": "error", "message": "could not load existing frames; continuation not persisted"})
                return
    if not store.update_message_and_frames_if_unchanged(
        message_id, expected_version, new_content, meta,
        frames=merged, layers=layers_used, k=k_used,
    ):
        emit({"type": "error", "message": "message changed during continuation; retry"})
        return
    emit(dict(
        done_holder,
        conversation_id=msg["conversation_id"],
        message_id=message_id,
        text=new_content,
        continued=True,
    ))


async def _watch_stop(ws, stop_event):
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "stop":
                stop_event.set()
    finally:
        stop_event.set()


async def _run_chat(ws, req):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    stop_event = ThreadingEvent()

    def emit(frame):
        loop.call_soon_threadsafe(queue.put_nowait, frame)

    try:
        token, snap = manager.coordinator.acquire(OperationType.GENERATE, requires_loaded=True)
        context = _captured_generation_context(token, snap, stop_event)
        if not req.get("lens"):
            context = GenerationContext(
                token=context.token, model_session_id=context.model_session_id, bundle=context.bundle,
                hf_model=context.hf_model, tokenizer=context.tokenizer, jl=context.jl, meta=context.meta,
                capability_profile=context.capability_profile, lens=None,
                intervention_snapshot=context.intervention_snapshot, stop_event=context.stop_event,
            )
    except OperationConflict as exc:
        await _ws_send(ws, json.dumps({"type": "error", "message": str(exc)}))
        return
    except ModelStateError as exc:
        await _ws_send(ws, json.dumps({"type": "error", "message": str(exc)}))
        return
    except Exception as exc:
        if token is not None:
            manager.coordinator.release(token)
        await _ws_send(ws, json.dumps({"type": "error", "message": str(exc)}))
        return

    try:
        payload = {"context": context, "request": dict(req)}
        dispatch = WorkerDispatch(manager.coordinator, token, stop_event, payload=payload)
    except Exception as exc:
        if token is not None:
            manager.coordinator.release(token)
        await _ws_send(ws, json.dumps({"type": "error", "message": str(exc)}))
        return
    snap = context = payload = None

    def worker_body():
        if not dispatch.claim():
            return _worker_success()
        payload = dispatch.take_payload()
        try:
            try:
                request = payload["request"]
                context = payload["context"]
                if "messages" in request:
                    _generate_safely(request["messages"], request.get("sampling", {}), stop_event, emit, context)
                else:
                    _persisted_generate(request, stop_event, emit, context)
                return _worker_success()
            except Exception as exc:
                return _worker_failure(exc)
        finally:
            payload = context = request = None
            dispatch.release_from_worker()

    try:
        worker = asyncio.create_task(asyncio.to_thread(worker_body))
    except Exception:
        dispatch.cancel_from_awaiter()
        raise
    await _retain_worker_task_or_cancel(worker, dispatch)
    try:
        receiver = asyncio.create_task(_watch_stop(ws, stop_event))
    except Exception as exc:
        stop_event.set()
        dispatch.cancel_from_awaiter()
        await asyncio.gather(asyncio.shield(worker), return_exceptions=True)
        await _ws_send(ws, json.dumps({"type": "error", "message": str(exc)}))
        return
    get_task = None
    try:
        while True:
            try:
                get_task = asyncio.create_task(queue.get())
            except Exception as exc:
                stop_event.set()
                receiver.cancel()
                dispatch.cancel_from_awaiter()
                await asyncio.gather(asyncio.shield(worker), receiver, return_exceptions=True)
                await _ws_send(ws, json.dumps({"type": "error", "message": str(exc)}))
                return
            done, pending = await asyncio.wait(
                {get_task, receiver, worker},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if receiver in done and not get_task.done():
                get_task.cancel()
                stop_event.set()
                dispatch.cancel_from_awaiter()
                break
            if worker in done and not get_task.done():
                get_task.cancel()
                break
            frame = get_task.result()
            get_task = None
            await _ws_send(ws, json.dumps(frame))
            if frame["type"] in ("done", "error"):
                break
    finally:
        stop_event.set()
        if get_task is not None and not get_task.done():
            get_task.cancel()
        if not worker.done():
            dispatch.cancel_from_awaiter()
        receiver.cancel()
        await asyncio.gather(
            asyncio.shield(worker),
            receiver,
            *( [get_task] if get_task is not None else [] ),
            return_exceptions=True,
        )
        get_task = receiver = worker = None


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_locks[ws] = asyncio.Lock()
    try:
        while True:
            req = json.loads(await ws.receive_text())
            if req.get("type") != "chat":
                continue
            await _run_chat(ws, req)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _ws_locks.pop(ws, None)


if config.UI_DIST.exists():
    app.mount("/", StaticFiles(directory=config.UI_DIST, html=True), name="ui")
