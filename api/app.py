import asyncio
import json
import logging
import mimetypes
import os
import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from core import editing, registry
from core.ablation import Interventions
from core.neighbors import TokenNeighbors
from core import fitting
from core.fitting import FitManager
from core.gpus import gpu_stats
from core.lens_manager import LensManager
from core.model_manager import (
    ModelManager,
    _resolve_revision,
    resolve_local_dir,
    resolve_source,
)
from core.store import Store

manager = ModelManager()
lens_manager = LensManager()
store = Store()
fit_manager = FitManager()
interventions = Interventions()
neighbors = TokenNeighbors()
app = FastAPI(title="J-Wash")

_ws_locks = {}
_loop_holder = {}


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


@app.on_event("startup")
async def _on_startup():
    _loop_holder["loop"] = asyncio.get_running_loop()
    logging.getLogger("uvicorn.access").addFilter(_QuietPolling())


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
_last_generation = None


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
    return {
        "loaded": manager.meta,
        "busy": manager.busy,
        "lens": lens_manager.meta,
        "gpus": gpu_stats(),
        "downloads": list(_downloads.values()),
        "convert": _convert_state,
        "fit": fit_manager.state,
        "gguf": dict(_gguf_state),
        "interventions": interventions.summary(),
        "interventions_scale": interventions.global_scale,
        "interventions_mode": interventions.mode,
        "last_generation": _last_generation,
    }


@app.post("/api/load")
async def api_load(req: LoadRequest):
    if req.dtype not in config.DTYPES:
        raise HTTPException(422, f"invalid dtype: {req.dtype}")
    if req.quant is not None and req.quant not in config.QUANTS:
        raise HTTPException(422, f"invalid quant: {req.quant}")
    if req.device not in _valid_devices():
        raise HTTPException(422, f"invalid device: {req.device}")
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    try:
        return await asyncio.to_thread(
            manager.load, req.model_id, req.dtype, req.quant, req.device
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/models/delete")
async def api_delete_model(req: DeleteModelRequest):
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    if manager.meta and manager.meta.get("model_id") == req.model_id:
        raise HTTPException(409, "unload this model before deleting it")
    from core.model_manager import delete_model
    try:
        return await asyncio.to_thread(delete_model, req.model_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))


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
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    lens_manager.unload()
    neighbors.reset()
    return await asyncio.to_thread(manager.unload)


@app.post("/api/lens/load")
async def api_lens_load(req: LensLoadRequest):
    if not req.repo_id and not req.path:
        raise HTTPException(422, "repo_id or path required")
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    try:
        return await asyncio.to_thread(
            lens_manager.load,
            manager,
            repo_id=req.repo_id,
            filename=req.filename,
            revision=req.revision,
            path=req.path,
            layers=req.layers,
            k=req.k,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/lens/unload")
def api_lens_unload():
    return lens_manager.unload()


@app.post("/api/lens/layers")
def api_lens_layers(req: LensLayersRequest):
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    try:
        return lens_manager.set_layers(manager, req.layers, k=req.k)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.get("/api/interventions")
def api_interventions():
    return {"rules": interventions.summary()}


@app.post("/api/interventions")
def api_interventions_add(req: InterventionRequest):
    if manager.hf_model is None or lens_manager.lens is None:
        raise HTTPException(422, "model and lens required")
    try:
        return {
            "rules": interventions.add(
                lens_manager,
                manager.jl,
                token_id=req.token_id,
                mode=req.mode,
                factor=req.factor,
                replacement_id=req.replacement_id,
                layers=req.layers,
            )
        }
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.patch("/api/interventions/{rule_id}")
def api_interventions_patch(rule_id: int, req: InterventionPatch):
    needs_dirs = any(
        x is not None for x in (req.layers, req.token_id, req.replacement_id, req.mode)
    )
    try:
        return {
            "rules": interventions.update(
                rule_id,
                factor=req.factor,
                layers=req.layers,
                enabled=req.enabled,
                token_id=req.token_id,
                replacement_id=req.replacement_id,
                mode=req.mode,
                lens_manager=lens_manager if needs_dirs else None,
                jl=manager.jl if needs_dirs else None,
            )
        }
    except ValueError as exc:
        raise HTTPException(404, str(exc))


@app.patch("/api/interventions")
def api_interventions_scale(req: InterventionsScale):
    if req.scale is not None:
        interventions.set_scale(req.scale)
    try:
        if req.mode is not None:
            capability = f"{req.mode}_supported" if req.mode in ("readthrough", "exact") else None
            if capability and manager.meta is not None and manager.meta.get(capability, manager.meta.get("rebase_supported")) is False:
                raise HTTPException(
                    422,
                    manager.meta.get(f"{req.mode}_reason", "read projection unavailable on this architecture"),
                )
            interventions.set_mode(req.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {
        "scale": interventions.global_scale,
        "mode": interventions.mode,
    }


@app.delete("/api/interventions/{rule_id}")
def api_interventions_remove(rule_id: int):
    return {"rules": interventions.remove(rule_id)}


@app.delete("/api/interventions")
def api_interventions_clear():
    return {"rules": interventions.remove()}


@app.get("/api/presets")
def api_presets():
    return {"presets": editing.list_presets()}


@app.post("/api/presets/{name}")
def api_presets_save(name: str):
    name = _safe_name(name)
    rules = interventions.summary()
    if not rules:
        raise HTTPException(422, "no active intervention to save")
    return editing.save_preset(
        name, rules, manager.meta.get("model_id") if manager.meta else None,
        scale=interventions.global_scale,
    )


@app.post("/api/presets/{name}/apply")
def api_presets_apply(name: str):
    name = _safe_name(name)
    if manager.hf_model is None or lens_manager.lens is None:
        raise HTTPException(422, "model and lens required")
    try:
        preset = editing.load_preset(name)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    warnings = []
    if preset.get("model_id") and manager.meta and preset["model_id"] != manager.meta["model_id"]:
        warnings.append(
            f"preset saved for {preset['model_id']}, loaded model: {manager.meta['model_id']}"
        )
    rules = None
    for rule in preset.get("rules", []):
        try:
            rules = interventions.add(
                lens_manager,
                manager.jl,
                token_id=rule["token_id"],
                mode=rule["mode"],
                factor=rule["factor"],
                replacement_id=rule.get("replacement_id"),
                layers=rule.get("layers"),
                enabled=rule.get("enabled", True),
            )
        except ValueError as exc:
            warnings.append(f"rule {rule.get('token')!r} skipped: {exc}")
    if preset.get("scale") is not None:
        interventions.set_scale(preset["scale"])
    return {
        "rules": rules or interventions.summary(),
        "scale": interventions.global_scale,
        "warnings": warnings,
    }


@app.delete("/api/presets/{name}")
def api_presets_delete(name: str):
    editing.delete_preset(_safe_name(name))
    return {"ok": True}


@app.post("/api/edit/export")
async def api_edit_export(req: ExportRequest):
    req.name = _safe_name(req.name)
    if manager.hf_model is None:
        raise HTTPException(422, "no model loaded")
    rules = interventions.active_rules_full()
    if not rules:
        raise HTTPException(422, "no active intervention to export (rules disabled or without layers?)")
    if req.format not in ("layers", "lora", "full"):
        raise HTTPException(422, f"unknown format: {req.format}")
    source_dir = resolve_local_dir(manager.meta["model_id"])
    mode = interventions.mode
    kwargs = {}
    if mode in ("readthrough", "exact"):
        export_fn = editing.export_rebase
        kwargs["exact"] = mode == "exact"
    elif mode == "abliteration":
        export_fn = editing.export_abliteration
    else:
        raise HTTPException(
            422,
            "export requires a pure-weights mode: switch to \"read projection\" "
            "(or \"global projection\" on write-norm architectures) — per-layer "
            "steering does not bake faithfully",
        )
    try:
        return await asyncio.to_thread(
            export_fn,
            rules,
            manager.jl,
            manager.meta,
            fmt=req.format,
            name=req.name,
            source_dir=source_dir,
            scale=interventions.global_scale,
            **kwargs,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))
    finally:
        # a full export builds ~2× the model in RAM (source + edited tensors);
        # on failure, force a collection so those copies don't linger
        import gc
        gc.collect()


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


def _gguf_worker(name, gguf_type, hf_dir, convert, quantize, gguf_py):
    import subprocess
    import sys
    try:
        out_dir = editing.EDITS_DIR / name
        base_type = gguf_type if gguf_type in GGUF_BASE_TYPES else "bf16"
        base_gguf = out_dir / f"{name}-{base_type}.gguf"
        env = dict(os.environ)
        if gguf_py is not None:  # vendored gguf package inside the llama.cpp repo
            env["PYTHONPATH"] = str(gguf_py) + os.pathsep + env.get("PYTHONPATH", "")
        if not base_gguf.exists():
            _gguf_state.update(step=f"converting to {base_type}")
            proc = subprocess.run(
                [sys.executable, "-X", "utf8", str(convert), str(hf_dir),
                 "--outfile", str(base_gguf), "--outtype", base_type],
                capture_output=True, text=True, env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"convert_hf_to_gguf failed: {proc.stderr[-2000:]}")
        result_path = base_gguf
        if gguf_type not in GGUF_BASE_TYPES:
            if quantize is None:
                raise RuntimeError(
                    "llama-quantize not found in the llama.cpp folder — only "
                    "bf16/f16 exports are possible"
                )
            _gguf_state.update(step=f"quantizing to {gguf_type}")
            result_path = out_dir / f"{name}-{gguf_type}.gguf"
            proc = subprocess.run(
                [str(quantize), str(base_gguf), str(result_path), gguf_type],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                result_path.unlink(missing_ok=True)
                raise RuntimeError(f"llama-quantize failed: {proc.stderr[-2000:]}")
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
        # no cached checkpoint: bake one from the ACTIVE rules (same path as a
        # plain full export)
        if manager.hf_model is None:
            raise HTTPException(422, "no model loaded (and no cached checkpoint for this name)")
        rules = interventions.active_rules_full()
        if not rules:
            raise HTTPException(422, "no active intervention to export")
        mode = interventions.mode
        if mode in ("readthrough", "exact"):
            export_fn, kwargs = editing.export_rebase, {"exact": mode == "exact"}
        elif mode == "abliteration":
            export_fn, kwargs = editing.export_abliteration, {}
        else:
            raise HTTPException(422, "export requires a pure-weights mode")
        source_dir = resolve_local_dir(manager.meta["model_id"])
        try:
            await asyncio.to_thread(
                export_fn, rules, manager.jl, manager.meta,
                fmt="full", name="/".join((*name_parts, "hf")), source_dir=source_dir,
                scale=interventions.global_scale, **kwargs,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except Exception as exc:
            raise HTTPException(500, str(exc))
        baked = "baked"

    _gguf_state.update(state="running", name=validated_name, step="starting", error=None, result=None)
    threading.Thread(
        target=_gguf_worker,
        args=(validated_name, req.gguf_type, hf_dir, convert, quantize, gguf_py),
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


@app.post("/api/generate")
async def api_generate_sync(req: GenerateSyncRequest):
    """Synchronous generation, no persistence or lens frames: for the CLI tools
    (scripts/jlab.py). Active interventions apply just like in the chat."""
    if manager.hf_model is None:
        raise HTTPException(422, "no model loaded")
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    done = {}

    def emit(frame):
        if frame["type"] == "done":
            done.update(frame)
        elif frame["type"] == "error":
            done["error"] = frame.get("message")

    try:
        await asyncio.to_thread(
            manager.generate,
            req.messages,
            req.sampling,
            threading.Event(),
            emit,
            lens=None,
            ablator=interventions if interventions.active else None,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))
    if done.get("error"):
        raise HTTPException(500, done["error"])
    global _last_generation
    _last_generation = {
        "n": (_last_generation or {}).get("n", 0) + 1,
        "prompt": next(
            (m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"),
            "",
        ),
        "text": done.get("text", ""),
        "stats": done.get("stats"),
    }
    # nudge any watching UI to refresh once this API generation is done (used by
    # the "API monitor" mode, which drops the 2s status poll for event-driven refresh)
    for ws in list(_ws_locks):
        try:
            await _ws_send(ws, json.dumps({"type": "api_generation"}))
        except Exception:
            pass
    return {"text": done.get("text", ""), "stats": done.get("stats")}


class NeighborsRequest(BaseModel):
    token_ids: list[int]
    k: int = 3


@app.post("/api/token-neighbors")
async def api_token_neighbors(req: NeighborsRequest):
    if manager.hf_model is None:
        raise HTTPException(422, "no model loaded")
    key = ((manager.meta or {}).get("model_id"), (manager.meta or {}).get("revision"))
    try:
        result = await asyncio.to_thread(
            neighbors.lookup, manager.jl, manager.tokenizer, key,
            req.token_ids[:64], req.k,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return {"neighbors": {str(tid): entries for tid, entries in result.items()}}


@app.get("/api/token-lookup")
def api_token_lookup(q: str):
    if manager.tokenizer is None:
        raise HTTPException(422, "no model loaded")
    tokenizer = manager.tokenizer
    candidates = {}
    for variant in (q, " " + q, q.lower(), " " + q.lower(),
                    q.capitalize(), " " + q.capitalize(), q.upper(), " " + q.upper()):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in candidates:
            candidates[ids[0]] = tokenizer.decode([ids[0]])
    return {"candidates": [{"id": tid, "str": s} for tid, s in candidates.items()]}


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
    if manager.hf_model is not None:
        raise HTTPException(
            409, "unload the model first: fitting needs all the VRAM"
        )
    valid = _valid_devices()
    bad = [d for d in req.devices if d not in valid]
    if bad:
        raise HTTPException(422, f"invalid device(s): {', '.join(bad)}")
    source = resolve_source(req.model_id)
    try:
        return fit_manager.start(
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
        )
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/fit/status")
def api_fit_status():
    return fit_manager.state


@app.post("/api/fit/stop")
def api_fit_stop():
    return fit_manager.stop()


@app.post("/api/lens/pin")
async def api_lens_pin(req: PinRequest):
    if manager.busy:
        raise HTTPException(409, f"busy: {manager.busy}")
    try:
        return await asyncio.to_thread(
            lens_manager.pin_ranks, req.gen_id, req.token_ids, manager.jl
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


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
        stop_poll = threading.Event()

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
    except ValueError as exc:
        raise HTTPException(404, str(exc))


class MessagePatch(BaseModel):
    content: str


@app.patch("/api/messages/{mid}")
def api_message_patch(mid: int, req: MessagePatch):
    """Edit a message's content (e.g. rewrite an assistant reply). Later turns
    are generated from the stored path, so the edit takes effect immediately."""
    try:
        store.update_message(mid, req.content)
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


def _generate_safely(messages, sampling, stop_event, emit, lens):
    try:
        manager.generate(
            messages, sampling, stop_event, emit, lens=lens,
            ablator=interventions if interventions.active else None,
        )
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


def _persisted_generate(req, stop_event, emit, lens):
    try:
        continue_id = req.get("continue_message_id")
        if continue_id is not None:
            _persisted_continue(req, continue_id, stop_event, emit, lens)
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
        context = store.path_to_root(parent_id)
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
            context, req.get("sampling", {}), stop_event, emit_inner, lens=lens,
            ablator=interventions if interventions.active else None,
        )
        meta = dict(
            done_holder.get("meta") or {},
            stats=done_holder.get("stats"),
            stopped=done_holder.get("stopped"),
        )
        message_id = store.add_message(
            conversation_id, parent_id, "assistant", done_holder.get("text", ""), meta=meta
        )
        if frames_acc:
            store.save_frames(message_id, frames_acc, layers_used, k_used)
        emit(dict(done_holder, conversation_id=conversation_id, message_id=message_id))
    except Exception as exc:
        emit({"type": "error", "message": str(exc)})


def _persisted_continue(req, message_id, stop_event, emit, lens):
    """Extend an existing assistant reply: generate with the turn left open,
    append the text to the message, and merge the new lens frames into its
    stored blob (positions keep increasing, so both parts stay coherent)."""
    msg = store.get_message(message_id)
    if msg["role"] != "assistant":
        emit({"type": "error", "message": "only an assistant reply can be continued"})
        return
    context = store.path_to_root(message_id)
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
        context, req.get("sampling", {}), stop_event, emit_inner, lens=lens,
        ablator=interventions if interventions.active else None,
        continue_final=True,
    )
    new_content = msg["content"] + done_holder.get("text", "")
    meta = json.loads(msg["meta"]) if msg.get("meta") else {}
    meta = dict(
        meta,
        stats=done_holder.get("stats"),
        stopped=done_holder.get("stopped"),
        continued=True,
    )
    store.update_message(message_id, new_content, meta=meta)
    if frames_acc:
        merged = frames_acc
        if msg.get("frames_file"):
            try:
                merged = store.load_frames(message_id)["frames"] + frames_acc
            except Exception:
                pass
        store.save_frames(message_id, merged, layers_used, k_used)
    emit(dict(
        done_holder,
        conversation_id=msg["conversation_id"],
        message_id=message_id,
        text=new_content,
        continued=True,
    ))


async def _watch_stop(ws, stop_event):
    while True:
        msg = json.loads(await ws.receive_text())
        if msg.get("type") == "stop":
            stop_event.set()


async def _run_chat(ws, req):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    stop_event = threading.Event()

    def emit(frame):
        loop.call_soon_threadsafe(queue.put_nowait, frame)

    lens = lens_manager if req.get("lens") and lens_manager.lens is not None else None
    if "messages" in req:
        worker = asyncio.create_task(
            asyncio.to_thread(
                _generate_safely, req["messages"], req.get("sampling", {}), stop_event, emit, lens
            )
        )
    else:
        worker = asyncio.create_task(
            asyncio.to_thread(_persisted_generate, req, stop_event, emit, lens)
        )
    receiver = asyncio.create_task(_watch_stop(ws, stop_event))
    try:
        while True:
            frame = await queue.get()
            await _ws_send(ws, json.dumps(frame))
            if frame["type"] in ("done", "error"):
                break
    finally:
        stop_event.set()
        receiver.cancel()
        await asyncio.gather(worker, receiver, return_exceptions=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_locks[ws] = asyncio.Lock()
    try:
        while True:
            req = json.loads(await ws.receive_text())
            if req.get("type") != "chat":
                continue
            if manager.hf_model is None:
                await _ws_send(
                    ws, json.dumps({"type": "error", "message": "no model loaded"})
                )
                continue
            if manager.busy:
                await _ws_send(
                    ws, json.dumps({"type": "error", "message": f"busy: {manager.busy}"})
                )
                continue
            await _run_chat(ws, req)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        _ws_locks.pop(ws, None)


if config.UI_DIST.exists():
    app.mount("/", StaticFiles(directory=config.UI_DIST, html=True), name="ui")
