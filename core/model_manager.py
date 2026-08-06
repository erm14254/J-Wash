import gc
import json
import os
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path

import torch
import transformers
from huggingface_hub import scan_cache_dir, try_to_load_from_cache

import config
import jlens
from core.lens_manager import ActivationCatcher
from core.model_session import GenerationContext, LoadedModelBundle, ModelSessionCoordinator, OperationConflict, OperationType

SKIP_LOCAL_DIRS = {"vendor", "ui", "data", "hf_cache", "lenses", "core", "api", "scripts"}

# Many "base" models (e.g. non-Instruct Llama-3.2-1B) ship no chat_template. The
# right one is their instruct sibling's, which shares the same tokenizer: so we
# look it up on the Hub before any fallback.
INSTRUCT_SIBLING_SUFFIXES = ("-Instruct", "-instruct", "-it", "-Chat", "-chat")


def _intervention_provenance(snapshot):
    if not isinstance(snapshot, Mapping):
        return None
    def clone_summary(items):
        out = []
        for item in items or ():
            cloned = dict(item)
            if cloned.get("layers") is not None:
                cloned["layers"] = list(cloned["layers"])
            out.append(cloned)
        return out
    return {
        "revision": snapshot.get("revision"),
        "mode": snapshot.get("mode"),
        "scale": snapshot.get("scale"),
        "model_session_id": snapshot.get("model_session_id"),
        "lens_binding_id": snapshot.get("lens_binding_id"),
        "summary": clone_summary(snapshot.get("summary")),
        "active_summary": clone_summary(snapshot.get("active_summary")),
    }

# End-of-turn markers per model family; added to the stop tokens when they appear
# in the applied template (useful when a base model is given an instruct template:
# it must stop on <|eot_id|>, <|im_end|>, <end_of_turn>, etc.)
TURN_END_MARKERS = ("<|eot_id|>", "<|im_end|>", "<end_of_turn>", "<|end|>", "<|endoftext|>")

# Last-resort fallback when no template can be found (offline, no reachable
# sibling): a readable "User:/Assistant:" format a completion model can continue.
FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}{{ message['content'] + '\n\n' }}"
    "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
    "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)


def _extract_template(chat_template):
    """chat_template may be a string or a list [{name, template}] (multi-template)."""
    if isinstance(chat_template, str):
        return chat_template
    if isinstance(chat_template, list):
        for entry in chat_template:
            if isinstance(entry, dict) and entry.get("name") == "default":
                return entry.get("template")
        if chat_template and isinstance(chat_template[0], dict):
            return chat_template[0].get("template")
    return None


def _read_hub_template(repo, token, revision=None):
    """Read a chat_template from a Hub repo: chat_template.jinja (raw) then the
    chat_template key of tokenizer_config.json / chat_template.json."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, "chat_template.jinja", token=token, revision=revision)
        text = Path(path).read_text(encoding="utf-8").strip()
        if text:
            return text
    except Exception:
        pass
    for fname in ("tokenizer_config.json", "chat_template.json"):
        try:
            path = hf_hub_download(repo, fname, token=token, revision=revision)
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        tmpl = _extract_template(data.get("chat_template") if isinstance(data, dict) else None)
        if tmpl:
            return tmpl
    return None


def fetch_chat_template(model_id, token, revision=None):
    """Look up the real chat_template on the Hub: first the model's own repo, then
    its instruct siblings (shared tokenizer). Returns (template, source_repo) or
    (None, None). Skips local models/paths (no Hub repo)."""
    if "/" not in model_id or model_id.startswith("local/") or os.path.isabs(model_id):
        return None, None
    candidates = [(model_id, revision)]
    for suffix in INSTRUCT_SIBLING_SUFFIXES:
        if not model_id.endswith(suffix):
            candidates.append((model_id + suffix, None))  # sibling revision unknown
    for repo, rev in candidates:
        try:
            tmpl = _read_hub_template(repo, token, revision=rev)
        except Exception:
            tmpl = None
        if tmpl:
            return tmpl, repo
    return None, None


def _config_n_layers(config_path):
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    tc = cfg.get("text_config", cfg)
    return tc.get("num_hidden_layers") or cfg.get("num_hidden_layers")


def _config_dtype(config_path):
    try:
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    return cfg.get("torch_dtype") or cfg.get("text_config", {}).get("torch_dtype")


# Model folders registered by hand (Browse): a list of absolute paths kept in
# the data dir. Registering never copies or moves anything; unregistering only
# forgets the entry, the files stay untouched.
REGISTERED_PATH = config.DATA_DIR / "registered_models.json"


def _read_registered():
    try:
        entries = json.loads(REGISTERED_PATH.read_text(encoding="utf-8"))
        return [str(e) for e in entries if isinstance(e, str)]
    except Exception:
        return []


def _write_registered(entries):
    REGISTERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTERED_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def register_model_dir(path):
    p = Path(path).expanduser().resolve()
    if not _dir_is_model(p):
        raise ValueError(f"not a model folder (config.json + weights required): {p}")
    entries = _read_registered()
    if str(p) not in entries:
        entries.append(str(p))
        _write_registered(entries)
    return {"registered": str(p)}


def unregister_model_dir(path):
    wanted = str(Path(path).expanduser().resolve())
    entries = _read_registered()
    kept = [e for e in entries if e != path and str(Path(e)) != wanted]
    if len(kept) == len(entries):
        raise ValueError(f"not a registered entry: {path}")
    _write_registered(kept)
    return {"unregistered": path}


def _registered_models():
    out = []
    for entry in _read_registered():
        path = Path(entry)
        missing = not _dir_is_model(path)
        stats = [] if missing else [f.stat() for f in path.glob("*.safetensors")]
        unique = {(s.st_ino, s.st_size): s.st_size for s in stats}
        out.append(
            {
                # the absolute path IS the id: resolve_source passes it through
                "id": entry,
                "source": "registered",
                "path": entry,
                "missing": missing,
                "size_bytes": sum(unique.values()),
                "n_layers": None if missing else _config_n_layers(path / "config.json"),
                "dtype": None if missing else _config_dtype(path / "config.json"),
            }
        )
    return out


def _local_models():
    found = []
    for child in sorted(config.LOCAL_MODELS_ROOT.iterdir()):
        if not child.is_dir() or child.name in SKIP_LOCAL_DIRS:
            continue
        if not (child / "config.json").exists():
            continue
        stats = [f.stat() for f in child.glob("*.safetensors")]
        if not stats:
            continue
        unique = {(s.st_ino, s.st_size): s.st_size for s in stats}
        found.append(
            {
                "id": f"local/{child.name}",
                "source": "local",
                "path": str(child),
                "size_bytes": sum(unique.values()),
                "n_layers": _config_n_layers(child / "config.json"),
                "dtype": _config_dtype(child / "config.json"),
            }
        )
    return found


def _cached_models():
    hub = config.HF_CACHE / "hub"
    if not hub.exists():
        return []
    out = []
    for repo in scan_cache_dir(hub).repos:
        if repo.repo_type != "model":
            continue
        config_path = None
        for rev in repo.revisions:
            for f in rev.files:
                if f.file_name == "config.json":
                    config_path = f.file_path
        if config_path is None:
            continue
        out.append(
            {
                "id": repo.repo_id,
                "source": "hf-cache",
                "size_bytes": repo.size_on_disk,
                "n_layers": _config_n_layers(config_path),
                "dtype": _config_dtype(config_path),
                "path": str(Path(config_path).parent),
            }
        )
    return sorted(out, key=lambda r: r["id"])


def _dir_is_model(path):
    return (path / "config.json").exists() and (
        any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
    )


def browse_dir(path=None):
    """Minimal file browser: subfolders + loadable model folders.
    Empty path -> list drive letters (Windows)."""
    import string

    if not path:
        drives = []
        for letter in string.ascii_uppercase:
            root = Path(f"{letter}:/")
            if root.exists():
                drives.append({"name": f"{letter}:", "path": str(root)})
        return {"path": "", "parent": None, "dirs": drives, "models": []}

    base = Path(path)
    if not base.is_dir():
        raise ValueError(f"folder not found: {path}")
    dirs = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        children = []
    for child in children:
        try:
            if child.is_dir():
                dirs.append({
                    "name": child.name,
                    "path": str(child),
                    "is_model": _dir_is_model(child),
                })
        except OSError:
            continue
    return {
        "path": str(base),
        "parent": str(base.parent) if base.parent != base else None,
        "dirs": dirs,
        "is_model": _dir_is_model(base),
    }


def delete_model(model_id):
    """Delete a model: local folder or HF cache repo.
    Refuses anything outside the managed roots (guards against arbitrary paths)."""
    import shutil

    if model_id.startswith("local/"):
        name = model_id.removeprefix("local/")
        if name in SKIP_LOCAL_DIRS or "/" in name or "\\" in name or ".." in name:
            raise ValueError("protected folder or invalid name")
        path = (config.LOCAL_MODELS_ROOT / name).resolve()
        root = config.LOCAL_MODELS_ROOT.resolve()
        if root not in path.parents or not (path / "config.json").exists():
            raise ValueError(f"unmanaged path: {path}")
        shutil.rmtree(path)
        return {"deleted": str(path), "freed_bytes": None}

    # otherwise: a Hugging Face cache repo (delete all of its revisions)
    hub = config.HF_CACHE / "hub"
    if not hub.exists():
        raise ValueError(f"unknown model: {model_id}")
    info = scan_cache_dir(hub)
    hashes, freed = [], 0
    for repo in info.repos:
        if repo.repo_id == model_id and repo.repo_type == "model":
            hashes = [rev.commit_hash for rev in repo.revisions]
            freed = repo.size_on_disk
            break
    if not hashes:
        raise ValueError(f"unknown model in cache: {model_id}")
    info.delete_revisions(*hashes).execute()
    return {"deleted": model_id, "freed_bytes": freed}


def convert_to_bf16(src_dir, out_dir=None):
    """Rewrite an fp32 model's safetensors as bf16 into a sibling local folder.
    Leaves the source untouched. Returns the new local id."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(src_dir)
    if not src.is_dir():
        raise ValueError(f"source not found: {src_dir}")
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors in the source")
    out = Path(out_dir) if out_dir else (config.LOCAL_MODELS_ROOT / f"{src.name}-bf16")
    name = out.name
    out.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            metadata = f.metadata()
            for key in f.keys():
                t = f.get_tensor(key)
                if t.dtype == torch.float32:
                    t = t.to(torch.bfloat16)
                tensors[key] = t
        save_file(tensors, str(out / shard.name), metadata=metadata)
    for extra in src.iterdir():
        if extra.suffix in (".json", ".txt", ".model") or extra.name.startswith("tokenizer"):
            data = extra.read_bytes()
            if extra.name == "config.json":
                cfg = json.loads(data)
                cfg["torch_dtype"] = "bfloat16"
                if "text_config" in cfg and isinstance(cfg["text_config"], dict):
                    cfg["text_config"]["torch_dtype"] = "bfloat16"
                (out / extra.name).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            else:
                (out / extra.name).write_bytes(data)
    return {"id": f"local/{name}", "path": str(out)}


def _torch_allocated():
    return {
        f"cuda:{i}": torch.cuda.memory_allocated(i)
        for i in range(torch.cuda.device_count())
    }


def _torch_reserved():
    return {
        f"cuda:{i}": torch.cuda.memory_reserved(i)
        for i in range(torch.cuda.device_count())
    }


def _free_cuda():
    """Hand the caching allocator's blocks back to the driver (gc then empty_cache).
    Call this on EVERY error/unload path: without it, allocations from an OOM load
    or from an aborted generation's KV cache stay reserved and pile up until the
    server restarts. We loop per device with a sync: pending frees must be visible
    before empty_cache can hand the segments back."""
    for _ in range(2):
        gc.collect()
    if not torch.cuda.is_available():
        return
    # cuBLAS keeps a persistent workspace (~8 MB) per device; under
    # expandable_segments:True (see config.setup_env) that single live allocation
    # pins the WHOLE segment (~8 GB) → empty_cache returns nothing after unload. So
    # we explicitly clear the cuBLAS workspaces first.
    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


def _input_device(hf_model):
    return hf_model.get_input_embeddings().weight.device


def resolve_source(model_id):
    if model_id.startswith("local/"):
        return str(config.LOCAL_MODELS_ROOT / model_id.removeprefix("local/"))
    return model_id


def resolve_local_dir(model_id):
    source = resolve_source(model_id)
    path = Path(source)
    if path.is_dir():
        return str(path)
    cached = try_to_load_from_cache(source, "config.json")
    if isinstance(cached, str):
        return str(Path(cached).parent)
    return None


def _resolve_revision(source):
    if Path(source).exists():
        return None
    cached = try_to_load_from_cache(source, "config.json")
    if isinstance(cached, str):
        parts = Path(cached).parts
        if "snapshots" in parts:
            return parts[parts.index("snapshots") + 1]
    return None


def _sample(logits, temperature, top_p, top_k, generator=None, penalty=1.0, penalty_ids=None):
    if penalty != 1.0 and penalty_ids is not None and penalty_ids.numel():
        score = logits[penalty_ids]
        logits[penalty_ids] = torch.where(score > 0, score / penalty, score * penalty)
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, -1)
    if top_k > 0:
        kth = probs.topk(top_k).values[-1]
        probs = probs.masked_fill(probs < kth, 0.0)
    if 0 < top_p < 1:
        sorted_probs, sorted_idx = probs.sort(descending=True)
        keep = sorted_probs.cumsum(-1) - sorted_probs < top_p
        sorted_probs = sorted_probs * keep
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=generator))


def _rebase_capability_meta(jl):
    """Compatibility wrapper around the single production policy."""
    from core import capabilities
    profile = capabilities.build_profile(
        jl, getattr(jl, "_jwash_declared_quant", None)
    )
    return capabilities.legacy(profile, loaded=True)


class ModelManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.hf_model = None
        self.tokenizer = None
        self.jl = None
        self.meta = None
        self.capability_profile = None
        self.coordinator = ModelSessionCoordinator()
        self.on_model_withdraw = None
        self.on_model_transition_prepare = None
        self.on_model_transition_rollback = None

    @property
    def busy(self):
        return self.coordinator.busy

    def _sync_from_bundle(self, bundle, *, prepared_meta=None):
        if bundle is None:
            self.hf_model = self.tokenizer = self.jl = self.meta = None
            self.capability_profile = None
            return
        self.hf_model = bundle.hf_model
        self.tokenizer = bundle.tokenizer
        self.jl = bundle.jl
        self.meta = prepared_meta
        self.capability_profile = bundle.capability_profile

    def session_snapshot(self):
        return self.coordinator.snapshot()

    def list_models(self):
        return _local_models() + _registered_models() + _cached_models()

    def load(self, model_id, dtype, quant, device):
        token = None
        candidate = None
        hf_model = tokenizer = jl = None
        published = False
        try:
            token, snap = self.coordinator.acquire(OperationType.LOAD, include_bundle=False)
            expected_session = snap.model_session_id
            if snap.loaded:
                intervention_record = None
                if self.on_model_transition_prepare is not None:
                    intervention_record = self.on_model_transition_prepare(expected_session + 1)
                try:
                    old_bundle, expected_session, _ = self.coordinator.withdraw_loaded(
                        token, intervention_record=intervention_record
                    )
                except Exception:
                    if self.on_model_transition_rollback is not None:
                        self.on_model_transition_rollback()
                    raise
                with self._lock:
                    self._sync_from_bundle(None)
                del old_bundle
                if self.on_model_withdraw is not None:
                    self.on_model_withdraw(expected_session, token)
                _free_cuda()
            torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
            source = resolve_source(model_id)
            kwargs = {"dtype": torch_dtype}
            if quant == "int8":
                kwargs["quantization_config"] = transformers.BitsAndBytesConfig(load_in_8bit=True)
            elif quant == "nf4":
                kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype, bnb_4bit_use_double_quant=True,
                )
            kwargs["device_map"] = "auto" if device == "auto" else {"": device}
            offline_ok = resolve_local_dir(model_id) is not None
            if offline_ok:
                kwargs["local_files_only"] = True
            tok_kwargs = {"local_files_only": True} if offline_ok else {}
            started = time.perf_counter()
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(source, **kwargs)
            tokenizer = transformers.AutoTokenizer.from_pretrained(source, **tok_kwargs)
            chat_template_source = None
            if not getattr(tokenizer, "chat_template", None):
                token_env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                fetched, src = fetch_chat_template(model_id, token_env, revision=_resolve_revision(source))
                if fetched:
                    tokenizer.chat_template = fetched
                    chat_template_source = src
                else:
                    tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
                    chat_template_source = "generic"
            chat_template_fallback = chat_template_source == "generic"
            hf_model.eval()
            text_config = hf_model.config.get_text_config()
            jl = jlens.from_hf(hf_model, tokenizer)
            jl._jwash_declared_quant = quant
            from core.capabilities import build_profile, legacy, normalize_profile
            capability_profile = build_profile(jl, quant)
            capability_meta = legacy(capability_profile, loaded=True)
            normalized_profile = normalize_profile(capability_profile)
            meta = {
                "model_id": model_id, "revision": _resolve_revision(source),
                "dtype": dtype, "quant": quant, "device": device,
                "n_layers": text_config.num_hidden_layers, "d_model": text_config.hidden_size,
                "has_packed_read_parameters": (normalized_profile["has_packed_read_parameters"] if normalized_profile is not None else False),
                **capability_meta,
                "chat_template_source": chat_template_source,
                "chat_template_fallback": chat_template_fallback,
                "load_seconds": round(time.perf_counter() - started, 1),
            }
            candidate = LoadedModelBundle.from_parts(hf_model, tokenizer, jl, meta, capability_profile)
            prepared_meta = dict(candidate.meta)
            result = dict(prepared_meta)
            self.coordinator.publish_loaded(token, candidate, expected_unloaded_session=expected_session)
            published = True
            with self._lock:
                self._sync_from_bundle(candidate, prepared_meta=prepared_meta)
            prepared_meta = None
            candidate = hf_model = tokenizer = jl = None
            return result
        except OperationConflict:
            raise
        except Exception as exc:
            failure_message = str(exc)
            try:
                exc.__traceback__ = None
                exc.__context__ = None
                exc.__cause__ = None
            except Exception:
                pass
            if not published:
                candidate = None
                hf_model = tokenizer = jl = None
                _free_cuda()
            raise RuntimeError(failure_message) from None
        finally:
            if token is not None:
                self.coordinator.release(token)

    def unload(self):
        token = None
        try:
            token, _snap = self.coordinator.acquire(OperationType.UNLOAD, include_bundle=False)
            return self._unload_locked(token=token)
        finally:
            if token is not None:
                self.coordinator.release(token)

    def _unload_locked(self, token=None):
        try:
            before = _torch_allocated()
        except Exception:
            before = None
        intervention_record = None
        if self.coordinator.snapshot(include_bundle=False).loaded and self.on_model_transition_prepare is not None:
            intervention_record = self.on_model_transition_prepare(self.coordinator.snapshot(include_bundle=False).model_session_id + 1)
        try:
            old, session_id, changed = self.coordinator.withdraw_loaded(token, intervention_record=intervention_record)
        except Exception:
            if self.on_model_transition_rollback is not None:
                self.on_model_transition_rollback()
            raise
        if not changed:
            with self._lock:
                self._sync_from_bundle(None)
            try:
                allocated = _torch_allocated()
            except Exception:
                allocated = None
            return {"unloaded": False, "vram_allocated": allocated}
        with self._lock:
            self._sync_from_bundle(None)
        old = None
        if self.on_model_withdraw is not None:
            self.on_model_withdraw(session_id, token)
        _free_cuda()
        try:
            allocated_after = _torch_allocated()
        except Exception:
            allocated_after = None
        try:
            reserved_after = _torch_reserved()
        except Exception:
            reserved_after = None
        return {
            "unloaded": True,
            "vram_allocated_before": before,
            "vram_allocated_after": allocated_after,
            "vram_reserved_after": reserved_after,
        }

    @torch.no_grad()
    def generate(self, messages, sampling, stop_event, emit, lens=None, ablator=None,
                 continue_final=False, capture_full_input_frames=False):
        """``continue_final=True``: the last message is an assistant reply to
        EXTEND — the template leaves its turn open instead of starting a new
        one, and the model picks up where it stopped."""
        if isinstance(lens, GenerationContext):
            context = lens
            hf_model, tokenizer = context.hf_model, context.tokenizer
            jl = context.jl
            meta = dict(context.meta)
            lens = context.lens
            ablator_snapshot = context.intervention_snapshot
            intervention_provenance = _intervention_provenance(ablator_snapshot)
            stop_event = context.stop_event
        else:
            context = None
            hf_model, tokenizer = self.hf_model, self.tokenizer
            jl = self.jl
            meta = self.meta or {}
            ablator_snapshot = None
            intervention_provenance = None
        reader = None
        attachment = None
        ok = False
        try:
            if ablator is not None:
                attachment = ablator.attach(jl, snapshot=ablator_snapshot)
            is_gpt_oss = "gpt-oss" in (meta or {}).get("model_id", "").lower()
            template_kwargs = {}
            if is_gpt_oss:
                # harmony format: the system slot always carries an identity —
                # "You are ChatGPT, a large language model trained by OpenAI."
                # unless model_identity overrides it — while a user "system"
                # message is APPENDED as a developer message. We make the
                # user's system prompt BE the identity (no OpenAI default, no
                # duplicated developer copy); with no system prompt, a neutral
                # identity replaces the default.
                sys_prompts = [m["content"] for m in messages if m["role"] == "system"]
                identity = (sys_prompts[0] or "").strip() if sys_prompts else ""
                template_kwargs["model_identity"] = identity or "You are a helpful assistant."
                if sys_prompts:
                    messages = [m for m in messages if m["role"] != "system"]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=not continue_final,
                continue_final_message=continue_final,
                return_tensors="pt",
                enable_thinking=False,
                **template_kwargs,
            )
            input_ids = encoded if isinstance(encoded, torch.Tensor) else encoded["input_ids"]
            input_ids = input_ids.to(_input_device(hf_model))

            # gpt-oss: enable_thinking does not apply to the harmony template.
            # We prime the "final" channel directly to skip the CoT ("analysis"
            # channel) → direct answer, no chain of thought.
            # (Not when continuing: the final message is already mid-channel.)
            if is_gpt_oss and not continue_final:
                final_prefix = torch.tensor(
                    [tokenizer.encode("<|channel|>final<|message|>", add_special_tokens=False)],
                    device=input_ids.device, dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, final_prefix], dim=1)

            read_from = 0
            gen_id = None
            generation_run_id = uuid.uuid4().hex
            if lens is not None and lens.lens is not None:
                reader = ActivationCatcher(jl.layers, lens.layers)
                gen_id = lens.start_gen(generation_run_id=generation_run_id)
                if not capture_full_input_frames and len(messages) > 1 and any(m["role"] != "system" for m in messages[:-1]):
                    prev = tokenizer.apply_chat_template(
                        messages[:-1],
                        add_generation_prompt=False,
                        return_tensors="pt",
                        enable_thinking=False,
                        **template_kwargs,
                    )
                    prev_ids = prev if isinstance(prev, torch.Tensor) else prev["input_ids"]
                    read_from = min(prev_ids.shape[1], input_ids.shape[1] - 1)

            temperature = float(sampling.get("temperature", config.DEFAULT_SAMPLING["temperature"]))
            top_p = float(sampling.get("top_p", config.DEFAULT_SAMPLING["top_p"]))
            top_k = int(sampling.get("top_k", config.DEFAULT_SAMPLING["top_k"]))
            max_tokens = int(sampling.get("max_tokens", config.DEFAULT_SAMPLING["max_tokens"]))
            seed = int(sampling.get("seed", config.DEFAULT_SAMPLING["seed"]))
            # repetition penalty (HF-style, over prompt + generated); 1.0 = off.
            # Deliberately NOT exposed through the MCP server.
            repetition_penalty = float(
                sampling.get("repetition_penalty", config.DEFAULT_SAMPLING["repetition_penalty"])
            )
            # base model with a generic template: the model has no notion of dialogue
            # turns, so we cut as soon as it reopens one (User: or a new Assistant:)
            stop_seqs = (
                ["\nUser:", "\nAssistant:"]
                if (meta or {}).get("chat_template_fallback")
                else []
            )
            hidden_special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())

            out = hf_model(input_ids=input_ids, use_cache=True)
            cache = out.past_key_values
            logits = out.logits[:, -1]
            eos = hf_model.generation_config.eos_token_id
            eos_ids = set(eos) if isinstance(eos, list) else {eos}
            # if the template applies end-of-turn markers (e.g. an instruct template
            # placed on a base model), add them to the stop tokens.
            applied_template = getattr(tokenizer, "chat_template", "") or ""
            if isinstance(applied_template, str) and not is_gpt_oss:
                unk = tokenizer.unk_token_id
                for marker in TURN_END_MARKERS:
                    if marker in applied_template:
                        tid = tokenizer.convert_tokens_to_ids(marker)
                        if isinstance(tid, int) and tid >= 0 and tid != unk:
                            eos_ids.add(tid)
            if is_gpt_oss:
                # in harmony <|end|> separates MESSAGES (analysis → final), but our
                # prompt primes the final channel directly (and "continue" resumes
                # mid-final), so there is never a transition to protect: the first
                # <|end|> IS the end of the turn. The model often emits it instead
                # of <|return|>; without this stop it then replays a whole
                # "assistant analysis ..." turn in plain text up to max_tokens.
                tid = tokenizer.convert_tokens_to_ids("<|end|>")
                if isinstance(tid, int) and tid >= 0:
                    eos_ids.add(tid)

            # seed >= 0: reproducible sampling; -1 = random
            generator = None
            if seed >= 0:
                generator = torch.Generator(device=logits.device).manual_seed(seed)

            if reader is not None:
                positions = list(range(read_from, input_ids.shape[1]))
                reading_frames = lens.compute_frames(
                    reader.acts,
                    positions,
                    "reading",
                    jl,
                    input_ids[0, read_from:].tolist(),
                    gen_id=gen_id,
                )
                for frame in reading_frames:
                    frame["generation_run_id"] = generation_run_id
                    emit(frame)

            reply_ids = []
            emitted = ""
            stop_reason = None
            started = time.perf_counter()
            penalty_ids = input_ids[0].to(logits.device) if repetition_penalty != 1.0 else None
            for _ in range(max_tokens):
                if stop_event.is_set():
                    break
                next_id = _sample(logits[0].float(), temperature, top_p, top_k, generator,
                                  repetition_penalty, penalty_ids)
                if next_id in eos_ids:
                    stop_reason = "eos"
                    break
                if next_id in hidden_special_ids:
                    stop_reason = "hidden_special_token"
                    break
                reply_ids.append(next_id)
                decoded_reply = tokenizer.decode(reply_ids, skip_special_tokens=True)
                matched_marker = next((marker for marker in stop_seqs if decoded_reply.endswith(marker)), None)
                if matched_marker is not None:
                    marker_tokens = 1
                    for suffix_len in range(1, len(reply_ids) + 1):
                        suffix_text = tokenizer.decode(reply_ids[-suffix_len:], skip_special_tokens=True)
                        if suffix_text.endswith(matched_marker):
                            marker_tokens = suffix_len
                            break
                    del reply_ids[-marker_tokens:]
                    stop_reason = "fallback_stop_sequence"
                    break
                if penalty_ids is not None:
                    penalty_ids = torch.cat(
                        [penalty_ids, torch.tensor([next_id], device=penalty_ids.device)]
                    )
                held_stop_prefix = 0
                for suffix_len in range(1, len(reply_ids) + 1):
                    suffix_text = tokenizer.decode(reply_ids[-suffix_len:], skip_special_tokens=True)
                    if any(marker.startswith(suffix_text) and suffix_text != marker for marker in stop_seqs):
                        held_stop_prefix = suffix_len
                visible_ids = reply_ids[:-held_stop_prefix] if held_stop_prefix else reply_ids
                text = tokenizer.decode(visible_ids, skip_special_tokens=True)
                if not text.endswith("�") and len(text) > len(emitted):
                    emit({"type": "token", "text": text[len(emitted):]})
                    emitted = text
                out = hf_model(
                    input_ids=torch.tensor([[next_id]], device=_input_device(hf_model)),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                logits = out.logits[:, -1]
                if reader is not None:
                    frame = lens.compute_frames(
                        reader.acts,
                        [-1],
                        "thinking",
                        jl,
                        [next_id],
                        gen_id=gen_id,
                        abs_positions=[input_ids.shape[1] + len(reply_ids) - 1],
                    )[0]
                    frame["generation_run_id"] = generation_run_id
                    emit(frame)

            elapsed = time.perf_counter() - started
            durable_reply_ids = tuple(int(token_id) for token_id in reply_ids)
            text = tokenizer.decode(durable_reply_ids, skip_special_tokens=True)
            durable_reply_tokens = len(durable_reply_ids)
            if not text.endswith("�") and len(text) > len(emitted):
                emit({"type": "token", "text": text[len(emitted):]})
                emitted = text
            suffix_start_pos = int(input_ids.shape[1])
            emit(
                {
                    "type": "done",
                    "text": text,
                    "gen_id": gen_id,
                    "generation_run_id": generation_run_id,
                    "durable_reply_token_ids": list(durable_reply_ids),
                    "durable_generated_token_count": durable_reply_tokens,
                    "stop_reason": stop_reason or ("cancelled" if stop_event.is_set() else "max_tokens"),
                    "generated_token_start_pos": suffix_start_pos,
                    "generated_token_end_pos": suffix_start_pos + durable_reply_tokens,
                    "continuation_suffix_start_pos": suffix_start_pos if continue_final else None,
                    "continuation_suffix_end_pos": (
                        suffix_start_pos + durable_reply_tokens if continue_final else None
                    ),
                    "stopped": stop_event.is_set(),
                    "stats": {
                        "tokens": durable_reply_tokens,
                        "seconds": round(elapsed, 2),
                        "tok_per_s": round(durable_reply_tokens / elapsed, 2) if durable_reply_tokens and elapsed > 0 else 0.0,
                    },
                    "meta": dict(
                        meta or {},
                        sampling=sampling,
                        lens=dict(lens.meta) if lens is not None and lens.meta else None,
                        interventions=intervention_provenance["summary"] if intervention_provenance is not None else (ablator.summary() if ablator is not None else None),
                        interventions_scale=intervention_provenance["scale"] if intervention_provenance is not None else (ablator.global_scale if ablator is not None else None),
                        interventions_revision=intervention_provenance["revision"] if intervention_provenance is not None else None,
                        interventions_mode=intervention_provenance["mode"] if intervention_provenance is not None else None,
                        interventions_model_session_id=intervention_provenance["model_session_id"] if intervention_provenance is not None else None,
                        interventions_lens_binding_id=intervention_provenance["lens_binding_id"] if intervention_provenance is not None else None,
                        interventions_active_summary=intervention_provenance["active_summary"] if intervention_provenance is not None else None,
                        intervention_provenance=intervention_provenance,
                        model_session_id=context.model_session_id if context is not None else None,
                        generation_run_id=generation_run_id,
                    ),
                }
            )
            ok = True
        finally:
            cleanup_error = None
            if attachment is not None:
                try:
                    attachment.close()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    cleanup_error = cleanup_error or exc
            # aborted generation (OOM/error/hard stop): the KV cache and captured
            # activations are now dereferenced — return the blocks
            try:
                if not ok:
                    _free_cuda()
                elif torch.cuda.is_available():
                    # success path: when the device is nearly full (big model + long
                    # KV cache), the freed cache fragments the reserve and the next
                    # prefill hits costly allocator retries — generation gets slower
                    # with every message. Hand segments back once the reserve crosses
                    # 92 % of the device; a no-op (no sync, no gc) below that.
                    for i in range(torch.cuda.device_count()):
                        total = torch.cuda.get_device_properties(i).total_memory
                        if torch.cuda.memory_reserved(i) > 0.92 * total:
                            with torch.cuda.device(i):
                                torch.cuda.empty_cache()
            except Exception as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                raise cleanup_error
