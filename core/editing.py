import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import config
from core import rebase
from core.ablation import abliteration_direction, effective_coeffs

EDITS_DIR = config.DATA_DIR / "edits"
PRESETS_DIR = config.DATA_DIR / "presets"

# Residual writes edited by the global abliteration (embed aside)
TARGET_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")
REBASE_EXPORT_ROW_BUDGET = 4096


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_presets():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(PRESETS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.append({"name": path.stem, "n_rules": len(data.get("rules", [])), "model_id": data.get("model_id")})
    return out


def save_preset(name, rules, model_id, scale=1.0, mode=None, preserve_lm_head=None):
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": model_id, "saved_at": _now(), "scale": scale, "rules": rules}
    # record the intervention mode and head-protection: a preset built with the
    # output head protected must not silently become a word-killing edit when
    # re-applied under different settings
    if mode is not None:
        payload["mode"] = mode
    if preserve_lm_head is not None:
        payload["preserve_lm_head"] = bool(preserve_lm_head)
    (PRESETS_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return payload


def load_preset(name):
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown preset {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_preset(name):
    (PRESETS_DIR / f"{name}.json").unlink(missing_ok=True)


def compute_abliteration(rules, jl, scale=1.0):
    """Global pure-weight edit reproducing the abliteration-mode preview.

    Applies to EVERY residual write (embed_tokens + o_proj/down_proj of every
    layer) the same transform as the abliteration-mode hooks: for each rule,
    ``out += scale·(v̂_A·out)·w`` (applied sequentially, like the hooks). Since
    the residual is the sum of all these writes, the direction is
    removed/redirected across the whole residual — hence the fidelity (~0.97
    cosine on the logits). This is the pure-weights path for architectures the
    rebase does not support (write norms, Gemma style).

    Returns ``(tensors, info)``:
      - ``tensors``: {param_name: W_new (cpu, float32)}
      - ``info``: {tied, embed_key, lm_head_key, path, delta_max, lowrank}
        where ``lowrank`` = {param_name: (B [out, r], A [r, in])} — the SAME edit
        as per-rule rank-1 factors (delta = B·A), exact, for the LoRA export.
        For the embed, delta = (B·A)ᵀ (PEFT lookup convention).
    """
    # layers=[] = disabled rule, in this mode too (consistent with the preview)
    rules = [r for r in rules if r["layers"]]
    if not rules:
        raise ValueError("no active rule (all have 0 layers): nothing to export")
    path = jl.layout.path
    weight_u = jl._lm_head.weight
    # (v_a, w_eff) per rule, with w_eff = alpha·v̂_A + beta·v̂_B: the SAME effective
    # coefficients (saturation included) as the preview hooks
    pairs = []
    for r in rules:
        v_a, v_b = abliteration_direction(weight_u, r)
        alpha, beta = effective_coeffs(r["mode"], r["factor"], scale, r.get("keep", 0.0))
        w_eff = alpha * v_a
        if beta:
            w_eff = w_eff + beta * v_b
        pairs.append((v_a, w_eff))

    # bake on CPU: the float32 matrices (embed ~1.5 GB) don't fit alongside the
    # model on the GPU (OOM measured on 12 GB with a 4B loaded)
    def apply_cols(W):  # [d_model, d_in]: residual output = rows
        cur, us, rows = W, [], []
        for v_a, w in pairs:
            row = v_a @ cur  # composed over the previous rules
            us.append(w)
            rows.append(row)
            cur = cur + torch.outer(w, row)
        return cur, torch.stack(us, dim=1), torch.stack(rows, dim=0)

    def apply_rows(E):  # [vocab, d_model]: each ROW is a residual vector
        cur, us, rows = E, [], []
        for v_a, w in pairs:
            col = cur @ v_a  # [vocab]
            us.append(w)
            rows.append(col)
            cur = cur + torch.outer(col, w)
        return cur, torch.stack(us, dim=1), torch.stack(rows, dim=0)

    tensors = {}
    lowrank = {}
    delta_max = 0.0

    embed_key = f"{path}.{jl.layout.embed}.weight"
    E = jl._embed_tokens.weight.detach().float().cpu()
    E_new, B, A = apply_rows(E)
    delta_max = max(delta_max, (E_new - E).abs().max().item())
    tensors[embed_key] = E_new
    lowrank[embed_key] = (B, A)  # delta_embed = (B·A)ᵀ = summed outer(A_k, B_k)

    skipped_writes = 0
    for i, block in enumerate(jl.layers):
        for suffix in TARGET_SUFFIXES:
            module = block
            for part in suffix.split("."):
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None:  # e.g. linear-attention blocks (no self_attn)
                skipped_writes += 1
                continue
            W = module.weight.detach().float().cpu()
            W_new, B, A = apply_cols(W)
            delta_max = max(delta_max, (W_new - W).abs().max().item())
            name = f"{path}.layers.{i}.{suffix}.weight"
            tensors[name] = W_new
            lowrank[name] = (B, A)

    tied = jl._lm_head.weight.data_ptr() == jl._embed_tokens.weight.data_ptr()
    info = {
        "tied": tied,
        "embed_key": embed_key,
        "lm_head_key": f"{jl.layout.lm_head}.weight",
        "path": path,
        "delta_max": delta_max,
        "lowrank": lowrank,
        "skipped_writes": skipped_writes,
    }
    return tensors, info


def _abliteration_warnings(rules):
    warns = []
    for r in rules:
        if r["mode"] == "scale" and r["factor"] > 1.0:
            warns.append(
                f"\"{(r['token'] or '').strip()}\" ×{r['factor']}: amplifying (factor > 1) "
                "is approximate in pure weights (the hook composes over the layers)"
            )
    return warns


def export_abliteration(rules, jl, model_meta, *, fmt, name, source_dir=None, scale=1.0):
    """Pure-weight export (global abliteration). Formats: ``full`` (full
    checkpoint), ``layers`` (safetensors of only the modified matrices) and
    ``lora`` (exact PEFT adapter, rank = n_rules; embed omitted if embeddings
    are tied). Unties ``lm_head`` (full/layers) if the model has tied embeddings,
    to preserve the original un-embedding."""
    rules = [r for r in rules if r["layers"]]  # layers=[] = disabled rule
    if not rules:
        raise ValueError("no active intervention to export")
    if fmt not in ("full", "layers", "lora"):
        raise ValueError(f"unknown format for abliteration: {fmt}")

    tensors, info = compute_abliteration(rules, jl, scale=scale)
    if info["delta_max"] < 1e-8:
        raise ValueError(
            "the bake changes no weight (neutral factors, scale=0 or null "
            "directions) — the export would be identical to the original model"
        )
    out_dir = EDITS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if model_meta.get("dtype") == "bf16" else torch.float16
    lm_head_key = info["lm_head_key"]
    summary = [
        {k: r.get(k) for k in ("token_id", "token_ids", "token", "mode", "factor", "keep",
                               "replacement_id", "replacement_ids", "replacement")}
        for r in rules
    ]
    meta = {
        "name": name,
        "format": fmt,
        "method": "abliteration-global",
        "model_id": model_meta.get("model_id"),
        "model_revision": model_meta.get("revision"),
        "dtype": model_meta.get("dtype"),
        "global_scale": scale,
        "untied_lm_head": info["tied"] and fmt in ("full", "layers"),
        "rules": summary,
        "modified_params_count": len(tensors) + (1 if info["tied"] else 0),
        "warnings": _abliteration_warnings(rules) + (
            [f"{info['skipped_writes']} residual write(s) without o_proj/down_proj "
             "(hybrid architecture) left untouched — the bake is partial there; "
             "prefer read projection when the architecture supports it"]
            if info["skipped_writes"] else []
        ),
        "note": (
            "global abliteration: the token's direction is removed/redirected in "
            "every residual write (embed + o_proj/down_proj of all layers). "
            "Reproduces the abliteration-mode preview (~0.97 cosine on the logits). "
            "Pure weights: a standard safetensors checkpoint."
        ),
        "created_at": _now(),
    }

    if fmt == "layers":
        out = {k: v.to(dtype) for k, v in tensors.items()}
        if info["tied"]:
            # original un-embedding (unedited embed) to write separately
            out[lm_head_key] = jl._embed_tokens.weight.detach().to(dtype).cpu()
        save_file(out, str(out_dir / "modified_layers.safetensors"))

    elif fmt == "lora":
        # The abliteration delta is EXACTLY rank-n_rules per matrix (delta = B·A),
        # so the LoRA is exact — except the embed of a tied-embeddings model: PEFT
        # can't untie lm_head, and editing the embed would corrupt the shared
        # un-embedding → we omit it (reduced fidelity).
        include_embed = not info["tied"]
        if not include_embed:
            meta["warnings"] = meta["warnings"] + [
                "tied embeddings: the embed is not included in the LoRA (PEFT "
                "cannot untie lm_head) — prefer \"full checkpoint\" for maximum "
                "fidelity"
            ]
        out = {}
        target_modules = set()
        for pname, (B, A) in info["lowrank"].items():
            base = pname.removesuffix(".weight")
            if pname == info["embed_key"]:
                if not include_embed:
                    continue
                target_modules.add(base.rsplit(".", 1)[-1])
                # PEFT Embedding convention: delta_lookup = (B·A)ᵀ,
                # A = lora_embedding_A [r, vocab], B = lora_embedding_B [d_model, r]
                out[f"base_model.model.{base}.lora_embedding_A"] = A.contiguous()
                out[f"base_model.model.{base}.lora_embedding_B"] = B.contiguous()
            else:
                target_modules.add(base.rsplit(".", 1)[-1])
                out[f"base_model.model.{base}.lora_A.weight"] = A.contiguous()
                out[f"base_model.model.{base}.lora_B.weight"] = B.contiguous()
        rank = len(rules)
        save_file(out, str(out_dir / "adapter_model.safetensors"))
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": model_meta.get("model_id"),
            "r": rank,
            "lora_alpha": rank,
            "lora_dropout": 0.0,
            "target_modules": sorted(target_modules),
            "bias": "none",
            "fan_in_fan_out": False,
            "task_type": "CAUSAL_LM",
        }
        (out_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=1), encoding="utf-8"
        )

    elif fmt == "full":
        if source_dir is None or not Path(source_dir).is_dir():
            raise ValueError("full checkpoint: model source folder not found")
        source_dir = Path(source_dir)
        shards = sorted(source_dir.glob("*.safetensors"))
        if not shards:
            raise ValueError("full checkpoint: no safetensors in the source")

        lm_head_value = None  # original un-embedding (if tied) = original embed from disk
        embed_shard_name = None
        seen = set()
        for shard in shards:
            ino = shard.stat().st_ino
            if ino in seen:
                continue
            seen.add(ino)
            out = {}
            with safe_open(str(shard), framework="pt") as f:
                keys = list(f.keys())
                for key in keys:
                    original = f.get_tensor(key)
                    if info["tied"] and key == info["embed_key"]:
                        lm_head_value = original.clone()  # BEFORE editing
                        embed_shard_name = shard.name
                    out[key] = tensors[key].to(original.dtype) if key in tensors else original
            # if this shard already carries lm_head (untied model), don't touch it
            save_file(out, str(out_dir / shard.name))

        # untie: add lm_head.weight (= original embed) into the embed's shard
        if info["tied"]:
            if lm_head_value is None:
                raise ValueError("cannot untie: embed not found in the source")
            target_shard = out_dir / embed_shard_name
            with safe_open(str(target_shard), framework="pt") as f:
                merged = {k: f.get_tensor(k) for k in f.keys()}
            merged[lm_head_key] = lm_head_value
            save_file(merged, str(target_shard))

        # config.json: copy, force tie_word_embeddings=False if untied
        cfg_path = source_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if info["tied"]:
                cfg["tie_word_embeddings"] = False
            (out_dir / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        # other tokenizer/config files (json, merges.txt, tokenizer.model…):
        # copy as-is, then fix the index if present
        for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"):
            for extra in source_dir.glob(pattern):
                if extra.name == "config.json":
                    continue
                shutil.copy2(extra, out_dir / extra.name)
        index_path = out_dir / "model.safetensors.index.json"
        if info["tied"] and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            wm = index.setdefault("weight_map", {})
            wm[lm_head_key] = embed_shard_name
            if "metadata" in index and "total_size" in index["metadata"]:
                index["metadata"]["total_size"] += lm_head_value.numel() * lm_head_value.element_size()
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

    (out_dir / "edit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"out_dir": str(out_dir), **meta}


def _disk_mapper(mem_embed_key, disk_keys):
    """Memory keys (instantiated model's layout) → disk checkpoint keys.

    transformers renames on load: e.g. Qwen3.5 is instantiated as ForCausalLM
    ("model.layers.*" in memory) but saved in ConditionalGeneration format
    ("model.language_model.layers.*"). Without this mapping, a "full" export
    would copy the source verbatim without transforming anything. We anchor the
    disk prefix on the embed, whose suffix is unique in the checkpoint."""
    if mem_embed_key in disk_keys:
        return lambda key: key
    suffix = "." + ".".join(mem_embed_key.rsplit(".", 2)[-2:])  # ".embed_tokens.weight"
    candidates = [k for k in disk_keys if k.endswith(suffix)]
    if len(candidates) != 1:
        raise ValueError(
            f"checkpoint prefix undecidable: {mem_embed_key} absent from the source "
            f"and {len(candidates)} key(s) end with {suffix}"
        )
    mem_prefix = mem_embed_key.removesuffix(suffix)
    disk_prefix = candidates[0].removesuffix(suffix)

    def to_disk(key):
        if key == mem_prefix or key.startswith(mem_prefix + "."):
            return disk_prefix + key[len(mem_prefix):]
        return key

    return to_disk


def apply_read_transform_bounded(entry, tensor, *, row_budget=REBASE_EXPORT_ROW_BUDGET,
                                 observer=None):
    """Apply a packed rank-3 read transform with bounded float32 temporaries."""
    if tensor.ndim != 3 or entry[0] != "read":
        raise ValueError("bounded transform requires a rank-3 read tensor")
    if row_budget < 1:
        raise ValueError("row_budget must be positive")
    source = tensor.detach().cpu().contiguous()
    rows = source.reshape(-1, source.shape[-1])
    destination = torch.empty_like(source)
    dest_rows = destination.reshape_as(rows)
    _kind, Ug, Vg = entry
    delta_max = 0.0
    for start in range(0, rows.shape[0], row_budget):
        count = min(row_budget, rows.shape[0] - start)
        if observer is not None:
            observer(count)
        chunk = rows[start:start + count].float()
        updated = chunk + (chunk @ Ug) @ Vg.T
        delta_max = max(delta_max, float((updated - chunk).abs().max()))
        dest_rows[start:start + count].copy_(updated.to(source.dtype))
    return destination, delta_max


def export_rebase(rules, jl, model_meta, *, fmt, name, source_dir=None, scale=1.0,
                  exact=False, preserve_lm_head=False):
    """Pure-weight export by change of basis of the reads (cf. core/rebase).

    ``readthrough`` (exact=False): the downstream read matrices + lm_head.
    ``exact``: adds the counter-transform of the downstream writes.
    ``preserve_lm_head``: leaves lm_head untransformed (output vocabulary
    intact); a tied model then needs no untying either.
    Formats: ``full`` (checkpoint), ``layers`` (safetensors of the modified
    matrices) and ``lora`` (PEFT adapter = the exact low-rank diff between the
    baked weights and the originals; the lm_head delta is applied at forward
    time, so tied embeddings need no untying). The bake is done streaming, one
    float32 CPU matrix at a time. Tied-embeddings model (full/layers): the embed
    stays INTACT, it's lm_head (untied) that receives the final read transform."""
    method = "rebase-exact" if exact else "rebase-readthrough"
    if fmt not in ("full", "layers", "lora"):
        raise ValueError(f"unknown format for {method}: {fmt}")
    transforms, info = rebase.build_plan(
        rules, jl, scale, exact=exact, preserve_lm_head=preserve_lm_head)
    if not transforms:
        raise ValueError(
            "nothing to bake: with the output head preserved, rules hooked on "
            "the last layer only transform no downstream read"
        )
    lm_head_key = info["lm_head_key"]
    if info["packed_moe"]:
        if exact:
            raise ValueError("Exact is not implemented for packed MoE; use Readthrough")
        if fmt == "lora":
            raise ValueError("LoRA is not implemented for packed MoE; use Full Readthrough")
        if fmt == "layers":
            raise ValueError("Layers is not implemented for packed MoE; use Full Readthrough")

    delta_max = 0.0
    applied = set()

    def bake(key, tensor):
        nonlocal delta_max
        if tensor.ndim == 3 and transforms[key][0] == "read":
            W_new, chunk_delta = apply_read_transform_bounded(
                transforms[key], tensor, row_budget=REBASE_EXPORT_ROW_BUDGET
            )
            delta_max = max(delta_max, chunk_delta)
            applied.add(key)
            return W_new
        W = tensor.detach().to("cpu", torch.float32)
        W_new, _B, _A = rebase.apply_transform(transforms[key], W)
        delta_max = max(delta_max, (W_new - W).abs().max().item())
        applied.add(key)
        return W_new

    out_dir = EDITS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if model_meta.get("dtype") == "bf16" else torch.float16
    warnings = []
    if exact and info["regularized_layers"]:
        warnings.append(
            "regularized inverse (full zap ⇒ singular transform) on layers "
            f"{info['regularized_layers']} — the effect there equals readthrough; "
            "prefer readthrough mode for full removals"
        )
    if fmt == "lora" and info["tied"] and lm_head_key in transforms:
        warnings.append(
            "tied embeddings: use the adapter at runtime (PEFT applies the "
            "lm_head delta at forward time, leaving the shared embed intact); "
            "merging it into the base weights (merge_and_unload) would write "
            "that delta into the embed too — export a full checkpoint if you "
            "need merged weights"
        )

    def source_weight(state, key):
        source = state.get(key)
        if source is None and key == lm_head_key and info["tied"]:
            source = state[info["embed_key"]]  # tied: the un-embedding IS the embed
        if source is None:
            raise ValueError(
                f"parameter {key} not found in the loaded model — "
                "unexpected layout, export cancelled"
            )
        return source

    if fmt == "layers":
        state = jl._hf_model.state_dict()
        to_disk = lambda key: key  # noqa: E731 — refined if the source is available
        if source_dir is not None and Path(source_dir).is_dir():
            disk_keys = set()
            for shard in Path(source_dir).glob("*.safetensors"):
                with safe_open(str(shard), framework="pt") as f:
                    disk_keys.update(f.keys())
            if disk_keys:
                to_disk = _disk_mapper(info["embed_key"], disk_keys)
        tensors = {}
        for key in transforms:
            tensors[to_disk(key)] = bake(key, source_weight(state, key)).to(dtype)
        save_file(tensors, str(out_dir / "modified_layers.safetensors"))

    elif fmt == "lora":
        # The rebase delta is low-rank by construction (delta = B·A exactly, cf.
        # rebase.apply_transform): the adapter is the exact diff between the
        # baked weights and the originals, not an approximation. lm_head: PEFT
        # adds the delta at forward time without writing to the (possibly tied)
        # weight, so the un-embedding is effectively untied while the embed
        # stays intact. Module names follow the model as instantiated by
        # AutoModelForCausalLM (the same loading path as the UI).
        state = jl._hf_model.state_dict()
        factors = {}
        max_rank = 0
        for key in transforms:
            W = source_weight(state, key).detach().to("cpu", torch.float32)
            W_new, B, A = rebase.apply_transform(transforms[key], W)
            delta_max = max(delta_max, (W_new - W).abs().max().item())
            applied.add(key)
            factors[key] = (B, A)
            max_rank = max(max_rank, B.shape[1])
        tensors = {}
        module_paths = []
        for key, (B, A) in factors.items():
            base = key.removesuffix(".weight")
            module_paths.append(base)
            if B.shape[1] < max_rank:  # pad so a single config `r` fits every module
                pad = max_rank - B.shape[1]
                B = torch.cat([B, torch.zeros(B.shape[0], pad)], dim=1)
                A = torch.cat([A, torch.zeros(pad, A.shape[1])], dim=0)
            tensors[f"base_model.model.{base}.lora_A.weight"] = A.contiguous()
            tensors[f"base_model.model.{base}.lora_B.weight"] = B.contiguous()
        save_file(tensors, str(out_dir / "adapter_model.safetensors"))
        # target_modules as an anchored regex over the modules actually edited:
        # a plain suffix list would wrap the same projections in EVERY layer and
        # leave benign but alarming "missing adapter keys" warnings at load time
        target_regex = "(.*\\.)?(" + "|".join(re.escape(p) for p in sorted(module_paths)) + ")"
        adapter_config = {
            "peft_type": "LORA",
            "base_model_name_or_path": model_meta.get("model_id"),
            "r": max_rank,
            "lora_alpha": max_rank,  # scaling alpha/r = 1: B·A is the raw delta
            "lora_dropout": 0.0,
            "target_modules": target_regex,
            "bias": "none",
            "fan_in_fan_out": False,
            "task_type": "CAUSAL_LM",
        }
        (out_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, indent=1), encoding="utf-8"
        )

    elif fmt == "full":
        if source_dir is None or not Path(source_dir).is_dir():
            raise ValueError("full checkpoint: model source folder not found")
        source_dir = Path(source_dir)
        shards = sorted(source_dir.glob("*.safetensors"))
        if not shards:
            raise ValueError("full checkpoint: no safetensors in the source")

        disk_keys = set()
        seen = set()
        for shard in shards:
            ino = shard.stat().st_ino
            if ino in seen:
                continue
            seen.add(ino)
            with safe_open(str(shard), framework="pt") as f:
                disk_keys.update(f.keys())
        to_disk = _disk_mapper(info["embed_key"], disk_keys)
        transforms = {to_disk(k): fn for k, fn in transforms.items()}
        lm_head_key = to_disk(lm_head_key)
        embed_key = to_disk(info["embed_key"])

        lm_head_written = False
        embed_shard_name = None
        seen = set()
        for shard in shards:
            ino = shard.stat().st_ino
            if ino in seen:
                continue
            seen.add(ino)
            out = {}
            with safe_open(str(shard), framework="pt") as f:
                for key in f.keys():
                    original = f.get_tensor(key)
                    if key in transforms:
                        out[key] = bake(key, original).to(original.dtype)
                        if key == lm_head_key:
                            lm_head_written = True
                    else:
                        out[key] = original
                    if key == embed_key:
                        embed_shard_name = shard.name
            save_file(out, str(out_dir / shard.name))
            del out

        # untie: the transformed un-embedding becomes a separate lm_head, baked
        # from the original embed (which stays intact). With the output head
        # preserved, lm_head is not transformed and no untying is needed.
        needs_untie = info["tied"] and lm_head_key in transforms
        if needs_untie and not lm_head_written:
            if embed_shard_name is None:
                raise ValueError("cannot untie: embed not found in the source")
            target_shard = out_dir / embed_shard_name
            with safe_open(str(target_shard), framework="pt") as f:
                merged = {k: f.get_tensor(k) for k in f.keys()}
            embed_original = merged[embed_key]
            lm_head_value = bake(lm_head_key, embed_original).to(embed_original.dtype)
            merged[lm_head_key] = lm_head_value
            save_file(merged, str(out_dir / embed_shard_name))
            del merged

        missing = set(transforms) - applied
        if missing:
            sample = sorted(missing)[:3]
            raise ValueError(
                f"{len(missing)} parameter(s) to transform absent from the source "
                f"checkpoint (e.g. {sample}) — unexpected key names, export cancelled "
                "(the written checkpoint would be partially original)"
            )

        cfg_path = source_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if needs_untie:
                cfg["tie_word_embeddings"] = False
                text_cfg = cfg.get("text_config")
                if isinstance(text_cfg, dict) and "tie_word_embeddings" in text_cfg:
                    text_cfg["tie_word_embeddings"] = False
            (out_dir / "config.json").write_text(
                json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja"):
            for extra in source_dir.glob(pattern):
                if extra.name == "config.json":
                    continue
                shutil.copy2(extra, out_dir / extra.name)
        index_path = out_dir / "model.safetensors.index.json"
        if needs_untie and not lm_head_written and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            wm = index.setdefault("weight_map", {})
            wm[lm_head_key] = embed_shard_name
            if "metadata" in index and "total_size" in index["metadata"]:
                index["metadata"]["total_size"] += (
                    lm_head_value.numel() * lm_head_value.element_size()
                )
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

    if delta_max < 1e-8:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise ValueError(
            "the bake changes no weight (null directions?) — the export would be "
            "identical to the original model, folder deleted"
        )

    summary = [
        {k: r.get(k) for k in ("token_id", "token_ids", "token", "mode", "factor", "keep",
                               "replacement_id", "replacement_ids", "replacement", "layers")}
        for r in rules if r["layers"]
    ]
    head_transformed = lm_head_key in transforms
    meta = {
        "name": name,
        "format": fmt,
        "method": method,
        "model_id": model_meta.get("model_id"),
        "model_revision": model_meta.get("revision"),
        "dtype": model_meta.get("dtype"),
        "global_scale": scale,
        "preserve_lm_head": preserve_lm_head,
        # lora: no physical untying — the lm_head delta lives in the adapter
        "untied_lm_head": info["tied"] and fmt != "lora" and head_transformed,
        "rules": summary,
        "layers_span": info["layers_span"],
        "rank": info["rank_final"],
        "modified_params_count": len(transforms),
        "delta_max": delta_max,
        "min_gamma": info["min_gamma"],
        "warnings": warnings,
        "note": (
            "change of basis of the reads: every matrix that READS the residual "
            "downstream of the hooked layers (q/k/v, in_proj*, gate/up"
            + (" + lm_head" if head_transformed else "; lm_head preserved") + ") sees "
            "the residual transformed by the same J-space directions as the live preview"
            + (" ; downstream writes counter-transformed (exact mode)" if exact else "")
            + ". Pure weights: a standard safetensors checkpoint."
        ),
        "created_at": _now(),
    }
    (out_dir / "edit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return {"out_dir": str(out_dir), **meta}
