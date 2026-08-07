import hashlib
import itertools
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch
from jlens.lens import JacobianLens

import config

GEN_STORE_MAX = 4


@dataclass(frozen=True)
class GenerationPublicationResult:
    published: bool
    gen_id: int
    generation_run_id: str


MASKS_DIR = config.DATA_DIR / "masks"

# Last range of layers captured per lens: {lens key: [layers]}.
# Avoids re-entering the range on every reload (user request).
LENS_PREFS_PATH = config.DATA_DIR / "lens_prefs.json"


def _load_lens_prefs():
    try:
        return json.loads(LENS_PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_lens_pref(key, layers):
    prefs = _load_lens_prefs()
    prefs[key] = [int(l) for l in layers]
    LENS_PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LENS_PREFS_PATH.write_text(json.dumps(prefs, indent=1), encoding="utf-8")


def _lens_pref_key(source):
    if source.get("path"):
        return f"path:{source['path']}"
    return f"hub:{source['repo_id']}:{source['filename']}@{source.get('revision') or 'main'}"


class ActivationCatcher:
    def __init__(self, layers, indices):
        self.acts = {}
        self._closed = False
        self._handles = []
        try:
            for i in indices:
                self._handles.append(layers[i].register_forward_hook(self._make(i)))
        except Exception:
            self.close()
            raise

    def _make(self, index):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            self.acts[index] = tensor.detach()

        return hook

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        handles, self._handles = self._handles, []
        first = None
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:
                if first is None:
                    first = exc
        if first is not None:
            raise first


def _vocab_fingerprint(tokenizer):
    payload = json.dumps(sorted(tokenizer.get_vocab().items()), ensure_ascii=False)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _wordlike(raw):
    s = raw.strip()
    if len(s) < 1 or "<|" in s or (s.startswith("<") and s.endswith(">")):
        return False
    if s.isascii():
        return (
            raw.startswith(" ")
            and len(s) > 2
            and s[0].isalpha()
            and all(c.isalpha() or c in "'-" for c in s)
        )
    return all(ch.isalnum() for ch in s)


def display_token_mask(tokenizer, vocab_size):
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    path = MASKS_DIR / f"{_vocab_fingerprint(tokenizer)}_{vocab_size}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    n_decodable = min(vocab_size, len(tokenizer))
    decoded = tokenizer.batch_decode(
        [[tid] for tid in range(n_decodable)], clean_up_tokenization_spaces=False
    )
    for tid, raw in enumerate(decoded):
        mask[tid] = _wordlike(raw)
    torch.save(mask, path)
    return mask


class LensGenerationView:
    def __init__(self, manager):
        self._manager = manager
        self.lens = manager.lens
        self.meta = dict(manager.meta) if manager.meta else None
        self.layers = list(manager.layers)
        self.k = manager.k
        self.mask = manager.mask
        self._J = manager._J
        self.binding_id = manager.binding_id
        self.model_session_id = manager.model_session_id

    def start_gen(self, generation_run_id=None, provisional=False):
        return self._manager.start_gen(
            generation_run_id=generation_run_id, provisional=provisional,
        )

    def finalize_gen(self, gen_id, *, retained_thinking=None, publish=True):
        return self._manager.finalize_gen(
            gen_id, retained_thinking=retained_thinking, publish=publish,
        )

    def discard_gen(self, gen_id):
        return self._manager.discard_gen(gen_id)

    def unpublish_gen(self, gen_id, *, generation_run_id, model_session_id, lens_binding_id):
        return self._manager.unpublish_gen(
            gen_id,
            generation_run_id=generation_run_id,
            model_session_id=model_session_id,
            lens_binding_id=lens_binding_id,
        )

    def compute_frames(self, acts, positions, phase, jl, token_ids, gen_id=None, abs_positions=None, chunk=None):
        old_layers, old_k, old_mask, old_J = self._manager.layers, self._manager.k, self._manager.mask, self._manager._J
        try:
            self._manager.layers, self._manager.k, self._manager.mask, self._manager._J = self.layers, self.k, self.mask, self._J
            return self._manager.compute_frames(acts, positions, phase, jl, token_ids, gen_id=gen_id, abs_positions=abs_positions, chunk=chunk)
        finally:
            self._manager.layers, self._manager.k, self._manager.mask, self._manager._J = old_layers, old_k, old_mask, old_J


class LensManager:
    def __init__(self):
        self._lock = threading.Lock()
        self.lens = None
        self.meta = None
        self.layers = []
        self.k = 8
        self.mask = None
        self._J = None
        self._tok_strs = {}
        self.gen_store = OrderedDict()
        self._provisional_gen_store = {}
        self._gen_counter = itertools.count(1)
        self._pref_key = None
        self.binding_id = 0
        self.model_session_id = None

    def snapshot_for_generation(self):
        with self._lock:
            return LensGenerationView(self) if self.lens is not None else None

    def state_record(self):
        with self._lock:
            return {
                "lens": self.lens,
                "meta": dict(self.meta) if self.meta else None,
                "layers": list(self.layers),
                "k": self.k,
                "mask": self.mask,
                "_J": self._J,
                "_tok_strs": dict(self._tok_strs),
                "_pref_key": self._pref_key,
                "binding_id": self.binding_id,
                "model_session_id": self.model_session_id,
            }

    def restore_state_record(self, record):
        with self._lock:
            self.lens = record["lens"]
            self.meta = dict(record["meta"]) if record["meta"] else None
            self.layers = list(record["layers"])
            self.k = record["k"]
            self.mask = record["mask"]
            self._J = record["_J"]
            self._tok_strs = dict(record["_tok_strs"])
            self._pref_key = record["_pref_key"]
            self.binding_id = record["binding_id"]
            self.model_session_id = record["model_session_id"]

    def load(self, model_manager, *, repo_id=None, filename="lens.pt", revision=None,
             path=None, layers=None, k=8):
        snap = model_manager.session_snapshot() if hasattr(model_manager, "session_snapshot") else None
        if model_manager.hf_model is None:
            raise ValueError("load a model first")
        model_session_id = snap.model_session_id if snap is not None else None
        if path:
            lens = JacobianLens.from_pretrained(path)
            source = {"path": path, "repo_id": None, "filename": None, "revision": None}
        else:
            lens = JacobianLens.from_pretrained(repo_id, filename=filename, revision=revision)
            source = {"path": None, "repo_id": repo_id, "filename": filename, "revision": revision}

        model_meta = dict(model_manager.meta)
        if lens.d_model != model_meta["d_model"]:
            raise ValueError(f"lens d_model ({lens.d_model}) != model ({model_meta['d_model']})")
        n_layers = model_meta["n_layers"]
        fitted = lens.source_layers
        if fitted[-1] >= n_layers:
            raise ValueError(f"the lens covers layer {fitted[-1]}, outside a model with {n_layers} layers")
        pref_key = _lens_pref_key(source)
        if layers:
            tapped = sorted(set(layers) & set(fitted))
            if not tapped:
                raise ValueError(f"no requested layer is fitted (fitted: {fitted[0]}..{fitted[-1]})")
        else:
            saved = _load_lens_prefs().get(pref_key)
            tapped = (sorted(set(saved) & set(fitted)) if saved else None) or list(fitted)
        _save_lens_pref(pref_key, tapped)

        device = model_manager.jl.input_device
        stacked = torch.stack([lens.jacobians[l].float() for l in tapped]).to(device)
        tokenizer = model_manager.tokenizer
        vocab_size = model_manager.hf_model.get_output_embeddings().weight.shape[0]
        mask = display_token_mask(tokenizer, vocab_size).to(device)

        warnings = []
        if model_meta.get("quant"):
            warnings.append(
                f"model loaded in {model_meta['quant']}: the lens was probably "
                "fitted on the unquantized weights, the readouts may drift"
            )
        if model_meta["model_id"].startswith("local/"):
            warnings.append("local model: cannot verify that the lens matches these exact weights")

        with self._lock:
            binding_id = self.binding_id + 1
            meta = {
                **source,
                "model_session_id": model_session_id,
                "lens_binding_id": binding_id,
                "model_id": model_meta["model_id"],
                "model_revision": model_meta.get("revision"),
                "d_model": lens.d_model,
                "n_prompts": lens.n_prompts,
                "fitted_layers": [int(fitted[0]), int(fitted[-1])],
                "fitted_layers_all": [int(l) for l in fitted],
                "tapped_layers": [int(l) for l in tapped],
                "k": int(k),
                "warnings": warnings,
            }
            self.binding_id = binding_id
            self.model_session_id = model_session_id
            self.lens = lens
            self.layers = tapped
            self.k = int(k)
            self.mask = mask
            self._J = stacked
            self._tok_strs = {}
            self._pref_key = pref_key
            self.meta = meta
            return dict(self.meta)

    def set_layers(self, model_manager, layers, k=None):
        with self._lock:
            if self.lens is None:
                raise ValueError("no lens loaded")
            lens = self.lens
            old_k = self.k
            pref_key = self._pref_key
            old_meta = dict(self.meta)
            binding_id = self.binding_id + 1
        fitted = lens.source_layers
        tapped = sorted(set(layers) & set(fitted))
        if not tapped:
            raise ValueError(f"no requested layer is fitted (fitted: {fitted[0]}..{fitted[-1]})")
        device = model_manager.jl.input_device
        stacked = torch.stack([lens.jacobians[l].float() for l in tapped]).to(device)
        new_k = int(k) if k else old_k
        if pref_key:
            _save_lens_pref(pref_key, tapped)
        with self._lock:
            if self.lens is not lens:
                raise ValueError("lens changed while updating layers")
            self.layers = tapped
            self._J = stacked
            self.k = new_k
            self.binding_id = binding_id
            self.meta = dict(old_meta, lens_binding_id=self.binding_id, tapped_layers=[int(l) for l in tapped], k=self.k)
            return dict(self.meta)

    def unload(self):
        old = self.withdraw()
        cleanup_error = self.cleanup_withdrawn(old)
        if cleanup_error is not None:
            raise cleanup_error
        return {"unloaded": True}

    def withdraw(self):
        with self._lock:
            old = {
                "lens": self.lens,
                "meta": self.meta,
                "layers": self.layers,
                "k": self.k,
                "mask": self.mask,
                "_J": self._J,
                "_tok_strs": self._tok_strs,
                "_pref_key": self._pref_key,
                "binding_id": self.binding_id,
                "model_session_id": self.model_session_id,
                "gen_store": self.gen_store,
                "provisional_gen_store": self._provisional_gen_store,
            }
            self.lens = None
            self.meta = None
            self.layers = []
            self.mask = None
            self._J = None
            self._tok_strs = {}
            self.gen_store = OrderedDict()
            self._provisional_gen_store = {}
            self.model_session_id = None
            return old

    def restore_withdrawn(self, old):
        if not old:
            return
        with self._lock:
            self.lens = old["lens"]
            self.meta = old["meta"]
            self.layers = old["layers"]
            self.k = old["k"]
            self.mask = old["mask"]
            self._J = old["_J"]
            self._tok_strs = old["_tok_strs"]
            self._pref_key = old["_pref_key"]
            self.binding_id = old["binding_id"]
            self.model_session_id = old["model_session_id"]
            self.gen_store = old["gen_store"]
            self._provisional_gen_store = old.get("provisional_gen_store", {})

    @staticmethod
    def cleanup_withdrawn(_old):
        import gc
        first_error = None
        if isinstance(_old, dict):
            _old.clear()
        _old = None
        try:
            gc.collect()
        except Exception as exc:
            first_error = exc
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            if first_error is None:
                first_error = exc
        return first_error

    def start_gen(self, generation_run_id=None, provisional=False):
        gen_id = next(self._gen_counter)
        store = {
            "generation_run_id": generation_run_id,
            "model_session_id": self.model_session_id,
            "lens_binding_id": self.binding_id,
            "layers": list(self.layers),
            "residuals": {l: [] for l in self.layers},
            "positions": [],
            "token_ids": [],
            "phases": [],
            "finalized_thinking": None,
        }
        if provisional:
            self._provisional_gen_store[gen_id] = store
        else:
            self.gen_store[gen_id] = store
            while len(self.gen_store) > GEN_STORE_MAX:
                self.gen_store.popitem(last=False)
        return gen_id

    def finalize_gen(self, gen_id, *, retained_thinking=None, publish=True):
        store = self._provisional_gen_store.get(gen_id)
        already_published = False
        if store is None:
            store = self.gen_store.get(gen_id)
            already_published = store is not None
        if store is None:
            raise ValueError("unknown provisional generation")
        run_id = store.get("generation_run_id")
        if (
            not isinstance(run_id, str) or len(run_id) != 32
            or any(ch not in "0123456789abcdef" for ch in run_id)
        ):
            raise ValueError("generation run identity is invalid")
        if (
            store.get("model_session_id") != self.model_session_id
            or store.get("lens_binding_id") != self.binding_id
        ):
            raise ValueError("generation binding is no longer current")

        retained = None if retained_thinking is None else tuple(
            (int(position), int(token_id)) for position, token_id in retained_thinking
        )
        previous = store.get("finalized_thinking")
        if previous is not None:
            if retained is not None and retained != previous:
                raise ValueError("generation was finalized with a different retained sequence")
        else:
            if retained is None:
                raise ValueError("retained thinking sequence is required before publication")
            positions = store["positions"]
            token_ids = store["token_ids"]
            phases = store["phases"]
            count = len(positions)
            if len(token_ids) != count or len(phases) != count:
                raise ValueError("generation residual metadata is not aligned")
            residual_rows = {}
            for layer in store["layers"]:
                chunks = store["residuals"].get(layer)
                if chunks is None:
                    raise ValueError("generation residual layer is missing")
                rows = sum(int(chunk.shape[0]) for chunk in chunks)
                if rows != count:
                    raise ValueError("generation residual rows are not aligned")
                residual_rows[layer] = torch.cat(chunks, dim=0) if chunks else None

            retained_index = 0
            keep = []
            for phase, position, token_id in zip(phases, positions, token_ids):
                if phase == "reading":
                    keep.append(True)
                elif phase == "thinking":
                    wanted = retained_index < len(retained) and retained[retained_index] == (position, token_id)
                    keep.append(wanted)
                    if wanted:
                        retained_index += 1
                else:
                    raise ValueError("generation residual phase is invalid")
            if retained_index != len(retained):
                raise ValueError("retained thinking sequence is not present in captured residuals")
            for layer, rows in residual_rows.items():
                if rows is None:
                    store["residuals"][layer] = []
                else:
                    mask = torch.tensor(keep, dtype=torch.bool, device=rows.device)
                    store["residuals"][layer] = [rows[mask]] if any(keep) else []
            store["positions"] = [value for value, selected in zip(positions, keep) if selected]
            store["token_ids"] = [value for value, selected in zip(token_ids, keep) if selected]
            store["phases"] = [value for value, selected in zip(phases, keep) if selected]
            store["finalized_thinking"] = retained

        if publish and not already_published:
            # Publication is an in-memory transaction.  Keep the provisional
            # owner until committed visibility and eviction both succeed, and
            # restore both maps exactly if any injected/runtime failure occurs.
            committed_before = OrderedDict(self.gen_store)
            provisional_before = dict(self._provisional_gen_store)
            try:
                self.gen_store[gen_id] = store
                while len(self.gen_store) > GEN_STORE_MAX:
                    self.gen_store.popitem(last=False)
                removed = self._provisional_gen_store.pop(gen_id, None)
                if removed is not store:
                    raise RuntimeError("provisional generation ownership changed during publication")
            except BaseException:
                self.gen_store.clear()
                self.gen_store.update(committed_before)
                self._provisional_gen_store.clear()
                self._provisional_gen_store.update(provisional_before)
                raise
        return GenerationPublicationResult(
            published=publish or already_published,
            gen_id=int(gen_id),
            generation_run_id=run_id,
        )

    def discard_gen(self, gen_id):
        store = self._provisional_gen_store.pop(gen_id, None)
        if store is not None:
            store.clear()
            return True
        return False

    def unpublish_gen(self, gen_id, *, generation_run_id, model_session_id, lens_binding_id):
        """Remove exactly one committed run after durable publication rollback."""
        store = self.gen_store.get(gen_id)
        if store is None:
            return False
        if (
            store.get("generation_run_id") != generation_run_id
            or store.get("model_session_id") != model_session_id
            or store.get("lens_binding_id") != lens_binding_id
        ):
            return False
        self.gen_store.pop(gen_id, None)
        store.clear()
        return True

    def has_published_gen(self, gen_id, generation_run_id):
        store = self.gen_store.get(gen_id)
        return bool(
            store is not None
            and store.get("generation_run_id") == generation_run_id
            and store.get("model_session_id") == self.model_session_id
            and store.get("lens_binding_id") == self.binding_id
        )

    @torch.no_grad()
    def pin_ranks(self, gen_id, token_ids, jl, chunk=32, generation_run_id=None):
        store = self.gen_store.get(gen_id)
        if store is not None and (store.get("model_session_id") != self.model_session_id or store.get("lens_binding_id") != self.binding_id):
            store = None
        if store is None:
            raise ValueError("unknown generation (residual store expired)")
        if generation_run_id is None or store.get("generation_run_id") != generation_run_id:
            raise ValueError("generation run identity is missing or no longer current")
        layers = store["layers"]
        device = self._J.device
        tids = torch.tensor(token_ids, dtype=torch.long, device=device)
        pins = {
            int(t): {"ranks": [], "p": []} for t in token_ids
        }
        for layer in layers:
            residuals = torch.cat(store["residuals"][layer]).to(device).float()
            J = self.lens.jacobians[layer].float().to(device)
            layer_ranks = {int(t): [] for t in token_ids}
            layer_p = {int(t): [] for t in token_ids}
            for start in range(0, residuals.shape[0], chunk):
                h = residuals[start : start + chunk]
                logits = jl.unembed(h @ J.T).float()
                probs = torch.softmax(logits, -1)
                sel = logits[:, tids]
                rank = (logits.unsqueeze(-1) > sel.unsqueeze(1)).sum(1)
                p_sel = probs[:, tids]
                rank_l, p_l = rank.tolist(), p_sel.tolist()
                for ti, t in enumerate(token_ids):
                    layer_ranks[int(t)].extend(row[ti] for row in rank_l)
                    layer_p[int(t)].extend(round(row[ti], 6) for row in p_l)
            for t in token_ids:
                pins[int(t)]["ranks"].append(layer_ranks[int(t)])
                pins[int(t)]["p"].append(layer_p[int(t)])
        return {
            "gen_id": gen_id,
            "generation_run_id": store["generation_run_id"],
            "layers": [int(l) for l in layers],
            "positions": store["positions"],
            "phases": store["phases"],
            "tokens": self._strs(jl.tokenizer, store["token_ids"]),
            "pins": pins,
        }

    def _strs(self, tokenizer, ids):
        out = []
        for tid in ids:
            s = self._tok_strs.get(tid)
            if s is None:
                s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
                self._tok_strs[tid] = s
            out.append(s)
        return out

    @torch.no_grad()
    def compute_frames(self, acts, positions, phase, jl, token_ids, gen_id=None,
                       abs_positions=None, chunk=None):
        tokenizer = jl.tokenizer
        if chunk is None:
            chunk = max(1, 96 // max(1, len(self.layers)))
        if abs_positions is None:
            abs_positions = positions
        frames = [
            {
                "type": "frame",
                "phase": phase,
                "pos": int(pos),
                "token_id": int(tid),
                "tok": self._strs(tokenizer, [tid])[0],
                "gen": gen_id,
                "layers": {},
            }
            for pos, tid in zip(abs_positions, token_ids)
        ]
        store = None
        if gen_id is not None:
            store = self._provisional_gen_store.get(gen_id) or self.gen_store.get(gen_id)
        if store is not None:
            store["positions"].extend(int(p) for p in abs_positions)
            store["token_ids"].extend(int(t) for t in token_ids)
            store["phases"].extend(phase for _ in abs_positions)
        device = self._J.device
        for start in range(0, len(positions), chunk):
            batch_positions = positions[start : start + chunk]
            gathered = []
            for layer in self.layers:
                full = acts[layer][0]
                gathered.append(full[list(batch_positions)].float().to(device))
            h = torch.stack(gathered)
            if store is not None:
                for li, layer in enumerate(self.layers):
                    store["residuals"][layer].append(h[li].half().cpu())
            # L2 norm of the residual per layer/position ("Activations" view)
            h_norms = h.norm(dim=-1).tolist()
            transported = torch.einsum("lij,lpj->lpi", self._J, h)
            logits = jl.unembed(transported).float()
            lse = logits.logsumexp(-1, keepdim=True)
            raw_v, raw_ids = logits.topk(self.k)
            raw_p = (raw_v - lse).exp()
            m_v, m_ids = logits.masked_fill(~self.mask, float("-inf")).topk(self.k)
            m_p = (m_v - lse).exp()
            sel = logits.gather(-1, m_ids)
            # rank of each top-k token in the full distribution. We loop over k
            # rather than materializing a boolean [L, P, k, V] (≈760 MB at k=32 /
            # 32 layers → OOM): each iteration only touches [L, P, V].
            m_rank = torch.empty_like(m_ids)
            for ki in range(m_ids.shape[-1]):
                m_rank[..., ki] = (logits > sel[..., ki : ki + 1]).sum(-1)
            del logits
            raw_ids_l, raw_p_l = raw_ids.tolist(), raw_p.tolist()
            m_ids_l, m_p_l, m_rank_l = m_ids.tolist(), m_p.tolist(), m_rank.tolist()
            for li, layer in enumerate(self.layers):
                for pi in range(len(batch_positions)):
                    ids = raw_ids_l[li][pi]
                    mids = m_ids_l[li][pi]
                    frames[start + pi]["layers"][str(layer)] = {
                        "ids": ids,
                        "p": [round(v, 5) for v in raw_p_l[li][pi]],
                        "strs": self._strs(tokenizer, ids),
                        "m_ids": mids,
                        "m_p": [round(v, 5) for v in m_p_l[li][pi]],
                        "m_rank": m_rank_l[li][pi],
                        "m_strs": self._strs(tokenizer, mids),
                        "h_norm": round(h_norms[li][pi], 2),
                    }
        return frames
