import errno
import json
import os
import re
import shutil
import stat
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import config
from core import rebase
from core.ablation import abliteration_direction, effective_coeffs

EDITS_DIR = config.DATA_DIR / "edits"
PRESETS_DIR = config.DATA_DIR / "presets"

DEFAULT_TEMP_STALE_AGE = 24 * 60 * 60
_TEMP_STAGE_RE = re.compile(r"^\..+\.tmp-[0-9a-f]{32}$")
_TEMP_GGUF_RE = re.compile(r"^\..+\.tmp-[0-9a-f]{32}\.gguf$")
_RMTREE_DIR_FD_SAFE = bool(getattr(shutil.rmtree, "avoids_symlink_attacks", False))


def internal_temp_leaf_kind(name):
    """Classify exact J-Wash temporary/lease leaf names."""
    artifact_name = name[:-6] if name.endswith(".lease") else name
    if _TEMP_STAGE_RE.fullmatch(artifact_name):
        return "directory-lease" if name.endswith(".lease") else "directory"
    if _TEMP_GGUF_RE.fullmatch(artifact_name):
        return "file-lease" if name.endswith(".lease") else "file"
    return None


class ArtifactLease:
    """Exclusive advisory lease for one temporary export artifact."""

    def __init__(self, artifact, *, create=True):
        self.artifact = Path(artifact)
        self.path = self.artifact.with_name(self.artifact.name + ".lease")
        self.create = create
        self._file = None

    def acquire(self, *, blocking=True):
        created = False
        if self.create:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                created = True
            except FileExistsError:
                fd = os.open(self.path, os.O_RDWR)
        else:
            fd = os.open(self.path, os.O_RDWR)
        file = os.fdopen(fd, "r+b")

        def close_unacquired():
            file.close()
            if created:
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass

        try:
            file.seek(0, os.SEEK_END)
            if file.tell() == 0:
                file.write(b"0")
                file.flush()
            file.seek(0)
            if os.name == "nt":
                import msvcrt
                flag = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                try:
                    msvcrt.locking(file.fileno(), flag, 1)
                except OSError as exc:
                    if not blocking and exc.errno in (
                        errno.EACCES, errno.EAGAIN, errno.EDEADLK,
                    ):
                        close_unacquired()
                        return False
                    raise
            else:
                import fcntl
                flag = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                try:
                    fcntl.flock(file.fileno(), flag)
                except BlockingIOError:
                    close_unacquired()
                    return False
            self._file = file
            return True
        except Exception:
            if not file.closed:
                close_unacquired()
            raise

    def release(self, *, remove=True):
        file, self._file = self._file, None
        if file is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            finally:
                file.close()
        if remove:
            self.path.unlink(missing_ok=True)

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"temporary artifact lease is already held: {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.release()
        except Exception:
            if exc is None:
                raise
        return False


def artifact_lease(artifact):
    return ArtifactLease(artifact)


def _is_link_or_reparse(path, path_stat=None):
    path_stat = path.lstat() if path_stat is None else path_stat
    return (
        stat.S_ISLNK(path_stat.st_mode)
        or getattr(path, "is_junction", lambda: False)()
        or bool(
            getattr(path_stat, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        )
    )


def _recognized_temp(path):
    """Return the owned temporary kind without following a symlink."""
    try:
        if _is_link_or_reparse(path):
            return None
        if internal_temp_leaf_kind(path.name) == "directory" and path.is_dir():
            return "directory"
        if internal_temp_leaf_kind(path.name) == "file" and path.is_file():
            return "file"
    except OSError:
        raise
    return None


def _stat_identity(value):
    return (
        value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode),
        getattr(value, "st_file_attributes", 0),
        getattr(value, "st_reparse_tag", 0),
    )


def _candidate_identity(value):
    return _stat_identity(value) + (
        getattr(value, "st_ctime_ns", None), getattr(value, "st_mtime_ns", None),
        value.st_size,
    )


def _absolute_no_follow(path):
    """Return an absolute normalized spelling without resolving any links."""
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))


def _path_chain(path):
    anchor = Path(path.anchor)
    current = anchor
    chain = [anchor]
    for part in path.parts[1:]:
        current = current / part
        chain.append(current)
    return chain


def _snapshot_directory_chain(path, *, missing_ok=False):
    snapshots = []
    for component in _path_chain(path):
        try:
            component_stat = component.lstat()
        except FileNotFoundError:
            if missing_ok:
                return None
            raise
        if _is_link_or_reparse(component, component_stat):
            raise ValueError(
                f"cleanup root chain contains a symlink or junction/reparse point: {component}"
            )
        if not stat.S_ISDIR(component_stat.st_mode):
            raise ValueError(f"cleanup path component is not a directory: {component}")
        snapshots.append((component, _stat_identity(component_stat)))
    return tuple(snapshots)


def _revalidate_directory_chain(snapshots):
    for component, expected in snapshots:
        current = component.lstat()
        if _is_link_or_reparse(component, current) or _stat_identity(current) != expected:
            raise ValueError(f"cleanup path component changed: {component}")


@dataclass(frozen=True)
class _TempCandidate:
    path: Path
    kind: str
    identity: tuple
    parent_chain: tuple


class _UnsafeAnchoredCleanup(RuntimeError):
    """Raised when the platform cannot safely anchor a destructive cleanup."""


class _AnchoredCleanupParent:
    """Descriptor-relative owner of one candidate's final POSIX operations.

    The descriptor chain is opened without following links and kept alive from
    the authoritative candidate check through claim verification and deletion.
    Windows deliberately fails closed until equivalent handle-relative recursive
    deletion is available.
    """

    def __init__(self, candidate):
        self.candidate = candidate
        self.parent_fd = None
        self._fds = []

    def __enter__(self):
        if os.name == "nt":
            raise _UnsafeAnchoredCleanup(
                "safe handle-relative temporary cleanup is unavailable on Windows"
            )
        flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        snapshots = self.candidate.parent_chain
        try:
            if not snapshots:
                raise _UnsafeAnchoredCleanup("candidate has no validated parent chain")
            anchor, expected = snapshots[0]
            fd = os.open(anchor, flags)
            self._fds.append(fd)
            self._verify_fd(fd, expected, anchor)
            for component, expected in snapshots[1:]:
                next_fd = os.open(component.name, flags, dir_fd=fd)
                self._fds.append(next_fd)
                self._verify_fd(next_fd, expected, component)
                fd = next_fd
            self.parent_fd = fd
            self.stat_leaf(self.candidate.path.name, self.candidate.identity)
            return self
        except Exception as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, _UnsafeAnchoredCleanup):
                raise
            raise _UnsafeAnchoredCleanup(
                f"cannot safely anchor cleanup parent: {exc}"
            ) from exc

    @staticmethod
    def _verify_fd(fd, expected, display):
        current = os.fstat(fd)
        if _stat_identity(current) != expected or not stat.S_ISDIR(current.st_mode):
            raise _UnsafeAnchoredCleanup(f"cleanup parent changed: {display}")

    def stat_leaf(self, leaf, expected_identity=None, *, full=False):
        current = os.stat(leaf, dir_fd=self.parent_fd, follow_symlinks=False)
        if _is_link_or_reparse(Path(leaf), current):
            raise _UnsafeAnchoredCleanup("temporary artifact became a link/reparse point")
        expected_type = stat.S_ISDIR if self.candidate.kind == "directory" else stat.S_ISREG
        if not expected_type(current.st_mode):
            raise _UnsafeAnchoredCleanup("temporary artifact type changed")
        if expected_identity is not None:
            actual = _candidate_identity(current) if full else _stat_identity(current)
            expected = expected_identity if full else expected_identity[:5]
            if actual != expected:
                raise _UnsafeAnchoredCleanup("temporary artifact identity changed")
        return current

    def claim(self, claim_leaf):
        os.replace(
            self.candidate.path.name, claim_leaf,
            src_dir_fd=self.parent_fd, dst_dir_fd=self.parent_fd,
        )
        current = self.stat_leaf(claim_leaf, self.candidate.identity)
        return _candidate_identity(current)

    def restore(self, claim_leaf, claim_identity):
        self.stat_leaf(claim_leaf, claim_identity, full=True)
        try:
            os.stat(self.candidate.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            os.replace(
                claim_leaf, self.candidate.path.name,
                src_dir_fd=self.parent_fd, dst_dir_fd=self.parent_fd,
            )

    def delete(self, claim_leaf, claim_identity):
        self.stat_leaf(claim_leaf, claim_identity, full=True)
        if self.candidate.kind == "file":
            os.unlink(claim_leaf, dir_fd=self.parent_fd)
            return
        if not _RMTREE_DIR_FD_SAFE:
            raise _UnsafeAnchoredCleanup(
                "descriptor-relative symlink-resistant directory deletion is unavailable"
            )
        shutil.rmtree(claim_leaf, dir_fd=self.parent_fd)

    def __exit__(self, exc_type, exc, tb):
        for fd in reversed(self._fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()
        self.parent_fd = None
        return False


def _revalidate_candidate(candidate):
    _revalidate_directory_chain(candidate.parent_chain)
    current = candidate.path.lstat()
    if _is_link_or_reparse(candidate.path, current):
        raise ValueError("temporary artifact changed or became a link/reparse point")
    if _candidate_identity(current) != candidate.identity:
        raise ValueError("temporary artifact changed after discovery")
    expected = stat.S_ISDIR if candidate.kind == "directory" else stat.S_ISREG
    if not expected(current.st_mode):
        raise ValueError("temporary artifact type changed after discovery")
    return current


def _read_completion_marker(marker):
    marker_stat = marker.lstat()
    if _is_link_or_reparse(marker, marker_stat) or not stat.S_ISREG(marker_stat.st_mode):
        raise ValueError("possible completion marker is not a safe regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(marker, flags)
    try:
        opened = os.fstat(fd)
        if _stat_identity(opened) != _stat_identity(marker_stat):
            raise ValueError("possible completion marker changed during inspection")
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            fd = None
            return json.load(stream)
    finally:
        if fd is not None:
            os.close(fd)


def _raise_walk_error(exc):
    raise exc


def _newest_lstat_mtime(path, kind):
    newest = path.lstat().st_mtime
    if kind == "file":
        return newest
    for current, dirs, files in os.walk(
        path, followlinks=False, onerror=_raise_walk_error,
    ):
        base = Path(current)
        # Never traverse directory symlinks, but include their own lstat time.
        kept = []
        for name in dirs:
            child = base / name
            newest = max(newest, child.lstat().st_mtime)
            if not _is_link_or_reparse(child):
                kept.append(name)
        dirs[:] = kept
        for name in files:
            newest = max(newest, (base / name).lstat().st_mtime)
    return newest


def cleanup_abandoned_export_temps(
    *, root=None, now=None, stale_age=DEFAULT_TEMP_STALE_AGE, observer=None
):
    """Remove only abandoned J-Wash transaction artifacts beneath ``root``.

    A held lease always wins.  An unlocked lease proves abandonment and permits
    immediate removal.  Legacy artifacts without a lease must be older than the
    configured threshold (24 hours by default).
    """
    root = _absolute_no_follow(root if root is not None else EDITS_DIR)
    current_time = time.time() if now is None else float(now)
    result = {
        "removed": [], "removed_count": 0, "skipped_active": [],
        "skipped_recent": [], "skipped_completed": [], "skipped_changed": [],
        "errors": [],
    }
    try:
        root_chain = _snapshot_directory_chain(root, missing_ok=True)
        if root_chain is None:
            return result
    except FileNotFoundError:
        return result
    except (OSError, RuntimeError, ValueError) as exc:
        result["errors"].append({"path": str(root), "error": str(exc)})
        return result

    candidates = []

    def record_walk_error(exc):
        result["errors"].append({
            "path": str(getattr(exc, "filename", root)), "error": str(exc),
        })

    for current, dirs, files in os.walk(
        root, topdown=True, followlinks=False, onerror=record_walk_error,
    ):
        base = Path(current)
        # Do not descend through any symlink, including one beneath EDITS_DIR.
        kept_dirs = []
        for name in dirs:
            child = base / name
            try:
                if not _is_link_or_reparse(child):
                    kept_dirs.append(name)
            except OSError as exc:
                result["errors"].append({"path": str(child), "error": str(exc)})
        dirs[:] = kept_dirs
        for name in list(dirs) + files:
            path = base / name
            try:
                kind = _recognized_temp(path)
                if kind:
                    path_stat = path.lstat()
                    parent_chain = _snapshot_directory_chain(path.parent)
                    candidates.append(_TempCandidate(
                        path, kind, _candidate_identity(path_stat), parent_chain,
                    ))
                    if kind == "directory" and name in dirs:
                        dirs.remove(name)
            except OSError as exc:
                result["errors"].append({"path": str(path), "error": str(exc)})

    if observer:
        observer("discovered", tuple(candidate.path for candidate in candidates))

    for candidate in candidates:
        path, kind = candidate.path, candidate.kind
        lease = ArtifactLease(path, create=False)
        acquired = False
        claimed = None
        anchored = None
        try:
            _revalidate_directory_chain(root_chain)
            path.relative_to(root)
            _revalidate_candidate(candidate)
            if observer:
                observer("before_inspect", path)
            _revalidate_candidate(candidate)
            try:
                lease_stat = lease.path.lstat()
            except FileNotFoundError:
                lease_stat = None
            if lease_stat is not None and (
                _is_link_or_reparse(lease.path, lease_stat)
                or not stat.S_ISREG(lease_stat.st_mode)
            ):
                raise ValueError("temporary artifact lease is not a safe regular file")
            has_lease = lease_stat is not None
            if has_lease:
                acquired = lease.acquire(blocking=False)
                if not acquired:
                    result["skipped_active"].append(str(path))
                    continue
            else:
                if kind == "directory":
                    marker = path / "edit_meta.json"
                    try:
                        marker.lstat()
                    except FileNotFoundError:
                        marker_present = False
                    except OSError as exc:
                        raise OSError(
                            f"cannot inspect possible completion marker: {exc}"
                        ) from exc
                    else:
                        marker_present = True
                    if marker_present:
                        try:
                            metadata = _read_completion_marker(marker)
                        except Exception as exc:
                            raise ValueError(f"cannot inspect possible completion marker: {exc}") from exc
                        relative_name = path.relative_to(root).as_posix()
                        if (
                            not isinstance(metadata, dict)
                            or not isinstance(metadata.get("name"), str)
                        ):
                            raise ValueError("possible completion marker has invalid metadata")
                        if metadata["name"] == relative_name:
                            result["skipped_completed"].append(str(path))
                            continue
                newest = _newest_lstat_mtime(path, kind)
                if current_time - newest < stale_age:
                    result["skipped_recent"].append(str(path))
                    continue
            _revalidate_candidate(candidate)
            if observer:
                observer("before_claim", path)
            _revalidate_candidate(candidate)
            claim_uuid = uuid.uuid4().hex
            if kind == "directory":
                claimed = path.with_name(f".{path.name.lstrip('.')}.tmp-{claim_uuid}")
            else:
                stem = path.name.split(".tmp-", 1)[0].lstrip(".")
                claimed = path.with_name(f".{stem}.tmp-{claim_uuid}.gguf")
            if observer:
                observer("before_anchored_claim", path)
            with _AnchoredCleanupParent(candidate) as anchored:
                claim_identity = anchored.claim(claimed.name)
                if observer:
                    observer("before_delete", (path, claimed))
                try:
                    anchored.delete(claimed.name, claim_identity)
                except Exception:
                    try:
                        anchored.restore(claimed.name, claim_identity)
                        claimed = None
                    except Exception:
                        # A changed claim is intentionally left untouched.
                        pass
                    raise
            claimed = None
            anchored = None
            result["removed"].append(str(path))
            result["removed_count"] += 1
        except Exception as exc:
            if (
                isinstance(exc, _UnsafeAnchoredCleanup)
                or "changed" in str(exc)
                or "cleanup root chain" in str(exc)
            ):
                result["skipped_changed"].append(str(path))
            result["errors"].append({"path": str(path), "error": str(exc)})
        finally:
            if acquired:
                try:
                    lease.release(remove=True)
                except Exception as exc:
                    result["errors"].append({"path": str(lease.path), "error": str(exc)})
    return result


# Residual writes edited by the global abliteration (embed aside)
TARGET_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_presets():
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for path in sorted(PRESETS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out.append({"name": path.stem, "n_rules": len(data.get("rules", [])), "model_id": data.get("model_id")})
    return out


def save_preset(name, rules, model_id, scale=1.0):
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"model_id": model_id, "saved_at": _now(), "scale": scale, "rules": rules}
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
        alpha, beta = effective_coeffs(r["mode"], r["factor"], scale)
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
    parts = validate_export_name(name)
    name = "/".join(parts)
    out_dir = EDITS_DIR.joinpath(*parts)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if model_meta.get("dtype") == "bf16" else torch.float16
    lm_head_key = info["lm_head_key"]
    summary = [
        {k: r[k] for k in ("token_id", "token", "mode", "factor", "replacement_id", "replacement")}
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


_INDEX_RESERVED = {
    "config.json", "model.safetensors.index.json", "edit_meta.json",
    "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "generation_config.json", "special_tokens_map.json",
    "chat_template.json",
}


def _indexed_shards(source_dir):
    """Return a fully schema-validated index and its logical shard paths."""
    path = source_dir / "model.safetensors.index.json"
    if not path.exists():
        return None, None
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"full checkpoint: invalid safetensors index: {exc}") from exc
    if not isinstance(index, dict):
        raise ValueError("full checkpoint: safetensors index root must be an object")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("full checkpoint: index metadata must be an object")
    total_size = metadata.get("total_size")
    if isinstance(total_size, bool) or not isinstance(total_size, int) or total_size < 0:
        raise ValueError("full checkpoint: metadata.total_size must be a nonnegative integer")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("full checkpoint: index weight_map must be a nonempty object")
    filenames = []
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not key:
            raise ValueError("full checkpoint: every weight_map key must be a nonempty string")
        if not isinstance(filename, str) or not filename:
            raise ValueError("full checkpoint: every shard filename must be a nonempty string")
        try:
            parts = _safe_relative_parts(filename)
        except ValueError as exc:
            raise ValueError(f"full checkpoint: unsafe shard filename {filename!r}: {exc}") from exc
        if len(parts) != 1 or Path(filename).name != filename or not filename.lower().endswith(".safetensors"):
            raise ValueError(f"full checkpoint: shard filename must be a plain .safetensors leaf: {filename!r}")
        filenames.append(filename)
    folded = [name.casefold() for name in dict.fromkeys(filenames)]
    if len(folded) != len(set(folded)):
        raise ValueError("full checkpoint: shard filenames have a case-fold collision")
    auxiliary = {p.name.casefold() for pattern in ("*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja")
                 for p in source_dir.glob(pattern)}
    forbidden = {name.casefold() for name in _INDEX_RESERVED} | auxiliary
    collisions = sorted({name for name in dict.fromkeys(filenames) if name.casefold() in forbidden})
    if collisions:
        raise ValueError(f"full checkpoint: shard filename collides with reserved/auxiliary file: {collisions}")
    shards = [source_dir / name for name in dict.fromkeys(filenames)]
    missing = [shard.name for shard in shards if not shard.is_file()]
    if missing:
        raise ValueError(f"full checkpoint: indexed shard(s) missing: {missing}")
    return index, shards


def _validate_index_contents(index, shard_paths):
    """Validate exact key placement, coverage, and logical total size."""
    key_sets = {}
    tensor_sizes = {}
    dtype_bytes = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                   "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                   "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for shard in shard_paths:
        try:
            with safe_open(str(shard), framework="pt") as f:
                keys = set(f.keys()); key_sets[shard.name] = keys
                tensor_sizes[shard.name] = {}
                for key in keys:
                    tensor_slice = f.get_slice(key)
                    size = dtype_bytes.get(tensor_slice.get_dtype())
                    if size is None:
                        raise ValueError(f"unsupported safetensors dtype {tensor_slice.get_dtype()}")
                    elements = 1
                    for dimension in tensor_slice.get_shape(): elements *= dimension
                    tensor_sizes[shard.name][key] = elements * size
        except Exception as exc:
            raise ValueError(f"invalid safetensors shard {shard.name}: {exc}") from exc
    weight_map = index["weight_map"]
    for key, filename in weight_map.items():
        if key not in key_sets.get(filename, set()):
            raise ValueError(f"index maps {key!r} to {filename!r}, but that shard does not contain it")
    staged_keys = set().union(*key_sets.values()) if key_sets else set()
    unindexed = staged_keys - set(weight_map)
    if unindexed:
        raise ValueError(f"indexed checkpoint contains unindexed tensor(s): {sorted(unindexed)[:3]}")
    logical_total = sum(tensor_sizes[filename][key] for key, filename in weight_map.items())
    if index["metadata"]["total_size"] != logical_total:
        raise ValueError(
            f"index metadata.total_size={index['metadata']['total_size']} does not match {logical_total}"
        )
    return key_sets, logical_total


REBASE_EXPORT_ROW_BUDGET = 4096
REBASE_EXPORT_CHUNK_OBSERVER = None

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                     *(f"LPT{i}" for i in range(1, 10)),
                     "COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"}


def validate_export_name(name):
    """Lexically validate a portable relative export name before resolution."""
    if not isinstance(name, str) or not name:
        raise ValueError("export name must be a nonempty relative path")
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        raise ValueError("export name must not be absolute, drive-qualified, or UNC")
    if "\\" in name:
        raise ValueError("export name must use portable '/' separators")
    raw = name.split("/")
    if any(part in ("", ".", "..") for part in raw):
        raise ValueError("export name contains an empty, '.', or '..' component")
    for part in raw:
        if internal_temp_leaf_kind(part) is not None:
            raise ValueError(
                f"export name component {part!r} is reserved for internal temporary export use"
            )
        if part.endswith((" ", ".")) or part.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
            raise ValueError(f"export name contains Windows-reserved component {part!r}")
        if any(ord(char) < 32 or char in '<>:"|?*' for char in part):
            raise ValueError(f"export name contains nonportable component {part!r}")
    return tuple(raw)


def _safe_relative_parts(name):
    """Backward-compatible private alias for the public validator."""
    return validate_export_name(name)


def apply_transform_bounded(entry, tensor, *, row_budget=REBASE_EXPORT_ROW_BUDGET,
                            observer=None):
    """Apply a read transform with bounded float32 row temporaries.

    The destination is allocated directly in the source dtype.  Write
    transforms are ordinary dense matrices and retain the established path.
    """
    kind, X, Y = entry
    if kind != "read":
        source = tensor.detach().to("cpu", torch.float32)
        updated, _B, _A = rebase.apply_transform(entry, source)
        return updated.to(tensor.dtype), float((updated - source).abs().max())
    if row_budget < 1:
        raise ValueError("row_budget must be positive")
    source = tensor.detach().cpu().contiguous()
    rows = source.reshape(-1, source.shape[-1])
    destination = torch.empty_like(source)
    dest_rows = destination.reshape_as(rows)
    delta_max = 0.0
    for start in range(0, rows.shape[0], row_budget):
        count = min(row_budget, rows.shape[0] - start)
        if observer is not None:
            observer(count)
        chunk = rows[start:start + count].float()
        B = chunk @ X
        updated = chunk + B @ Y.T
        delta_max = max(delta_max, float((updated - chunk).abs().max()))
        dest_rows[start:start + count].copy_(updated.to(source.dtype))
    return destination, delta_max


def export_rebase(rules, jl, model_meta, *, fmt, name, source_dir=None, scale=1.0, exact=False):
    """Transactionally construct and atomically publish a rebase export.

    Overwrite policy is deliberately conservative: an existing destination is
    rejected and never modified.  All construction occurs in a unique sibling
    staging directory which is removed on every failure.
    """
    parts = validate_export_name(name)
    root = EDITS_DIR.resolve()
    final_dir = root.joinpath(*parts).resolve()
    try:
        relative = final_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("export name must remain beneath the edits directory") from exc
    if not relative.parts:
        raise ValueError("export name must identify a directory beneath the edits directory")
    if final_dir.exists():
        raise ValueError(f"export destination already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = final_dir.parent / f".{final_dir.name}.tmp-{uuid.uuid4().hex}"
    with artifact_lease(stage):
        stage.mkdir()
        try:
            result = _export_rebase_impl(
                rules, jl, model_meta, fmt=fmt, name=name, source_dir=source_dir,
                scale=scale, exact=exact, out_dir=stage,
            )
            stage.replace(final_dir)
            result["out_dir"] = str(final_dir)
            return result
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def _export_rebase_impl(rules, jl, model_meta, *, fmt, name, source_dir=None,
                        scale=1.0, exact=False, out_dir):
    """Pure-weight export by change of basis of the reads (cf. core/rebase).

    ``readthrough`` (exact=False): the downstream read matrices + lm_head.
    ``exact``: adds the counter-transform of the downstream writes.
    Formats: ``full`` (checkpoint), ``layers`` (safetensors of the modified
    matrices) and ``lora`` (PEFT adapter = the exact low-rank diff between the
    baked weights and the originals; the lm_head delta is applied at forward
    time, so tied embeddings need no untying). The bake is done streaming, one
    float32 CPU matrix at a time. Tied-embeddings model (full/layers): the embed
    stays INTACT, it's lm_head (untied) that receives the final read transform."""
    method = "rebase-exact" if exact else "rebase-readthrough"
    if fmt not in ("full", "layers", "lora"):
        raise ValueError(f"unknown format for {method}: {fmt}")
    transforms, info = rebase.build_plan(rules, jl, scale, exact=exact)
    lm_head_key = info["lm_head_key"]

    packed = [key for key, target in info["targets"].items()
              if not target.lora_supported or target.tensor(jl.layers[int(key.split(".layers.", 1)[1].split(".", 1)[0])]).ndim > 2]
    if packed and fmt == "lora":
        raise ValueError(
            "LoRA export is unavailable for packed MoE parameters. "
            "Use full-checkpoint export."
        )
    if packed and fmt == "layers":
        raise ValueError(
            "modified-layers export is unavailable for packed MoE parameters: "
            "bounded-memory sharding is not implemented. Use full-checkpoint export."
        )

    delta_max = 0.0
    applied = set()

    def bake(key, tensor):
        nonlocal delta_max
        W_new, chunk_delta = apply_transform_bounded(
            transforms[key], tensor, row_budget=REBASE_EXPORT_ROW_BUDGET,
            observer=REBASE_EXPORT_CHUNK_OBSERVER,
        )
        delta_max = max(delta_max, chunk_delta)
        applied.add(key)
        return W_new

    dtype = torch.bfloat16 if model_meta.get("dtype") == "bf16" else torch.float16
    warnings = []
    indexed_artifact = None
    if exact and info["regularized_layers"]:
        warnings.append(
            "regularized inverse (full zap ⇒ singular transform) on layers "
            f"{info['regularized_layers']} — the effect there equals readthrough; "
            "prefer readthrough mode for full removals"
        )
    if fmt == "lora" and info["tied"]:
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
        source_index, indexed_paths = _indexed_shards(source_dir)
        if source_index is not None:
            shards = indexed_paths
            _validate_index_contents(source_index, shards)
        else:
            shards = sorted(source_dir.glob("*.safetensors"))
        if not shards:
            raise ValueError("full checkpoint: no safetensors in the source")

        disk_keys = set()
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                disk_keys.update(f.keys())
        to_disk = _disk_mapper(info["embed_key"], disk_keys)
        transforms = {to_disk(k): fn for k, fn in transforms.items()}
        lm_head_key = to_disk(lm_head_key)
        embed_key = to_disk(info["embed_key"])
        required = set(transforms)
        if info["tied"] and lm_head_key not in disk_keys and embed_key in disk_keys:
            required.remove(lm_head_key)
        missing_source = required - disk_keys
        if missing_source:
            sample = sorted(missing_source)[:3]
            raise ValueError(
                f"{len(missing_source)} parameter(s) to transform absent from the source "
                f"checkpoint (e.g. {sample}) — export cancelled before writing"
            )
        if any(key.startswith("mtp.") for key in disk_keys):
            warnings.append(
                "MTP weights were preserved but not transformed. Ordinary Transformers "
                "generation does not use them, but MTP/speculative decoding fidelity is not "
                "guaranteed for this edited checkpoint."
            )

        lm_head_written = False
        embed_shard_name = None
        for shard in shards:
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
        # from the original embed (which stays intact)
        if info["tied"] and not lm_head_written:
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
            # A streamed full export may already have written earlier shards.
            # Never leave a checkpoint that looks usable but is only partially
            # edited when the source inventory fails the completeness guard.
            raise ValueError(
                f"{len(missing)} parameter(s) to transform absent from the source "
                f"checkpoint (e.g. {sample}) — unexpected key names, export cancelled "
                "(the written checkpoint would be partially original)"
            )

        cfg_path = source_dir / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if info["tied"]:
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
        if info["tied"] and not lm_head_written and index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            wm = index.setdefault("weight_map", {})
            wm[lm_head_key] = embed_shard_name
            if "metadata" in index and "total_size" in index["metadata"]:
                index["metadata"]["total_size"] += (
                    lm_head_value.numel() * lm_head_value.element_size()
                )
            index_path.write_text(json.dumps(index, indent=1), encoding="utf-8")

        if source_index is not None:
            indexed_artifact = (index_path, [out_dir / shard.name for shard in shards], set(transforms))

    if delta_max < 1e-8:
        raise ValueError(
            "the bake changes no weight (null directions?) — the export would be "
            "identical to the original model, folder deleted"
        )

    summary = [
        {k: r[k] for k in ("token_id", "token", "mode", "factor", "replacement_id", "replacement", "layers")}
        for r in rules if r["layers"]
    ]
    meta = {
        "name": name,
        "format": fmt,
        "method": method,
        "model_id": model_meta.get("model_id"),
        "model_revision": model_meta.get("revision"),
        "dtype": model_meta.get("dtype"),
        "global_scale": scale,
        # lora: no physical untying — the lm_head delta lives in the adapter
        "untied_lm_head": info["tied"] and fmt != "lora",
        "rules": summary,
        "layers_span": info["layers_span"],
        "rank": info["rank_final"],
        "modified_params_count": len(transforms),
        "delta_max": delta_max,
        "min_gamma": info["min_gamma"],
        "warnings": warnings,
        "note": (
            "change of basis of the reads: every matrix that READS the residual "
            "downstream of the hooked layers (q/k/v, in_proj*, gate/up + lm_head) sees "
            "the residual transformed by the same J-space directions as the live preview"
            + (" ; downstream writes counter-transformed (exact mode)" if exact else "")
            + ". Pure weights: a standard safetensors checkpoint."
        ),
        "created_at": _now(),
    }
    (out_dir / "edit_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if indexed_artifact is not None:
        index_path, staged_shards, transformed_keys = indexed_artifact
        staged_index, _ = _indexed_shards(out_dir)
        key_sets, _ = _validate_index_contents(staged_index, staged_shards)
        for key in transformed_keys:
            filename = staged_index["weight_map"].get(key)
            if filename is None or key not in key_sets.get(filename, set()):
                raise ValueError(f"transformed key {key} is not validly placed by the final index")
    return {"out_dir": str(out_dir), **meta}
