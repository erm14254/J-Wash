import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

HOST = "127.0.0.1"
PORT = 8381

# HF cache root. Default: the shared Hugging Face cache (a pre-set HF_HOME, else
# the standard ~/.cache/huggingface — the same one other HF tools use). Pass
# `run.py --hf-cache PATH` (e.g. ./hf_cache) for an isolated, project-local cache;
# run.py sets HF_HOME from that argument before this module is imported.
_DEFAULT_HF_HOME = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "huggingface"
HF_CACHE = Path(os.environ.get("HF_HOME") or _DEFAULT_HF_HOME)
# Runtime data root (SQLite history, frames, presets, edits, masks). Overridable
# so several instances can run side by side; run.py sets it from --data-dir.
DATA_DIR = Path(os.environ.get("JWASH_DATA_DIR") or (ROOT / "data"))
LENSES_DIR = ROOT / "lenses"
UI_DIST = ROOT / "ui" / "dist"
LOCAL_MODELS_ROOT = ROOT

DTYPES = ("bf16", "fp16")
QUANTS = ("int8", "nf4")
DEFAULT_DTYPE = "bf16"
DEFAULT_DEVICE = "cuda:0"

DEFAULT_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    # min_p and repetition_penalty default to llama.cpp / LM Studio parity:
    # edited models have flattened logit distributions, and without these two
    # protections the SAME weights loop in J-Wash while running fine elsewhere.
    "min_p": 0.05,  # 0 = off; filters tokens below min_p × top probability
    "max_tokens": 512,
    "seed": -1,  # -1 = random
    "repetition_penalty": 1.1,  # 1.0 = off
    # thinking-capable chat templates (Qwen3.5 style): False renders the
    # non-thinking turn; True lets the model reason in <think> first — models
    # finetuned FOR thinking degrade when it is forced off.
    "thinking": False,
}


def setup_env():
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _tolerant_subprocess_text()


def _tolerant_subprocess_text():
    """Windows: libraries (quantization backends, driver probes) spawn tools
    whose console output is localized (cp850/cp1252). Under ``-X utf8`` the
    stdlib decodes their pipes as STRICT UTF-8, and the reader thread dies
    with a noisy — though harmless — UnicodeDecodeError. Default text-mode
    pipes to ``errors="replace"`` when the caller didn't choose otherwise."""
    if os.name != "nt":
        return
    import subprocess

    if getattr(subprocess.Popen.__init__, "_jwash_tolerant", False):
        return
    orig = subprocess.Popen.__init__

    def patched(self, *args, **kwargs):
        wants_text = (
            kwargs.get("text")
            or kwargs.get("universal_newlines")
            or kwargs.get("encoding")
        )
        if wants_text and not kwargs.get("errors"):
            kwargs["errors"] = "replace"
        orig(self, *args, **kwargs)

    patched._jwash_tolerant = True
    subprocess.Popen.__init__ = patched
