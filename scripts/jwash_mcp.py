"""MCP server for J-Wash — let an external LLM autonomously test token-direction
edits on a model that is ALREADY loaded in the running J-Wash app (port 8381).

It exposes only what is needed to experiment, and nothing else:

  * generate                — (re)generate text from the current model
  * scale_token / replace_token — attempt a Readthrough token operation, any intensity
  * set_intensity           — global multiplier over all edits (sweep the intensity)
  * list_edits / reset_edits — inspect / clear the current edits

This is a thin HTTP client of the J-Wash REST API (same server as scripts/jlab.py),
spoken over MCP/stdio so any MCP client (Claude Desktop, another agent, ...) can
drive a model you loaded yourself. By design it never loads models or lenses,
never changes the sampling defaults beyond the call, and never exports anything:
a model AND a Jacobian lens must already be loaded from the J-Wash UI.

Token edits attempt Readthrough mode. Capability diagnostics are advisory: an
experimental architecture can reach the strict transform planner and fail with
a concrete implementation error. Preview/export equivalence is only established
for positively validated Readthrough implementations.

Run it from an MCP client over stdio:

    pip install mcp
    python -X utf8 scripts/jwash_mcp.py

Point it at a non-default J-Wash instance with an env var:

    JWASH_BASE=http://127.0.0.1:8382
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP

BASE = os.environ.get("JWASH_BASE", "http://127.0.0.1:8381").rstrip("/")

mcp = FastMCP(
    "j-wash",
    instructions=(
        "Drive a model ALREADY loaded in the running J-Wash app to test "
        "token-direction edits. Typical loop: (1) `generate` a baseline reply; "
        "(2) find the exact token with `find_token` and the layers to target with "
        "`list_layers`; (3) apply edits with `scale_token`/`replace_token` (layers "
        "are required) — Readthrough is attempted and may report a concrete "
        "planner error on an experimental architecture; "
        "(4) `generate` again to see the effect, tuning each edit's `factor` or the "
        "global `set_intensity`; (5) `reset_edits` to start over. A model AND a "
        "Jacobian lens must be loaded from the J-Wash UI first; this server never "
        "loads models, lenses, or exports checkpoints."
    ),
)


# --- HTTP plumbing (stdlib only, like scripts/jlab.py) ----------------------

def _call(method, path, body=None, timeout=600):
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("detail", detail)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ValueError(f"J-Wash {method} {path} -> HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise ValueError(
            f"J-Wash server unreachable at {BASE} ({exc.reason}). Start it "
            "(python -X utf8 run.py) and load a model + lens, then retry."
        )


def _status():
    return _call("GET", "/api/status")


def _resolve_token(text):
    """The EXACT single token for ``text`` (a leading space is significant)."""
    r = _call("GET", "/api/token-lookup?q=" + urllib.parse.quote(text.strip()))
    cands = r.get("candidates", [])
    for c in cands:
        if c["str"] == text:
            return c
    listing = ", ".join(f"{c['id']}:{c['str']!r}" for c in cands) or "none"
    raise ValueError(
        f"No exact single-token match for {text!r}. A leading space is "
        f"significant (mid-sentence words usually need one, e.g. ' model'). "
        f"Candidates: {listing}"
    )


def _parse_layers(spec, n_layers):
    """None -> server default band; 'all'/'none'/'19-31'/'3,5,7' -> explicit list."""
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return []
    if spec == "all":
        if not n_layers:
            raise ValueError("layers='all' needs a loaded model to know the layer count")
        return list(range(n_layers))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


_PURE_WEIGHTS_MODES = ("readthrough", "exact", "abliteration")


def _edits_summary():
    st = _status()
    mode = st.get("interventions_mode")
    return {
        "mode": mode,
        "pure_weights": mode in _PURE_WEIGHTS_MODES,
        "global_intensity": st.get("interventions_scale"),
        "edits": [
            {
                "id": r["id"],
                "token": r["token"],
                "op": r["mode"],
                "factor": r["factor"],
                "replacement": r.get("replacement"),
                "layers": r["layers"],
            }
            for r in (st.get("interventions") or [])
        ],
    }


def _add_rule(token, op, factor, replacement, layers):
    st = _status()
    loaded = st.get("loaded")
    if not loaded:
        raise ValueError(
            "No model loaded in J-Wash — load a model and a Jacobian lens from "
            "the app first."
        )
    if not st.get("lens"):
        raise ValueError(
            "No Jacobian lens loaded — load one in the Lens tab of J-Wash before "
            "editing tokens."
        )
    # Readthrough is the stable MCP default. Legacy capability aliases are
    # diagnostics only and never reroute an attempt to another implementation.
    _call("PATCH", "/api/interventions", {"mode": "readthrough"})

    body = {"token_id": _resolve_token(token)["id"], "mode": op, "factor": float(factor)}
    if op == "replace":
        body["replacement_id"] = _resolve_token(replacement)["id"]
    parsed = _parse_layers(layers, loaded.get("n_layers"))
    if parsed is not None:
        body["layers"] = parsed
    _call("POST", "/api/interventions", body)
    return _edits_summary()


# --- MCP tools --------------------------------------------------------------

@mcp.tool()
def generate(prompt: str, system: str | None = None, max_tokens: int = 200,
             temperature: float = 0.0, seed: int = 1234) -> str:
    """(Re)generate a reply from the model currently loaded in J-Wash, with the
    active token edits applied — call it again to regenerate.

    At temperature 0 generation is deterministic, so the reply changes only when
    the edits change: this is the clean way to compare behaviour before vs after
    an edit. Raise `temperature` (or set `seed=-1` for a random seed) to sample
    varied continuations instead. `system` is an optional system prompt.
    """
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    r = _call("POST", "/api/generate", {
        "messages": messages,
        "sampling": {"temperature": temperature, "max_tokens": max_tokens, "seed": seed},
    }, timeout=1800)
    return r.get("text", "")


@mcp.tool()
def scale_token(token: str, factor: float, layers: str) -> dict:
    """Multiply a token's own direction by `factor` using Readthrough.

    `factor` is the intensity: 0 removes the token's direction, 0<factor<1
    attenuates it, factor>1 amplifies it. `token` is the exact token string — a
    leading space is usually significant (e.g. ' model'); use `find_token` to get
    it. `layers` is REQUIRED: it selects where the edit acts and an edit that
    targets no layer does nothing — pass a 0-based range or list ('19-25', '20',
    '20,24', or 'all') and call `list_layers` to see the model's layers. The mode
    attempts Readthrough; experimental architectures may return a concrete
    planner error rather than producing a preview or export.
    """
    return _add_rule(token, "scale", factor, None, layers)


@mcp.tool()
def replace_token(token: str, replacement: str, layers: str, factor: float = 1.0) -> dict:
    """Attempt to rewrite `token` onto `replacement` using Readthrough,
    e.g. token=' model', replacement=' fish' to make the model talk like a fish.

    You MUST pass `layers` — it selects the layers where the replacement is
    applied, and WITHOUT it nothing happens. Give a 0-based range or list
    ('19-25', '20,24', or 'all'); call `list_layers` for the model's layers and
    `find_token` for the exact ' token' strings (both must be single tokens, a
    leading space usually being significant). `factor` scales the strength
    (1.0 = full). Readthrough is attempted without consulting advisory
    capability metadata; validated implementations establish export fidelity.
    """
    return _add_rule(token, "replace", factor, replacement, layers)


@mcp.tool()
def set_intensity(scale: float) -> dict:
    """Set the global multiplier applied to ALL active edits — sweep the overall
    intensity without touching each rule: 0 disables every edit, 1 is nominal,
    >1 pushes them harder. Returns the current edits.
    """
    _call("PATCH", "/api/interventions", {"scale": scale})
    return _edits_summary()


@mcp.tool()
def list_edits() -> dict:
    """Show the active token edits, selected attempted mode, and the global
    intensity — a read-only snapshot of the current experiment.
    """
    return _edits_summary()


@mcp.tool()
def reset_edits() -> dict:
    """Remove every token edit, returning the model to its unedited behaviour.
    Use it to start a fresh experiment. Returns the (now empty) edits.
    """
    _call("DELETE", "/api/interventions")
    return _edits_summary()


@mcp.tool()
def find_token(text: str) -> dict:
    """Look up the single-token forms of `text` so you can pick the exact token to
    edit before calling scale_token/replace_token.

    A leading space is significant (' model' and 'model' are different tokens), so
    the lookup also tries the space-prefixed and capitalization variants and
    returns those that are exactly one token. Use a returned `token` string
    verbatim; words that split into several tokens can't be edited directly.
    """
    r = _call("GET", "/api/token-lookup?q=" + urllib.parse.quote(text.strip()))
    return {
        "query": text,
        "candidates": [{"id": c["id"], "token": c["str"]} for c in r.get("candidates", [])],
    }


@mcp.tool()
def list_layers() -> dict:
    """List the layers you can target with `scale_token`/`replace_token`.

    Returns the model's total layer count (indices are 0-based, so the valid
    range is 0..n_layers-1) and the layers the loaded Jacobian lens actually
    covers — those are the calibrated ones to edit; targeting a layer outside
    them falls back to a less reliable logit-lens direction.
    """
    st = _status()
    loaded = st.get("loaded")
    if not loaded:
        raise ValueError(
            "No model loaded in J-Wash — load a model and a Jacobian lens from "
            "the app first."
        )
    n = loaded.get("n_layers")
    lens = st.get("lens") or {}
    out = {"n_layers": n, "valid_range": f"0-{n - 1}" if n else None}
    if lens.get("fitted_layers_all"):
        out["lens_fitted_layers"] = lens["fitted_layers_all"]
    if lens.get("tapped_layers"):
        out["lens_tapped_layers"] = lens["tapped_layers"]
    return out


if __name__ == "__main__":
    mcp.run(transport="stdio")
