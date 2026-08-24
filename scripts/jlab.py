# CLI client for the J-Wash server (port 8381): drive the model, lens,
# intervention rules, generation and export without going through the UI.
#
#   python -X utf8 scripts/jlab.py status
#   ... load Qwen/Qwen3.5-4B --device cuda:0
#   ... lens --repo neuronpedia/jacobian-lens --file <file> --layers all
#   ... rule-add " assistant" --mode replace --repl " fish" --layers 19-31
#   ... mode readthrough ; ... scale 1.5
#   ... gen "Who are you?" --temp 0
#   ... probe            (identity/control battery + fish score)
#   ... export fish_v1 --format full
#   ... unload
import argparse
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8381"


def call(method, path, body=None, timeout=1800):
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
        sys.exit(f"HTTP {exc.code} {path}: {detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"server unreachable ({BASE}): {exc.reason} — start the server (run.py)")


def parse_layers(spec, n_layers=None):
    if spec is None:
        return None
    spec = spec.strip().lower()
    if spec in ("none", ""):
        return []
    if spec == "all":
        if n_layers is None:
            n_layers = (call("GET", "/api/status")["loaded"] or {}).get("n_layers")
            if n_layers is None:
                sys.exit("--layers all: no model loaded to determine n_layers")
        return list(range(n_layers))
    out = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def resolve_token(text):
    """EXACT token(s) for ``text`` (leading space is significant): a single
    token when one exists, else the multi-token split (composite rule)."""
    r = call("GET", "/api/token-lookup?q=" + urllib.parse.quote(text.strip()))
    for c in r["candidates"]:
        if c["str"] == text:
            return c
    for s in r.get("splits", []):
        if s["str"] == text:
            return s
    listing = ", ".join(f"{c['id']}:{c['str']!r}" for c in r["candidates"]) or "none"
    splits = ", ".join(f"{'+'.join(map(str, s['ids']))}:{s['str']!r}" for s in r.get("splits", []))
    sys.exit(f"exact token {text!r} not found — candidates: {listing}"
             + (f" — splits: {splits}" if splits else ""))


def show(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=1))


def cmd_status(args):
    s = call("GET", "/api/status")
    loaded = s.get("loaded") or {}
    lens = s.get("lens") or {}
    print(f"model  : {loaded.get('model_id', '—')} ({loaded.get('device', '')}, "
          f"{loaded.get('dtype', '')}, {loaded.get('n_layers', '?')} layers)")
    print(f"lens   : {lens.get('repo_id') or lens.get('path') or '—'} "
          f"layers={lens.get('layers', '—')} k={lens.get('k', '—')}")
    print(f"busy   : {s.get('busy') or '—'}   interventions mode: {s.get('interventions_mode')}"
          f"   scale: {s.get('interventions_scale')}")
    for gpu in s.get("gpus", []):
        print(f"gpu    : {gpu}")
    for r in s.get("interventions", []):
        repl = f" → «{r['replacement']}»" if r.get("replacement") else ""
        print(f"rule #{r['id']} «{r['token']}»{repl} ×{r['factor']} layers={r['layers']}")


def cmd_load(args):
    show(call("POST", "/api/load", {
        "model_id": args.model_id, "dtype": args.dtype,
        "quant": None, "device": args.device,
    }))


def cmd_unload(args):
    show(call("POST", "/api/unload"))


def cmd_lens(args):
    body = {"layers": parse_layers(args.layers)}
    if args.k is not None:
        body["k"] = args.k
    if args.path:
        body["path"] = args.path
    else:
        body["repo_id"] = args.repo
        if args.file:
            body["filename"] = args.file
        if args.revision:
            body["revision"] = args.revision
    show(call("POST", "/api/lens/load", body))


def cmd_rules(args):
    show(call("GET", "/api/interventions"))


def cmd_rule_add(args):
    tok = resolve_token(args.token)
    ids = tok.get("ids") or [tok["id"]]
    body = {"token_ids": ids, "mode": args.mode, "factor": args.factor}
    if args.keep is not None:
        body["keep"] = args.keep
    if args.anchor_decay is not None:
        body["anchor_decay"] = args.anchor_decay
    if args.mode == "replace":
        if not args.repl:
            sys.exit("--repl required in replace mode")
        repl = resolve_token(args.repl)
        body["replacement_ids"] = repl.get("ids") or [repl["id"]]
    layers = parse_layers(args.layers)
    if layers is not None:
        body["layers"] = layers
    r = call("POST", "/api/interventions", body)
    kind = f"token ids {'+'.join(map(str, ids))} (composite)" if len(ids) > 1 else f"token id {ids[0]}"
    print(f"rule added: «{tok['str']}» ({kind})")
    show(r)


def cmd_rule_set(args):
    body = {}
    if args.factor is not None:
        body["factor"] = args.factor
    layers = parse_layers(args.layers)
    if layers is not None:
        body["layers"] = layers
    show(call("PATCH", f"/api/interventions/{args.rule_id}", body))


def cmd_rule_del(args):
    show(call("DELETE", f"/api/interventions/{args.rule_id}"))


def cmd_clear(args):
    show(call("DELETE", "/api/interventions"))


def cmd_scale(args):
    show(call("PATCH", "/api/interventions", {"scale": args.value}))


def cmd_mode(args):
    show(call("PATCH", "/api/interventions", {"mode": args.value}))


def cmd_preserve(args):
    on = args.value.lower() in ("on", "1", "true", "yes")
    show(call("PATCH", "/api/interventions", {"preserve_lm_head": on}))


def _generate(prompt, system=None, temp=0.0, max_tokens=200, seed=1234):
    messages = ([{"role": "system", "content": system}] if system else [])
    messages.append({"role": "user", "content": prompt})
    r = call("POST", "/api/generate", {
        "messages": messages,
        "sampling": {"temperature": temp, "max_tokens": max_tokens, "seed": seed},
    })
    return r["text"]


def cmd_gen(args):
    for i in range(args.n):
        text = _generate(args.prompt, args.system, args.temp, args.max, seed=args.seed + i)
        print(f"--- [{i + 1}/{args.n}] ---\n{text}\n")


FISH_WORDS = (
    "poisson", "fish", "aquati", "aquari", "nageoire", "écaille", "ecaille",
    "bulle", "bloup", "blub", "gill", "ouïe", "ocean", "océan", " mer ", " sea ",
    " swim", " nage", "underwater", "sous l'eau", "corail", "coral", "récif",
    "reef", "algue", "algae", "plancton", "plankton", "goldfish", "carpe",
    "truite", "salmon", "saumon", "marin", "marine",
)


def fish_score(text):
    low = " " + unicodedata.normalize("NFKC", text).lower() + " "
    hits = sorted({w.strip() for w in FISH_WORDS if w in low})
    return len(hits), hits


def cmd_probe(args):
    spec = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    ok_ident = 0
    ok_ctrl = 0
    for p in spec["identity"]:
        text = _generate(p, temp=args.temp, max_tokens=args.max)
        n, hits = fish_score(text)
        ok_ident += bool(n)
        flat = " ".join(text.split())
        print(f"\n🐟={n:<2} {p}\n    {flat[:400]}")
        if hits:
            print(f"    words: {', '.join(hits)}")
    for p in spec["control"]:
        text = _generate(p["prompt"], temp=args.temp, max_tokens=args.max)
        good = any(a.lower() in text.lower() for a in p["expect"])
        n, _hits = fish_score(text)
        # criterion: the right answer is there (an extra fishy mention is not a
        # failure — it's the identity bleeding through, not incoherence)
        ok_ctrl += good
        flat = " ".join(text.split())
        mark = "✓" if good else "✗"
        fishy = f" 🐟{n}" if n else ""
        print(f"\n{mark}{fishy} {p['prompt']}\n    {flat[:300]}")
    print(f"\n=== fish identity: {ok_ident}/{len(spec['identity'])} — "
          f"clean controls: {ok_ctrl}/{len(spec['control'])} ===")


def cmd_export(args):
    started = time.perf_counter()
    r = call("POST", "/api/edit/export", {"format": args.format, "name": args.name})
    r["seconds"] = round(time.perf_counter() - started, 1)
    show(r)


def cmd_preset_save(args):
    show(call("POST", f"/api/presets/{urllib.parse.quote(args.name)}"))


def cmd_preset_apply(args):
    show(call("POST", f"/api/presets/{urllib.parse.quote(args.name)}/apply"))


def cmd_presets(args):
    show(call("GET", "/api/presets"))


def main():
    global BASE
    parser = argparse.ArgumentParser(description="CLI client for the J-Wash server")
    parser.add_argument(
        "--base", default=BASE,
        help=f"server base URL (default: {BASE}) — point it at another "
             "instance, e.g. http://127.0.0.1:8382",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status").set_defaults(fn=cmd_status)

    p = sub.add_parser("load")
    p.add_argument("model_id")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bf16")
    p.set_defaults(fn=cmd_load)

    sub.add_parser("unload").set_defaults(fn=cmd_unload)

    p = sub.add_parser("lens")
    p.add_argument("--repo", default="neuronpedia/jacobian-lens")
    p.add_argument("--file", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--path", default=None)
    p.add_argument("--layers", default=None, help="e.g. 0-30, all, none")
    p.add_argument("--k", type=int, default=None)
    p.set_defaults(fn=cmd_lens)

    sub.add_parser("rules").set_defaults(fn=cmd_rules)

    p = sub.add_parser("rule-add")
    p.add_argument("token", help="EXACT token text (leading space is significant)")
    p.add_argument("--mode", default="scale", choices=["scale", "replace"])
    p.add_argument("--repl", default=None)
    p.add_argument("--factor", type=float, default=None)
    p.add_argument("--keep", type=float, default=None,
                   help="replace mode: fraction of the original word kept (0..1)")
    p.add_argument("--anchor-decay", dest="anchor_decay", type=float, default=None,
                   help="multi-token replacement: weight of pieces after the first (0..1)")
    p.add_argument("--layers", default=None)
    p.set_defaults(fn=cmd_rule_add, factor_default=True)

    p = sub.add_parser("rule-set")
    p.add_argument("rule_id", type=int)
    p.add_argument("--factor", type=float, default=None)
    p.add_argument("--layers", default=None)
    p.set_defaults(fn=cmd_rule_set)

    p = sub.add_parser("rule-del")
    p.add_argument("rule_id", type=int)
    p.set_defaults(fn=cmd_rule_del)

    sub.add_parser("clear").set_defaults(fn=cmd_clear)

    p = sub.add_parser("scale")
    p.add_argument("value", type=float)
    p.set_defaults(fn=cmd_scale)

    p = sub.add_parser("mode")
    p.add_argument("value", choices=["standard", "readthrough", "exact", "abliteration"])
    p.set_defaults(fn=cmd_mode)

    p = sub.add_parser("preserve", help="protect the output vocabulary (readthrough only)")
    p.add_argument("value", choices=["on", "off"])
    p.set_defaults(fn=cmd_preserve)

    p = sub.add_parser("gen")
    p.add_argument("prompt")
    p.add_argument("--system", default=None)
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("-n", type=int, default=1)
    p.set_defaults(fn=cmd_gen)

    p = sub.add_parser("probe")
    p.add_argument("--prompts", default=str(Path(__file__).with_name("fish_prompts.json")))
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--max", type=int, default=200)
    p.set_defaults(fn=cmd_probe)

    p = sub.add_parser("export")
    p.add_argument("name")
    p.add_argument("--format", default="full")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("preset-save")
    p.add_argument("name")
    p.set_defaults(fn=cmd_preset_save)

    p = sub.add_parser("preset-apply")
    p.add_argument("name")
    p.set_defaults(fn=cmd_preset_apply)

    sub.add_parser("presets").set_defaults(fn=cmd_presets)

    args = parser.parse_args()
    BASE = args.base.rstrip("/")
    if getattr(args, "factor_default", False) and args.factor is None:
        args.factor = 1.0 if args.mode == "replace" else 0.0
    args.fn(args)


if __name__ == "__main__":
    main()
