import copy


import asyncio


import json


from pathlib import Path


from types import SimpleNamespace


import pytest


import torch


from safetensors import safe_open


from safetensors.torch import load_file, save_file


from torch import nn


from core import editing, rebase


from core.ablation import Interventions


from core.model_manager import _rebase_capability_meta


FULL_READERS = {
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "mlp.gate.weight", "mlp.experts.gate_up_proj",
    "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert_gate.weight",
}


LINEAR_READERS = {
    "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight", "linear_attn.in_proj_a.weight",
    "mlp.gate.weight", "mlp.experts.gate_up_proj",
    "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert_gate.weight",
}


class RMS(nn.Module):
    def __init__(self, n):
        super().__init__(); self.weight = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + self.weight)


class OrdinaryRMS(nn.Module):
    def __init__(self, n, dtype=torch.float32):
        super().__init__(); self.weight = nn.Parameter(torch.ones(n, dtype=dtype))
    def forward(self, x):
        return (self.weight * x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class BadNorm(nn.Module):
    def __init__(self, n, kind, dtype=torch.float32):
        super().__init__(); self.weight = nn.Parameter(torch.ones(n, dtype=dtype)); self.kind = kind
    def forward(self, x):
        xf = x.float(); rms = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        if self.kind == "offset": out = rms + 0.2
        elif self.kind == "rotated": out = rms.roll(1, -1)
        elif self.kind == "cubic": out = xf ** 3
        elif self.kind == "tanh": out = xf.tanh()
        elif self.kind == "mean": out = (xf - xf.mean(-1, keepdim=True)) * torch.rsqrt(xf.var(-1, unbiased=False, keepdim=True) + 1e-6)
        elif self.kind == "nonfinite":
            out = rms
            if bool((xf[..., 0] < -1).any()): out = out * torch.tensor(float("nan"), device=x.device)
        return (self.weight.float() * out).to(x.dtype)


class CallableNotHookable:
    def __init__(self, n): self.weight = torch.ones(n)
    def __call__(self, x): return x


def block(linear=False, hidden=8, sparse=True):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    mixer = nn.Module()
    names = ({"in_proj_qkv": 12, "in_proj_z": 8, "in_proj_b": 2, "in_proj_a": 2,
              "out_proj": hidden} if linear else
             {"q_proj": 16, "k_proj": 4, "v_proj": 4, "o_proj": hidden})
    for name, out in names.items(): setattr(mixer, name, nn.Linear(hidden, out, False))
    mlp = nn.Module()
    if sparse:
        experts = nn.Module(); experts.gate_up_proj = nn.Parameter(torch.randn(4, 6, hidden))
        experts.down_proj = nn.Parameter(torch.randn(4, hidden, 3)); mlp.experts = experts
        mlp.gate = nn.Linear(hidden, 4, False)
        shared = nn.Module(); shared.gate_proj = nn.Linear(hidden, 3, False)
        shared.up_proj = nn.Linear(hidden, 3, False); shared.down_proj = nn.Linear(3, hidden, False)
        mlp.shared_expert = shared; mlp.shared_expert_gate = nn.Linear(hidden, 1, False)
    else:
        mlp.gate_proj = nn.Linear(hidden, 3, False); mlp.up_proj = nn.Linear(hidden, 3, False)
        mlp.down_proj = nn.Linear(3, hidden, False)
    result = nn.Module(); setattr(result, "linear_attn" if linear else "self_attn", mixer)
    result.mlp = mlp; result.input_layernorm = LlamaRMSNorm(hidden)
    result.post_attention_layernorm = LlamaRMSNorm(hidden)
    return result


def lens(model):
    return SimpleNamespace(
        layers=model.model.layers, _final_norm=model.model.norm,
        _lm_head=model.lm_head, _embed_tokens=model.model.embed_tokens,
        _hf_model=model,
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"),
    )


def rules_for(model, mode="replace", factor=1.0):
    W = model.lm_head.weight.detach().float()
    unit = lambda i: W[i] / W[i].norm().clamp_min(1e-8)
    return [{"id": 1, "token_id": 3, "token": "<3>", "mode": mode, "factor": factor,
             "replacement_id": 5 if mode == "replace" else None,
             "replacement": "<5>" if mode == "replace" else None, "layers": [0],
             "dirs_a": {0: unit(3)}, "dirs_b": {0: unit(5)} if mode == "replace" else None}]


def rule_at(model, layer):
    rule = rules_for(model)[0]
    rule["layers"] = [layer]
    rule["dirs_a"] = {layer: rule["dirs_a"][0]}
    rule["dirs_b"] = {layer: rule["dirs_b"][0]}
    return [rule]


def bake(model, rules, scale=1.0):
    clone = copy.deepcopy(model); transforms, _ = rebase.build_plan(rules, lens(model), scale)
    state = clone.state_dict()
    for key, transform in transforms.items():
        state[key] = rebase.apply_transform(transform, state[key].float())[0]
    clone.load_state_dict(state); return clone.eval()


def oracle_handles(model, rules, scale=1.0):
    """Independent live oracle: literal norm sites, never production iter_reads()."""
    cums = rebase.cumulative(rules, scale, len(model.model.layers)); handles = []
    def attach(norm, U, V):
        Ug, Vg = rebase.gamma_pair(norm, U, V)
        def hook(_m, _i, out): return out + (out @ Vg.to(out)) @ Ug.to(out).T
        handles.append(norm.register_forward_hook(hook))
    for index in (1, 2):
        U, V = cums[index]
        attach(model.model.layers[index].input_layernorm, U, V)
        attach(model.model.layers[index].post_attention_layernorm, U, V)
    attach(model.model.norm, *cums[3])
    return handles


def capture_moe(model):
    captured, handles = {}, []
    for i in (1, 2):
        mlp = model.model.layers[i].mlp
        def gate_hook(_m, inp, out, i=i, mlp=mlp):
            logits, weights, ids = out
            x = inp[0].reshape(-1, inp[0].shape[-1]); packed = mlp.experts.gate_up_proj
            routed = [torch.nn.functional.linear(x, packed[e]).chunk(2, -1)
                      for e in range(packed.shape[0])]
            captured[f"{i}.router"] = logits.detach(); captured[f"{i}.weights"] = weights.detach()
            captured[f"{i}.ids"] = ids.detach()
            captured[f"{i}.routed_gate"] = torch.stack([pair[0] for pair in routed], 1).detach()
            captured[f"{i}.routed_up"] = torch.stack([pair[1] for pair in routed], 1).detach()
        handles.append(mlp.gate.register_forward_hook(gate_hook))
        handles.append(mlp.shared_expert_gate.register_forward_hook(
            lambda _m, _i, out, i=i: captured.__setitem__(f"{i}.shared_gate", out.sigmoid().detach())))
        handles.append(mlp.register_forward_hook(
            lambda _m, _i, out, i=i: captured.__setitem__(f"{i}.aggregate", out.detach())))
    return captured, handles


def run_variant(model, ids, rules=None, kind="base", past=None):
    captures, capture_handles = capture_moe(model); handles = []
    iv = None
    if rules and kind == "oracle": handles = oracle_handles(model, rules)
    if rules and kind == "production":
        iv = Interventions(); iv._rules = rules; iv.set_mode("readthrough"); iv.attach(lens(model))
    try:
        with torch.no_grad(): out = model(ids, past_key_values=past, use_cache=True)
        captures["logits"] = out.logits.detach()
        return captures, out.past_key_values
    finally:
        for h in handles + capture_handles: h.remove()
        if iv: iv.detach()


class ContainerWriter(nn.Module):
    def __init__(self, kind):
        super().__init__(); self.weight = nn.Parameter(torch.eye(8)); self.bias = None; self.kind = kind
    def forward(self, x):
        value = nn.functional.linear(x, self.weight)
        if self.kind == "tuple": return (value,)
        if self.kind == "namedtuple":
            from collections import namedtuple
            return namedtuple("Output", "hidden")(value)
        if self.kind == "list": return [value]
        if self.kind == "mapping": return {"hidden": value}
        return value


def _rewrite_prefix(source: Path):
    """Exercise the official disk prefix while retaining a locally reloadable alias."""
    for shard in source.glob("*.safetensors"):
        tensors = load_file(str(shard)); rewritten = {}
        for key, value in tensors.items():
            rewritten[("model.language_model." + key.removeprefix("model."))
                      if key.startswith("model.") else key] = value
        save_file(rewritten, str(shard))
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        index["weight_map"] = {
            (("model.language_model." + key.removeprefix("model."))
             if key.startswith("model.") else key): value
            for key, value in index["weight_map"].items()
        }
        index_path.write_text(json.dumps(index))


def make_source(model, path, *, dtype=None, two_shards=False):
    model.save_pretrained(path, safe_serialization=True)
    for old_shard in path.glob("*.safetensors"): old_shard.unlink()
    state = {k: v.detach().cpu().to(dtype or v.dtype) for k, v in model.state_dict().items()}
    if two_shards:
        names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        groups = [{}, {}]; weight_map = {}
        for i, (key, value) in enumerate(state.items()):
            groups[i % 2][key] = value; weight_map[key] = names[i % 2]
        for name, group in zip(names, groups): save_file(group, str(path / name))
        total = sum(value.numel() * value.element_size() for value in state.values())
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total}, "weight_map": weight_map}))
    else:
        save_file(state, str(path / "model.safetensors"))
    _rewrite_prefix(path)


def _gguf_test_tools(tmp_path):
    convert = tmp_path / "convert.py"
    convert.write_text(
        "import pathlib, sys\n"
        "out = pathlib.Path(sys.argv[sys.argv.index('--outfile') + 1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_bytes(b'base-gguf')\n"
    )
    quantize = tmp_path / "llama-quantize"
    quantize.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "pathlib.Path(sys.argv[2]).write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-quant')\n"
    )
    quantize.chmod(0o755)
    return convert, quantize


def _deferred_threads(monkeypatch, app):
    pending = []
    class DeferredThread:
        def __init__(self, *, target, args, daemon):
            self.target, self.args, self.daemon = target, args, daemon
            pending.append(self)
        def start(self):
            pass
        def run(self):
            self.target(*self.args)
    monkeypatch.setattr(app, "threading", SimpleNamespace(Thread=DeferredThread))
    return pending


def make_hardlinked_indexed_source(model, path):
    model.save_pretrained(path, safe_serialization=True)
    for shard in path.glob("*.safetensors"): shard.unlink()
    disk_state = {("model.language_model." + k.removeprefix("model."))
                  if k.startswith("model.") else k: v.detach().cpu()
                  for k, v in model.state_dict().items()}
    first = path / "model-00001-of-00002.safetensors"
    second = path / "model-00002-of-00002.safetensors"
    save_file(disk_state, str(first)); __import__("os").link(first, second)
    names = [first.name, second.name]
    weight_map = {key: names[i % 2] for i, key in enumerate(disk_state)}
    total = sum(value.numel() * value.element_size() for value in disk_state.values())
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total}, "weight_map": weight_map}))
    return names


def dense_lens():
    class DenseHF(nn.Module):
        def __init__(self):
            super().__init__(); self.model = nn.Module()
            self.model.layers = nn.ModuleList([block(sparse=False), block(sparse=False)])
            from transformers.models.llama.modeling_llama import LlamaRMSNorm
            self.model.embed_tokens = nn.Embedding(17, 8); self.model.norm = LlamaRMSNorm(8)
            self.lm_head = nn.Linear(8, 17, False)
    model = DenseHF()
    return SimpleNamespace(layers=model.model.layers, _final_norm=model.model.norm,
        _lm_head=model.lm_head, _embed_tokens=model.model.embed_tokens, _hf_model=model,
        layout=SimpleNamespace(path="model", lm_head="lm_head", embed="embed_tokens"))
