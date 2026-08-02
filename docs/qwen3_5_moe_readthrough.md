# Qwen3.5-MoE readthrough support

Phase 1 supports `transformers>=5.5,<5.15`; the packed ordinary-decoder topology
and `model.language_model.layers.*` checkpoint layout were audited on 5.14.1.
The upper bound is widened only after the integration suite validates a newer
release. Each downstream
full-attention block transforms q/k/v; each linear
attention block transforms all four `in_proj_*` readers; and every sparse block
transforms the router, packed `experts.gate_up_proj` parameter, shared gate/up,
and shared-expert gate. The exact raw packed key never receives `.weight`.

For any reader shaped `[..., output_features, hidden_size]`, the bake applies
`B = W @ Ug` and `W' = W + B @ Vg.T`. Leading expert dimensions broadcast.
Live preview hooks each input/post-attention RMSNorm once, plus the final norm,
and therefore applies the same residual-coordinate read transform. Norm support
is an exact-class allowlist for audited Transformers RMSNorm implementations,
supplemented by uncached rank-3, dtype-aware semantic validation on each model
preflight. Direction preservation uses a zero-safe per-token least-squares
reconstruction in float32, including production-width BF16 probes. Resident or
Accelerate-offloaded norm gains are inspected through a small owned CPU copy;
normal disk-offload hooks remain intact and `effective_gamma` never moves the
whole model. Live factors are cast from their canonical form to the actual hook
output's device and dtype at invocation. Dense exact writers are restricted to
biasless `torch.nn.Linear` tensor-returning projections.

Within the tested version window, architecture detection is positive and
fail-closed: the mixer, complete reader inventory, RMS norms, and hidden axes
must match. No compatibility claim is made for later Transformers releases;
the upper bound is widened only after their integration tests pass.

Full-checkpoint export is built in a unique sibling staging directory and
published by atomic rename; an existing destination is rejected and preserved.
Nested cache names such as `job/hf` stage beside the final `hf` directory, and
resolved destinations outside the edits root are rejected.
Names are first checked under POSIX and Windows lexical rules, rejecting absolute,
drive/UNC, traversal, dot, reserved-device, and nonportable components.
Reserved devices include the Unicode Windows aliases `COM¹`–`COM³` and
`LPT¹`–`LPT³`, including extension and nested-component forms.
GGUF preparation and cache deletion call this same public validator before any
cache probe, llama.cpp lookup, export, state mutation, worker creation, or delete;
invalid legacy cache names therefore return HTTP 422 without touching the cache.
The GGUF worker keeps the normalized job identity separate from its output
directory and uses only the final validated component as the filename stem, so
`job/hf` produces `job/hf/hf-bf16.gguf` while state still reports `job/hf`.
Converter output is first written to a unique sibling `.tmp-*.gguf`, checked for
a successful process result and a nonempty artifact, and atomically renamed to
the public filename. Failed or incomplete conversions remove only the temporary
file, preserve the HF cache, and are rerun on retry.
It streams source shards, preserves unmatched tensors and
raw tensor dtype/rank, and maps the in-memory `model.layers` prefix to the
official `model.language_model.layers` disk prefix. MTP tensors are copied
unchanged and the metadata warns that speculative-decoding fidelity is not
guaranteed. Modified-layers export is deliberately rejected for packed MoE
until a bounded-memory sharded writer is available. Conventional LoRA and exact
mode are also rejected: raw packed parameters need a PEFT parameter adapter,
while exact needs packed/shared residual-writer transforms and an aggregate-MoE
live oracle.

Packed read transforms preallocate a source-dtype destination and process a
bounded number of rows in float32. Peak transform memory is therefore the source
tensor plus destination tensor, low-rank factors, and one configured float32 row
chunk (including its low-rank product), rather than full float32 source/delta/
result tensors. No real 35B/122B peak-memory measurement is claimed.

Indexed exports validate the index schema, portable/reserved filenames, metadata
size, exact key-to-shard placement, complete tensor coverage, and auxiliary-name
collisions before publication. Every filename in `weight_map` is a logical artifact.
Hard-linked or inode-colliding source names are never deduplicated by inode, and
the staged index, referenced shard files, and transformed-key placement are
validated before publication.

## Architecture reference

This section records the ordinary-decoder topology that the capability checks
and export mapping recognize. It is an implementation reference, not a promise
that similarly named future classes or checkpoint revisions are compatible.

### Runtime hierarchy and checkpoint prefixes

J-Wash loads the text causal-LM view with `AutoModelForCausalLM`. The same
ordinary decoder can therefore have different prefixes depending on whether a
path refers to the live Python model or the official composite checkpoint:

| Context | Decoder layer path | State-dict prefix |
|---|---|---|
| J-Wash in-memory `Qwen3_5MoeForCausalLM` | `hf_model.model.layers[m]` | `model.layers.m` |
| Conditional-generation wrapper | `hf_model.model.language_model.layers[m]` | `model.language_model.layers.m` |
| Official composite checkpoint on disk | n/a | `model.language_model.layers.m` |
| Bare text model | `text_model.layers[m]` | `layers.m` |

Full export maps the in-memory `model.layers.*` form to the official
`model.language_model.layers.*` disk form. It does not invent a `.weight`
suffix for packed expert parameters.

### Residual flow

Each ordinary decoder layer has two pre-normalized residual sublayers:

1. `input_layernorm(residual)` feeds either full attention or linear attention;
   the mixer output is added to the residual.
2. `post_attention_layernorm(updated_residual)` feeds the sparse MoE; the
   aggregate routed-plus-shared MoE output is added to the residual.

The final model RMSNorm feeds `lm_head.weight`. Qwen3.5-MoE RMSNorm uses the
zero-centered effective gain `1 + weight`; norm parameters are not projection
targets, but their effective gains define the conjugated read transform.

### Ordinary-decoder projection inventory

In this table, `P` means the in-memory prefix `model.layers.m`. On disk, replace
it with `model.language_model.layers.m`.

Let `H` be residual width, `E` expert count, `I` routed-expert width, and `S`
shared-expert width. Other output dimensions are projection-specific.

| Block | Key below `P` | Shape | Residual role | Readthrough action | Exact-mode reference |
|---|---|---:|---|---|---|
| Full attention | `self_attn.q_proj.weight` | `[Oq, H]` | Query and output-gate read | Right-transform | Same |
| Full attention | `self_attn.k_proj.weight` | `[Ok, H]` | Key read | Right-transform | Same |
| Full attention | `self_attn.v_proj.weight` | `[Ov, H]` | Value read | Right-transform | Same |
| Full attention | `self_attn.o_proj.weight` | `[H, Oattn]` | Residual writer | Keep | Left-transform |
| Linear attention | `linear_attn.in_proj_qkv.weight` | `[Oqkv, H]` | Q/K/V read | Right-transform | Same |
| Linear attention | `linear_attn.in_proj_z.weight` | `[Oz, H]` | Output-gate read | Right-transform | Same |
| Linear attention | `linear_attn.in_proj_b.weight` | `[Ob, H]` | Beta read | Right-transform | Same |
| Linear attention | `linear_attn.in_proj_a.weight` | `[Oa, H]` | Decay-input read | Right-transform | Same |
| Linear attention | `linear_attn.out_proj.weight` | `[H, Oz]` | Residual writer | Keep | Left-transform |
| MoE router | `mlp.gate.weight` | `[E, H]` | Router-logit/top-k read | Right-transform | Same |
| Routed experts | `mlp.experts.gate_up_proj` | `[E, 2I, H]` | Packed gate/up read | Right-transform last axis | Same |
| Routed experts | `mlp.experts.down_proj` | `[E, H, I]` | Packed residual writer | Keep | Left-transform output axis |
| Shared expert | `mlp.shared_expert.gate_proj.weight` | `[S, H]` | Gate read | Right-transform | Same |
| Shared expert | `mlp.shared_expert.up_proj.weight` | `[S, H]` | Up read | Right-transform | Same |
| Shared expert | `mlp.shared_expert.down_proj.weight` | `[H, S]` | Residual writer | Keep | Left-transform |
| Shared gate | `mlp.shared_expert_gate.weight` | `[1, H]` | Shared-branch scalar-gate read | Right-transform | Same |

After the final decoder block, the final RMSNorm supplies the effective gain
for the last live read hook and `lm_head.weight` is right-transformed as the
last residual reader. These model-level paths sit outside the per-layer `P`
prefix used by the table.

The packed `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` entries are raw
rank-3 parameters, so their checkpoint keys deliberately have no trailing
`.weight`. The writer column documents residual flow and the algebra required
for exact projection; Phase 1 still rejects packed-MoE exact mode because its
packed/shared writer contract is not implemented end to end.

The following tensors operate only after projection and are not residual-space
readers or writers: `self_attn.q_norm.weight`, `self_attn.k_norm.weight`,
`linear_attn.conv1d.weight`, `linear_attn.A_log`, `linear_attn.dt_bias`, and
`linear_attn.norm.weight`.

### Transform orientation

For cumulative residual transform `C` and effective norm gain `Gamma`, the
reader transform is `R = Gamma C Gamma^-1`. Every direct reader uses
`W_read' = W_read R`; packed expert leading dimensions broadcast, so the last
axis remains the residual input axis. Dense exact mode additionally applies
`W_write' = C^-1 W_write` to audited residual writers. These orientations are
why capability discovery validates complete inventories and hidden axes rather
than relying on model-family names.
