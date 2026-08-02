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
