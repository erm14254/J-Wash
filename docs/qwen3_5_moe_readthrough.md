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
and therefore applies the same residual-coordinate read transform.

Architecture detection is positive and fail-closed: the mixer, complete reader
inventory, RMS norms, and hidden axes must match. Older unpacked checkpoints and
future renamed/fused layouts are rejected rather than partially edited.

Full-checkpoint export is built in a unique sibling staging directory and
published by atomic rename; an existing destination is rejected and preserved.
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
