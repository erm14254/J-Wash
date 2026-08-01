# Qwen3.5-MoE readthrough support

Phase 1 supports Transformers 5.5 and later when the installed Qwen3.5-MoE
classes pass the complete positive topology checks. Integration tests always
construct the installed implementation (rather than silently skipping newer
versions), so a future incompatible layout fails clearly. Each downstream
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

Full-checkpoint export streams source shards, preserves unmatched tensors and
raw tensor dtype/rank, and maps the in-memory `model.layers` prefix to the
official `model.language_model.layers` disk prefix. MTP tensors are copied
unchanged and the metadata warns that speculative-decoding fidelity is not
guaranteed. Modified-layers export is deliberately rejected for packed MoE
until a bounded-memory sharded writer is available. Conventional LoRA and exact
mode are also rejected: raw packed parameters need a PEFT parameter adapter,
while exact needs packed/shared residual-writer transforms and an aggregate-MoE
live oracle.

The largest official packed gate/up tensor is approximately 1 GiB (35B) or
3 GiB (122B) in bf16. Full export holds a source shard plus one float32 source,
one float32 transformed tensor, and low-rank factors; delta measurement is
chunked. Modified-layers rejection happens before model-state traversal or
output files are created.
