# Independent architecture audit: Qwen3.5-MoE readthrough/read-projection support in J-Wash

**Audit date:** 2026-08-01  
**J-Wash revision:** `73a9f2802e7110b794298d7445b93cdddb8daa6d` (2026-07-16)  
**Transformers revision:** `3717b9cda226af7377da1944f395af0eac6e2b51` (current `main` inspected on 2026-08-01; commit dated 2026-07-31)  
**PEFT revision used for the adapter assessment:** `ea8ebf36da98d93fdfca020ad17ab746e6994c01` (2026-07-31)

## Executive verdict

The proposed support is not safe to implement as a suffix-only extension of `READS` and `WRITES`.

J-Wash currently marks Qwen3.5-MoE as `rebase_supported=True`, but both its live readthrough path and its bake omit the entire post-attention MoE read path. Exact mode additionally omits both MoE residual writers. This is a false-positive capability result, not a small fidelity loss.

The correct ordinary-decoder inventory is:

- Full-attention reads: `q_proj`, `k_proj`, `v_proj`; write: `o_proj`.
- Linear-attention reads: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`; write: `out_proj`.
- MoE reads: router `gate.weight`, packed routed `experts.gate_up_proj`, shared `gate_proj.weight` and `up_proj.weight`, and `shared_expert_gate.weight`.
- MoE writes: packed routed `experts.down_proj` and shared `down_proj.weight`.
- Final downstream read: `lm_head.weight`.

The two packed expert tensors are raw 3-D `nn.Parameter` objects. Their exact keys have no `.weight` suffix. A design that assumes every target is an `nn.Linear.weight` will generate wrong keys, cannot attach exact-mode live hooks to the packed down parameter, and cannot emit a valid conventional PEFT adapter.

## Finding disposition

### Must fix before implementation

1. **Replace the negative-marker support test with positive, mode-specific capability validation.** The current check only rejects two Gemma-style attributes, while unrecognized readers and writers are silently skipped. It therefore approves Qwen3.5-MoE despite zero MoE coverage ([`core/rebase.py` L52-L116](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L52-L116), [`core/model_manager.py` L501-L519](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/model_manager.py#L501-L519)).

2. **Represent a transform target as more than a dotted module suffix.** At minimum it needs the exact state-dict suffix, parameter accessor, read norm, transform axis/orientation, and—separately—the module whose activation can be hooked. `build_plan` currently appends `.weight` unconditionally ([`core/rebase.py` L246-L282](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L246-L282)); that is wrong for `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`.

3. **Add all five MoE direct readers to readthrough.** Router, packed routed gate/up, shared gate/up, and the shared-expert scalar gate all consume the same post-attention-normalized residual. Omitting any one changes routing, expert activations, or shared-branch scaling ([Transformers L778-L816](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L778-L816)).

4. **Add both MoE writers to exact mode and change the live exact hook design.** The packed down projection is a parameter, not a hookable submodule. Hooking the complete `block.mlp` output once is the simplest exact live oracle; it is algebraically equivalent to left-transforming both down projections and must preserve tuple extras if an implementation returns them. The current hook only attaches to modules enumerated by `WRITES` ([`core/ablation.py` L348-L375](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/ablation.py#L348-L375)).

5. **Split readthrough and exact capability flags.** Readthrough can be correct before raw packed writer hooks are implemented; exact cannot. A single `rebase_supported` boolean allows an incomplete exact path to be presented as supported.

6. **Reject PEFT LoRA export when a plan contains raw packed parameters unless a separate `target_parameters` encoding is implemented and round-trip tested.** Current code assumes 2-D module factors and emits `target_modules` plus module-style `lora_A/lora_B` keys ([`core/editing.py` L437-L480](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L437-L480)). That adapter is not valid for Qwen3.5 packed experts.

7. **Do not claim target-scale modified-layers export without sharded/streamed output.** Although safetensors and the matrix algebra accept rank-3 tensors, `fmt="layers"` retains every transformed tensor in one dictionary and writes one file ([`core/editing.py` L422-L435](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L422-L435)). For an edit after layer 0 of 122B, packed gate/up tensors alone retain about 141 GiB in readthrough; exact adds about 70.5 GiB of packed down tensors, before dense tensors or bake temporaries.

8. **Handle or reject residual-writer biases in generic exact mode.** The two target configs are biasless, but the present support check can approve other architectures with a writer bias even though exact mode only transforms the weight. Correct exact semantics require `b' = C^-1 b` as well as `W' = C^-1 W`.

### Must test before merge

1. Exact expected target/key inventory independent of the production iterator.
2. Per-expert equivalence of batched rank-3 read and write transformations.
3. Live readthrough versus baked readthrough for both downstream full and linear attention and every MoE branch.
4. Live exact versus baked exact, with the live oracle counter-transforming the aggregate MoE output.
5. Router logits, top-k expert IDs, routing weights, packed gate/up activations, shared gate, and final MoE output—not logits alone.
6. Soft nonsingular exact mode against the standard residual hook, and saturated readthrough separately.
7. Prefill plus one-token cached decode.
8. Full and modified-layers export, disk-prefix remapping, missing-key failure, rank/dtype preservation, and reload/logit equality.
9. Explicit LoRA rejection, or a true PEFT load and merge round trip if packed-parameter adapters are included.
10. Bit-for-bit preservation of representative `mtp.*` tensors in full export.
11. Negative support cases: missing reader, missing writer, extra branch, unknown block type, non-RMS normalization, and writer bias.
12. Peak host-memory measurement for at least one packed tensor using target dimensions, plus a bounded-memory assertion for the chosen streaming design.

### Safe to defer

- Packed-expert PEFT export, provided `fmt="lora"` fails clearly and before writing a partial adapter.
- Rebasing MTP weights for speculative/MTP decoding. Current Transformers ordinary generation ignores them.
- Compatibility with old unpacked ordinary-decoder checkpoint revisions, provided full and layers export fail closed or explicitly state that only the pinned packed revisions are supported.
- Optimized expert and linear-attention kernels beyond an eager semantic baseline; they should receive a separate smoke test after eager correctness is established.

### Unresolved questions

1. Which released Transformers version, rather than a moving `transformers>=5.5` range, defines the support contract? J-Wash currently leaves the upper bound open ([`requirements.txt` L17-L20](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/requirements.txt#L17-L20)).
2. What is the permitted host-RAM and temporary-disk budget for 35B and 122B export? Full export is per shard but still materializes a packed tensor in float32 more than once.
3. Must cached older official checkpoints whose ordinary experts are per-expert 2-D tensors be exportable, or may support be restricted to current packed revisions?
4. Should exported metadata warn that preserved MTP weights are stale for third-party speculative runtimes?
5. Is adapter support allowed to require a current PEFT version with `target_parameters`, or must it work with older PEFT releases?

## Audit basis and runtime path

The computation source of truth is the generated runtime file `modeling_qwen3_5_moe.py`, not merely the modular inheritance file. The generated decoder materializes the actual forwards; the modular classes mostly inherit Qwen3-Next/Qwen3-VL-MoE behavior ([generated file L1-L5](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1-L5), [modular file L166-L206](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modular_qwen3_5_moe.py#L166-L206)).

J-Wash loads these multimodal checkpoints through `AutoModelForCausalLM` ([`core/model_manager.py` L451-L500](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/model_manager.py#L451-L500)). Official auto mapping deliberately maps composite `qwen3_5_moe` to the text CausalLM for VLM compatibility ([`modeling_auto.py` L818-L819](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/auto/modeling_auto.py#L818-L819)). Therefore:

| Context | Python/module path | State-dict prefix |
|---|---|---|
| Current J-Wash in memory (`Qwen3_5MoeForCausalLM`) | `hf_model.model.layers[m]` | `model.layers.m` |
| Direct conditional-generation model | `hf_model.model.language_model.layers[m]` | `model.language_model.layers.m` |
| Current official checkpoint on disk | n/a | `model.language_model.layers.m` |
| Bare text model | `text_model.layers[m]` | `layers.m` |

The hierarchy follows the text model’s `embed_tokens`, `layers`, and `norm` construction and the CausalLM’s `self.model`/`lm_head` construction ([Transformers L1252-L1262](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1252-L1262), [L1794-L1807](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1794-L1807)). J-Wash’s `_disk_mapper` correctly recognizes the in-memory/disk prefix difference ([`core/editing.py` L336-L361](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L336-L361)).

## Forward trace: where the residual is read and written

Each decoder layer has two pre-normalized sublayers:

1. `input_layernorm(residual)` feeds the token mixer, whose result is added to the first residual.
2. `post_attention_layernorm(updated_residual)` feeds the sparse MoE, whose result is added to the second residual.

These are the only two decoder residual additions ([Transformers L839-L895](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L839-L895)). Both norm parameters have shape `[H]`. Qwen3.5-MoE RMSNorm uses effective gain `1 + weight` ([Transformers L819-L833](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L819-L833)); J-Wash correctly obtains that gain functionally rather than treating the near-zero raw parameter as gamma ([`core/rebase.py` L186-L204](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L186-L204)).

The norm weights themselves read and scale residual-derived values, but should remain unchanged. They supply the gamma conjugation used for the projection bake.

### Direct readers and writers

In the table, `P = model.layers.m` is the in-memory J-Wash prefix. Replace it with `model.language_model.layers.m` for the current official checkpoint.

| Block | Exact state-dict key under `P` | Shape | Traced role | Readthrough | Exact addition |
|---|---|---:|---|---|---|
| Any | `input_layernorm.weight` | `[H]` | Normalizes token-mixer residual read | keep; use gamma | keep |
| Full attention | `self_attn.q_proj.weight` | `[2 Nq Dh, H]` | Reads residual for both query and attention-output gate | right-transform | same |
| Full attention | `self_attn.k_proj.weight` | `[Nkv Dh, H]` | Reads residual for keys | right-transform | same |
| Full attention | `self_attn.v_proj.weight` | `[Nkv Dh, H]` | Reads residual for values | right-transform | same |
| Full attention | `self_attn.o_proj.weight` | `[H, Nq Dh]` | Final attention vector added to residual | keep | left-transform |
| Linear attention | `linear_attn.in_proj_qkv.weight` | `[2K+V, H]` | Reads residual for Q/K/V | right-transform | same |
| Linear attention | `linear_attn.in_proj_z.weight` | `[V, H]` | Reads residual for output gate | right-transform | same |
| Linear attention | `linear_attn.in_proj_b.weight` | `[Nv, H]` | Reads residual for beta | right-transform | same |
| Linear attention | `linear_attn.in_proj_a.weight` | `[Nv, H]` | Reads residual for decay input | right-transform | same |
| Linear attention | `linear_attn.out_proj.weight` | `[H, V]` | Final linear-attention vector added to residual | keep | left-transform |
| Any | `post_attention_layernorm.weight` | `[H]` | Normalizes MoE residual read | keep; use gamma | keep |
| MoE router | `mlp.gate.weight` | `[E, H]` | Reads residual to produce router logits/top-k | right-transform | same |
| Routed experts | `mlp.experts.gate_up_proj` | `[E, 2I, H]` | Raw packed direct read for expert gate/up | right-transform on last axis | same |
| Routed experts | `mlp.experts.down_proj` | `[E, H, I]` | Raw packed per-expert residual-write contribution | keep | left-transform on H axis |
| Shared expert | `mlp.shared_expert.gate_proj.weight` | `[S, H]` | Direct residual read | right-transform | same |
| Shared expert | `mlp.shared_expert.up_proj.weight` | `[S, H]` | Direct residual read | right-transform | same |
| Shared expert | `mlp.shared_expert.down_proj.weight` | `[H, S]` | Shared residual-write contribution | keep | left-transform |
| Shared gate | `mlp.shared_expert_gate.weight` | `[1, H]` | Direct residual read producing a scalar sigmoid gate | right-transform | same |

Full attention constructs all three direct reads, splits query and output gate from the doubled `q_proj` output, gates the attention result, and only then applies `o_proj` ([Transformers L645-L719](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L645-L719)). Therefore the entire `q_proj` tensor is a reader; `q_norm` and `k_norm` operate after projection and are not residual-coordinate readers.

Linear attention calls all four `in_proj_*` matrices directly on normalized `hidden_states` ([Transformers L444-L470](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L444-L470)). Its convolution, delta rule, and inner gated norm operate on projected features; `out_proj` is the only linear-attention output returned for residual addition ([Transformers L472-L559](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L472-L559)).

The MoE forward sends the same flattened normalized residual independently to the shared expert, router, routed experts, and shared-expert gate ([Transformers L797-L816](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L797-L816)). The router’s actual operation is `F.linear(hidden_states, weight)` ([L778-L794](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L778-L794)). Each selected expert calls `F.linear` with one slice of packed gate/up, then one slice of packed down, scales by a routing scalar, and accumulates with `index_add_` ([L738-L775](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L738-L775)). The shared gate and router influence the residual write but emit scalars/indices, not residual-width vectors; they are readers, not writers.

### Internal tensors that must not be mistaken for residual-coordinate readers/writers

| Tensor | Shape | Why it is not transformed |
|---|---:|---|
| `self_attn.q_norm.weight`, `k_norm.weight` | `[Dh]` | Normalize already-projected head features. |
| `linear_attn.conv1d.weight` | `[2K+V, 1, C]` | Convolves projected Q/K/V channels. |
| `linear_attn.A_log`, `dt_bias` | `[Nv]` | Act on projected decay features. |
| `linear_attn.norm.weight` | `[Dv]` | Normalizes/gates delta-rule output before `out_proj`. |

The target configs set `attention_bias=false`, and all Qwen3.5-MoE MLP and linear-attention projections shown above are constructed biasless ([full attention L657-L668](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L657-L668), [linear attention L402-L447](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L402-L447), [MoE L722-L803](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L722-L803)).

## Concrete target shapes

Let `K = Nk Dk` and `V = Nv Dv`. The current official model configs establish the following values ([35B config L13-L80](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json#L13-L80), [122B config L13-L88](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/dc4d348443bc740c68e2d77492492c11606384d5/config.json#L13-L88)):

| Quantity | Qwen3.5-35B-A3B | Qwen3.5-122B-A10B |
|---|---:|---:|
| `H` / layers | 2048 / 40 | 3072 / 48 |
| `Nq / Nkv / Dh` | 16 / 2 / 256 | 32 / 2 / 256 |
| `Nk / Nv / Dk / Dv` | 16 / 32 / 128 / 128 | 16 / 64 / 128 / 128 |
| `K / V / (2K+V)` | 2048 / 4096 / 8192 | 2048 / 8192 / 12288 |
| `E / top-k` | 256 / 8 | 256 / 8 |
| `I / S` | 512 / 512 | 1024 / 1024 |
| Layer mix | 30 linear + 10 full | 36 linear + 12 full |

Full-attention layers are zero-based indices 3, 7, 11, and so on; the default interval generation is explicit in configuration source ([Transformers configuration L123-L130](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L123-L130)).

| Tensor | 35B-A3B | 122B-A10B |
|---|---:|---:|
| `self_attn.q_proj.weight` | `[8192, 2048]` | `[16384, 3072]` |
| `self_attn.k_proj.weight` | `[512, 2048]` | `[512, 3072]` |
| `self_attn.v_proj.weight` | `[512, 2048]` | `[512, 3072]` |
| `self_attn.o_proj.weight` | `[2048, 4096]` | `[3072, 8192]` |
| `linear_attn.in_proj_qkv.weight` | `[8192, 2048]` | `[12288, 3072]` |
| `linear_attn.in_proj_z.weight` | `[4096, 2048]` | `[8192, 3072]` |
| `linear_attn.in_proj_a.weight` / `in_proj_b.weight` | `[32, 2048]` | `[64, 3072]` |
| `linear_attn.conv1d.weight` | `[8192, 1, 4]` | `[12288, 1, 4]` |
| `linear_attn.out_proj.weight` | `[2048, 4096]` | `[3072, 8192]` |
| `mlp.gate.weight` | `[256, 2048]` | `[256, 3072]` |
| `mlp.experts.gate_up_proj` | `[256, 1024, 2048]` | `[256, 2048, 3072]` |
| `mlp.experts.down_proj` | `[256, 2048, 512]` | `[256, 3072, 1024]` |
| Shared `gate_proj.weight` / `up_proj.weight` | `[512, 2048]` | `[1024, 3072]` |
| Shared `down_proj.weight` | `[2048, 512]` | `[3072, 1024]` |
| `shared_expert_gate.weight` | `[1, 2048]` | `[1, 3072]` |

The packed shapes and raw parameter status are defined directly in the official expert class ([Transformers L738-L749](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L738-L749)). The official tensor-parallel/expert-parallel plans corroborate the exact paths ([configuration L59-L88](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L59-L88)).

Current checkpoint examples also prove that ordinary decoder experts are packed and use keys without `.weight` ([35B index L20-L97](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/model.safetensors.index.json#L20-L97), [122B index L8-L150](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/dc4d348443bc740c68e2d77492492c11606384d5/model.safetensors.index.json#L8-L150)).

## Transformation semantics

For a downstream block `m`, let `C_m` be the cumulative residual transform and let `Gamma_m` be the diagonal effective RMSNorm gain. J-Wash’s read transform is:

`R_m = Gamma_m C_m Gamma_m^-1`

and every direct reader must become:

`W_read' = W_read R_m`.

For packed gate/up:

`W_read'[e, :, :] = W_read[e, :, :] R_m` for every expert `e`.

Readthrough applies this to the token-mixer reader set, all five MoE reader families, and the final `lm_head.weight`. It does not transform a residual writer.

Exact mode adds:

`W_write' = C_m^-1 W_write`,

including each packed down slice:

`W_write'[e, :, :] = C_m^-1 W_write[e, :, :]`.

No extra transform is needed for routing weights or the shared-expert gate after their readers are corrected: those are scalars, so `C_m^-1` commutes with their multiplication and with the routed/shared summation.

### Live readthrough versus weight bake

J-Wash’s live read hook applies the low-rank `R_m` to the RMSNorm output, while `apply_read` folds the same transform into a reading matrix ([`core/ablation.py` L337-L346](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/ablation.py#L337-L346), [`core/rebase.py` L199-L210](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L199-L210)). In real arithmetic, if every direct MoE reader is included:

`W (R z) = (W R) z`.

Therefore router logits, top-k choices, routed gate/up, shared gate/up, and shared-gate values are identical before subsequent nonlinearities. Top-k does not break the proof because its logits are identical; finite-precision near-ties can still amplify rounding and must be tested.

The precise answer for the current repository is:

- **Current live and current bake are mutually equivalent only as the same incomplete implementation.** Neither discovers `post_attention_layernorm` for a sparse MoE block, because the only MLP entries are nonexistent top-level `mlp.gate_proj/up_proj` paths.
- **They are not equivalent to a complete MoE readthrough bake.**
- If the production inventory is extended correctly, the existing norm-deduplication logic can hook `post_attention_layernorm` once and will be mathematically equivalent to transforming all five MoE readers ([`core/ablation.py` L358-L366](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/ablation.py#L358-L366)).
- Exact live equivalence additionally requires a hook on the aggregate MoE output (or equivalent separate routed/shared output hooks).

This live-versus-bake equivalence is distinct from equivalence to a standard raw-residual hook. J-Wash explicitly approximates the RMS denominator as unchanged for the latter relationship ([`core/rebase.py` L28-L34](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L28-L34)).

## Support-check failure modes

The existing check is negative and structural only: it rejects `pre_feedforward_layernorm` and `post_feedforward_layernorm`, then `iter_reads`/`iter_writes` skip absent paths ([`core/rebase.py` L73-L116](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L73-L116)). It can falsely approve:

- Qwen3.5-MoE as currently implemented.
- A block with fused/renamed QKV or fused dense MLP projections.
- A block with no recognized MLP at all.
- A block with an additional residual-reading branch.
- An unknown token-mixer `block_type`.
- LayerNorm or another non-RMS normalization that happens to expose `weight`.
- A residual writer with bias in exact mode.
- A writer implemented as a raw parameter or functional operation.
- A future Transformers refactor that keeps the two marker attributes absent.

The full-export missing-key check is valuable but only proves that every *planned* key was found; it cannot detect a required key that the plan never included ([`core/editing.py` L517-L552](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L517-L552)).

A correct check should positively identify the supported block topology, require the complete mode-specific reader/writer set and expected hidden axes, validate the norm semantics, validate bias handling, and confirm planned state keys against the instantiated model. Unknown or partial blocks should fail closed.

## Export assessment for rank-3 expert tensors

### Full checkpoint

**Shape-wise: yes, after raw key handling is fixed.** `apply_read` and `apply_write` use `torch.matmul` in a way that broadcasts over the leading expert dimension ([`core/rebase.py` L207-L243](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/rebase.py#L207-L243)):

- Read: `[E,O,H] @ [H,r] -> [E,O,r]`, then `[E,O,r] @ [r,H] -> [E,O,H]`.
- Write: `[r,H] @ [E,H,I] -> [E,r,I]`, then `[H,r] @ [E,r,I] -> [E,H,I]`.

The full exporter replaces arbitrary-rank matched tensors and preserves all other keys ([`core/editing.py` L486-L552](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L486-L552)). `_disk_mapper` is prefix-only, so it also works for raw keys without `.weight`.

Operationally, full export still needs a target-size memory test. For 122B, one bf16 packed gate/up tensor is 3 GiB and its float32 form is 6 GiB. `bake` simultaneously creates float32 source/new tensors and a full-size delta reduction ([`core/editing.py` L384-L390](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L384-L390)). “One matrix at a time” is not automatically a safe peak.

### Modified layers

**Format-wise: yes; target-scale implementation-wise: no, not yet.** The saved tensor can remain rank 3, but all results remain resident until one `save_file` call. Packed tensor sizes are:

| Tensor per layer | 35B bf16 | 122B bf16 |
|---|---:|---:|
| `experts.gate_up_proj` | 1 GiB | 3 GiB |
| `experts.down_proj` | 0.5 GiB | 1.5 GiB |

For 47 downstream layers of 122B, that is 141 GiB readthrough or 211.5 GiB exact for packed tensors alone. The output must be sharded/streamed, with an index, or the product must impose a much narrower supported layer span and explicit RAM requirement.

### Raw-checkpoint layout compatibility

Current official ordinary decoder weights are packed. Transformers also contains a conversion mapping that merges older per-expert `gate_proj`/`up_proj` and stacks per-expert down tensors into the packed runtime representation ([`conversion_mapping.py` L1062-L1075](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/conversion_mapping.py#L1062-L1075), [L1835-L1836](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/conversion_mapping.py#L1835-L1836)). J-Wash export operates on raw checkpoint keys and does not run that conversion in reverse. Older revisions therefore need a separate policy.

## PEFT LoRA assessment

The existing LoRA export is incompatible with raw packed expert parameters.

For a packed read delta, J-Wash produces `B[E,O,r]` and `A[r,H]`. For a packed exact down delta it produces `B[H,r]` and `A[E,r,I]`. Current export:

- derives rank from `B.shape[1]`;
- pads as though both factors are 2-D;
- removes only a trailing `.weight`;
- lists every target under `target_modules`;
- writes ordinary module `lora_A.weight`/`lora_B.weight` keys.

Those assumptions are visible at [`core/editing.py` L437-L480](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L437-L480). For `gate_up_proj`, `B.shape[1]` is the output dimension `2I`, not rank `r`. There is also no module at the parameter path for PEFT to replace as an ordinary linear layer.

Current PEFT supports raw 2-D/3-D parameters through the separate `target_parameters` mechanism and explicitly cites fused MoE parameters as its use case ([PEFT `config.py` L521-L532](https://github.com/huggingface/peft/blob/ea8ebf36da98d93fdfca020ad17ab746e6994c01/src/peft/tuners/lora/config.py#L521-L532)). Its `ParamWrapper` packs per-expert factors into a distinct 2-D storage layout and reconstructs per-expert deltas by reshape/einsum ([PEFT `layer.py` L2236-L2246](https://github.com/huggingface/peft/blob/ea8ebf36da98d93fdfca020ad17ab746e6994c01/src/peft/tuners/lora/layer.py#L2236-L2246), [L2289-L2355](https://github.com/huggingface/peft/blob/ea8ebf36da98d93fdfca020ad17ab746e6994c01/src/peft/tuners/lora/layer.py#L2289-L2355), [L2418-L2435](https://github.com/huggingface/peft/blob/ea8ebf36da98d93fdfca020ad17ab746e6994c01/src/peft/tuners/lora/layer.py#L2418-L2435)).

Thus packed LoRA is possible in principle, but not through the existing serializer. The safe initial scope is to reject LoRA when any raw parameter target appears.

## MTP representation and safety

Both target configs declare one MTP layer and no dedicated MTP embedding ([35B config L65-L73](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json#L65-L73), [122B config L73-L81](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/dc4d348443bc740c68e2d77492492c11606384d5/config.json#L73-L81)).

The checkpoint stores MTP under a separate top-level `mtp.*` tree:

- `mtp.pre_fc_norm_embedding.weight`, `mtp.pre_fc_norm_hidden.weight`, `mtp.fc.weight`;
- `mtp.layers.0` with a full-attention block, router, shared expert, shared-expert gate, and norms;
- routed experts as 256 *unpacked* triples `mtp.layers.0.mlp.experts.e.{gate_proj,up_proj,down_proj}.weight`;
- `mtp.norm.weight`.

This differs from the ordinary decoder’s packed `experts.gate_up_proj`/`down_proj`. Representative exact keys appear in the current 35B index ([L345-L347 and L1226-L1447](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/model.safetensors.index.json#L345-L1447)) and 122B index ([L430-L432 and L1328-L1616](https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/dc4d348443bc740c68e2d77492492c11606384d5/model.safetensors.index.json#L430-L1616)).

Current Transformers does not instantiate an MTP module and explicitly ignores unexpected keys matching `^mtp.*` ([base class L898-L910](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L898-L910), [CausalLM L1794-L1807](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1794-L1807)). Ordinary forward uses only the text model and `lm_head` ([L1852-L1871](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1852-L1871)).

Full export copies all unmatched tensors unchanged ([`core/editing.py` L517-L528](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/core/editing.py#L517-L528)). Preserving MTP byte-for-byte is therefore safe for ordinary Transformers generation and avoids destructive loss. It is not a claim of correctness for a third-party runtime that actively uses MTP after the main decoder basis has been changed.

## Precise tiny-random-model test plan

No 35B or 122B weights or tokenizer are needed.

### Fixture

Instantiate `Qwen3_5MoeForCausalLM` directly on CPU in float32 from a local config:

~~~python
cfg = Qwen3_5MoeTextConfig(
    vocab_size=67,
    hidden_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    layer_types=["linear_attention", "full_attention", "linear_attention"],
    linear_conv_kernel_dim=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    moe_intermediate_size=16,
    shared_expert_intermediate_size=12,
    num_experts=4,
    num_experts_per_tok=2,
    tie_word_embeddings=False,
    use_cache=False,
    max_position_embeddings=64,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
)
cfg._attn_implementation = "eager"
cfg._experts_implementation = "eager"
model = Qwen3_5MoeForCausalLM(cfg).float().eval()
~~~

Transformers’ own tests already use a 32-hidden hybrid Qwen3.5-MoE configuration with explicit full/linear layer types and tiny expert settings ([official test L56-L69](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/tests/models/qwen3_5_moe/test_modeling_qwen3_5_moe.py#L56-L69), [L205-L232](https://github.com/huggingface/transformers/blob/3717b9cda226af7377da1944f395af0eac6e2b51/tests/models/qwen3_5_moe/test_modeling_qwen3_5_moe.py#L205-L232)).

Use a tiny namespace rather than a tokenizer or downloaded `jlens` model:

- `layers = model.model.layers`
- `_final_norm = model.model.norm`
- `_embed_tokens = model.model.embed_tokens`
- `_lm_head = model.lm_head`
- `_hf_model = model`
- layout `path="model"`, `embed="embed_tokens"`, `lm_head="lm_head"`

Use fixed input IDs, fixed seeds, and one normalized 32-vector rule at layer 0. Regenerate or explicitly set router weights until the top-k boundary has a nontrivial margin, so a routing mismatch indicates a semantic error rather than an exact floating-point tie.

### Inventory assertions

With a hook after layer 0, the literal readthrough plan must contain:

- Layer 1 full attention: q/k/v plus the five MoE reads = 8 keys.
- Layer 2 linear attention: four `in_proj_*` plus the five MoE reads = 9 keys.
- `lm_head.weight` = 1 key.

Total: 18 transforms. The raw packed keys must end exactly in `mlp.experts.gate_up_proj`.

Exact must add three writers for each downstream layer:

- `self_attn.o_proj.weight` or `linear_attn.out_proj.weight`;
- `mlp.experts.down_proj`;
- `mlp.shared_expert.down_proj.weight`.

Total exact plan: 24 transforms.

Assert this set from a test-owned literal list, not by reusing `READS`/`WRITES`. The current Llama test lets live and bake share the same incomplete iterator and therefore cannot catch a mutually omitted branch ([`scripts/test_rebase.py` L23-L24 and L72-L144](https://github.com/erm14254/J-Wash/blob/73a9f2802e7110b794298d7445b93cdddb8daa6d/scripts/test_rebase.py#L23-L144)).

### Algebra and forward assertions

1. Apply each packed read/write transform and assert the result keeps its exact rank-3 shape.
2. Compare batched transformation to an explicit loop over all four expert slices.
3. Build an independent live oracle that manually hooks both downstream `input_layernorm` and `post_attention_layernorm` once plus final norm; do not discover hook sites through the production iterator.
4. Compare oracle, production live, and baked logits with tight float32 tolerances.
5. Capture and compare every layer’s router logits, selected expert IDs, routing weights, routed gate/up outputs, shared-expert gate, and aggregate MoE output.
6. For exact, counter-transform the complete live token-mixer output and complete `mlp` output. Compare against the bake.
7. Run a soft scale case for exact and a saturated removal/replacement case for readthrough.
8. Re-run with `use_cache=True` for a short prefill and one-token decode.

### Support and export assertions

1. Mutate a synthetic block to remove each required reader/writer in turn; capability detection must fail for the relevant mode.
2. Add an unknown residual branch; fail closed.
3. Add a writer bias; reject exact unless bias transformation is implemented.
4. Save the tiny base with safetensors and very small shards. Full export, reload locally, and compare logits to the in-memory bake.
5. Inspect full and modified-layers tensors: exact expected keys, no `.weight` on packed parameters, rank 3 preserved, original dtype preserved.
6. Create a synthetic disk-prefix fixture using `model.language_model.*` to exercise `_disk_mapper` independently of the in-memory prefix.
7. Remove one targeted raw key and assert full export aborts through its missing-key guard.
8. Add representative `mtp.*` sentinels and assert full export copies them bit-for-bit; modified-layers should omit them because it is an overlay.
9. Until raw-parameter LoRA is implemented, assert `fmt="lora"` raises a clear error before creating adapter files. If implemented later, require both PEFT runtime-logit equality and `merge_and_unload` equality; inspecting filenames is insufficient.
10. Add a dry-run memory estimator using the two official configurations and test that the selected modified-layers writer shards rather than accumulating all target tensors.

## Final recommendation

Proceed only with a descriptor-based, fail-closed implementation. The minimal safe feature is:

1. complete readthrough coverage;
2. separate exact capability and aggregate-MoE live hook;
3. correct raw packed keys and rank-3 bake;
4. full-checkpoint export plus sharded modified-layers export;
5. explicit LoRA rejection for raw parameters;
6. the independent tiny-random test suite above.

MTP rebasing and packed PEFT adapters can then remain separate follow-up work without weakening ordinary-generation correctness.
