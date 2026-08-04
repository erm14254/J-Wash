# PR #9 — Capability-aware intervention and export UX

## 1. Executive summary

This audit covers the current **master** branch of **erm14254/J-Wash** at commit **edd012e69f1974ccfdb651f0ceabcef60cd1313c** (“Migrate startup lifecycle and safely inspect abandoned export artifacts (#8)”), fetched on 2026-08-04. The checkout matched origin/master when inspected. No repository code was changed.

PR #9 should be a narrow capability-contract and UX integration PR, but it cannot be frontend-only. The backend already computes model-wide readthrough and exact-mode decisions during model load and publishes them in manager metadata. The React editor ignores the newer fields, consumes only legacy rebase_supported, infers global projection from every readthrough failure, and offers full, modified-layers, and LoRA exports for every non-standard mode. GGUF visibility is based only on whether a llama.cpp path string is nonempty.

The minimum safe design is:

1. Compute and cache intrinsic model facts once at load: standard-edit eligibility, readthrough, exact, positively validated global projection, packed-parameter presence, quantized-load policy, and tied-embedding notes.
2. Add one structured, mode-aware capability snapshot to /api/status. Combine cached model facts with cheap current checks for source safetensors and llama.cpp tools.
3. Keep the existing flat rebase_supported, readthrough_supported/reason, and exact_supported/reason fields for compatibility, deriving them from the same decisions.
4. Make mode PATCH, regular export, GGUF preparation, live attach, and deep export execution enforce the same policy. Frontend disabling remains advisory.
5. Replace the editor’s Boolean fallback with explicit mode and format decisions, visible reasons, deterministic selection reset, and model-session cleanup.
6. Add a focused frontend CI job and behavioral component tests without expanding PR #9 into a broad UI or dependency rewrite.

Five backend alignment fixes are required for the UI metadata to be truthful:

- Unknown architectures must not be treated as global-projection capable merely because readthrough failed.
- Packed-MoE modified-layers and LoRA rejection must be model-wide, not dependent on which rule layers happen to produce packed transform targets.
- Quantized loads must fail closed for editing and current-edit export until deliberately validated; this matches the README’s stated policy.
- Interventions, mode, scale, lens state, and editor session state must not survive a model unload or replacement.
- A cached GGUF HF bake must be matched to the current edit before reuse; config.json alone is not an edit identity.

The intended packed Qwen3.5-MoE Phase 1 boundary is substantially verified: live readthrough and full-checkpoint export are implemented; exact, packed modified-layers, and conventional LoRA are rejected. GGUF is available only through a valid full bake plus a usable converter, with quantized GGUF types additionally requiring llama-quantize. These are intentional Phase 1 limits, not missing frontend affordances.

Validation performed during this audit:

- npm ci and npm run build completed successfully on Node 24.14.0 / npm 11.9.0; Vite transformed 603 modules.
- The Python suite was inspected but not executed because the audit runtime did not contain pytest, torch, Transformers, Accelerate, FastAPI, or safetensors. Existing CI remains the execution authority for those tests.

## 2. Verified current state

### 2.1 Current capability computation and status contract

**Verified current behavior**

- **core/model_manager.py::_rebase_capability_meta** at lines 439–458 is the public model-capability builder.
- It calls **core/rebase.py::model_preflight** twice: once with exact=False and once with exact=True.
- It runs inside **ModelManager.load** after AutoModelForCausalLM loading, tokenizer setup, model eval, and jlens.from_hf, but before manager.meta is finalized (core/model_manager.py:473–541).
- manager.meta receives:
  - model_id, revision, dtype, quant, device, n_layers, and d_model;
  - rebase_supported;
  - readthrough_supported and readthrough_reason;
  - exact_supported and exact_reason;
  - chat-template source/fallback and load time.
- rebase_supported is intentionally the legacy alias of readthrough_supported.
- **api/app.py::api_status** at lines 228–243 exposes manager.meta unchanged as loaded, plus busy, lens, GPU statistics, downloads, conversion state, fit state, GGUF worker state, intervention rules, intervention scale, intervention mode, and last generation.
- No structured export-format capability, global-projection capability, source-readiness capability, quantized-edit capability, or llama.cpp installation capability is exposed.

**Relevant topology logic**

**core/rebase.py::block_capabilities** (lines 250–330) positively validates a complete topology and fails closed:

- Exactly one token mixer must exist: self_attn or linear_attn.
- Every expected residual reader and its hidden axis must match.
- The read RMSNorms must be exact audited Transformers classes and pass structural and semantic probes.
- pre_feedforward_layernorm or post_feedforward_layernorm rejects readthrough.
- Dense exact additionally requires a complete writer inventory, rank-2 output axes, no bias, exact torch.nn.Linear type, hookability, and tensor-returning behavior.
- Packed sparse/MoE blocks can pass readthrough but always fail exact because packed/shared residual-writer support and an aggregate-MoE live oracle do not exist.

**core/rebase.py::model_preflight** (lines 333–348) applies this to every decoder block and the final norm. The Qwen3.5-MoE read inventory is explicit in _MOE_READS (lines 121–127), including the raw rank-3 mlp.experts.gate_up_proj parameter without inventing a .weight suffix.

Norm inspection copies only a small gain tensor, including through supported Accelerate weights_map dispatch (core/rebase.py:144–246). Capability scanning is performed at load, not on each status poll. The current implementation nevertheless scans the model twice because exact preflight repeats readthrough validation.

### 2.2 Current intervention-mode behavior

**Verified current behavior**

- Legal names are standard, readthrough, exact, and abliteration (core/ablation.py:60–75).
- **PATCH /api/interventions** checks readthrough_supported or exact_supported only when the requested mode is readthrough or exact (api/app.py:444–462).
- The lookup falls back to legacy rebase_supported when a newer field is absent.
- Abliteration is never checked against a positive capability.
- If no model is loaded, manager.meta is None and the model-dependent check is skipped, so a pure mode can be stored without a model.
- Scale is applied before mode validation; a request containing scale plus an unsupported mode can return 422 after mutating scale.
- Live readthrough/exact remains fail-closed because **Interventions._attach_rebase** reruns model_preflight before registering hooks (core/ablation.py:322–390).
- Live global projection has no corresponding preflight. **Interventions._attach_abliteration** applies the transformation to embeddings and every block output without proving that the bake handles the same residual writer inventory (core/ablation.py:289–320).
- Interventions.add rejects only when lm_head.weight has a non-floating dtype (core/ablation.py:170–172); it does not use model_meta.quant.

### 2.3 Current export behavior

**Regular export**

**api/app.py::api_edit_export** (lines 536–581):

- requires a loaded model and at least one active rule;
- accepts only full, layers, and lora;
- routes readthrough/exact to editing.export_rebase and abliteration to editing.export_abliteration;
- rejects standard mode;
- maps ValueError to HTTP 422 and retains deep execution checks.

**Readthrough/exact**

**core/editing.py::export_rebase** and **_export_rebase_impl** (lines 1551–1881):

- publish through a unique sibling staging directory and atomic rename;
- call rebase.build_plan, which reruns model_preflight;
- reject packed plan targets for layers and LoRA at lines 1603–1614;
- require a local safetensors source for full export;
- map in-memory and disk checkpoint prefixes, validate required keys, preserve unmatched tensors and MTP with a warning, and verify indexed checkpoints;
- support layers and LoRA from the loaded state dict;
- detect tied embeddings using data_ptr, untie lm_head for full/layers, and keep readthrough LoRA runtime-safe with a warning not to merge it into shared base weights.

**Global projection**

**core/editing.py::compute_abliteration** and **export_abliteration** (lines 1077–1357):

- have no positive architecture preflight;
- hard-code the bake inventory to self_attn.o_proj and mlp.down_proj (TARGET_SUFFIXES at lines 1040–1041);
- silently count missing writer modules and later emit a partial-bake warning;
- do not handle linear_attn.out_proj or packed routed/shared expert writers;
- implement full, layers, and LoRA;
- omit the embedding from a tied global-projection LoRA and warn that fidelity is reduced.

There are no dedicated global-projection capability or export tests.

### 2.4 Current GGUF behavior

**Verified current behavior**

- The only setting is a raw llamacpp_dir string (api/app.py:279–324).
- **_llamacpp_paths** (api/app.py:600–622) checks that the configured directory exists, that convert_hf_to_gguf.py exists, and searches the root, bin, and build/bin for llama-quantize or llama-quantize.exe.
- The checks use exists(), not regular-file/readability/executability checks.
- bf16/f16 need the converter; q8_0, q6_k, q5_k_m, q4_k_m, and q3_k_m additionally need a quantizer.
- **api_edit_export_gguf** (api/app.py:704–761) bakes fmt=full unless data/edits/<name>/hf/config.json already exists, then launches the worker.
- On cache reuse, config.json alone is accepted; current model, revision, mode, rules, scale, and edit_meta.json are not compared.
- A cached checkpoint can therefore be converted with no model loaded.
- /api/status.gguf is worker progress only; it is not installation or model capability metadata.

### 2.5 Current frontend behavior

**ui/src/App.jsx**

- App polls /api/status initially and every two seconds (lines 284–304).
- In API-monitor mode it stops polling and refreshes only after an api_generation WebSocket message. The server emits that message only after /api/generate (api/app.py:832–838).
- App copies intervention rules, scale, and mode from status into React state (lines 379–384).
- It passes only status.loaded.rebase_supported to Editor (lines 1784–1803).
- readthrough_supported/reason and exact_supported/reason are not consumed.
- Settings are fetched once, and Editor receives llamaCppSet as the Boolean value of settings.llamacpp_dir.
- Load and unload handlers do not immediately refresh status.
- Editor is always mounted and is not keyed to a model instance; open only controls whether Editor returns markup.

**ui/src/Editor.jsx**

- ModeToggle is a two-position standard-versus-pure control (lines 177–213).
- Every non-standard mode is displayed as the same pure side. Exact is intentionally API-only, so an API-selected exact mode is mislabeled.
- pureMode is derived as abliteration when rebaseSupported is exactly false; otherwise readthrough. The prop defaults to true (lines 258–265).
- This maps no metadata and every unknown readthrough failure to an optimistic mode.
- exportFmt is local state initialized to full and is reset only when a llama.cpp path disappears while GGUF is selected (lines 521–528).
- Full, layers, and LoRA options are always present for every non-standard mode; GGUF appears for any nonempty path string (lines 741–803).
- All GGUF types are always shown, with q4_k_m as the default even when no quantizer exists.
- Static MODE_INFO copy claims all three non-GGUF formats for readthrough, exact, and global projection (lines 215–256).
- The non-GGUF Export button checks rules.length, not whether any rule is enabled and has layers.
- The disabled section relies partly on pointer-events:none; descendant keyboard focus is not reliably disabled by that CSS alone (ui/src/styles.css:126–133).
- Labels are visually adjacent but not associated with controls through htmlFor/id, disabled choices have no associated reason, important explanations rely on title, and the close glyph lacks an accessible label.

**Bundled non-React client**

- **scripts/jwash_mcp.py::_add_rule** (lines 148–173) duplicates the legacy fallback: rebase_supported false means abliteration, otherwise readthrough.

### 2.6 Existing tests, CI, and documentation

**Python coverage already present**

- Dense positive exact and fail-closed reader/norm cases: tests/test_capabilities.py:34–53.
- Packed model-wide exact rejection before hook registration: tests/test_capabilities.py:69–79.
- Flat legacy and newer capability fields: tests/test_capabilities.py:82–92.
- Adversarial norm and writer contracts: tests/test_capabilities.py:172–223.
- Accelerate offload inspection: tests/test_accelerate_offload.py.
- Packed full success and layers/LoRA/exact rejection: tests/test_transactional_exports.py:50–93.
- Dense full/layers/LoRA success: tests/test_transactional_exports.py:204–225.
- GGUF preparation, worker, atomicity, and retry behavior: tests/test_gguf_workers.py.
- Tied indexed-export rollback: tests/test_indexed_checkpoints.py:133–161.

No test verifies a structured /api/status capability snapshot, positive global projection, quantized reporting, reporting/enforcement parity, settings-driven GGUF capability changes, or unload/replacement cleanup.

**Frontend infrastructure**

- ui/package.json has dev, build, and preview only; no test or lint command.
- There are no React test files or frontend test dependencies.
- A Python source-string assertion in tests/test_fitting.py is not behavioral React coverage.

**CI**

- .github/workflows/ci.yml has independent j-wash and jacobian-lens Python matrix jobs across Linux/Windows, Python 3.11/3.12, and Transformers 5.5.0/5.14.1.
- It has no Node setup, npm install, frontend tests, or frontend build.

**Documentation**

- README.md:196–214 already contains a useful dense/packed/unknown read-projection and export support matrix.
- docs/qwen3_5_moe_readthrough.md:31–57 already documents packed full-checkpoint support and the intentional exact, layers, and LoRA limitations.
- README.md:20–25, 62–67, and 317–333 still make broad export/fidelity statements that require qualification.
- README.md:74–77 says Node.js 18+, but the committed marked 18.0.6 package requires Node >=20 (ui/package-lock.json:1728–1737).
- Dockerfile:15–16 installs EOL Node 20.

### 2.7 Assumptions in the request that are outdated or incomplete

1. The structured readthrough/exact metadata is not merely apparent; it is present on current master and exposed under /api/status.loaded.
2. The React editor does still consume only the legacy Boolean, so that concern is current.
3. The README support matrix and packed Qwen3.5-MoE technical documentation already exist. PR #9 should refine them, not create a new architecture document.
4. “The backend rejects unsupported operations” is only partly true. Readthrough/exact and packed plan targets are guarded, but global projection lacks positive validation, quantized policy is incidental, and packed layers/LoRA rejection is active-plan-dependent.
5. A configured llama.cpp path is not a validated installation. The frontend currently treats any nonempty string as availability.
6. Packed full-checkpoint support is conditional on a valid local safetensors source and complete disk-key mapping; it is not an unconditional model-family promise.
7. GGUF cache reuse is name-specific and can operate without a model. It must not make the generic “export the current edit” capability optimistic.
8. README’s Node 18 minimum is incompatible with the committed direct dependency set.

## 3. Findings ranked by severity

### High 1 — Readthrough failure is incorrectly treated as proof of global-projection support

**Verified current behavior:** Editor.jsx derives abliteration from rebase_supported === false. scripts/jwash_mcp.py does the same. The backend mode PATCH does not validate abliteration. The bake handles only embeddings, self_attn.o_proj, and mlp.down_proj, skipping missing writers.

**Inference:** An unknown, hybrid linear-attention, packed-MoE, modified, or incomplete architecture can be presented as faithfully exportable through global projection even though the live and baked writer inventories differ.

**Recommendation:** Add a positive global_projection_preflight shared by metadata, mode PATCH, live attach, and export. It must validate exactly the inventory the live and bake paths handle. Unknown, packed, hybrid, extra-writer, missing-writer, wrong-axis, biased, and non-floating layouts fail closed. Never define global support as not readthrough.

### High 2 — Model unload/replacement retains old rules, mode, scale, and editor work

**Verified current behavior:** ModelManager.load unloads only the old model. api_unload unloads the lens, neighbors, and model but not Interventions. Editor remains mounted and keeps export format/name, selected rules, pending debounced PATCH timers, and scale timer. Load/unload handlers do not immediately refresh status.

**Inference:** Old-dimensional directions can cross a model boundary, a stale unsupported mode can remain selected, delayed requests can mutate the new session, and UI capability state can stay stale indefinitely in API-monitor mode.

**Recommendation:** Add Interventions.reset(), use one lifecycle reset on unload/replacement/failure, restore mode=standard and scale=1.0, and issue a model_session_id for every successful load. Key/remount Editor by that ID, close it when model/lens disappears, clear prefill, and cancel pending timers on session cleanup.

### High 3 — The UI ignores authoritative fields and knowingly offers invalid packed exports

**Verified current behavior:** App passes only rebase_supported. Editor never reads readthrough/exact reasons, has no exact choice, and always offers full/layers/LoRA outside standard mode. Packed layers/LoRA then fail in editing._export_rebase_impl.

**Inference:** Users discover intentional Phase 1 boundaries only after an expensive or confusing submission failure, and exact selected through an API can be mislabeled.

**Recommendation:** Render explicit backend decisions for standard/readthrough/exact/global and full/layers/LoRA/GGUF. Missing metadata is unsupported. Show concise reasons adjacent to controls, reset invalid selections deterministically, and retain backend checks.

### High 4 — Packed layers/LoRA rejection is rule-plan-dependent

**Verified current behavior:** _export_rebase_impl examines only info.targets produced by the current build plan for packed parameters.

**Inference from the exact control flow:** A final-layer-only rule produces only the final lm_head transform. info.targets is then empty, so the current model-wide documented prohibition can be bypassed.

**Recommendation:** Cache a model-wide has_packed_read_parameters fact during load and reject layers/LoRA for packed topology regardless of selected rule layers. Add first-, middle-, and final-layer tests. This is enforcement parity, not Phase 2 implementation.

### High 5 — GGUF cache reuse can silently convert a stale edit

**Verified current behavior:** <name>/hf/config.json alone selects the reuse branch. Current model, revision, mode, rule set, scale, and edit metadata are ignored.

**Inference:** Reusing a name after changing the model or rules can produce a GGUF from an older bake while the editor says it is exporting the current edit.

**Recommendation:** Store a canonical edit fingerprint in the HF cache metadata and compare it before reuse. Permit same-edit multi-type conversion; reject mismatches before worker creation. Keep conversion of an explicitly existing cache as a name-specific API operation, not generic current-edit capability.

### High 6 — Quantized editing policy is contradictory and unproven

**Verified current behavior:** README.md:441 says interventions and lens readouts are unavailable on int8/nf4 weights. Capability computation ignores model_meta.quant. Interventions.add checks only lm_head dtype, and BitsAndBytes can leave heads floating. Exact can fail incidentally on writer type while readthrough may still be reported true.

**Inference:** The UI and API can make inconsistent claims depending on incidental module dtypes rather than declared load mode.

**Recommendation:** Until a tested quantized edit path exists, quant != null makes all intervention modes and current-edit formats unsupported with reason_code quantized_model_unsupported. Enforce the same rule in API submission. This follows existing documentation and does not add a feature.

### Medium 7 — Status freshness is insufficient for capability-sensitive state

**Verified current behavior:** API-monitor mode refreshes only after api_generation. It does not refresh on load, unload, settings changes, mode changes, or GGUF transitions. Fetch errors silently preserve the last status, and requests are not sequenced.

**Inference:** Capability choices and GGUF progress can remain actionable but stale, and a late response can overwrite a newer snapshot.

**Recommendation:** Hoist a sequenced refreshStatus(), call it after capability-relevant mutations, track freshness, and disable capability-sensitive actions while status is unknown/stale. Broadcast a generic status_changed event or retain a low-frequency safety poll in API-monitor mode. Poll while GGUF is running.

### Medium 8 — GGUF installation readiness is optimistic and subtype-blind

**Verified current behavior:** Any nonempty settings path shows GGUF and every q-type. The backend checks the directory/converter only on submission and requires a quantizer only for q-types.

**Inference:** A converter-only installation defaults the user to an invalid q4_k_m request even though bf16/f16 could work.

**Recommendation:** Derive GGUF from full-export eligibility plus a shared llama.cpp probe. Report each type separately. If the quantizer disappears, reset a q-type to bf16/f16; if the converter/full bake becomes unavailable, reset the format away from GGUF.

### Medium 9 — Source, tied-weight, and reason reporting can disagree with execution

**Verified current behavior:** Full export requires local safetensors, but model listing can accept other layouts. Tied detection uses data_ptr at execution. Capability reasons are raw exception strings.

**Inference:** A loadable .bin-only or removed source can be advertised as full-exportable; distinct meta-device parameters can both have pointer zero; low-level offload/probe details can leak into UI text.

**Recommendation:** Cheaply recheck source safetensors in the status composer and again at submission. Centralize tied detection using shared Parameter identity first, with configuration/disk evidence where necessary. Publish stable reason_code plus curated reason and log deeper diagnostic details server-side.

### Medium 10 — Mode PATCH is not transactional

**Verified current behavior:** Scale changes before unsupported mode validation.

**Inference:** A failed UI action can still alter generation behavior.

**Recommendation:** Validate the requested mode before applying either field, then mutate scale/mode under one lock or reject without state change. Add a regression test.

### Medium 11 — Frontend behavior is not built or tested in CI

**Verified current behavior:** CI has no Node job; package.json has no test command. The current production build does succeed locally.

**Inference:** JSX/build breakage and capability-state regressions can merge despite the Python matrix passing.

**Recommendation:** Add one independent Ubuntu/Node 24 frontend job with npm cache, npm ci, focused component tests, and npm run build. Do not multiply it across the Python matrix or make npm audit a PR #9 gate.

## 4. Capability model and terminology

### 4.1 Separate intrinsic facts, current readiness, and execution validation

Use three layers, all driven by the same policy functions:

1. **Intrinsic model profile — computed once at load**
   - float/edit eligibility;
   - readthrough decision;
   - exact decision;
   - positively validated global-projection decision;
   - model-wide packed-reader fact;
   - tied-embedding fact and warnings;
   - quantized-load policy.

2. **Current capability snapshot — assembled for /api/status**
   - cached intrinsic profile;
   - current intervention mode;
   - source-folder and safetensors presence;
   - current llama.cpp converter/quantizer readiness;
   - per-format and per-GGUF-type decisions.

3. **Request/execution preflight — run on every mutation**
   - current model/session still matches;
   - active rules, neutral factors, name validity, destination collision, busy state;
   - deep source index/key validation;
   - actual model preflight before hooks/transforms;
   - converter/quantizer recheck and subprocess result.

The status snapshot is an authoritative policy snapshot, not a guarantee against later filesystem races, state changes, OOM, disk failure, or converter failure. Submission remains fail-closed.

### 4.2 Smallest public contract

Add one top-level capabilities object to /api/status. Keep manager.meta’s existing flat fields for compatibility. Cache intrinsic facts on ModelManager rather than permanently storing current mode/settings-derived format decisions in manager.meta.

Recommended shape:

    {
      "loaded": {
        "model_id": "Qwen/...",
        "model_session_id": "opaque-load-id",
        "rebase_supported": true,
        "readthrough_supported": true,
        "readthrough_reason": "Supported.",
        "exact_supported": false,
        "exact_reason": "Exact mode is not implemented for packed MoE writers."
      },
      "capabilities": {
        "contract_version": 1,
        "model_session_id": "opaque-load-id",
        "intervention_modes": {
          "standard": {
            "supported": true,
            "reason_code": "supported",
            "reason": "Supported."
          },
          "readthrough": {
            "supported": true,
            "reason_code": "supported",
            "reason": "Supported."
          },
          "exact": {
            "supported": false,
            "reason_code": "packed_moe_exact_unimplemented",
            "reason": "Exact mode is intentionally unavailable for packed MoE in Phase 1."
          },
          "abliteration": {
            "supported": false,
            "reason_code": "global_projection_topology_unsupported",
            "reason": "This architecture has not passed global-projection validation."
          }
        },
        "exports": {
          "for_mode": "readthrough",
          "formats": {
            "full": {
              "supported": true,
              "reason_code": "supported",
              "reason": "Supported."
            },
            "layers": {
              "supported": false,
              "reason_code": "packed_moe_layers_unimplemented",
              "reason": "Packed-MoE modified-layers export is not implemented."
            },
            "lora": {
              "supported": false,
              "reason_code": "packed_moe_lora_unimplemented",
              "reason": "Conventional LoRA cannot represent packed parameters."
            },
            "gguf": {
              "supported": true,
              "reason_code": "supported",
              "reason": "A full bake and converter are available.",
              "types": {
                "bf16": {"supported": true, "reason_code": "supported"},
                "f16": {"supported": true, "reason_code": "supported"},
                "q4_k_m": {
                  "supported": false,
                  "reason_code": "llamacpp_quantizer_missing",
                  "reason": "Install llama-quantize; bf16 and f16 remain available."
                }
              }
            }
          }
        }
      }
    }

Return the same fail-closed object with no_model reasons when no model is loaded; do not omit fields and invite optimistic frontend defaults.

### 4.3 Boolean versus a tri-state

Use **supported: true/false**, not a generic supported/unsupported/conditionally_available enum.

Rationale:

- The UI’s submission decision is binary: the option is enabled now or it is not.
- “Configure llama.cpp,” “restore source safetensors,” and “install the quantizer” are actionable false states identified by stable reason codes and curated messages.
- A generic conditional state leaves clients to decide whether it is selectable, recreating policy duplication.
- Partial GGUF readiness is represented exactly through the types map: the format can be supported while q-types are false.
- Supported formats may carry warnings without becoming unsupported, for example tied LoRA merge semantics.

If a future API needs remediation categories, add a non-authoritative remediation field; do not change enablement semantics.

### 4.4 Stable reason codes

Use codes for branching/tests and human reasons for display. A compact initial set is sufficient:

| Area | Reason codes |
|---|---|
| General | supported, no_model, capability_data_unavailable, quantized_model_unsupported |
| Modes | readthrough_topology_unsupported, exact_topology_unsupported, packed_moe_exact_unimplemented, global_projection_topology_unsupported |
| Exports | mode_not_exportable, source_checkpoint_unavailable, source_safetensors_missing, packed_moe_layers_unimplemented, packed_moe_lora_unimplemented |
| GGUF | full_export_required, llamacpp_not_configured, llamacpp_directory_invalid, llamacpp_converter_missing, llamacpp_quantizer_missing |
| Warnings | tied_lora_do_not_merge, tied_global_lora_reduced_fidelity, mtp_preserved_untransformed |

Do not expose raw exception text as capability reason. Tests should assert codes and broad semantics, not complete prose.

### 4.5 Backward compatibility

- Retain rebase_supported as the exact alias of structured readthrough.supported.
- Retain readthrough_supported/reason and exact_supported/reason.
- Sanitize legacy reason text to the same public reason; log detailed diagnostics separately.
- Additive /api/status fields do not break existing clients.
- Do not add a Pydantic response model for the entire status payload; modeling every existing dynamic status branch would be scope creep. A small internal dataclass or typed helper for decisions is enough.
- Update scripts/jwash_mcp.py to use structured mode decisions, but keep server rejection authoritative for older clients.

## 5. Expected model/mode/export matrix

The format columns below mean “eligible for a fresh export in the supported pure mode,” before request-local conditions such as active rules, a valid name, an unused destination, sufficient disk/RAM, and a later filesystem race. Full and GGUF also require source/runtime conditions shown separately.

| Model state | Standard live | Readthrough | Exact | Global projection | Full | Layers | LoRA |
|---|---|---|---|---|---|---|---|
| No model | No | No | No | No | No | No | No |
| Audited dense RMSNorm, unquantized | Yes | Yes | Yes when writer preflight passes | Only if its own positive preflight passes; not inferred or preferred | Yes, with local safetensors | Yes | Yes |
| Packed Qwen3.5-MoE Phase 1 | Yes | Yes | No — intentional packed-writer boundary | No — current bake omits packed writers | Yes, with valid source | No, model-wide | No, model-wide |
| Positively validated write-norm fallback | Yes | No | No | Yes | Yes, with valid source | Yes if the validated inventory is fully represented | Yes, with tied warning where applicable |
| Unknown/unaudited architecture | Only if a minimal float/hookability live preflight passes | No | No | No | No | No | No |
| int8/nf4 loaded model | No under current documented policy | No | No | No | No current-edit bake | No | No |
| Dense model with tied embeddings | Same as dense mode support | Yes | As preflighted | As preflighted | Yes; full/layers untie lm_head | Yes; unties lm_head | Readthrough LoRA supported at runtime with do-not-merge warning; global LoRA has reduced-fidelity warning |

### Runtime/source matrix for GGUF

| State | Generic current-edit GGUF | Type behavior |
|---|---|---|
| Full export unsupported for model/mode | Disabled: full_export_required | All types disabled |
| Source missing or no safetensors | Disabled: source_checkpoint_unavailable/source_safetensors_missing | All types disabled |
| llama.cpp not configured | Disabled: llamacpp_not_configured | All types disabled |
| Configured path is not a directory | Disabled: llamacpp_directory_invalid | All types disabled |
| Directory exists, converter missing | Disabled: llamacpp_converter_missing | All types disabled |
| Converter present, quantizer missing | Enabled | bf16/f16 enabled; q-types disabled with llamacpp_quantizer_missing |
| Converter and usable quantizer present | Enabled | All listed types enabled |
| Existing named HF cache, no model | Not generic editor capability | Explicit name-specific conversion may remain available through API/CLI after cache validation |

### Transition behavior

| Transition | Required behavior |
|---|---|
| Model unload | Reset backend interventions/mode/scale, remove capability identity, close or empty the editor, cancel pending UI patches, and expose no enabled modes/formats |
| Model replacement | New model_session_id; old lens/rules cannot cross; recompute intrinsic facts; remount editor |
| Current format remains supported | Preserve it |
| Current format becomes unsupported | Prefer full if supported, otherwise first supported format in a fixed order, otherwise no selection; announce the reset |
| q-type becomes unsupported but GGUF remains supported | Select bf16, then f16, then the first supported type |
| Capability status becomes stale/unavailable | Disable submission and show a concise reconnecting/capability-unavailable explanation |

## 6. Recommended backend design

### 6.1 Compute intrinsic facts once

Add a small **core/capabilities.py** policy module, not a framework. Keep topology-specific inspection close to the code it describes:

- Extend **core/rebase.py** with a one-pass model profile that returns readthrough, exact, and has_packed_read_parameters. Avoid running the same semantic norm probes twice.
- Add **global_projection_preflight(jl)** in **core/ablation.py** or core/capabilities.py, using one shared target inventory consumed by both Interventions._attach_abliteration and editing.compute_abliteration.
- Validate embedding/head presence, floating dtype, rank, hidden axes, writer presence, bias semantics, and unexpected extra residual writers. Reject linear-attention and packed writer layouts until implemented.
- Use declared model_meta.quant as an explicit fail-closed input.
- Centralize tied detection. Prefer object/Parameter identity; do not infer tying solely from data_ptr on meta tensors.
- Store the intrinsic profile on ModelManager, clear it in _unload_locked, and issue model_session_id for each successful load.
- Derive the legacy fields from this profile.

This profile may inspect shapes/classes and run the existing small norm probes during load. It must not move full projection tensors, iterate state_dict, or materialize the model on every status request.

### 6.2 Compose current status cheaply

Add **capability_snapshot(manager, interventions.mode, settings)**:

- Return a complete no_model snapshot when unloaded.
- Combine cached intrinsic facts with current mode.
- Resolve the source directory and check only directory/safetensors presence for readiness. Leave index/key completeness to execution.
- Probe llama.cpp with cheap filesystem checks only.
- Report converter and quantizer separately and validate regular-file/readability/executability as appropriate to how each is invoked.
- Include for_mode so the UI cannot accidentally use an export map for the previous mode.
- Never place settings-aware GGUF decisions permanently in manager.meta.

Repeated /api/status calls must not rerun rebase.model_preflight, semantic norm probes, state_dict traversal, tensor movement, or Accelerate weights_map inspection.

### 6.3 Reuse policy in all mutations

**api/app.py::api_interventions_scale**

- Validate the entire requested update before mutating scale or mode.
- Enforce all four explicit mode decisions, including global projection and no-model cases.
- Return the updated mode/scale and, additively, the fresh capabilities snapshot for immediate UI reconciliation.

**api/app.py::api_edit_export**

- Check the current mode/format decision before starting export work.
- Preserve existing active-rule, name, busy, and deep export validation.
- Keep ValueError-to-422 behavior.

**core/ablation.py / core/editing.py**

- Global live attach and export must call the same positive preflight.
- Keep rebase.model_preflight and build_plan execution checks even when metadata said supported.
- Make packed layers/LoRA rejection use the cached/model-wide fact or a model-wide helper, never active-plan targets alone.

**api/app.py::api_edit_export_gguf**

- Use the same llama.cpp probe as status.
- Require current full-export support for a new bake.
- Preserve per-type quantizer enforcement.
- Recheck tools immediately before worker creation.
- Do not use frontend metadata as authorization.

### 6.4 Make model lifecycle a capability boundary

Add **Interventions.reset()**:

- detach any handles;
- clear rules and pending direction tensors;
- restore scale=1.0;
- restore mode=standard.

Use one API lifecycle helper on explicit unload, replacement load, and failed replacement to reset interventions, lens, and neighbors consistently. A failed replacement currently leaves no model; it must not retain the old capability/rule identity.

Do not depend only on frontend remounting. Legacy API clients and scripts must receive the same clean state.

### 6.5 Validate GGUF cache identity

Define a deterministic current-edit fingerprint from normalized, output-relevant data:

- model_id and resolved revision;
- model_session-independent model identity, dtype, and relevant topology/version marker;
- intervention mode;
- global scale;
- ordered active rules including token/replacement IDs, operations, factors, enabled state, and layer lists;
- export method/version.

Write it into the cached HF edit metadata. Before reuse:

- require edit_meta.json, not config.json alone;
- compare the fingerprint with the current edit for “export current edit”;
- reject mismatches before state mutation or worker creation;
- allow additional GGUF types for the same fingerprint;
- include the fingerprint or model key in _gguf_state so the frontend can ignore a completion from another session.

Do not redesign PR #7/#8 atomic publication or abandoned-artifact inspection.

### 6.6 Known disagreement risks and boundaries

- **Source readiness:** status can prove basic readiness, not immutability. Execution rechecks and performs deep index/key validation.
- **Packed parameters:** model-wide profile controls format policy; execution still validates actual plan and keys.
- **Quantization:** declared quant is authoritative until a tested supported path exists.
- **Tied embeddings:** centralized detection and warnings keep status and execution aligned.
- **llama.cpp:** one probe produces both metadata and submission inputs; subprocess failure remains possible.
- **Offload:** intrinsic scans occur once at load; status does no model probing.
- **Rules:** capability metadata describes policy eligibility, not whether the current form has active/non-neutral rules.

## 7. Recommended frontend design

### 7.1 App-level status and session flow

Modify **ui/src/App.jsx** at the status effect, settings patch, load/unload handlers, intervention synchronization, and Editor props.

1. Hoist a stable, sequenced **refreshStatus()**. Ignore an older response when a newer request has completed, and track whether the last snapshot is fresh.
2. Call refreshStatus after:
   - model load and unload;
   - capability-relevant settings changes;
   - mode changes;
   - GGUF start and cache deletion.
3. In API-monitor mode, either consume a generic server status_changed event or keep a low-frequency safety poll. Independently poll while GGUF state is running so “progress shown below” is true.
4. Observe capabilities.model_session_id:
   - a new value closes/remounts the editor and clears prefill;
   - null closes the editor or renders a non-actionable “Load a model and lens” state;
   - status stale/failed disables capability-sensitive controls.
5. Pass the structured capability object and session ID to Editor. Remove rebaseSupported and llamaCppSet.
6. Continue taking rules, scale, and actual current mode from status; do not infer mode from capabilities.
7. Disable model load/unload while a synchronous export is actually in progress, not merely while manager.busy is nonempty.

### 7.2 Intervention-mode control

Replace the two-state ModeToggle with a compact native radio fieldset:

- Per-layer steering — standard.
- Read projection — readthrough.
- Exact compensated — exact, labeled advanced/soft factors.
- Global projection — abliteration.

The control should:

- render the actual current mode rather than treating every non-standard value as the same side;
- use the backend decision for each disabled state;
- show unsupported reasons directly under or beside the option;
- never derive global support from readthrough failure;
- default missing decisions to disabled;
- disable all model-dependent options while status is stale or the model session is changing;
- submit the selected mode to the existing endpoint and reconcile with its returned capability snapshot.

Showing exact as an advanced option is justified by PR #9’s explicit requirement. It does not implement packed exact mode. Packed models display the intentional Phase 1 reason.

### 7.3 Export format and type controls

Keep the existing export section; a broad editor redesign is unnecessary.

Recommended minimum accessible implementation:

- Keep a native select if desired, but mark unsupported options disabled and label them as unavailable.
- Associate the format label with the control through htmlFor/id.
- Add a visible capability-reason list immediately below and connect it through aria-describedby.
- Do not rely on title tooltips.
- If native disabled-option presentation is inconsistent across platforms, extract the four formats into a compact radio fieldset.
- MODE_INFO should contain mode semantics only. Delete its unconditional format lists.

Use a pure helper such as **chooseSupported(current, decisions, preferredOrder)**:

1. Keep current if decisions[current].supported is true.
2. Otherwise choose full when supported.
3. Otherwise choose the first supported entry in layers, lora, gguf order.
4. Otherwise return an empty selection.

Run this on:

- model session change;
- current mode change;
- capability snapshot change;
- source/tool readiness change.

Use the React functional state setter so the effect does not loop on exportFmt. Announce an automatic reset through an aria-live=polite region.

For GGUF:

- Render only types present in the backend map.
- Keep a type only while supported.
- Prefer bf16, then f16, then the first supported type after a reset.
- Explain that converter-only installations still support bf16/f16.
- Separate running job progress from the currently selected format so a format reset does not hide an active worker.
- Include the job’s model/edit fingerprint or session identity when deciding whether progress belongs to this editor session.

### 7.4 Form readiness versus model capability

Do not overload the backend capability map with request-local form conditions. In Editor compute:

- **hasActiveRules** = at least one rule with enabled !== false and a nonempty layers list;
- non-neutral scale/factor readiness where practical;
- nonempty export name;
- no current export request;
- manager/job not busy;
- selected format/type still supported.

The submit button requires all of these. doExport must recheck local support immediately before fetch, but the server remains authoritative.

Cached GGUF conversion without active rules is not “export the current edit.” Keep it API/CLI-visible unless a separate, explicitly named cache-conversion UI is later designed.

### 7.5 Cleanup and stale-work prevention

In **ui/src/Editor.jsx**:

- cancel every pendingRef timer and clear its map on model-session cleanup;
- cancel scaleTimer;
- discard optimistic localFactors and editing selections;
- reset export name/format/type on new model session;
- clear any old error/reason announcement;
- prevent a late fetch response from updating a newer model session;
- use native disabled attributes rather than pointer-events as the behavior boundary.

Closing and reopening the editor within the same model session may preserve harmless draft fields if desired. Unload/replacement must not.

### 7.6 Accessibility and copy

Required details:

- Use fieldset/legend and native radio semantics, or role=group with pressed buttons; do not combine radiogroup with aria-pressed buttons.
- Associate every label and form control.
- Connect each reason with aria-describedby.
- Add aria-label to the close button.
- Use aria-live=polite for format resets and GGUF transitions.
- Keep disabled reasons visible to keyboard, touch, and screen-reader users.
- Do not assert that “pure-weights” always means exportable.

Suggested concise copy:

- **Standard:** “Preview only. J-Wash intentionally refuses export because no faithful bake reproduces these per-layer hooks.”
- **Packed exact:** “Unavailable in packed-MoE Phase 1; this is an intentional support boundary.”
- **Packed layers:** “Packed expert parameters currently require a full checkpoint.”
- **GGUF without converter:** “Configure a valid llama.cpp folder containing convert_hf_to_gguf.py.”
- **No quantizer:** “bf16/f16 are available; quantized GGUF types require llama-quantize.”
- **Unknown topology:** “This model did not pass the safety preflight for this pure-weight mode.”

### 7.7 Bundled MCP client

Update **scripts/jwash_mcp.py::_add_rule** as a narrow compatibility integration:

- choose readthrough only when intervention_modes.readthrough.supported;
- otherwise choose abliteration only when intervention_modes.abliteration.supported;
- otherwise return the backend’s concise reason;
- never interpret rebase_supported=false as global support.

Do not redesign MCP tools or remove the legacy fields.

## 8. Testing strategy

### 8.1 Python tests

Add a focused **tests/test_api_capabilities.py** or **tests/test_export_capabilities.py**, then extend existing topology/export suites. Keep slow real-model work out of the status-policy tests by using current synthetic fixtures.

| Test area | File(s) | Required cases |
|---|---|---|
| Intrinsic metadata | tests/test_capabilities.py | Dense self-attention and linear-attention positive cases; packed readthrough true/exact false; unknown fail closed; positive global-only fixture; quantized flag with a deliberately floating fake lm_head; stable reason codes |
| Legacy compatibility | tests/test_capabilities.py | rebase_supported equals structured readthrough.supported; flat reason fields remain present |
| Global projection | tests/test_capabilities.py and a focused editing test | Validated write-norm inventory passes; missing/extra/linear-attention/packed/wrong-axis/bias layouts fail; live attach and export use the same inventory |
| Packed format policy | tests/test_transactional_exports.py | First-, middle-, and final-layer-only rules all reject layers/LoRA model-wide; full still succeeds |
| Exact restrictions | tests/test_capabilities.py | Packed exact false; dense writer bias/type/inventory failures; API PATCH rejects before mutation |
| Unknown architecture | tests/test_api_capabilities.py | Status reports no pure modes/formats; mode PATCH and export return 422 before traversal/publication |
| Quantized policy | tests/test_api_capabilities.py | quant=int8/nf4 disables modes/formats even when head dtype is floating; legacy client attempts still receive 422 |
| Source readiness | tests/test_export_capabilities.py | Missing source, missing directory, .bin-only, no safetensors, valid safetensors; layers/LoRA remain independently decided |
| Tied embeddings | tests/test_transactional_exports.py / test_indexed_checkpoints.py | Tied and untied dense status warnings; full/layers untie; readthrough LoRA do-not-merge warning; global LoRA reduced-fidelity warning; distinct meta-device parameters are not falsely tied |
| GGUF tools | tests/test_gguf_workers.py | Blank path, bad directory, missing converter, converter-only, unusable quantizer, valid converter+quantizer; bf16/f16 versus q-type map |
| Settings changes | tests/test_gguf_workers.py or API capability tests | PATCH settings changes GGUF snapshot without model reload |
| GGUF cache identity | tests/test_gguf_workers.py | Same fingerprint reuses; changed model/rules/mode/scale rejects before worker/state mutation; no-model explicit cache conversion remains separately defined |
| Enforcement parity | API capability tests | Every reported false action is rejected server-side; a stale formerly true snapshot cannot bypass a later deep preflight |
| Lifecycle | tests/test_lifecycle.py / API capability tests | Unload/replacement/failure clears lens, rules, scale, mode, intrinsic profile; new model_session_id; no old GGUF state is attributed to the new session |
| Poll cost | tests/test_api_capabilities.py | Multiple status calls do not invoke model_preflight, state_dict, tensor movement, or offload materialization |
| Transactionality | API capability tests | scale+unsupported mode leaves both scale and mode unchanged |

Do not assert complete human reason strings. Assert supported, reason_code, and only short semantic fragments where useful.

### 8.2 Frontend behavioral tests

There is no suitable current React test harness. For PR #9, adding a focused **Vitest + jsdom** setup is justified because the critical behavior is rerender/reset/disabled/accessibility state, not a static source contract. Reuse Vite’s JSX transform; do not add Playwright, Cypress, a browser farm, or broad end-to-end infrastructure.

Extract the controlled mode/export controls and capability normalization from Editor so tests do not mount the entire application. Pin a Vitest version compatible with the existing Vite 5 dependency; do not upgrade Vite as part of this PR.

Required component cases:

1. Missing/no-model capability data fails closed.
2. Dense decisions enable standard/readthrough/exact and the reported formats.
3. Packed decisions enable readthrough/full, disable exact/layers/LoRA, and display backend-provided reasons.
4. A failed readthrough decision does not enable global projection.
5. A positively supported global-only fixture shows standard/global and no readthrough/exact.
6. Rerender from dense+lora selected to packed resets to full and announces the change.
7. Rerender to no supported format leaves no selection and disables submit.
8. Model session change resets format/name and pending local state; unload shows the empty/non-actionable state.
9. Converter absent, converter-only, and converter+quantizer maps update GGUF and types correctly.
10. q4_k_m becoming unsupported resets to bf16/f16.
11. All rules disabled or layerless prevents a fresh export.
12. Controls have native disabled state, associated labels, descriptions, and live announcements.
13. API-selected exact/global is labeled truthfully.
14. A late old-session response cannot overwrite a new-session state.

Vitest/jsdom does not prove live backend integration or browser-specific native-select rendering. Retain a short manual smoke checklist:

- dense load → edit → exact/readthrough switch → export;
- packed load → intentional disabled reasons;
- valid/invalid llama.cpp transitions;
- model replacement and unload while the editor is open;
- API-monitor mode and GGUF progress;
- keyboard and screen-reader traversal.

If maintainers reject new test dependencies, the fallback is node:test coverage of a pure capability reducer plus the same mandatory manual checklist. That fallback is lower confidence and must not be described as behavioral React coverage.

### 8.3 Audit validation note

The production UI build passed in this audit. Python tests could not be run in the audit runtime because their dependencies were absent; the implementation agent must run the full existing suite plus the new tests in the project CI environment.

## 9. CI changes

Add one independent Linux-only job to **.github/workflows/ci.yml**. Do not attach it to the Python jobs through needs, and do not repeat it across OS, Python, or Transformers matrices.

Recommended job:

    frontend:
      name: Frontend (Node 24)
      runs-on: ubuntu-latest
      timeout-minutes: 10
      defaults:
        run:
          working-directory: ui
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-node@v7
          with:
            node-version: "24"
            cache: npm
            cache-dependency-path: ui/package-lock.json
        - name: Install frontend dependencies
          run: npm ci
        - name: Run frontend tests
          run: npm test
        - name: Build frontend
          run: npm run build

Rationale:

- As of 2026-08-04, Node 24 is an LTS release, while Node 18 and Node 20 are EOL. See the official [Node.js release table](https://nodejs.org/en/about/previous-releases).
- The lockfile’s marked 18.0.6 already requires Node >=20, so the README’s Node 18 claim is false.
- Node 24 is enough for this PR; a Node matrix would add little value for an application UI.
- setup-node’s npm cache uses the lockfile hash and does not cache node_modules. The subdirectory lockfile therefore needs cache-dependency-path. See the official [actions/setup-node documentation](https://github.com/actions/setup-node).
- Linux is enough for Vite compile/component tests; OS-specific backend behavior remains in the Python matrix.

**npm audit policy**

A read-only audit during this review reported two high and one moderate advisory in the existing Vite/PostCSS/esbuild toolchain. Therefore:

- do not add npm audit as a blocking PR #9 step;
- do not run npm audit fix;
- do not upgrade Vite or other dependencies in PR #9;
- record the advisories in a dedicated security/dependency issue.

Keep ui/dist ignored and uncommitted.

## 10. Documentation changes

### README.md

Update these exact areas:

- **Lines 20–25 and 62–67:** qualify universal full/layers/LoRA and “what you see is what you get” wording by supported pure mode and backend-reported availability.
- **Lines 74–89:** replace Node.js 18+ with Node 24 LTS as the recommended development/CI release; use npm ci for a lockfile checkout.
- **Lines 196–214:** retain the existing support matrix. Clarify that it describes unquantized, positively preflighted read projection; full requires source safetensors; GGUF is separate runtime readiness; the table does not imply global fallback for unknowns.
- **Lines 297–315:** describe standard, readthrough, exact, and positively validated global projection separately. Remove the implication that every readthrough failure becomes global projection.
- **Lines 317–333:** say the editor displays formats valid for the current model and mode. Clarify that modified layers are a partial tensor artifact and LoRA is a PEFT adapter, not a standalone full checkpoint.
- **Lines 335–340:** state that GGUF requires supported full bake plus a validated converter, and q-types require llama-quantize.
- **Line 441:** align the quantized restriction with the new reason shown in UI.

Do not add implementation-internal tensor inventories to the main user flow.

### docs/qwen3_5_moe_readthrough.md

The detailed Phase 1 explanation already exists. Add a short user-facing table near the top:

| Operation | Phase 1 |
|---|---|
| Live readthrough | Supported |
| Live exact | Unsupported |
| Full checkpoint | Supported when source validation passes |
| Modified layers | Unsupported |
| LoRA | Unsupported |
| GGUF | Conditional on full bake plus validated llama.cpp tools |

State explicitly that disabled exact/layers/LoRA choices are intentional boundaries, not a UI failure or a broken model. Preserve the technical architecture section and PR #7/#8 publication guarantees.

### UI and bundled client help

- Remove unconditional format lists from Editor.jsx MODE_INFO.
- Replace App.jsx’s “setting a path enables GGUF” wording with converter/quantizer validation status.
- Keep backend reasons concise and user-facing.
- Update scripts/jwash_mcp.py help/errors to use the structured mode choice without redesigning its interface.

## 11. Security, safety, and regression risks

| Risk/adversarial case | Required mitigation | Verification |
|---|---|---|
| Frontend/backend policy drift | One backend policy snapshot and the same require helpers at submission | Pair false metadata with direct 422 tests |
| Capability metadata stale after replacement | model_session_id, backend reset, sequenced refresh, Editor remount | Dense→packed→unload tests |
| Invalid React selection retained | Deterministic chooseSupported effect and no optimistic defaults | Rerender tests |
| Raw internal exceptions or paths in UI reasons | Stable codes and curated public text; diagnostic logging only | Inject probe/tool failure and inspect status |
| Unknown architecture receives global support | Positive global preflight; never negate readthrough | Unknown/extra-writer fixtures |
| GGUF supported from path string alone | Shared directory/file/executable probe and per-type map | Missing converter/quantizer cases |
| Expensive capability work on 2-second poll | Cache intrinsic model profile; status performs only cheap filesystem checks | Mock-call-count tests |
| Repeated preflight disturbs Accelerate offload | Load-time scan once; keep deep checks only on attach/export | Extend offload tests; hook/weights_map unchanged |
| Metadata moves tensors or grows memory | Shapes/classes and small norm gains only; never state_dict/full projection copies | Monkeypatch state_dict/tensor moves to fail during status |
| Rank-3 packed parameter treated as normal LoRA | Model-wide packed fact and execution guard | Final-layer-only rule regression |
| Packed exact exposed through fallback | Explicit exact decision; no cross-mode fallback | Mode PATCH and UI packed tests |
| Legacy client bypasses UI | Server mode/export checks remain authoritative | Direct API tests and old-field fixture |
| GGUF cache belongs to another edit | Canonical fingerprint and session-aware worker state | Same-name changed-edit test |
| Source folder changes after status | Recheck on submission and deep export validation | Delete/change source between status and request |
| Distinct meta tensors look tied via pointer zero | Centralized identity-aware tied detection | Meta-device fixture |
| Pending UI patch crosses model boundary | Session-key cleanup, request sequencing, backend reset | Fake timers and late-response tests |
| Failed mode PATCH changes scale | Validate before mutation/transactional update | Combined-request regression |
| Tests pin mutable prose | Assert codes, semantic DOM, and small fragments | Review test style |
| Scope expands into Phase 2 | Enforce false capabilities; no new packed writer/PEFT implementation | Diff review against non-goals |

The current GGUF worker displays raw converter stderr in its runtime error state. PR #9 must not reuse that text as capability reason. Broader runtime-error redaction can be a separate hardening item if maintainers want to change existing diagnostics.

## 12. Ordered implementation plan

The sequence below is suitable for a single reviewable Codex Cloud PR. Add tests with each behavior change rather than deferring all tests to the end.

### Step 1 — Lock the policy with fixtures and reason codes

**Files/symbols:** new core/capabilities.py; tests/test_capabilities.py fixtures; optional tests/test_api_capabilities.py.

**Behavior:** Define the small decision representation and stable codes. Write expected dense, packed, global-only, unknown, quantized, tied, source, and tool matrices before wiring the API.

**Tests:** Unit tests for decision serialization and no-model fail-closed defaults.

**Dependency:** None.

**Failure/rollback:** This step is additive. If names need revision, change codes before frontend consumption; do not ship codes that are derived from exception strings.

### Step 2 — Produce one intrinsic model profile

**Files/symbols:** core/rebase.py::block_capabilities/model_preflight and a new model profile helper; core/model_manager.py::_rebase_capability_meta, ModelManager.load/_unload_locked.

**Behavior:** Compute readthrough/exact once, include model-wide packed fact, explicit quant policy, tied notes, and model_session_id. Cache the profile and derive all flat legacy fields.

**Tests:** Dense/packed/unknown/quantized/tied profile tests; legacy equality; repeated scan call-count test; Accelerate offload remains intact.

**Dependency:** Step 1.

**Failure/rollback:** Preserve existing flat fields throughout. If one-pass optimization is risky, keep the existing probes temporarily but still cache results; never move probing into status.

### Step 3 — Add positive global-projection preflight

**Files/symbols:** core/ablation.py::Interventions._attach_abliteration and new global_projection_preflight; core/editing.py::TARGET_SUFFIXES/compute_abliteration/export_abliteration.

**Behavior:** Define one exact writer inventory shared by live and bake paths. Validate complete supported write-norm topology and reject unknown, hybrid, packed, extra, or missing writers. Remove skip-and-warn as a capability path; unexpected skips become fail-closed.

**Tests:** Positive synthetic write-norm model; missing/extra/wrong-axis/bias/linear-attention/packed failures; live/bake inventory equality.

**Dependency:** Steps 1–2.

**Failure/rollback:** If the repository cannot prove a real positive topology, ship global support false rather than preserve the unsafe not-readthrough fallback. Do not add a new family implementation.

### Step 4 — Compose /api/status capabilities

**Files/symbols:** core/capabilities.py::capability_snapshot; api/app.py::api_status; model_manager.resolve_local_dir; shared llama.cpp probe.

**Behavior:** Add the complete top-level capability object for no-model and loaded states. Combine cached profile with current mode, basic source readiness, converter, quantizer, and per-GGUF-type state. Keep status cheap.

**Tests:** TestClient snapshots for every matrix row; settings transitions; no state_dict/model_preflight calls during repeated status.

**Dependency:** Steps 1–3.

**Failure/rollback:** The field is additive. If source/tool probing errors, return supported=false with a safe code; do not make /api/status fail or expose exceptions.

### Step 5 — Enforce the shared policy at mode and regular export endpoints

**Files/symbols:** api/app.py::api_interventions_scale/api_edit_export; core/ablation.py; core/editing.py::_export_rebase_impl.

**Behavior:** Validate mode before any mutation, guard abliteration/no-model/quantized states, and guard mode-aware formats before export. Make packed layers/LoRA model-wide. Retain deep preflights.

**Tests:** False-reporting/direct-422 parity; final-layer packed rejection; scale+mode transactionality; legacy client direct calls.

**Dependency:** Step 4.

**Failure/rollback:** Do not remove existing build_plan/source/export validation. If early checks disagree with a deep check, fail closed and fix the shared fact; never relax the deep check.

### Step 6 — Reset lifecycle state and identify model sessions

**Files/symbols:** core/ablation.py::Interventions.reset; api/app.py::api_load/api_unload and a shared lifecycle helper; core/model_manager.py session metadata; lens/neighbors integration.

**Behavior:** Clear rules/directions/handles, scale, mode, lens, neighbors, and capability profile on unload/replacement/failure. Issue a new model_session_id per successful load.

**Tests:** Load A→load B, unload, and failed replacement; old rules/mode do not survive; status has a new/null session ID.

**Dependency:** Steps 2 and 4.

**Failure/rollback:** Presets remain the explicit persistence mechanism. If maintainers insist on keeping scale preference, rules and unsupported mode must still reset; document the choice.

### Step 7 — Align GGUF tools and cache identity

**Files/symbols:** api/app.py::_llamacpp_paths (replace with a shared probe/require pair), api_edit_export_gguf, _gguf_state; core/editing.py export metadata.

**Behavior:** Validate regular files/executability, use the same per-type decisions as status, require full support for new bake, fingerprint cache metadata, reject stale-name reuse, and tag worker state with edit/session identity.

**Tests:** Complete tool matrix; same/different fingerprint reuse; rejection before state mutation/worker creation; current PR #7/#8 atomic/retry suites unchanged.

**Dependency:** Steps 4–6.

**Failure/rollback:** Do not change publication paths, atomic rename, temporary cleanup, or abandoned-artifact inspection. Cache mismatch returns a clear 422 and preserves the cache.

### Step 8 — Update the bundled MCP consumer

**Files/symbols:** scripts/jwash_mcp.py::_add_rule and help/errors.

**Behavior:** Choose only a structured supported pure mode and surface its public reason when none exists.

**Tests:** Small unit/source-independent client tests with dense, global-only, and unknown status fixtures.

**Dependency:** Step 4.

**Failure/rollback:** Keep legacy response fields and MCP tool signatures. Older clients remain protected by Step 5.

### Step 9 — Make App status/session-aware

**Files/symbols:** ui/src/App.jsx status effect, patchSettings, onLoad/onUnload, intervention sync, Editor props.

**Behavior:** Add sequenced refreshStatus, freshness state, post-mutation refresh, GGUF-running polling/status_changed consumption, model-session handling, and structured props.

**Tests:** Mocked status transitions, late-response ordering, API-monitor changes, unload/replacement remount.

**Dependency:** Steps 4 and 6.

**Failure/rollback:** If WebSocket event changes are too large, retain a low-frequency safety poll. Never keep the current event-only behavior without an alternate refresh path.

### Step 10 — Replace mode/export Boolean UI

**Files/symbols:** ui/src/Editor.jsx::ModeToggle/MODE_INFO/export state/doExport/export markup; ui/src/styles.css.

**Behavior:** Explicit modes, backend-driven formats/reasons, deterministic format/type reset, active-rule readiness, session cleanup, native disabled semantics, labels/descriptions/live announcements.

**Tests:** Vitest/jsdom matrix and rerender/accessibility cases from Section 8.

**Dependency:** Step 9.

**Failure/rollback:** Keep the old visual styling where practical, but do not keep the optimistic fallback. If a control cannot represent disabled reasons accessibly, prefer a small radio group over hidden tooltips.

### Step 11 — Update documentation and user copy

**Files/symbols:** README.md sections listed in Section 10; docs/qwen3_5_moe_readthrough.md; App/Editor/MCP help text.

**Behavior:** Qualify export claims, explain intentional Phase 1 restrictions, document validated GGUF prerequisites, and correct Node guidance.

**Tests:** Documentation link check/manual review; avoid exact-string policy tests.

**Dependency:** Steps 4, 7, and 10 so docs match final fields and UX.

**Failure/rollback:** Preserve the existing detailed Qwen architecture reference and PR #7/#8 guarantees. Do not replace them with generic capability prose.

### Step 12 — Add frontend CI and run the complete suite

**Files/symbols:** ui/package.json/package-lock.json test script and focused test dependencies; .github/workflows/ci.yml frontend job.

**Behavior:** Node 24 Ubuntu job runs npm ci, npm test, and npm run build independently.

**Tests:** Run all Python tests, frontend tests/build, compileall, and git diff --check. Manually execute the browser checklist.

**Dependency:** Steps 9–11.

**Failure/rollback:** Do not resolve npm advisories by widening this PR. If new test dependencies force a Vite upgrade, choose a compatible test version or use the documented node:test fallback.

## 13. Acceptance criteria

- [ ] /api/status returns a complete fail-closed capability object with stable reason codes, including when no model is loaded.
- [ ] Intrinsic model inspection runs at load, not on status polls.
- [ ] rebase_supported remains exactly compatible with structured readthrough support.
- [ ] Dense audited topology reports and enforces readthrough/exact/full/layers/LoRA as appropriate.
- [ ] Packed Qwen3.5-MoE reports readthrough/full only; exact/layers/LoRA remain intentionally unavailable for first-, middle-, and final-layer rule plans.
- [ ] Unknown architectures never receive global projection or export through a Boolean fallback.
- [ ] A positively validated write-norm fixture can use global projection through the same live/bake inventory.
- [ ] Declared int8/nf4 loads fail closed for editing/current-edit export.
- [ ] Full/GGUF reflects basic source safetensors readiness and rechecks deeply at execution.
- [ ] GGUF distinguishes converter readiness from quantizer readiness and reports each type.
- [ ] Same-edit GGUF cache reuse works; changed edit/model/mode/scale with the same name fails safely without modifying the cache.
- [ ] Backend mode/export checks remain active for legacy/direct API clients.
- [ ] Unsupported combined scale+mode PATCH is side-effect-free.
- [ ] Unload/replacement clears interventions/lens/neighbors, resets mode/scale, changes model_session_id, and cannot retain old editor requests.
- [ ] Editor shows actual readthrough/exact/global support and concise reasons.
- [ ] Unsupported formats are disabled/omitted with visible adjacent explanations.
- [ ] Invalid selected format/type resets deterministically and is announced.
- [ ] No-model, unload, replacement, stale-status, and missing-metadata states are non-actionable.
- [ ] Accessible labels, native disabled semantics, descriptions, close label, and live regions are present.
- [ ] scripts/jwash_mcp.py no longer infers global support from rebase_supported=false.
- [ ] Frontend component tests cover rerender/reset/disabled/reason/accessibility behavior.
- [ ] npm ci, npm test, and npm run build pass on Node 24 in an independent Ubuntu CI job.
- [ ] Existing Python Linux/Windows matrices remain unchanged and pass.
- [ ] README and Qwen docs accurately describe model/mode-dependent formats and conditional GGUF.
- [ ] No Phase 2 packed feature, dependency modernization, broad UI redesign, or PR #7/#8 rewrite enters the diff.

## 14. Explicit non-goals

- Packed-MoE exact-mode implementation.
- Packed modified-layers writer/sharding implementation.
- Packed-parameter PEFT/LoRA support.
- MTP rebasing.
- Hook-factor device/dtype caching.
- Accelerate weights_map performance optimization beyond preserving current safe inspection.
- New model-family support or optimistic family-name allowlisting.
- A 122B benchmark or new peak-memory claim.
- Starlette/httpx warning cleanup.
- Dependency modernization, Vite upgrade, npm audit fix, or broad advisory remediation.
- Automatic destructive cleanup of abandoned export artifacts.
- Reworking PR #7/#8 atomic publication, cleanup, or lifespan behavior.
- Version tagging or release publication.
- A broad editor redesign.
- Playwright/Cypress or a full end-to-end testing platform.
- Windows frontend CI or a Node-version matrix.
- Redesign of MCP/CLI interfaces.

## 15. Deferred backlog

Items discovered but not required for PR #9:

1. Remediate the current npm advisories in a dedicated, reviewed security/dependency change.
2. Move Dockerfile from EOL Node 20 to a maintained LTS with a dedicated image-build test; do not hide this inside capability UX.
3. Consider broader Vitest/Testing Library or browser E2E coverage if frontend state complexity continues to grow.
4. Review whether GGUF runtime error messages should redact converter stderr and local paths while preserving useful server logs.
5. Consider a separately named “convert an existing cached HF export” UI if users need the current API-only no-model workflow.
6. Support source layouts without safetensors only through a separately designed conversion/export feature.
7. Broaden positively audited global-projection topologies only with semantic live-versus-bake tests.
8. Revisit quantized editing only with an explicit supported design and integration tests.

## 16. Open questions

There are no blocking open questions for the conservative PR #9 plan:

- README.md already resolves quantized policy: fail closed.
- Packed Phase 1 policy is documented: full only.
- Current global code does not prove broad architecture support, so the safe default is false unless the new exact-inventory preflight passes.
- Current tied LoRA behavior can remain selectable with explicit format-specific warnings for backward compatibility.
- Existing-cache GGUF conversion can remain an explicit API/CLI behavior while the editor models only a fresh current-edit export.

Any decision to relax these defaults is new feature work and belongs in the deferred backlog, not in this capability UX PR.
