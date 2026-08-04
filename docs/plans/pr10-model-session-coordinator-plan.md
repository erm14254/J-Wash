# Pre-implementation plan: atomic model-session and operation coordination

Planning basis: `erm14254/J-Wash`, branch `master`, commit `0fa0f74cac889312f25df9e71431854dce8c28c0` (merged PR #9). This is a backend-only maintenance plan. It does not propose implementation code or any GitHub-state change.

## 1. Executive recommendation

Add a focused `ModelSessionCoordinator` that is the sole authority for the published active-model session and the one active model operation. The first implementation should deliberately serialize all operations that retain or mutate active model, lens, intervention, capability, or model-bound cache state. `/api/status` remains non-owning and reads a short, coherent snapshot.

The recommended design is:

- Use an always-present, process-local, monotonic integer `model_session_id`; initialize it to `0`, representing the initial unloaded generation.
- Publish the active model as one bundle containing the HF model, tokenizer, Jacobian Lens adapter, metadata, and canonical PR #9 capability profile. Never publish or clear those fields independently.
- Represent each operation with a unique monotonic operation token and explicit type. Acquisition, state publication, owner validation, cancellation marking, and release are short atomic actions under one synchronous mutex.
- Record ownership for the duration of work, but never hold the mutex during model loading, inference, CUDA cleanup, disk or network I/O, subprocesses, thread joins, or an `await`.
- Use one exclusive model operation at a time. Do not add reader/writer compatibility or concurrent generation in this PR.
- Pass a captured session snapshot and the same token to internal helpers. Helpers do not reacquire and cannot release the root token.
- Transfer release responsibility to the actual synchronous worker when using `asyncio.to_thread`. Cancellation of the awaiting request must not make the coordinator appear idle while the thread is still running.
- Replace `Interventions._handles` with operation-local, idempotent attachment objects. A cleanup removes exactly the hooks that attachment installed.
- Build model and lens candidates off to the side, then publish them atomically only after validating the exact operation token and expected session.
- Clear intervention rules on every model unload/reload/replacement, including a same-ID reload. Do not migrate them. Preserve the numeric scale, reset mode to `standard`, and keep rule IDs monotonic.
- Keep existing rules across a lens-only unload/replacement in the same model session because their direction tensors are already materialized. Tag their lens provenance; never recompute them automatically. A user-initiated direction-changing edit or preset application may explicitly rebuild against the current lens.
- Keep regular export and fresh GGUF checkpoint baking under model-operation ownership. Release model ownership after the cached HF checkpoint is atomically published; subsequent llama.cpp conversion/quantization uses a separate GGUF job token and remains usable with no active model.
- Preserve PR #9’s capability contract and all existing transactional export, fitting ownership, lifespan, and non-destructive artifact-inspection guarantees.

This is intentionally more than a Boolean lock: the session ID rejects stale requests, the operation token rejects stale completion and cleanup, replace-whole state prevents torn snapshots, and exact handle ownership prevents one generation from detaching another operation’s hooks.

## 2. Current shared-state and operation map

### 2.1 Active model and related state

All line references below refer to the pinned base commit.

| File and symbol | Mutable state | Present synchronization | Current concern |
| --- | --- | --- | --- |
| `core/model_manager.py` — `ModelManager.__init__` (448–456) | `_lock`, `hf_model`, `tokenizer`, `jl`, `meta`, `capability_profile`, `busy` | `_lock` is used only by load/unload; `busy` is unlocked | The related fields do not form one published object, and most readers do not take `_lock`. |
| `ModelManager.load` (461–548) | Clears old model; assigns new model fields; sets/clears `busy` | Holds `threading.Lock` across the full load | It holds a synchronous lock across long model/network/CUDA work, clears the old model before publishing `busy`, and publishes the new bundle field-by-field at 508–535. |
| `ModelManager._unload_locked` (553–570) | Clears model fields and frees CUDA memory | Called with the model lock held | Readers can see partial clearing; long CUDA cleanup occurs while the lock is held. |
| `ModelManager.generate` (573–791) | Sets/clears `busy`; attaches hooks; reads model/lens/intervention state; frees CUDA cache | No model lock | It captures only `hf_model` and `tokenizer` at 578, then repeatedly reads live `self.jl` and `self.meta`; its finalizer unconditionally detaches global intervention handles and clears `busy`. |
| `core/lens_manager.py` — `LensManager.__init__` (96–108) | `lens`, `meta`, `layers`, `k`, `mask`, `_J`, `_tok_strs`, `gen_store`, counters and preference key | One lens lock for load/set/unload | Generation, pinning, string caching, and frame computation bypass the lock. No model-session binding exists. |
| `LensManager.load` (110–186) | Builds and publishes active lens state | Holds its lock across file/download, tensor stack, GPU transfer, mask construction, and preference I/O | It separately reads `model_manager.hf_model`, `meta`, `jl`, `tokenizer`, and embeddings, so one lens load can combine multiple model sessions. It publishes lens fields sequentially. |
| `LensManager.set_layers` / `unload` (188–220) | Replaces capture layers/Jacobians or clears lens state | Lens lock | Generation reads the same state unlocked. `unload` clears generation records and calls CUDA cleanup without model-operation ownership. |
| `LensManager.start_gen`, `pin_ranks`, `compute_frames` (222–353) | Generation record store, residuals, token strings, frame inputs | Unlocked | A stale `gen_id` can be combined with the current lens, `_J`, and model `jl`; concurrent generations mutate the same stores. |
| `ActivationCatcher` (41–58) | Per-generation hook handles and activations | Operation-local | This is the correct ownership shape, but constructor registration is not rollback-safe if an intermediate hook registration fails. |
| `core/ablation.py` — `Interventions.__init__` (78–85) | Rule counter/list, one global handle list, scale, mode | One lock on selected mutations | Reads, attach, and detach are not consistently locked; `_handles` has no operation identity. |
| `Interventions.set_scale_and_mode` / `state_snapshot` (111–125) | Mode and scale | Locked and validated before mutation | This is a PR #9 guarantee to retain. It is not currently coordinated with generation or export ownership. |
| `rules_full`, `active_rules_full`, `summary` (127–155) | Rule views | Unlocked; shallow copies | A generation or export can retain dictionaries/lists that another request mutates in place. |
| `Interventions.add` / `update` / `remove` (171–271) | Model/lens-derived directions and rule configuration | Mutations mostly locked | Directions belong to a model session and the lens used to derive them, but neither identity is stored. `remove` calls global `detach`. |
| `Interventions.attach`, `_attach_rebase`, `detach` (273–390) | Live model forward hooks | Unlocked; one shared `_handles` list | One operation overwrites another’s handles; late cleanup removes the successor’s hooks. Standard-mode attachment lacks partial-registration rollback; rebase attachment already has the correct transactional plan/rollback behavior. |
| `core/capabilities.py` | Canonical capability profile construction, normalization, status and deep enforcement | Pure functions | The policy is sound. The problem is that the profile is read and published separately from its model. It must be stored in the same session bundle without changing its semantics. |
| `core/neighbors.py` — `TokenNeighbors` | `_key`, GPU mask/norm tensors, result cache | Internal lock | API keying uses only `(model_id, revision)`, so a same-ID reload can reuse old-session tensors. |

Additional application globals matter to completion and status:

| File and symbol | State | Current issue | Coordination boundary |
| --- | --- | --- | --- |
| `api/app.py` — `_last_generation` (127) | Last HTTP generation result and counter | Unlocked and not session-bound; an old completion can publish after replacement | Move into the coordinated session state; clear on session change and update only for the owning session/token. |
| `_gguf_state` (605) | One plain-dict GGUF job status | Admission and updates are non-atomic; a stale worker can overwrite a retry | Give this independent job IDs and a short lock. It is not the model-session owner after the HF cache is published. |
| `fit_manager.state` | Long-running fit status | Fit ownership is internally reliable, but model-load admission is not atomic with “model must be unloaded” | Hold a narrow coordinator `FIT` reservation for the existing fit run lifetime; do not rewrite fitting orchestration. |
| `_downloads` / `_downloads_lock` (123–124) | Per-repository download states | Some worker updates bypass the lock | Not active-model state. Leave outside this PR unless copying it under the existing lock for status. |
| `_convert_state` (1215) | Background BF16 conversion status | Plain dict | Disk-only and outside model-session ownership. Do not fold it into the coordinator. |
| `_ws_locks`, `_loop_holder` (44–45) | WebSocket send locks and lifespan loop | Existing lifespan ownership is intentional | Preserve; only add strong tracking for owned `to_thread` tasks if required. |
| `Store` and conversation state | Messages and frame files | Independent persistence | Generation owns the model snapshot through model-derived frame/meta finalization; conversation storage itself is not part of the session coordinator. |

### 2.2 Current operation entry points

| Entry point / symbol | Active state read or mutation | Current gate | Planned treatment |
| --- | --- | --- | --- |
| `GET /api/status` — `api_status` (229–253) | Reads model, profile, lens, rules, scale/mode, busy, last generation separately | None | Non-owning coherent coordinator snapshot; append GPU/download/fit/GGUF telemetry afterward. |
| `POST /api/load` — `api_load` (256–271) | Replaces active model implicitly | Check-then-read of `manager.busy` | Exclusive `LOAD`; atomic withdrawal and candidate publication. |
| `POST /api/unload` — `api_unload` (361–367) | Clears lens/neighbors, then model | Check-then-read of `busy` | Exclusive `UNLOAD`; one atomic withdrawal of all model-bound state, then off-lock disposal. |
| `POST /api/models/delete` — `api_delete_model` (274–286) | Checks active metadata, deletes a possible load source | Non-atomic busy/meta check | Exclusive `MODEL_DELETE` through the delete so a load cannot race source removal. |
| Lens load/unload/layers/pin (370–403, 954–963) | Reads model/JL; replaces or consumes lens state | Inconsistent `busy` checks; lens unload has none | Exclusive typed operations using one captured model/lens binding. |
| Intervention GET/add/patch/scale/remove/clear (406–478) | Reads/mutates rules, mode, scale, directions, handles | No model-operation ownership | GET uses a coherent snapshot. Mutations acquire exclusive ownership and commit copy-on-write state. |
| Preset save/apply (486–537) | Reads model/rules or creates model/lens-derived directions | None | Exclusive `PRESET_SAVE`/`PRESET_APPLY`; preset application remains an explicit recreation path, not migration. |
| Regular export — `api_edit_export` (546–598) | Reads model, metadata, capability, rules, scale/mode; retains JL in a worker | No `busy` or model lock | Exclusive `EXPORT` held by the actual worker through publication and cleanup. |
| Fresh GGUF preparation — `api_edit_export_gguf` (720–783) | On cache miss, performs a full model-bound bake | Only plain `_gguf_state` check, set after the bake | Reserve GGUF job first; exclusive `GGUF_BAKE` only until the HF checkpoint is atomically published. |
| Cached GGUF conversion — `_gguf_worker` (641–717) | Uses only published files and tool paths | Plain `_gguf_state` | Separate GGUF job token; deliberately no model owner and no loaded-model requirement. |
| GGUF cache delete (790–806) | Deletes a cache used by a job | Non-atomic state/name check | Same-name GGUF job/deletion reservation; no model-session owner. |
| HTTP generation — `api_generate_sync` (814–861) | Inference, live interventions, last-generation publication | Check-then-`to_thread` | Exclusive `GENERATE`; captured snapshot; worker-owned cleanup/release. |
| WebSocket generation — `_run_chat`, `_generate_safely`, `_persisted_generate`, `_persisted_continue`, `ws_endpoint` (1299–1484) | Inference, lens frames, intervention hooks, message metadata | Per-socket serialization only; global check-then-read of `busy` | One exclusive `GENERATE` across all sockets and HTTP; immediate conflict frame for competitors. |
| Token neighbors / lookup (869–895) | Retain tokenizer/JL and GPU model-derived cache | Model-presence check only | Short exclusive model reads; neighbors keyed by session. |
| Fit start/stop (913–951) | Requires active model to remain unloaded for VRAM | Non-atomic unloaded check; fit has separate reliable owner | `FIT` reserves the unloaded model domain for the run lifetime. Stop signals the existing run; release occurs only when that exact run terminates. |

Catalog listing/registration, downloads, BF16 conversion, registry lookup, settings, preset listing/deletion, and ordinary conversation APIs do not retain the active model and remain outside model-session ownership. Model deletion is the exception because it can remove a source concurrently selected by load.

## 3. Concrete race and failure scenarios

1. **Generation versus unload.** HTTP or WebSocket generation passes the `busy` check before the worker sets `busy = "generating"`. Unload clears the lens and model and calls CUDA cleanup. Generation retains an old `hf_model`/tokenizer but later reads `self.jl = None` or a successor `jl`, causing incorrect hooks, frames, or failure during live inference.

2. **Generation versus replacement.** Old inference continues on model A while `self.meta` and `self.jl` become model B. The response can be labeled as B, hooks can target B, and A’s finalizer can detach B’s hooks or clear B’s `busy` value.

3. **Two generations.** Two clients can both pass the check. They run one PyTorch model concurrently, mutate the lens generation store together, overwrite `Interventions._handles`, and allow the earlier completion to report idle while the later generation still runs.

4. **Two accepted loads.** Two requests can both observe `busy is None`. The internal model lock merely queues the second; it does not return a conflict. The second request later unloads the first model even though the first request returned success.

5. **Unload during load.** The API clears lens and neighbor state before waiting on `ModelManager._lock`. It can then unload the model that the in-flight load just published. Load replacement itself does not clear old lens/intervention state.

6. **Lens load versus model replacement.** Lens loading can validate against A’s metadata, transfer tensors to B’s device/JL, build a mask from B’s tokenizer/vocabulary, and publish metadata claiming A.

7. **Lens unload/layer change versus generation.** Frame computation can observe new `layers` with old `_J`, `mask = None`, a cleared `gen_store`, or a replacement lens mid-generation. Pinning can combine an old record with the current model and lens.

8. **Intervention mutation versus generation/export.** Hooks retain mutable rule dictionaries and read live scale. A patch can change a factor, mode, layer list, or directions in the middle of inference. `remove()` can detach every live hook. Export’s shallow rule list can change after capability/mode selection.

9. **Regular export versus replacement.** The handler reads rules, source model ID, mode/scale, JL, metadata, and capability profile in separate steps. A complete but semantically hybrid export can be transactionally published from A’s directions, B’s metadata, and a source selected from either session.

10. **Cancellation releases too early.** Canceling an await of `asyncio.to_thread` does not reliably stop the underlying callable. Releasing from an async context manager would allow unload/replacement while load, generation, or export still uses the model. Current regular-export `gc.collect()` can run in the handler while its worker is still active.

11. **GGUF double admission and stale completion.** `_gguf_state == "running"` is checked before a potentially long fresh bake and set only afterward. Two jobs can enter. A late worker from job A can overwrite job B’s step, result, or error because job name is not an ownership identity.

12. **GGUF cache deletion race.** Deletion can pass its check immediately before a same-name job reserves or begins preparing the cache. Fresh preparation is invisible to `_gguf_state` during its longest phase.

13. **Torn status.** `api_status` can report new `hf_model` with no metadata/profile, old lens/rules with a new model, a capability snapshot from one mode alongside rules from another moment, or `busy = None` after a stale cleanup.

14. **Stale load failure or cleanup.** Current load failure clears all shared model fields unconditionally. The current long lock happens to limit this today, but cancellation-safe refactoring without token checks would let an old failure erase a successor.

15. **Stale model-derived caches.** A same-ID/revision reload can reuse `TokenNeighbors` GPU tensors from the prior model. Old lens generation records and `_last_generation` likewise lack a session identity.

16. **Fit versus load.** Fit can pass “no model loaded,” start its worker, and then a model load can begin because the fit is not represented by `manager.busy`. Both compete for VRAM despite the fitting manager’s otherwise correct internal ownership.

## 4. Proposed coordinator data model

### 4.1 Authoritative records

| Record | Required fields | Invariant |
| --- | --- | --- |
| `LoadedModelBundle` | `hf_model`, `tokenizer`, `jl`, copied/frozen `meta`, normalized canonical `capability_profile` | The five values are created together and published/withdrawn as one unit. A bundle never changes session after publication. |
| `LensBinding` | `bound_session_id`, monotonic `lens_binding_id`, lens object, copied metadata, immutable tapped layers, `k`, mask, stacked Jacobians, session/binding-tagged generation store | A nonempty lens binding always names the current model session. Candidate construction does not mutate the active binding. |
| `InterventionConfigSnapshot` | `bound_session_id`, revision, scale, mode, copy-isolated rules, per-rule source lens-binding provenance | Containers are immutable for an operation. Direction tensors may be referenced rather than copied because exclusive ownership blocks mutation and rule commits replace values. |
| `ModelSessionSnapshot` | `model_session_id`, loaded/unloaded state, model bundle or `None`, lens binding or `None`, intervention snapshot, copied last-generation record | Every model-bound operation receives one coherent tuple and uses no later manager globals. |
| `OperationToken` | monotonic `operation_id`, explicit operation type, acquired session ID, internal ownership nonce/state, start time, cancellation flag | Only the exact current token may commit or release. Operation IDs, not Python object IDs or thread IDs, define ownership. Tokens are never accepted from clients. |
| `OperationStatus` | public operation ID, type, acquired session, current session, start time, cancel requested | Diagnostic/status representation only; it does not grant authority. |
| `GGUFJobToken` / tracker | unique job ID, validated name, phase/state, cancellation flag, result/error, optional source session | Job identity is independent of name and model session. Every progress, file-publication guard, terminal update, and release validates it. |

Metadata and capability dictionaries exposed to callers must be copies or read-only views. Large model objects, lens tensors, and direction tensors stay by reference inside the captured bundle; serialization is provided by ownership, not by duplicating GPU memory.

### 4.2 Compatibility policy

The coordinator supports one exclusive model operation. Status snapshotting is compatible with an active operation because it only copies already-published state. Cached GGUF conversion, downloads, settings, registry reads, BF16 conversion, and ordinary conversation operations remain outside the model-operation domain.

Do not introduce a read/write matrix now. Generation mutates hook state and lens records, export retains large model references, and even “reads” such as neighbors populate GPU caches. Sharing would require additional isolation with no current production need.

Do not implement implicit reentrancy through `threading.RLock`, thread identity, or task identity. Async work crosses threads. The outer request acquires once; internal helpers borrow the same token and call an owner/session assertion. Borrowed helpers cannot release it. This covers conceptual nesting such as fresh GGUF preparation invoking the full-export helper without a second acquisition.

### 4.3 Locking hierarchy

Use a synchronous `threading.Lock` because both FastAPI request threads and ordinary worker threads access the coordinator. The mutex protects only owner/session comparison, replace-whole publication/withdrawal, coherent snapshot copying, cancellation marking, and release.

If component-local locks remain, the only permitted order is coordinator mutex first, then a component’s short state lock. No code may acquire the coordinator while holding a component lock. Model/lens preparation, intervention direction computation, hook registration, generation, export, filesystem work, subprocesses, CUDA cleanup, and callbacks all occur with every publication mutex released.

`/api/status` copies the authoritative published records while holding only the coordinator mutex (and, if unavoidable, short component locks in the documented order). It releases the mutex before `gpu_stats()`, fit/download snapshots, or any I/O.

## 5. Session-ID lifecycle

### 5.1 Exact rules

- Initialize `model_session_id` to `0`, the initial unloaded generation.
- The ID is process-local and resets on server restart. It is an optimistic-concurrency identity, not a persistent model fingerprint.
- Increment only when an authoritative model-state publication changes the loaded/unloaded generation.
- Reloading the same model ID and revision still creates a new session.
- Lens, intervention, generation, and export changes do not increment the model session.
- Rejected validation, stale-session rejection, operation conflict, and a failed initial load from an already-unloaded state do not increment it.

| Transition | Session effect | Published state |
| --- | --- | --- |
| Process start | Set `0` | Unloaded, no lens/rules/last generation |
| Initial load begins while unloaded | No increment | Same unloaded generation plus `operation=load` |
| Initial load succeeds | Increment once | Complete new loaded bundle |
| Initial load fails | No increment | Original unloaded generation; operation error/release only |
| Explicit unload of a loaded model | Increment once | New unloaded generation with model/lens/rules/last generation cleared |
| Redundant unload of an already clean unloaded state | No increment | Same unloaded generation |
| Replacement/reload begins with a loaded model | Increment once when the old bundle is withdrawn | New unloaded generation plus `operation=load` |
| Replacement succeeds | Increment again at complete bundle publication | New loaded generation |
| Replacement fails | No second increment | The new unloaded generation remains authoritative |

The two increments for a successful loaded-to-loaded replacement are intentional. Replacement is an unload publication followed by a load publication, and the unloaded interval has its own identity. A single target epoch reused for both “unloaded/loading” and “loaded” would allow a client that observed that epoch before publication to reuse it afterward without being detected as stale.

The load context therefore carries the unique operation token and the exact unloaded session from which publication is permitted. Success validates both, increments once, and installs the candidate. Failure disposes candidate-local resources only.

### 5.2 Lens and rule identity

Maintain a separate monotonic `lens_binding_id` within the process. Successful lens load/replacement, unload, and layer-binding publication advance it as appropriate; failed candidates do not. Lens generation records carry both model session and lens binding IDs.

Every intervention rule records the model session and the lens binding used to derive its directions. Rules from an older model session are always cleared during withdrawal and are never migrated, even when the replacement has the same model ID/revision. Rule IDs continue increasing so stale numeric IDs are not reused.

For a lens-only replacement within the same model session, retain existing materialized rules for compatibility; they continue to identify their source lens binding. Do not silently recompute them. A factor/enabled change preserves the existing directions. A user-requested token/layer/replacement/mode change explicitly rebuilds the whole rule from the current lens and updates its provenance. Applying a preset is likewise an explicit recreation path and retains the existing model-mismatch warning.

On model withdrawal, clear rules and live attachment registry, reset mode to `standard`, preserve the numeric scale, clear lens generation records and neighbor caches, and clear `last_generation`. Preserving scale is safe because it is a model-independent scalar; resetting mode avoids carrying a capability-dependent selection into a different topology.

## 6. Operation acquisition and release protocol

### 6.1 Acquisition

For syntactically valid requests, perform one atomic coordinator call with this decision order:

1. If the client supplied an expected model session and it differs, reject with `412` before any side effect.
2. If an owner already exists, reject immediately with `409`; do not wait or queue.
3. Validate the required loaded/unloaded state against the same snapshot, preserving existing `422` no-model and `409` fit/model conflicts.
4. Install a new owner record and return the token plus coherent session snapshot.

Request schema validation and name/type validation may occur before acquisition where they are side-effect free. Capability checks occur after acquisition against the captured profile, then release on rejection.

Suggested explicit types are `LOAD`, `UNLOAD`, `GENERATE`, `LENS_LOAD`, `LENS_UPDATE`, `LENS_PIN`, `INTERVENTION_UPDATE`, `PRESET_SAVE`, `PRESET_APPLY`, `EXPORT`, `GGUF_BAKE`, `TOKEN_NEIGHBORS`, `TOKEN_LOOKUP`, `MODEL_DELETE`, and `FIT`. They currently share one compatibility rule but give stable status, errors, logs, and tests.

### 6.2 Synchronous and async work

Short synchronous operations use a top-level lease with an outermost `finally`. Long async endpoints use a dispatch object with a small pending/claimed/cancelled handshake:

1. The handler acquires a token and creates the worker dispatch.
2. The synchronous wrapper atomically claims the dispatch before touching model state.
3. Once claimed, the wrapper is the sole releaser and executes cleanup before release.
4. If request cancellation or executor-submission failure cancels a still-pending dispatch, the handler releases because the worker cannot start.
5. If the worker already claimed, outer cancellation only sets the operation cancellation flag and any cooperative stop event. It must not release.
6. Keep a strong reference to the `to_thread` task until it completes; shield the worker lifecycle from cancellation of the request waiter.

This distinguishes cancellation before executor start from cancellation after a real thread is using the model. It also prevents a leaked owner when a cancelled executor work item never invokes its wrapper.

### 6.3 Release and commit

- Every state-changing commit calls `require_owner(token, expected_session_id)` under the mutex.
- `release(token)` compare-and-clears only the exact current operation ID/nonce. It returns success for the first release and a safe false/no-op for a stale or duplicate release.
- A stale commit raises an internal controlled stale-operation exception; it never mutates shared state. API code maps a request-visible stale commit to a stable conflict response, while cleanup paths log/no-op as appropriate.
- Physical cleanup of operation-local objects is always allowed: remove the exact local hook handles, close an exact activation catcher, discard a candidate, or remove that operation’s unique staging directory. Cleanup of shared state, `busy`, last generation, GGUF status, or global CUDA cache requires current ownership/session.
- Legacy `busy` is derived from the owner record. No manager method assigns or clears it directly.

## 7. Load, unload, and replacement transaction

### 7.1 Load admission and old-state withdrawal

1. Validate dtype, quantization, device, and source request without shared mutation.
2. Acquire `LOAD`, requiring no fit/model operation conflict.
3. Under the coordinator mutex, if a model is loaded, atomically withdraw the model bundle, lens binding, model-bound rules, last generation, and neighbor-session key; reset mode to `standard`; preserve scale; increment to an unloaded generation. Record that unloaded ID in the load context.
4. If already unloaded, leave the session unchanged and use it as the permitted publication base.
5. Release the mutex. Dispose the retired model/lens tensors and run CUDA cleanup outside the mutex while the `LOAD` owner remains recorded. Skip global CUDA cleanup if the token has somehow become stale.

There is no attempt to keep the old model serving during replacement. The current unload-first behavior is necessary for VRAM. The improvement is that the unloaded state is published coherently and protected by ownership.

### 7.2 Candidate preparation and publication

Refactor model loading so all work occurs in locals: HF model, tokenizer/template, `jlens.from_hf`, declared quantization, canonical `capabilities.build_profile`, normalized legacy metadata, and model metadata. Candidate construction must not assign active manager fields.

On success, acquire the mutex briefly, validate the exact `LOAD` token and recorded unloaded session, increment the session, and publish the entire `LoadedModelBundle` in one action. The new lens binding is empty, rules are empty, mode is `standard`, preserved scale remains, and the capability profile is the exact PR #9 profile built for that JL/model.

On candidate failure, clear and free candidate-local references only. Do not write `None` to active state. Leave the recorded unloaded generation unchanged and release in the worker’s outermost finalizer. A stale failure callback is a no-op against a successor.

If request cancellation occurs after the loader thread has started, loading is not safely interruptible inside `transformers.from_pretrained`. Preserve current effective semantics: mark cancellation, keep ownership, allow the worker to finish, and publish only if its token/session are still current. The caller may disconnect, but another model operation cannot overlap the live loader. Cancellation before worker claim releases without starting.

### 7.3 Unload

Acquire `UNLOAD`, atomically withdraw all model-bound state, and increment only if a loaded bundle existed. Return the new session ID in the response. Dispose exact withdrawn objects and perform CUDA cleanup outside the mutex while the token remains owner. A redundant clean unload returns `unloaded: false` and does not increment.

Lens/intervention cleanup is part of the same withdrawal publication, not a sequence of API calls. Rules from the old model are discarded rather than migrated. No new operation may start until off-lock disposal completes and the exact unload token releases.

### 7.4 Lens load and replacement

Lens load acquires a captured loaded-model snapshot. Build the candidate lens, compatibility metadata, tapped layers, stacked Jacobians, mask, and preference update inputs outside publication locks. Publish the complete binding only if the token and model session still match. A failed or cancelled candidate leaves the previous active lens and existing materialized rules unchanged.

Lens unload/replacement clears session/binding-tagged generation records. It does not automatically rewrite rules in the same model session. `set_layers` publishes a complete replacement lens-state record after building `_J`; no generation can overlap because it owns the same exclusive domain.

## 8. Generation and hook lifecycle

### 8.1 Generation start and snapshot

Both HTTP and WebSocket paths acquire `GENERATE` before selecting the model, lens, or interventions. The snapshot freezes:

- model session ID, HF model, tokenizer, JL, metadata, and capability profile;
- requested lens binding, tapped layers, `k`, mask, and stacked Jacobians;
- intervention mode, scale, and copy-isolated active rules;
- operation ID for generation-store and hook provenance.

Refactor `ModelManager.generate` to accept this explicit model/lens/intervention context. It must not read `self.jl`, `self.meta`, global managers, or set/clear `busy`.

Only one generation is allowed across all HTTP clients and WebSocket connections. A competitor receives an immediate `409` or WebSocket error frame. This is the smallest safe policy for current PyTorch, hooks, lens records, and GPU memory.

### 8.2 Live intervention attachment

`Interventions.attach` should return an operation-specific attachment object rather than mutate a global `_handles` list. It captures the mode, numeric scale, and rule snapshot, plans every hook site, registers transactionally, and stores only the exact returned handles.

- Standard attachment must gain the same partial-registration rollback already present in `_attach_rebase`.
- Rebase/exact attachment retains model-wide topology preflight, full hook planning, audited site selection, and rollback.
- Quantized and global-projection fail-closed checks remain before hook side effects.
- Hook closures read captured scale/rules, not live intervention state.
- `close()` removes every exact handle once, attempts all removals even if one raises, records closed state, and is idempotent.
- Closing attachment A can never remove B’s handles, even in a forced stale-completion test.

Make `ActivationCatcher` construction transactional as well. It remains local to the generation and closes before the operation token releases.

### 8.3 Success, error, stop, cancellation, and disconnect

The synchronous generation wrapper owns this order:

1. Create local intervention attachment if needed.
2. Create local activation catcher/lens generation record if requested.
3. Run generation using only captured state.
4. Finalize frames and model-derived metadata, tagging them with `model_session_id` and `lens_binding_id`.
5. Identity-check any coordinated `last_generation` publication.
6. Close the activation catcher and exact intervention attachment, attempting both on all paths.
7. Perform failure CUDA cleanup only while the token is still current.
8. Release the operation token last.

An exception before hooks exist closes nothing and still releases. An exception after partial hook installation rolls it back. A stop or WebSocket disconnect sets the operation-local `threading.Event`; `_run_chat` should race the queue, receiver, and worker so a disconnect wakes cleanup immediately rather than waiting indefinitely on `queue.get()`. It then awaits/gathers the real worker without releasing early.

If a forced stale generation completes after a replacement, its text may be retained in conversation history with the old session in metadata, but it cannot publish current-session last-generation state, clear current ownership, remove successor hooks, or mutate current lens records. Normal operation serialization should prevent such overlap; identity checks are defense in depth.

## 9. Export and GGUF interaction

### 9.1 Regular edited-model export

Acquire `EXPORT` and capture one session/intervention snapshot before capability selection or source resolution. Use the captured profile for `capabilities.require`, captured JL/meta for deep checks, captured rules/mode/scale, and a source directory resolved from the captured model ID. Do not read manager globals after acquisition.

The synchronous export wrapper owns the token through export construction, the existing transactional stage publication, stage cleanup, and worker-side `gc.collect()`. Move `gc.collect()` out of the async handler’s `finally`.

Add an optional synchronous liveness/publication guard to `editing.export_rebase`. It is invoked at bounded safe checkpoints and immediately before `stage.replace(final_dir)`. Direct internal/test callers may omit it. For coordinated API exports, it validates exact operation ownership, session, and cancellation state. If stale/cancelled before publication, it raises and the existing unique stage `finally` removes only that stage. If cancellation arrives after the final check and atomic rename has linearized, publication wins and the complete artifact remains.

Do not change the existing conservative overwrite policy, semantic inventory checks, bounded transforms, indexed-checkpoint validation, or transactional publication. Do not enable `export_abliteration`; PR #9’s deep rejection remains.

### 9.2 Fresh GGUF checkpoint preparation

Validate the export name/type/tool configuration, then atomically reserve a GGUF job before checking or building the cache. On cache miss:

1. Acquire `GGUF_BAKE` and a coherent loaded-model/intervention snapshot.
2. Set owner-checked GGUF phase to `baking checkpoint`.
3. Run the same coordinated transactional full export to `<name>/hf` with a publication guard.
4. Treat successful atomic HF-directory publication as the handoff point.
5. Release the model operation after bake cleanup/publication, while retaining the GGUF job token.
6. Start conversion/quantization using only the published cache and job token.

If fresh-preparation request cancellation occurs before HF publication, mark both tokens cancelled; the export guard rolls back if it has not linearized. If the cache already published, preserve it. Do not start conversion after a cancelled preparation; finish model cleanup, mark the exact GGUF job cancelled/error in a controlled way, and leave the complete cache reusable.

### 9.3 Cached conversion and GGUF job state

When `<name>/hf/config.json` already exists, preserve current no-model behavior: reserve only the GGUF job, never inspect or acquire the active model, and start conversion even when unloaded. An optional expected model-session precondition is ignored for this sessionless cached path; adding cache fingerprint/session validation is explicitly out of scope.

Replace naked `_gguf_state` reads/writes with an owner-aware tracker supporting atomic reserve, owner-checked progress/terminal update, cancellation marking, locked snapshot, idempotent release, and same-name deletion reservation. `_gguf_worker` receives the unique job token. Check it before publishing a base or quantized temp file and before every shared-state update. A stale worker still removes only its own UUID temp but cannot overwrite a successor’s state or final file.

During cached conversion, `/api/status` intentionally shows model `busy: null` while `gguf.state: running`; model load/unload may proceed. A second GGUF job or same-name cache deletion conflicts independently.

## 10. API and status behavior

### 10.1 Backward-compatible preconditions and responses

Clients are not required to send a session ID. Without one, a request binds atomically to the current session at acquisition. Add an optional `X-Model-Session-Id` header for HTTP model-bound operations and optional `model_session_id` in WebSocket chat messages. This permits later frontend stale-action protection without frontend work in this PR.

Successful model-bound responses and WebSocket terminal frames should echo the session used. `/api/status` remains authoritative. Owner tokens are internal and are never accepted from clients.

| Condition | HTTP behavior | WebSocket behavior |
| --- | --- | --- |
| Supplied expected session does not equal current | `412 Precondition Failed`; stable string detail such as `stale model session: expected X, current Y` | `{type: "error", code: "stale_model_session", message, model_session_id}` |
| Another model operation owns the domain | `409 Conflict`; stable detail naming active type/session | Error frame with `code: "operation_conflict"` and current session/operation summary |
| Required model/lens absent | Preserve existing `422` domain errors | Existing error message plus stable code/session fields |
| Invalid request/name/type/capability/quantization | Preserve existing `422` behavior and canonical PR #9 reason strings | Corresponding error frame |
| Unexpected execution failure | Preserve controlled `500` mapping | Error frame; owner cleanup still occurs |

Keep FastAPI’s `detail` value a string for current UI compatibility. If structured sibling fields are added through a custom error response, they are additive.

### 10.2 `/api/status`

Retain all existing fields and derive all model-bound values from one coordinator snapshot. Add:

| New field | Meaning |
| --- | --- |
| `model_session_id` | Always-present integer, including unloaded generations |
| `model_state` | `loaded` or `unloaded`; the active operation separately explains loading/unloading |
| `operation` | `null` or `{id, type, session_id, current_session_id, started_at, cancel_requested}` |

Continue returning `loaded`, `capabilities`, `busy`, `lens`, `interventions`, `interventions_scale`, `interventions_mode`, and `last_generation`. Build `loaded` legacy capability fields and the structured capability snapshot from the same bundle/profile and captured mode. `busy` is a legacy string derived from operation type; preserve `loading` and `generating`, and use stable labels such as `unloading`, `exporting`, `loading lens`, `editing`, and `fitting` for other owners.

Copy `_downloads`, fit state, GGUF state, and conversion state under their own short locks/snapshot methods after releasing the coordinator mutex. Call `gpu_stats()` last. The response may combine independently timed GPU/download telemetry, but its model/profile/lens/intervention/operation tuple must be one coherent instant.

## 11. Exception and cancellation cleanup

| Failure point | Required cleanup | Shared-state rule |
| --- | --- | --- |
| Acquisition rejected or cancelled before owner install | None | No session/owner change |
| Owner installed, executor submission fails or dispatch cancelled before claim | Handler releases exact token once | No worker may subsequently claim/use it |
| Model candidate load fails | Drop candidate locals; token-current CUDA cleanup; release | Never clear published state in the failure handler |
| Lens candidate fails | Drop candidate locals; preserve active binding/rules; release | No partial preference/binding publication |
| Exception before hooks | Release after local finalizers | No global detach |
| Exception during hook registration | Remove every handle registered by that attachment | Do not publish attachment as live; preserve PR #9 fail-closed behavior |
| Exception after hooks | Attempt activation-catcher and attachment close independently, then token-current CUDA cleanup and release | Cleanup acts on exact local handles only |
| HTTP cancellation after thread claim | Mark cancel and signal stop if supported; retain owner until real thread exit | Awaiting coroutine never clears `busy` |
| WebSocket disconnect | Signal stop, wake queue/receiver race, gather worker, close local resources, release in worker | No release merely because socket closed |
| Export cancellation/staleness before rename | Liveness guard raises; unique stage cleanup runs; worker releases | No final artifact publication |
| Export cancellation after atomic rename | Complete artifact remains; cleanup/release runs | Publication is the linearization winner |
| Fresh GGUF cancellation after cache publication | Preserve complete cache; do not start converter; terminate exact job state | No destructive cache rollback |
| Cached GGUF worker failure | Remove exact temp, preserve any prior valid output/cache, owner-check error update/release | Never overwrite successor job state |
| Old completion after successor | Local cleanup allowed; shared commits/releases return stale/no-op | Successor owner, session, busy, state, and handles remain unchanged |
| Double release | First exact release succeeds; later releases return false/no-op | A later owner cannot be cleared |

Cleanup code must attempt all independent local finalizers even when one raises, then run the outer token release. Do not let hook-removal, progress callback, WebSocket send, or memory-cleanup exceptions leak ownership.

## 12. File-by-file implementation plan

| File | Symbol/class | Purpose | Expected behavior | Relevant tests |
| --- | --- | --- | --- | --- |
| New `core/model_session.py` | `OperationType`, coordinator errors, `OperationToken`, `LoadedModelBundle`, `ModelSessionSnapshot`, `ModelSessionCoordinator`, worker-dispatch helper or protocol | Centralize session identity, exclusive ownership, coherent snapshot/publication, cancellation, stale checks, and idempotent release | Short mutex only; exact token/session checks; replace-whole state; deterministic conflicts; no long work under lock | New coordinator unit tests for acquisition, session sequence, stale commit, cancellation handoff, double release, coherent snapshot |
| `core/model_manager.py` | `ModelManager.__init__`, `load`, `_unload_locked`, `unload`, `generate`; new candidate/disposal helpers | Split candidate preparation from publication; stop global busy mutation; make generation consume captured bundle | No field-by-field publication; failure cleans locals; API generation never reads live manager state; standalone script compatibility retained through a safe convenience wrapper/read-only bundle properties | Load failure/publication tests; generation-vs-unload; two loads; same-ID reload; existing direct script/test smoke |
| `core/lens_manager.py` | `LensManager.load`, `set_layers`, `unload`, `start_gen`, `pin_ranks`, `compute_frames`, `ActivationCatcher` | Build/publish lens state transactionally and bind generation records to session/lens identity | No long lens lock; failed candidate preserves prior lens; stale pin rejected; catcher registration rollback/idempotent close | Lens-load-vs-replacement; stale pin; coherent status; partial catcher registration; bounded cleanup |
| `core/ablation.py` | `Interventions` state/snapshot/mutations; new attachment object; `attach`/`detach` transition | Produce coherent immutable config snapshots and exact hook ownership | Preserve atomic scale/mode; copy-on-write rules with session/lens provenance; standard/rebase registration rollback; old attachment cannot detach new; global projection/quantized checks unchanged | Existing capability/atomicity tests; stale hook cleanup; registration failure; rule transition/provenance tests |
| `core/neighbors.py` | `TokenNeighbors.lookup`, `_prepare`, `reset` or API key contract | Prevent same-ID replacement from reusing old GPU cache | Include `model_session_id` in cache key and reset during atomic withdrawal; no old tensor reuse | Neighbor session replacement test; lookup-vs-unload conflict |
| `core/editing.py` | `export_rebase` and safe bounded/publication checkpoints | Add optional coordinator liveness/publication guard without altering export semantics | Stale/cancelled coordinated exports roll back their unique stage; atomic rename remains the publication boundary; direct callers unchanged | Existing transactional/indexed/temp-cleanup suites plus stale prepublication guard and cancellation tests |
| `core/fitting.py` | Existing run record and terminal/finally path; small reservation field/finalizer; state snapshot | Keep model domain unloaded for exact fit lifetime without rewriting fitting workers | `FIT` token is stored on the exact run and released only after actual terminal cleanup; stop does not release early; stale run cannot release successor | Existing fitting ownership suite plus fit-vs-load, start-failure release, terminal/double-finalizer tests |
| `api/app.py` | Global construction and `api_status` | Instantiate coordinator/job tracker and replace individual global reads | Status model tuple is coherent; legacy fields remain; task registry/lifespan behavior preserved | Status barrier tests; lifecycle tests; malformed-profile fail-closed test |
| `api/app.py` | `api_load`, `api_unload`, `api_delete_model`, lens/intervention/preset/token/fit endpoints | Replace check-then-act busy logic with typed acquisition and optional session precondition | Stable 412/409/422 mapping; successful responses echo session; no uncoordinated active-state access | Endpoint conflict/stale tests, load/lens/intervention/fit races |
| `api/app.py` | `api_generate_sync`, `_generate_safely`, `_persisted_generate`, `_persisted_continue`, `_run_chat`, `ws_endpoint` | Capture once, transfer worker ownership, make disconnect/cancellation deterministic | One generation globally; exact hook/catcher cleanup; last-generation owner check; no live manager reads | HTTP and WS generation-vs-unload, second generation conflict, disconnect, cancellation, stale cleanup |
| `api/app.py` | `api_edit_export`, `api_edit_export_gguf`, `_gguf_worker`, GGUF cache delete; new job tracker | Coordinate regular/fresh export and separate cached conversion ownership | Model owner ends at HF cache publication; cached no-model path retained; stale workers/deletes cannot clobber job/output | Export-vs-replacement, fresh handoff, cached no-model, two jobs, stale worker, delete-vs-start, existing GGUF atomicity suite |
| `tests/helpers.py` | New coherent synthetic-session factory, event gates, bounded join helper; `_deferred_threads` update | Make concurrency tests deterministic and compatible with new locks/events | `join(timeout)` always asserts worker terminated; no sleeps; patch only thread creation rather than replacing the whole `threading` module | Used by all new coordination/GGUF tests; existing helper-based tests remain passing |
| `tests/conftest.py` | Coordinator/GGUF isolation fixtures | Prevent cross-test owner/session/job leakage | Fresh coordinator/job tracker per coordination test; teardown fails clearly if an owner/worker remains; stop mutating a naked `_gguf_state` dict | Fixture self-check and full suite isolation |
| New `tests/test_model_session_coordinator.py` | Pure coordinator/session/dispatch tests | Prove invariants without FastAPI timing | Exact session increments, conflict ordering, stale commit/release, cancellation claim handshake, double release, coherent snapshots | All unit-level acceptance cases |
| New `tests/test_operation_coordination.py` | API/thread/WebSocket race tests | Prove competing production operations are serialized and cleanup-safe | Event/barrier-driven generation/load/export/lens/intervention/fit scenarios with bounded joins | Required race matrix in section 13 |
| Existing `tests/test_capabilities.py`, `tests/test_capability_contract.py`, `tests/test_accelerate_offload.py` | API fixtures and attach/detach assertions | Adapt impossible field-by-field monkeypatches to coherent session installation and attachment leases | All PR #9 reason strings, topology validation, deep checks, live/baked equivalence, and scale/mode atomicity remain unchanged | Entire existing files plus new coordination conflict assertions |
| Existing `tests/test_gguf_workers.py`, `tests/test_transactional_exports.py`, `tests/test_indexed_checkpoints.py`, `tests/test_export_temp_cleanup.py` | Job-state setup and publication assertions | Adapt to job tokens while retaining filesystem guarantees | Atomic temp publication, retry, prior-valid preservation, stage rollback, leases, and non-destructive inspection remain intact | Entire existing suites plus job/session guard cases |
| Existing `tests/test_lifecycle.py`, `tests/test_fitting.py` | Task tracking and fit lease finalization | Preserve lifespan ownership and fitting cleanup | No deprecated event handlers; cancellation does not leak tasks/owners; exact fit run releases reservation | Existing tests plus targeted additions |

`core/capabilities.py` is intentionally not a policy-change target. Consumers will read its profile from the active bundle, but profile structure, canonical reasons, supported topology, quantization rejection, packed policy, export decisions, and global-projection disablement must remain unchanged.

## 13. Deterministic test matrix

Use `threading.Event`, `threading.Barrier`, injected callbacks at named phases, `asyncio.wait_for`, and bounded thread joins. Every join must assert `not worker.is_alive()` with a failure message naming the phase/operation. Do not use timing sleeps as synchronization.

| Required case | Deterministic orchestration | Assertions |
| --- | --- | --- |
| Generation versus unload | Fake generation signals after worker claim and blocks on an event; call unload | Unload returns 409; no model/lens/hook cleanup occurs; after release and bounded join, unload succeeds and increments once |
| Export versus replacement | Fake export blocks after snapshot; attempt load/replacement | Replacement returns 409 and loader is not invoked; export arguments all carry one session; replacement succeeds only after worker release |
| Two competing loads | First loader blocks in candidate preparation; second request attempts acquisition | Second returns immediately with 409, does not queue, invoke loader, or change session; first publishes once |
| Stale load failure after newer success | Retain old token/failure callback; release/retire it in a controlled unit setup, publish successor bundle, then invoke old failure | Successor model/meta/profile/lens/rules/owner/session are unchanged; old cleanup returns stale/no-op |
| Failed replacement | Start with loaded A; withdraw A; make B preparation fail | Exactly one increment to unloaded generation; no partial B or A restoration; rules/lens/last generation cleared |
| Stale generation cleanup | Complete/release A, acquire B, then rerun A final release/status cleanup | B remains owner and `busy`; current last generation/session unchanged |
| Stale hook cleanup | Install fake attachment A and B handle sets; close A after B exists | Only A handles are removed; B removal counts remain zero; repeated A close is harmless |
| Hook registration exception | Fail the Nth standard and catcher registration | Every prior exact handle is removed; no attachment is published; owner releases; next generation can acquire |
| Stale export cleanup/publication | Pause real export at liveness guard before rename, mark token stale/cancelled, continue | No final directory; exact stage removed; successor state and any prior valid destination unchanged |
| Exception-safe ownership release | Inject exceptions before hooks, after hooks, during model load, lens load, export, and cleanup | Local cleanup attempted; owner cleared exactly once; next operation acquires; original exception mapping retained |
| Cancellation before worker claim | Hold executor dispatch pending, cancel requester, then allow executor | Handler releases; worker cannot touch model; no owner leak |
| Cancellation after worker claim | Worker signals claimed and blocks; cancel requester | Stop/cancel flag set; ownership remains and successor gets 409 until real worker exits; bounded completion releases |
| Double release | Release A twice; then acquire B and invoke A release again | First release succeeds; later calls return false/no-op and never clear B |
| Session increment rules | Exercise initial success/failure, direct/no-op unload, failed/successful replacement, and same-ID reload | Exact integer sequence matches section 5, including two increments for successful replacement |
| Coherent `/api/status` | Pause at old loaded, withdrawn/unloaded, candidate-ready, and new-published barriers; query status | Every response is entirely old, unloaded, or new; capability/legacy/mode/lens/rules/session agree; busy derives from owner |
| Lens load versus replacement | Block candidate after captured model; attempt replacement | Replacement conflicts; lens publishes only to captured session; failed/cancelled lens candidate preserves prior binding |
| Stale lens pin | Save old gen ID, change lens binding or model session, invoke pin | Controlled expired/stale error; no computation against current JL/lens |
| Rules across model transition | Create rules, unload/reload same model ID | Rules/attachments cleared, mode reset, scale preserved, IDs not reused, no automatic migration |
| Rules across lens-only transition | Replace/unload lens in same session | Existing directions remain unchanged with old provenance; no automatic recomputation; explicit direction edit rebuilds against current lens |
| Neighbor session identity | Populate cache in session A; publish same-ID session B | B does not reuse A’s GPU mask/norm/cache; old references are released/reset |
| Multiple generations | Hold HTTP or WS generation; start another HTTP/WS request | Exactly one model call/attachment; competitor gets deterministic conflict |
| WebSocket disconnect | Receiver raises/disconnects while worker is blocked on a cooperative gate | Stop is signaled immediately; worker, catcher, attachment, receiver all terminate within bounds; owner releases |
| Cached no-model GGUF | Create valid cache while coordinator unloaded; block converter | Request succeeds without model owner/session increment; model load can acquire concurrently; second GGUF and same-name delete conflict |
| Fresh GGUF handoff | Block bake, then converter separately | Replacement conflicts during bake; after atomic HF publication/model-token release it may acquire while converter remains running |
| Two GGUF admissions | Barrier two cache-hit requests | Exactly one job token/worker; other returns 409; no double dispatch |
| Stale GGUF worker | Start A, transition tracker to B in controlled unit setup, deliver A progress/done/error (including same name) | Every A update/publication is rejected; B’s state/result/error/output remain unchanged |
| Cache delete versus job | Block deletion after reservation, attempt same-name job; repeat inverse | Both directions return 409 for same name; no partial deletion/start |
| Intervention scale/mode regression | Keep existing lock-signaling tests and add operation-conflict case | Pair update remains validate-before-commit and atomic; denied patch changes neither field |
| Transactional export regression | Run existing export/index/temp suites and inject coordinator guard failure | No partial/fake final output, existing destination untouched, unique stages cleaned, atomic publication retained |
| Capability regression | Run all PR #9 capability/topology/live/export tests | No capability widening, reason change, quantized bypass, packed Phase 2 support, or global projection path |
| Fit versus load | Hold existing fit run after exact reservation; attempt load | Load returns 409 through stopping/cleanup; exact terminal run releases; stale fit finalizer cannot release successor |

## 14. Compatibility and migration considerations

- No database or on-disk migration is needed. Session and operation IDs are process-local runtime state.
- Preserve current request bodies. The optional HTTP header and WebSocket field are additive; existing UI, CLI, scripts, and MCP callers can omit them.
- Preserve string `detail` errors and legacy `/api/status` fields. Additional status and response session fields are additive.
- Preserve `busy` truthiness and the existing `loading`/`generating` values the UI recognizes. Short operations may expose additional stable labels.
- The safety change intentionally rejects overlapping generations/exports/intervention edits that currently race unpredictably. This is a compatibility restriction, not a queued behavior; clients should retry after 409.
- Retain safe standalone `ModelManager`/`LensManager` convenience behavior for `scripts/verify_accuracy.py`, or update that script to install/use one local bundle. Server production code must use the coordinator; unsafe public setters should not be preserved merely for tests.
- Replace tests that monkeypatch `manager.hf_model`, `jl`, `meta`, and `capability_profile` independently with a helper that publishes one coherent synthetic session.
- Adapt direct `Interventions.attach()/detach()` tests to the returned attachment lease while keeping every fail-closed and live-versus-baked assertion.
- Rules do not survive a model session change. Presets remain the explicit recreation mechanism and retain mismatch warnings. No automatic migration is added.
- Rules may survive lens-only changes in the same model session as materialized directions with provenance. This chosen behavior is the smallest compatibility-preserving option; changing it to invalidate rules would need an explicit product decision and acceptance update.
- Cached GGUF behavior remains name/cache based. No fingerprint, automatic invalidation, or model-session requirement is added.
- Fitting internals, worker ownership, progress, stop behavior, and cleanup remain unchanged except for carrying/releasing one exact model-domain reservation.
- FastAPI lifespan ownership and non-destructive abandoned-export inspection remain intact. Strong task tracking must not reintroduce deprecated event handlers or swallow startup cancellation.

## 15. Risks and alternatives

| Risk or alternative | Assessment and decision |
| --- | --- |
| One exclusive operation reduces throughput | Accepted for this PR. Current shared hooks, lens stores, model execution, and GPU memory are not safe for concurrent readers. Add compatibility classes only after separate isolation work and measurements. |
| A cancelled non-interruptible load can keep `busy` for a long time | Correct and safer than false-idle overlap. Status exposes cancellation requested. Do not force-release a live loader or start a second model allocation. |
| Successful replacement advances the session twice | Intentional: the unloaded interval and new loaded bundle are distinct authoritative publications. It is clearer than one epoch that changes meaning. |
| Clearing model-session rules changes current accidental persistence | Required. Old directions are derived from the old model/lens and automatic migration is unsafe. Presets provide explicit recreation. |
| Retaining rules across lens-only replacement can surprise users | Directions are self-contained and remain bound to the same model. Provenance makes this explicit; no automatic recomputation occurs. The alternative is explicit invalidation, not silent migration. |
| A coordinator that delegates snapshots to long-held component locks can deadlock | Avoided by refactoring heavy work outside locks and enforcing coordinator-first, short-lock-only publication/snapshot order. Prefer replace-whole immutable records. |
| A plain global Boolean or `manager.busy` | Rejected: it has no session identity, owner identity, stale cleanup protection, coherent snapshot, or cancellation transfer semantics. |
| Hold one global lock for the whole operation | Rejected: it would block status/request threads across inference/I/O and violates async/thread safety. Ownership is recorded while the mutex is released. |
| Use `asyncio.Lock` only | Rejected: sync FastAPI handlers and ordinary/GGUF/fitting threads also need coordination. |
| Use Python object IDs or thread/task IDs | Rejected: IDs can be reused, do not express model generations, and async work changes threads/tasks. Use monotonic session and operation IDs plus internal token identity. |
| Queue conflicting operations | Rejected for v1. A queued load could unexpectedly replace a model long after the caller’s context; fail-fast 409 is deterministic. |
| Release ownership immediately on request cancellation | Rejected because `to_thread` work can continue. The pending/claimed handshake determines the only safe releaser. |
| Reuse filesystem artifact leases as model locks | Rejected. Artifact leases protect unique staging paths; they do not bind model/lens/capability state or stale workers. |
| Bind cached GGUF conversion to the current session | Rejected. Once the HF checkpoint is atomically published, conversion is file-only. Session binding would break required no-model cache reuse and drift into excluded fingerprinting work. |
| Recompute capability profiles on every status/operation | Rejected. Keep PR #9’s once-per-load canonical profile and publish it with the model bundle. |

## 16. Exact acceptance criteria

1. The implementation is based on `master` at `0fa0f74cac889312f25df9e71431854dce8c28c0` and does not redesign PR #9 capability decisions.
2. One `ModelSessionCoordinator` is authoritative for the active server model bundle, session ID, model-bound lens/intervention snapshots, last-generation state, and active model operation.
3. `model_session_id` starts at `0` unloaded and follows exactly the transition table in section 5; tests assert every increment and no-increment case.
4. The active HF model, tokenizer, JL adapter, metadata, and capability profile are published and withdrawn as one coherent bundle. No production API path reads those fields independently after operation acquisition.
5. A successful loaded-to-loaded replacement publishes an intervening unloaded generation, then a new loaded generation. Failed replacement leaves the unloaded generation; failed initial load leaves the prior unloaded ID.
6. Rules from the withdrawn model session, live attachments, lens generation records, neighbor cache, and last generation are cleared atomically. Mode resets to `standard`; scale is preserved; rule IDs are not reused.
7. No intervention rule is automatically migrated or recomputed across model sessions. Lens-only behavior follows the explicit provenance policy in section 5.
8. Every model-bound operation has a unique typed token. Acquisition and release are atomic; conflicts are fail-fast; stale/double releases cannot clear a successor.
9. There is no synchronous publication lock held across model inference/load, export, I/O, subprocess work, CUDA cleanup, thread join, callback, or `await`.
10. Async-to-thread dispatch distinguishes pending from worker-claimed work. Cancellation before claim releases safely; cancellation after claim leaves release to the real worker.
11. Legacy `manager.busy` mutation/check-then-act logic is removed from production coordination. `/api/status` derives `busy` and structured operation state from the coordinator.
12. HTTP and WebSocket generation use only one captured model/lens/intervention snapshot and reject a second generation. Success, exception, stop, cancellation, and disconnect all close exact local resources before release.
13. Intervention attachment is transactional and operation-local for both standard and rebase/exact modes. Close is idempotent; stale attachment cleanup cannot remove successor hooks.
14. PR #9’s global-projection disablement, quantized-edit rejection, packed readthrough/full-only policy, canonical reasons, topology validation, and atomic scale/mode updates remain unchanged and fully tested.
15. Regular export owns the model operation through guarded transactional publication and cleanup. A stale/cancelled export before rename cannot publish; a publication that already linearized remains complete.
16. Fresh GGUF preparation owns a model operation only through atomic HF-cache publication. Cached conversion/quantization uses a separate exact job token and works with no loaded model.
17. GGUF job admission, progress, terminal state, output publication, retry, and same-name cache deletion are owner-checked. A stale worker cannot overwrite successor state or output.
18. Existing transactional export staging, indexed-checkpoint validation, atomic GGUF temp publication, previous-valid preservation, non-destructive abandoned-artifact inspection, and retry behavior remain passing.
19. Fit holds an exact unloaded-model reservation for the existing run lifetime; load cannot overlap a running/stopping fit; existing fitting worker ownership and cleanup behavior is otherwise unchanged.
20. `/api/status` retains legacy fields and adds `model_session_id`, `model_state`, and structured `operation`. All model/profile/lens/intervention fields in one response are from one coherent snapshot.
21. Optional expected-session input is backward compatible. Supplied mismatch returns 412 with no side effects; active-owner conflict returns 409; existing 422 capability/no-model/request behavior remains.
22. All minimum races listed in the request are covered with event/barrier-driven tests: generation/unload, export/replacement, two loads, stale load failure, stale generation/hook/export cleanup, exception release, cancellation, double release, session rules, coherent status, cached no-model GGUF, scale/mode regression, and transactional export regression.
23. Every concurrency test uses bounded waits/joins and fails with a clear message if a worker remains alive. No race assertion depends on `sleep`.
24. The full existing backend/Jacobian Lens test suites pass with no capability widening or export-publication regression.
25. The PR contains no frontend/React, MCP, README/release docs, workflow, dependency, GGUF fingerprinting, new model-family support, global projection, packed Phase 2, or broad fitting-manager changes.

## 17. Recommended PR title and branch

- Title: `Add atomic model-session and operation coordination`
- Branch: `maintenance/model-session-coordinator`

