recommended scope
=================

Implement one backend-only maintenance PR, titled **Harden export publication and GGUF job integrity**, containing B3, B4, and B5 together. Base it directly and only on `ca343a5cb02dcd191433912d92c087124149910c`. Do not include capability-policy, UI, or other work from draft PR #12.

The coherent boundary is: an export may publish only while its exact operation owns publication; every transformed disk tensor must satisfy an operation-private structural contract; and a GGUF conversion, its status, its cache interaction, and its filesystem publication must have one exact private owner from request reservation through terminal completion.

This plan is based on a detached, clean, read-only inspection of that exact commit, its tests, the current CI workflow, and the frozen PR #12 commit `18c337a70ccffc621444333bd65f710d20d63a8b`. The requested base was independently verified as the current `master` head. No repository or GitHub state was changed.

Recommended PR metadata:

| Field | Value |
|---|---|
| Title | `Harden export publication and GGUF job integrity` |
| Future branch | `maintenance/export-publication-integrity` |
| Exact base | `ca343a5cb02dcd191433912d92c087124149910c` |
| Scope | B3 + B4 + B5, backend and tests only |
| Public API schema | Unchanged |
| UI | No change |
| Dependencies/workflows | No change |

## 1. Verified root cause of B3

### Verdict on the independent assertions

| Assertion | Verdict | Source-inspection result |
|---|---|---|
| Both normal export and fresh GGUF bake close over a mutable coordinator-token local. | **Confirmed.** | The two predicates are at `api/app.py:1903-1905` and `api/app.py:2170-2172`. |
| Cleanup can clear that token before publication. | **Confirmed and strengthened.** | This is deterministic, not merely possible: `_prepare_export_handoff()` assigns `token = None` at line 1912 before creating the worker task; `_prepare_gguf_bake_handoff()` repeats it at line 2179. Both `finally` blocks clear the cell again. |
| The later guard can receive `None`. | **Confirmed.** | Python closures retain the cell. `PublicationGate.publish()` calls the predicate at `core/editing.py:1593-1602`; `ModelSessionCoordinator.is_cancelled(None)` dereferences `token.id` at `core/model_session.py:431-433` and raises `AttributeError`. |
| A conceptual immutable binding is sufficient. | **Refined.** | Bind only the exact coordinator and immutable `OperationToken`, but add one atomic coordinator query rather than retaining the existing two separately locked queries. This also removes the cancellation/currentness check gap. |
| Session, dispatch/handoff, or stop state must also survive in the guard. | **Disproved.** | Exact current token ownership already incorporates the acquired session. Waiter cancellation first cancels the gate and then the dispatch. Capturing handoff, dispatch, snapshot, bundle, or stop event is unnecessary and would weaken lifetime cleanup. |
| Other production handoffs have the same mutable-token gate closure. | **Disproved for this base.** | A repository-wide search found only these two production `PublicationGate` predicates with this mutable-token pattern. |

The current failure sequence is deterministic:

1. The coordinator creates and installs an immutable `OperationToken`.
2. The setup function constructs a `PublicationGate` whose lambda references the local `token` cell.
3. The dispatch receives the gate and the exact token, but setup then assigns `token = None` before `create_thread_task()`.
4. The worker successfully claims the dispatch and constructs a unique staging directory.
5. At real publication, the gate evaluates the lambda with `token is None`.
6. `is_cancelled(None)` raises before `stage.replace(final_dir)`.
7. `export_rebase()` removes the stage in `finally`; the worker releases the exact operation; the legitimate request is reported as a false server error. For a fresh GGUF request, HF publication fails and conversion never starts.

The bug affects enabled Readthrough/Exact normal exports and cache-miss GGUF bakes through `editing.export_rebase()`. Cached GGUF conversion bypasses this gate. At this base, Global Projection/abliteration export remains fail-closed and is not a reachable counterexample.

## 2. Verified root cause of B4

### Verdict on the independent assertions

| Assertion | Verdict | Source-inspection result |
|---|---|---|
| The live model is validated but streamed disk tensors are not bound to the validated inventory. | **Confirmed.** | `model_preflight()` builds a strong private inventory, but `_build_plan_from_inventory()` emits transform tuples without exact source shapes. Full export proves key presence, index placement, and byte totals, then transforms tensors read separately from disk. |
| Same keys with wrong rank/leading dimensions can publish. | **Confirmed.** | `apply_transform_bounded()` at `core/editing.py:1540-1568` reshapes every read tensor to `(-1, tensor.shape[-1])`. A malformed packed tensor such as `[E*2, I, H]` instead of `[E, 2I, H]` can retain the same last axis and publish. |
| Wrong row count can publish. | **Confirmed.** | The current row count is derived from the disk tensor itself; it is not compared with the inventory tensor. |
| Wrong final dimension silently publishes. | **Refined.** | It normally reaches an uncontrolled matrix-multiplication `RuntimeError`, not false success. It still needs an explicit keyed `ValueError` before arithmetic. |
| Integer/non-floating transformed tensors can falsely succeed. | **Confirmed.** | The tensor is converted to float32 for the transform and cast back to the source integer dtype. Integer MTP/unmatched tensors are a different case and must remain copyable. |
| Dense Exact writers have equivalent protection. | **Disproved.** | They have a distinct hole: the write path converts the arbitrary disk tensor to float32 and calls generic transform code. A wrong column count can pass if residual output dimension/axis 0 is compatible. |
| Exact runtime/disk dtype equality is the correct contract. | **Disproved.** | Loading deliberately converts to BF16/FP16 at `core/model_manager.py:582-599`; Accelerate/offload and original FP32/F64 disk weights are legitimate. Shape/topology and usable floating representation are authoritative, not equality to runtime dtype. |
| A local source can change after load. | **Confirmed.** | Full export resolves the path again and streams it later. Local directories are mutable. |
| The recorded Hub revision already pins export input. | **Disproved.** | Load records a resolved revision at `core/model_manager.py:620`, but normal/GGUF preflight call `resolve_local_dir(meta["model_id"])` at `api/app.py:1958` and 2009. `resolve_local_dir()` does not accept or pass a revision. |

The precise missing contract is not “same number of elements,” “same last dimension,” or “same dtype as the loaded model.” For every active transform it is:

- the mapped disk source corresponds injectively to the inventory-derived memory key;
- exact logical shape equality, including rank and every packed/leading dimension;
- the expected transform role and residual orientation;
- a supported floating disk representation;
- an explicit source/output binding for tied head fallback;
- output shape and dtype equal to the original disk source shape and dtype.

The current indexed validation at `core/editing.py:1416-1500` calculates schema, key placement, and `metadata.total_size`, but discards shape and dtype descriptors. The full streaming path at `core/editing.py:1821-1861` then checks only mapped-key presence before reading and transforming payloads.

## 3. Verified root cause of B5

### Verdict on the independent assertions

| Assertion | Verdict | Source-inspection result |
|---|---|---|
| Singleton state is checked before a potentially long fresh bake and reserved only afterward. | **Confirmed.** | `api_edit_export_gguf()` reads `_gguf_state` at `api/app.py:2202`, awaits the fresh bake at 2218-2234, and writes `state="running"` only at 2237. |
| Any second fresh request can always overlap the first bake. | **Refined.** | A second cache-miss request is normally blocked by `OperationType.GGUF_BAKE`, but a cached request bypasses that coordinator. A same-name cached request can also enter after A publishes `config.json` and before A reserves conversion state. |
| Shared state is unversioned and stale workers can overwrite newer status. | **Confirmed.** | `_gguf_worker()` makes unconditional progress and terminal `.update()` calls at lines 2077, 2109, and 2132-2141. Terminal updates do not reset `name`, so a result/error can be displayed under another job's name. |
| The endpoint is the only state writer. | **Disproved.** | There are two production writer families: route reservation and conversion-worker progress/terminal updates. `/api/status` and cache deletion are unlocked readers. There is no shutdown writer. |
| Existing-base reuse accepts invalid filesystem objects. | **Confirmed.** | `base_gguf.exists()` at line 2076 accepts a zero-byte file, a directory, and a symlink. A base-format request can skip conversion and report `done`. |
| A public job ID or general job framework is necessary. | **Disproved.** | A private immutable GGUF token plus a `threading.Lock` is sufficient. The public state can retain `{state,name,step,error,result}`. |
| Full GGUF parsing is required. | **Disproved.** | No reusable parser/validator exists in this repository. Matching the generated-output invariant—no-follow regular file and positive size—is the appropriate scoped check. |

Additional defects confirmed during source inspection:

- `threading.Thread(...).start()` at `api/app.py:2238-2243` is outside cleanup handling; construction/start failure can strand public state at `running` forever.
- Cache deletion at lines 2261-2266 is neither linearized with reservation nor with another deletion.
- Valid nested names create a path-prefix hazard: the cache directory for `job` is also the job directory for `job/hf`.
- The conversion thread is an untracked daemon using blocking `subprocess.run`; there is no lifespan cancellation/join hook.

Stale hidden temp names are not the reuse defect: their names do not equal the exact published base path. Existing generated-temp cleanup remains important, but reuse must validate the exact base file.

## 4. Exact production call paths

### B3 normal export

| Order | Call/ownership transition | Base location |
|---:|---|---|
| 1 | `POST /api/edit/export` enters `api_edit_export()` | `api/app.py:1924-1945` |
| 2 | `OperationHandoff.acquire_scope()` installs cleanup ownership before acquire | `api/app.py:400-422` |
| 3 | `OperationHandoff.acquire()` calls coordinator | `api/app.py:508-523` |
| 4 | Coordinator installs exact token and returns operation-private snapshot | `core/model_session.py:396-418` |
| 5 | `_prepare_export_handoff()` performs preflight, constructs gate/payload/dispatch | `api/app.py:1892-1921` |
| 6 | `create_thread_task()` retains task, transitions through `TRANSFERRED`, and clears heavy setup refs | `api/app.py:686-711` |
| 7 | `_make_export_worker()` enters `_execute_dispatched_worker()` | `api/app.py:1877-1889`, 282-351 |
| 8 | Worker claims dispatch, takes payload, and calls `editing.export_rebase()` | `core/model_session.py:116-135`; `core/editing.py:1605-1649` |
| 9 | Export builds stage; `PublicationGate.publish()` evaluates ownership and runs `stage.replace(final_dir)` | `core/editing.py:1593-1602`, 1640-1641 |
| 10 | Worker clears payload and releases that exact token in `finally` | `api/app.py:330-346`; `core/model_session.py:435-441` |

Intentional survivors are the retained task, dispatch, coordinator/token inside the dispatch, stop event, payload until claim, and publication gate. Snapshot/bundle references survive only where the worker payload needs them. Handoff-heavy/setup locals are intentionally cleared after transfer.

Cancellation is already ordered usefully: `_await_transferred_worker()` at `api/app.py:851-897` calls `gate.cancel()` before `dispatch.cancel_from_awaiter()`. `PublicationGate.cancel()` and `.publish()` share a lock. Before claim, dispatch cancellation releases immediately; after claim, it marks the exact token cancelled, sets the stop event, drains the late worker, and lets the worker perform exact release.

### B3/B5 fresh GGUF path

| Order | Call/ownership transition | Base location |
|---:|---|---|
| 1 | `POST /api/edit/export-gguf` validates name/type/tools and checks public singleton | `api/app.py:2193-2211` |
| 2 | Cache decision uses only `<job>/hf/config.json` existence | `api/app.py:2213-2215` |
| 3 | On miss, acquire `GGUF_BAKE` and prepare handoff | `api/app.py:2218-2231`, 2159-2190 |
| 4 | Worker performs `export_rebase(fmt="full", name="<job>/hf")` | `api/app.py:2144-2156` |
| 5 | Route drains bake; exact model-operation ownership is released | `api/app.py:2232-2234`, 330-346 |
| 6 | Only now public GGUF state is set to running and daemon conversion starts | `api/app.py:2237-2243` |

The cached path skips steps 3-5, including coordinator ownership and the only await. That is why it can bypass A's unreserved interval.

### B4 full-checkpoint transform

1. Normal/GGUF preflight re-resolves `model_id` to a source directory without the recorded revision (`api/app.py:1958`, 2009; `core/model_manager.py:436-450`).
2. `export_rebase()` builds one operation-private `ModelInventory` before staging (`core/editing.py:1605-1639`; inventory validation in `core/rebase.py:496-1033`).
3. `_build_plan_from_inventory()` emits algebraic transforms and diagnostics, but no frozen source tensor specification (`core/rebase.py:1212-1262`).
4. Full export validates index schema/placement/total size or enumerates safetensors, maps keys, and checks required key presence (`core/editing.py:1416-1500`, 1821-1838).
5. It streams each source shard. A transformed key is passed to `apply_transform_bounded()`, which derives rows from the current disk tensor (`core/editing.py:1705-1713`, 1848-1861, 1540-1568).
6. Tied fallback reopens a selected staged shard and creates `lm_head`; indexed final placement is validated at `core/editing.py:1959-1966`.
7. Only after all work does the publication gate atomically rename the stage.

### B5 conversion and cache deletion

`_gguf_worker()` (`api/app.py:2065-2141`) creates the job directory, derives the base path, reuses anything for which `.exists()` is true, otherwise generates a unique leased temp and atomically replaces the base. Quantized output follows another validated temp/replace. Every progress/terminal update currently mutates the shared dict unconditionally.

`/api/status` copies the dict at `api/app.py:1128` without a lock. Cache deletion validates `<name>/hf`, reads only public name/state, computes size, and runs `shutil.rmtree()` at lines 2251-2267. Lifespan at lines 917-950 does not track the conversion thread.

## 5. Proposed implementation design

### B3: immutable, atomic publication ownership

Add one coordinator method with a single-lock predicate, for example:

```python
def can_publish(self, token: OperationToken | None) -> bool:
    with self._lock:
        return (
            token is not None
            and self._operation == token
            and token.id not in self._cancelled
        )
```

Use `functools.partial(handoff.coordinator.can_publish, token)` (or an equivalent default-argument closure) when constructing both gates. Bind the values before any setup-local clearing. Prefer `handoff.coordinator`, not the mutable global `manager.coordinator`, because it is the coordinator that issued the token and dispatch.

Do not capture `handoff`, `dispatch`, snapshot, bundle, session ID, or stop event. The exact token includes acquired/expected session identity. Gate cancellation supplies the atomic cancel-versus-publish boundary, while the coordinator method supplies current, uncancelled ownership. Capturing heavier objects can create a `dispatch -> payload -> gate -> dispatch/handoff` cycle and retain a model-bearing payload beyond worker completion.

Map a deliberate `PublicationCancelled` that reaches a still-live endpoint caller to a controlled HTTP 409 rather than a generic 500. Caller cancellation itself continues to re-raise `CancelledError` only after the worker and exact ownership have drained. Do not make publication reversible after a rename has already won the gate lock.

### B4: inventory-derived disk source contract

Add an immutable internal specification in `core/rebase.py`:

```python
@dataclass(frozen=True)
class SourceTensorSpec:
    memory_key: str
    shape: tuple[int, ...]
    transform_kind: Literal["read", "write"]
    residual_axis: int
    site: Literal["reader", "writer", "final_head"]
```

Populate exactly one spec with every active transform inside `_build_plan_from_inventory()`:

- each `BlockInventory.reads` tensor: exact inventory shape, `read`, residual axis last, site `reader`;
- each active Exact dense writer's `module.weight`: exact shape, `write`, residual axis 0, site `writer`;
- `inventory.final_head[2]`: exact shape, `read`, residual axis last, site `final_head`.

Return the specs as private plan metadata, for example `info["source_specs"]`. Do not add disk paths or source dtype to `ModelInventory`; the inventory remains topology/shape authority, and valid disk dtype can differ.

Map transforms and specs together through `_disk_mapper()`. Reject a mapped-key collision before writing instead of letting dict construction overwrite one entry. Introduce an explicit internal binding for each physical source, conceptually:

```python
@dataclass(frozen=True)
class DiskSourceBinding:
    output_key: str
    source_key: str
    spec: SourceTensorSpec
    owner_shard: str | None = None
```

For ordinary transforms, output and source keys match. For tied fallback, only when the inventory is tied, no explicit disk head exists, and the embedding exists, bind logical head output to the embedding source. If an explicit disk head exists, it must validate against the head spec; malformed explicit head must fail rather than fall back.

For indexed tied fallback, `weight_map[embed_key]` is the authoritative owner shard. Do not use the current “last physical occurrence encountered” value. Preserve duplicate/hard-linked shard aliases by shard name, not inode: validate every physical occurrence that the streaming loop will transform, and use the index owner only to choose the tied source/output shard. For an unindexed duplicate source, preserve deterministic sorted-shard selection while recording the selected shard explicitly; same-shape value identity remains outside this structural contract.

Use two defensive validation layers:

1. **Metadata-only preflight before the first staged shard write.** After key mapping and tied binding, collect `get_slice(key).get_shape()` and `.get_dtype()` for only active transformed physical occurrences and the tied source. Extend `_validate_index_contents()` to retain descriptors keyed by `(shard_name, key)` while it already scans headers. Use an equivalent header pass for unindexed sources. No payload or full checkpoint is loaded.
2. **Mandatory transform-boundary validation.** Require `apply_transform_bounded()` to receive the exact spec and a diagnostic disk key. Before indexing shape or doing arithmetic, validate kind, role, exact shape, allowed dtype, and transform-matrix dimensions. Recheck result shape and original dtype afterward. This covers direct callers and a source replacement between header scan and streaming.

Allowed transformed-source safetensors dtypes for this PR: `F16`, `BF16`, `F32`, and `F64`. Reject integer, unsigned, boolean, complex, and FP8 transformed sources. FP8 stays fail-closed until there is an explicit round-trip accuracy/reload contract. Apply the rule only to transformed sources; unmatched tensors and MTP tensors may be integer and remain byte/logically preserved.

Boundary matrix checks must be explicit:

- reject any kind other than exactly `read` or `write`;
- require `X` and `Y` rank 2 and equal low-rank width;
- for reads/final head, require both residual dimensions to equal the spec's last-axis size;
- for writes, require both residual dimensions to equal spec axis 0;
- require exact source shape, not merely compatible last axis or element count;
- require output shape equal spec shape and output dtype equal the source disk dtype.

Wrong rank, row count, packed layout, final dimension, integer source, or Exact writer columns then produce a keyed `ValueError` before arbitrary reshape/matmul behavior.

Keep staging placement unchanged. It is acceptable for the empty leased unique stage to exist during header preflight; `export_rebase()` removes it on every failure and no final publication is possible. Do not load all shards or tensors to move validation earlier.

Make source resolution revision-aware:

```python
resolve_local_dir(model_id, *, revision=None)
```

Pass `revision` to `try_to_load_from_cache()`. Normal and GGUF preflight pass `meta.get("revision")`. A missing pinned Hub snapshot must fail closed; do not silently fall back to the current moving cache ref. Local paths ignore the revision argument and remain protected structurally by the two validation layers.

Preserve the existing MoE/MTP rules:

- packed `gate_up_proj` is an inventory rank-3 source and must match its exact `[E, 2I, H]`-style shape;
- packed Layers/LoRA and unsupported Exact arrangements remain rejected;
- shared/router readers use their exact inventory rank-2 shapes;
- MTP remains outside the transform manifest, copied unchanged, with the existing warning;
- tied output is created exactly once, untied in config/index as today.

### B5: one private GGUF owner from reservation through publication

Use one GGUF-specific `threading.Lock`, a frozen private `GGUFJobToken` containing at least a unique ID and normalized name, a private current owner, and a private cache-deletion claim. Keep `_gguf_state`'s public keys unchanged. Do not reuse the model coordinator token and do not hold `OperationType.GGUF_BAKE` throughout conversion; that would block unrelated model operations and would not fit cached conversion.

Recommended narrow helpers:

- `_reserve_gguf_job(name) -> GGUFJobToken`;
- `_gguf_state_snapshot() -> dict`;
- `_gguf_job_is_current(token) -> bool`;
- `_gguf_job_progress(token, step) -> bool`;
- `_finish_gguf_job(token, *, result=None, error=None) -> bool`;
- `_publish_gguf_temp(token, temp_path, final_path) -> bool`;
- `_reserve_gguf_cache_delete()` / exact release in `finally`;
- `_validate_gguf_artifact(path)`.

Every helper compares the exact token, never just `name`. Construct the token and replacement public state before acquiring the lock; install them together under the lock. All readers use the snapshot helper. Public state mutation occurs only under the lock and each job update preserves its own normalized name.

Route order and policy:

1. Validate portable name, GGUF type, and llama.cpp tool requirements. Invalid requests do not alter state.
2. Before cache inspection or any await, atomically reserve the private job and set public state to `running`, step `checking checkpoint`, with old error/result cleared.
3. Any active job conflicts globally with any second GGUF export, same or different name, cached or fresh: HTTP 409, no queue, no state change.
4. If the HF cache is absent, update the exact owner to `baking checkpoint`, then acquire/await `GGUF_BAKE`. Any preflight, coordinator conflict, setup failure, worker failure, or caller cancellation terminally errors/releases only that GGUF token. On caller cancellation, first let `_await_transferred_worker()` cancel and drain the model operation, then finish the GGUF token and re-raise.
5. If cached, retain the same reservation and skip only the model bake.
6. Construct and start the conversion thread inside `try`. Thread construction/start failure terminally errors/releases the token and returns a controlled server error. Once `start()` succeeds, ownership belongs to the worker until its exact terminal update.
7. The worker takes the token as an argument. It checks ownership at entry, before each subprocess phase, and at each publication. A stale worker may finish writing its unique temp, but cannot rename it or mutate public state; `finally` removes the temp.
8. Hold the GGUF lock only for state changes and the final owner-check-plus-`Path.replace()` linearization. Never hold it over an await, subprocess, directory walk, or bake.
9. Success revalidates the exact final result and atomically writes `done`/result and clears the owner. Failure atomically writes only that job's `error` and clears the owner. A stale terminal call is a no-op.

Artifact validation should use `path.stat(follow_symlinks=False)`, require `stat.S_ISREG(mode)`, and require `st_size > 0`. Apply it to:

- an existing/reused base before either reuse or quantization;
- newly generated base temp before replace;
- newly generated quantized temp before replace;
- the exact final result immediately before `done`.

Reject an invalid existing base in place; do not delete or overwrite it implicitly. For a quantized request, do not invoke `llama-quantize` if the base is invalid. Do not add magic-header parsing.

Cache deletion uses the same lock and a private deletion claim. Serialize it globally against all GGUF jobs and other deletions: active job or deletion means 409. Claim under lock, perform the directory walk/removal outside the lock, and clear only the exact claim in `finally`; job reservation also rejects while a deletion is claimed. Before removal, fail closed if the nominal cache subtree contains nested export evidence (a published `*.gguf` or a descendant checkpoint `config.json` below the cache root), so deleting cache `job` cannot erase the valid nested job `job/hf` while inactive.

Do not expose the job ID. `/api/status` and the POST response use the locked public snapshot. A very fast worker may legitimately make the response snapshot already `done`.

Shutdown semantics remain scoped: cancellation during the awaited bake drains and releases both ownership systems. After the conversion thread starts, the request does not cancel it. Do not clear ownership from lifespan while a daemon worker could still publish. A cancellable `Popen` registry and graceful worker join are a separate feature; handled failures/cancellation must still leave no temp artifact, and process termination can only leave a hidden temp rather than a falsely published final.

## 6. Transaction and ownership invariants

### Publication/handoff

1. A gate is permanently bound to the same coordinator/token pair installed in its dispatch.
2. Its ownership query is one coordinator-lock operation: exact current token and not cancelled.
3. Setup-local cleanup cannot mutate gate ownership inputs.
4. Gate cancellation and publication remain exactly-once transitions under the gate lock.
5. Current/uncancelled owner may execute one atomic stage-to-final callback. Cancelled or stale owner executes none.
6. A failed callback never marks the gate published; export cleanup removes the stage.
7. Old-worker release is exact and cannot release a successor.
8. Caller cancellation does not return until claimed late-worker cleanup and exact release complete.
9. The gate retains no handoff, dispatch, model snapshot, bundle, or other heavy setup graph.

### Full checkpoint

1. Every active transform has exactly one inventory-derived `SourceTensorSpec` and one resolved disk source binding.
2. Mapped keys are injective; collisions and missing sources fail before a staged shard write.
3. Every transformed physical occurrence has exact shape and approved floating representation.
4. Runtime dtype may differ from disk dtype; transformed output preserves the disk source dtype and exact shape.
5. Header preflight is bounded-memory, and the arithmetic boundary rechecks the contract.
6. Explicit disk head wins and must validate; tied embedding fallback occurs only when the head is absent and is emitted once.
7. Indexed `weight_map` names, not inode identity or traversal order, determine authoritative tied ownership.
8. MTP/unmatched tensors are not subjected to the transformed-source float rule and remain unchanged.
9. Any incompatibility leaves final absent, removes stage, leaves source bytes untouched, and releases coordinator ownership.

### GGUF

1. At most one private GGUF job owner exists per process, including during a fresh bake.
2. Every public state snapshot corresponds to that owner or to the latest terminal job; no update is name-only.
3. A stale worker can neither update state nor publish a final artifact.
4. Cache delete and job reservation are mutually exclusive and deletion claims are exact-release.
5. No lock is held over slow work.
6. Only a no-follow regular, nonempty exact result may produce `done`.
7. Invalid reused artifacts remain in place and produce `error`, never false success.
8. Thread-start, bake, and conversion failure always release the exact job owner.
9. Public state remains `{state,name,step,error,result}`; the private token never crosses the API.

## 7. Per-file change plan

| File | Planned changes |
|---|---|
| `api/app.py` | Bind both gates to immutable coordinator/token inputs; map reached `PublicationCancelled` to 409; pass recorded revision to source resolution; add GGUF token/lock/state helpers, pre-await reservation, owner-conditional worker publication/state, locked status/response snapshots, artifact validation, thread-start cleanup, and serialized fail-closed cache deletion. |
| `core/model_session.py` | Add and document the single-lock `can_publish(token)` query. Keep existing `is_current`, cancellation, release, and token semantics intact. |
| `core/rebase.py` | Add immutable `SourceTensorSpec`; generate reader/writer/head specs from the operation-private inventory alongside plan entries; reject duplicate plan/spec keys. No family-name heuristic or policy relaxation. |
| `core/editing.py` | Retain header shape/dtype descriptors; map transforms/specs with collision checks; build explicit tied disk bindings; use index-authoritative tied shard; run transformed-key metadata preflight; require/recheck specs in `apply_transform_bounded()`; preserve output dtype/shape, MTP, sharding, leases, staging, and final index validation. |
| `core/model_manager.py` | Add optional revision to `resolve_local_dir()` and pass it to Hub cache resolution while leaving local paths unchanged. |
| `tests/test_model_session_coordinator.py` | Test `can_publish()` for current, cancelled, released, stale/successor, and `None`, including proof that stale release cannot affect successor. |
| `tests/test_operation_coordination.py` | Add real coordinator/handoff normal-export publication seam tests for current, cancellation, stale/superseded, late completion, once-only publication, terminal gate state, stage cleanup, and exact release; add route-level corrupted-source 422/release coverage. |
| `tests/test_transactional_exports.py` | Add real exporter rejecting-gate cleanup test; direct boundary contract matrix; full-source mutation/corruption matrix; legitimate runtime/disk dtype mismatch; source-byte and stage/final assertions. |
| `tests/test_indexed_checkpoints.py` | Add same-key wrong-shape/dtype/packed corruption with valid index totals; tied fallback owner placement; malformed explicit head; mapped-key diagnostics; hardlink/alias preservation. |
| `tests/test_gguf_workers.py` | Make all fresh-bake fakes invoke `publication_guard.publish()`; add B3 fresh current/cancel/stale tests and B5 reservation, stale-state, stale-filesystem, delete, thread-start, failure, success, and reused-base integrity tests. |
| `tests/conftest.py` | Restore/snapshot GGUF public and private state under its lock; assert no live owner/deletion at teardown rather than clearing ownership under a running worker. |
| `tests/helpers.py` | Add a focused safetensors tensor-rewrite helper with optional correct index-total recalculation, without changing unrelated checkpoint contents. |
| `tests/test_capabilities.py` | Replace the existing direct unlocked `_gguf_state` mutation in invalid-name coverage with locked helper/snapshot use and prove validation leaves prior state unchanged. |

## 8. Regression-test plan

### Existing tests that gave false confidence

| Area | Existing coverage | Why it missed the safety seam |
|---|---|---|
| B3 | `tests/test_operation_coordination.py:417-435` | Uses a constant `lambda: True`; no real coordinator ownership is evaluated. |
| B3 | `tests/test_operation_coordination.py:1823-1848`, 3405-3468 | Source-shape/cancellation/release tests do not reach guarded publication. |
| B3/GGUF | `tests/test_gguf_workers.py:81-99`, 281-312, 324-403 | Fake exporters accept `**kwargs`, ignore the gate, and create final cache directly. |
| B4 | `tests/test_transactional_exports.py:18-62` | Arbitrary shapes are accepted or expected rows are derived from the disk tensor; inventory agreement is never asserted. |
| B4 | `tests/test_transactional_exports.py:78-108`, 196-216 | Covers missing keys, mapping, MTP, and same runtime/disk BF16, but not a same-key incompatible source or valid dtype difference. |
| B4 | `tests/test_indexed_checkpoints.py:75-162` | Strong index schema/placement/total tests, but totals do not prove rank, shape, dtype role, or tied source owner. |
| B5 | `tests/test_gguf_workers.py:151-191`, 249-269 | Captures/no-ops workers and manually resets the unversioned dict; it bypasses job ownership. |
| B5 | `tests/test_gguf_workers.py:281-403` | Defers work but executes requests/workers serially. |
| B5 | `tests/test_gguf_workers.py:433-507` | Direct worker tests manually seed public state; valid reuse is covered, but zero-byte/directory existing base and stale owner are not. |

### B3 cases

Use the real `ModelSessionCoordinator`, `OperationHandoff.acquire_scope()`, dispatch, retained thread task, endpoint, cancellation awaiter, and real `PublicationGate`. Patch only `editing.export_rebase` with a guarded exporter seam that creates a unique stage/marker, signals stage-ready, waits for an explicit allow-publication event, calls `publication_guard.publish(lambda: stage.replace(final))`, and removes stage in `finally`.

Add:

- normal current owner: final exists, callback count one, gate `published`, stage absent, coordinator free;
- cancel after stage-ready and before publication: no final, stage absent, gate `cancelled-before-publication`, API task remains pending until late worker rejects/cleans/releases;
- stale/superseded old token: no final, old worker returns controlled conflict, successor remains current, old release does not clear it;
- second `.publish()` after success: second callback is never invoked and state stays `published`;
- real `editing.export_rebase()` with an explicitly rejecting gate: final absent and real unique stage removed;
- fresh GGUF current: guarded fake publishes `<job>/hf/config.json`, then exactly one conversion thread is scheduled;
- fresh GGUF cancelled/stale: HF cache/final stage absent, no conversion thread, exact model and GGUF ownership released.

Use real `Path.replace()` in the seam. It is deterministic, cheap, and exercises the actual atomic callback instead of a boolean stand-in.

### B4 cases

Direct `apply_transform_bounded()` tests, all with a required spec and diagnostic key:

- valid read and valid write;
- wrong rank with identical element count and compatible hidden axis;
- wrong leading shape/row count;
- wrong final dimension;
- exact-shape integer source;
- Exact writer with correct output rows but wrong columns;
- packed wrong leading dimensions such as `[E*2, I, H]` for expected `[E, 2I, H]`;
- unknown transform kind and role/kind mismatch;
- malformed X/Y rank, hidden width, and low-rank width;
- result shape/dtype postcondition.

Full unindexed export tests mutate a transformed same-key source after live inventory agreement was established. Parameterize wrong rank, row/leading shape, final dimension, integer dtype, and packed representation. For each failure assert deliberate keyed `ValueError`, endpoint 422 where routed, final absent, no `.tmp-*` stage except persistent lease metadata, source shard bytes unchanged, and coordinator free for a successor.

Positive cases:

- runtime BF16 with disk F32 succeeds and transformed output remains F32;
- disk F16/BF16 coverage where inexpensive;
- untouched integer MTP sentinel remains unchanged;
- unchanged valid packed Readthrough source succeeds.

Indexed cases:

- wrong-rank and wrong-leading-dimension corruptions with the same element count and unchanged valid total size;
- F32-to-I32 exact-shape corruption, preserving byte total;
- changed-numel corruption with `metadata.total_size` recalculated so the new contract, not the old total guard, rejects it;
- tied runtime, absent disk `lm_head`: embedding is validated as head source, one transformed head is emitted into `weight_map[embed_key]`'s shard, both tie declarations become false, and final total is correct;
- explicit malformed disk head fails without embedding fallback;
- prefix-remapped failure message names the actual mapped disk key;
- duplicate/hard-linked shard names are not inode-deduplicated and final index remains valid.

### B5 cases

- A reserves a fresh bake; cached B with same and different names receives 409 and starts neither bake nor conversion.
- Two simultaneous same-name reservations and two different-name reservations each produce exactly one owner.
- Same-name B arriving after A's cache publication but before A's conversion-thread start still gets 409.
- Old progress/success/error calls after a newer reservation all return false and leave the newer public snapshot unchanged.
- Old subprocess completes after a newer reservation: its temp is cleaned, no final rename occurs, and newer state is unchanged.
- Active job makes cache delete return 409; an active deletion claim makes export return 409; two deletes serialize.
- Inactive cache delete refuses nested exported `.gguf`/checkpoint evidence.
- Fresh bake exception, preflight conflict, and caller cancellation yield terminal error/release and allow a successor.
- Thread construction/start exception yields terminal error/release, never stranded `running`.
- Conversion success/failure changes only the current token's state and releases it.
- Existing base parameterization: zero-byte file, directory, symlink, valid nonempty regular file. Invalid inputs produce `error`, are preserved, and never invoke converter/quantizer or report `done`; valid input reuses and succeeds.
- Newly generated empty/missing output continues to fail; valid generated and quantized temps publish atomically.
- Status during a paused bake is `running` with correct name/step and has no private ID.

## 9. Concurrency-test synchronization design

No test should use `sleep()` to create ordering. Every wait has a bounded timeout for deadlock diagnosis, and every test releases events and joins/drains tasks/threads in `finally`.

| Scenario | Synchronization | Required observation at the held point |
|---|---|---|
| B3 current publication | Worker `threading.Event(stage_ready)` + `allow_publish` | Stage exists; coordinator reports exact operation; final absent until release. |
| B3 cancellation/late completion | Recording real gate sets `cancel_entered`; worker remains on `allow_publish` | Cancelled API task is not done before worker rejects, cleans, and releases. |
| B3 stale owner | Record acquired token; block worker at stage; release old/acquire successor explicitly | Gate rejects old token; successor remains current after old worker exits. |
| Fresh bake overlap | A route runs in its own task/thread; fake guarded exporter signals `bake_blocked` and waits | GGUF public owner already says A/running; cached B returns 409. |
| Post-bake/pre-start gap | Fake thread-start seam signals `start_entered` and blocks A in a separate OS thread | Published cache exists but A still owns GGUF reservation; same-name B returns 409. |
| Simultaneous reservation | Two OS threads released by `threading.Barrier` | Exactly one token/helper call succeeds, one gets conflict, snapshot names winner. |
| Stale filesystem worker | Fake `subprocess.run` writes unique temp, signals `temp_ready`, waits | Finish old/reserve new through helpers, then release; old cannot replace final. |
| Delete versus export | Fake `shutil.rmtree` signals after delete claim and waits | Export conflicts until deletion releases; reverse direction holds job and delete conflicts. |

Bridge worker-thread events into async tests with `asyncio.to_thread(event.wait)` plus `asyncio.wait_for`, or run an endpoint's `asyncio.run()` in a controlled OS thread when the tested synchronous start seam must block. Do not poll shared dicts.

Fixtures must never reset `_gguf_state` or private ownership underneath a live worker. Teardown first releases all barriers and joins/drains work, then asserts no active token/deletion claim, then restores the locked public snapshot.

## 10. Expected base-to-head file set

The expected diff is exactly these 13 files:

```text
api/app.py
core/editing.py
core/model_manager.py
core/model_session.py
core/rebase.py
tests/conftest.py
tests/helpers.py
tests/test_capabilities.py
tests/test_gguf_workers.py
tests/test_indexed_checkpoints.py
tests/test_model_session_coordinator.py
tests/test_operation_coordination.py
tests/test_transactional_exports.py
```

No UI, README/docs, config, dependency lock, workflow, vendored Jacobian Lens, MCP, or other production file should change. If implementation genuinely needs another file, stop and justify it against this plan before expanding the diff.

## 11. Validation commands

### Baseline and focused Codex/Linux validation

Run from the repository root in offline mode:

```bash
test "$(git merge-base HEAD ca343a5cb02dcd191433912d92c087124149910c)" = "ca343a5cb02dcd191433912d92c087124149910c"
test "$(cat requirements/jacobian-lens.commit)" = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

python -m pytest -q tests/test_model_session_coordinator.py -k "publish"
python -m pytest -q tests/test_operation_coordination.py -k "publication or late_worker or corrupted_source"
python -m pytest -q tests/test_transactional_exports.py -k "publication_guard or source_contract"
python -m pytest -q tests/test_indexed_checkpoints.py -k "source_contract or tied"
python -m pytest -q tests/test_gguf_workers.py
python -m pytest -q tests/test_capabilities.py -k "gguf"
```

Name the new tests consistently enough that those `-k` filters select them. Then run complete affected modules and cleanup coverage:

```bash
python -m pytest -q tests/test_model_session_coordinator.py
python -m pytest -q tests/test_operation_coordination.py
python -m pytest -q -W error::RuntimeWarning tests/test_operation_coordination.py
python -m pytest -q tests/test_transactional_exports.py tests/test_indexed_checkpoints.py
python -m pytest -q tests/test_gguf_workers.py tests/test_export_temp_cleanup.py
python -m pytest -q tests/test_capabilities.py tests/test_capability_contract.py
```

Full repository validation:

```bash
python scripts/bootstrap_jlens.py
test "$(git -C vendor/jacobian-lens rev-parse HEAD)" = "581d398613e5602a5af361e1c34d3a92ea82ba8e"
python -m pytest -q tests/
python -m pytest -q vendor/jacobian-lens/tests/
python -m compileall -q api core scripts tests
git diff --check
git diff --check ca343a5cb02dcd191433912d92c087124149910c...HEAD
git diff --name-only ca343a5cb02dcd191433912d92c087124149910c...HEAD
git status --short
```

The GitHub matrix must remain green exactly as defined in `.github/workflows/ci.yml`:

| OS | Python | Transformers | Accelerate | Suites |
|---|---:|---:|---:|---|
| Ubuntu | 3.11 | 5.5.0 | 1.14.0 | J-Wash + pinned Jacobian Lens |
| Ubuntu | 3.11 | 5.14.1 | 1.14.0 | J-Wash + pinned Jacobian Lens |
| Ubuntu | 3.12 | 5.5.0 | 1.14.0 | J-Wash + pinned Jacobian Lens |
| Ubuntu | 3.12 | 5.14.1 | 1.14.0 | J-Wash + pinned Jacobian Lens |
| Windows | 3.12 | 5.14.1 | 1.14.0 | J-Wash + pinned Jacobian Lens |

### Local Windows validation

Use a temporary Python 3.12 virtual environment and UTF-8 mode. PowerShell:

```powershell
$TaskVenv = Join-Path $env:TEMP "jwash-pr13-py312"
py -3.12 -m venv $TaskVenv
$TaskPython = Join-Path $TaskVenv "Scripts\python.exe"
& $TaskPython -m pip install --upgrade pip
& $TaskPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $TaskPython -m pip install -r requirements.txt "transformers==5.14.1" "accelerate==1.14.0" pytest

$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:TOKENIZERS_PARALLELISM = "false"

& $TaskPython -X utf8 scripts/bootstrap_jlens.py
& $TaskPython -X utf8 -m pytest -q tests/test_model_session_coordinator.py
& $TaskPython -X utf8 -m pytest -q tests/test_operation_coordination.py
& $TaskPython -X utf8 -m pytest -q -W error::RuntimeWarning tests/test_operation_coordination.py
& $TaskPython -X utf8 -m pytest -q tests/test_transactional_exports.py tests/test_indexed_checkpoints.py
& $TaskPython -X utf8 -m pytest -q tests/test_gguf_workers.py tests/test_export_temp_cleanup.py
& $TaskPython -X utf8 -m pytest -q tests/
& $TaskPython -X utf8 -m pytest -q vendor/jacobian-lens/tests/
& $TaskPython -X utf8 -m compileall -q api core scripts tests
git diff --check
git diff --name-only ca343a5cb02dcd191433912d92c087124149910c...HEAD
git status --short
```

Do not add a global warning filter or convert known platform-specific skips into unconditional passes. The explicit RuntimeWarning-as-error run must expose new un-awaited coroutine/thread-handoff warnings.

No Node/UI test or build is required because the expected diff has no UI file and the public status schema is unchanged. If implementation unexpectedly touches UI or changes shared public behavior, that is first a scope exception; after justification also run `npm --prefix ui test` and `npm --prefix ui run build`.

## 12. Integration plan with frozen PR #12

Do not implement PR13 on `18c337a70ccffc621444333bd65f710d20d63a8b`, cherry-pick PR #12 into it, or alter PR #12 now. Merge/review PR13 independently on the exact master base. Only after PR13 is merged should the then-current master be merged into the still-draft PR #12 branch.

The frozen PR #12 diff was inspected. Current direct overlaps are:

| File | PR #12 change | Later merge rule |
|---|---|---|
| `api/app.py` | Removes capability `require`/`ensure_unquantized` preflight calls and passes `publication_guard` to both rebase and abliteration workers. | Preserve PR #12's advisory policy and unconditional guard propagation; also preserve PR13's immutable gate binding, revision pinning, GGUF owner/lock, and error handling. Do not re-add capability gating while resolving. |
| `core/editing.py` | Removes capability enforcement from rebase paths, makes abliteration accept a guard parameter, and keeps abliteration fail-closed. | Preserve PR12 advisory semantics and fail-closed Global Projection; retain all PR13 source specs, staging, source validation, and publication behavior. Do not make abliteration reachable by accident. |
| `core/rebase.py` | Removes `ensure_unquantized()` from `build_plan()` and refines unsupported-decoder diagnostics. | Keep those PR12 changes and layer PR13's spec construction/collision checks into `_build_plan_from_inventory()`. |
| `core/model_manager.py` | Changes capability/profile behavior elsewhere in model loading. | Retain that behavior and PR13's revision-aware `resolve_local_dir()` signature/call semantics. |
| `tests/test_capabilities.py` | Updates advisory-capability expectations. | Preserve them; adapt any direct GGUF state setup to the PR13 locked helper/fixture. |
| `tests/test_gguf_workers.py` | Replaces “missing profile rejects” with “missing profile reaches exporter”; its fake still publishes cache directly. | Keep the advisory assertion, but make the fake call the real publication guard and participate in the private GGUF reservation lifecycle. |

At the frozen commit, PR #12 does **not** modify `tests/test_transactional_exports.py`, `tests/test_operation_coordination.py`, `tests/test_model_session_coordinator.py`, or `tests/test_indexed_checkpoints.py`; these should normally merge cleanly, but the union must still be rerun because PR12 makes additional export modes reach deeper execution.

After conflict resolution, run the entire PR13 focused matrix plus PR12 capability/UI tests and the full backend, UI (because PR12 touches it), and pinned Jacobian Lens suites. Re-audit guard propagation for every now-reachable exporter. In particular, PR12's `export_abliteration()` remains deliberately fail-closed; if a future PR implements it, it must independently gain transactional staging, ownership-guarded publication, and target/source verification rather than inheriting assumptions from rebase.

## 13. Explicit scope exclusions

This PR must not include:

- capability-policy changes or any part of PR #12;
- UI capability warnings, model compatibility messaging, or UI refactors;
- dense Qwen3.5/Qwen3.6 support, generic decoder adapters, or model-family-name heuristics;
- Global Projection/abliteration implementation;
- Standard export redesign, Layers/LoRA feature expansion, or a new checkpoint format;
- lens fitting, MCP behavior, Store/continuation, or intervention semantics;
- a general application job framework, public GGUF job IDs, persistent job database, or multi-process locking;
- full GGUF parsing/magic validation;
- cancellable subprocess/process registry or graceful daemon-thread shutdown framework;
- load-time full checkpoint hashing or immutable local-source snapshotting;
- unrelated cleanup, renames, type-system migrations, dependency changes, workflow changes, docs, or UI files.

Do not weaken any existing topology, packing, index, source-key, lease, destination, staging, or publication validation to make new tests pass.

## 14. Acceptance criteria

The implementation is acceptable only when all of the following hold:

### B3

- Both setup functions bind the exact immutable coordinator/token before clearing locals.
- The coordinator current-plus-not-cancelled query is atomic and safe for `None`.
- A legitimate normal export and fresh GGUF bake execute the guard and publish once without `AttributeError` or false 500.
- Cancellation, stale ownership, and late completion execute no final rename, remove the stage, produce the expected gate terminal state, and release only the old operation.
- A successor operation remains current after an old worker exits.
- No additional handoff/gate closure has the mutable-local defect.

### B4

- Every active read, Exact writer, and final head transform has exactly one inventory-derived exact spec.
- Every mapped transformed disk occurrence is header-validated before staged shard writing and revalidated at transform entry.
- Wrong rank, any dimension, row count, packed layout, role, or non-floating source fails with a deliberate keyed `ValueError`/HTTP 422.
- Legitimate BF16 runtime/F32 disk input succeeds and preserves F32 output; transformed output always preserves disk shape/dtype.
- Tied explicit-head and absent-head fallback semantics are unambiguous; indexed fallback uses the authoritative embedding shard and index totals/placement remain valid.
- Integer MTP/unmatched tensors remain preserved.
- Failure leaves final absent, stage removed, source bytes untouched, and coordinator available.
- Hub cache resolution uses the recorded revision and never silently falls forward.

### B5

- A GGUF job is reserved before cache inspection/await and remains the exact owner through conversion terminal state.
- Same-name, different-name, cached, and fresh overlaps all return controlled 409 without starting a second job.
- Every public read/write is lock-protected and every worker update/rename is token-conditional.
- Stale workers cannot alter newer status/result/error or publish a final artifact.
- Bake, cancellation, coordinator conflict, thread-start, converter, and quantizer failure release only the current job and never strand `running`.
- Cache deletion and export reservation are linearized; nested export evidence is not recursively deleted as cache.
- Reused and generated final artifacts are no-follow regular nonempty files before success; zero-byte, directory, and symlink bases never report done.
- Public status shape has no job ID and remains backward-compatible.

### Whole PR

- Base-to-head file list matches section 10 or has a separately reviewed scope justification.
- Focused tests, RuntimeWarning-as-error coordination, full backend suite, pinned Jacobian Lens suite, compileall, whitespace check, and the full CI matrix pass.
- No UI, dependency, workflow, source-checkpoint, PR #12, or GitHub state change is present.

## 15. Risks and subtle cases Codex must not overlook

1. **Lock ordering:** `PublicationGate.publish()` holds the gate lock while calling `coordinator.can_publish()`. Cancellation already takes gate lock before coordinator cancellation. Preserve that one-way order; never add a coordinator callback that acquires a gate lock.
2. **Two coordinator reads are not equivalent:** binding the token but retaining separate `is_cancelled()` and `is_current()` calls leaves an avoidable gap. Use one lock/query.
3. **Publication is not reversible:** if rename wins the gate lock, later cancellation cannot undo it. Tests must accept rename-before-cancel as committed and force cancel-before-rename when testing rejection.
4. **Callback failure:** if `Path.replace()` raises, the gate currently remains pending and the export removes the stage. Do not mark it published before the callback returns.
5. **Retention cycles:** do not capture handoff/dispatch merely to obtain token or stop state. The exact coordinator/token pair is enough.
6. **Controlled stale status:** artificial supersession may reach `PublicationCancelled`; map it deliberately without confusing it with caller cancellation.
7. **Dtype is a usability policy, not identity:** exact runtime/disk dtype equality would reject valid load-time conversion. Conversely, casting integer weights through float is not valid preservation.
8. **Shape means every dimension:** numel, rank alone, or hidden-axis compatibility does not protect packed MoE or writer orientation.
9. **Header preflight is not sufficient alone:** a local file can be replaced after scanning. Keep the transform-boundary check and keyed diagnostics.
10. **Structural validation has a limit:** same-shape/same-dtype value replacement, including a square transpose, cannot be detected without hashes or an immutable snapshot. Revision pinning addresses Hub provenance; value identity for mutable local paths remains an explicit non-goal.
11. **Revision fallback:** if the recorded Hub snapshot is unavailable, fail closed. Falling back to the current cached ref recreates the provenance bug.
12. **Aliases:** do not inode-deduplicate hard-linked indexed shard names. The index names are logical artifacts; validate/write them as names and use `weight_map` for authority.
13. **Tied explicit head:** presence of a malformed explicit head must fail. Falling back to embedding would conceal source corruption.
14. **Index totals can give false confidence:** corruption tests must keep or correctly recompute totals so the new source contract is the rejecting layer.
15. **MTP scope:** a global floating-dtype check would reject legitimate untouched integer sentinels. Check active bindings only.
16. **GGUF lock duration:** never hold the state lock while awaiting, running subprocesses, walking/deleting a tree, or baking a model. Hold it across the final owner check and atomic replace only.
17. **Filesystem ownership matters as much as status ownership:** token-conditional `.update()` calls alone still let a stale worker overwrite a final file. Gate each temp-to-final replace.
18. **Fast completion:** the worker can finish before the POST response snapshot; `done` in that response is valid, not a race failure.
19. **Thread-start failure:** state must be terminally released if either thread construction or `.start()` throws.
20. **No-follow semantics:** `Path.is_file()` follows symlinks. Reuse validation must use no-follow stat if symlinks are to be rejected as required.
21. **Invalid-base preservation:** do not silently delete or regenerate an invalid existing base in this PR; report the exact integrity failure and leave user data untouched.
22. **Nested cache namespace:** valid names allow a job beneath another job's `hf` cache path. Global runtime serialization is insufficient when idle; fail closed on nested export evidence before recursive deletion.
23. **Fixture safety:** tests must never clear a private owner while its daemon worker is alive. Always release barriers/join first.
24. **Shutdown and process count:** `run.py` starts one uvicorn process, so a `threading.Lock` is sufficient for the supported deployment. It is not a multi-worker/process lock. Do not pretend a daemon `subprocess.run` is cancellable without a larger design.
25. **Handled versus crash cleanup:** handled stale/failure paths must unlink unique temps. A process crash may leave a recognized hidden temp/lease, but must not turn it into a published final or public success.
26. **PR #12 conflict resolution:** preserving PR13 integrity must not accidentally reintroduce PR12's capability gates; preserving PR12 advisory behavior must not bypass PR13's source/ownership checks.
