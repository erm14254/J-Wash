# PR #11 planning report: capability-aware UI

Audit target: [`erm14254/J-Wash`](https://github.com/erm14254/J-Wash) at commit [`e4a4388097a31fc72c72aa8d121d8b2a67519830`](https://github.com/erm14254/J-Wash/commit/e4a4388097a31fc72c72aa8d121d8b2a67519830), which was also the tip of `origin/master` during this audit.

This is a read-only pre-implementation report. No repository file, branch, commit, pull request, or GitHub state was changed.

## 1. Executive recommendation

PR #11 should make the complete `status.capabilities` object—not `loaded.rebase_supported`, model names, lens availability, or static JavaScript copy—the sole authority for first-party edit-mode and export availability.

The smallest coherent implementation has five parts:

1. Add a pure `ui/src/capabilityState.js` adapter. It validates only the UI-required shape, passes valid server decisions and their `reason` strings through unchanged, and fails closed with one explicitly client-owned fallback when the snapshot is missing, malformed, stale, or internally out of sync.
2. Replace the two-position mode toggle with four capability-driven rows. Keep Standard and Readthrough as the normal choices; mark Exact as an advanced but selectable choice; keep Global Projection visible but disabled while the server reports `global_projection_unvalidated`.
3. Drive each export format independently from `capabilities.exports[format]`. Treat llama.cpp configuration and active-rule/name/busy checks as separate local prerequisites.
4. Make `App` own a reusable, race-safe `refreshStatus()` and an invalidate → mutate → refresh capability-transition wrapper. Mode, load, unload, and replacement must never leave the previous snapshot authorized. API-monitor mode additionally needs a narrow `capabilities_changed` notification for external load/unload/mode transitions, followed by a normal `GET /api/status`; reconnect must refresh in case an event was missed.
5. Preserve every existing backend guard and close one narrow enforcement gap found by the audit: direction-changing rule PATCH, preset application, and active-rule generation must fail closed when the captured capability profile is unavailable. This is enforcement of existing semantics, not broader model support.

No intervention PATCH response expansion is needed. A full status refresh is preferable because it atomically reconciles model session, mode, rules, lens metadata, and export decisions and is reusable for every transition. A PATCH-only capability response would fix only a first-party mode change and would not fix load/unload, replacement failure, out-of-order status responses, or external API changes in API-monitor mode.

### Classification

| Classification | Items |
| --- | --- |
| **Production defects/mismatches—must fix** | Legacy boolean drives the UI; false maps to forbidden Global Projection; Exact is collapsed into an ambiguous “pure” side; missing data defaults permissively; export formats ignore per-format decisions; standard export copy is duplicated; active-rule export precondition is wrong; mode/model changes retain stale permissions; API-monitor can remain stale indefinitely; three backend enforcement seams can bypass an unavailable profile. |
| **UX work required to expose the contract** | Four visible mode rows, Exact marked Advanced, canonical reasons next to disabled controls, four visible export rows, separate GGUF local status, disabled LensView edit affordance, and a clear unavailable/current-mode warning. |
| **Optional/nonblocking** | Rich llama.cpp installation validation beyond “configured”; eventual removal of legacy capability fields; a broader status event bus; broader busy-state cleanup outside editor/capability actions; redesign of unrelated editor layout. |

## 2. Exact current mismatch between backend contract and UI

### Authoritative backend behavior

- [`core/capabilities.py:3-28`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/core/capabilities.py#L3-L28) owns canonical public reasons and decision construction.
- [`core/capabilities.py:48-94`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/core/capabilities.py#L48-L94) rejects malformed or contradictory profiles as a whole.
- [`core/capabilities.py:105-159`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/core/capabilities.py#L105-L159) constructs the intrinsic profile and the selected-mode export snapshot.
- [`api/app.py:1082-1117`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/api/app.py#L1082-L1117) returns the complete snapshot as `status.capabilities`. The legacy fields are separately derived compatibility fields.
- [`core/model_session.py:362-389`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/core/model_session.py#L362-L389) captures model profile, model session, lens, and intervention state coherently under the coordinator lock.

### Mismatch inventory

| Current code | Exact mismatch and impact | Classification |
| --- | --- | --- |
| [`App.jsx:1776-1796`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/App.jsx#L1776-L1796) passes `status.loaded.rebase_supported` | The editor never receives `status.capabilities`, so it cannot answer which modes/formats are supported or display canonical reasons. | Must fix |
| [`Editor.jsx:258-265`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L258-L265) defaults `rebaseSupported` to `true` and maps `false` to `abliteration` | Missing data becomes permissive. Unsupported readthrough, quantization, and unavailable profile data can all advertise Global Projection, although the backend always denies it. | Must fix |
| [`Editor.jsx:177-213`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L177-L213) treats every non-standard mode as the same “pure · exportable” side | Exact is not represented distinctly. If Exact is selected externally, the thumb implies the generic pure side and clicking it requests another mode. Unknown/unsupported modes also look exportable. | Must fix |
| [`Editor.jsx:215-255`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L215-L255) contains static availability/export claims | Copy recommends Read Projection when it may be denied, claims Global Projection is the path for unsupported architectures, and lists formats that packed MoE denies. | Must fix; retain only mode semantics as UX copy |
| [`Editor.jsx:582-717`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L582-L717) leaves effectful editor actions usable | Quantized and capability-unavailable models still present usable scale, factor, layer, add, enable, and preset controls. The backend then rejects some of them after a click. | Must fix |
| [`Editor.jsx:741-803`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L741-L803) uses essentially `mode !== "standard"` for export | It ignores `exports.full/layers/lora/gguf`, so packed Layers/LoRA and other denied formats appear valid. | Must fix |
| [`Editor.jsx:745-752`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L745-L752) hard-codes standard-mode denial text | The UI duplicates `standard_not_exportable` instead of displaying the server decision and may recommend an unsupported alternative. | Must fix |
| [`Editor.jsx:759-783`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L759-L783) hides GGUF unless llama.cpp is configured | Server capability and a local tool prerequisite are conflated, so the UI cannot say “the model supports this, but local tooling is not configured.” | UX required |
| [`Editor.jsx:775-779`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L775-L779) checks raw `rules.length`; GGUF omits even that check | Backend export preflight uses coordinated `active_rules`: enabled rules with nonempty layers. A disabled or zero-layer rule is not exportable. | Must fix |
| [`LensView.jsx:577-587`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/LensView.jsx#L577-L587) always presents the ☢ edit affordance | Quantized/unavailable profiles still advertise a usable edit action even though lens visualization itself may remain valid. | UX required |
| [`App.jsx:285-305`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/App.jsx#L285-L305) owns refresh inside one effect | Load/unload and mode handlers cannot explicitly refresh. Fetch failures retain old permissions; overlapping interval requests can resolve out of order. | Must fix |
| [`App.jsx:380-385`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/App.jsx#L380-L385) copies mode/rules in a later effect | A render can combine a new status/capability snapshot with old local mode/rules. Export decisions are mode-specific, so mixed snapshots must not authorize. | Must fix |
| [`App.jsx:703-743`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/App.jsx#L703-L743) does not refresh after load/unload | Normal mode is stale until a poll. API-monitor mode can retain the previous model’s lens, rules, and permissions indefinitely, including after a failed replacement withdrew the old model. | Must fix |
| [`Editor.jsx:334-341`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/ui/src/Editor.jsx#L334-L341) applies PATCH mode locally without status refresh | `capabilities.exports` still describes the previous mode until polling, so switching from an exportable mode to Standard can temporarily leave permissive export controls. | Must fix |
| API-monitor listens only for `api_generation`; the server emits that only at [`api/app.py:2311-2334`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/api/app.py#L2311-L2334) | External load, unload, replacement, and mode changes are invisible until a later successful external generation. Reconnect also does not reconcile missed state. | Must fix |

### Important backend reason precedence

The UI must render the supplied decision independently rather than derive one decision from another:

- No model or an invalid loaded profile is handled before mode-specific export logic, so all exports are `no_model_loaded` or `capability_data_unavailable`.
- A valid profile in Standard mode returns `standard_not_exportable` for every export format—even on a quantized load whose Standard mode decision itself is `quantized_model_unsupported`.
- A supported packed readthrough profile returns Full and GGUF supported, but Layers and LoRA return `packed_moe_format_unsupported`.
- An unsupported selected mode propagates that mode’s reason to all formats.
- An unknown selected mode returns `mode_unsupported` for formats.

These are server semantics; JavaScript must not reproduce their precedence.

## 3. User-visible behavior matrix by model/capability case

| Server/client case and selected mode | Mode presentation | Editor behavior | Export behavior | Visualization |
| --- | --- | --- | --- | --- |
| No model | All four visible and disabled with canonical `no_model_loaded`. | No effectful edits. Cleanup remains available if stale local rows are still displayed. | All four disabled with canonical `no_model_loaded`. | Previously persisted frame viewing may remain read-only; no model-dependent edit prefill. |
| Server returns a valid fail-closed `capability_data_unavailable` snapshot | All four disabled; show the exact server reason. | Block add/edit/factor/scale/enable/preset apply. Allow disable/remove/inspection. | All four disabled with the exact server reason. | Do not gate lens/frame display on editing capability. |
| Capability object missing, malformed, stale, or `exports.mode` does not equal `interventions_mode` | All four fail closed with one UI-owned “Authoritative capability status is unavailable” message. Do not label it as a backend reason code. | Same conservative action policy as above. | All four fail closed until a coherent status arrives. | Read-only display remains. |
| Unquantized unsupported architecture, Standard selected | Standard enabled. Readthrough/Exact disabled with canonical `architecture_unsupported`; Global disabled with the global reason. | Standard per-layer rule editing remains enabled when a lens is loaded. Do **not** blanket-disable the editor because readthrough is unavailable. | All formats disabled with canonical `standard_not_exportable`. | Unchanged. |
| Normal non-packed model, Readthrough selected | Standard and Readthrough enabled. Exact follows its own decision. Global is disabled. | Editing enabled. | Each format follows its supplied decision; normally all server formats are supported. GGUF still has a separate local prerequisite. | Unchanged. |
| Dense Exact-supported model, Exact selected | Exact is visibly selected in the Advanced group and enabled. | Editing enabled. Exact-specific educational copy may remain. | Each format follows the Exact-mode server snapshot. | Unchanged. |
| Packed MoE, Readthrough selected | Readthrough enabled. Exact disabled with `packed_moe_exact_unsupported`. Global disabled. | Standard/readthrough editing remains enabled. | Full and GGUF enabled by the server; Layers and LoRA disabled with `packed_moe_format_unsupported`. | Unchanged. |
| Quantized load, Standard selected | Standard/Readthrough/Exact disabled with `quantized_model_unsupported`; Global disabled with its own global reason. | Block edit creation/application, direction changes, factor/scale, enabling, and preset apply. Permit disabling/removing and read-only inspection. | Render the actual format decisions—normally `standard_not_exportable` because Standard is selected—not a JavaScript-derived quantized reason. | Model/lens visualization remains available where backend production behavior permits it; lens load currently warns about drift rather than forbidding display. |
| Global Projection | Visible in Advanced, never inferred from another denial, disabled with canonical `global_projection_unvalidated`. | No application/edit mutation while it is the current unsupported mode; supported mode choices remain usable for recovery. | All formats disabled according to the supplied snapshot. | Unchanged. |
| Known mode selected externally but its decision is unsupported | Keep it visibly selected, show its canonical reason prominently, and do not silently switch it. Other individually supported modes remain selectable. | Disable effectful mutations; preserve cleanup/read-only actions. | Use the current snapshot’s per-format denials. | Unchanged. |
| GGUF server-supported, llama.cpp setting empty | Mode UI unchanged. | Editing unchanged. | GGUF row says server-supported but locally unavailable; other formats are unaffected. | Unchanged. |
| Selected export format becomes unsupported after a model/mode transition | Mode UI follows new status. | Unchanged. | Keep the format visibly selected so its canonical reason remains clear; disable Export until the user chooses an enabled format. Do not silently substitute another artifact type. | Unchanged. |

## 4. Mode-control design recommendation

Replace `ModeToggle` and the `pureMode` heuristic with a compact capability-driven `ModeSelector` containing four radio-style rows:

1. **Standard — Per-layer steering**
2. **Readthrough — Read projection**
3. **Exact — Exact compensated** (`Advanced` badge)
4. **Abliteration — Global projection** (`Advanced` badge)

Standard and Readthrough should be the always-visible primary group. Exact and Global Projection can sit under an “Advanced modes” disclosure to keep the default editor compact. Auto-open that disclosure when the current mode is Exact or Abliteration, including an externally selected unsupported state.

Exact should be an **advanced selectable mode**. It is meaningfully distinct, can be positively supported on dense audited models, and has caveats around soft factors and singular full removal/replacement. Keeping it API-only would continue to hide authoritative state; promoting it to an ordinary default toggle would overstate its generality.

For every row:

- `enabled` comes only from `capabilities.modes[id].supported === true` after adapter validation.
- A disabled row remains visible and displays `decision.reason` inline through `aria-describedby`; do not rely only on a tooltip on a disabled element.
- A supported row may show semantic UI copy, but no static text may claim that it is exportable or list formats.
- The selected unsupported row remains selected/read-only; selecting a different supported row is the recovery path.
- While a mode change is pending or status is invalidated, disable all mode rows and show “Refreshing authoritative capability status.”
- Remove all model-family and legacy-boolean branching.

The old `MODE_INFO` object may retain labels and mode-mechanics explanations after removing availability, fallback, and format claims. For example, “Exact applies read projection plus compensated writes” is UI education; “Formats: full/layers/LoRA” is policy and must come from the snapshot.

## 5. Export-control design recommendation

Show all four formats as compact radio rows rather than hiding unavailable formats:

| Format | Server authority | Additional local prerequisites |
| --- | --- | --- |
| Full | `capabilities.exports.full` | Nonempty valid name, at least one active rule, no conflicting busy operation. |
| Layers | `capabilities.exports.layers` | Same local prerequisites. |
| LoRA | `capabilities.exports.lora` | Same local prerequisites. |
| GGUF | `capabilities.exports.gguf` | Same prerequisites, llama.cpp configured, and no running GGUF job. The backend remains responsible for validating the directory, conversion script, quantizer, and requested type. |

For each row, retain three distinct pieces of state:

```js
{
  decision,       // exact server object
  serverEnabled,  // decision.supported === true
  local: { ready, reason },
  enabled,        // serverEnabled && local.ready
}
```

Presentation rules:

- If the server denies a format, display its exact canonical `decision.reason` first.
- If GGUF is server-supported but llama.cpp is unset, display a clearly local message such as “Configure llama.cpp in Options.” Do not invent a backend `reason_code`.
- If both server and local checks fail, show both under explicit “Model/mode” and “Local setup” labels rather than replacing the canonical reason.
- Keep GGUF visible when unconfigured.
- Keep a now-unsupported selected format selected for explanation; the Export button remains disabled.
- The primary Export button must require `rules.some(r => r.enabled !== false && Array.isArray(r.layers) && r.layers.length > 0)` for **all** formats, including GGUF.
- GGUF cache cleanup is cleanup, not export authorization. Keep it available when a name is present and no GGUF job is using the cache, regardless of the current capability decision.
- Replace the hard-coded Standard warning with the four supplied `standard_not_exportable` decisions. Optional surrounding text may say “Choose a supported exportable mode,” but must not predict which mode that is.

## 6. Exact handling of canonical reason messages

1. For a structurally valid server decision, render its `reason` string verbatim. Prefixes such as “Exact unavailable:” or “Model/mode:” are allowed; rewriting the explanation is not.
2. Do not add a JavaScript map from `reason_code` to prose. Do not special-case model families, topology, packed layout, or quantization to choose copy.
3. Preserve `reason_code` only for test assertions, stable data attributes, and diagnostics. The visible explanation comes from `reason`.
4. Structural validation should accept future unsupported reason codes if they carry `supported: false` and a nonempty server reason. This keeps the UI forward-compatible without knowing new policy.
5. A supported decision must have strict Boolean `supported: true` and `reason_code: "supported"`; an unsupported decision must not use the supported code. Non-Booleans never authorize.
6. When no valid server decision exists, use one local sentinel with `reason_code: null`, `source: "client-fallback"`, and text such as “Authoritative capability status is unavailable. Refresh and try again.” This is a transport/freshness message, not a recreation of backend policy.
7. Make reasons visible adjacent to controls and available to assistive technology. Disabled native controls often do not emit hover/focus events, so a visible reason or enabled wrapper is required.

## 7. Immediate capability-refresh design after mode mutation

### Alternatives considered

| Approach | Advantage | Problem | Decision |
| --- | --- | --- | --- |
| Return `capabilities` from `PATCH /api/interventions` | One response and atomic for that local mode mutation. | Partial merge into App state; still needs race protection; does not solve load/unload, failed replacement, or external clients. | Do not choose. |
| App-owned explicit `GET /api/status` after mutations | Reuses the complete coherent contract and reconciles the whole session. | One extra request; requires a latest-request guard and fail-closed invalidation. | **Choose.** |
| Continue relying on the 2-second poll | No code seam. | Stale window, indefinite API-monitor staleness, swallowed errors, out-of-order responses. | Reject. |
| Add a narrow WebSocket invalidation event | Immediate external-client signal while preserving API-monitor’s no-poll purpose. | Small backend/client event change. | **Choose in addition to explicit local refresh.** |

### Chosen first-party flow

Create a single App-owned `runCapabilityMutation(mutate)` used by local load, unload, replacement, and mode changes:

1. Mark capability freshness unavailable before issuing the mutation. Disable all effectful edit/export controls immediately and cancel pending debounced factor/scale writes.
2. Execute the mutation.
3. In `finally`, call and await the shared `refreshStatus()`, even after a failed mode PATCH, unload, or replacement.
4. Apply mode, rules, scale, session, lens, and capabilities from the returned status in one React update path. Do not set the new mode from PATCH as authorization.
5. If refresh fails, keep capability state unavailable. A failure must never restore the last known permissive snapshot.

`refreshStatus()` must:

- be the sole function that applies `/api/status` responses;
- assign a monotonically increasing **request epoch** and ignore any older completion, including an older failure;
- invalidate capabilities when the latest request fails;
- structurally validate status before it can authorize;
- apply `status.interventions_mode` and `status.capabilities.exports.mode` together;
- reset model-bound editor state when the accepted `model_session_id` changes;
- return the applied status to mutation callers.

Use request epochs rather than assuming `model_session_id` is globally monotonic: a server restart can reset its counter. Session ID remains the right key for clearing local rules/forms within a running server, but not the only network-response ordering mechanism.

### External/API-monitor flow

Add a narrow WebSocket notification:

```json
{
  "type": "capabilities_changed",
  "model_session_id": 12
}
```

Emit it after:

- successful load from unloaded;
- successful replacement;
- replacement failure if withdrawing the old model changed the session to unloaded;
- successful unload that changed the session;
- successful intervention mode change.

Do not put capability decisions in the event. It is only an invalidation signal; the UI immediately invalidates and fetches `/api/status`. On WebSocket reconnect, also refresh to recover any missed event. Keep the existing `api_generation` refresh behavior.

This addition must use the existing send/connection machinery and must not change WebSocket ownership, cancellation, generation streaming, or connection lifecycle. If maintainers reject even this narrow event, the fallback is a two-second safety poll whenever the editor is open in API-monitor mode; that is less desirable because it weakens the mode’s performance purpose and still has a polling window.

## 8. Exact editor actions to enable or disable and why

Define:

```js
canApplyEdits = capabilityState.valid
  && capabilityState.modes.standard.decision.supported === true
  && capabilityState.selectedMode.decision.supported === true
  && lensAvailable
  && !busy
  && !capabilityTransitionPending
```

Using Standard here does not infer architecture policy. It consumes the backend decision that authorizes creation/resolution of per-layer edit directions. Requiring the selected mode as well prevents effect mutations while an externally selected mode is unavailable. Unsupported-architecture profiles still allow Standard editing because their Standard decision is supported.

| Action | Exact rule | Why |
| --- | --- | --- |
| Select Standard/Readthrough/Exact/Abliteration | Gate the target only on its own mode decision, plus busy/pending state. | Allows recovery from an unsupported current mode without a global Boolean. |
| Change global scale | Require `canApplyEdits`. | It changes the applied effect; backend currently accepts scale-only broadly, but the UI must not imply an unavailable edit can be used. |
| Add a rule | Require `canApplyEdits`, valid form, and nonempty layers. | Backend add requires loaded model+lens and Standard capability. |
| Change token, replacement, rule scale/replace type, or layers | Require `canApplyEdits`. | These recompute direction data and require model+lens; quantized loads are deeply denied. |
| Change one/group factor | Require `canApplyEdits`. | Factor can increase or otherwise alter an effect; it is not cleanup. |
| Apply group layers | Require `canApplyEdits`. | Direction-changing operation. |
| Apply a preset | Require `canApplyEdits`. | It creates edit directions and changes scale. |
| Enable a disabled rule | Require `canApplyEdits`. | Reactivates an effect. |
| Disable an enabled rule | Allow without capability authorization, but still block on a coordinator busy conflict. | Safe reduction/cleanup; production PATCH does not need model/lens. |
| Remove one/all rules | Allow without capability authorization, subject to busy. | Cleanup is intentionally profile-independent in production. |
| Inspect, select, expand, copy, or cancel editing | Always allow. | Read-only/local UI behavior. |
| Save a preset | Do not capability-gate; require loaded session, name, rules, and no busy conflict. | Serializes state; it does not apply an edit. |
| Delete a preset | Do not capability-gate. | Cleanup independent of model capability. |
| Lens/frame/pin/hide/token visualization | Do not capability-gate. | Quantized loads can still expose visualization with a drift warning; editing capability is separate. |
| LensView ☢ prefill | Disable when `canApplyEdits` is false and show the blocking server reason. | Prevents an apparently actionable edit affordance while preserving visualization. |
| Choose/export a format | Require that format’s server decision, then local prerequisites. | Matches authoritative per-format enforcement. |
| Clean GGUF cache | Do not capability-gate; require a name and idle cache/job state. | Cleanup, not export authorization. |

When capability state becomes invalid, cancel queued `firePatch` and scale timers before they can mutate a new session. On accepted `model_session_id` change, clear optimistic factors, selections, expanded rule, edit form, prefill, and export name; the backend already clears coordinated rules/mode on model transition.

## 9. Proposed pure frontend capability adapter API

Add `ui/src/capabilityState.js` with this public surface:

```js
export const MODE_IDS = ['standard', 'readthrough', 'exact', 'abliteration']
export const EXPORT_IDS = ['full', 'layers', 'lora', 'gguf']

export function readCapabilityState(status, {
  fresh = true,
  llamaCppConfigured = false,
  lensAvailable = false,
  busy = false,
  rules = [],
} = {})

export function activeRuleCount(rules)
export function invalidateStatusCapabilities(status)
export function isCapabilityRefreshEvent(message)
```

Suggested return shape:

```js
{
  valid,
  sessionId,
  selectedMode,
  exportsSynchronized,
  fallbackDecision,
  modes: {
    standard: { decision, enabled, selected },
    readthrough: { decision, enabled, selected },
    exact: { decision, enabled, selected },
    abliteration: { decision, enabled, selected },
  },
  editing: {
    enabled,
    blockingDecision,
    localReason,
  },
  formats: {
    full: { decision, serverEnabled, local: { ready, reason }, enabled },
    layers: { /* same */ },
    lora: { /* same */ },
    gguf: { /* same */ },
  },
  actions: {
    changeScale,
    addRule,
    changeRuleDirections,
    changeFactor,
    enableRule,
    disableRule,
    removeRule,
    clearRules,
    applyPreset,
    viewLens,
    cleanGgufCache,
  },
  hasActiveRules,
}
```

Validation rules are structural, not policy reconstruction:

- Require the four known UI mode decisions and four known UI format decisions; ignore extra keys for forward compatibility.
- Require exact Boolean `supported` and nonempty string `reason_code`/`reason`.
- Require supported/code consistency.
- Require a known `status.interventions_mode` and exact equality with `status.capabilities.exports.mode` before any export can authorize.
- On any required-shape failure, return the client fallback for every authorizing decision; never reuse a last-known-good decision.
- Never inspect model ID/family, topology, quantization, lens metadata, legacy flags, or reason-code-specific policy.

Add `ui/src/statusState.js` for testable refresh orchestration:

```js
export function createLatestStatusRefresher({ fetchStatus, applyStatus, invalidate })
export async function runCapabilityMutation({ invalidate, mutate, refresh })
```

The first helper owns request epochs and latest-error invalidation. The second guarantees invalidate → mutate → refresh-in-finally ordering. `App` should use these production helpers directly, so Node tests cover the real synchronization seam without a React test framework.

## 10. Narrow backend changes

### Response contract

No HTTP response schema change is required. Keep `PATCH /api/interventions` returning `{scale, mode}` and keep its existing exact-response compatibility test. The fresh full status remains authoritative.

### Narrow notification change

Add the `capabilities_changed` invalidation event described in section 7. This is a notification, not a capability response and not an alternate authority.

### Concrete enforcement defect found during audit

Despite the intended fail-closed contract, three production seams do not currently consult the captured profile-level decision:

1. Direction-changing rule PATCH reaches `Interventions.update()` with only the deep quantization check ([`api/app.py:1489-1528`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/api/app.py#L1489-L1528), [`core/ablation.py:326-385`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/core/ablation.py#L326-L385)).
2. Preset application uses `ensure_unquantized` but not the Standard decision ([`api/app.py:1739-1824`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/api/app.py#L1739-L1824)).
3. Generation carries `capability_profile` in `GenerationContext` but active intervention attachment does not require the selected mode decision before attaching hooks ([`core/model_manager.py:710-739`](https://github.com/erm14254/J-Wash/blob/e4a4388097a31fc72c72aa8d121d8b2a67519830/core/model_manager.py#L710-L739)).

A malformed/unavailable unquantized profile can therefore block Add and Export correctly yet still receive rules through preset/direction edits and reach Standard hook attachment. Close this gap narrowly:

- Before a direction-changing PATCH, require `modes.standard` from the loaded bundle profile and return canonical rejection as HTTP 422. Preserve factor-only disable and removal behavior.
- Before preset application, require `modes.standard` before mutation.
- In `ModelManager.generate`, when a captured intervention snapshot has active rules, require its captured selected mode before hook attachment. Ordinary generation with no active rules must remain unaffected.
- Do not remove `ensure_unquantized`, topology preflight, export guards, or deep global-projection denial.

This does not enable/disable any new model family or alter any decision; it makes manipulated/stale clients obey the already published contract.

## 11. Exact files expected to change

| File | Change |
| --- | --- |
| `ui/src/capabilityState.js` (new) | Structural adapter, action/format derivation, active-rule helper, client fallback, refresh-event classifier. |
| `ui/src/statusState.js` (new) | Latest-request status controller and invalidate → mutate → refresh wrapper. |
| `ui/src/App.jsx` | Consume full capabilities; centralize status application; explicit transition invalidation/refresh; remove legacy prop; pass derived state; handle `capabilities_changed` and reconnect refresh; reset session-bound UI state. |
| `ui/src/Editor.jsx` | Replace two-position toggle; remove `pureMode`/legacy prop; capability-gate exact actions; canonical reasons; per-format export rows; active-rule prerequisite; cancel pending writes on invalidation/session change. |
| `ui/src/LensView.jsx` | Disable the edit-prefill affordance when editing is unavailable and expose the blocking reason; leave visualization unchanged. |
| `ui/src/styles.css` | Remove obsolete two-position/all-or-nothing export styles; add compact mode/format row, reason, Advanced, pending, and disabled styling. |
| `ui/tests/capabilityState.test.mjs` (new) | Capability profiles, action matrix, canonical reason identity, local/server separation. |
| `ui/tests/statusState.test.mjs` (new) | Request ordering, mutation sequencing, failures, session transition, refresh-event classification. |
| `api/app.py` | Narrow Standard guards for direction-changing PATCH/preset apply; emit capability invalidation events after mode/model transitions. |
| `core/model_manager.py` | Final active-intervention selected-mode require before hook attachment. |
| `tests/test_capability_contract.py` | Guard and cleanup regression tests; mode-change event tests; keep current PATCH response assertion. |
| `tests/test_lifecycle.py` or `tests/test_operation_coordination.py` | Load/unload/replacement/failure event and session-transition coverage at the existing lifecycle seam. |

`ui/package.json` and `ui/package-lock.json` should not change. The existing `node --test tests/*.test.mjs` script already discovers both new test files, and no frontend dependency is justified.

## 12. Test matrix and specific test cases

### Pure Node tests: capability adapter

| # | Fixture/action | Required assertions |
| --- | --- | --- |
| 1 | No model snapshot | All modes/formats/effectful actions disabled; exact `no_model_loaded` reasons retained; cleanup/view allowed. |
| 2 | `null`, missing capabilities, missing key, non-Boolean support, empty reason, contradictory supported/code, or mode mismatch | `valid === false`; every authorizing decision is the client fallback; no legacy/default-true authorization. |
| 3 | Unquantized Standard-only/unsupported architecture | Standard edit actions enabled with lens; Readthrough/Exact disabled using exact canonical reasons; all formats use `standard_not_exportable`. |
| 4 | Normal Readthrough-supported profile | Standard/Readthrough selectable; all formats follow supplied Readthrough decisions. |
| 5 | Exact-supported dense profile | Exact advanced row selectable; selected Exact and its exports remain distinct. |
| 6 | Packed MoE | Readthrough enabled; Exact disabled with `packed_moe_exact_unsupported`. |
| 7 | Packed format restrictions | Full/GGUF enabled; Layers/LoRA disabled with `packed_moe_format_unsupported`. |
| 8 | Quantized profile | Add/edit/factor/scale/enable/preset apply disabled; disable/remove/view remain enabled; reasons retained. |
| 9 | Global projection unavailable | Global remains visible/disabled with the exact global reason; no fallback from Readthrough denial. |
| 10 | Standard selected | All four formats disabled from their supplied `standard_not_exportable` decisions. |
| 11 | Full supported, Layers/LoRA unsupported | Per-format states differ; no mode-wide export Boolean leaks in. |
| 12 | GGUF server-supported, local setting unset | `serverEnabled === true`, `local.ready === false`, effective `enabled === false`; local reason remains separate. |
| 13 | Mode transition with old export snapshot | Invalidation blocks immediately; `exports.mode !== interventions_mode` never authorizes; matching refreshed status restores only its supplied decisions. |
| 14 | Dense → quantized/new-session transition | No reducer/controller state between invalidation and fresh status retains dense edit/export permissions. |
| 15 | Forced known unsupported current mode | It stays visibly selected/denied; supported target modes remain selectable; effectful actions and exports are blocked. |
| 16 | Active-rule predicate | Disabled and zero-layer rules do not count; one enabled nonempty-layer rule satisfies the local precondition; GGUF follows the same rule. |
| 17 | Canonical reason identity | Supply a distinctive server reason and assert exact object/string identity through mode and format presentation; proves no JavaScript mapping. |
| 18 | Server/local reason precedence | Server denial remains visible even when GGUF local setup also fails; local reason alone blocks when server supports. |
| 19 | Selected format becomes unsupported | Selection remains for explanation; Export disables until an enabled row is chosen. |
| 20 | Lens edit affordance | Capability-derived action disabled with the same blocking decision while view action stays enabled. |

### Pure Node tests: status controller

1. A later-started status request resolves first; the older response cannot overwrite it.
2. An older request fails after a newer success; it cannot invalidate the newer state.
3. The latest request fails; capabilities invalidate and remain unavailable until a valid response.
4. `runCapabilityMutation` calls `invalidate`, then mutation, then refresh in `finally` on success.
5. A rejected mode PATCH still refreshes to restore unchanged authoritative state.
6. A successful mutation followed by failed refresh remains fail closed.
7. A pre-transition request cannot apply after the transition epoch increments.
8. Session change clears model-bound form/optimistic state but not general visualization preferences.
9. `api_generation`, `capabilities_changed`, and reconnect are refresh triggers; unrelated WebSocket frames are not.

This small pure controller is preferable to adding jsdom, React Testing Library, or another frontend test framework. The critical App/Editor sequencing must use the tested helper in production rather than merely testing a duplicate reducer.

### Backend tests

Add the following narrow cases:

1. Malformed loaded profile + direction-changing rule PATCH returns 422 with canonical `capability_data_unavailable` and leaves the rule unchanged.
2. Malformed loaded profile + preset apply returns 422 canonically and leaves rules/scale unchanged.
3. Malformed profile + captured active Standard rule is rejected before `Interventions.attach`/hook registration; ordinary generation with no active rule still proceeds to the normal generation seam.
4. Factor-only disable, delete-one, and clear-all remain available as cleanup under unavailable capability data.
5. Unquantized unsupported-architecture profile still permits Standard rule creation/editing; the hardening must not incorrectly require Readthrough.
6. Successful mode mutation emits one `capabilities_changed` event after commit; rejected mode and scale-only requests emit none.
7. Successful load, replacement, and unload emit the event with the resulting session ID.
8. Failed replacement that already withdrew the old model emits the event for the new unloaded session; a load failure that changed no session does not emit a false transition.
9. Event-send failure cannot roll back or fail a committed model/mode mutation; notification is best-effort and the next reconnect refresh recovers.
10. Existing `test_intervention_patch_returns_the_committed_pair` continues to assert exactly `{scale, mode}` because no response expansion is chosen.

Optional but useful contract assertions, even though semantics do not change:

- Standard snapshot returns `standard_not_exportable` for all formats.
- Packed Readthrough returns Full/GGUF supported and Layers/LoRA `packed_moe_format_unsupported`.

### Verification commands for implementation

```bash
pytest -q tests/test_capability_contract.py tests/test_lifecycle.py tests/test_operation_coordination.py
pytest -q
cd ui && npm test
cd ui && npm run build
```

No new package installation should be necessary if the existing UI dependencies are already installed.

## 13. Scope exclusions

Keep PR #11 out of:

- coordinator/session redesign;
- Store/frame publication and continuation behavior;
- WebSocket ownership, cancellation, generation streaming, or connection lifecycle (the one new invalidation event reuses existing machinery only);
- FIT behavior;
- preset file semantics or migration;
- new model-family support;
- Global Projection enablement;
- packed MoE Phase 2;
- GGUF fingerprints/tracker redesign;
- model lookup/download redesign;
- performance tuning unrelated to capability refresh;
- MCP;
- dependency upgrades;
- broad visual/editor redesign.

Do not enable Abliteration, broaden model support, weaken a deep guard, or replace backend rejection with UI-only enforcement.

## 14. Risks and edge cases

| Risk/edge case | Required mitigation |
| --- | --- |
| Old status response arrives after model/mode mutation | Request/transition epoch; only latest current request may apply. |
| Status fetch fails | Invalidate permissions; never keep last good permissions active. Preserve read-only visualization where possible. |
| Server restarts and session ID resets | Use network request epochs for ordering; use session ID only for within-process model-bound state identity. |
| Status contains new session but old local rule/mode state | Apply status/mode/rules/scale through one batched path; derive selected mode from status rather than a separately optimistic mode state. |
| Debounced factor/scale write crosses a model transition | Cancel timers and optimistic values on invalidation/session change. |
| Current mode is unsupported or unknown | Preserve and explain it; block effects/exports; allow supported known mode recovery. Unknown mode fails closed. |
| Future backend reason code | Accept structurally valid unsupported decision and show its server reason; do not enumerate codes in JS. |
| Supported field is truthy but not Boolean | Reject structurally and fail closed. |
| Quantized Standard export reason differs from mode reason | Render the format decision actually supplied; never derive export reason from mode reason. |
| Rules exist but none are active | Local Export prerequisite remains false for every format, including GGUF. |
| llama.cpp path is nonempty but invalid/missing tools | UI may say “configured,” not “ready”; backend validation remains authoritative. Rich readiness is optional follow-up. |
| Capability event is missed or send fails | Refresh on reconnect; normal non-monitor polling still recovers; committed server mutation must not depend on event delivery. |
| Accessibility of disabled reasons | Visible adjacent text/`aria-describedby`; do not rely on disabled-control tooltip behavior. |
| Backend hardening accidentally blocks cleanup | Tests explicitly preserve disable/remove/clear and no-active-rule generation. |

## 15. Suggested branch name

`maintenance/pr11-capability-aware-ui`

## 16. Suggested PR title

`Make editor and export controls capability-aware`

## 17. Step-by-step Codex implementation plan

1. Add `capabilityState.js` with strict structural decision validation, client fallback, mode/export derivation, active-rule counting, editor action derivation, and refresh-event classification. Do not import or encode model-family/reason policy.
2. Add `statusState.js` with the latest-request controller and `runCapabilityMutation` wrapper; unit-test it before wiring React.
3. Refactor `App` so `refreshStatus()` is reusable and is the only status-application path. Apply status, rules, scale, session, lens, and mode-coherent capabilities together; remove the later mixed-snapshot mode copy.
4. Remove `rebaseSupported` and `onMode` from the Editor boundary. Pass the derived capability state, current authoritative mode/session, and App-owned mode mutation callback.
5. Wrap local load, unload/replacement, and mode PATCH in invalidate → mutate → refresh-in-finally. Cancel/reset model-bound editor state at the appropriate transition/session boundary.
6. Add the best-effort `capabilities_changed` event after actual model-session and mode transitions; refresh on that event and WebSocket reconnect. Do not include capability decisions in the event.
7. Replace `ModeToggle` with the four-row selector. Keep Exact selectable under Advanced and keep Global visible/disabled with canonical reason. Strip policy/format claims from `MODE_INFO`.
8. Gate global scale, factor, layers, add/update, enabling, and preset apply from the adapter action state. Preserve disable/remove/clear/read-only operations.
9. Pass edit availability/reason into `LensView` and disable only the ☢ edit-prefill affordance; leave pinning, frame display, hiding, and other visualization behavior untouched.
10. Replace the export select/all-or-nothing grid with four format rows. Wire server decisions, local GGUF status, active-rule/name/busy/job prerequisites, canonical reasons, and stable unsupported selection.
11. Add the narrow backend Standard/profile guards for direction-changing PATCH and preset application, then add the final active-selected-mode generation guard. Preserve all existing deep checks.
12. Update CSS only for the new mode/format/reason/pending states; remove obsolete sliding-toggle and pointer-events-based whole-grid assumptions.
13. Add the pure Node matrices and backend regression cases listed above. Do not add frontend test dependencies.
14. Run targeted backend tests, the full backend suite, UI Node tests, and the Vite production build. Verify manually with fixtures or a development server that canonical reasons are visible for Standard, packed MoE, quantized, unavailable, and Global Projection cases.
15. Review the diff for forbidden policy duplication: search JavaScript for public reason strings, model-family names, quantization branches, `rebase_supported`, and `mode !== "standard"` export authorization. None should remain as capability policy.
16. Confirm the backend still rejects every unsupported manipulated-client request independently, and that notification failure cannot affect committed state.

