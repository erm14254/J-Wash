import { useEffect, useMemo, useRef, useState } from 'react'
import { fmtTok } from './tok'
import { createEditorMutationState, isEditorMutationCancellation, throwIfEditorMutationCancelled } from './editorMutationState.js'
import { selectedExportState, shouldOpenAdvanced } from './capabilityState.js'

// Default layer slice for a new rule, as fractions of the model's layer count
// (aligned with core/ablation.py): 56 layers -> 33 to 44.
const DEFAULT_LAYERS_FRAC_LO = 3 / 5
const DEFAULT_LAYERS_FRAC_HI = 4 / 5
// default radius of the auto-selected slice around an edited token's "peak"
// layer (band = peak ± radius) — adjustable in the Options tab. More inclusive
// = more robust edit in readthrough.
const AUTO_LAYER_RADIUS_DEFAULT = 2

async function jsonFetch(url, options) {
  const res = await fetch(url, options)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || res.statusText)
  return body
}

const patchJson = (url, body, options = {}) =>
  jsonFetch(url, { ...options, method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

/* Layer selector: clickable cells, shift+click = range, shortcuts. */
export function LayerPicker({ all, value, onChange, compact, defaults, fitted }) {
  const [anchor, setAnchor] = useState(null)
  const set = useMemo(() => new Set(value), [value])
  // "paint" drag: the state (on/off) is fixed by the first clicked layer, then
  // applied to the hovered layers as long as the button stays held.
  const dragRef = useRef(null) // { turnOn, sel } during the drag
  useEffect(() => {
    const up = () => { dragRef.current = null }
    window.addEventListener('mouseup', up)
    return () => window.removeEventListener('mouseup', up)
  }, [])
  if (!all.length) return null

  function onCellDown(layer, ev) {
    ev.preventDefault() // prevents text selection during the drag
    if (ev.shiftKey && anchor != null) {
      const [lo, hi] = anchor < layer ? [anchor, layer] : [layer, anchor]
      const range = all.filter((l) => l >= lo && l <= hi)
      const turnOn = !set.has(layer)
      const next = new Set(set)
      range.forEach((l) => (turnOn ? next.add(l) : next.delete(l)))
      setAnchor(layer)
      onChange([...next].sort((a, b) => a - b))
      return
    }
    const turnOn = !set.has(layer)
    const sel = new Set(set)
    turnOn ? sel.add(layer) : sel.delete(layer)
    dragRef.current = { turnOn, sel }
    setAnchor(layer)
    onChange([...sel].sort((a, b) => a - b))
  }

  function onCellEnter(layer) {
    const d = dragRef.current
    if (!d) return
    if (d.turnOn === d.sel.has(layer)) return // already in the desired state
    d.turnOn ? d.sel.add(layer) : d.sel.delete(layer)
    onChange([...d.sel].sort((a, b) => a - b))
  }

  return (
    <div className={`layerpicker ${compact ? 'lp-compact' : ''}`}>
      <div className="lp-cells">
        {all.map((l) => (
          <span
            key={l}
            className={`lp-cell ${set.has(l) ? 'on' : ''} ${fitted && !fitted.has(l) ? 'lp-approx' : ''}`}
            title={`layer ${l} (click-drag = paint, shift+click = range)${fitted && !fitted.has(l) ? ' — outside the lens: direct logit lens (approx.)' : ''}`}
            onMouseDown={(e) => onCellDown(l, e)}
            onMouseEnter={() => onCellEnter(l)}
          >{l}</span>
        ))}
      </div>
      {!compact && (
        <div className="lp-quick">
          {defaults?.length > 0 && <button onClick={() => onChange(defaults)}>default</button>}
          <button onClick={() => onChange([...all])}>all</button>
          <button onClick={() => onChange([])}>none</button>
        </div>
      )}
    </div>
  )
}

/* Mini-bar: one segment per available layer, filled if the rule is active there. */
function RuleLayerBar({ all, layers, onClick }) {
  const set = new Set(layers)
  return (
    <div
      className="rulebar"
      title={layers.length ? `layers ${layers.join(', ')} — click to edit` : 'no layer — inactive rule, click to edit'}
      onClick={onClick}
    >
      {all.map((l) => <span key={l} className={`rb-seg ${set.has(l) ? 'on' : ''}`} />)}
    </div>
  )
}

/* Token field with live resolution (debounce) and clickable candidates. */
function TokenField({ label, value, onChange, placeholder }) {
  const [cands, setCands] = useState([])
  const timerRef = useRef(null)

  // clears the suggestion when the field is reset by the parent (after an add,
  // or an add from a view): otherwise the old candidate stays displayed
  useEffect(() => {
    if (!value.text.trim()) setCands([])
  }, [value.text])

  function lookup(text) {
    clearTimeout(timerRef.current)
    if (!text.trim()) { setCands([]); return }
    timerRef.current = setTimeout(async () => {
      try {
        const body = await jsonFetch(`/api/token-lookup?q=${encodeURIComponent(text.trim())}`)
        setCands(body.candidates)
        const preferred = body.candidates.find((c) => c.str.startsWith(' ')) || body.candidates[0]
        if (preferred) onChange({ text, id: preferred.id, str: preferred.str })
      } catch { setCands([]) }
    }, 280)
  }

  return (
    <>
      <div className="row"><label>{label}</label>
        <input type="text" value={value.text} placeholder={placeholder}
          onChange={(e) => { onChange({ text: e.target.value, id: null, str: '' }); lookup(e.target.value) }} />
      </div>
      {cands.length > 0 && (
        <div className="ed-cands">
          {cands.map((c) => (
            <button key={c.id} className={value.id === c.id ? 'ed-cand-on' : ''}
              onClick={() => { onChange({ text: value.text, id: c.id, str: c.str }); }}>
              {fmtTok(c.str)}
            </button>
          ))}
          {value.id == null && <span className="src">no single token — multi-token word?</span>}
        </div>
      )}
    </>
  )
}

// Layers as compact ranges: [20,21,22,24] → "20-22, 24".
export function formatRanges(layers) {
  const s = [...layers].sort((a, b) => a - b)
  const out = []
  let start = null, prev = null
  for (const l of s) {
    if (start === null) { start = prev = l; continue }
    if (l === prev + 1) { prev = l; continue }
    out.push(start === prev ? `${start}` : `${start}-${prev}`)
    start = prev = l
  }
  if (start !== null) out.push(start === prev ? `${start}` : `${start}-${prev}`)
  return out.join(', ')
}

// Multi-line tooltip for a rule: full (untruncated) tokens, ids, mode, factor,
// layers — especially useful for replacement (words get cut off in the row).
function ruleTitle(r) {
  const lines = [`token: "${fmtTok(r.token)}"  (id ${r.token_id})`]
  if (r.mode === 'replace') {
    lines.push(`replacement: "${fmtTok(r.replacement)}"  (id ${r.replacement_id})`)
  }
  lines.push(`mode: ${r.mode === 'replace' ? 'replace' : 'scale'} × ${r.factor}`)
  const ls = r.layers || []
  lines.push(`layers (${ls.length}): ${ls.length ? formatRanges(ls) : 'none → inactive'}`)
  if (r.enabled === false) lines.push('— rule disabled —')
  return lines.join('\n')
}

// Four independent rows mirror the authoritative server decisions.
function ModeSelector({ state, onChange }) {
  const [advanced, setAdvanced] = useState(shouldOpenAdvanced(state.selectedMode))
  useEffect(() => {
    if (shouldOpenAdvanced(state.selectedMode)) setAdvanced(true)
  }, [state.selectedMode])
  const rows = (ids) => ids.map((id) => {
    const item = state.modes[id]
    const info = MODE_INFO[id]
    const reasonId = `mode-${id}-reason`
    return <label key={id} className={`cap-row ${item.selected ? 'selected' : ''}`}>
      <input type="radio" name="intervention-mode" checked={item.selected}
        disabled={!item.enabled} aria-describedby={!item.enabled ? reasonId : undefined}
        onChange={() => onChange(id)} />
      <span><strong>{info.label}</strong><span className="src">{info.subtitle}</span>
        {!item.decision.supported && <span id={reasonId} className="cap-reason">{item.decision.reason}</span>}
        {item.decision.supported && !item.enabled && <span id={reasonId} className="cap-local">Refreshing or another operation is in progress.</span>}
      </span>
    </label>
  })
  return (
    <div className="mode-selector" role="radiogroup" aria-label="intervention mode">
      {rows(['standard', 'readthrough'])}
      <button type="button" className="advanced-toggle" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}>Advanced {advanced ? '▾' : '▸'}</button>
      {advanced && rows(['exact', 'abliteration'])}
    </div>
  )
}

const MODE_INFO = {
  standard: {
    label: 'Standard', subtitle: 'Per-layer steering',
    help: 'J-space hooks apply on the chosen layers.',
  },
  readthrough: {
    label: 'Readthrough', subtitle: 'Read projection',
    help: 'every read of the residual downstream of the chosen layers (q/k/v, gate/up, lm_head) '
      + 'sees the transformed residual: the preview = the exported checkpoint. Recommended for '
      + 'removals and replacements. Regenerate after a change.',
  },
  exact: {
    label: 'Exact', subtitle: 'Exact compensated',
    help: 'read projection + counter-transform of the downstream writes: reproduces a '
      + 'hook applied exactly once. ⚠ a full zap/replace makes the inverse singular '
      + '(regularized ≈ read projection) — reserve this mode for partial factors.',
  },
  abliteration: {
    label: 'Abliteration', subtitle: 'Global projection',
    help: 'Global projection transforms residual writes throughout the model.',
  },
}

export default function Editor({
  open, onClose, rules, scale, mode, lensMeta, nLayers, genId, generationRunId, busy,
  prefill, onPrefillConsumed, onRules, onScale, onModeChange, onNotice,
  capabilityState, exportFmt, onExportFmtChange, autoLayerRadius, llamaCppSet = false, ggufState,
}) {
  const layerRadius = autoLayerRadius ?? AUTO_LAYER_RADIUS_DEFAULT
  // All the model's layers; those outside the lens use the direct logit lens.
  const allLayers = useMemo(() => {
    if (nLayers) return Array.from({ length: nLayers }, (_, i) => i)
    if (!lensMeta) return []
    if (lensMeta.fitted_layers_all?.length) return lensMeta.fitted_layers_all
    const [lo, hi] = lensMeta.fitted_layers || [0, -1]
    return Array.from({ length: hi - lo + 1 }, (_, i) => lo + i)
  }, [nLayers, lensMeta])

  const fittedSet = useMemo(() => {
    if (!lensMeta?.fitted_layers_all?.length) return null
    return new Set(lensMeta.fitted_layers_all)
  }, [lensMeta])

  const defaultLayers = useMemo(() => {
    const n = allLayers.length
    if (!n) return []
    const lo = Math.floor(n * DEFAULT_LAYERS_FRAC_LO)
    const hi = Math.min(Math.floor(n * DEFAULT_LAYERS_FRAC_HI), n - 1)
    return allLayers.filter((l) => l >= lo && l <= hi)
  }, [allLayers])

  // --- optimistic factor editing + debounced PATCH with flush ---
  const [localFactors, setLocalFactors] = useState({})
  const pendingRef = useRef(new Map()) // ruleId -> {timer, body}
  const mutationStateRef = useRef(createEditorMutationState())

  function resetPendingWrites({ renew = true } = {}) {
    mutationStateRef.current.invalidate()
    if (renew) mutationStateRef.current = createEditorMutationState()
    pendingRef.current.clear()
    scaleTimer.current = null
    setLocalFactors({})
    setScaleEdit(null)
  }

  function reportMutationError(error) {
    if (!isEditorMutationCancellation(error)) onNotice(String(error.message || error))
  }

  function firePatch(id) {
    const entry = pendingRef.current.get(id)
    if (!entry) return Promise.resolve()
    pendingRef.current.delete(id)
    mutationStateRef.current.cancelTimer(entry.timer)
    const state = mutationStateRef.current
    return state.request(
      (signal) => patchJson(`/api/interventions/${id}`, entry.body, { signal }),
      (r) => {
        onRules(r.rules)
        setLocalFactors((prev) => { const n = { ...prev }; delete n[id]; return n })
      },
    )
      .catch((err) => {
        if (isEditorMutationCancellation(err)) return
        setLocalFactors((prev) => { const n = { ...prev }; delete n[id]; return n })
        reportMutationError(err)
      })
  }

  function schedulePatch(id, body) {
    const prev = pendingRef.current.get(id)
    if (prev) { mutationStateRef.current.cancelTimer(prev.timer); body = { ...prev.body, ...body } }
    const timer = mutationStateRef.current.schedule(() => firePatch(id), 350)
    pendingRef.current.set(id, { timer, body })
  }

  // --- global scale ---
  const [scaleEdit, setScaleEdit] = useState(null)
  const scaleTimer = useRef(null)
  const scaleShown = scaleEdit ?? scale ?? 1

  function setGlobalScale(v) {
    setScaleEdit(v)
    if (scaleTimer.current != null) mutationStateRef.current.cancelTimer(scaleTimer.current)
    scaleTimer.current = mutationStateRef.current.schedule(() => flushScale(v), 300)
  }

  function flushScale(v) {
    if (scaleTimer.current != null) mutationStateRef.current.cancelTimer(scaleTimer.current)
    scaleTimer.current = null
    const state = mutationStateRef.current
    return state.request(
      (signal) => patchJson('/api/interventions', { scale: +v }, { signal }),
      (r) => { onScale(r.scale); setScaleEdit(null) },
    )
      .catch((err) => {
        if (isEditorMutationCancellation(err)) return
        setScaleEdit(null)
        reportMutationError(err)
      })
  }

  function setMode(next) {
    onModeChange(next)
      .then(() => onNotice(`${MODE_INFO[next]?.label || next} — regenerate to see the effect.`, 'ok'))
      .catch((err) => onNotice(String(err.message || err)))
  }

  useEffect(() => {
    if (capabilityState.valid) return
    resetPendingWrites()
  }, [capabilityState.valid, capabilityState.sessionId])

  useEffect(() => () => resetPendingWrites({ renew: false }), [])

  async function flushAll() {
    const jobs = [...pendingRef.current.keys()].map(firePatch)
    if (scaleTimer.current != null) jobs.push(flushScale(scaleEdit ?? scale ?? 1))
    await Promise.all(jobs)
  }

  // --- multiple selection ---
  const [selected, setSelected] = useState(new Set())
  const [groupLayers, setGroupLayers] = useState([])
  const [groupFactor, setGroupFactor] = useState('')
  const selIds = [...selected].filter((id) => rules.some((r) => r.id === id))

  async function applyGroup(body) {
    try {
      const state = mutationStateRef.current
      await state.request(async (signal) => {
        let last = null
        for (const id of selIds) last = await patchJson(`/api/interventions/${id}`, body, { signal })
        return last
      }, (last) => {
        if (last) onRules(last.rules)
        onNotice(`${selIds.length} rule(s) updated`, 'ok')
      })
    } catch (err) { reportMutationError(err) }
  }

  // --- per-rule layers (inline picker) ---
  const [expandedRule, setExpandedRule] = useState(null)

  // --- add / edit form (editRuleId != null: the form UPDATES that rule) ---
  const [addToken, setAddToken] = useState({ text: '', id: null, str: '' })
  const [addRepl, setAddRepl] = useState({ text: '', id: null, str: '' })
  const [addMode, setAddMode] = useState('scale')
  const [addFactor, setAddFactor] = useState(0)
  const [addLayers, setAddLayers] = useState([])
  const [editRuleId, setEditRuleId] = useState(null)
  const [flash, setFlash] = useState(false)
  const addFormRef = useRef(null)

  function startEditRule(r) {
    setEditRuleId(r.id)
    setAddToken({ text: (r.token || '').trim(), id: r.token_id, str: r.token })
    setAddRepl(r.replacement_id != null
      ? { text: (r.replacement || '').trim(), id: r.replacement_id, str: r.replacement }
      : { text: '', id: null, str: '' })
    setAddMode(r.mode)
    setAddFactor(r.factor)
    setAddLayers(r.layers || [])
    setFlash(true)
    setTimeout(() => setFlash(false), 1600)
    setTimeout(() => addFormRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
  }

  function resetAddForm() {
    setEditRuleId(null)
    setAddToken({ text: '', id: null, str: '' })
    setAddRepl({ text: '', id: null, str: '' })
  }

  // Set the default slice ONCE per model (layer count). Definitely not on every
  // change of the defaultLayers reference: lensMeta is rebuilt on every
  // /api/status poll, and "no layer" would refill itself.
  const layersInitRef = useRef(0)
  useEffect(() => {
    if (!defaultLayers.length) return
    if (layersInitRef.current === allLayers.length) return
    layersInitRef.current = allLayers.length
    setAddLayers(defaultLayers)
  }, [defaultLayers])

  // Auto-select the "peak" layer of the added token: as soon as a token is
  // resolved and a generation with frames exists, we ask for its per-layer ranks
  // (same data as the pins) and set the layers to peak ± 1. Silent if there is
  // no generation, the token was never seen, or the server is busy.
  useEffect(() => {
    if (addToken.id == null || genId == null || generationRunId == null) return
    let stale = false
    jsonFetch('/api/lens/pin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gen_id: genId, generation_run_id: generationRunId, token_ids: [addToken.id] }),
    })
      .then((body) => {
        if (stale) return
        const d = body.pins?.[addToken.id]
        if (!d?.ranks?.length) return
        // most-relevant layer = the row that is lightest on average (the heatmap
        // metric: 1 - log10(rank+1)/5) — consistent with peakLayerOf in the J-lens
        let bestLi = 0, bestScore = -Infinity
        d.ranks.forEach((layerRanks, li) => {
          if (!layerRanks.length) return
          const score = layerRanks.reduce(
            (s, r) => s + (1 - Math.min(1, Math.log10(r + 1) / 5)), 0,
          ) / layerRanks.length
          if (score > bestScore) { bestScore = score; bestLi = li }
        })
        const peak = body.layers[bestLi]
        const band = allLayers.filter((l) => Math.abs(l - peak) <= layerRadius)
        if (band.length) setAddLayers(band)
      })
      .catch(() => {})
    return () => { stale = true }
  }, [addToken.id, genId, generationRunId])

  useEffect(() => {
    if (!prefill) return
    setAddToken({ text: (prefill.str || '').trim(), id: prefill.id, str: prefill.str })
    setAddMode('scale')
    setAddFactor(0)
    // Pre-select the most-relevant layer (± 1 for an effective edit), otherwise
    // keep the default slice already in place.
    if (prefill.layer != null && allLayers.length) {
      const band = allLayers.filter((l) => Math.abs(l - prefill.layer) <= layerRadius)
      if (band.length) setAddLayers(band)
    }
    setFlash(true)
    setTimeout(() => setFlash(false), 1600)
    setTimeout(() => addFormRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
    onPrefillConsumed()
  }, [prefill])

  async function resolveId(field, signal) {
    if (field.id != null) return field.id
    const body = await jsonFetch(`/api/token-lookup?q=${encodeURIComponent(field.text.trim())}`, { signal })
    if (!body.candidates.length) throw new Error(`no single token for "${field.text}"`)
    return (body.candidates.find((c) => c.str.startsWith(' ')) || body.candidates[0]).id
  }

  async function addRule() {
    const state = mutationStateRef.current
    try {
      await state.request(async (signal) => {
        const tokenId = await resolveId(addToken, signal)
        const replId = addMode === 'replace' ? await resolveId(addRepl, signal) : null
        throwIfEditorMutationCancelled(signal)
        if (editRuleId != null) {
          return patchJson(`/api/interventions/${editRuleId}`, {
            token_id: tokenId, mode: addMode, factor: +addFactor,
            replacement_id: replId, layers: addLayers,
          }, { signal })
        }
        return jsonFetch('/api/interventions', {
          method: 'POST', signal,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
          token_id: tokenId, mode: addMode, factor: +addFactor,
          replacement_id: replId, layers: addLayers.length ? addLayers : null,
          }),
        })
      }, (result) => {
        onRules(result.rules)
        resetAddForm()
        onNotice(editRuleId != null
          ? 'rule updated — regenerate to see the effect'
          : 'rule added — regenerate to see the effect', 'ok')
      })
    } catch (err) { reportMutationError(err) }
  }

  // --- presets ---
  const [presets, setPresets] = useState([])
  const [presetName, setPresetName] = useState('')
  const refreshPresets = () => jsonFetch('/api/presets').then((b) => setPresets(b.presets)).catch(() => {})
  useEffect(() => { if (open) refreshPresets() }, [open])

  async function savePreset() {
    try {
      await flushAll() // ensures the in-flight edits are the ones being saved
      await jsonFetch(`/api/presets/${encodeURIComponent(presetName.trim())}`, { method: 'POST' })
      setPresetName('')
      refreshPresets()
      onNotice('preset saved', 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
  }

  async function applyPreset(name) {
    try {
      const state = mutationStateRef.current
      await state.request(
        (signal) => jsonFetch(`/api/presets/${encodeURIComponent(name)}/apply`, { method: 'POST', signal }),
        (result) => {
          onRules(result.rules)
          if (result.scale != null) onScale(result.scale)
          const warn = (result.warnings || []).join(' ; ')
          onNotice(warn || `preset "${name}" applied`, warn ? 'err' : 'ok')
        },
      )
    } catch (err) { reportMutationError(err) }
  }

  // --- export ---
  const [exportName, setExportName] = useState('')
  const [ggufType, setGgufType] = useState('q4_k_m')
  async function doExport() {
    onNotice('exporting...', 'ok')
    try {
      await flushAll()
      if (exportFmt === 'gguf') {
        const r = await jsonFetch('/api/edit/export-gguf', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: exportName.trim(), gguf_type: ggufType }),
        })
        onNotice(
          `GGUF conversion started (checkpoint ${r.checkpoint}) — progress shown below`,
          'ok',
        )
        return
      }
      const r = await jsonFetch('/api/edit/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: exportFmt, name: exportName.trim() }),
      })
      const nParams = r.modified_params?.length ?? r.modified_params_count
      const warn = (r.warnings || []).join(' ; ')
      onNotice(
        `exported to ${r.out_dir} (${nParams} matrices${r.untied_lm_head ? ', lm_head untied' : ''})`
          + (warn ? ` — ⚠ ${warn}` : ''),
        warn ? 'err' : 'ok',
      )
      if (!warn) setExportName('')
    } catch (err) { onNotice(String(err.message || err)) }
  }

  async function cleanGgufCache() {
    try {
      const r = await jsonFetch('/api/edit/gguf-cache/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: exportName.trim() }),
      })
      onNotice(`cache cleaned — ${(r.freed_bytes / 2 ** 30).toFixed(1)} GB freed`, 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
  }

  if (!open) return null

  return (
    <div className="editor">
      <div className="ed-head">
        <span>☢ Token editor</span>
        <button className="ed-close" onClick={onClose}>✕</button>
      </div>

      <div className="ed-body">
        <div className="ed-section">
          <h3>Global multiplier</h3>
          <div className="row">
            <input type="range" min="0" max="3" step="0.05" value={scaleShown}
              disabled={!capabilityState.actions.changeScale}
              onChange={(e) => setGlobalScale(+e.target.value)} style={{ flex: 1 }} />
            <input type="number" step="0.05" value={scaleShown}
              disabled={!capabilityState.actions.changeScale}
              onChange={(e) => setGlobalScale(e.target.value)}
              style={{ width: 64, flexShrink: 0 }} />
          </div>
          <div className="src">
            all alterations × {(+scaleShown).toFixed(2)}
            {+scaleShown === 1 ? ' (neutral)' : +scaleShown === 0 ? ' (all disabled)' : ''}
          </div>
          <div style={{ marginTop: 10 }}>
            <ModeSelector state={capabilityState} onChange={setMode} />
          </div>
          <div className="src" style={{ marginTop: 8 }}>{MODE_INFO[mode]?.help || ''}</div>
        </div>

        <div className="ed-section">
          <h3>Active rules ({rules.length})</h3>
          {rules.length === 0 && <div className="src">none — add a token below or from the J-lens (☢)</div>}
          {rules.map((r) => (
            <div key={r.id} className={`ed-rule ${r.enabled === false ? 'ed-rule-off' : ''}`}>
              <div className="ed-rule-main">
                <input type="checkbox" checked={selected.has(r.id)}
                  onChange={(e) => {
                    const next = new Set(selected)
                    e.target.checked ? next.add(r.id) : next.delete(r.id)
                    setSelected(next)
                  }} />
                <button className="ed-rule-toggle"
                  disabled={r.enabled === false ? !capabilityState.actions.enableRule : !capabilityState.actions.disableRule}
                  title={r.enabled === false ? 'rule disabled — click to enable (layers kept)' : 'rule active — click to disable without losing the layers'}
                  onClick={async () => {
                    try {
                      const state = mutationStateRef.current
                      await state.request(
                        (signal) => patchJson(`/api/interventions/${r.id}`, { enabled: r.enabled === false }, { signal }),
                        (resp) => onRules(resp.rules),
                      )
                    } catch (err) { reportMutationError(err) }
                  }}>{r.enabled === false ? '○' : '●'}</button>
                <span className="ed-rule-tok" title={ruleTitle(r)}>
                  «{fmtTok(r.token)}»{r.mode === 'replace' ? ` → «${fmtTok(r.replacement)}»` : ''}
                </span>
                <span className="src">×</span>
                <input type="number" step="0.05" className="ed-rule-factor"
                  disabled={!capabilityState.actions.changeFactor}
                  value={localFactors[r.id] ?? r.factor}
                  onChange={(e) => {
                    setLocalFactors((prev) => ({ ...prev, [r.id]: e.target.value }))
                    schedulePatch(r.id, { factor: +e.target.value })
                  }} />
                <RuleLayerBar all={allLayers} layers={r.layers}
                  onClick={() => setExpandedRule(expandedRule === r.id ? null : r.id)} />
                <button className="ed-rule-del" title="edit this rule (token, replacement, mode, factor, layers) in the form below"
                  disabled={!capabilityState.actions.changeRuleDirections}
                  onClick={() => startEditRule(r)}>✎</button>
                <button className="ed-rule-del" title="delete this rule"
                  disabled={!capabilityState.actions.removeRule} onClick={async () => {
                  try {
                    const state = mutationStateRef.current
                    await state.request(
                      (signal) => jsonFetch(`/api/interventions/${r.id}`, { method: 'DELETE', signal }),
                      (resp) => {
                        onRules(resp.rules)
                        if (editRuleId === r.id) resetAddForm()
                      },
                    )
                  } catch (err) { reportMutationError(err) }
                }}>✕</button>
              </div>
              {expandedRule === r.id && (
                <div className="ed-rule-layers">
                  <LayerPicker all={allLayers} value={r.layers} defaults={defaultLayers} fitted={fittedSet}
                    onChange={capabilityState.actions.changeRuleDirections ? async (layers) => {
                      try {
                        const state = mutationStateRef.current
                        await state.request(
                          (signal) => patchJson(`/api/interventions/${r.id}`, { layers }, { signal }),
                          (resp) => onRules(resp.rules),
                        )
                      } catch (err) { reportMutationError(err) }
                    } : () => {}} />
                </div>
              )}
            </div>
          ))}
          {rules.length > 1 && (
            <div className="row" style={{ marginTop: 4 }}>
              <button onClick={() => setSelected(new Set(rules.map((r) => r.id)))}>select all</button>
              {selIds.length > 0 && (
                <button onClick={() => setSelected(new Set())}>deselect all</button>
              )}
              <button disabled={!capabilityState.actions.clearRules} onClick={async () => {
                try {
                  const state = mutationStateRef.current
                  await state.request(
                    (signal) => jsonFetch('/api/interventions', { method: 'DELETE', signal }),
                    (resp) => onRules(resp.rules),
                  )
                } catch (err) { reportMutationError(err) }
              }}>remove all</button>
            </div>
          )}
        </div>

        {selIds.length > 0 && (
          <div className="ed-section ed-group">
            <h3>{selIds.length} rule(s) selected</h3>
            <div className="src">layers to apply (none = inactive rules):</div>
            <LayerPicker all={allLayers} value={groupLayers} defaults={defaultLayers} fitted={fittedSet}
              onChange={setGroupLayers} />
            <div className="row">
              <button className="primary"
                disabled={!capabilityState.actions.changeRuleDirections}
                onClick={() => applyGroup({ layers: groupLayers })}>Apply layers</button>
            </div>
            <div className="row">
              <label>factor</label>
              <input type="number" step="0.05" value={groupFactor} placeholder="—"
                onChange={(e) => setGroupFactor(e.target.value)} />
              <button disabled={groupFactor === '' || !capabilityState.actions.changeFactor}
                onClick={() => applyGroup({ factor: +groupFactor })}>Apply</button>
            </div>
            <button onClick={() => setSelected(new Set())}>deselect</button>
          </div>
        )}

        <div className={`ed-section ${flash ? 'ed-flash' : ''}`} ref={addFormRef}>
          <h3>{editRuleId != null
            ? <>Edit the rule <button style={{ marginLeft: 8, fontSize: 11 }} onClick={resetAddForm}>cancel</button></>
            : 'Add a rule'}</h3>
          <TokenField label="token" value={addToken} onChange={setAddToken} placeholder="word (e.g. Euro)" />
          <div className="row"><label>mode</label>
            <select value={addMode} onChange={(e) => { setAddMode(e.target.value); setAddFactor(e.target.value === 'replace' ? 1 : 0) }}>
              <option value="scale">multiply (×0 = remove)</option>
              <option value="replace">replace with</option>
            </select>
          </div>
          {addMode === 'replace' && (
            <TokenField label="with" value={addRepl} onChange={setAddRepl} placeholder="replacement token" />
          )}
          <div className="row"><label>factor</label>
            <input type="number" step="0.05" value={addFactor} onChange={(e) => setAddFactor(e.target.value)} />
          </div>
          <div className="src">layers ({addLayers.length}):</div>
          <LayerPicker all={allLayers} value={addLayers} defaults={defaultLayers} fitted={fittedSet}
            onChange={setAddLayers} />
          <button className="primary"
            disabled={!capabilityState.actions.addRule || !addToken.text.trim() || (addMode === 'replace' && !addRepl.text.trim()) || !addLayers.length || !!busy}
            onClick={addRule}>{editRuleId != null ? 'Update' : 'Add'}</button>
        </div>

        <div className="ed-section">
          <h3>Presets</h3>
          {presets.map((p) => (
            <div key={p.name} className="reg-item" title={p.model_id ? `saved for ${p.model_id}` : ''}>
              <span className="preset-info">
                <span className="preset-name">{p.name} · {p.n_rules} rule(s)</span>
                {p.model_id && <span className="src preset-model">{p.model_id.replace(/^local\//, '')}</span>}
              </span>
              <button disabled={!capabilityState.actions.applyPreset} onClick={() => applyPreset(p.name)}>Apply</button>
              <button onClick={async () => {
                await jsonFetch(`/api/presets/${encodeURIComponent(p.name)}`, { method: 'DELETE' }).catch(() => {})
                refreshPresets()
              }}>✕</button>
            </div>
          ))}
          <div className="row">
            <input type="text" placeholder="preset name" value={presetName} onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && presetName.trim() && rules.length && !busy) savePreset() }} />
            <button disabled={!presetName.trim() || !rules.length} onClick={savePreset}>Save</button>
          </div>
        </div>

        <div className="ed-section">
          <h3>Export the edit</h3>
          <div className="exp-grid" role="radiogroup" aria-label="export format">
            {Object.entries({ full: 'Full checkpoint', layers: 'Layers', lora: 'LoRA', gguf: 'GGUF' }).map(([id, label]) => {
              const item = selectedExportState(capabilityState, id)
              return <label key={id} className={`cap-row ${exportFmt === id ? 'selected' : ''}`}>
                <input type="radio" name="export-format" checked={exportFmt === id}
                  onChange={() => onExportFmtChange(id)} />
                <span><strong>{label}</strong>
                  <span className={item.serverEnabled ? 'cap-local' : 'cap-reason'}>Model/mode: {item.decision.reason}</span>
                  {!item.local.ready && <span className="cap-local">Local: {item.local.reason}</span>}
                </span>
              </label>
            })}
            {exportFmt === 'gguf' && (
              <div className="row"><label title="bf16/f16 = plain conversion; q* = quantized with llama-quantize. The intermediate HF checkpoint is cached so other types don't re-bake.">type</label>
                <select value={ggufType} onChange={(e) => setGgufType(e.target.value)}>
                  {['q4_k_m', 'q5_k_m', 'q6_k', 'q8_0', 'q3_k_m', 'bf16', 'f16'].map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
            )}
            <div className="row"><label>name</label>
              <input type="text" placeholder="edit name" value={exportName}
                onChange={(e) => setExportName(e.target.value)} />
            </div>
            <button className="primary"
              disabled={!capabilityState.formats[exportFmt].enabled || !exportName.trim()
                || !!busy
                || ggufState?.state === 'running'}
              onClick={doExport}>Export</button>
            {exportFmt === 'gguf' && (
              <div className="row" style={{ marginTop: 2 }}>
                <button disabled={!capabilityState.actions.cleanGgufCache || !exportName.trim() || ggufState?.state === 'running'}
                  title="delete the cached intermediate HF checkpoint of this export (the .gguf files stay)"
                  onClick={cleanGgufCache}>clean cache</button>
              </div>
            )}
            {ggufState?.state === 'running' && (
              <div className="src">⏳ GGUF “{ggufState.name}”: {ggufState.step}…</div>
            )}
            {ggufState?.state === 'done' && ggufState.result && (
              <div className="src ok">✔ GGUF ready: {ggufState.result.gguf}
                {' '}({(ggufState.result.size_bytes / 2 ** 30).toFixed(1)} GB) — the HF
                checkpoint stays cached for other types (clean cache to reclaim).</div>
            )}
            {ggufState?.state === 'error' && (
              <div className="src reg-reason">GGUF failed: {ggufState.error}</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
