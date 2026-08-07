import { useEffect, useMemo, useRef, useState } from 'react'
import { fmtTok } from './tok'

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

const patchJson = (url, body) =>
  jsonFetch(url, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

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

// Two-position toggle: steering (exploration) ↔ the pure-weights mode the
// architecture supports (read projection, or global projection on write-norm
// models like Gemma). The "exact" mode stays reachable through the API; if it
// is active, the thumb sits on the pure side and a click brings it back.
function ModeToggle({ mode, onChange, pureMode = 'readthrough' }) {
  const isPure = mode !== 'standard'
  return (
    <div className={`mode-toggle ${isPure ? 'mt-read' : 'mt-steer'}`} role="radiogroup"
      aria-label="intervention mode">
      <div className="mt-thumb" />
      <button type="button" className={`mt-opt ${!isPure ? 'mt-on' : ''}`}
        aria-pressed={!isPure}
        onClick={() => mode !== 'standard' && onChange('standard')}>
        <svg viewBox="0 0 14 14" width="15" height="15" aria-hidden="true">
          <path d="M2.5 1v5M2.5 10.6V13M7 1v1.4M7 7V13M11.5 1v7.4M11.5 13v-2"
            stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" fill="none" />
          <circle cx="2.5" cy="8.3" r="1.7" fill="currentColor" />
          <circle cx="7" cy="4.7" r="1.7" fill="currentColor" />
          <circle cx="11.5" cy="10.7" r="1.7" fill="currentColor" />
        </svg>
        <span className="mt-lab">Per-layer steering</span>
        <span className="mt-sub">preview only</span>
      </button>
      <button type="button" className={`mt-opt ${isPure ? 'mt-on' : ''}`}
        aria-pressed={isPure}
        onClick={() => mode !== pureMode && onChange(pureMode)}>
        <svg viewBox="0 0 14 14" width="15" height="15" aria-hidden="true">
          <path d="M1.2 7S3.4 3.2 7 3.2 12.8 7 12.8 7 10.6 10.8 7 10.8 1.2 7 1.2 7Z"
            fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
          <circle cx="7" cy="7" r="1.9" fill="currentColor" />
        </svg>
        <span className="mt-lab">{pureMode === 'abliteration' ? 'Global projection' : 'Read projection'}</span>
        <span className="mt-sub">pure-weights · exportable</span>
      </button>
    </div>
  )
}

const MODE_INFO = {
  standard: {
    label: 'Per-layer steering (preview only)',
    tag: 'not exportable',
    help: 'J-space hooks on the chosen layers: the most expressive way to explore, '
      + 'but the export bake only captures ~1-2 % of the effect. Switch to '
      + '"read projection" to export what you see.',
    exportHelp: 'no export in this mode — switch to "read projection" for a '
      + 'faithful pure-weights bake.',
  },
  readthrough: {
    label: 'Read projection (faithful bake)',
    tag: 'pure-weights',
    help: 'every read of the residual downstream of the chosen layers (q/k/v, gate/up, lm_head) '
      + 'sees the transformed residual: the preview = the exported checkpoint. Recommended for '
      + 'removals and replacements. Regenerate after a change.',
    exportHelp: 'change of basis of the downstream reads + lm_head (untied if embeddings are tied). '
      + 'Formats: full checkpoint (standard safetensors), modified layers, or LoRA '
      + '(exact low-rank diff vs the original weights).',
  },
  exact: {
    label: 'Exact compensated (soft factors)',
    tag: 'pure-weights',
    help: 'read projection + counter-transform of the downstream writes: reproduces a '
      + 'hook applied exactly once. ⚠ a full zap/replace makes the inverse singular '
      + '(regularized ≈ read projection) — reserve this mode for partial factors.',
    exportHelp: 'downstream reads + writes transformed, lm_head untied if needed. '
      + 'Formats: full checkpoint, modified layers, or LoRA (exact low-rank diff).',
  },
  abliteration: {
    label: 'Global projection (W_U abliteration)',
    tag: 'pure-weights',
    help: 'W_U projection on every residual write (embed + all layers): the pure-weights '
      + 'mode for architectures where the read projection is unavailable (write norms, '
      + 'Gemma style). Faithful for full removals/replacements; the rules\' layers are '
      + 'ignored (global projection).',
    exportHelp: 'global abliteration × scale: removes/redirects the direction in embed + '
      + 'every o_proj/down_proj. Formats: full checkpoint, modified layers, or LoRA '
      + '(exact delta; embed omitted if embeddings are tied). ⚠ amplifying (factor > 1) '
      + 'stays approximate.',
  },
}

export default function Editor({
  open, onClose, rules, scale, mode, lensMeta, nLayers, genId, generationRunId, busy,
  prefill, onPrefillConsumed, onRules, onScale, onMode, onNotice,
  rebaseSupported = true, autoLayerRadius, llamaCppSet = false, ggufState,
}) {
  // pure-weights mode this architecture can bake (cf. ModeToggle)
  const pureMode = rebaseSupported === false ? 'abliteration' : 'readthrough'
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

  function firePatch(id) {
    const entry = pendingRef.current.get(id)
    if (!entry) return Promise.resolve()
    pendingRef.current.delete(id)
    clearTimeout(entry.timer)
    return patchJson(`/api/interventions/${id}`, entry.body)
      .then((r) => {
        onRules(r.rules)
        setLocalFactors((prev) => { const n = { ...prev }; delete n[id]; return n })
      })
      .catch((err) => {
        setLocalFactors((prev) => { const n = { ...prev }; delete n[id]; return n })
        onNotice(String(err.message || err))
      })
  }

  function schedulePatch(id, body) {
    const prev = pendingRef.current.get(id)
    if (prev) { clearTimeout(prev.timer); body = { ...prev.body, ...body } }
    const timer = setTimeout(() => firePatch(id), 350)
    pendingRef.current.set(id, { timer, body })
  }

  // --- global scale ---
  const [scaleEdit, setScaleEdit] = useState(null)
  const scaleTimer = useRef(null)
  const scaleShown = scaleEdit ?? scale ?? 1

  function setGlobalScale(v) {
    setScaleEdit(v)
    clearTimeout(scaleTimer.current)
    scaleTimer.current = setTimeout(() => flushScale(v), 300)
  }

  function flushScale(v) {
    clearTimeout(scaleTimer.current)
    scaleTimer.current = null
    return patchJson('/api/interventions', { scale: +v })
      .then((r) => { onScale(r.scale); setScaleEdit(null) })
      .catch((err) => { setScaleEdit(null); onNotice(String(err.message || err)) })
  }

  function setMode(next) {
    patchJson('/api/interventions', { mode: next })
      .then((r) => {
        onMode(r.mode)
        onNotice(`${MODE_INFO[r.mode]?.label || r.mode} — regenerate to see the effect.`, 'ok')
      })
      .catch((err) => onNotice(String(err.message || err)))
  }

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
      let last = null
      for (const id of selIds) last = await patchJson(`/api/interventions/${id}`, body)
      if (last) onRules(last.rules)
      onNotice(`${selIds.length} rule(s) updated`, 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
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

  async function resolveId(field) {
    if (field.id != null) return field.id
    const body = await jsonFetch(`/api/token-lookup?q=${encodeURIComponent(field.text.trim())}`)
    if (!body.candidates.length) throw new Error(`no single token for "${field.text}"`)
    return (body.candidates.find((c) => c.str.startsWith(' ')) || body.candidates[0]).id
  }

  async function addRule() {
    try {
      const tokenId = await resolveId(addToken)
      const replId = addMode === 'replace' ? await resolveId(addRepl) : null
      if (editRuleId != null) {
        // rewrite the existing rule in place (directions re-resolved server-side)
        const resp = await patchJson(`/api/interventions/${editRuleId}`, {
          token_id: tokenId, mode: addMode, factor: +addFactor,
          replacement_id: replId, layers: addLayers,
        })
        onRules(resp.rules)
        resetAddForm()
        onNotice('rule updated — regenerate to see the effect', 'ok')
        return
      }
      const r = await jsonFetch('/api/interventions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token_id: tokenId, mode: addMode, factor: +addFactor,
          replacement_id: replId, layers: addLayers.length ? addLayers : null,
        }),
      })
      onRules(r.rules)
      resetAddForm()
      onNotice('rule added — regenerate to see the effect', 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
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
      const r = await jsonFetch(`/api/presets/${encodeURIComponent(name)}/apply`, { method: 'POST' })
      onRules(r.rules)
      if (r.scale != null) onScale(r.scale)
      const warn = (r.warnings || []).join(' ; ')
      onNotice(warn || `preset "${name}" applied`, warn ? 'err' : 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
  }

  // --- export ---
  const [exportFmt, setExportFmt] = useState('full')
  const [exportName, setExportName] = useState('')
  const [ggufType, setGgufType] = useState('q4_k_m')
  useEffect(() => {
    // llama.cpp path removed from Options: don't leave an orphan gguf format
    if (!llamaCppSet && exportFmt === 'gguf') setExportFmt('full')
  }, [llamaCppSet, exportFmt])

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
              onChange={(e) => setGlobalScale(+e.target.value)} style={{ flex: 1 }} />
            <input type="number" step="0.05" value={scaleShown}
              onChange={(e) => setGlobalScale(e.target.value)}
              style={{ width: 64, flexShrink: 0 }} />
          </div>
          <div className="src">
            all alterations × {(+scaleShown).toFixed(2)}
            {+scaleShown === 1 ? ' (neutral)' : +scaleShown === 0 ? ' (all disabled)' : ''}
          </div>
          <div style={{ marginTop: 10 }}>
            <ModeToggle mode={mode || 'standard'} onChange={setMode} pureMode={pureMode} />
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
                  title={r.enabled === false ? 'rule disabled — click to enable (layers kept)' : 'rule active — click to disable without losing the layers'}
                  onClick={async () => {
                    try {
                      const resp = await patchJson(`/api/interventions/${r.id}`, { enabled: r.enabled === false })
                      onRules(resp.rules)
                    } catch (err) { onNotice(String(err.message || err)) }
                  }}>{r.enabled === false ? '○' : '●'}</button>
                <span className="ed-rule-tok" title={ruleTitle(r)}>
                  «{fmtTok(r.token)}»{r.mode === 'replace' ? ` → «${fmtTok(r.replacement)}»` : ''}
                </span>
                <span className="src">×</span>
                <input type="number" step="0.05" className="ed-rule-factor"
                  value={localFactors[r.id] ?? r.factor}
                  onChange={(e) => {
                    setLocalFactors((prev) => ({ ...prev, [r.id]: e.target.value }))
                    schedulePatch(r.id, { factor: +e.target.value })
                  }} />
                <RuleLayerBar all={allLayers} layers={r.layers}
                  onClick={() => setExpandedRule(expandedRule === r.id ? null : r.id)} />
                <button className="ed-rule-del" title="edit this rule (token, replacement, mode, factor, layers) in the form below"
                  onClick={() => startEditRule(r)}>✎</button>
                <button className="ed-rule-del" title="delete this rule" onClick={async () => {
                  try {
                    const resp = await jsonFetch(`/api/interventions/${r.id}`, { method: 'DELETE' })
                    onRules(resp.rules)
                    if (editRuleId === r.id) resetAddForm()
                  } catch (err) { onNotice(String(err.message || err)) }
                }}>✕</button>
              </div>
              {expandedRule === r.id && (
                <div className="ed-rule-layers">
                  <LayerPicker all={allLayers} value={r.layers} defaults={defaultLayers} fitted={fittedSet}
                    onChange={async (layers) => {
                      try {
                        const resp = await patchJson(`/api/interventions/${r.id}`, { layers })
                        onRules(resp.rules)
                      } catch (err) { onNotice(String(err.message || err)) }
                    }} />
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
              <button onClick={async () => {
                try {
                  const resp = await jsonFetch('/api/interventions', { method: 'DELETE' })
                  onRules(resp.rules)
                } catch (err) { onNotice(String(err.message || err)) }
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
                onClick={() => applyGroup({ layers: groupLayers })}>Apply layers</button>
            </div>
            <div className="row">
              <label>factor</label>
              <input type="number" step="0.05" value={groupFactor} placeholder="—"
                onChange={(e) => setGroupFactor(e.target.value)} />
              <button disabled={groupFactor === ''}
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
            disabled={!addToken.text.trim() || (addMode === 'replace' && !addRepl.text.trim()) || !addLayers.length || !!busy}
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
              <button disabled={!!busy} onClick={() => applyPreset(p.name)}>Apply</button>
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
          <h3>Export the edit
            <span className="exp-tag">{MODE_INFO[mode]?.tag || ''}</span>
          </h3>
          {mode === 'standard' ? (
            <div className="src exp-disabled-note" style={{ marginBottom: 6 }}>
              ⚠ export disabled in "per-layer steering": no bake reproduces the
              per-layer hooks faithfully. Switch the mode to
              "{pureMode === 'abliteration' ? 'global projection' : 'read projection'}"
              above for a faithful checkpoint.
            </div>
          ) : null}
          <div className={mode === 'standard' ? 'exp-grid exp-grid-off' : 'exp-grid'}>
            <div className="row"><label>format</label>
              <select value={exportFmt} onChange={(e) => setExportFmt(e.target.value)} disabled={mode === 'standard'}>
                <option value="full">full checkpoint</option>
                <option value="layers">modified layers (safetensors)</option>
                <option value="lora">LoRA (PEFT)</option>
                {llamaCppSet && <option value="gguf">GGUF (via llama.cpp)</option>}
              </select>
            </div>
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
                onChange={(e) => setExportName(e.target.value)} disabled={mode === 'standard'} />
            </div>
            <button className="primary"
              disabled={mode === 'standard' || !exportName.trim()
                || (exportFmt !== 'gguf' && !rules.length) || !!busy
                || ggufState?.state === 'running'}
              onClick={doExport}>Export</button>
            {!llamaCppSet && (
              <div className="src">tip: set the llama.cpp folder in the Options tab to
                unlock a direct GGUF export.</div>
            )}
            {exportFmt === 'gguf' && (
              <div className="row" style={{ marginTop: 2 }}>
                <button disabled={!exportName.trim() || ggufState?.state === 'running'}
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
            <div className="src">{MODE_INFO[mode]?.exportHelp || ''}</div>
          </div>
        </div>
      </div>
    </div>
  )
}
