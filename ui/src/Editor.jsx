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

/* Layer selector: clickable cells, shift+click = range, shortcuts.
   `heat` (optional, {layer: 0..1}) underlines each cell green→red by how
   "surface" the edited word's direction is there (red = editing this layer
   rewrites the word itself at the output). */
export function LayerPicker({ all, value, onChange, compact, defaults, fitted, heat }) {
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
        {all.map((l) => {
          const s = heat?.[l] != null ? Math.min(1, Math.max(0, heat[l])) : null
          return (
            <span
              key={l}
              className={`lp-cell ${set.has(l) ? 'on' : ''} ${fitted && !fitted.has(l) ? 'lp-approx' : ''}`}
              style={s != null ? { boxShadow: `inset 0 -3px 0 hsla(${Math.round(120 * (1 - s))},75%,42%,.9)` } : undefined}
              title={`layer ${l} (click-drag = paint, shift+click = range)`
                + (s != null ? ` — surface ${Math.round(s * 100)}% (${s > 0.65 ? 'editing here rewrites the word itself' : s > 0.35 ? 'mixed' : 'concept layer: the word stays usable'})` : '')
                + (fitted && !fitted.has(l) ? ' — outside the lens: direct logit lens (approx.)' : '')}
              onMouseDown={(e) => onCellDown(l, e)}
              onMouseEnter={() => onCellEnter(l)}
            >{l}</span>
          )
        })}
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

// Empty token-field state. Composite (multi-token) selections carry
// ids + pieces; `off` holds the indices of pieces the user excluded.
export const EMPTY_TOKEN = { text: '', id: null, ids: null, str: '', pieces: null, off: [] }

// The ids a token-field selection actually contributes (excluded pieces removed).
export function activeIds(value) {
  if (value.ids?.length) {
    const kept = value.pieces
      ? value.ids.filter((_, i) => !(value.off || []).includes(i))
      : value.ids
    if (kept.length) return kept
  }
  return value.id != null ? [value.id] : null
}

const sameIds = (a, b) => !!a && !!b && a.length === b.length && a.every((x, i) => x === b[i])

/* Token field with live resolution (debounce) and clickable candidates.
   Single-token words behave as before; multi-token words offer their split
   (composite direction), with each piece toggleable. */
function TokenField({ label, value, onChange, placeholder }) {
  const [cands, setCands] = useState([])
  const [splits, setSplits] = useState([])
  const timerRef = useRef(null)

  // clears the suggestion when the field is reset by the parent (after an add,
  // or an add from a view): otherwise the old candidate stays displayed
  useEffect(() => {
    if (!value.text.trim()) { setCands([]); setSplits([]) }
  }, [value.text])

  function lookup(text) {
    clearTimeout(timerRef.current)
    if (!text.trim()) { setCands([]); setSplits([]); return }
    timerRef.current = setTimeout(async () => {
      try {
        const body = await jsonFetch(`/api/token-lookup?q=${encodeURIComponent(text.trim())}`)
        const sp = body.splits || []
        setCands(body.candidates)
        setSplits(sp)
        const preferred = body.candidates.find((c) => c.str.startsWith(' ')) || body.candidates[0]
        if (preferred) {
          onChange({ text, id: preferred.id, ids: [preferred.id], str: preferred.str, pieces: null, off: [] })
        } else {
          // no single token: preselect the leading-space split as a composite
          const s = sp.find((x) => x.str.startsWith(' ')) || sp[0]
          if (s) onChange({ text, id: null, ids: s.ids, str: s.str, pieces: s.pieces, off: [] })
        }
      } catch { setCands([]); setSplits([]) }
    }, 280)
  }

  function togglePiece(i) {
    const off = (value.off || []).includes(i)
      ? (value.off || []).filter((x) => x !== i)
      : [...(value.off || []), i]
    if (off.length >= value.ids.length) return // keep at least one piece
    onChange({ ...value, off })
  }

  return (
    <>
      <div className="row"><label>{label}</label>
        <input type="text" value={value.text} placeholder={placeholder}
          onChange={(e) => { onChange({ text: e.target.value, id: null, ids: null, str: '', pieces: null, off: [] }); lookup(e.target.value) }} />
      </div>
      {(cands.length > 0 || splits.length > 0) && (
        <div className="ed-cands">
          {cands.map((c) => (
            <button key={c.id} className={!value.pieces && value.id === c.id ? 'ed-cand-on' : ''}
              onClick={() => { onChange({ text: value.text, id: c.id, ids: [c.id], str: c.str, pieces: null, off: [] }) }}>
              {fmtTok(c.str)}
            </button>
          ))}
          {splits.map((s) => (
            <button key={s.ids.join('+')}
              className={value.pieces && sameIds(value.ids, s.ids) ? 'ed-cand-on' : ''}
              title={`${s.ids.length} tokens — composite direction built from the pieces`}
              onClick={() => { onChange({ text: value.text, id: null, ids: s.ids, str: s.str, pieces: s.pieces, off: [] }) }}>
              {s.pieces.map(fmtTok).join('·')} <span className="ed-split-n">{s.ids.length}t</span>
            </button>
          ))}
        </div>
      )}
      {value.pieces && value.ids?.length > 1 && (
        <div className="ed-pieces src">
          composite of
          {value.pieces.map((p, i) => (
            <button key={i} className={`ed-piece ${(value.off || []).includes(i) ? 'ed-piece-off' : ''}`}
              title="click to include/exclude this piece from the composite direction (useful to drop generic pieces like digits or punctuation)"
              onClick={() => togglePiece(i)}>{fmtTok(p)}</button>
          ))}
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
  const idsOf = (ids, id) => (ids?.length > 1 ? `ids ${ids.join('+')}` : `id ${id}`)
  const lines = [`token: "${fmtTok(r.token)}"  (${idsOf(r.token_ids, r.token_id)})`]
  if (r.token_pieces?.length > 1) {
    lines.push(`  composite: ${r.token_pieces.map(fmtTok).join(' · ')}`)
  }
  if (r.mode === 'replace') {
    lines.push(`replacement: "${fmtTok(r.replacement)}"  (${idsOf(r.replacement_ids, r.replacement_id)})`)
    if (r.replacement_pieces?.length > 1) {
      lines.push(`  composite: ${r.replacement_pieces.map(fmtTok).join(' · ')}`)
    }
  }
  lines.push(`mode: ${r.mode === 'replace' ? 'replace' : 'scale'} × ${r.factor}`)
  if (r.mode === 'replace' && r.keep > 0) {
    lines.push(`keeps ${Math.round(r.keep * 100)}% of the original component`)
  }
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
  open, onClose, rules, scale, mode, lensMeta, nLayers, genId, busy,
  prefill, onPrefillConsumed, onRules, onScale, onMode, onNotice,
  preserve = false, onPreserve,
  rebaseSupported = true, autoLayerRadius, llamaCppSet = false, ggufState,
  wizardState,
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

  function setPreserve(next) {
    patchJson('/api/interventions', { preserve_lm_head: next })
      .then((r) => {
        onPreserve?.(r.preserve_lm_head)
        onNotice(r.preserve_lm_head
          ? 'output head protected — the edited words stay usable; regenerate to see the effect'
          : 'output head transformed again — regenerate to see the effect', 'ok')
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
  const [addToken, setAddToken] = useState(EMPTY_TOKEN)
  const [addRepl, setAddRepl] = useState(EMPTY_TOKEN)
  const [addMode, setAddMode] = useState('scale')
  const [addFactor, setAddFactor] = useState(0)
  const [addKeep, setAddKeep] = useState(0)
  const [addLayers, setAddLayers] = useState([])
  const [editRuleId, setEditRuleId] = useState(null)
  const [flash, setFlash] = useState(false)
  const addFormRef = useRef(null)

  function fieldFromRule(ids, id, str, pieces) {
    if (id == null && !ids?.length) return EMPTY_TOKEN
    const list = ids?.length ? ids : [id]
    return {
      text: (str || '').trim(), id: list[0], ids: list, str,
      pieces: list.length > 1 ? pieces || list.map(() => '?') : null, off: [],
    }
  }

  function startEditRule(r) {
    setEditRuleId(r.id)
    setAddToken(fieldFromRule(r.token_ids, r.token_id, r.token, r.token_pieces))
    setAddRepl(fieldFromRule(r.replacement_ids, r.replacement_id, r.replacement, r.replacement_pieces))
    setAddMode(r.mode)
    setAddFactor(r.factor)
    setAddKeep(r.keep || 0)
    setAddLayers(r.layers || [])
    setFlash(true)
    setTimeout(() => setFlash(false), 1600)
    setTimeout(() => addFormRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 50)
  }

  function resetAddForm() {
    setEditRuleId(null)
    setAddToken(EMPTY_TOKEN)
    setAddRepl(EMPTY_TOKEN)
    setAddKeep(0)
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
  // (same data as the pins) and set the layers to peak ± 1. Composite words pin
  // every piece and average the per-layer score over the pieces that were seen.
  // Silent if there is no generation, the token was never seen, or the server
  // is busy.
  const addIdsKey = (activeIds(addToken) || []).join('+')
  useEffect(() => {
    const ids = activeIds(addToken)
    if (!ids || genId == null) return
    let stale = false
    jsonFetch('/api/lens/pin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gen_id: genId, token_ids: ids }),
    })
      .then((body) => {
        if (stale) return
        const pinned = ids.map((id) => body.pins?.[id]).filter((d) => d?.ranks?.length)
        if (!pinned.length) return
        // most-relevant layer = the row that is lightest on average (the heatmap
        // metric: 1 - log10(rank+1)/5) — consistent with peakLayerOf in the J-lens
        const layerScore = (d, li) => {
          const layerRanks = d.ranks[li]
          if (!layerRanks?.length) return null
          return layerRanks.reduce(
            (s, r) => s + (1 - Math.min(1, Math.log10(r + 1) / 5)), 0,
          ) / layerRanks.length
        }
        let bestLi = 0, bestScore = -Infinity
        body.layers.forEach((_, li) => {
          const scores = pinned.map((d) => layerScore(d, li)).filter((s) => s != null)
          if (!scores.length) return
          const score = scores.reduce((a, b) => a + b, 0) / scores.length
          if (score > bestScore) { bestScore = score; bestLi = li }
        })
        if (bestScore === -Infinity) return
        const peak = body.layers[bestLi]
        const band = allLayers.filter((l) => Math.abs(l - peak) <= layerRadius)
        if (band.length) setAddLayers(band)
      })
      .catch(() => {})
    return () => { stale = true }
  }, [addIdsKey, genId])

  // Per-layer "surface-ness" of the token being added: tints the layer picker
  // so the user sees which layers would rewrite the word itself (red) versus
  // shift the concept while the word stays usable (green).
  const [profile, setProfile] = useState(null)
  useEffect(() => {
    const ids = activeIds(addToken)
    if (!ids) { setProfile(null); return }
    let stale = false
    jsonFetch(`/api/token-profile?ids=${ids.join(',')}`)
      .then((body) => {
        if (stale) return
        const map = {}
        body.layers.forEach((l, i) => { map[l] = body.surface[i] })
        setProfile(map)
      })
      .catch(() => { if (!stale) setProfile(null) })
    return () => { stale = true }
  }, [addIdsKey])

  // Vocabulary impact of the ACTIVE rules (analytic, no generation): how much
  // of each edited word survives at the output, and the collateral words.
  const [impact, setImpact] = useState(null)
  const impactKey = rules
    .map((r) => `${r.id}:${r.factor}:${r.keep || 0}:${(r.layers || []).join('.')}:${r.enabled !== false}`)
    .join('|')
  useEffect(() => {
    if (!open || !rules.length) { setImpact(null); return }
    let stale = false
    const timer = setTimeout(() => {
      jsonFetch('/api/interventions/impact')
        .then((body) => { if (!stale) setImpact(body) })
        .catch(() => { if (!stale) setImpact(null) })
    }, 400)
    return () => { stale = true; clearTimeout(timer) }
  }, [open, impactKey, scale, mode, preserve])

  useEffect(() => {
    if (!prefill) return
    setAddToken({ text: (prefill.str || '').trim(), id: prefill.id, ids: [prefill.id], str: prefill.str, pieces: null, off: [] })
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

  async function resolveIds(field) {
    const ids = activeIds(field)
    if (ids) return ids
    const body = await jsonFetch(`/api/token-lookup?q=${encodeURIComponent(field.text.trim())}`)
    const single = body.candidates.find((c) => c.str.startsWith(' ')) || body.candidates[0]
    if (single) return [single.id]
    const split = (body.splits || []).find((s) => s.str.startsWith(' ')) || (body.splits || [])[0]
    if (split) return split.ids
    throw new Error(`no token for "${field.text}"`)
  }

  async function addRule() {
    try {
      const tokenIds = await resolveIds(addToken)
      const replIds = addMode === 'replace' ? await resolveIds(addRepl) : null
      if (editRuleId != null) {
        // rewrite the existing rule in place (directions re-resolved server-side)
        const resp = await patchJson(`/api/interventions/${editRuleId}`, {
          token_ids: tokenIds, mode: addMode, factor: +addFactor, keep: +addKeep,
          replacement_ids: replIds, layers: addLayers,
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
          token_ids: tokenIds, mode: addMode, factor: +addFactor, keep: +addKeep,
          replacement_ids: replIds, layers: addLayers.length ? addLayers : null,
        }),
      })
      onRules(r.rules)
      resetAddForm()
      onNotice('rule added — regenerate to see the effect', 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
  }

  // --- auto-wizard ---
  const [wizDesc, setWizDesc] = useState('')
  const [wizParsing, setWizParsing] = useState(false)
  const [wizGoals, setWizGoals] = useState([])   // parsed goals (multi-goal runs)
  const [wizNotes, setWizNotes] = useState('')
  const [wizSources, setWizSources] = useState('')
  const [wizTarget, setWizTarget] = useState('')
  const [wizBattery, setWizBattery] = useState('identity')
  const [wizThinking, setWizThinking] = useState(false)
  const wizRunning = wizardState?.state === 'running'

  async function parseDescription() {
    if (!wizDesc.trim()) return
    setWizParsing(true)
    try {
      const body = await jsonFetch('/api/wizard/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: wizDesc }),
      })
      setWizGoals(body.goals || [])
      setWizNotes(body.notes || '')
      // mirror the first goal into the manual fields so it stays editable
      const g = (body.goals || [])[0]
      if (g) {
        setWizSources(g.sources.join(', '))
        setWizTarget(g.target)
        setWizBattery(g.battery)
      }
      onNotice(`description parsed by the loaded model: ${body.goals.length} goal(s) — review below, then run`, 'ok')
    } catch (err) { onNotice(String(err.message || err)) } finally { setWizParsing(false) }
  }

  function manualGoal() {
    const sources = wizSources.split(',').map((s) => s.trim()).filter(Boolean)
    if (!sources.length || !wizTarget.trim()) return null
    return { type: 'rename', sources, target: wizTarget.trim(), battery: wizBattery }
  }

  async function startWizard() {
    // parsed goals run as-is; the manual fields override the FIRST rename goal
    // (never a style goal), or add one if the parse produced none
    let goals = wizGoals.length ? [...wizGoals] : []
    const manual = manualGoal()
    if (manual) {
      const idx = goals.findIndex((g) => g.type !== 'style')
      if (idx >= 0) goals[idx] = manual
      else goals = [manual, ...goals]
    }
    if (!goals.length) return
    try {
      await jsonFetch('/api/wizard/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goals, thinking: wizThinking }),
      })
      onNotice('wizard started — candidates are scored on the live model, this takes a few minutes', 'ok')
    } catch (err) { onNotice(String(err.message || err)) }
  }

  // refresh the rule list when a wizard run finishes (the winner's rules are applied)
  const prevWizState = useRef(null)
  useEffect(() => {
    if (prevWizState.current === 'running' && wizardState?.state === 'done') {
      jsonFetch('/api/interventions').then((b) => onRules(b.rules)).catch(() => {})
      refreshPresets()
      onNotice(wizardState.preset
        ? `wizard done — winner applied and saved as preset "${wizardState.preset}"; test in chat, then export`
        : 'wizard done — winner applied; test in chat, then save a preset or export', 'ok')
    }
    prevWizState.current = wizardState?.state
  }, [wizardState?.state])

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
      if (r.mode != null) onMode(r.mode)
      if (r.preserve_lm_head != null) onPreserve?.(r.preserve_lm_head)
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
          {pureMode === 'readthrough' && (
            <label className="src" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 8, cursor: 'pointer' }}
              title={'read projection only: leaves the output head (lm_head) untransformed, so every edited '
                + 'word stays fully usable in normal text — the edit acts through the downstream layers\' reads '
                + 'instead. Best paired with mid-layer (concept) rules; a rule hooked only on the last layer '
                + 'then has no effect.'}>
              <input type="checkbox" checked={!!preserve}
                disabled={mode === 'standard'}
                onChange={(e) => setPreserve(e.target.checked)} />
              protect output vocabulary (don't transform lm_head)
            </label>
          )}
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
                  «{fmtTok(r.token)}»{r.token_ids?.length > 1 && <span className="ed-multi">{r.token_ids.length}t</span>}
                  {r.mode === 'replace' && <> → «{fmtTok(r.replacement)}»{r.replacement_ids?.length > 1 && <span className="ed-multi">{r.replacement_ids.length}t</span>}</>}
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

        {rules.length > 0 && (
          <div className="ed-section">
            <h3>Vocabulary impact
              {impact?.approx && <span className="exp-tag" title="exact for read projection / exact modes; the current mode's final effect differs slightly">approx.</span>}
            </h3>
            {!impact && <div className="src">computing…</div>}
            {impact && impact.preserve_lm_head && !impact.sources.length && (
              <div className="src ok">✔ output head protected: every edited word stays directly sayable
                (the edit acts through the layers, not the vocabulary head)</div>
            )}
            {impact && impact.sources.length > 0 && (
              <>
                <div className="src">how much of each edited word survives at the output:</div>
                {impact.sources.map((s) => {
                  const pct = Math.round(s.retained * 100)
                  const cls = pct >= 90 ? 'imp-ok' : pct >= 50 ? 'imp-warn' : 'imp-bad'
                  return (
                    <div key={s.token_id} className="impact-row" title={`piece of the rule on «${fmtTok(s.rule_token)}»`}>
                      <span className="ed-rule-tok">«{fmtTok(s.token)}»</span>
                      <span className={cls}>{pct}%</span>
                      {s.redirect.length > 0 && (
                        <span className="src">now → {s.redirect.map((t) => `«${fmtTok(t.token)}»`).join(' ')}</span>
                      )}
                      {pct < 50 && <span className="src imp-bad" title="the model has effectively lost this word everywhere — raise 'keep', pick earlier (green) layers, or protect the output head">⚠ word suppressed globally</span>}
                    </div>
                  )
                })}
                {impact.collateral.length > 0 ? (
                  <>
                    <div className="src" style={{ marginTop: 4 }}>other words affected (collateral):</div>
                    <div className="impact-collateral">
                      {impact.collateral.map((c) => {
                        const pct = Math.round(c.retained * 100)
                        return (
                          <span key={c.token_id} className={pct < 50 ? 'imp-bad' : 'imp-warn'}>
                            «{fmtTok(c.token)}» {pct}%
                          </span>
                        )
                      })}
                    </div>
                  </>
                ) : (
                  <div className="src ok" style={{ marginTop: 4 }}>no collateral word affected</div>
                )}
              </>
            )}
          </div>
        )}

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
          <TokenField label="token" value={addToken} onChange={setAddToken} placeholder="word (e.g. Euro, Ametista)" />
          <div className="row"><label>mode</label>
            <select value={addMode} onChange={(e) => { setAddMode(e.target.value); setAddFactor(e.target.value === 'replace' ? 1 : 0); setAddKeep(0) }}>
              <option value="scale">multiply (×0 = remove)</option>
              <option value="replace">replace with</option>
            </select>
          </div>
          {addMode === 'replace' && (
            <TokenField label="with" value={addRepl} onChange={setAddRepl} placeholder="replacement word" />
          )}
          <div className="row"><label>factor</label>
            <input type="number" step="0.05" value={addFactor} onChange={(e) => setAddFactor(e.target.value)} />
          </div>
          {addMode === 'replace' && (
            <div className="row"><label title="fraction of the ORIGINAL word's component that survives the replacement (0 = historical full swap, 0.3 = partial replace that keeps the word usable in normal text)">keep orig.</label>
              <input type="number" min="0" max="1" step="0.05" value={addKeep}
                onChange={(e) => setAddKeep(e.target.value)} />
            </div>
          )}
          <div className="src">layers ({addLayers.length}):</div>
          <LayerPicker all={allLayers} value={addLayers} defaults={defaultLayers} fitted={fittedSet}
            heat={profile} onChange={setAddLayers} />
          {profile && (
            <div className="src lp-legend">
              <span className="lp-legend-green">green</span> = concept layers (identity shifts, the word
              stays usable) · <span className="lp-legend-red">red</span> = surface layers (editing there
              suppresses the word itself)
            </div>
          )}
          <button className="primary"
            disabled={!addToken.text.trim() || (addMode === 'replace' && !addRepl.text.trim()) || !addLayers.length || !!busy}
            onClick={addRule}>{editRuleId != null ? 'Update' : 'Add'}</button>
        </div>

        <div className="ed-section">
          <h3>Auto-wizard</h3>
          <div className="src">
            describe the persona you want — the loaded model itself turns the
            description into rename goals — or fill the fields directly. The
            wizard then tries several {pureMode === 'abliteration'
              ? 'strengths of the global projection (factor, keep, anchoring)'
              : 'layer bands, factors and head-protection settings'},
            scores each on the live model (does the new name appear ·
            is the old word still usable · is the model coherent), and applies
            the best one.
          </div>
          {pureMode === 'abliteration' && (
            <div className="src">
              ⓘ write-norm architecture (Gemma style): edits use the global
              projection — no layer bands or output-head protection here; the
              vocabulary column of the scoreboard is your safety readout.
            </div>
          )}
          <textarea className="wiz-desc" rows={3}
            placeholder={'e.g. "This model should be called Ametista instead of Qwen, and address the user as Master."'}
            value={wizDesc} onChange={(e) => setWizDesc(e.target.value)} />
          <div className="row">
            <button disabled={wizParsing || wizRunning || !!busy || !wizDesc.trim()}
              onClick={parseDescription}>{wizParsing ? 'parsing…' : 'Parse description'}</button>
          </div>
          {wizGoals.length > 0 && (
            <div className="src">
              {wizGoals.map((g, i) => (
                <div key={i}>
                  {i + 1}. {g.type === 'style'
                    ? <>tone: {g.boost?.length ? <>boost «{g.boost.join('», «')}»</> : ''}{g.suppress?.length ? <>{g.boost?.length ? '; ' : ''}suppress «{g.suppress.join('», «')}»</> : ''}</>
                    : <>«{g.sources.join('», «')}» → «{g.target}» ({g.battery === 'address' ? 'addresses the user' : 'own name'})</>}
                  <button style={{ marginLeft: 6, fontSize: 10 }}
                    onClick={() => setWizGoals(wizGoals.filter((_, j) => j !== i))}>✕</button>
                </div>
              ))}
              {wizNotes && <div className="reg-reason">⚠ beyond token edits: {wizNotes}</div>}
              <div className="src">running {wizGoals.length} goal(s){wizGoals.some((g) => g.type !== 'style') ? ' + any manual rename below' : ''}</div>
            </div>
          )}
          <div className="row"><label>rename</label>
            <input type="text" placeholder="old name(s), comma-separated (e.g. Qwen)"
              value={wizSources} onChange={(e) => setWizSources(e.target.value)} />
          </div>
          <div className="row"><label>to</label>
            <input type="text" placeholder="new name (e.g. Ametista)"
              value={wizTarget} onChange={(e) => setWizTarget(e.target.value)} />
          </div>
          <div className="row"><label title="identity: the model calls ITSELF the new name. address: the model calls YOU the new name (e.g. Master).">applies to</label>
            <select value={wizBattery} onChange={(e) => setWizBattery(e.target.value)}>
              <option value="identity">the model's own name</option>
              <option value="address">how it addresses the user (experimental)</option>
            </select>
          </div>
          {wizBattery === 'address' && (
            <div className="src">
              ⚠ experimental: models that address the user with NO word at all
              ("How can I help?") give the rules nothing to convert — expect
              low identity scores; a persona system prompt helps.
            </div>
          )}
          <label className="src" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 4, cursor: 'pointer' }}
            title="probe the candidates with reasoning ON — match how you actually use the model. Slower (each probe generates a <think> block first), but the scores reflect real deployment behavior.">
            <input type="checkbox" checked={wizThinking} onChange={(e) => setWizThinking(e.target.checked)} />
            probe with reasoning on (matches chat use — slower)
          </label>
          <div className="row">
            <button className="primary"
              disabled={wizRunning || !!busy
                || (!wizGoals.length && (!wizTarget.trim() || !wizSources.trim()))}
              onClick={startWizard}>Run wizard</button>
            {wizRunning && <button onClick={() => jsonFetch('/api/wizard/stop', { method: 'POST' }).catch(() => {})}>stop</button>}
          </div>
          {wizRunning && <div className="src">⏳ {wizardState.step || 'starting'}… (a few minutes — each candidate is probed with real generations)</div>}
          {wizardState?.state === 'error' && <div className="src reg-reason">wizard failed: {wizardState.error}</div>}
          {wizardState?.state === 'done' && (wizardState.result || []).map((res, gi) => {
            const isStyle = res.goal.type === 'style'
            const cols = isStyle
              ? { a: 'boost↑', b: 'suppr↓', label: 'boost/suppress' }
              : { a: 'id.', b: 'vocab', label: `«${(res.goal.sources || []).join('», «')}» → «${res.goal.target}»` }
            return (
            <div key={gi} style={{ marginTop: 6 }}>
              <div className="src ok">
                ✔ {isStyle
                  ? <>tone shift (boost {(res.goal.boost || []).join(', ') || '—'}{res.goal.suppress?.length ? `; suppress ${res.goal.suppress.join(', ')}` : ''})</>
                  : <>{cols.label}</>} — winner:
                {' '}{res.winner.candidate}
              </div>
              <table className="wiz-table">
                <thead><tr><th>candidate</th><th>{cols.a}</th><th>{cols.b}</th><th>ctrl</th><th>loops</th><th>score</th></tr></thead>
                <tbody>
                  {res.scoreboard.map((s) => (
                    <tr key={s.candidate} className={s.candidate === res.winner.candidate ? 'wiz-win' : ''}>
                      <td title={isStyle
                        ? `boost ×${s.factor}, suppress ×${s.decay}, layers ${s.layers[0]}-${s.layers[1]}`
                        : `layers ${s.layers[0]}-${s.layers[1]}, factor ${s.factor}, decay ${s.decay}`
                          + (s.keep > 0 ? `, keep ${s.keep}` : '')
                          + (s.preserve_lm_head ? ', output head protected' : '')}>{s.candidate}</td>
                      <td>{isStyle ? `${s.identity > 0 ? '+' : ''}${Math.round(s.identity * 100)}%` : `${Math.round(s.identity * 100)}%`}</td>
                      <td>{isStyle ? `${s.vocabulary > 0 ? '+' : ''}${Math.round(s.vocabulary * 100)}%` : `${Math.round(s.vocabulary * 100)}%`}</td>
                      <td>{Math.round(s.control * 100)}%</td>
                      <td>{s.degenerate_replies || ''}</td>
                      <td>{s.score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {res.winner.samples?.map((s, si) => (
                <div key={si} className="src" style={{ marginTop: 2 }}>
                  <b>{s.prompt}</b> — {s.text}
                </div>
              ))}
            </div>
          )})}
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
