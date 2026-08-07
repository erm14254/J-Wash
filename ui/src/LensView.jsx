import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import * as d3 from 'd3'
import { fmtTok } from './tok'
import { LayerPicker, formatRanges } from './Editor.jsx'

// Mouse-following tooltip that never overflows the viewport: measured after
// render (before paint), it flips to the left of / above the cursor if needed.
function Tip({ x, y, children }) {
  const ref = useRef(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const { width, height } = el.getBoundingClientRect()
    let left = x + 12
    let top = y + 14
    if (left + width > window.innerWidth - 8) left = Math.max(8, x - width - 12)
    if (top + height > window.innerHeight - 8) top = Math.max(8, y - height - 14)
    el.style.left = `${left}px`
    el.style.top = `${top}px`
  })
  return (
    <div className="lv-tip" ref={ref} style={{ left: x + 12, top: y + 14 }}>
      {children}
    </div>
  )
}

const trimTok = (s) => (s || '').replace(/^\s+|\s+$/g, '')

function copyText(s) {
  const clean = trimTok(s)
  if (!clean) return
  navigator.clipboard?.writeText(clean).catch(() => {})
}

// Short name of the loaded lens, for the header badge ("which lens do I have?").
function lensLabel(meta) {
  if (!meta) return ''
  if (meta.path) return meta.path.replace(/[/\\]+$/, '').split(/[/\\]/).pop()
  if (meta.filename) {
    const stem = meta.filename.split('/').pop().replace('_jacobian_lens', '').replace('.pt', '')
    return meta.revision ? `${stem} @${meta.revision}` : stem
  }
  return meta.repo_id || 'lens'
}

const CELL_W = 58
const CELL_H = 22
const HEADER_H = 52
const LABEL_W = 44
const MAX_COLS = 400
const PIN_COLORS = ['#e8a13c', '#6cb8e0', '#7ec97e', '#c98bd4', '#d0654f']

/* ============================================================================
   "PINS" VIEW SETTINGS (rank curves + heatmap of a pinned token)

   ⚠ After ANY change here you MUST rebuild the front-end and hard-reload the
   page with Ctrl+F5:
       cd ui && npm run build
   (without a rebuild nothing changes: the browser serves the compiled bundle,
   not this source file)
   ============================================================================ */
const PIN_FILL_WIDTH = true   // true = the block stretches to the full available width
const PIN_PX_PER_TOKEN = 6    // minimum width of a column (position), in px
                              // (raise to 12-20 for wider cells: past the
                              //  available width, horizontal scroll kicks in)
const PIN_MAX_W = 1600        // max graph width (px) — beyond it: horizontal scroll
const PIN_MIN_W = 120         // minimum graph width (px)
const PIN_CURVE_H = 90        // height of the curves graph (px)
const PIN_ROW_H_MIN = 5       // minimum height of a row (layer) in the heatmap (px)
const PIN_ROW_H_MAX = 12      // maximum height of a row (px)
const PIN_MAP_H = 180         // target heatmap height: row height ≈ PIN_MAP_H / n_layers
/* ========================================================================== */

function cellData(frame, layer, maskOn) {
  const d = frame.layers[layer]
  if (!d) return null
  const strs = maskOn ? d.m_strs : d.strs
  const ps = maskOn ? d.m_p : d.p
  const ids = maskOn ? d.m_ids : d.ids
  return {
    str: strs[0],
    rank: maskOn ? d.m_rank[0] : 0,
    p: ps[0],
    tid: ids[0],
    top: strs.map((s, i) => ({ s, p: ps[i], r: maskOn ? d.m_rank[i] : i, tid: ids[i] })),
  }
}

/* First pinned token present in the cell's top-k (concept localization). */
function pinHit(frame, layer, maskOn, pinnedIds) {
  const d = frame.layers[layer]
  if (!d || !pinnedIds.length) return null
  const ids = maskOn ? d.m_ids : d.ids
  for (let i = 0; i < ids.length; i++) {
    if (pinnedIds.includes(ids[i])) return { tid: ids[i], idx: i }
  }
  return null
}

function rankColor(rank) {
  const t = 1 - Math.min(1, Math.log10(rank + 1) / 5)
  return d3.interpolateInferno(0.15 + 0.8 * t)
}

// CJK, kana, hangul, cyrillic, arabic, hebrew...: candidates for "nearest tokens"
const NONLATIN_RE = /[Ѐ-ӿ֐-ۿऀ-෿฀-໿ᄀ-ᇿ⺀-鿿ꀀ-꯿가-힯豈-﫿︰-﹏]/

function parseLayerSpec(spec) {
  if (!spec.trim()) return null
  const set = new Set()
  for (const part of spec.split(',')) {
    const m = part.trim().match(/^(\d+)\s*-\s*(\d+)$/)
    if (m) for (let i = +m[1]; i <= +m[2]; i++) set.add(i)
    else if (part.trim() && !isNaN(+part.trim())) set.add(+part.trim())
  }
  return set.size ? set : null
}

export default function LensView({ frames, tick, genId, generationRunId, lensMeta, onNotice, onEditToken, hidden, onHideToken, maxH, editorOpen, streaming }) {
  const [maskOn, setMaskOn] = useState(true)
  const [view, setView] = useState('agg')
  const [filterInput, setFilterInput] = useState('')
  const [dragLayers, setDragLayers] = useState(null)
  const [aggLimit, setAggLimit] = useState(() => {
    const v = +localStorage.getItem('jlens_agg_limit')
    return v > 0 ? v : 80
  })
  const [autoScroll, setAutoScroll] = useState(() => localStorage.getItem('jlens_autoscroll') !== '0')
  const gridScrollRef = useRef(null)
  const [pinInput, setPinInput] = useState('')
  const [pinCands, setPinCands] = useState([])
  const pinTimerRef = useRef(null)
  const [pinned, setPinned] = useState({})
  const [pinData, setPinData] = useState(null)
  const [tip, setTip] = useState(null)
  const [aggTip, setAggTip] = useState(null)
  const [pinHover, setPinHover] = useState(null) // { tid, li }: layer hovered in a pin block
  const transCache = useRef(new Map()) // tid -> [{id, str, sim}] | null (request in flight)
  const [, setTransTick] = useState(0)

  // available width for the pin blocks (PIN_FILL_WIDTH): measured on the
  // .lv-pins container. editorOpen is an explicit dependency — the 400px
  // editor panel opening/closing is the main reason this width changes, and
  // ResizeObserver alone proved unreliable for it — plus window resizes.
  const pinsRef = useRef(null)
  const [pinsW, setPinsW] = useState(0)
  useEffect(() => {
    const el = pinsRef.current
    if (!el) return
    const measure = () => setPinsW(el.clientWidth)
    // synchronous first (reading clientWidth forces the reflow, and rAF /
    // ResizeObserver never fire in a hidden tab), then again next frame in
    // case the flex layout still settles
    measure()
    const raf = requestAnimationFrame(measure)
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    window.addEventListener('resize', measure)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      window.removeEventListener('resize', measure)
    }
  }, [pinData, editorOpen])

  useEffect(() => { localStorage.setItem('jlens_agg_limit', String(aggLimit)) }, [aggLimit])
  useEffect(() => { localStorage.setItem('jlens_autoscroll', autoScroll ? '1' : '0') }, [autoScroll])

  // horizontal auto-scroll to the right during generation (new tokens appended
  // on the right) — Heatmap view only, toggleable. Deps without gridW (declared
  // below → TDZ): tick changes on every batch of frames and the effect reads
  // scrollWidth live at execution time.
  useEffect(() => {
    if (autoScroll && view === 'grid' && gridScrollRef.current) {
      const el = gridScrollRef.current
      el.scrollLeft = el.scrollWidth
    }
  }, [tick, autoScroll, view])

  // manual pinning of a token by its text, with suggestions (variants with/without
  // a leading space): the pin colors the grid and draws the curves
  function lookupPin(text) {
    clearTimeout(pinTimerRef.current)
    if (!text.trim()) { setPinCands([]); return }
    pinTimerRef.current = setTimeout(async () => {
      try {
        const res = await fetch(`/api/token-lookup?q=${encodeURIComponent(text.trim())}`)
        const body = await res.json()
        setPinCands(body.candidates || [])
      } catch { setPinCands([]) }
    }, 250)
  }
  function addPin(c) {
    if (!c) return
    if (!pinned[c.id]) togglePin(c.id, c.str)
    setPinInput('')
    setPinCands([])
  }
  async function fetchTrans(entries) {
    const missing = [...new Set(
      entries.filter((e) => NONLATIN_RE.test(e.str || '')).map((e) => e.tid)
    )].filter((t) => !transCache.current.has(t))
    if (!missing.length) return
    missing.forEach((t) => transCache.current.set(t, null))
    try {
      const res = await fetch('/api/token-neighbors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_ids: missing, k: 6 }),
      })
      if (!res.ok) throw new Error()
      const body = await res.json()
      Object.entries(body.neighbors).forEach(([tid, list]) => transCache.current.set(+tid, list))
      setTransTick((x) => x + 1)
    } catch {
      missing.forEach((t) => transCache.current.delete(t))
    }
  }

  const transLabel = (tid) => {
    const list = transCache.current.get(tid)
    if (!list?.length) return null
    return list.slice(0, 2).map((n) => fmtTok(n.str)).join(', ')
  }

  const shown = frames.length > MAX_COLS ? frames.slice(-MAX_COLS) : frames
  const layers = useMemo(() => {
    if (!shown.length) return []
    return Object.keys(shown[shown.length - 1].layers).map(Number).sort((a, b) => a - b)
  }, [tick, frames])

  const filterSet = useMemo(() => parseLayerSpec(filterInput), [filterInput])
  const filtered = filterSet ? layers.filter((l) => filterSet.has(l)) : layers
  const filterEmpty = filterSet != null && filtered.length === 0
  const layersShown = filterEmpty ? layers : filtered

  // Defer applying a layer drag-selection until the mouse is released, so the
  // Frequencies cloud reflows once (on mouseup) instead of on every dragged cell.
  const dragLayersRef = useRef(null)
  dragLayersRef.current = dragLayers
  const layersRef = useRef(layers)
  layersRef.current = layers
  useEffect(() => {
    const commit = () => {
      const sel = dragLayersRef.current
      if (sel == null) return
      dragLayersRef.current = null
      setFilterInput(sel.length >= layersRef.current.length ? '' : formatRanges(sel))
      setDragLayers(null)
    }
    window.addEventListener('mouseup', commit)
    return () => window.removeEventListener('mouseup', commit)
  }, [])

  const pinnedIds = useMemo(() => Object.keys(pinned).map(Number).sort((a, b) => a - b), [pinned])
  const pinColor = (tid) => PIN_COLORS[pinnedIds.indexOf(tid) % PIN_COLORS.length]

  // "Most-relevant" layer for a pinned token = where it reaches its best rank
  // (the lowest) across all positions. Used to prefill the editor on the right
  // slice rather than the default one.
  function peakLayerOf(tid) {
    const d = pinData?.pins?.[tid]
    if (!d?.ranks?.length) return null
    // "most-relevant" layer = the row that is LIGHTEST on average in the heatmap
    // (rankColor metric: 1 - log10(rank+1)/5), not the single best rank at one
    // position (a spike doesn't make the layer read as strong).
    let bestLi = 0, bestScore = -Infinity
    d.ranks.forEach((layerRanks, li) => {
      if (!layerRanks.length) return
      const score = layerRanks.reduce(
        (s, r) => s + (1 - Math.min(1, Math.log10(r + 1) / 5)), 0,
      ) / layerRanks.length
      if (score > bestScore) { bestScore = score; bestLi = li }
    })
    return pinData.layers[bestLi]
  }

  const agg = useMemo(() => {
    if (view !== 'agg') return []
    const visible = new Set(layersShown)
    const map = new Map()
    frames.forEach((f) => {
      const seenHere = new Set()
      Object.entries(f.layers).forEach(([layer, d]) => {
        if (!visible.has(+layer)) return
        const strs = maskOn ? d.m_strs : d.strs
        const ps = maskOn ? d.m_p : d.p
        const ids = maskOn ? d.m_ids : d.ids
        strs.forEach((s, i) => {
          const tid = ids[i]
          let entry = map.get(tid)
          if (!entry) {
            entry = { tid, str: s, appearances: 0, maxP: 0, peakLayer: null }
            map.set(tid, entry)
          }
          if (!seenHere.has(tid)) {
            seenHere.add(tid)
            entry.appearances++
          }
          if (ps[i] > entry.maxP) {
            entry.maxP = ps[i]
            entry.peakLayer = +layer
          }
        })
      })
    })
    return [...map.values()]
      .filter((e) => !hidden.has(trimTok(e.str)))
      .sort((a, b) => b.appearances - a.appearances || b.maxP - a.maxP)
      .slice(0, aggLimit)
  }, [tick, frames, maskOn, view, layersShown, hidden, aggLimit])

  const gridW = shown.length * CELL_W
  const height = HEADER_H + layersShown.length * CELL_H

  // Sequence token: two quick pin/unpin clicks race their POSTs — only the
  // LAST request may write pinData, or a stale response resurrects a token
  // that was just unpinned (ghost graph showing the raw token id).
  const pinReqRef = useRef(0)

  async function refreshPins(nextPinned) {
    const ids = Object.keys(nextPinned).map(Number)
    setPinned(nextPinned)
    const reqId = ++pinReqRef.current
    if (!ids.length || genId == null || generationRunId == null) {
      setPinData(null)
      return
    }
    try {
      const res = await fetch('/api/lens/pin', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gen_id: genId, generation_run_id: generationRunId, token_ids: ids }),
      })
      const body = await res.json()
      if (!res.ok) throw new Error(body.detail || res.statusText)
      if (reqId === pinReqRef.current) setPinData(body)
    } catch (err) {
      if (reqId === pinReqRef.current) {
        onNotice?.(String(err.message || err))
        setPinData(null)
      }
    }
  }

  // Keep the pin graphs in sync with the generation being VIEWED: refetch when
  // it changes (new message generated / another message selected) — debounced
  // on tick so a streaming generation triggers one fetch at the end, not one
  // per frame batch.
  const pinnedRef = useRef(pinned)
  pinnedRef.current = pinned
  useEffect(() => {
    // never fetch mid-stream: slow generations space their frame batches past
    // the debounce, and the server 409s /api/lens/pin while it generates —
    // when `streaming` flips back to false this effect refires and fetches once
    if (streaming) return
    if (genId == null || generationRunId == null || !Object.keys(pinnedRef.current).length) return
    const t = setTimeout(() => refreshPins(pinnedRef.current), 400)
    return () => clearTimeout(t)
  }, [genId, generationRunId, tick, streaming])

  function togglePin(tid, str) {
    const next = { ...pinned }
    if (next[tid]) delete next[tid]
    else next[tid] = str
    if (genId == null) {
      // in replay: no server curves, but the grid coloring stays available
      setPinned(next)
      setPinData(null)
      return
    }
    refreshPins(next)
  }

  // Memoized grid: doesn't depend on the tooltip (smooth hover even with 400 columns).
  const gridCells = useMemo(() => {
    if (view !== 'grid') return null
    return (
      <>
        {layersShown.map((layer, ri) => ri % 2 === 1 && (
          <rect key={`z${layer}`} x="0" y={HEADER_H + ri * CELL_H} width={gridW} height={CELL_H}
            fill="rgba(255,255,255,.025)" />
        ))}
        {shown.map((f, ci) => (
          <g key={ci} transform={`translate(${ci * CELL_W},0)`}>
            <rect
              x="0" y="0" width={CELL_W - 1} height={HEADER_H - 6}
              fill={f.phase === 'reading' ? 'rgba(232,161,60,.10)' : 'rgba(108,184,224,.10)'}
            />
            <text
              x={CELL_W / 2} y={HEADER_H - 12}
              className="lv-toklabel"
              transform={`rotate(-38 ${CELL_W / 2} ${HEADER_H - 12})`}
            >
              {fmtTok(f.tok).slice(0, 9) || '·'}
            </text>
          </g>
        ))}
        {layersShown.map((layer, ri) => (
          <g key={layer} transform={`translate(0,${HEADER_H + ri * CELL_H})`}>
            {shown.map((f, ci) => {
              const c = cellData(f, String(layer), maskOn)
              if (!c) return null
              const opacity = Math.max(0.28, Math.min(1, Math.sqrt(c.p) * 2.4))
              const hit = pinHit(f, String(layer), maskOn, pinnedIds)
              return (
                <g
                  key={ci}
                  transform={`translate(${ci * CELL_W},0)`}
                  className="lv-cell"
                  onMouseMove={(e) => {
                    setTip({ x: e.clientX, y: e.clientY, c, layer, ri, ci, f })
                    fetchTrans(c.top.map((t) => ({ tid: t.tid, str: t.s })))
                  }}
                  onClick={() => togglePin(c.tid, c.str)}
                >
                  {/* hover capture over the whole cell (without the 1px gap between
                      visible rects). We do NOT clear the tooltip per cell: only a
                      hover leaving the whole grid clears it (see lv-scroll). So
                      moving from one cell to the next fires a single event (no
                      intermediate null state = no flicker) and hovering a gap keeps
                      the last info instead of jumping. */}
                  <rect x="0" y="0" width={CELL_W} height={CELL_H} fill="transparent" />
                  <rect
                    x="0" y="0" width={CELL_W - 1} height={CELL_H - 1}
                    fill={hit
                      ? d3.color(pinColor(hit.tid)).copy({ opacity: 0.14 + 0.42 * (1 - hit.idx / Math.max(1, (lensMeta?.k ?? 8) - 1)) }).formatRgb()
                      : f.phase === 'reading' ? 'rgba(232,161,60,.06)' : 'rgba(108,184,224,.06)'}
                    stroke={pinned[c.tid] ? pinColor(c.tid) : 'transparent'}
                  />
                  <text x="3" y={CELL_H - 7} className="lv-word" opacity={opacity}>
                    {fmtTok(c.str).slice(0, 7)}
                    {maskOn && c.rank > 0 && <tspan className="lv-rank" dy="-4">{c.rank}</tspan>}
                  </text>
                </g>
              )
            })}
          </g>
        ))}
      </>
    )
  }, [tick, frames, maskOn, pinned, genId, view, layersShown, gridW])

  // "Activations" view: L2 norm of the residual per layer/position, normalized
  // PER LAYER (the norm grows strongly with depth: without this, the last layers
  // would crush everything) between the min and the 95th PERCENTILE of the
  // layer: the first token is an "attention sink" with a norm ~100× larger than
  // the rest — without this clamp it would be the only thing visible and all the
  // rest uniformly dark. Light color = high norm for the layer; outliers saturate
  // to yellow (exact value in the tooltip).
  const actHasData = useMemo(
    () => view === 'act' && shown.some((f) => Object.values(f.layers).some((d) => d?.h_norm != null)),
    [view, tick, frames],
  )
  const actCells = useMemo(() => {
    if (view !== 'act') return null
    const extent = {}
    layersShown.forEach((layer) => {
      const vals = []
      shown.forEach((f) => {
        const v = f.layers[String(layer)]?.h_norm
        if (v != null) vals.push(v)
      })
      vals.sort((a, b) => a - b)
      const lo = vals[0] ?? Infinity
      const hi = vals.length ? vals[Math.floor(0.95 * (vals.length - 1))] : -Infinity
      extent[layer] = [lo, hi]
    })
    return (
      <>
        {shown.map((f, ci) => (
          <g key={ci} transform={`translate(${ci * CELL_W},0)`}>
            <rect
              x="0" y="0" width={CELL_W - 1} height={HEADER_H - 6}
              fill={f.phase === 'reading' ? 'rgba(232,161,60,.10)' : 'rgba(108,184,224,.10)'}
            />
            <text
              x={CELL_W / 2} y={HEADER_H - 12}
              className="lv-toklabel"
              transform={`rotate(-38 ${CELL_W / 2} ${HEADER_H - 12})`}
            >
              {fmtTok(f.tok).slice(0, 9) || '·'}
            </text>
          </g>
        ))}
        {layersShown.map((layer, ri) => (
          <g key={layer} transform={`translate(0,${HEADER_H + ri * CELL_H})`}>
            {shown.map((f, ci) => {
              const v = f.layers[String(layer)]?.h_norm
              const [lo, hi] = extent[layer]
              const t = v == null || !isFinite(lo) ? null
                : hi > lo ? Math.min(1, (v - lo) / (hi - lo)) : 0.5
              return (
                <g
                  key={ci}
                  transform={`translate(${ci * CELL_W},0)`}
                  onMouseMove={(e) => setTip({ x: e.clientX, y: e.clientY, layer, ri, ci, f, act: { v, lo, hi } })}
                >
                  <rect x="0" y="0" width={CELL_W} height={CELL_H} fill="transparent" />
                  <rect
                    x="0" y="0" width={CELL_W - 1} height={CELL_H - 1}
                    fill={t == null ? 'rgba(255,255,255,.04)' : d3.interpolateInferno(0.08 + 0.84 * t)}
                  />
                </g>
              )
            })}
          </g>
        ))}
      </>
    )
  }, [tick, frames, view, layersShown, gridW])

  // No frames (e.g. while regenerating the very first reply): render an empty
  // shell instead of unmounting — unmounting would wipe the pinned tokens.
  if (!shown.length) return <div className="lensview" />

  return (
    <div className="lensview" style={maxH ? { maxHeight: maxH, height: maxH } : undefined}>
      <div className="lv-controls">
        {/* ['act', 'Activations'] disabled (TODO) — the view==='act' render below
            stays in place, just re-add the entry here to re-enable it */}
        {[['agg', 'Frequencies'], ['grid', 'Heatmap']].map(([v, label]) => (
          <button key={v} className={view === v ? 'lv-view-on' : ''}
            onClick={() => { setView(v); setTip(null); setAggTip(null) }}>{label}</button>
        ))}
        {lensMeta && (
          <span className="lv-lensbadge" title={lensMeta.path || `${lensMeta.repo_id || ''} ${lensMeta.filename || ''}`.trim()}>
            🔎 {lensLabel(lensMeta)}
          </span>
        )}
        <span className="lv-sep" />
        <label>
          <input type="checkbox" checked={maskOn} onChange={(e) => setMaskOn(e.target.checked)} />
          BPE/punctuation mask
        </label>
        {view === 'grid' && (
          <label title="follow the right edge (latest tokens) during generation">
            <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />
            auto-scroll →
          </label>
        )}
        <span className="lv-sep" />
        <label title="local display filter — doesn't affect the server capture">display</label>
        <input
          type="text"
          placeholder={`all (e.g. ${layers[0]}-${layers[layers.length - 1]})`}
          value={filterInput}
          onChange={(e) => setFilterInput(e.target.value)}
          style={{ width: 120 }}
        />
        {filterSet != null && (
          <span className={`lv-filter-badge ${filterEmpty ? 'err' : ''}`}>
            {filterEmpty ? 'no layer matches' : `${layersShown.length}/${layers.length} layers`}
            <span className="lv-filter-clear" onClick={() => setFilterInput('')}> ✕</span>
          </span>
        )}
        <label title="pin a token by its text — pick the variant with (˽) or without a space">pin</label>
        <span className="lv-pinadd">
          <input type="text" placeholder="token…" value={pinInput}
            onChange={(e) => { setPinInput(e.target.value); lookupPin(e.target.value) }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && pinCands.length) {
                addPin(pinCands.find((c) => c.str.startsWith(' ')) || pinCands[0])
              }
            }}
            style={{ width: 90 }} />
          {pinCands.length > 0 && (
            <span className="lv-pinadd-cands">
              {pinCands.slice(0, 6).map((c) => (
                <button key={c.id} title={`id ${c.id}`} onClick={() => addPin(c)}>{fmtTok(c.str)}</button>
              ))}
            </span>
          )}
        </span>
        <span className="lv-sep" />
        {pinnedIds.map((tid) => (
          <span key={tid} className="lv-pin" style={{ borderColor: pinColor(tid), color: pinColor(tid) }}
            title={transLabel(tid) ? `≈ ${transLabel(tid)}` : undefined}
            onMouseEnter={() => fetchTrans([{ tid, str: String(pinned[tid]) }])}>
            <span onClick={() => togglePin(tid, pinned[tid])}>{fmtTok(pinned[tid])} ✕</span>
            <span
              className="lv-ablate"
              title="open the editor with this token prefilled (most-relevant layer pre-selected)"
              onClick={() => onEditToken?.(+tid, String(pinned[tid]), peakLayerOf(+tid))}
            > ☢</span>
          </span>
        ))}
        <span className="fcount" style={{ marginLeft: 'auto' }}>
          {frames.length} tokens · {layersShown.length}{layersShown.length !== layers.length ? `/${layers.length}` : ''} layers
          {frames.length > MAX_COLS ? ` · last ${MAX_COLS} shown` : ''}
        </span>
      </div>

      {/* visual selection of the displayed layers (click/click-drag, number on hover) —
          synced with the "display" text field above via filterInput */}
      <div className="lv-layerrow">
        <span className="src">layers shown:</span>
        <LayerPicker all={layers} value={dragLayers ?? layersShown} compact
          onChange={setDragLayers} />
      </div>

      {view === 'agg' && (
        <div className="lv-agg-wrap">
          <div className="lv-agg-bar">
            <label>show{' '}
              {/* min/default multiples of the step: an unaligned value makes the
                  spinner's first click "snap" by 1 instead of stepping */}
              <input type="number" min="10" max="2000" step="10" value={aggLimit}
                onChange={(e) => setAggLimit(Math.max(10, Math.trunc(+e.target.value) || 10))}
                style={{ width: 62 }} /> tokens
            </label>
            <span className="lv-sep" />
            <span className="src">click = pin · right-click = hide</span>
          </div>
          <div className="lv-agg">
            {agg.map((e) => (
              <span
                key={e.tid}
                className={`lv-agg-word ${pinned[e.tid] ? 'lv-agg-pinned' : ''}`}
                style={{
                  opacity: Math.max(0.35, Math.min(1, Math.sqrt(e.maxP) * 2.4)),
                  fontSize: `${Math.min(19, 11 + Math.log2(e.appearances) * 1.6)}px`,
                  ...(pinned[e.tid] ? { color: pinColor(e.tid) } : {}),
                }}
                onMouseEnter={(ev) => {
                  setAggTip({ x: ev.clientX, y: ev.clientY, e })
                  fetchTrans([{ tid: e.tid, str: e.str }])
                }}
                onMouseLeave={() => setAggTip(null)}
                onClick={() => togglePin(e.tid, e.str)}
                onContextMenu={(ev) => { ev.preventDefault(); onHideToken(e.str); setAggTip(null) }}
              >
                {fmtTok(e.str)}<sup>{e.appearances}</sup>
              </span>
            ))}
            {agg.length === 0 && <span className="src">no token to show (all hidden or no frames)</span>}
          </div>
        </div>
      )}

      {aggTip && (() => {
        const e = aggTip.e
        const trans = NONLATIN_RE.test(e.str || '') ? transCache.current.get(e.tid) : null
        return (
          <Tip x={aggTip.x} y={aggTip.y}>
            <div className="lv-tip-head">"{fmtTok(e.str)}" — click = pin</div>
            <div className="lv-tip-row"><span>appearances</span><span>{e.appearances}</span></div>
            <div className="lv-tip-row"><span>peak</span><span>L{e.peakLayer}</span></div>
            <div className="lv-tip-row"><span>max p</span><span>{(e.maxP * 100).toFixed(2)} %</span></div>
            {trans?.length > 0 && (
              <>
                <div className="lv-tip-head" style={{ marginTop: 6 }}>≈ nearest tokens (W_U cosine)</div>
                {trans.map((n) => (
                  <div key={n.id} className="lv-tip-row">
                    <span className="lv-trans">{fmtTok(n.str)}</span>
                    <span>{n.sim.toFixed(2)}</span>
                  </div>
                ))}
              </>
            )}
          </Tip>
        )
      })()}

      {view === 'grid' && (
      <div className="lv-gridwrap" ref={gridScrollRef}>
        <svg width={LABEL_W} height={height} className="lv-svg lv-labels">
          {layersShown.map((layer, ri) => (
            <g key={layer} transform={`translate(0,${HEADER_H + ri * CELL_H})`}>
              {ri % 2 === 1 && <rect x="0" y="0" width={LABEL_W} height={CELL_H} fill="rgba(255,255,255,.025)" />}
              <text
                x={LABEL_W - 6} y={CELL_H / 2 + 4}
                className={`lv-laylabel ${tip?.layer === layer ? 'lv-laylabel-hot' : ''}`}
              >L{layer}</text>
            </g>
          ))}
        </svg>
        <div className="lv-scroll" onMouseLeave={() => setTip(null)}>
          <svg width={gridW} height={height} className="lv-svg">
            {gridCells}
            {tip && tip.ri != null && (
              <g pointerEvents="none">
                <rect x="0" y={HEADER_H + tip.ri * CELL_H} width={gridW} height={CELL_H - 1}
                  fill="rgba(108,184,224,.08)" stroke="rgba(108,184,224,.35)" />
                <rect x={tip.ci * CELL_W} y="0" width={CELL_W - 1} height={height}
                  fill="rgba(108,184,224,.06)" />
              </g>
            )}
          </svg>
        </div>
      </div>
      )}

      {view === 'act' && (
      <div className="lv-gridwrap">
        <svg width={LABEL_W} height={height} className="lv-svg lv-labels">
          {layersShown.map((layer, ri) => (
            <g key={layer} transform={`translate(0,${HEADER_H + ri * CELL_H})`}>
              {ri % 2 === 1 && <rect x="0" y="0" width={LABEL_W} height={CELL_H} fill="rgba(255,255,255,.025)" />}
              <text
                x={LABEL_W - 6} y={CELL_H / 2 + 4}
                className={`lv-laylabel ${tip?.layer === layer ? 'lv-laylabel-hot' : ''}`}
              >L{layer}</text>
            </g>
          ))}
        </svg>
        <div className="lv-scroll" onMouseLeave={() => setTip(null)}>
          {!actHasData && (
            <div className="status-line" style={{ padding: '6px 8px' }}>
              activation norms are absent from these frames — regenerate (new
              generations capture them)
            </div>
          )}
          <svg width={gridW} height={height} className="lv-svg">
            {actCells}
            {tip && tip.ri != null && (
              <g pointerEvents="none">
                <rect x="0" y={HEADER_H + tip.ri * CELL_H} width={gridW} height={CELL_H - 1}
                  fill="none" stroke="rgba(108,184,224,.45)" />
                <rect x={tip.ci * CELL_W} y="0" width={CELL_W - 1} height={height}
                  fill="rgba(108,184,224,.06)" />
              </g>
            )}
          </svg>
        </div>
      </div>
      )}

      {tip && tip.act && (
        <Tip x={tip.x} y={tip.y}>
          <div className="lv-tip-head">
            pos {tip.f.pos} · L{tip.layer} · {tip.f.phase === 'reading' ? 'reading' : 'thinking'} · tok «{fmtTok(tip.f.tok)}»
          </div>
          {tip.act.v == null ? (
            <div className="lv-tip-row"><span>‖h‖ unavailable (older generation)</span></div>
          ) : (
            <>
              <div className="lv-tip-row"><span>‖h‖ (residual norm)</span><span>{tip.act.v}</span></div>
              <div className="lv-tip-row"><span>layer range</span><span>{tip.act.lo} – {tip.act.hi}</span></div>
            </>
          )}
        </Tip>
      )}

      {tip && !tip.act && (() => {
        // non-latin tokens of the cell whose "nearest tokens" we show at the tooltip's end
        const nonLatin = tip.c.top.filter((t) => NONLATIN_RE.test(t.s || ''))
        return (
          <Tip x={tip.x} y={tip.y}>
            <div className="lv-tip-head">
              pos {tip.f.pos} · L{tip.layer} · {tip.f.phase === 'reading' ? 'reading' : 'thinking'} · tok «{fmtTok(tip.f.tok)}»
            </div>
            {tip.c.top.map((t, i) => (
              <div key={i} className="lv-tip-row" style={pinned[t.tid] ? { color: pinColor(t.tid) } : undefined}>
                <span>{fmtTok(t.s)}</span>
                <span>r{t.r} · {(t.p * 100).toFixed(2)}%</span>
              </div>
            ))}
            {nonLatin.map((t) => {
              const list = transCache.current.get(t.tid)
              if (!list?.length) return null
              return (
                <div key={`tr${t.tid}`} className="lv-tip-trans">
                  <div className="lv-tip-head">"{fmtTok(t.s)}" ≈ nearest tokens (W_U cosine)</div>
                  {list.slice(0, 6).map((n) => (
                    <div key={n.id} className="lv-tip-row">
                      <span className="lv-trans">{fmtTok(n.str)}</span>
                      <span>{n.sim.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )
            })}
          </Tip>
        )
      })()}

      {pinData && (
        <div className="lv-pins" ref={pinsRef}>
          {/* only tokens still pinned: a stale pinData must never resurrect an
              unpinned token's graph */}
          {Object.entries(pinData.pins).filter(([tid]) => pinned[tid]).map(([tid, data]) => {
            const N = pinData.positions.length
            const L = pinData.layers.length
            // width: fills the container (PIN_FILL_WIDTH), at least
            // PIN_PX_PER_TOKEN per position, bounded by PIN_MIN_W / PIN_MAX_W
            const fillW = PIN_FILL_WIDTH && pinsW ? pinsW - 46 - 10 : 0
            const w = Math.min(PIN_MAX_W, Math.max(PIN_MIN_W, N * PIN_PX_PER_TOKEN, fillW))
            const curveH = PIN_CURVE_H
            const rowH = Math.max(PIN_ROW_H_MIN, Math.min(PIN_ROW_H_MAX, Math.floor(PIN_MAP_H / L)))
            const mapH = L * rowH
            // column bands: position pi occupies [xL(pi), xL(pi)+cw] — the
            // curve passes through the centers. A [0, N-1] → [0, w] point
            // scale would push the LAST heatmap column past the svg edge
            // (clipped) and misalign the hover column math.
            const cw = w / N
            const xL = (pi) => 40 + pi * cw
            const xC = (pi) => 40 + (pi + 0.5) * cw
            const y = d3.scaleLinear([0, 5.2], [2, curveH - 2])
            const layerColor = (li) => d3.interpolateCool(0.15 + 0.7 * (li / Math.max(1, L - 1)))
            const hov = pinHover?.tid === tid ? pinHover.li : null
            const hovPi = pinHover?.tid === tid ? pinHover.pi : null
            // layer labels: all if few, otherwise sampled
            const labelEvery = L <= 16 ? 1 : Math.ceil(L / 12)
            return (
              <div key={tid} className="lv-pinblock">
                <div className="lv-pinname" style={{ color: pinColor(+tid) }}>
                  "{fmtTok(pinned[tid] ?? tid)}"
                  <button className="lv-copybtn" title="copy the token to the clipboard (without spaces)"
                    onClick={() => copyText(String(pinned[tid] ?? ''))}>⧉</button>
                  <span className="lv-pinsub"> — token rank per layer and position</span>
                  {hov != null && (
                    <span className="lv-pinhov">
                      {' '}· hover: L{pinData.layers[hov]}
                      {hovPi != null && pinData.tokens?.[hovPi] != null
                        ? ` — ${fmtTok(pinData.tokens[hovPi])}` : ''}
                    </span>
                  )}
                </div>
                {/* legend: one chip per layer, hover = highlights the curve and the row */}
                <div className="lv-laylegend">
                  {pinData.layers.map((layer, li) => (
                    <span
                      key={layer}
                      className={`lv-laychip ${hov != null && hov !== li ? 'dim' : ''}`}
                      style={{ borderColor: layerColor(li), color: layerColor(li) }}
                      onMouseEnter={() => setPinHover({ tid, li })}
                      onMouseLeave={() => setPinHover(null)}
                    >L{layer}</span>
                  ))}
                </div>
                <svg width={w + 46} height={curveH + 10}>
                  {[0, 1, 2, 3, 4, 5].map((d) => (
                    <g key={d}>
                      <line x1="40" x2={w + 40} y1={y(d)} y2={y(d)} className="lv-grid" />
                      <text x="36" y={y(d) + 3} className="lv-axis">{d === 0 ? '1' : `1e${d}`}</text>
                    </g>
                  ))}
                  {data.ranks.map((layerRanks, li) => (
                    <polyline
                      key={li}
                      fill="none"
                      stroke={layerColor(li)}
                      strokeWidth={hov === li ? 2.4 : 1.2}
                      opacity={hov == null ? 0.75 : hov === li ? 1 : 0.12}
                      points={layerRanks.map((r, pi) => `${xC(pi)},${y(Math.log10(r + 1))}`).join(' ')}
                      onMouseEnter={() => setPinHover({ tid, li })}
                      onMouseLeave={() => setPinHover(null)}
                      style={{ cursor: 'pointer' }}
                    />
                  ))}
                </svg>
                <svg width={w + 46} height={mapH + 16}
                  onMouseMove={(e) => {
                    // continuous hover: each Y pixel of the heatmap falls on a row
                    // (layer), with no gap between rows → no more jumping
                    const rect = e.currentTarget.getBoundingClientRect()
                    const ry = e.clientY - rect.top
                    const li = Math.floor(ry / rowH)
                    // column = token position (X axis), to show the hovered token
                    const pi = Math.max(0, Math.min(N - 1,
                      Math.floor((e.clientX - rect.left - 40) / cw)))
                    if (li >= 0 && li < L) setPinHover({ tid, li, pi })
                  }}
                  onMouseLeave={() => setPinHover(null)}
                >
                  {data.ranks.map((layerRanks, li) => (
                    <g key={li}>
                      {(li % labelEvery === 0 || hov === li) && (
                        <text x="36" y={li * rowH + rowH - 1}
                          className={`lv-axis ${hov === li ? 'lv-axis-hot' : ''}`}>L{pinData.layers[li]}</text>
                      )}
                      {layerRanks.map((r, pi) => (
                        <rect
                          key={pi}
                          x={xL(pi)} y={li * rowH}
                          width={Math.max(1.5, cw)} height={rowH - 1}
                          fill={rankColor(r)}
                          opacity={hov == null || hov === li ? 1 : 0.25}
                        />
                      ))}
                      {hov === li && (
                        <rect x="40" y={li * rowH} width={w} height={rowH - 1}
                          fill="none" stroke="var(--accent2)" strokeWidth="1" pointerEvents="none" />
                      )}
                    </g>
                  ))}
                  <text x="40" y={mapH + 12} className="lv-axis" style={{ textAnchor: 'start' }}>
                    color = rank (light = rank 1) · X axis = token position · hover a layer above
                  </text>
                </svg>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
