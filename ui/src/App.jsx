import { useEffect, useMemo, useRef, useState } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import LensView from './LensView.jsx'
import LensDiff from './Diff.jsx'
import Editor from './Editor.jsx'
import { fmtTok } from './tok'

const GB = 2 ** 30

marked.setOptions({ breaks: true, gfm: true })

// Markdown bubble content: parsed + sanitized (a local model can still emit
// arbitrary HTML — never inject it raw).
function Md({ text }) {
  const html = useMemo(() => DOMPurify.sanitize(marked.parse(text || '')), [text])
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />
}

// Polluter tokens excluded by default from the Frequencies token cloud (compared
// on the form without edge spaces). Managed at the bottom of the Chat tab.
const DEFAULT_HIDDEN = [
  '热门推荐', '阅读全文', '网友评论', '点击查看答案', '查看全文', '最新发布', '展开全文',
  '相关问答', '続きを読む', 'Читать', 'Weiterlesen', 'スポンサーリンク',
]
const trimTok = (s) => (s || '').replace(/^\s+|\s+$/g, '')

const LENS_PRESETS = {
  'Qwen/Qwen3.5-4B': {
    repo_id: 'neuronpedia/jacobian-lens',
    revision: 'qwen-n1000',
    filename: 'qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt',
  },
  'Qwen/Qwen3-4B': {
    repo_id: 'neuronpedia/jacobian-lens',
    revision: 'main',
    filename: 'qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt',
  },
}

async function jsonFetch(url, options) {
  const res = await fetch(url, options)
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || res.statusText)
  return body
}

const SAMPLING_DEFAULT = { temperature: 0.7, top_p: 0.95, top_k: 40, max_tokens: 512, seed: -1, repetition_penalty: 1 }

// Human-readable name of the loaded lens (local path or Hub file) for "which lens do I have?".
function lensName(meta) {
  if (!meta) return ''
  if (meta.path) return meta.path.replace(/[/\\]+$/, '').split(/[/\\]/).pop()
  if (meta.filename) {
    const stem = meta.filename.split('/').pop().replace('_jacobian_lens', '').replace('.pt', '')
    return meta.revision ? `${stem} @${meta.revision}` : stem
  }
  return meta.repo_id || 'lens'
}

// Note about the chat template fetched when loading a base model.
function templateNote(meta) {
  const src = meta.chat_template_source
  if (!src) return { suffix: '', warn: false } // the model has its own template
  if (src === 'generic') {
    return { suffix: ' — no chat template found on the Hub: generic User:/Assistant: template (limited results on a base model)', warn: true }
  }
  return { suffix: ` — chat template fetched from ${src}`, warn: false }
}

function TreeNode({ node, childs, depth, activeIds, onSelect }) {
  return (
    <>
      <div
        className={`tree-node ${activeIds.has(node.id) ? 'on-path' : ''}`}
        style={{ paddingLeft: 8 + depth * 14 }}
        onClick={() => onSelect(node.id)}
      >
        <span className={`tn-role ${node.role}`}>{node.role[0].toUpperCase()}</span>
        <span className="tn-text" title={node.content || '(empty)'}>#{node.id} {node.content.slice(0, 70) || '(empty)'}</span>
        {node.has_frames && <span className="tn-frames">◈</span>}
      </div>
      {(childs[node.id] || []).map((c) => (
        <TreeNode key={c.id} node={c} childs={childs} depth={depth + 1} activeIds={activeIds} onSelect={onSelect} />
      ))}
    </>
  )
}

export default function App() {
  const [models, setModels] = useState([])
  const [status, setStatus] = useState(null)
  const [selected, setSelected] = useState(null)
  const [dtype, setDtype] = useState('bf16')
  const [quant, setQuant] = useState('')
  const [device, setDevice] = useState('cuda:0')
  const [notice, setNotice] = useState(null)
  const [downloadRepo, setDownloadRepo] = useState('')
  const [browseOpen, setBrowseOpen] = useState(false)
  const [browse, setBrowse] = useState(null)

  const [lensForm, setLensForm] = useState({ repo_id: 'neuronpedia/jacobian-lens', revision: '', filename: '', path: '', k: 8 })
  const [reg, setReg] = useState(null)
  // model being loaded right now (id requested) + lens queued to chain-load
  const [loadingId, setLoadingId] = useState(null)
  const queuedLensRef = useRef(null)
  const [queuedLensName, setQueuedLensName] = useState(null)
  const [lensOn, setLensOn] = useState(true)
  const [framesCount, setFramesCount] = useState(0)
  const [selectedIdx, setSelectedIdx] = useState(null)
  const [diffSel, setDiffSel] = useState([])
  const [tab, setTab] = useState('chat')
  const [fitName, setFitName] = useState('')

  const [hidden, setHidden] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('jlens_hidden_tokens') || 'null')
      return new Set(Array.isArray(saved) ? saved : DEFAULT_HIDDEN)
    } catch { return new Set(DEFAULT_HIDDEN) }
  })
  const [hideInput, setHideInput] = useState('')

  // chat display: markdown rendering of assistant replies (default ON)
  const [chatMd, setChatMd] = useState(() => localStorage.getItem('jlens_chat_md') !== '0')
  useEffect(() => { localStorage.setItem('jlens_chat_md', chatMd ? '1' : '0') }, [chatMd])

  // "API monitor" mode: instead of polling status every 2s, refresh only when an
  // API/MCP generation completes (the server pushes on /ws). Off = normal 2s poll.
  const [apiMonitor, setApiMonitor] = useState(() => localStorage.getItem('jlens_api_monitor') === '1')
  useEffect(() => { localStorage.setItem('jlens_api_monitor', apiMonitor ? '1' : '0') }, [apiMonitor])

  // server-side settings (Options tab)
  const [settings, setSettings] = useState(null)
  useEffect(() => {
    jsonFetch('/api/settings').then((s) => {
      setSettings(s)
      setChatMd(!!s.chat_markdown)
      if (s.default_quant != null) setQuant(s.default_quant)
    }).catch(() => {})
  }, [])
  async function patchSettings(patch) {
    try {
      const s = await jsonFetch('/api/settings', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(patch),
      })
      setSettings(s)
      if (patch.chat_markdown != null) setChatMd(!!s.chat_markdown)
      if (patch.default_quant != null) setQuant(s.default_quant)
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  // manual height of the lens view (drag handle above it; null = auto 46vh)
  const [lensViewH, setLensViewH] = useState(() => {
    const v = +localStorage.getItem('jlens_lensview_h')
    return v >= 80 ? v : null
  })
  useEffect(() => {
    if (lensViewH) localStorage.setItem('jlens_lensview_h', String(Math.round(lensViewH)))
    else localStorage.removeItem('jlens_lensview_h')
  }, [lensViewH])
  function startLensResize(e) {
    e.preventDefault()
    const el = document.querySelector('.lensview')
    const startH = lensViewH ?? el?.getBoundingClientRect().height ?? 300
    const startY = e.clientY
    const move = (ev) => setLensViewH(
      Math.max(80, Math.min(window.innerHeight * 0.85, startH + (startY - ev.clientY))),
    )
    const up = () => {
      window.removeEventListener('mousemove', move)
      window.removeEventListener('mouseup', up)
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
  }
  // reply being edited in place: { idx, text } | null
  const [editingMsg, setEditingMsg] = useState(null)
  // assistant message being CONTINUED (ref: onFrame is a stale closure)
  const continuingIdRef = useRef(null)
  const [continuingId, setContinuingId] = useState(null)

  const [ivRules, setIvRules] = useState([])
  const [ivScale, setIvScale] = useState(1)
  const [ivMode, setIvMode] = useState('standard')
  const [editorOpen, setEditorOpen] = useState(false)
  const [editorPrefill, setEditorPrefill] = useState(null)
  const [capLayers, setCapLayers] = useState('')
  const [capK, setCapK] = useState('')

  const [convs, setConvs] = useState([])
  const [convQuery, setConvQuery] = useState('')
  const [conv, setConv] = useState(null)
  const [tree, setTree] = useState([])
  const [showTree, setShowTree] = useState(false)
  const [editParent, setEditParent] = useState(undefined)

  const [fitModel, setFitModel] = useState('')
  const [fitN, setFitN] = useState(100)
  // fit corpus library: tick one or several; several = mixed in equal parts.
  // persisted so the user's added datasets survive a refresh.
  const [fitDatasets, setFitDatasets] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('jlens_fit_datasets') || 'null')
      const restored = Array.isArray(saved)
        ? saved.filter((d) => d && d.id).map((d) => ({ id: String(d.id), on: !!d.on }))
        : []
      if (restored.length) return restored
    } catch { /* ignore corrupt storage */ }
    return [
      { id: 'Salesforce/wikitext-103-raw-v1', on: true },
      { id: 'heretic-org/Semantic-Harmless', on: false },
    ]
  })
  useEffect(() => {
    localStorage.setItem('jlens_fit_datasets', JSON.stringify(fitDatasets))
  }, [fitDatasets])
  const [fitDatasetInput, setFitDatasetInput] = useState('')
  const addFitDataset = () => {
    const id = fitDatasetInput.trim()
    if (!id) return
    setFitDatasets((prev) =>
      prev.some((d) => d.id === id)
        ? prev.map((d) => (d.id === id ? { ...d, on: true } : d))
        : [...prev, { id, on: true }])
    setFitDatasetInput('')
  }
  const [fitQuant, setFitQuant] = useState('')
  const [fitDevices, setFitDevices] = useState([])
  const [fitDimBatch, setFitDimBatch] = useState('')
  const [fitMaxSeq, setFitMaxSeq] = useState(128)
  const [fitLayers, setFitLayers] = useState('')
  const [fitContinue, setFitContinue] = useState('')
  const [localLenses, setLocalLenses] = useState([])
  const [fitAdvanced, setFitAdvanced] = useState(false)

  const [system, setSystem] = useState('')
  const [messages, setMessages] = useState([])
  const [draft, setDraft] = useState(null)
  const [input, setInput] = useState('')
  const [sampling, setSampling] = useState(() => {
    // restore sampling params after a refresh (temperature, etc.)
    try {
      const saved = JSON.parse(localStorage.getItem('jlens_sampling') || 'null')
      return saved && typeof saved === 'object' ? { ...SAMPLING_DEFAULT, ...saved } : SAMPLING_DEFAULT
    } catch { return SAMPLING_DEFAULT }
  })

  const wsRef = useRef(null)
  const draftRef = useRef('')
  const framesRef = useRef([])
  const messagesRef = useRef(null)
  const lastPaintRef = useRef(0)
  const convRef = useRef(null)
  convRef.current = conv

  const refreshModels = () =>
    jsonFetch('/api/models').then((b) => setModels(b.models)).catch(() => {})

  const refreshConvs = (query = convQuery) =>
    jsonFetch(`/api/conversations${query ? `?query=${encodeURIComponent(query)}` : ''}`)
      .then((b) => setConvs(b.conversations))
      .catch(() => {})

  useEffect(() => {
    refreshModels()
    refreshConvs()
    // restore the current conversation after a page reload
    const saved = Number(localStorage.getItem('jlens_conv'))
    if (saved) {
      jsonFetch(`/api/conversations/${saved}`).then((body) => {
        setTree(body.messages)
        setConv({ id: body.id, title: body.title, tags: body.tags })
        if (body.messages.length) {
          applyPath(Math.max(...body.messages.map((m) => m.id)), body.messages)
        }
      }).catch(() => localStorage.removeItem('jlens_conv'))
    }
  }, [])

  // status refresh — normally a 2s poll; in "API monitor" mode it is event-driven
  // instead, refetched only when an API/MCP generation ends (pushed over /ws).
  useEffect(() => {
    const tick = () => jsonFetch('/api/status').then(setStatus).catch(() => {})
    tick()
    if (!apiMonitor) {
      const id = setInterval(tick, 2000)
      return () => clearInterval(id)
    }
    let ws
    let timer
    let closed = false
    const open = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onmessage = (ev) => { try { if (JSON.parse(ev.data)?.type === 'api_generation') tick() } catch { /* ignore */ } }
      ws.onclose = () => { if (!closed) timer = setTimeout(open, 2000) }
    }
    open()
    return () => { closed = true; clearTimeout(timer); if (ws) ws.close() }
  }, [apiMonitor])

  useEffect(() => {
    if (conv?.id) localStorage.setItem('jlens_conv', String(conv.id))
  }, [conv?.id])

  useEffect(() => {
    localStorage.setItem('jlens_sampling', JSON.stringify(sampling))
  }, [sampling])

  useEffect(() => {
    localStorage.setItem('jlens_hidden_tokens', JSON.stringify([...hidden]))
  }, [hidden])

  function hideToken(s) {
    const clean = trimTok(s)
    if (!clean) return
    setHidden((prev) => new Set(prev).add(clean))
  }
  function unhideToken(s) {
    setHidden((prev) => { const n = new Set(prev); n.delete(s); return n })
  }

  // fit devices follow the GPUs actually present: drop the missing ones,
  // default to all of them when nothing (valid) is selected
  useEffect(() => {
    const avail = (status?.gpus || []).map((g) => `cuda:${g.index}`)
    if (!avail.length) return
    setFitDevices((prev) => {
      const filtered = prev.filter((d) => avail.includes(d))
      return filtered.length ? filtered : avail
    })
  }, [status?.gpus?.length])

  useEffect(() => { refreshConvs() }, [convQuery])

  // Auto-scroll to the bottom while a reply streams in — but only if the user
  // is already near the bottom, so scrolling up to read is never hijacked.
  useEffect(() => {
    const el = messagesRef.current
    if (!el || draft === null) return
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 160) {
      el.scrollTop = el.scrollHeight
    }
  }, [draft])

  // After a refresh / branch change, messages are restored without their frames
  // (applyPath sets frames: undefined) → the lens view won't show. We reload the
  // frames of the last message that has some (they exist server-side), to get the
  // heatmap/frequencies back without having to regenerate.
  const framesTriedRef = useRef(new Set())
  useEffect(() => {
    // draft !== null == streaming (const "streaming" declared below → avoid the TDZ)
    if (draft !== null || messages.some((m) => m.frames?.length)) return
    let lastIdx = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].has_frames && messages[i].id != null) { lastIdx = i; break }
    }
    if (lastIdx < 0) return
    const mid = messages[lastIdx].id
    if (framesTriedRef.current.has(mid)) return
    framesTriedRef.current.add(mid)
    jsonFetch(`/api/messages/${mid}/frames`)
      .then((body) => {
        if (body.frames?.length) {
          setMessages((prev) => prev.map((x) => (x.id === mid ? { ...x, frames: body.frames } : x)))
        }
      })
      .catch(() => {})
  }, [messages, draft])

  useEffect(() => {
    if (tab === 'fit') jsonFetch('/api/registry/local').then((b) => setLocalLenses(b.lenses)).catch(() => {})
  }, [tab, status?.fit?.state])

  useEffect(() => {
    if (!status) return
    setIvRules(status.interventions || [])
    if (status.interventions_scale != null) setIvScale(status.interventions_scale)
    if (status.interventions_mode != null) setIvMode(status.interventions_mode)
  }, [status])

  // auto-open the token editor when a lens just got loaded
  // (status?.lens, NOT lensMeta: that const is declared further down — TDZ)
  const hadLensRef = useRef(false)
  useEffect(() => {
    const has = !!status?.lens
    if (has && !hadLensRef.current) setEditorOpen(true)
    hadLensRef.current = has
  }, [status?.lens])

  const loadedId = status?.loaded?.model_id
  const lensMeta = status?.lens
  const busy = status?.busy
  const streaming = draft !== null

  useEffect(() => {
    // the registry is queryable for the model being LOADED too: pick a lens
    // while the weights stream in, it chain-loads once the model is ready
    const target = loadedId || loadingId
    const preset = LENS_PRESETS[target]
    if (preset) setLensForm((f) => ({ ...f, ...preset }))
    if (!target) { setReg(null); return }
    const rev = loadedId ? status?.loaded?.revision : null
    jsonFetch(`/api/registry/for-model?model_id=${encodeURIComponent(target)}${rev ? `&revision=${rev}` : ''}`)
      .then(setReg)
      .catch(() => setReg(null))
  }, [loadedId, loadingId])

  function ensureWs() {
    return new Promise((resolve, reject) => {
      const cur = wsRef.current
      if (cur && cur.readyState === WebSocket.OPEN) return resolve(cur)
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      ws.onopen = () => resolve(ws)
      ws.onerror = () => reject(new Error('websocket unavailable'))
      ws.onmessage = (ev) => onFrame(JSON.parse(ev.data))
      ws.onclose = () => { wsRef.current = null }
      wsRef.current = ws
    })
  }

  async function fetchTree(cid, { keepPath = true } = {}) {
    try {
      const body = await jsonFetch(`/api/conversations/${cid}`)
      setTree(body.messages)
      setConv({ id: body.id, title: body.title, tags: body.tags })
      return body
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
      return null
    }
  }

  function onFrame(frame) {
    if (frame.type === 'token') {
      draftRef.current += frame.text
      setDraft(draftRef.current)
    } else if (frame.type === 'frame') {
      framesRef.current.push(frame)
      const now = performance.now()
      if (now - lastPaintRef.current > 60) {
        lastPaintRef.current = now
        setFramesCount(framesRef.current.length)
      }
    } else if (frame.type === 'persisted') {
      if (!convRef.current) setConv({ id: frame.conversation_id, title: '', tags: [] })
      setMessages((prev) => {
        const copy = [...prev]
        for (let i = copy.length - 1; i >= 0; i--) {
          if (copy[i].role === 'user' && copy[i].id == null) {
            copy[i] = { ...copy[i], id: frame.user_message_id }
            break
          }
        }
        return copy
      })
    } else if (frame.type === 'done') {
      const frames = framesRef.current
      framesRef.current = []
      setFramesCount(0)
      setSelectedIdx(null)
      if (frame.continued && frame.message_id != null) {
        // continuation: update the extended reply in place (frame.text is the
        // FULL new content) and reload its merged frames blob
        continuingIdRef.current = null
        setContinuingId(null)
        setMessages((prev) => prev.map((m) => (m.id === frame.message_id
          ? {
              ...m, content: frame.text, stats: frame.stats,
              gen_id: frame.gen_id, generation_run_id: frame.generation_run_id,
              has_frames: m.has_frames || frames.length > 0,
              frames: undefined,
            }
          : m)))
        framesTriedRef.current.delete(frame.message_id)
        jsonFetch(`/api/messages/${frame.message_id}/frames`)
          .then((body) => {
            if (body.frames?.length) {
              setMessages((prev) => prev.map((x) => (x.id === frame.message_id
                ? { ...x, frames: body.frames } : x)))
            }
          })
          .catch(() => {})
      } else {
        setMessages((prev) => [...prev, {
          id: frame.message_id ?? null,
          role: 'assistant',
          content: frame.text,
          meta: frame.meta,
          stats: frame.stats,
          gen_id: frame.gen_id,
          generation_run_id: frame.generation_run_id,
          has_frames: frames.length > 0,
          frames,
        }])
      }
      draftRef.current = ''
      setDraft(null)
      if (!frame.text && !frame.continued) {
        setNotice({
          kind: 'err',
          text: 'the model emitted end-of-turn immediately (0 tokens) — strong '
            + 'global rules can do this on short prompts: soften the factor/scale '
            + 'or rephrase',
        })
      }
      if (frame.conversation_id) {
        fetchTree(frame.conversation_id)
        refreshConvs()
      }
    } else if (frame.type === 'error') {
      setNotice({ kind: 'err', text: frame.message })
      framesRef.current = []
      draftRef.current = ''
      setDraft(null)
      continuingIdRef.current = null
      setContinuingId(null)
    }
  }

  function lastPathId() {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].id != null) return messages[i].id
    }
    const root = tree.find((m) => m.role === 'system')
    return root ? root.id : null
  }

  async function sendChat({ content, parentId }) {
    try {
      const ws = await ensureWs()
      draftRef.current = ''
      framesRef.current = []
      setFramesCount(0)
      setDraft('')
      ws.send(JSON.stringify({
        type: 'chat',
        conversation_id: conv?.id ?? null,
        parent_id: parentId,
        content,
        system: conv ? undefined : (system.trim() || undefined),
        sampling,
        lens: lensOn && !!lensMeta,
      }))
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
      setDraft(null)
    }
  }

  function onSend() {
    const text = input.trim()
    if (!text || streaming) return
    const parentId = editParent !== undefined ? editParent : lastPathId()
    setMessages((prev) => [...prev, { id: null, role: 'user', content: text }])
    setInput('')
    setEditParent(undefined)
    sendChat({ content: text, parentId })
  }

  function onStop() {
    wsRef.current?.send(JSON.stringify({ type: 'stop' }))
  }

  function onRegenerate() {
    if (streaming) return
    const history = [...messages]
    while (history.length && history[history.length - 1].role === 'assistant') history.pop()
    const lastUser = [...history].reverse().find((m) => m.role === 'user')
    if (!lastUser || lastUser.id == null) return
    setMessages(history)
    sendChat({ content: null, parentId: lastUser.id })
  }

  // Extend the last assistant reply: the server generates with the turn left
  // open and appends to the stored message (frames merged too).
  async function onContinue() {
    if (streaming) return
    const last = messages[messages.length - 1]
    if (!last || last.role !== 'assistant' || last.id == null) return
    continuingIdRef.current = last.id
    setContinuingId(last.id)
    try {
      const ws = await ensureWs()
      draftRef.current = ''
      framesRef.current = []
      setFramesCount(0)
      setDraft('')
      ws.send(JSON.stringify({
        type: 'chat',
        continue_message_id: last.id,
        sampling,
        lens: lensOn && !!lensMeta,
      }))
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
      setDraft(null)
      continuingIdRef.current = null
      setContinuingId(null)
    }
  }

  async function saveEditedMsg() {
    if (!editingMsg) return
    const m = messages[editingMsg.idx]
    if (!m || m.id == null) { setEditingMsg(null); return }
    try {
      await jsonFetch(`/api/messages/${m.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: editingMsg.text }),
      })
      setMessages((prev) => prev.map((x, i) => (i === editingMsg.idx ? { ...x, content: editingMsg.text } : x)))
      setEditingMsg(null)
      setNotice({ kind: 'ok', text: 'reply updated — later turns will use the edited text' })
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  function onEditLast() {
    if (streaming) return
    const history = [...messages]
    while (history.length && history[history.length - 1].role === 'assistant') history.pop()
    const last = history.pop()
    if (!last) return
    setMessages(history)
    setInput(last.content)
    const parent = tree.find((m) => m.id === last.id)?.parent_id
    setEditParent(parent ?? null)
  }

  function pathFor(nodeId, msgs) {
    const byId = Object.fromEntries(msgs.map((m) => [m.id, m]))
    const path = []
    let cur = byId[nodeId]
    while (cur) {
      path.unshift(cur)
      cur = cur.parent_id != null ? byId[cur.parent_id] : null
    }
    return path
  }

  function applyPath(nodeId, msgs) {
    const path = pathFor(nodeId, msgs)
    const sys = path.find((m) => m.role === 'system')
    setSystem(sys ? sys.content : '')
    setMessages(path.filter((m) => m.role !== 'system').map((m) => ({ ...m, frames: undefined })))
    setSelectedIdx(null)
    setEditParent(undefined)
  }

  async function openConv(cid) {
    const body = await fetchTree(cid)
    if (!body) return
    const maxId = Math.max(...body.messages.map((m) => m.id))
    applyPath(maxId, body.messages)
  }

  function newConv() {
    localStorage.removeItem('jlens_conv')
    setConv(null)
    setTree([])
    setMessages([])
    setSystem('')
    setSelectedIdx(null)
    setEditParent(undefined)
  }

  async function deleteConv(cid, ev) {
    ev.stopPropagation()
    if (!window.confirm(`Delete conversation ${cid}?`)) return
    await jsonFetch(`/api/conversations/${cid}`, { method: 'DELETE' }).catch(() => {})
    if (conv?.id === cid) newConv()
    refreshConvs()
  }

  async function patchConv(fields) {
    if (!conv?.id) return
    await jsonFetch(`/api/conversations/${conv.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(fields),
    }).catch(() => {})
    refreshConvs()
  }

  async function selectMessage(i) {
    const m = messages[i]
    if (m.has_frames && !m.frames?.length && m.id != null) {
      try {
        const body = await jsonFetch(`/api/messages/${m.id}/frames`)
        setMessages((prev) => prev.map((x, j) => (j === i ? { ...x, frames: body.frames } : x)))
      } catch (err) {
        setNotice({ kind: 'err', text: String(err.message || err) })
        return
      }
    }
    setSelectedIdx(i)
  }

  // Consume the lens queued during the model load (chain-load), if any.
  function flushQueuedLens() {
    const q = queuedLensRef.current
    queuedLensRef.current = null
    setQueuedLensName(null)
    if (q) lensLoadBy(q)
  }

  async function onLoad() {
    if (!selected) return
    setNotice({ kind: 'ok', text: `loading ${selected}...` })
    setLoadingId(selected)
    try {
      const meta = await jsonFetch('/api/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: selected, dtype, quant: quant || null, device }),
      })
      const note = templateNote(meta)
      setNotice({
        kind: note.warn ? 'err' : 'ok',
        text: `loaded in ${meta.load_seconds}s (${meta.n_layers} layers, d=${meta.d_model})` + note.suffix,
      })
      flushQueuedLens()
    } catch (err) {
      queuedLensRef.current = null
      setQueuedLensName(null)
      setNotice({ kind: 'err', text: String(err.message || err) })
    } finally {
      setLoadingId(null)
    }
  }

  async function onUnload() {
    try {
      const r = await jsonFetch('/api/unload', { method: 'POST' })
      let text = 'nothing to unload'
      if (r.unloaded) {
        // VRAM actually returned = allocated before − reserved after (the rest = CUDA context)
        const before = Object.values(r.vram_allocated_before || {}).reduce((a, b) => a + b, 0)
        const reserved = Object.values(r.vram_reserved_after || {}).reduce((a, b) => a + b, 0)
        text = `model unloaded · ${(before / GB).toFixed(1)} GB returned`
          + (reserved > 64 * 2 ** 20 ? ` (${(reserved / GB).toFixed(1)} GB still reserved)` : ' (VRAM returned to the driver)')
      }
      setNotice({ kind: 'ok', text })
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  async function lensLoadBy(payload) {
    setNotice({ kind: 'ok', text: 'loading the lens...' })
    // form top-k applied to ALL load paths (registry included)
    if (payload.k == null && +lensForm.k > 0) payload = { ...payload, k: +lensForm.k }
    try {
      const meta = await jsonFetch('/api/lens/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: +lensForm.k || 8, ...payload }),
      })
      const warn = (meta.warnings || []).join(' ; ')
      setNotice({ kind: warn ? 'err' : 'ok', text: warn || `lens loaded (layers ${meta.tapped_layers.join(', ')})` })
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  function onLensLoad() {
    if (lensForm.path.trim()) {
      lensClick({ path: lensForm.path.trim() })
      return
    }
    lensClick({
      repo_id: lensForm.repo_id || null,
      revision: lensForm.revision || null,
      filename: lensForm.filename,
    })
  }

  // While a model is loading, clicking a lens QUEUES it (chain-load).
  const modelLoading = !!loadingId && busy === 'loading'
  function lensClick(payload) {
    if (modelLoading) {
      queuedLensRef.current = payload
      setQueuedLensName(payload.path || payload.filename?.split('/').pop() || 'lens')
      setNotice({ kind: 'ok', text: 'lens queued — it loads as soon as the model is ready' })
      return
    }
    lensLoadBy(payload)
  }

  function renderHubLens(h) {
    const nMatch = h.filename.match(/_n(\d+)/)
    const stem = h.filename.split('/').pop().replace('_jacobian_lens', '').replace('.pt', '')
    const isBase = h.via === 'base-model' || h.via === 'base-guess'
    return (
      <div key={`${h.repo_id}/${h.filename}`} className={`reg-item2 ${h.compatible === false ? 'reg-warn' : ''}`}>
        <div className="reg-main">
          <div className="reg-name" title={`${h.filename || stem}`}>
            {stem} <span className="reg-tag hub">Hub</span>
            {isBase && <span className="reg-tag base" title={h.reason}>base model</span>}
          </div>
          <div className="src">
            {nMatch ? `${nMatch[1]} prompts` : 'n unspecified (repo default fit)'}
            {h.base_model ? ` · base ${h.base_model}` : ''}
            {h.cached ? ' · ✓ cached' : ' · to download'}
          </div>
          <div className="src">{h.repo_id} @{h.revision}</div>
          {h.reason && <div className="src reg-reason">⚠ {h.reason}</div>}
        </div>
        <button disabled={h.compatible === false || (!!busy && !modelLoading)}
          title={modelLoading ? 'queue: loads right after the model'
            : h.cached ? 'load (already cached)' : 'download and load'}
          onClick={() => lensClick({ repo_id: h.repo_id, filename: h.filename, revision: h.revision })}>
          {modelLoading ? '⏲' : h.cached ? '▶' : '⬇'}</button>
      </div>
    )
  }

  async function onLensUnload() {
    await jsonFetch('/api/lens/unload', { method: 'POST' }).catch(() => {})
  }

  function openEditorWith(prefill) {
    setEditorOpen(true)
    if (prefill) setEditorPrefill(prefill)
  }

  // "Visualized" generation: the live stream, else the selected message if it has
  // frames, else the last message with frames. Shared between the J-lens and the
  // editor (auto-selecting the peak layer of an added token).
  function currentGenView() {
    const live = streaming && framesRef.current.length ? framesRef.current : null
    let idx = selectedIdx
    if (idx == null || !messages[idx]?.frames?.length) {
      idx = -1
      for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].frames?.length) { idx = i; break }
      }
    }
    // gen_id is only known for messages generated THIS page session; after a
    // reload, fall back to the id carried by the persisted frames themselves —
    // the server-side residual store survives a page refresh.
    const msgGen = idx >= 0
      ? messages[idx].gen_id ?? messages[idx].frames[messages[idx].frames.length - 1]?.gen ?? null
      : null
    const genId = live ? live[live.length - 1]?.gen ?? null : msgGen
    const msgRunId = idx >= 0
      ? messages[idx].generation_run_id
        ?? messages[idx].frames[messages[idx].frames.length - 1]?.generation_run_id ?? null
      : null
    const generationRunId = live
      ? live[live.length - 1]?.generation_run_id ?? null
      : msgRunId
    return { live, idx, genId, generationRunId }
  }

  async function applyCapture() {
    const list = []
    for (const part of capLayers.split(',')) {
      const m = part.trim().match(/^(\d+)\s*-\s*(\d+)$/)
      if (m) for (let i = +m[1]; i <= +m[2]; i++) list.push(i)
      else if (part.trim()) list.push(+part.trim())
    }
    const layers = list.filter((x) => !isNaN(x))
    try {
      const body = await jsonFetch('/api/lens/layers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          // empty layers field = keep the current layers (changing k only)
          layers: layers.length ? layers : lensMeta.tapped_layers,
          k: capK ? +capK : null,
        }),
      })
      setCapLayers('')
      setCapK('')
      setNotice({ kind: 'ok', text: `capture: layers ${body.tapped_layers.join(', ')} · k=${body.k}` })
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  async function openBrowse(path) {
    try {
      const body = await jsonFetch(`/api/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`)
      setBrowse(body)
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  // NATIVE folder picker (server-side tkinter dialog — the server is local, and
  // tkinter works on Windows/Linux/macOS alike). Returns the chosen path.
  async function pickFolder() {
    try {
      const body = await jsonFetch('/api/pick-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind: 'dir' }),
      })
      return body.path || null
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
      return null
    }
  }

  async function pickNative() {
    const path = await pickFolder()
    if (path) await registerPath(path)
  }

  // Browse result: the folder is ADDED to the model list (nothing is loaded
  // yet) — load it from the list with the dtype/quant/device of your choice.
  async function registerPath(path) {
    try {
      const r = await jsonFetch('/api/models/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })
      setSelected(r.registered)
      setNotice({ kind: 'ok', text: `added to the list: ${r.registered} — pick dtype/quant/device and Load` })
      refreshModels()
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  async function unregisterPath(m, ev) {
    ev.stopPropagation()
    try {
      await jsonFetch('/api/models/unregister', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: m.path }),
      })
      setNotice({ kind: 'ok', text: `removed from the list (files untouched): ${m.path}` })
      if (selected === m.id) setSelected(null)
      refreshModels()
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  async function deleteModel(m, ev) {
    ev.stopPropagation()
    const size = m.size_bytes ? ` (${(m.size_bytes / GB).toFixed(1)} GB)` : ''
    if (!window.confirm(`Permanently delete ${m.id}${size}?\nThis erases the files from disk.`)) return
    try {
      const r = await jsonFetch('/api/models/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: m.id }),
      })
      const freed = r.freed_bytes ? ` · ${(r.freed_bytes / GB).toFixed(1)} GB freed` : ''
      setNotice({ kind: 'ok', text: `deleted: ${m.id}${freed}` })
      if (selected === m.id) setSelected(null)
      refreshModels()
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  async function convertBf16(path) {
    try {
      await jsonFetch('/api/convert-bf16', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })
      setNotice({ kind: 'ok', text: 'bf16 conversion started (see status below)' })
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  async function onDownload() {
    if (!downloadRepo.trim()) return
    try {
      await jsonFetch('/api/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: downloadRepo.trim() }),
      })
      setNotice({ kind: 'ok', text: `download of ${downloadRepo} started` })
    } catch (err) {
      setNotice({ kind: 'err', text: String(err.message || err) })
    }
  }

  const doneDownloads = (status?.downloads || []).filter((d) => d.state === 'done').length
  useEffect(() => {
    if (doneDownloads > 0) refreshModels()
  }, [doneDownloads])

  useEffect(() => {
    if (status?.convert?.state === 'done') refreshModels()
  }, [status?.convert?.state])

  const childs = {}
  tree.forEach((m) => { (childs[m.parent_id] ??= []).push(m) })
  const activeIds = new Set(messages.map((m) => m.id).filter((x) => x != null))
  tree.filter((m) => m.role === 'system' && activeIds.size).forEach((m) => activeIds.add(m.id))

  return (
    <>
      <div className="sidebar">
        <h1>J-Wash</h1>

        <h2>GPU</h2>
        <div>
          {(status?.gpus || []).map((g) => (
            <div className="gpu" key={g.index}>
              <div className="name">
                <span>cuda:{g.index} · {g.name.replace('NVIDIA GeForce ', '')}</span>
                <span>{(g.vram_used / GB).toFixed(1)} / {(g.vram_total / GB).toFixed(0)} GB</span>
              </div>
              <div className="bar"><div className="fill" style={{ width: `${(100 * g.vram_used) / g.vram_total}%` }} /></div>
            </div>
          ))}
        </div>

        <div className="tabs">
          {[['chat', 'Chat'], ['model', 'Model'], ['lens', 'Lens'], ['fit', 'Fit'], ['options', 'Options']].map(([id, label]) => (
            <button key={id} className={tab === id ? 'tab-on' : ''} onClick={() => setTab(id)}>{label}</button>
          ))}
        </div>

        {tab === 'chat' && (<>
        <h2>Conversations <button style={{ float: 'right', padding: '1px 7px' }} onClick={newConv}>+ new</button></h2>
        <input type="text" placeholder="full-text search..." value={convQuery} onChange={(e) => setConvQuery(e.target.value)} />
        <div className="conv-list">
          {convs.map((c) => (
            <div key={c.id} className={`conv-item ${conv?.id === c.id ? 'selected' : ''}`} onClick={() => openConv(c.id)}>
              <div className="conv-title" title={c.title || `conversation ${c.id}`}>
                {c.title || `conversation ${c.id}`}
                <button className="conv-del" onClick={(e) => deleteConv(c.id, e)}>✕</button>
              </div>
              <div className="src">
                {c.n_messages} msgs{c.tags?.length ? ` · ${c.tags.join(', ')}` : ''}
                {c.snippet ? <div className="conv-snippet">{c.snippet}</div> : null}
              </div>
            </div>
          ))}
        </div>

        <h2>Ignored tokens{hidden.size ? ` (${hidden.size})` : ''}</h2>
        <input type="text" placeholder="hide a token…" value={hideInput}
          onChange={(e) => setHideInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { hideToken(hideInput); setHideInput('') } }} />
        {hidden.size > 0 ? (
          <div className="lv-hidden-row">
            {[...hidden].map((h) => (
              <span key={h} className="lv-hidden-chip" title="click to show again"
                onClick={() => unhideToken(h)}>{fmtTok(h)} ✕</span>
            ))}
          </div>
        ) : (
          <div className="src">tokens excluded from the Frequencies view (right-click a token there to add one)</div>
        )}

        </>)}

        {tab === 'model' && (<>
        <h2>Models <button style={{ float: 'right', padding: '1px 7px' }} onClick={refreshModels}>↻</button></h2>
        <div className="model-list">
          {models.map((m) => {
            const isFp32 = m.dtype === 'float32' || m.dtype === 'fp32'
            const isRegistered = m.source === 'registered'
            const label = isRegistered ? (m.path.split(/[\\/]/).pop() || m.id) : m.id
            return (
              <div
                key={m.id}
                className={`model-item ${selected === m.id ? 'selected' : ''} ${loadedId === m.id ? 'loaded' : ''}`}
                onClick={() => setSelected(m.id)}
              >
                <div className="model-item-head">
                  <span title={m.path || m.id}>{label} {loadedId === m.id ? '●' : ''}{m.missing ? ' ⚠ folder missing' : ''}</span>
                  {isRegistered ? (
                    <button className="model-del model-unreg"
                      title="remove this entry from the list — the model files are NOT touched"
                      disabled={loadedId === m.id || !!busy}
                      onClick={(e) => unregisterPath(m, e)}>
                      <svg viewBox="0 0 14 14" width="13" height="13" aria-hidden="true">
                        <path d="M2 3.5h6M2 7h6M2 10.5h4M9.5 8.5l3 3m0-3l-3 3"
                          stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" fill="none" />
                      </svg>
                    </button>
                  ) : (
                    <button className="model-del model-trash"
                      title={loadedId === m.id ? 'unload it first' : 'PERMANENTLY delete the model files from disk'}
                      disabled={loadedId === m.id || !!busy}
                      onClick={(e) => deleteModel(m, e)}>
                      <svg viewBox="0 0 14 14" width="13" height="13" aria-hidden="true">
                        <path d="M2 3.5h10M5.5 3.5V2.2c0-.4.3-.7.7-.7h1.6c.4 0 .7.3.7.7v1.3M3.2 3.5l.6 8.1c0 .5.4.9.9.9h4.6c.5 0 .9-.4.9-.9l.6-8.1M5.6 6v4M8.4 6v4"
                          stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" fill="none" />
                      </svg>
                    </button>
                  )}
                </div>
                <div className="src">
                  {m.source} · {(m.size_bytes / GB).toFixed(1)} GB
                  {m.n_layers != null ? ` · ${m.n_layers} layers` : ''}
                  {m.dtype ? ` · ${m.dtype}` : ''}
                </div>
                {isFp32 && (
                  <div className="src">
                    fp32 — <button className="linkbtn"
                      onClick={(e) => { e.stopPropagation(); convertBf16(m.path) }}
                      disabled={status?.convert?.state === 'running'}>convert to bf16 (÷2 space)</button>
                  </div>
                )}
              </div>
            )
          })}
        </div>

        <div className="row"><label>dtype</label>
          <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option value="bf16">bf16</option>
            <option value="fp16">fp16</option>
          </select>
        </div>
        <div className="row"><label>quant</label>
          <select value={quant} onChange={(e) => setQuant(e.target.value)}>
            <option value="">none</option>
            <option value="int8">int8</option>
            <option value="nf4">nf4</option>
          </select>
        </div>
        <div className="row"><label>device</label>
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            {(status?.gpus || []).map((g) => (
              <option key={g.index} value={`cuda:${g.index}`}>
                cuda:{g.index} · {g.name.replace('NVIDIA GeForce ', '')}
              </option>
            ))}
            {(status?.gpus?.length || 0) > 1 && <option value="auto">auto (all GPUs)</option>}
          </select>
        </div>
        <div className="row">
          <button className="primary" style={{ flex: 1 }} disabled={!selected || !!busy} onClick={onLoad}>Load</button>
          <button className="danger" disabled={!loadedId || !!busy} onClick={onUnload}>Unload</button>
        </div>
        </>)}

        {tab === 'lens' && (<>
        <h2>J-Lens</h2>
        {lensMeta ? (
          <div className="lens-status">
            <div className="status-line ok" title={lensMeta.path || `${lensMeta.repo_id || ''} ${lensMeta.filename || ''}`.trim()}>
              ▶ <b>{lensName(lensMeta)}</b>
            </div>
            <div className="status-line">
              layers {lensMeta.tapped_layers[0]}–{lensMeta.tapped_layers[lensMeta.tapped_layers.length - 1]} ({lensMeta.tapped_layers.length}) · k={lensMeta.k} · n={lensMeta.n_prompts}
            </div>
            {(lensMeta.warnings || []).map((w, i) => <div key={i} className="status-line err">⚠ {w}</div>)}
            <div className="row" style={{ marginTop: 6 }}>
              <label style={{ width: 'auto' }}>
                <input type="checkbox" checked={lensOn} onChange={(e) => setLensOn(e.target.checked)} /> active in chat
              </label>
              <button className="danger" onClick={onLensUnload}>Remove</button>
            </div>
            <div className="row" style={{ marginTop: 6 }}>
              <label title="layers read by the lens on the next generations (fitted: see the lens)">capture</label>
              <input type="text" value={capLayers}
                placeholder={`${lensMeta.tapped_layers[0]}-${lensMeta.tapped_layers[lensMeta.tapped_layers.length - 1]}`}
                onChange={(e) => setCapLayers(e.target.value)} />
              <label title="number of tokens read per cell (top-k) — applies to the next generations" style={{ width: 'auto' }}>k</label>
              <input type="number" min="1" max="32" value={capK} placeholder={lensMeta.k}
                onChange={(e) => setCapK(e.target.value)} style={{ width: 52, flexShrink: 0 }} />
              <button disabled={(!capLayers.trim() && !capK) || !!busy} onClick={applyCapture}>OK</button>
            </div>
            <div className="status-line">captured: {lensMeta.tapped_layers.join(', ')} · k={lensMeta.k} — applies to the next generations</div>
          </div>
        ) : (
          <>
            {modelLoading && (
              <div className="status-line ok">
                ⏳ {loadingId} is loading — pick a lens now, it will load automatically
                once the model is ready{queuedLensName ? ` · queued: ${queuedLensName}` : ''}.
              </div>
            )}
            {reg && (reg.local.length + reg.hub.length > 0) && (
              <div className="reg-list">
                {reg.local.map((l) => {
                  const m = l.meta || {}
                  const sl = m.source_layers
                  return (
                    <div key={l.path} className={`reg-item2 ${l.compatible === false ? 'reg-warn' : ''}`}>
                      <div className="reg-main">
                        <div className="reg-name" title={l.name}>{l.name} <span className="reg-tag">local</span>{l.compatible === false ? ' ⚠' : ''}</div>
                        <div className="src">
                          {m.n_prompts != null ? `${m.n_prompts} prompts` : 'n unknown'}
                          {sl ? ` · layers ${sl[0]}–${sl[1]}` : ''}
                          {m.dtype ? ` · ${m.quant || m.dtype}` : ''}
                          {m.d_model ? ` · d=${m.d_model}` : ''}
                        </div>
                        {m.created_at && <div className="src">fit {m.created_at.slice(0, 10)}{m.fit_seconds ? ` · ${Math.round(m.fit_seconds / 60)} min` : ''}{m.corpus ? ` · ${m.corpus.split(' ')[0]}` : ''}</div>}
                        {l.reason && <div className="src reg-reason">⚠ {l.reason}</div>}
                      </div>
                      <button disabled={l.compatible === false || (!!busy && !modelLoading)}
                        onClick={() => lensClick({ path: l.path })}>{modelLoading ? '⏲' : 'Load'}</button>
                    </div>
                  )
                })}
                {reg.hub.map(renderHubLens)}
                {reg.hub_error && <div className="status-line err">{reg.hub_error}</div>}
              </div>
            )}
            {reg && reg.local.length + reg.hub.length === 0 && (loadedId || loadingId) && (
              <div className="status-line">no known lens for this model — fit one (Fit tab), or
                load a lens fitted for a compatible model below</div>
            )}
            {reg && (reg.other?.length || 0) > 0 && (
              <details>
                <summary style={{ fontSize: 12, color: 'var(--muted)', cursor: 'pointer' }}>
                  lenses fitted for other models ({reg.other.length})
                </summary>
                <div className="src" style={{ margin: '6px 0' }}>
                  For your own finetune or merge without a matching lens: a lens fitted
                  on a compatible model of the SAME architecture can work (d_model and
                  layer count are checked at load). Readouts drift with the distance
                  between the weights — treat them as approximate.
                </div>
                <div className="reg-list">{reg.other.map(renderHubLens)}</div>
              </details>
            )}
            <details>
              <summary style={{ fontSize: 12, color: 'var(--muted)', cursor: 'pointer' }}>manual load</summary>
              <div className="src" style={{ margin: '6px 0' }}>
                Load any Jacobian-lens <code>.pt</code>: either a Hugging Face repo + the
                file path inside it (mirror the entries above), or the local path of a
                lens you fitted (Fit tab writes <code>lenses/&lt;name&gt;/lens.pt</code>).
              </div>
              <div className="row" style={{ marginTop: 6 }}><label>repo</label>
                <input type="text" value={lensForm.repo_id} placeholder="neuronpedia/jacobian-lens"
                  onChange={(e) => setLensForm({ ...lensForm, repo_id: e.target.value })} />
              </div>
              <div className="row"><label>file</label>
                <input type="text" value={lensForm.filename} placeholder="gemma-3-1b-it/jlens/…/gemma-3-1b-it_jacobian_lens.pt"
                  onChange={(e) => setLensForm({ ...lensForm, filename: e.target.value })} />
              </div>
              <div className="row"><label>revision</label>
                <input type="text" value={lensForm.revision} placeholder="main (default)"
                  onChange={(e) => setLensForm({ ...lensForm, revision: e.target.value })} />
              </div>
              <div className="row"><label>or path</label>
                <input type="text" value={lensForm.path} placeholder="lenses/my-fit/lens.pt (local file)"
                  onChange={(e) => setLensForm({ ...lensForm, path: e.target.value })} />
              </div>
              <div className="row"><label>top-k</label>
                <input type="number" min="1" max="32" value={lensForm.k} onChange={(e) => setLensForm({ ...lensForm, k: e.target.value })} />
              </div>
              <button className="primary"
                disabled={(!loadedId && !modelLoading) || (!lensForm.filename && !lensForm.path.trim()) || (!!busy && !modelLoading)}
                onClick={onLensLoad}>{modelLoading ? 'Queue lens' : 'Load lens'}</button>
            </details>
          </>
        )}

        {lensMeta && (
          <>
            <h2>Token editing ☢</h2>
            <button className="primary" onClick={() => openEditorWith(null)}>
              Open editor{ivRules.length ? ` (${ivRules.length} rule${ivRules.length > 1 ? 's' : ''})` : ''}
            </button>
            {ivScale !== 1 && <div className="status-line">global multiplier: ×{(+ivScale).toFixed(2)}</div>}
          </>
        )}
        </>)}

        {tab === 'fit' && (<>
        <h2>Fitting</h2>
        {(() => {
          const fit = status?.fit
          if (fit?.state === 'running' || fit?.state === 'stopping') {
            return (
              <div>
                <div className="status-line">
                  {fit.name} · {fit.phase === 'loading' ? 'loading model' : fit.phase === 'fitting' ? `fitting · ${fit.done} of ${fit.total} sequences complete` : fit.phase === 'merge' ? 'merging' : fit.phase} · {fit.done}/{fit.total}
                  {fit.elapsed != null ? ` · elapsed ${Math.max(0, Math.round(fit.elapsed))} sec` : ''}
                  {fit.eta_seconds ? ` · ETA ${Math.max(1, Math.round(fit.eta_seconds / 60))} min` : ''}
                </div>
                {(fit.workers || []).map((w, i) => (
                  <div className="gpu" key={i}>
                    <div className="name"><span>{w.device} · {w.state}{w.state === 'fitting' && w.done < w.total ? ` sequence ${w.done + 1}/${w.total}` : ''} · dim_batch ${w.dim_batch}</span><span>{w.done}/{w.total}</span></div>
                    <div className="bar"><div className="fill" style={{ width: `${(100 * w.done) / Math.max(1, w.total)}%` }} /></div>
                  </div>
                ))}
                <button className="danger" onClick={() => jsonFetch('/api/fit/stop', { method: 'POST' }).catch(() => {})}>
                  Stop (resumable)
                </button>
              </div>
            )
          }
          return (
            <>
              {fit?.state === 'done' && (
                <div className="status-line ok">
                  fit done: {fit.name} ({Math.round((fit.meta?.fit_seconds || 0) / 60)} min) — reload the model then{' '}
                  <button onClick={async () => {
                    try {
                      await jsonFetch('/api/lens/load', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ path: fit.lens_path }),
                      })
                      setNotice({ kind: 'ok', text: `lens ${fit.name} loaded` })
                    } catch (err) {
                      setNotice({ kind: 'err', text: String(err.message || err) })
                    }
                  }}>load lens</button>
                </div>
              )}
              {fit?.state === 'error' && <div className="status-line err">fit error: {fit.error}</div>}
              {fit?.state === 'stopped' && <div className="status-line">fit stopped (checkpoints kept, restart = resume)</div>}
              <div className="row"><label>continue</label>
                <select value={fitContinue} onChange={(e) => {
                  setFitContinue(e.target.value)
                  const l = localLenses.find((x) => x.path === e.target.value)
                  if (l?.meta?.model_id) setFitModel(l.meta.model_id)
                }}>
                  <option value="">— new lens —</option>
                  {localLenses.map((l) => (
                    <option key={l.path} value={l.path}>{l.name} (n={l.meta?.n_prompts ?? '?'})</option>
                  ))}
                </select>
              </div>
              {fitContinue && (
                <div className="status-line">resume: {fitN} prompts are added to the {localLenses.find((l) => l.path === fitContinue)?.meta?.n_prompts ?? '?'} existing ones (weighted average = fit over the union); source layers inherited</div>
              )}
              <div className="row"><label>model</label>
                <select value={fitModel} onChange={(e) => setFitModel(e.target.value)} disabled={!!fitContinue}>
                  <option value="">—</option>
                  {models.map((m) => <option key={m.id} value={m.id}>{m.id}{m.n_layers != null ? ` (${m.n_layers} l.)` : ''}</option>)}
                </select>
              </div>
              <div className="row"><label>name</label>
                <input type="text" placeholder="(auto: model_nN)" value={fitName} onChange={(e) => setFitName(e.target.value)} />
              </div>
              <div className="row"><label title="number of training sequences the fit iterates over — what you set is exactly what runs (existing lenses were made with n=100 unless marked _nNNN)">sequences</label>
                <input type="number" min="4" step="1" value={fitN} onChange={(e) => setFitN(e.target.value)} />
              </div>
              <div className="row"><label title="fit corpus. Tick one or several HuggingFace datasets; several ticked = mixed in equal parts (rounded to the nearest sequence), shuffled">datasets</label>
                <span style={{ display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
                  {fitDatasets.map((d, i) => (
                    <label key={d.id} style={{ width: 'auto', display: 'flex', alignItems: 'center', gap: 6 }} title={d.id}>
                      <input type="checkbox" checked={d.on}
                        onChange={(e) => setFitDatasets(fitDatasets.map((x, j) => j === i ? { ...x, on: e.target.checked } : x))} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{d.id}</span>
                      <button className="linkbtn" title="remove from the list (does not delete anything on disk)"
                        onClick={() => setFitDatasets(fitDatasets.filter((_, j) => j !== i))}>✕</button>
                    </label>
                  ))}
                  <span style={{ display: 'flex', gap: 6 }}>
                    <input type="text" placeholder="org/dataset (HuggingFace id)" value={fitDatasetInput}
                      onChange={(e) => setFitDatasetInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); addFitDataset() } }} />
                    <button onClick={addFitDataset} title="add this dataset to the list">+</button>
                  </span>
                </span>
              </div>
              <div className="row"><label>quant</label>
                <select value={fitQuant} onChange={(e) => setFitQuant(e.target.value)} disabled={!!fitContinue}>
                  <option value="">none (bf16)</option>
                  <option value="int8">int8</option>
                  <option value="nf4">nf4</option>
                </select>
              </div>

              <div className="row">
                <button className="linkbtn" onClick={() => setFitAdvanced(!fitAdvanced)}>
                  {fitAdvanced ? '▾' : '▸'} advanced settings
                </button>
              </div>
              {fitAdvanced && (
                <div className="fit-adv">
                  <div className="row"><label title="GPUs used (several = corpus split across them)">devices</label>
                    <span style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                      {(status?.gpus || []).map((g) => {
                        const d = `cuda:${g.index}`
                        return (
                          <label key={d} style={{ width: 'auto' }} title={g.name}>
                            <input type="checkbox" checked={fitDevices.includes(d)}
                              onChange={(e) => setFitDevices(e.target.checked
                                ? [...fitDevices, d].sort()
                                : fitDevices.filter((x) => x !== d))} /> {d}
                          </label>
                        )
                      })}
                    </span>
                  </div>
                  <div className="row"><label title="auto is selected conservatively by the backend from available VRAM; the resolved value appears in worker status">dim_batch</label>
                    <input type="number" min="1" step="1" placeholder="auto" value={fitDimBatch}
                      onChange={(e) => setFitDimBatch(e.target.value)} />
                  </div>
                  <div className="row"><label>max_seq_len</label>
                    <input type="number" min="8" step="8" value={fitMaxSeq} onChange={(e) => setFitMaxSeq(e.target.value)} />
                  </div>
                  <div className="row"><label title="layers to fit, e.g. 3-30 or 5,10,15; empty = library default">layers</label>
                    <input type="text" placeholder="(default)" value={fitLayers} disabled={!!fitContinue}
                      onChange={(e) => setFitLayers(e.target.value)} />
                  </div>
                  <div className="status-line">
                    max = second-to-last layer: the last one is the lens TARGET
                    (its readout is already exact, J = I, nothing to fit)
                  </div>
                  <div className="status-line">resume: restarting with the same name picks up from the checkpoints in data/fits/&lt;name&gt;/</div>
                </div>
              )}
              <button
                className="primary"
                disabled={(!fitModel && !fitContinue) || !fitDevices.length || !fitDatasets.some((d) => d.on) || !!loadedId || !!busy}
                onClick={async () => {
                  try {
                    const layers = []
                    for (const part of fitLayers.split(',')) {
                      const mm = part.trim().match(/^(\d+)\s*-\s*(\d+)$/)
                      if (mm) for (let i = +mm[1]; i <= +mm[2]; i++) layers.push(i)
                      else if (part.trim() && !isNaN(+part.trim())) layers.push(+part.trim())
                    }
                    await jsonFetch('/api/fit', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        model_id: fitModel, n_prompts: +fitN, quant: fitQuant || null,
                        datasets: fitDatasets.filter((d) => d.on).map((d) => d.id),
                        name: fitName.trim() || null, devices: fitDevices,
                        dim_batch: fitDimBatch ? +fitDimBatch : null,
                        max_seq_len: +fitMaxSeq,
                        source_layers: layers.length ? layers : null,
                        continue_from: fitContinue || null,
                      }),
                    })
                    setNotice({ kind: 'ok', text: `fitting started on ${fitDevices.join(', ')}` })
                  } catch (err) {
                    setNotice({ kind: 'err', text: String(err.message || err) })
                  }
                }}
              >{fitContinue ? 'Continue fit' : 'Start fit'}</button>
              {loadedId && <div className="status-line">unload the model first (VRAM required)</div>}
            </>
          )
        })()}
        </>)}

        {tab === 'model' && (<>
        <h2>Download</h2>
        <div className="row">
          <input type="text" placeholder="org/repo" value={downloadRepo} onChange={(e) => setDownloadRepo(e.target.value)} />
          <button onClick={onDownload}
            disabled={(status?.downloads || []).some((d) => d.repo_id === downloadRepo.trim() && d.state === 'running')}>↓</button>
        </div>
        {(status?.downloads || []).map((d) => {
          const planTxt = d.plan
            ? ` (${(d.plan.size_bytes / GB).toFixed(1)} GB${d.plan.kind === 'fallback' ? ' · full repo' : ''})`
            : ''
          const pr = d.progress
          const pct = pr && pr.total ? Math.min(100, (100 * pr.done) / pr.total) : null
          return (
            <div key={d.repo_id} className={`status-line ${d.state === 'error' ? 'err' : d.state === 'done' ? 'ok' : ''}`}>
              {d.state === 'running' && `downloading: ${d.repo_id}${planTxt}${pct != null ? ` — ${pct.toFixed(0)} %` : '...'}`}
              {d.state === 'converting' && `auto bf16 conversion: ${d.repo_id} (fp32 detected)...`}
              {d.state === 'done' && `downloaded: ${d.repo_id}${d.converted ? ` → converted: ${d.converted}` : ''}`}
              {d.state === 'error' && `failed ${d.repo_id}: ${d.error}`}
              {(d.state === 'done' || d.state === 'error') && (
                <span style={{ cursor: 'pointer', marginLeft: 6 }}
                  onClick={() => jsonFetch(`/api/download/${encodeURIComponent(d.repo_id)}`, { method: 'DELETE' }).catch(() => {})}> ✕</span>
              )}
              {d.state === 'running' && pct != null && (
                <div className="dl-bar"><div className="dl-fill" style={{ width: `${pct}%` }} /></div>
              )}
            </div>
          )
        })}
        {status?.convert?.state === 'running' && (
          <div className="status-line">bf16 conversion in progress: {status.convert.path}...</div>
        )}
        {status?.convert?.state === 'done' && (
          <div className="status-line ok">converted to bf16: {status.convert.result?.id}</div>
        )}
        {status?.convert?.state === 'error' && (
          <div className="status-line err">conversion failed: {status.convert.error}</div>
        )}

        <h2>Browse <button style={{ float: 'right', padding: '1px 7px' }}
          onClick={() => { const n = !browseOpen; setBrowseOpen(n); if (n && !browse) openBrowse(null) }}>
          {browseOpen ? 'hide' : 'open'}</button></h2>
        <div className="row">
          <button disabled={!!busy} onClick={() => pickNative('dir')}>🗂 Browse (folder)…</button>
        </div>
        {browseOpen && browse && (
          <div className="browser">
            <div className="browser-path">
              {browse.parent != null && <button className="linkbtn" onClick={() => openBrowse(browse.parent)}>⬆ parent</button>}
              <button className="linkbtn" onClick={() => openBrowse(null)}>drives</button>
              <span className="src"> {browse.path || '(drives)'}</span>
            </div>
            {browse.is_model && (
              <div className="reg-item2" style={{ marginBottom: 4 }}>
                <div className="reg-main"><div className="reg-name">📦 this folder is a model</div></div>
                <button title="add to the model list (nothing is loaded or copied)"
                  onClick={() => registerPath(browse.path)}>+ Add</button>
              </div>
            )}
            <div className="browser-list">
              {browse.dirs.map((d) => (
                <div key={d.path} className={`browser-row ${d.is_model ? 'is-model' : ''}`}>
                  <span className="browser-name" title={d.path} onClick={() => openBrowse(d.path)}>
                    {d.is_model ? '📦' : '📁'} {d.name}
                  </span>
                  {d.is_model && <button title="add to the model list (nothing is loaded or copied)"
                    onClick={() => registerPath(d.path)}>+ Add</button>}
                </div>
              ))}
              {!browse.dirs.length && <div className="src">(empty)</div>}
            </div>
          </div>
        )}
        </>)}

        {tab === 'options' && (<>
        <h2>Defaults</h2>
        <div className="row"><label title="quantization preselected in the Model tab">quant</label>
          <select value={settings?.default_quant ?? ''}
            onChange={(e) => patchSettings({ default_quant: e.target.value })}>
            <option value="">none</option>
            <option value="int8">int8</option>
            <option value="nf4">nf4</option>
          </select>
        </div>
        <div className="row">
          <label title="editor: when a token is added, its layers are auto-selected around the most-relevant one (peak ± this radius). Wider = more robust edit, narrower = more surgical.">auto layers ±</label>
          <input type="number" min="0" max="8" step="1" value={settings?.auto_layer_radius ?? 2}
            onChange={(e) => patchSettings({ auto_layer_radius: Math.max(0, Math.min(8, Math.trunc(+e.target.value) || 0)) })}
            style={{ width: 64, flex: 'none' }} />
        </div>
        <div className="row"><label>chat</label>
          <label style={{ width: 'auto' }}>
            <input type="checkbox" checked={settings ? !!settings.chat_markdown : chatMd}
              onChange={(e) => patchSettings({ chat_markdown: e.target.checked })} /> render replies as markdown
          </label>
        </div>
        <div className="row"><label title="while an external API/MCP client drives the model: stop the 2s status polling and refresh the view only when a generation finishes — keeps the UI fast">API monitor</label>
          <label style={{ width: 'auto' }}>
            <input type="checkbox" checked={apiMonitor} onChange={(e) => setApiMonitor(e.target.checked)} /> refresh only when an API generation ends
          </label>
        </div>

        <h2>Paths</h2>
        <div className="row">
          <label title="Hugging Face cache directory (downloaded models and lenses). Applied at the next server start; run.py --hf-cache overrides it.">HF cache</label>
          <input type="text" value={settings?.hf_cache ?? ''} placeholder="(shared HF cache)"
            onChange={(e) => setSettings((s) => ({ ...s, hf_cache: e.target.value }))}
            onBlur={(e) => patchSettings({ hf_cache: e.target.value.trim() })} />
          <button title="browse folders"
            onClick={async () => { const p = await pickFolder(); if (p) patchSettings({ hf_cache: p }) }}>🔍</button>
        </div>
        <div className="row">
          <label title="llama.cpp folder — needs convert_hf_to_gguf.py (and llama-quantize for quantized outputs). Setting it enables the direct GGUF export in the token editor.">llama.cpp</label>
          <input type="text" value={settings?.llamacpp_dir ?? ''} placeholder="(disables GGUF export)"
            onChange={(e) => setSettings((s) => ({ ...s, llamacpp_dir: e.target.value }))}
            onBlur={(e) => patchSettings({ llamacpp_dir: e.target.value.trim() })} />
          <button title="browse folders"
            onClick={async () => { const p = await pickFolder(); if (p) patchSettings({ llamacpp_dir: p }) }}>🔍</button>
        </div>
        <div className="src" style={{ marginTop: 4 }}>
          HF cache applies at the next server start. llama.cpp enables the direct
          GGUF export (editor → Export).
        </div>
        </>)}

        <div className="status-line">
          {busy && <span>({busy}) </span>}
          {/* no duplicate "busy: generating" when busy already shows the state */}
          {notice && !(busy && /^busy\s*:/.test(notice.text)) && <span className={notice.kind}>{notice.text}</span>}
        </div>
      </div>

      <div className="main">
        {conv?.id && (
          <div className="conv-header">
            <input
              className="conv-title-input"
              defaultValue={conv.title}
              key={`t${conv.id}`}
              onBlur={(e) => e.target.value !== conv.title && patchConv({ title: e.target.value })}
            />
            <input
              className="conv-tags-input"
              placeholder="tags (comma-separated)"
              defaultValue={conv.tags?.join(', ')}
              key={`g${conv.id}`}
              onBlur={(e) => patchConv({ tags: e.target.value.split(',').map((t) => t.trim()).filter(Boolean) })}
            />
            <button onClick={() => setShowTree(!showTree)}>{showTree ? 'Hide' : 'Tree'}</button>
            <a href={`/api/conversations/${conv.id}/export?format=json&frames=1`} download>JSON</a>
            <a href={`/api/conversations/${conv.id}/export?format=markdown`} download>MD</a>
          </div>
        )}

        {showTree && conv?.id && (
          <div className="tree-panel">
            {(childs[null] || []).map((root) => (
              <TreeNode
                key={root.id}
                node={root}
                childs={childs}
                depth={0}
                activeIds={activeIds}
                onSelect={(nodeId) => applyPath(nodeId, tree)}
              />
            ))}
            <div className="tree-hint">click a node = resume/branch from there</div>
          </div>
        )}

        <details className="sys">
          <summary>System prompt {system.trim() ? '●' : ''} {conv?.id ? '(set at creation)' : ''}</summary>
          <textarea value={system} onChange={(e) => setSystem(e.target.value)} placeholder="(none)" disabled={!!conv?.id} />
        </details>

        {status?.last_generation?.text && (
          <div style={{ border: '1px solid var(--border)', borderRadius: 6, padding: '6px 8px', marginBottom: 6, maxHeight: 160, overflow: 'auto' }}
            title="latest text generated through the API (CLI/MCP) — not saved to the conversation">
            <div style={{ color: 'var(--muted)', fontSize: 12, marginBottom: 4 }}>
              generated via API/MCP{status.last_generation.prompt ? ` · ${status.last_generation.prompt.slice(0, 80)}` : ''}
            </div>
            <div style={{ whiteSpace: 'pre-wrap', fontSize: 13 }}>{status.last_generation.text}</div>
          </div>
        )}

        <div className="messages" ref={messagesRef}>
          {messages.map((m, i) => (
            <div
              key={m.id ?? `tmp${i}`}
              className={`msg ${m.role} ${m.has_frames || m.frames?.length ? 'has-frames' : ''} ${selectedIdx === i ? 'msg-selected' : ''}`}
              onClick={() => { if (m.has_frames || m.frames?.length) selectMessage(i) }}
            >
              {editingMsg?.idx === i ? (
                <div className="msg-edit" onClick={(e) => e.stopPropagation()}>
                  <textarea
                    value={editingMsg.text}
                    onChange={(e) => setEditingMsg({ idx: i, text: e.target.value })}
                    rows={Math.min(16, Math.max(4, editingMsg.text.split('\n').length + 1))}
                  />
                  <div className="msg-edit-actions">
                    <button className="primary" onClick={saveEditedMsg}>Save</button>
                    <button onClick={() => setEditingMsg(null)}>Cancel</button>
                  </div>
                </div>
              ) : (() => {
                const text = continuingId != null && m.id === continuingId && draft !== null
                  ? m.content + draft
                  : m.content
                return m.role === 'assistant' && chatMd ? <Md text={text} /> : text
              })()}
              {m.meta && (
                <div className="msgmeta">
                  {m.meta.model_id} · {m.meta.quant || m.meta.dtype} · {m.meta.device} · T={m.meta.sampling?.temperature ?? '?'}
                  {m.meta.sampling?.seed >= 0 ? ` · seed=${m.meta.sampling.seed}` : ''} · {(m.stats || m.meta.stats)?.tok_per_s} tok/s
                  {m.meta.interventions?.length ? ` · ☢ ${m.meta.interventions.length} intervention(s)` : ''}
                  {m.has_frames || m.frames?.length ? ' · lens frames ◈' : ''}
                  {m.role === 'assistant' && m.id != null && (
                    <button
                      className="diff-btn"
                      title="edit this reply (later turns will use the edited text)"
                      onClick={(e) => { e.stopPropagation(); setEditingMsg({ idx: i, text: m.content }) }}
                    >✎</button>
                  )}
                  {(m.has_frames || m.frames?.length) ? (
                    <button
                      className="diff-btn"
                      title="A/B lens diff — pick TWO replies to compare their lens frames side by side"
                      onClick={async (e) => {
                        e.stopPropagation()
                        const key = m.id ?? `tmp${i}`
                        if (diffSel.find((x) => x.key === key)) {
                          setDiffSel((prev) => prev.filter((x) => x.key !== key))
                          return
                        }
                        let frames = m.frames
                        if (!frames?.length && m.id != null) {
                          try {
                            frames = (await jsonFetch(`/api/messages/${m.id}/frames`)).frames
                          } catch { return }
                        }
                        if (!frames?.length) return
                        setDiffSel((prev) => [...prev, { key, frames }].slice(-2))
                      }}
                    >{diffSel.find((x) => x.key === (m.id ?? `tmp${i}`)) ? `◧ ${diffSel.findIndex((x) => x.key === (m.id ?? `tmp${i}`)) === 0 ? 'A' : 'B'}` : '◧'}</button>
                  ) : null}
                </div>
              )}
            </div>
          ))}
          {draft !== null && continuingId == null && (
            <div className="msg assistant">{chatMd && draft ? <Md text={draft} /> : (draft || '…')}</div>
          )}
        </div>

        {diffSel.length === 2 ? (
          <LensDiff
            framesA={diffSel[0].frames}
            framesB={diffSel[1].frames}
            labelA={`#${diffSel[0].key}`}
            labelB={`#${diffSel[1].key}`}
            onClose={() => setDiffSel([])}
          />
        ) : lensMeta && lensOn || messages.some((m) => m.frames?.length) ? (() => {
          const { live, idx, genId: viewGen, generationRunId: viewRunId } = currentGenView()
          const viewFrames = live || (idx >= 0 ? messages[idx].frames : [])
          // keep LensView MOUNTED even with no frames (it renders an empty
          // shell): unmounting here would wipe the pinned tokens, e.g. while
          // regenerating the first reply of a conversation
          return (
            <>
              {viewFrames.length > 0 && (
                <div className="v-resizer" title="drag to set the lens view height — double-click for auto"
                  onMouseDown={startLensResize} onDoubleClick={() => setLensViewH(null)} />
              )}
              <LensView
                frames={viewFrames}
                tick={`${framesCount}-${messages.length}-${idx}`}
                genId={viewGen ?? null}
                generationRunId={viewRunId ?? null}
                lensMeta={lensMeta}
                hidden={hidden}
                onHideToken={hideToken}
                onNotice={(text, kind) => setNotice({ kind: kind || 'err', text })}
                onEditToken={(id, str, layer) => openEditorWith({ id, str, layer })}
                maxH={lensViewH}
                editorOpen={editorOpen}
                streaming={streaming}
              />
            </>
          )
        })() : null}

        <div className="composer">
          <div className="controls">
            <label>temp <input type="number" step="0.05" min="0" max="2" value={sampling.temperature}
              onChange={(e) => setSampling({ ...sampling, temperature: +e.target.value })} /></label>
            <label>top_p <input type="number" step="0.01" min="0" max="1" value={sampling.top_p}
              onChange={(e) => setSampling({ ...sampling, top_p: +e.target.value })} /></label>
            <label>top_k <input type="number" step="1" min="0" value={sampling.top_k}
              onChange={(e) => setSampling({ ...sampling, top_k: +e.target.value })} /></label>
            <label>max <input type="number" step="16" min="1" value={sampling.max_tokens}
              onChange={(e) => setSampling({ ...sampling, max_tokens: +e.target.value })} /></label>
            <label title="-1 = random; ≥ 0 = reproducible sampling">seed <input type="number" step="1" min="-1" value={sampling.seed}
              onChange={(e) => setSampling({ ...sampling, seed: Math.trunc(+e.target.value) })} /></label>
            <label title="repetition penalty — 1 = off, >1 penalizes tokens already in the context (curbs loops)">rep <input type="number" step="0.05" min="1" max="2" value={sampling.repetition_penalty ?? 1}
              onChange={(e) => setSampling({ ...sampling, repetition_penalty: +e.target.value })} /></label>
            <button onClick={onRegenerate} disabled={streaming || !messages.some((m) => m.role === 'assistant')}>Regenerate</button>
            <button onClick={onContinue}
              title="extend the last reply: the model picks up exactly where it stopped"
              disabled={streaming || messages[messages.length - 1]?.role !== 'assistant' || messages[messages.length - 1]?.id == null}>
              Continue</button>
            <button onClick={onEditLast} disabled={streaming || !messages.some((m) => m.role === 'user')}>Edit last</button>
            {lensMeta && (
              <button
                className={editorOpen ? 'ed-toggle-on' : ''}
                style={{ marginLeft: 'auto' }}
                onClick={() => setEditorOpen(!editorOpen)}
              >☢ Editor{ivRules.length ? ` (${ivRules.length})` : ''}{ivScale !== 1 ? ` ×${(+ivScale).toFixed(2)}` : ''}</button>
            )}
            {editParent !== undefined && <span className="fork-note">branch from #{editParent ?? 'root'}</span>}
          </div>
          <div className="inputrow">
            <textarea
              value={input}
              placeholder={loadedId ? 'Message... (Enter to send, Shift+Enter for a new line)' : 'Load a model to chat'}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend() }
              }}
            />
            {streaming
              ? <button className="danger" onClick={onStop}>Stop</button>
              : <button className="primary" onClick={onSend} disabled={!loadedId || !input.trim()}>Send</button>}
          </div>
        </div>
      </div>

      <Editor
        open={editorOpen}
        onClose={() => setEditorOpen(false)}
        rules={ivRules}
        scale={ivScale}
        mode={ivMode}
        lensMeta={lensMeta}
        nLayers={status?.loaded?.n_layers}
        genId={currentGenView().genId}
        generationRunId={currentGenView().generationRunId}
        busy={busy}
        prefill={editorPrefill}
        onPrefillConsumed={() => setEditorPrefill(null)}
        onRules={setIvRules}
        onScale={setIvScale}
        onMode={setIvMode}
        onNotice={(text, kind) => setNotice({ kind: kind || 'err', text })}
        rebaseSupported={status?.loaded?.rebase_supported}
        autoLayerRadius={settings?.auto_layer_radius}
        llamaCppSet={!!settings?.llamacpp_dir}
        ggufState={status?.gguf}
      />
    </>
  )
}
