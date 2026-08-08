export const MODE_IDS = ['standard', 'readthrough', 'exact', 'abliteration']
export const EXPORT_IDS = ['full', 'layers', 'lora', 'gguf']

export const FALLBACK_DECISION = Object.freeze({
  source: 'client-fallback',
  supported: false,
  reason_code: null,
  reason: 'Capability diagnostics are unavailable; this operation has not been validated.',
})

function validDecision(value) {
  return value && typeof value.supported === 'boolean'
    && typeof value.reason_code === 'string' && value.reason_code.trim()
    && typeof value.reason === 'string' && value.reason.trim()
    && (value.supported ? value.reason_code === 'supported' : value.reason_code !== 'supported')
}

export function activeRuleCount(rules) {
  return Array.isArray(rules)
    ? rules.filter((r) => r?.enabled !== false && Array.isArray(r?.layers) && r.layers.length > 0).length
    : 0
}

export function readCapabilityState(status, options = {}) {
  const { fresh = true, llamaCppConfigured = null, lensAvailable = false,
    busy = false, rules = [], transitionPending = false } = options
  const caps = status?.capabilities
  const selectedMode = status?.interventions_mode
  const sessionReady = !!(fresh && status?.loaded && Number.isInteger(status?.model_session_id)
    && MODE_IDS.includes(selectedMode))
  const diagnosticsValid = !!(caps && MODE_IDS.every((id) => validDecision(caps.modes?.[id])))
  const exportsSynchronized = !!(caps?.exports && caps.exports.mode === selectedMode
    && EXPORT_IDS.every((id) => validDecision(caps.exports?.[id])))
  const valid = diagnosticsValid && exportsSynchronized
  const blocked = busy || transitionPending
  const decisions = caps || { modes: {}, exports: {} }
  const modes = Object.fromEntries(MODE_IDS.map((id) => {
    const decision = validDecision(decisions.modes?.[id]) ? decisions.modes[id] : FALLBACK_DECISION
    return [id, { decision, enabled: sessionReady && !blocked, selected: id === selectedMode }]
  }))
  const editingEnabled = sessionReady && lensAvailable && !blocked
  let localReason = null
  if (!sessionReady) localReason = 'Load a model and wait for fresh session status.'
  else if (!lensAvailable) localReason = 'Load a lens to edit interventions.'
  else if (blocked) localReason = transitionPending
    ? 'Refreshing authoritative capability status.' : 'Another operation is in progress.'
  const active = activeRuleCount(rules)
  const formats = Object.fromEntries(EXPORT_IDS.map((id) => {
    const decision = validDecision(decisions.exports?.[id]) ? decisions.exports[id] : FALLBACK_DECISION
    const reasons = []
    if (!active) reasons.push('Add at least one enabled rule with one or more layers.')
    if (id === 'gguf' && llamaCppConfigured === false) reasons.push('Configure llama.cpp in Options.')
    if (!sessionReady) reasons.push('Load a model and wait for fresh session status.')
    if (blocked) reasons.push(transitionPending ? 'Refreshing authoritative capability status.' : 'Another operation is in progress.')
    const local = { ready: reasons.length === 0, reason: reasons.join(' ') || null }
    return [id, { decision, diagnosticValidated: decision.supported === true,
      local, enabled: sessionReady && local.ready }]
  }))
  const cleanup = !busy && !transitionPending
  return {
    valid, diagnosticsValid, sessionReady, sessionId: status?.model_session_id ?? null, selectedMode,
    exportsSynchronized, fallbackDecision: FALLBACK_DECISION, modes,
    editing: { enabled: editingEnabled, localReason }, formats,
    actions: {
      changeScale: editingEnabled, addRule: editingEnabled,
      changeRuleDirections: editingEnabled, changeFactor: editingEnabled,
      enableRule: editingEnabled, applyPreset: editingEnabled,
      disableRule: cleanup, removeRule: cleanup, clearRules: cleanup,
      viewLens: true, cleanGgufCache: cleanup,
    },
    hasActiveRules: active > 0,
  }
}

export function isCapabilityRefreshEvent(message) {
  return message?.type === 'api_generation' || message?.type === 'capabilities_changed'
}

export function selectedExportState(capabilityState, exportId) {
  return { id: exportId, ...capabilityState.formats[exportId] }
}
