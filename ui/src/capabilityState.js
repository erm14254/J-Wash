export const MODE_IDS = ['standard', 'readthrough', 'exact', 'abliteration']
export const EXPORT_IDS = ['full', 'layers', 'lora', 'gguf']

export const FALLBACK_DECISION = Object.freeze({
  source: 'client-fallback',
  supported: false,
  reason_code: null,
  reason: 'Authoritative capability status is unavailable. Refresh and try again.',
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
  const { fresh = true, llamaCppConfigured = false, lensAvailable = false,
    busy = false, rules = [], transitionPending = false } = options
  const caps = status?.capabilities
  const selectedMode = status?.interventions_mode
  const structural = fresh && caps && MODE_IDS.includes(selectedMode)
    && caps.exports?.mode === selectedMode
    && MODE_IDS.every((id) => validDecision(caps.modes?.[id]))
    && EXPORT_IDS.every((id) => validDecision(caps.exports?.[id]))
  const valid = !!structural
  const blocked = busy || transitionPending
  const decisions = valid ? caps : { modes: {}, exports: {} }
  const modes = Object.fromEntries(MODE_IDS.map((id) => {
    const decision = valid ? decisions.modes[id] : FALLBACK_DECISION
    return [id, { decision, enabled: valid && decision.supported && !blocked, selected: id === selectedMode }]
  }))
  const selectedDecision = valid ? decisions.modes[selectedMode] : FALLBACK_DECISION
  const standardDecision = valid ? decisions.modes.standard : FALLBACK_DECISION
  const editingEnabled = valid && standardDecision.supported && selectedDecision.supported
    && lensAvailable && !blocked
  let localReason = null
  if (valid && !lensAvailable) localReason = 'Load a lens to edit interventions.'
  else if (valid && blocked) localReason = transitionPending
    ? 'Refreshing authoritative capability status.' : 'Another operation is in progress.'
  const blockingDecision = !valid ? FALLBACK_DECISION
    : !standardDecision.supported ? standardDecision
      : !selectedDecision.supported ? selectedDecision : null
  const active = activeRuleCount(rules)
  const formats = Object.fromEntries(EXPORT_IDS.map((id) => {
    const decision = valid ? decisions.exports[id] : FALLBACK_DECISION
    const reasons = []
    if (!active) reasons.push('Add at least one enabled rule with one or more layers.')
    if (id === 'gguf' && !llamaCppConfigured) reasons.push('Configure llama.cpp in Options.')
    if (blocked) reasons.push(transitionPending ? 'Refreshing authoritative capability status.' : 'Another operation is in progress.')
    const local = { ready: reasons.length === 0, reason: reasons.join(' ') || null }
    const serverEnabled = valid && decision.supported === true
    return [id, { decision, serverEnabled, local, enabled: serverEnabled && local.ready }]
  }))
  const cleanup = !busy && !transitionPending
  return {
    valid, sessionId: status?.model_session_id ?? null, selectedMode,
    exportsSynchronized: valid, fallbackDecision: FALLBACK_DECISION, modes,
    editing: { enabled: editingEnabled, blockingDecision, localReason }, formats,
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

export function shouldOpenAdvanced(selectedMode) {
  return selectedMode === 'exact' || selectedMode === 'abliteration'
}
