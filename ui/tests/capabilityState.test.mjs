import test from 'node:test'
import assert from 'node:assert/strict'
import { activeRuleCount, readCapabilityState, isCapabilityRefreshEvent, selectedExportState, shouldOpenAdvanced } from '../src/capabilityState.js'

const yes = { supported: true, reason_code: 'supported', reason: 'Supported for this model.' }
const no = (code = 'future_denial', reason = 'Server says no.') => ({ supported: false, reason_code: code, reason })
function snapshot({ mode = 'readthrough', modes = {}, exports = {} } = {}) {
  return { model_session_id: 4, interventions_mode: mode, rebase_supported: true, capabilities: {
    modes: Object.fromEntries(['standard', 'readthrough', 'exact', 'abliteration'].map((id) => [id, { ...(modes[id] || yes) }])),
    exports: { mode, ...Object.fromEntries(['full', 'layers', 'lora', 'gguf'].map((id) => [id, { ...(exports[id] || yes) }])) },
  } }
}

test('malformed and stale status fail closed without legacy fallback', () => {
  for (const value of [null, {}, { rebase_supported: true }]) {
    const state = readCapabilityState(value, { fresh: true, lensAvailable: true, rules: [{ layers: [1] }] })
    assert.equal(state.valid, false)
    assert.equal(state.actions.addRule, false)
    assert.equal(state.formats.full.enabled, false)
    assert.equal(state.fallbackDecision.source, 'client-fallback')
  }
  assert.equal(readCapabilityState(snapshot(), { fresh: false }).valid, false)
})

test('every structural incoherence fails closed', () => {
  const cases = []
  let value = snapshot(); delete value.capabilities.modes.exact; cases.push(value)
  value = snapshot(); delete value.capabilities.exports.lora; cases.push(value)
  value = snapshot(); value.interventions_mode = 'future'; cases.push(value)
  value = snapshot(); value.capabilities.exports.mode = 'exact'; cases.push(value)
  value = snapshot(); value.capabilities.modes.standard.supported = 1; cases.push(value)
  value = snapshot(); value.capabilities.modes.standard.reason_code = 'denied'; cases.push(value)
  for (const malformed of cases) assert.equal(readCapabilityState(malformed).valid, false)
})

test('future reasons and independent export decisions pass through verbatim', () => {
  const reason = 'A future server explanation.'
  const state = readCapabilityState(snapshot({ exports: { layers: no('new_code', reason) } }), {
    lensAvailable: true, llamaCppConfigured: true, rules: [{ layers: [2] }],
  })
  assert.equal(state.valid, true)
  assert.equal(state.formats.full.enabled, true)
  assert.equal(state.formats.layers.enabled, false)
  assert.equal(state.formats.layers.decision.reason, reason)
})

test('GGUF local setup remains separate from its server decision', () => {
  const state = readCapabilityState(snapshot({ exports: { gguf: no('denied', 'Canonical denial.') } }), { rules: [] })
  assert.equal(state.formats.gguf.decision.reason, 'Canonical denial.')
  assert.match(state.formats.gguf.local.reason, /llama\.cpp/)
})

test('GGUF supported reason remains available beside missing local setup', () => {
  const state = readCapabilityState(snapshot(), { rules: [{ layers: [0] }] })
  assert.equal(state.formats.gguf.serverEnabled, true)
  assert.equal(state.formats.gguf.decision.reason, yes.reason)
  assert.equal(state.formats.gguf.local.reason, 'Configure llama.cpp in Options.')
})

test('active rule counting uses enabled nonempty-layer rules exactly', () => {
  assert.equal(activeRuleCount([{}, { enabled: false, layers: [1] }, { layers: [] }, { layers: [0] }]), 1)
})

test('unsupported selected mode stays selected while recovery remains enabled', () => {
  const state = readCapabilityState(snapshot({ modes: { readthrough: no() } }), { lensAvailable: true })
  assert.equal(state.modes.readthrough.selected, true)
  assert.equal(state.modes.readthrough.enabled, false)
  assert.equal(state.modes.standard.enabled, true)
  assert.equal(state.editing.enabled, false)
})

test('refresh event classifier ignores generation frames', () => {
  assert.equal(isCapabilityRefreshEvent({ type: 'api_generation' }), true)
  assert.equal(isCapabilityRefreshEvent({ type: 'capabilities_changed' }), true)
  assert.equal(isCapabilityRefreshEvent({ type: 'token' }), false)
})

test('export selection is external and remains denied across sessions', () => {
  const selected = 'lora'
  const next = readCapabilityState(snapshot({ exports: { lora: no('new_denial', 'LoRA denied on session B.') } }), {
    rules: [{ layers: [0] }], llamaCppConfigured: true,
  })
  const item = selectedExportState(next, selected)
  assert.equal(item.id, 'lora')
  assert.equal(item.enabled, false)
  assert.equal(item.decision.reason, 'LoRA denied on session B.')
})

test('cleanup survives stale status unless a transition or busy operation blocks it', () => {
  const state = readCapabilityState(snapshot(), { fresh: false })
  assert.equal(state.actions.disableRule, true)
  assert.equal(state.actions.removeRule, true)
  assert.equal(state.actions.clearRules, true)
  assert.equal(state.actions.addRule, false)
})

test('advanced disclosure opens for authoritative advanced selections', () => {
  assert.equal(shouldOpenAdvanced('exact'), true)
  assert.equal(shouldOpenAdvanced('abliteration'), true)
  assert.equal(shouldOpenAdvanced('standard'), false)
})
