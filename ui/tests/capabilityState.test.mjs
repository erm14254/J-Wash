import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { activeRuleCount, readCapabilityState, isCapabilityRefreshEvent } from '../src/capabilityState.js'

const yes = { supported: true, reason_code: 'supported', reason: 'Supported for this model.' }
const no = (code = 'future_denial', reason = 'Server says no.') => ({ supported: false, reason_code: code, reason })
function snapshot({ mode = 'readthrough', modes = {}, exports = {} } = {}) {
  return { loaded: { model_id: 'fake/model' }, model_session_id: 4, interventions_mode: mode, rebase_supported: true, capabilities: {
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
  assert.equal(readCapabilityState(snapshot(), { fresh: false }).sessionReady, false)
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

test('future reasons are advisory and pass through verbatim', () => {
  const reason = 'A future server explanation.'
  const state = readCapabilityState(snapshot({ exports: { layers: no('new_code', reason) } }), {
    lensAvailable: true, llamaCppConfigured: true, rules: [{ layers: [2] }],
  })
  assert.equal(state.valid, true)
  assert.equal(state.formats.full.enabled, true)
  assert.equal(state.formats.layers.enabled, true)
  assert.equal(state.formats.layers.decision.reason, reason)
})

test('GGUF local setup remains separate from its server decision', () => {
  const state = readCapabilityState(snapshot({ exports: { gguf: no('denied', 'Canonical denial.') } }), { rules: [], llamaCppConfigured: false })
  assert.equal(state.formats.gguf.decision.reason, 'Canonical denial.')
  assert.match(state.formats.gguf.local.reason, /llama\.cpp/)
})

test('GGUF supported reason remains available beside missing local setup', () => {
  const state = readCapabilityState(snapshot(), { rules: [{ layers: [0] }], llamaCppConfigured: false })
  assert.equal(state.formats.gguf.decision.reason, yes.reason)
  assert.equal(state.formats.gguf.local.reason, 'Configure llama.cpp in Options.')
})

test('active rule counting uses enabled nonempty-layer rules exactly', () => {
  assert.equal(activeRuleCount([{}, { enabled: false, layers: [1] }, { layers: [] }, { layers: [0] }]), 1)
})

test('unsupported selected mode stays selected while all operations remain enabled', () => {
  const state = readCapabilityState(snapshot({ modes: { readthrough: no() } }), { lensAvailable: true })
  assert.equal(state.modes.readthrough.selected, true)
  assert.equal(state.modes.readthrough.enabled, true)
  assert.equal(state.modes.standard.enabled, true)
  assert.equal(state.editing.enabled, true)
})

test('refresh event classifier ignores generation frames', () => {
  assert.equal(isCapabilityRefreshEvent({ type: 'api_generation' }), true)
  assert.equal(isCapabilityRefreshEvent({ type: 'capabilities_changed' }), true)
  assert.equal(isCapabilityRefreshEvent({ type: 'token' }), false)
})

test('export selection is external and advisory across sessions', () => {
  const selected = 'lora'
  const next = readCapabilityState(snapshot({ exports: { lora: no('new_denial', 'LoRA denied on session B.') } }), {
    rules: [{ layers: [0] }], llamaCppConfigured: true,
  })
  const item = next.formats[selected]
  assert.equal(item.enabled, true)
  assert.equal(item.decision.reason, 'LoRA denied on session B.')
})

test('cleanup survives stale status unless a transition or busy operation blocks it', () => {
  const state = readCapabilityState(snapshot(), { fresh: false })
  assert.equal(state.actions.disableRule, true)
  assert.equal(state.actions.removeRule, true)
  assert.equal(state.actions.clearRules, true)
  assert.equal(state.actions.addRule, false)
  assert.equal(state.modes.readthrough.enabled, false)
  assert.equal(state.modes.readthrough.mechanicalReason,
    'Load a model and wait for fresh session status.')
})

test('malformed diagnostics remain advisory for a fresh coherent session', () => {
  const value = snapshot()
  delete value.capabilities.modes.exact
  const state = readCapabilityState(value, { lensAvailable: true, rules: [{ layers: [0] }] })
  assert.equal(state.diagnosticsValid, false)
  assert.equal(state.sessionReady, true)
  assert.equal(state.modes.exact.enabled, true)
  assert.equal(state.actions.addRule, true)
  assert.equal(state.formats.full.enabled, true)
  assert.match(state.modes.exact.decision.reason, /not been validated/)
})

test('missing diagnostics remain advisory for a fresh coherent session', () => {
  const value = snapshot()
  delete value.capabilities
  const state = readCapabilityState(value, {
    lensAvailable: true, rules: [{ layers: [0] }], llamaCppConfigured: true,
  })
  assert.equal(state.diagnosticsValid, false)
  assert.equal(state.sessionReady, true)
  assert.equal(state.modes.readthrough.enabled, true)
  assert.equal(state.actions.addRule, true)
  assert.equal(state.formats.full.enabled, true)
})

test('busy and transition states expose mechanical mode reasons and block attempts', () => {
  const busy = readCapabilityState(snapshot(), { busy: true, lensAvailable: true })
  assert.equal(busy.modes.readthrough.enabled, false)
  assert.equal(busy.modes.readthrough.mechanicalReason, 'Another operation is in progress.')
  const transition = readCapabilityState(snapshot(), { transitionPending: true, lensAvailable: true })
  assert.equal(transition.modes.exact.enabled, false)
  assert.equal(transition.modes.exact.mechanicalReason, 'Refreshing session status.')
})

test('normal Editor mode and export rows do not present capability confidence', () => {
  const source = readFileSync(new URL('../src/Editor.jsx', import.meta.url), 'utf8')
  assert.doesNotMatch(source, /item\.decision\.(?:reason|supported)/)
  assert.doesNotMatch(source, /cap-advisory|⚠ Experimental|Validated:/)
})

test('llama.cpp readiness is tri-state', () => {
  const options = { rules: [{ layers: [0] }] }
  assert.equal(readCapabilityState(snapshot(), { ...options, llamaCppConfigured: true }).formats.gguf.enabled, true)
  assert.equal(readCapabilityState(snapshot(), { ...options, llamaCppConfigured: null }).formats.gguf.enabled, true)
  assert.equal(readCapabilityState(snapshot(), { ...options, llamaCppConfigured: false }).formats.gguf.enabled, false)
})
