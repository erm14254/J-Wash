import test from 'node:test'
import assert from 'node:assert/strict'
import { activeRuleCount, readCapabilityState, isCapabilityRefreshEvent } from '../src/capabilityState.js'

const yes = { supported: true, reason_code: 'supported', reason: 'Supported for this model.' }
const no = (code = 'future_denial', reason = 'Server says no.') => ({ supported: false, reason_code: code, reason })
function snapshot({ mode = 'readthrough', modes = {}, exports = {} } = {}) {
  return { model_session_id: 4, interventions_mode: mode, rebase_supported: true, capabilities: {
    modes: Object.fromEntries(['standard', 'readthrough', 'exact', 'abliteration'].map((id) => [id, modes[id] || yes])),
    exports: { mode, ...Object.fromEntries(['full', 'layers', 'lora', 'gguf'].map((id) => [id, exports[id] || yes])) },
  } }
}

test('malformed and stale status fail closed without legacy fallback', () => {
  for (const value of [null, {}, { rebase_supported: true }, snapshot()]) {
    const state = readCapabilityState(value, { fresh: value !== snapshot, lensAvailable: true, rules: [{ layers: [1] }] })
    if (value?.capabilities) continue
    assert.equal(state.valid, false)
    assert.equal(state.actions.addRule, false)
    assert.equal(state.formats.full.enabled, false)
    assert.equal(state.fallbackDecision.source, 'client-fallback')
  }
  assert.equal(readCapabilityState(snapshot(), { fresh: false }).valid, false)
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
