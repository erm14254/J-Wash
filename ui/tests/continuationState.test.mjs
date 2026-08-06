import test from 'node:test'
import assert from 'node:assert/strict'
import { applyGenerationTerminal } from '../src/continuationState.js'

const existing = [{ id: 7, role: 'assistant', content: 'old', frames: [{ gen: 1 }] }]

test('successful continuation updates its existing assistant', () => {
  const result = applyGenerationTerminal(existing, {
    type: 'done', continued: true, message_id: 7, text: 'old plus',
    gen_id: 2, generation_run_id: 'a'.repeat(32),
  }, [{ gen: 2 }])
  assert.equal(result.messages.length, 1)
  assert.equal(result.messages[0].content, 'old plus')
  assert.equal(result.reloadFrames, true)
  assert.equal(result.clearContinuation, true)
})

test('continuation no-op preserves durable message and clears attempt state', () => {
  const result = applyGenerationTerminal(existing, {
    type: 'done', continuation_noop: true, continued: false, message_id: 7,
    text: 'old', gen_id: 9, generation_run_id: 'b'.repeat(32),
  }, [{ gen: 9 }])
  assert.equal(result.messages, existing)
  assert.equal(result.messages.length, 1)
  assert.equal(result.messages[0].content, 'old')
  assert.equal(result.clearContinuation, true)
  assert.equal(result.clearAttempt, true)
  assert.equal(result.reloadFrames, false)
})

test('ordinary assistant appends once and duplicate id is ignored', () => {
  const created = applyGenerationTerminal([], { type: 'done', message_id: 8, text: 'new' }, [])
  assert.equal(created.messages.length, 1)
  const duplicate = applyGenerationTerminal(created.messages, { type: 'done', message_id: 8, text: 'new' }, [])
  assert.equal(duplicate.messages.length, 1)
})

test('error clears continuation attempt state', () => {
  const result = applyGenerationTerminal(existing, { type: 'error', message: 'failed' })
  assert.equal(result.messages, existing)
  assert.equal(result.clearContinuation, true)
  assert.equal(result.clearAttempt, true)
})
