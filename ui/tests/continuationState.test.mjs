import test from 'node:test'
import assert from 'node:assert/strict'
import { applyGenerationTerminal, frameArchivePinIdentity, generationPinIdentity } from '../src/continuationState.js'

const existing = [{ id: 7, role: 'assistant', content: 'old', frames: [{ gen: 1 }] }]

test('successful continuation updates its existing assistant', () => {
  const result = applyGenerationTerminal(existing, {
    type: 'done', continued: true, message_id: 7, text: ' plus', content: 'old plus',
    gen_id: 2, generation_run_id: 'a'.repeat(32), pin_publication_state: 'published',
  }, [{ gen: 2 }])
  assert.equal(result.messages.length, 1)
  assert.equal(result.messages[0].content, 'old plus')
  assert.equal(result.messages[0].content === ' plus', false)
  assert.equal(result.reloadFrames, true)
  assert.equal(result.clearContinuation, true)
})

test('frame archive exposes pin identity only after durable publication', () => {
  assert.deepEqual(frameArchivePinIdentity({
    pin_publication_state: 'published', pin_gen_id: 2,
    pin_generation_run_id: 'a'.repeat(32),
  }), {
    pin_publication_state: 'published', gen_id: 2,
    generation_run_id: 'a'.repeat(32),
  })
  for (const state of ['pending', 'unavailable']) {
    assert.deepEqual(frameArchivePinIdentity({
      pin_publication_state: state, pin_gen_id: 9,
      pin_generation_run_id: 'b'.repeat(32),
    }), { pin_publication_state: state, gen_id: null, generation_run_id: null })
  }
})

test('continuation no-op preserves durable message and clears attempt state', () => {
  const result = applyGenerationTerminal(existing, {
    type: 'done', continuation_noop: true, continued: false, message_id: 7,
    text: '', content: 'old',
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

test('ordinary published assistant is immediately pinnable', () => {
  const runId = 'c'.repeat(32)
  const created = applyGenerationTerminal([], {
    type: 'done', message_id: 8, text: 'new', gen_id: 4,
    generation_run_id: runId, pin_publication_state: 'published',
  }, [{ gen: 4 }])
  assert.deepEqual(generationPinIdentity(created.messages[0]), {
    gen_id: 4, generation_run_id: runId, pin_publication_state: 'published',
  })
})

test('ordinary nonpublished assistant fails closed for pinning', () => {
  for (const state of ['pending', 'unavailable', undefined]) {
    const frame = {
      type: 'done', message_id: 8, text: 'new', gen_id: 4,
      generation_run_id: 'd'.repeat(32),
    }
    if (state !== undefined) frame.pin_publication_state = state
    const message = applyGenerationTerminal([], frame, [{ gen: 4 }]).messages[0]
    assert.equal(message.gen_id, undefined)
    assert.equal(message.generation_run_id, undefined)
    assert.deepEqual(generationPinIdentity(message), {
      gen_id: null, generation_run_id: null,
      pin_publication_state: state ?? 'unavailable',
    })
  }
})

test('error clears continuation attempt state', () => {
  const result = applyGenerationTerminal(existing, { type: 'error', message: 'failed' })
  assert.equal(result.messages, existing)
  assert.equal(result.clearContinuation, true)
  assert.equal(result.clearAttempt, true)
})
