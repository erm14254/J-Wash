import test from 'node:test'
import assert from 'node:assert/strict'
import { createEditorMutationState, isEditorMutationCancellation, throwIfEditorMutationCancelled } from '../src/editorMutationState.js'

const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }

test('invalidation and unmount teardown cancel queued factor and scale timers', () => {
  const callbacks = new Map(); let id = 0
  const state = createEditorMutationState({
    setTimer: (fn) => { callbacks.set(++id, fn); return id },
    clearTimer: (timer) => callbacks.delete(timer),
  })
  state.schedule(() => assert.fail('factor request fired'), 350)
  state.schedule(() => assert.fail('scale request fired'), 300)
  state.invalidate()
  assert.equal(callbacks.size, 0)
})

test('delayed old-session response cannot publish parent rules or scale', async () => {
  const pending = deferred(); const published = []
  const state = createEditorMutationState()
  const request = state.request(() => pending.promise, (value) => published.push(value))
  state.invalidate()
  pending.resolve({ rules: ['old'], scale: 9 })
  await assert.rejects(request, isEditorMutationCancellation)
  assert.deepEqual(published, [])
})

test('request after invalidation never invokes its factory', async () => {
  const state = createEditorMutationState(); let invoked = false
  state.invalidate()
  await assert.rejects(state.request(() => { invoked = true }, () => {}), isEditorMutationCancellation)
  assert.equal(invoked, false)
})

test('token preflight cannot continue to intervention mutation after turnover', async () => {
  const preflight = deferred(); const state = createEditorMutationState()
  const effects = []
  const operation = state.request(async (signal) => {
    await preflight.promise
    throwIfEditorMutationCancelled(signal)
    effects.push('intervention-request')
    return { rules: [] }
  }, () => effects.push('publish/reset/success'))
  state.invalidate()
  preflight.resolve(1)
  await assert.rejects(operation, isEditorMutationCancellation)
  assert.deepEqual(effects, [])
})

test('expected cancellation is silent-classified while real failures remain reportable', () => {
  assert.equal(isEditorMutationCancellation(new DOMException('aborted', 'AbortError')), true)
  assert.equal(isEditorMutationCancellation(new Error('server failed')), false)
})
