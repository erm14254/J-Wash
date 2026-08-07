import test from 'node:test'
import assert from 'node:assert/strict'
import { createEditorMutationState } from '../src/editorMutationState.js'

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
  await request
  assert.deepEqual(published, [])
})
