import test from 'node:test'
import assert from 'node:assert/strict'
import { createLatestStatusRefresher, runCapabilityMutation } from '../src/statusState.js'

const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b }); return { promise, resolve, reject } }

test('latest request wins and older completion cannot overwrite', async () => {
  const a = deferred(), b = deferred(), applied = [], invalid = []
  let n = 0
  const refresh = createLatestStatusRefresher({ fetchStatus: () => (++n === 1 ? a.promise : b.promise), applyStatus: (x) => applied.push(x), invalidate: (x) => invalid.push(x) })
  const pa = refresh(), pb = refresh(); b.resolve('new'); await pb; a.resolve('old'); await pa
  assert.deepEqual(applied, ['new']); assert.deepEqual(invalid, [])
})

test('old failure cannot invalidate newer success; latest failure does', async () => {
  const a = deferred(), b = deferred(), c = deferred(), applied = [], invalid = []
  let n = 0
  const refresh = createLatestStatusRefresher({ fetchStatus: () => [a, b, c][n++].promise, applyStatus: (x) => applied.push(x), invalidate: (x) => invalid.push(x) })
  const pa = refresh().catch(() => {}), pb = refresh(); b.resolve('new'); await pb; a.reject(new Error('old')); await pa
  assert.deepEqual(applied, ['new']); assert.equal(invalid.length, 0)
  const latest = refresh().catch(() => {}); c.reject(new Error('latest')); await latest
  assert.equal(invalid.length, 1)
})

test('mutation always orders invalidate, mutation, refresh', async () => {
  const calls = []
  await runCapabilityMutation({ invalidate: () => calls.push('invalidate'), mutate: async () => calls.push('mutate'), refresh: async () => calls.push('refresh') })
  assert.deepEqual(calls, ['invalidate', 'mutate', 'refresh'])
})

test('rejected mutation still refreshes', async () => {
  const calls = []
  await assert.rejects(runCapabilityMutation({ invalidate: () => calls.push('invalidate'), mutate: async () => { calls.push('mutate'); throw new Error('no') }, refresh: async () => calls.push('refresh') }))
  assert.deepEqual(calls, ['invalidate', 'mutate', 'refresh'])
})
