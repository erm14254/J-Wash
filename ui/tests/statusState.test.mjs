import test from 'node:test'
import assert from 'node:assert/strict'
import { createLatestStatusRefresher, runCapabilityMutation, createCapabilityMutationLatch, capabilitySocketOpened, capabilitySocketClosed } from '../src/statusState.js'

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
  assert.deepEqual(calls, ['invalidate', 'mutate', 'invalidate', 'refresh'])
})

test('rejected mutation still refreshes', async () => {
  const calls = []
  await assert.rejects(runCapabilityMutation({ invalidate: () => calls.push('invalidate'), mutate: async () => { calls.push('mutate'); throw new Error('no') }, refresh: async () => calls.push('refresh') }))
  assert.deepEqual(calls, ['invalidate', 'mutate', 'invalidate', 'refresh'])
})

test('successful mutation with failed reconciliation surfaces refresh failure', async () => {
  await assert.rejects(runCapabilityMutation({
    invalidate: () => {}, mutate: async () => 'ok',
    refresh: async () => { throw new Error('refresh failed') },
  }), /refresh failed/)
})

test('mutation failure remains primary when reconciliation also fails', async () => {
  await assert.rejects(runCapabilityMutation({
    invalidate: () => {}, mutate: async () => { throw new Error('mutation failed') },
    refresh: async () => { throw new Error('refresh failed') },
  }), /mutation failed/)
})

for (const rejected of [false, true]) test(`mutation barrier survives an incidental old poll (${rejected ? 'rejected' : 'successful'})`, async () => {
  const mutation = deferred(), reconciliation = deferred(), events = []
  let pending = false
  const run = createCapabilityMutationLatch({
    invalidate: () => events.push('invalidate'),
    refresh: async () => { events.push('refresh'); return reconciliation.promise },
    setPending: (value) => { pending = value; events.push(`pending:${value}`) },
  })
  const operation = run(() => mutation.promise)
  assert.equal(pending, true)
  events.push('old-poll-applied')
  assert.equal(pending, true, 'accepted pre-mutation poll must not release barrier')
  rejected ? mutation.reject(new Error('rejected')) : mutation.resolve('ok')
  await Promise.resolve(); await Promise.resolve()
  assert.deepEqual(events.slice(0, 5), ['pending:true', 'invalidate', 'old-poll-applied', 'invalidate', 'refresh'])
  assert.equal(pending, true)
  reconciliation.resolve('new-status')
  if (rejected) await assert.rejects(operation, /rejected/); else assert.equal(await operation, 'ok')
  assert.equal(pending, false)
})

test('socket open and close invalidate before refresh or reconnect', async () => {
  const events = []
  await capabilitySocketOpened({ invalidate: () => events.push('invalidate-open'), refresh: async () => events.push('refresh') })
  capabilitySocketClosed({ invalidate: () => events.push('invalidate-close'), reconnect: () => events.push('reconnect') })
  assert.deepEqual(events, ['invalidate-open', 'refresh', 'invalidate-close', 'reconnect'])
})
