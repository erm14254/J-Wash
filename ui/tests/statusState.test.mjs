import test from 'node:test'
import assert from 'node:assert/strict'
import { createLatestStatusRefresher, runCapabilityMutation, createCapabilityMutationLatch, capabilitySocketConnecting, capabilitySocketClosed } from '../src/statusState.js'

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

test('socket connecting invalidates before handshake refresh and close invalidates before reconnect', async () => {
  const events = []
  capabilitySocketConnecting({ invalidate: () => events.push('invalidate-connecting') })
  events.push('socket-constructed')
  await (async () => events.push('refresh-on-open'))()
  capabilitySocketClosed({ invalidate: () => events.push('invalidate-close'), reconnect: () => events.push('reconnect') })
  assert.deepEqual(events, ['invalidate-connecting', 'socket-constructed', 'refresh-on-open', 'invalidate-close', 'reconnect'])
})

for (const finishBFirst of [true, false]) test(`overlapping mutations retain barrier when ${finishBFirst ? 'B' : 'A'} finishes first`, async () => {
  const a = deferred(), b = deferred(), refreshes = []
  const pendingEvents = []; let pending = false
  const run = createCapabilityMutationLatch({
    invalidate: () => {},
    refresh: () => { const item = deferred(); refreshes.push(item); return item.promise },
    setPending: (value) => { pending = value; pendingEvents.push(value) },
  })
  const operationA = run(() => a.promise)
  const operationB = run(() => b.promise)
  assert.equal(pending, true)
  const first = finishBFirst ? b : a
  first.resolve('first')
  await Promise.resolve(); await Promise.resolve()
  assert.equal(refreshes.length, 1)
  refreshes[0].resolve('first-status')
  await (finishBFirst ? operationB : operationA)
  assert.equal(pending, true, 'finishing a non-last operation must not clear pending')

  const second = finishBFirst ? a : b
  if (finishBFirst) second.reject(new Error('A rejected')); else second.resolve('second')
  await Promise.resolve(); await Promise.resolve()
  assert.equal(refreshes.length, 2)
  refreshes[1].resolve('second-status')
  if (finishBFirst) await assert.rejects(operationA, /A rejected/); else await operationB
  assert.equal(pending, false)
  assert.deepEqual(pendingEvents, [true, false])
})
