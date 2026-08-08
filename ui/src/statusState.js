export function createLatestStatusRefresher({ fetchStatus, applyStatus, invalidate }) {
  let epoch = 0
  const refresh = async () => {
    const requestEpoch = ++epoch
    try {
      const status = await fetchStatus()
      if (requestEpoch === epoch) applyStatus(status)
      return status
    } catch (error) {
      if (requestEpoch === epoch) invalidate(error)
      throw error
    }
  }
  refresh.invalidate = (reason) => { ++epoch; invalidate(reason) }
  return refresh
}

export async function runCapabilityMutation({ invalidate, mutate, refresh }) {
  invalidate()
  let value
  let mutationError
  try {
    value = await mutate()
  } catch (error) {
    mutationError = error
  }
  invalidate()
  try {
    await refresh()
  } catch (refreshError) {
    if (!mutationError) throw refreshError
  }
  if (mutationError) throw mutationError
  return value
}

export function createCapabilityMutationLatch({ invalidate, refresh, setPending }) {
  const active = new Set()
  return async function run(mutate) {
    const token = Symbol('capability-mutation')
    active.add(token)
    if (active.size === 1) setPending(true)
    try {
      return await runCapabilityMutation({ invalidate, mutate, refresh })
    } finally {
      active.delete(token)
      if (active.size === 0) setPending(false)
    }
  }
}

export function capabilitySocketConnecting({ invalidate }) {
  invalidate()
}

export function capabilitySocketClosed({ invalidate, reconnect }) {
  invalidate()
  return reconnect()
}
