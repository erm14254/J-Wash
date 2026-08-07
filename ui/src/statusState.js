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
  let epoch = 0
  return async function run(mutate) {
    const ownEpoch = ++epoch
    setPending(true)
    try {
      return await runCapabilityMutation({ invalidate, mutate, refresh })
    } finally {
      if (ownEpoch === epoch) setPending(false)
    }
  }
}

export function capabilitySocketOpened({ invalidate, refresh }) {
  invalidate()
  return refresh()
}

export function capabilitySocketClosed({ invalidate, reconnect }) {
  invalidate()
  return reconnect()
}
