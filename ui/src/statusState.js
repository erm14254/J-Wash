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
  try {
    return await mutate()
  } finally {
    await refresh()
  }
}
