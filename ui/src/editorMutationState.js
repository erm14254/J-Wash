export class EditorMutationCancellation extends Error {
  constructor() {
    super('Editor mutation cancelled.')
    this.name = 'EditorMutationCancellation'
  }
}

export function isEditorMutationCancellation(error) {
  return error instanceof EditorMutationCancellation || error?.name === 'AbortError'
}

export function throwIfEditorMutationCancelled(signal) {
  if (signal?.aborted) throw new EditorMutationCancellation()
}

export function createEditorMutationState({ setTimer = setTimeout, clearTimer = clearTimeout } = {}) {
  let current = true
  const timers = new Set()
  const controllers = new Set()

  function schedule(callback, delay) {
    const timer = setTimer(() => {
      timers.delete(timer)
      if (current) callback()
    }, delay)
    timers.add(timer)
    return timer
  }

  function cancelTimer(timer) {
    clearTimer(timer)
    timers.delete(timer)
  }

  async function request(factory, publish) {
    if (!current) throw new EditorMutationCancellation()
    const controller = new AbortController()
    controllers.add(controller)
    try {
      try {
        const value = await factory(controller.signal)
        if (!current) throw new EditorMutationCancellation()
        publish?.(value)
        return value
      } catch (error) {
        if (!current) throw new EditorMutationCancellation()
        throw error
      }
    } finally {
      controllers.delete(controller)
    }
  }

  function invalidate() {
    current = false
    for (const timer of timers) clearTimer(timer)
    timers.clear()
    for (const controller of controllers) controller.abort()
    controllers.clear()
  }

  return { schedule, cancelTimer, request, invalidate, isCurrent: () => current }
}
