export function applyGenerationTerminal(messages, frame, frames = []) {
  if (frame.type === 'error') {
    return { messages, clearContinuation: true, clearAttempt: true, reloadFrames: false }
  }
  if (frame.type !== 'done') {
    return { messages, clearContinuation: false, clearAttempt: false, reloadFrames: false }
  }
  if (frame.continuation_noop) {
    return { messages, clearContinuation: true, clearAttempt: true, reloadFrames: false }
  }
  if (frame.continued && frame.message_id != null) {
    const durableContent = frame.content !== undefined ? frame.content : frame.text
    const pin = generationPinIdentity(frame)
    return {
      messages: messages.map((message) => message.id === frame.message_id ? {
        ...message,
        content: durableContent,
        stats: frame.stats,
        gen_id: pin.gen_id ?? undefined,
        generation_run_id: pin.generation_run_id ?? undefined,
        pin_publication_state: pin.pin_publication_state,
        has_frames: message.has_frames || frames.length > 0,
        frames: undefined,
      } : message),
      clearContinuation: true,
      clearAttempt: true,
      reloadFrames: true,
    }
  }
  if (frame.message_id != null && messages.some((message) => message.id === frame.message_id)) {
    return { messages, clearContinuation: true, clearAttempt: true, reloadFrames: false }
  }
  const pin = generationPinIdentity(frame)
  return {
    messages: [...messages, {
      id: frame.message_id ?? null,
      role: 'assistant',
      content: frame.text,
      meta: frame.meta,
      stats: frame.stats,
      gen_id: pin.gen_id ?? undefined,
      generation_run_id: pin.generation_run_id ?? undefined,
      pin_publication_state: pin.pin_publication_state,
      has_frames: frames.length > 0,
      frames,
    }],
    clearContinuation: false,
    clearAttempt: true,
    reloadFrames: false,
  }
}

export function generationPinIdentity(value) {
  const state = value?.pin_publication_state ?? 'unavailable'
  const genId = value?.gen_id
  const runId = value?.generation_run_id
  if (
    state !== 'published'
    || !Number.isInteger(genId) || genId < 0
    || !/^[0-9a-f]{32}$/.test(runId || '')
  ) {
    return { gen_id: null, generation_run_id: null, pin_publication_state: state }
  }
  return { gen_id: genId, generation_run_id: runId, pin_publication_state: 'published' }
}

export function frameArchivePinIdentity(archive) {
  return generationPinIdentity({
    pin_publication_state: archive?.pin_publication_state,
    gen_id: archive?.pin_gen_id,
    generation_run_id: archive?.pin_generation_run_id,
  })
}
