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
    return {
      messages: messages.map((message) => message.id === frame.message_id ? {
        ...message,
        content: durableContent,
        stats: frame.stats,
        gen_id: frame.pin_publication_state === 'published' ? frame.gen_id : undefined,
        generation_run_id: frame.pin_publication_state === 'published' ? frame.generation_run_id : undefined,
        pin_publication_state: frame.pin_publication_state,
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
  return {
    messages: [...messages, {
      id: frame.message_id ?? null,
      role: 'assistant',
      content: frame.text,
      meta: frame.meta,
      stats: frame.stats,
      gen_id: frame.gen_id,
      generation_run_id: frame.generation_run_id,
      has_frames: frames.length > 0,
      frames,
    }],
    clearContinuation: false,
    clearAttempt: true,
    reloadFrames: false,
  }
}

export function frameArchivePinIdentity(archive) {
  if (archive?.pin_publication_state !== 'published') {
    return { gen_id: null, generation_run_id: null, pin_publication_state: archive?.pin_publication_state ?? 'unavailable' }
  }
  const genId = archive.pin_gen_id
  const runId = archive.pin_generation_run_id
  if (!Number.isInteger(genId) || genId < 0 || !/^[0-9a-f]{32}$/.test(runId || '')) {
    return { gen_id: null, generation_run_id: null, pin_publication_state: 'unavailable' }
  }
  return { gen_id: genId, generation_run_id: runId, pin_publication_state: 'published' }
}
