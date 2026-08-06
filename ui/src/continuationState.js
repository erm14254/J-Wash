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
    return {
      messages: messages.map((message) => message.id === frame.message_id ? {
        ...message,
        content: frame.text,
        stats: frame.stats,
        gen_id: frame.gen_id,
        generation_run_id: frame.generation_run_id,
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
