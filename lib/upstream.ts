// Upstream calls to anthropic-wire and gemini-wire providers.
//
// The entry normalises every client request to OpenAI Chat Completions shape (the Anthropic client
// is rewritten by lib/wire.ts's anthropicToOpenAI). These helpers take that OpenAI-shaped request,
// speak the provider's own wire, and return the provider's answer back in OpenAI shape, so the
// caller only ever deals with one wire. Streaming is translated event by event; a provider that
// answers a non-2xx is returned as that status so the caller can fail over.

import { isRecord, type Json } from './providers.ts';

/** An OpenAI-shaped request body, as the entry normalises it. */
export type OpenAIRequest = Json;

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** An OpenAI message as an Anthropic message (system is handled by the caller). */
function openAIMessageToAnthropic(m: Json): Json | null {
  const role = m.role === 'assistant' ? 'assistant' : 'user';
  const content = m.content;
  if (typeof content === 'string') return { role, content };
  if (!Array.isArray(content)) return null;
  const blocks: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      if (url.startsWith('data:')) {
        const [head, data] = url.slice(5).split(',', [2] as never) as unknown as [string, string];
        const media = head.split(';')[0] || 'image/png';
        blocks.push({ type: 'image', source: { type: 'base64', media_type: media, data: data ?? '' } });
      } else blocks.push({ type: 'image', source: { type: 'url', url } });
    }
  }
  if (m.role === 'tool' && typeof m.tool_call_id === 'string') {
    return { role: 'user', content: [{ type: 'tool_result', tool_use_id: m.tool_call_id, content: textOf(content) }] };
  }
  if (m.role === 'assistant' && Array.isArray(m.tool_calls)) {
    for (const call of m.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: Json = {};
      try {
        const parsed: unknown = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Empty input.
      }
      blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
    }
    if (blocks.length === 0 && textOf(content) !== '') blocks.push({ type: 'text', text: textOf(content) });
    return { role, content: blocks };
  }
  if (blocks.length === 0) return null;
  return { role, content: blocks };
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    const m = openAIMessageToAnthropic(raw);
    if (m !== null) messages.push(m);
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

/** An Anthropic Messages object as an OpenAI Chat Completions object (non-streaming). */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({ id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) } });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const stop = typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn';
  const finish = stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : 'stop';
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-anthropic',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) }, finish_reason: finish }],
    usage: { prompt_tokens: typeof usage.input_tokens === 'number' ? usage.input_tokens : 0, completion_tokens: typeof usage.output_tokens === 'number' ? usage.output_tokens : 0, total_tokens: (typeof usage.input_tokens === 'number' ? usage.input_tokens : 0) + (typeof usage.output_tokens === 'number' ? usage.output_tokens : 0) },
  };
}

/** An OpenAI Chat Completions request as a Gemini generateContent request naming `model`. The system
 *  prompt goes in systemInstruction (the provider echoes it from there); a system message in
 *  contents would echo as user text, not as the system prompt. */
export function openAIToGeminiRequest(body: Json, model: string): { urlModel: string; payload: Json } {
  const contents: Json[] = [];
  let systemText = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' || raw.role === 'model' ? 'model' : 'user';
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemText = systemText === '' ? t : `${systemText}\n${t}`;
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') parts.push({ text: content });
    else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          if (url.startsWith('data:')) {
            const [head, data] = url.slice(5).split(',', [2] as never) as unknown as [string, string];
            parts.push({ inlineData: { mimeType: head.split(';')[0] || 'image/png', data: data ?? '' } });
          }
        }
      }
    }
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        try {
          const parsed: unknown = JSON.parse(typeof call.function.arguments === 'string' ? call.function.arguments : '{}');
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Empty args.
        }
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) config.responseMimeType = 'application/json';
  const payload: Json = { contents };
  if (systemText !== '') payload.systemInstruction = { parts: [{ text: systemText }] };
  if (Object.keys(config).length > 0) payload.generationConfig = config;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    const decls = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (decls.length > 0) payload.tools = [{ functionDeclarations: decls }];
  }
  return { urlModel: model, payload };
}

/** A Gemini generateContent response as an OpenAI Chat Completions object (non-streaming). */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  const toolCalls: Json[] = [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    if (isRecord(part.functionCall)) {
      toolCalls.push({ id: `call_${toolCalls.length}`, type: 'function', function: { name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '', arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) } });
    }
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const finish = typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP';
  const mapped = finish === 'MAX_TOKENS' ? 'length' : toolCalls.length > 0 ? 'tool_calls' : finish === 'SAFETY' ? 'content_filter' : 'stop';
  const prompt = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const completion = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) }, finish_reason: mapped }],
    usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
  };
}

/** One OpenAI SSE chunk (a `data: {...}` frame) from a Gemini generateContent chunk. */
export function geminiChunkToOpenAIChunk(chunk: Json, model: string): string | null {
  const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
  const cand = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const usage = isRecord(chunk.usageMetadata) ? chunk.usageMetadata : {};
  const finish = typeof cand.finishReason === 'string' ? cand.finishReason : null;
  const mapped = finish === 'MAX_TOKENS' ? 'length' : finish === 'SAFETY' ? 'content_filter' : finish === 'STOP' ? 'stop' : null;
  const data: Json = {
    id: 'chatcmpl-gemini',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta: text !== '' ? { content: text } : {}, finish_reason: mapped }],
  };
  if (Object.keys(usage).length > 0) {
    const prompt = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
    const completion = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
    data.usage = { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion };
  }
  return `data: ${JSON.stringify(data)}\n\n`;
}

/** An Anthropic SSE stream as OpenAI SSE chunks: text deltas become content chunks, the message
 *  delta's stop reason and usage become the final chunk, then [DONE]. */
export async function* anthropicStreamToOpenAIStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  let done = false;
  const chunk = (delta: Json, finish: string | null, usage: Json | null): string => {
    const data: Json = { id: 'chatcmpl-anthropic', object: 'chat.completion.chunk', created: 1767225600, model, choices: [{ index: 0, delta, finish_reason: finish }] };
    if (usage !== null) data.usage = usage;
    return `data: ${JSON.stringify(data)}\n\n`;
  };
  const flush = function* (): Generator<string> {
    if (done) return;
    done = true;
    yield chunk({}, pendingStop ?? 'stop', { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens });
    yield 'data: [DONE]\n\n';
  };
  try {
    for (;;) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '') continue;
          let event: unknown;
          try {
            event = JSON.parse(payload);
          } catch {
            continue;
          }
          if (!isRecord(event)) continue;
          const type = event.type;
          if (type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
            inputTokens = typeof event.message.usage.input_tokens === 'number' ? event.message.usage.input_tokens : 0;
          } else if (type === 'content_block_delta' && isRecord(event.delta)) {
            if (event.delta.type === 'text_delta' && typeof event.delta.text === 'string' && event.delta.text !== '') {
              yield chunk({ content: event.delta.text }, null, null);
            }
          } else if (type === 'message_delta') {
            if (isRecord(event.delta) && typeof event.delta.stop_reason === 'string') {
              const stop = event.delta.stop_reason;
              pendingStop = stop === 'max_tokens' ? 'length' : stop === 'tool_use' ? 'tool_calls' : stop === 'refusal' ? 'content_filter' : 'stop';
            }
            if (isRecord(event.usage) && typeof event.usage.output_tokens === 'number') outputTokens = event.usage.output_tokens;
          } else if (type === 'message_stop') {
            for (const e of flush()) yield e;
          }
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      // A trailing partial frame: ignore it; the finish is emitted below.
    }
  } finally {
    reader.releaseLock();
  }
  for (const e of flush()) yield e;
}

/** A Gemini SSE stream as OpenAI SSE chunks, then [DONE]. */
export async function* geminiStreamToOpenAIStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let done = false;
  try {
    for (;;) {
      const { value, done: streamDone } = await reader.read();
      if (streamDone) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const line of frame.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '' || payload === '[DONE]') continue;
          let event: unknown;
          try {
            event = JSON.parse(payload);
          } catch {
            continue;
          }
          if (!isRecord(event)) continue;
          const out = geminiChunkToOpenAIChunk(event, model);
          if (out !== null) yield out;
        }
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  if (!done) {
    done = true;
    yield 'data: [DONE]\n\n';
  }
}
