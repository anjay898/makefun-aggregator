import { isRecord, type Json } from './providers.ts';

const CREATED = 1767225600;

function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isRecord) : [];
}

function textOf(value: unknown): string {
  if (typeof value === 'string') return value;
  if (!Array.isArray(value)) return '';
  return value
    .filter(isRecord)
    .map((part) => (typeof part.text === 'string' ? part.text : ''))
    .join('');
}

function parsed(value: unknown, fallback: unknown = {}): unknown {
  if (typeof value !== 'string') return value ?? fallback;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return fallback;
  }
}

function argumentsOf(value: unknown): string {
  return JSON.stringify(value ?? {}) ?? '{}';
}

function dataImage(part: Json): { mime_type: string; data: string } | undefined {
  if (part.type !== 'image_url') return undefined;
  const image = part.image_url;
  const url = typeof image === 'string'
    ? image
    : isRecord(image) && typeof image.url === 'string'
      ? image.url
      : '';
  const match = /^data:([^;,]+);base64,([\s\S]*)$/i.exec(url);
  if (!match) return undefined;
  return { mime_type: match[1]!, data: match[2]! };
}

function stopSequences(value: unknown): string[] | undefined {
  if (typeof value === 'string') return [value];
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === 'string');
  }
  return undefined;
}

function anthropicFinish(value: unknown): string {
  switch (value) {
    case 'max_tokens': return 'length';
    case 'tool_use': return 'tool_calls';
    case 'refusal': return 'content_filter';
    case 'end_turn':
    case 'stop_sequence':
    default: return 'stop';
  }
}

function geminiFinish(value: unknown): string {
  switch (value) {
    case 'MAX_TOKENS': return 'length';
    case 'SAFETY': return 'content_filter';
    default: return 'stop';
  }
}

function count(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function anthropicUsage(value: Json): Json {
  const prompt = count(value.input_tokens);
  const completion = count(value.output_tokens);
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: prompt + completion,
  };
}

function geminiUsage(value: Json): Json {
  const prompt = count(value.promptTokenCount);
  const completion = count(value.candidatesTokenCount);
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: typeof value.totalTokenCount === 'number'
      ? value.totalTokenCount
      : prompt + completion,
  };
}

function completion(
  id: unknown,
  model: string,
  message: Json,
  finishReason: string,
  usage: Json,
): Json {
  return {
    id,
    object: 'chat.completion',
    created: CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReason }],
    usage,
  };
}

function chunk(
  id: string,
  model: string,
  delta: Json,
  finishReason: string | null = null,
  index = 0,
): Json {
  return {
    id,
    object: 'chat.completion.chunk',
    created: CREATED,
    model,
    choices: [{ index, delta, finish_reason: finishReason }],
  };
}

function usageChunk(id: string, model: string, usage: Json): Json {
  return {
    id,
    object: 'chat.completion.chunk',
    created: CREATED,
    model,
    choices: [],
    usage,
  };
}

function anthropicContent(value: unknown): Json[] {
  if (typeof value === 'string') return [{ type: 'text', text: value }];
  const content: Json[] = [];
  for (const part of records(value)) {
    if (part.type === 'text' && typeof part.text === 'string') {
      content.push({ type: 'text', text: part.text });
    } else {
      const image = dataImage(part);
      if (image) {
        content.push({
          type: 'image',
          source: {
            type: 'base64',
            media_type: image.mime_type,
            data: image.data,
          },
        });
      }
    }
  }
  return content;
}

export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const system: string[] = [];
  const messages: Json[] = [];
  let pendingToolResults: Json[] | undefined;

  for (const message of records(body.messages)) {
    const role = message.role;
    if (role !== 'tool') pendingToolResults = undefined;

    if (role === 'system' || role === 'developer') {
      system.push(textOf(message.content));
      continue;
    }

    if (role === 'tool') {
      const result: Json = {
        type: 'tool_result',
        tool_use_id: message.tool_call_id,
        content: textOf(message.content),
      };
      if (pendingToolResults) {
        pendingToolResults.push(result);
      } else {
        pendingToolResults = [result];
        messages.push({ role: 'user', content: pendingToolResults });
      }
      continue;
    }

    if (role !== 'user' && role !== 'assistant') continue;
    const content = anthropicContent(message.content);

    if (role === 'assistant') {
      for (const call of records(message.tool_calls)) {
        if (!isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: call.id,
          name: call.function.name,
          input: parsed(call.function.arguments),
        });
      }
    }

    messages.push({ role, content });
  }

  const result: Json = {
    model,
    messages,
    max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 1024,
  };
  if (system.length) result.system = system.join('\n\n');

  if (Array.isArray(body.tools)) {
    result.tools = records(body.tools).flatMap((tool) => {
      if (!isRecord(tool.function)) return [];
      const definition = tool.function;
      const converted: Json = {
        name: definition.name,
        input_schema: definition.parameters ?? {
          type: 'object',
          properties: {},
        },
      };
      if (definition.description !== undefined) {
        converted.description = definition.description;
      }
      return [converted];
    });
  }

  if (body.tool_choice === 'auto') {
    result.tool_choice = { type: 'auto' };
  } else if (body.tool_choice === 'required') {
    result.tool_choice = { type: 'any' };
  } else if (body.tool_choice === 'none') {
    result.tool_choice = { type: 'none' };
  } else if (
    isRecord(body.tool_choice) &&
    isRecord(body.tool_choice.function) &&
    typeof body.tool_choice.function.name === 'string'
  ) {
    result.tool_choice = {
      type: 'tool',
      name: body.tool_choice.function.name,
    };
  }

  for (const key of ['temperature', 'top_p', 'stream']) {
    if (body[key] !== undefined) result[key] = body[key];
  }
  const stop = stopSequences(body.stop);
  if (stop !== undefined) result.stop_sequences = stop;
  return result;
}

export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const texts: string[] = [];
  const calls: Json[] = [];

  for (const block of records(body.content)) {
    if (block.type === 'text' && typeof block.text === 'string') {
      texts.push(block.text);
    } else if (block.type === 'tool_use') {
      calls.push({
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: argumentsOf(block.input),
        },
      });
    }
  }

  const message: Json = {
    role: 'assistant',
    content: texts.length ? texts.join('') : calls.length ? null : '',
  };
  if (calls.length) message.tool_calls = calls;

  return completion(
    body.id,
    model,
    message,
    anthropicFinish(body.stop_reason),
    anthropicUsage(isRecord(body.usage) ? body.usage : {}),
  );
}

type SSEFrame = { event: string; data: string };

async function* sseFrames(
  upstream: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEFrame> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let event = '';
  let data: string[] = [];
  let ended = false;

  function line(value: string): SSEFrame | undefined {
    if (value === '') {
      const frame = data.length ? { event, data: data.join('\n') } : undefined;
      event = '';
      data = [];
      return frame;
    }
    if (value.startsWith(':')) return undefined;

    const colon = value.indexOf(':');
    const field = colon < 0 ? value : value.slice(0, colon);
    let content = colon < 0 ? '' : value.slice(colon + 1);
    if (content.startsWith(' ')) content = content.slice(1);

    if (field === 'event') event = content;
    if (field === 'data') data.push(content);
    return undefined;
  }

  try {
    while (!ended) {
      const next = await reader.read();
      ended = next.done;
      buffer += ended
        ? decoder.decode()
        : decoder.decode(next.value, { stream: true });

      let offset = 0;
      for (let index = 0; index < buffer.length; index++) {
        const character = buffer[index];
        if (character !== '\n' && character !== '\r') continue;
        if (character === '\r' && index + 1 === buffer.length && !ended) break;

        const frame = line(buffer.slice(offset, index));
        if (character === '\r' && buffer[index + 1] === '\n') index++;
        offset = index + 1;
        if (frame) yield frame;
      }
      buffer = buffer.slice(offset);
    }

    if (buffer.length) {
      const frame = line(buffer);
      if (frame) yield frame;
    }
    const finalFrame = line('');
    if (finalFrame) yield finalFrame;
  } finally {
    if (!ended) {
      try {
        await reader.cancel();
      } catch {
        // Cancellation must not hide a parsing or upstream error.
      }
    }
    reader.releaseLock();
  }
}

function frameObject(frame: SSEFrame): Json | undefined {
  if (!frame.data.trim() || frame.data.trim() === '[DONE]') return undefined;
  const value: unknown = JSON.parse(frame.data);
  if (!isRecord(value)) throw new Error('Expected an SSE JSON object');
  return value;
}

function upstreamError(value: Json): Error {
  const detail = isRecord(value.error) ? value.error : value;
  const message = typeof detail.message === 'string'
    ? detail.message
    : typeof value.error === 'string'
      ? value.error
      : 'Upstream stream error';
  return new Error(message);
}

type AnthropicToolState = {
  index: number;
  initialInput: unknown;
  receivedArguments: boolean;
  closed: boolean;
};

export async function* openAIChunksFromAnthropic(
  upstream: ReadableStream<Uint8Array>,
): AsyncGenerator<Json> {
  let id = '';
  let model = '';
  let finishReason: string | undefined;
  let finished = false;
  let usageProvided = false;
  let usage: Json = {};
  let nextToolIndex = 0;
  const tools = new Map<number, AnthropicToolState>();

  function toolArguments(state: AnthropicToolState, value: string): Json {
    return chunk(id, model, {
      tool_calls: [{ index: state.index, function: { arguments: value } }],
    });
  }

  function closeTool(state: AnthropicToolState): Json | undefined {
    if (state.closed) return undefined;
    state.closed = true;
    if (!state.receivedArguments) {
      return toolArguments(state, argumentsOf(state.initialInput));
    }
    return undefined;
  }

  for await (const frame of sseFrames(upstream)) {
    const value = frameObject(frame);
    if (!value) continue;
    const type = typeof value.type === 'string' ? value.type : frame.event;

    if (type === 'error' || value.error !== undefined) {
      throw upstreamError(value);
    }

    if (type === 'message_start') {
      const message = isRecord(value.message) ? value.message : {};
      if (typeof message.id === 'string') id = message.id;
      if (typeof message.model === 'string') model = message.model;
      if (isRecord(message.usage)) usage = { ...usage, ...message.usage };
      yield chunk(id, model, { role: 'assistant', content: '' });
    } else if (type === 'content_block_start') {
      const block = isRecord(value.content_block) ? value.content_block : {};
      const blockIndex = typeof value.index === 'number' ? value.index : 0;
      if (block.type === 'text' && typeof block.text === 'string' && block.text) {
        yield chunk(id, model, { content: block.text });
      } else if (block.type === 'tool_use') {
        const state: AnthropicToolState = {
          index: nextToolIndex++,
          initialInput: block.input,
          receivedArguments: false,
          closed: false,
        };
        tools.set(blockIndex, state);
        yield chunk(id, model, {
          tool_calls: [{
            index: state.index,
            id: block.id,
            type: 'function',
            function: { name: block.name, arguments: '' },
          }],
        });
      }
    } else if (type === 'content_block_delta') {
      const delta = isRecord(value.delta) ? value.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string') {
        yield chunk(id, model, { content: delta.text });
      } else if (
        delta.type === 'input_json_delta' &&
        typeof delta.partial_json === 'string'
      ) {
        const state = tools.get(typeof value.index === 'number' ? value.index : 0);
        if (state && !state.closed) {
          if (delta.partial_json.length) state.receivedArguments = true;
          yield toolArguments(state, delta.partial_json);
        }
      }
    } else if (type === 'content_block_stop') {
      const state = tools.get(typeof value.index === 'number' ? value.index : 0);
      if (state) {
        const closing = closeTool(state);
        if (closing) yield closing;
      }
    } else if (type === 'message_delta') {
      const delta = isRecord(value.delta) ? value.delta : {};
      if (delta.stop_reason != null) {
        finishReason = anthropicFinish(delta.stop_reason);
      }
      if (isRecord(value.usage)) {
        usage = { ...usage, ...value.usage };
        usageProvided = true;
      }
    } else if (type === 'message_stop') {
      for (const state of tools.values()) {
        const closing = closeTool(state);
        if (closing) yield closing;
      }
      yield chunk(id, model, {}, finishReason ?? 'stop');
      if (usageProvided) yield usageChunk(id, model, anthropicUsage(usage));
      finished = true;
      break;
    }
  }

  if (!finished && finishReason !== undefined) {
    for (const state of tools.values()) {
      const closing = closeTool(state);
      if (closing) yield closing;
    }
    yield chunk(id, model, {}, finishReason);
    if (usageProvided) yield usageChunk(id, model, anthropicUsage(usage));
  }
}

function stripSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripSchema);
  if (!isRecord(value)) return value;
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => key !== '$schema')
      .map(([key, child]) => [key, stripSchema(child)]),
  );
}

function geminiParts(value: unknown): Json[] {
  if (typeof value === 'string') return [{ text: value }];
  const parts: Json[] = [];
  for (const part of records(value)) {
    if (part.type === 'text' && typeof part.text === 'string') {
      parts.push({ text: part.text });
    } else {
      const image = dataImage(part);
      if (image) parts.push({ inlineData: { mimeType: image.mime_type, data: image.data } });
    }
  }
  return parts;
}

export function openAIToGeminiRequest(body: Json, model: string): Json {
  void model;
  const system: string[] = [];
  const contents: Json[] = [];
  const toolNames = new Map<string, string>();

  for (const message of records(body.messages)) {
    if (message.role === 'system' || message.role === 'developer') {
      system.push(textOf(message.content));
      continue;
    }

    if (message.role === 'tool') {
      const callId = typeof message.tool_call_id === 'string'
        ? message.tool_call_id
        : '';
      const text = textOf(message.content);
      contents.push({
        role: 'user',
        parts: [{
          functionResponse: {
            name: toolNames.get(callId) ?? callId,
            response: { result: parsed(text, text) },
          },
        }],
      });
      continue;
    }

    if (message.role !== 'user' && message.role !== 'assistant') continue;
    const parts = geminiParts(message.content);
    if (message.role === 'assistant') {
      for (const call of records(message.tool_calls)) {
        if (!isRecord(call.function)) continue;
        const definition = call.function;
        if (typeof call.id === 'string' && typeof definition.name === 'string') {
          toolNames.set(call.id, definition.name);
        }
        parts.push({
          functionCall: {
            name: definition.name,
            args: parsed(definition.arguments),
          },
        });
      }
    }
    contents.push({
      role: message.role === 'assistant' ? 'model' : 'user',
      parts,
    });
  }

  const result: Json = { contents };
  if (system.length) {
    result.systemInstruction = { parts: [{ text: system.join('\n\n') }] };
  }

  if (Array.isArray(body.tools)) {
    const declarations = records(body.tools).flatMap((tool) => {
      if (!isRecord(tool.function)) return [];
      const definition = tool.function;
      const declaration: Json = { name: definition.name };
      if (definition.description !== undefined) {
        declaration.description = definition.description;
      }
      if (definition.parameters !== undefined) {
        declaration.parameters = stripSchema(definition.parameters);
      }
      return [declaration];
    });
    if (declarations.length) {
      result.tools = [{ functionDeclarations: declarations }];
    }
  }

  const config: Json = {};
  const maxTokens = body.max_tokens ?? body.max_completion_tokens;
  if (maxTokens !== undefined && maxTokens !== null) {
    config.maxOutputTokens = maxTokens;
  }
  if (body.temperature !== undefined) config.temperature = body.temperature;
  if (body.top_p !== undefined) config.topP = body.top_p;
  const stop = stopSequences(body.stop);
  if (stop !== undefined) config.stopSequences = stop;
  if (
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' ||
      body.response_format.type === 'json_schema')
  ) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length) result.generationConfig = config;
  return result;
}

export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const candidate = records(body.candidates)[0] ?? {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const texts: string[] = [];
  const calls: Json[] = [];

  for (const part of records(content.parts)) {
    if (typeof part.text === 'string') texts.push(part.text);
    if (isRecord(part.functionCall)) {
      calls.push({
        id: `call_${calls.length}`,
        type: 'function',
        function: {
          name: part.functionCall.name,
          arguments: argumentsOf(part.functionCall.args),
        },
      });
    }
  }

  const message: Json = {
    role: 'assistant',
    content: texts.length ? texts.join('') : calls.length ? null : '',
  };
  if (calls.length) message.tool_calls = calls;
  return completion(
    'gemini-aggregate',
    model,
    message,
    geminiFinish(candidate.finishReason),
    geminiUsage(isRecord(body.usageMetadata) ? body.usageMetadata : {}),
  );
}

type GeminiCandidateState = {
  text: string;
  calls: Map<number, { name: unknown; arguments: string }>;
  nextCallIndex: number;
  finishReason: string | undefined;
};

export async function* openAIChunksFromGemini(
  upstream: ReadableStream<Uint8Array>,
): AsyncGenerator<Json> {
  const id = 'gemini-aggregate';
  let model = '';
  let usage: Json | undefined;
  const candidates = new Map<number, GeminiCandidateState>();

  for await (const frame of sseFrames(upstream)) {
    if (frame.data.trim() === '[DONE]') break;
    const value = frameObject(frame);
    if (!value) continue;
    if (frame.event === 'error' || value.error !== undefined) {
      throw upstreamError(value);
    }
    if (typeof value.modelVersion === 'string') model = value.modelVersion;
    if (isRecord(value.usageMetadata)) {
      usage = { ...usage, ...value.usageMetadata };
    }

    const incoming = records(value.candidates);
    for (let position = 0; position < incoming.length; position++) {
      const candidate = incoming[position]!;
      const index = typeof candidate.index === 'number' ? candidate.index : position;
      let state = candidates.get(index);
      if (!state) {
        state = {
          text: '',
          calls: new Map(),
          nextCallIndex: 0,
          finishReason: undefined,
        };
        candidates.set(index, state);
        yield chunk(id, model, { role: 'assistant', content: '' }, null, index);
      }

      const content = isRecord(candidate.content) ? candidate.content : {};
      const parts = records(content.parts);
      const text = parts
        .map((part) => typeof part.text === 'string' ? part.text : '')
        .join('');

      if (text) {
        // Accept cumulative snapshots as well as Gemini's incremental text frames.
        const suffix = text.startsWith(state.text)
          ? text.slice(state.text.length)
          : text;
        state.text += suffix;
        if (suffix) yield chunk(id, model, { content: suffix }, null, index);
      }

      for (const part of parts) {
        if (!isRecord(part.functionCall)) continue;
        const callIndex = state.nextCallIndex++;
        const name = part.functionCall.name;
        const args = argumentsOf(part.functionCall.args);
        state.calls.set(callIndex, { name, arguments: args });
        yield chunk(id, model, {
          tool_calls: [{
            index: callIndex,
            id: `call_${callIndex}`,
            type: 'function',
            function: { name, arguments: args },
          }],
        }, null, index);
      }

      if (candidate.finishReason != null) {
        state.finishReason = geminiFinish(candidate.finishReason);
      }
    }
  }

  const finished = [...candidates.entries()]
    .filter(([, state]) => state.finishReason !== undefined)
    .map(([index, state]) => ({
      index,
      delta: {},
      finish_reason: state.finishReason,
    }));

  if (finished.length) {
    const finalChunk: Json = {
      id,
      object: 'chat.completion.chunk',
      created: CREATED,
      model,
      choices: finished,
    };
    if (usage) finalChunk.usage = geminiUsage(usage);
    yield finalChunk;
  } else if (usage) {
    yield usageChunk(id, model, geminiUsage(usage));
  }
}
