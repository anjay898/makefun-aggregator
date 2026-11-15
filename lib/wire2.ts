import { isRecord, type Json } from './providers.ts';

const CREATED = 1767225600;

function records(value: unknown): Json[] {
  return Array.isArray(value) ? value.filter(isRecord) : ([] as Json[]);
}

function textOf(value: unknown): string {
  if ('string' === typeof value) return value;
  if (!Array.isArray(value)) return '';
  return value
    .filter(isRecord)
    .map((part) => ('string' === typeof part.text ? part.text : ''))
    .join('');
}

function parsed(value: unknown, fallback: unknown = {}): unknown {
  if ('string' !== typeof value) return value ?? fallback;
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
  if ('image_url' !== part.type) return undefined;
  const image = part.image_url;
  const url = 'string' === typeof image
    ? image
    : isRecord(image) && 'string' === typeof image.url
      ? image.url
      : '';
  const match = /^data:([^;,]+);base64,([\s\S]*)$/i.exec(url);
  if (match === null) return undefined;
  return { mime_type: match[1]!, data: match[2]! };
}

function stopSequences(value: unknown): string[] | undefined {
  if ('string' === typeof value) return [value];
  if (Array.isArray(value)) {
    return value.filter((item): item is string => 'string' === typeof item);
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
  return 'number' === typeof value && Number.isFinite(value) ? value : 0;
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
    total_tokens: 'number' === typeof value.totalTokenCount
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
  if ('string' === typeof value) return [{ type: 'text', text: value }];
  const content: Json[] = [];
  for (const part of records(value)) {
    if ('text' === part.type && 'string' === typeof part.text) {
      content.push({ type: 'text', text: part.text });
    } else {
      const image = dataImage(part);
      if (image !== undefined) {
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
    if ('tool' !== role) pendingToolResults = undefined;

    if ('system' === role || 'developer' === role) {
      system.push(textOf(message.content));
      continue;
    }

    if ('tool' === role) {
      const result: Json = {
        type: 'tool_result',
        tool_use_id: message.tool_call_id,
        content: textOf(message.content),
      };
      if (pendingToolResults !== undefined) {
        pendingToolResults.push(result);
      } else {
        pendingToolResults = [result];
        messages.push({ role: 'user', content: pendingToolResults });
      }
      continue;
    }

    if ('user' !== role && 'assistant' !== role) continue;
    const content = anthropicContent(message.content);

    if ('assistant' === role) {
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
      if (undefined !== definition.description) {
        converted.description = definition.description;
      }
      return [converted];
    });
  }

  if ('auto' === body.tool_choice) {
    result.tool_choice = { type: 'auto' };
  } else if ('required' === body.tool_choice) {
    result.tool_choice = { type: 'any' };
  } else if ('none' === body.tool_choice) {
    result.tool_choice = { type: 'none' };
  } else if (
    isRecord(body.tool_choice) &&
    isRecord(body.tool_choice.function) &&
    'string' === typeof body.tool_choice.function.name
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
    if ('text' === block.type && 'string' === typeof block.text) {
      texts.push(block.text);
    } else if ('tool_use' === block.type) {
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
    if ('' === value) {
      const frame = data.length ? { event, data: data.join('\n') } : undefined;
      event = '';
      data = [];
      return frame;
    }
    if (value.startsWith(':')) return undefined;

    const colon = value.indexOf(':');
    const field = 0 > colon ? value : value.slice(0, colon);
    let content = 0 > colon ? '' : value.slice(colon + 1);
    if (content.startsWith(' ')) content = content.slice(1);

    if ('event' === field) event = content;
    if ('data' === field) data.push(content);
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
        if ('\n' !== character && '\r' !== character) continue;
        if ('\r' === character && index + 1 === buffer.length && !ended) break;

        const frame = line(buffer.slice(offset, index));
        if ('\r' === character && '\n' === buffer[index + 1]) index++;
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
  if (!frame.data.trim() || '[DONE]' === frame.data.trim()) return undefined;
  const value: unknown = JSON.parse(frame.data);
  if (!isRecord(value)) throw new Error('Expected an SSE JSON object');
  return value;
}

function upstreamError(value: Json): Error {
  const detail = isRecord(value.error) ? value.error : value;
  const message = 'string' === typeof detail.message
    ? detail.message
    : 'string' === typeof value.error
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

    if ('message_start' === type) {
      const message = isRecord(value.message) ? value.message : {};
      if ('string' === typeof message.id) id = message.id;
      if ('string' === typeof message.model) model = message.model;
      if (isRecord(message.usage)) usage = { ...usage, ...message.usage };
      yield chunk(id, model, { role: 'assistant', content: '' });
    } else if ('content_block_start' === type) {
      const block = isRecord(value.content_block) ? value.content_block : {};
      const blockIndex = 'number' === typeof value.index ? value.index : 0;
      if ('text' === block.type && 'string' === typeof block.text && block.text) {
        yield chunk(id, model, { content: block.text });
      } else if ('tool_use' === block.type) {
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
    } else if ('content_block_delta' === type) {
      const delta = isRecord(value.delta) ? value.delta : {};
      if ('text_delta' === delta.type && 'string' === typeof delta.text) {
        yield chunk(id, model, { content: delta.text });
      } else if (
        'input_json_delta' === delta.type &&
        'string' === typeof delta.partial_json
      ) {
        const state = tools.get('number' === typeof value.index ? value.index : 0);
        if (state && !state.closed) {
          if (delta.partial_json.length) state.receivedArguments = true;
          yield toolArguments(state, delta.partial_json);
        }
      }
    } else if ('content_block_stop' === type) {
      const state = tools.get('number' === typeof value.index ? value.index : 0);
      if (state !== undefined) {
        const closing = closeTool(state);
        if (closing) yield closing;
      }
    } else if ('message_delta' === type) {
      const delta = isRecord(value.delta) ? value.delta : {};
      if (delta.stop_reason != null) {
        finishReason = anthropicFinish(delta.stop_reason);
      }
      if (isRecord(value.usage)) {
        usage = { ...usage, ...value.usage };
        usageProvided = true;
      }
    } else if ('message_stop' === type) {
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
      .filter(([key]) => '$schema' !== key)
      .map(([key, child]) => [key, stripSchema(child)]),
  );
}

function geminiParts(value: unknown): Json[] {
  if ('string' === typeof value) return [{ text: value }];
  const parts: Json[] = [];
  for (const part of records(value)) {
    if ('text' === part.type && 'string' === typeof part.text) {
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
    if ('system' === message.role || 'developer' === message.role) {
      system.push(textOf(message.content));
      continue;
    }

    if ('tool' === message.role) {
      const callId = 'string' === typeof message.tool_call_id
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

    if ('user' !== message.role && 'assistant' !== message.role) continue;
    const parts = geminiParts(message.content);
    if ('assistant' === message.role) {
      for (const call of records(message.tool_calls)) {
        if (!isRecord(call.function)) continue;
        const definition = call.function;
        if ('string' === typeof call.id && 'string' === typeof definition.name) {
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
      role: 'assistant' === message.role ? 'model' : 'user',
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
  if (undefined !== maxTokens && null !== maxTokens) {
    config.maxOutputTokens = maxTokens;
  }
  if (undefined !== body.temperature) config.temperature = body.temperature;
  if (undefined !== body.top_p) config.topP = body.top_p;
  const stop = stopSequences(body.stop);
  if (undefined !== stop) config.stopSequences = stop;
  if (
    isRecord(body.response_format) &&
    ('json_object' === body.response_format.type ||
      'json_schema' === body.response_format.type)
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
    if ('string' === typeof part.text) texts.push(part.text);
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
    if ('[DONE]' === frame.data.trim()) break;
    const value = frameObject(frame);
    if (undefined === value) continue;
    if ('error' === frame.event || value.error !== undefined) {
      throw upstreamError(value);
    }
    if ('string' === typeof value.modelVersion) model = value.modelVersion;
    if (isRecord(value.usageMetadata)) {
      usage = { ...usage, ...value.usageMetadata };
    }

    const incoming = records(value.candidates);
    for (let position = 0; position < incoming.length; position++) {
      const candidate = incoming[position]!;
      const index = 'number' === typeof candidate.index ? candidate.index : position;
      let state = candidates.get(index);
      if (undefined === state) {
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
        .map((part) => 'string' === typeof part.text ? part.text : '')
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

      if (null != candidate.finishReason) {
        state.finishReason = geminiFinish(candidate.finishReason);
      }
    }
  }

  const finished = [...candidates.entries()]
    .filter(([, state]) => undefined !== state.finishReason)
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
