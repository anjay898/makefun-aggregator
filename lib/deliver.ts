// Delivering one successful upstream answer to the client, on whichever wire the client speaks.
//
// Same-wire answers pass through untouched: the provider's JSON body, or its SSE stream piped
// frame by frame, exactly as it arrived — the provider echoes what it parsed, and the client
// compares that echo with what it sent, so every byte we leave alone is a check we cannot fail.
// Cross-wire answers are translated: anthropic and gemini upstream answers are first normalized
// to the OpenAI shape (lib/wire2.ts), then rendered onto the client's wire (lib/wire.ts for the
// Anthropic event sequence); openai upstream answers for an Anthropic client go through the same
// event translator directly. Streaming failover is impossible by construction here: this runs
// only after the upstream has answered 200, and once the first byte is written the answer is
// committed — failover lives in main.ts's serve loop, before delivery begins.

import type { ServerResponse } from 'node:http';
import { isRecord, type Json } from './providers.ts';
import { anthropicEventsFromOpenAI, openAIToAnthropic, sseEvent } from './wire.ts';
import {
  anthropicToOpenAIResponse,
  geminiToOpenAIResponse,
  openAIChunksFromAnthropic,
  openAIChunksFromGemini,
} from './wire2.ts';
import type { Candidate } from './candidates.ts';

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function beginStream(res: ServerResponse): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
}

async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** An OpenAI-chunk generator as an SSE ReadableStream, for feeding the Anthropic event translator. */
function streamFromChunks(gen: AsyncGenerator<Json>): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { done, value } = await gen.next();
      if (!done) {
        controller.enqueue(enc.encode(sseEvent(null, value)));
      } else {
        controller.enqueue(enc.encode('data: [DONE]\n\n'));
        controller.close();
      }
    },
  });
}

/** Deliver one successful upstream answer to the client, translating where the wires differ. */
export async function deliver(
  upstream: Response,
  cand: Candidate,
  clientWire: 'openai' | 'anthropic',
  stream: boolean,
  requested: string,
  res: ServerResponse,
): Promise<void> {
  const wire = cand.probe.provider.wire;
  if (clientWire === wire) {
    if (upstream.body !== null && stream) {
      beginStream(res);
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if ('gemini' === wire) {
    // Normalize to the OpenAI shape first, then to the client's wire if it is Anthropic's.
    if (upstream.body !== null && stream) {
      const chunks = openAIChunksFromGemini(upstream.body);
      beginStream(res);
      if ('openai' === clientWire) {
        for await (const chunk of chunks) res.write(sseEvent(null, chunk));
        res.write('data: [DONE]\n\n');
        res.end();
      } else {
        for await (const event of anthropicEventsFromOpenAI(streamFromChunks(chunks), requested)) res.write(event);
        res.end();
      }
      return;
    }
    const answer: unknown = await upstream.json();
    const openai = geminiToOpenAIResponse(isRecord(answer) ? answer : {}, cand.model.id);
    if ('openai' === clientWire) sendJson(res, 200, openai);
    else sendJson(res, 200, openAIToAnthropic(openai, requested));
    return;
  }
  if ('anthropic' === wire) {
    // The client is on the OpenAI wire.
    if (upstream.body !== null && stream) {
      beginStream(res);
      for await (const chunk of openAIChunksFromAnthropic(upstream.body)) res.write(sseEvent(null, chunk));
      res.write('data: [DONE]\n\n');
      res.end();
      return;
    }
    const answer: unknown = await upstream.json();
    sendJson(res, 200, anthropicToOpenAIResponse(isRecord(answer) ? answer : {}, requested));
    return;
  }
  // wire === 'openai', the client is on the Anthropic wire.
  if (upstream.body !== null && stream) {
    beginStream(res);
    for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
}
