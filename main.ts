// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// Built from the kit's starter. The routing, failover and cross-wire serving are the entry's own:
//   - candidates: every free model (per the sealed quality table) meeting the request's requirements,
//     ranked same-wire first (a request passed through in its own wire keeps every field the provider
//     echoes), then by rating_milli; a JSON-mode request prefers the openai wire, the one wire that
//     carries response_format; a provider that has failed twice this trial is demoted below equals.
//   - failover: a 402, 429, 5xx or a network error moves the request to the next candidate; the
//     failure is counted per provider and remembered for the rest of the trial.
//   - wires: openai, anthropic and gemini upstreams all serve both client wires. Same-wire traffic
//     passes through untouched (only the model id is swapped); cross-wire traffic is translated by
//     lib/wire.ts and lib/wire2.ts, in JSON and in SSE.
// Everything is a pure function of the brief, the requests and the providers' answers: no clock and
// no randomness, so a replay makes the same choices.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicToOpenAIResponse,
  geminiToOpenAIResponse,
  openAIChunksFromAnthropic,
  openAIChunksFromGemini,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
} from './lib/wire2.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;

/** Provider calls made this trial, and how many of them failed, per provider id. */
const providerCalls = new Map<string, number>();
const providerFails = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free model meeting the request, ranked for this client wire (see the header comment). */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const jsonReq = requires.json === true;
  out.sort((a, b) => {
    const aw = a.probe.provider.wire === clientWire ? 0 : 1;
    const bw = b.probe.provider.wire === clientWire ? 0 : 1;
    if (aw !== bw) return aw - bw;
    if (jsonReq) {
      const aj = a.probe.provider.wire === 'openai' ? 0 : 1;
      const bj = b.probe.provider.wire === 'openai' ? 0 : 1;
      if (aj !== bj) return aj - bj;
    }
    const af = (providerFails.get(a.probe.provider.id) ?? 0) >= 2 ? 1 : 0;
    const bf = (providerFails.get(b.probe.provider.id) ?? 0) >= 2 ? 1 : 0;
    if (af !== bf) return af - bf;
    if (b.rating !== a.rating) return b.rating - a.rating;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One upstream call on the candidate's own wire, the request translated from the client's wire. */
async function callUpstream(cand: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = cand.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const up = clientWire === 'openai' ? { ...body, model: cand.model.id } : anthropicToOpenAI(body, cand.model.id);
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(up) });
  }
  if (provider.wire === 'anthropic') {
    const up = clientWire === 'anthropic' ? { ...body, model: cand.model.id } : openAIToAnthropicRequest(body, cand.model.id);
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(up) });
  }
  const openaiBody = clientWire === 'openai' ? body : anthropicToOpenAI(body, cand.model.id);
  const up = openAIToGeminiRequest(openaiBody, cand.model.id);
  const method = stream ? 'streamGenerateContent' : 'generateContent';
  return fetch(`${base}/v1beta/models/${cand.model.id}:${method}${stream ? '?alt=sse' : ''}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(up),
  });
}

/** An OpenAI-chunk generator as an SSE ReadableStream, for feeding the Anthropic event translator. */
function streamFromChunks(gen: AsyncGenerator<Json>): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const { value, done } = await gen.next();
      if (done) {
        controller.enqueue(enc.encode('data: [DONE]\n\n'));
        controller.close();
      } else {
        controller.enqueue(enc.encode(sseEvent(null, value)));
      }
    },
  });
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

/** Deliver one successful upstream answer to the client, translating where the wires differ. */
async function deliver(
  upstream: Response,
  cand: Candidate,
  clientWire: 'openai' | 'anthropic',
  stream: boolean,
  requested: string,
  res: ServerResponse,
): Promise<void> {
  const wire = cand.probe.provider.wire;
  if (wire === clientWire) {
    if (stream && upstream.body !== null) {
      beginStream(res);
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  if (wire === 'gemini') {
    // Normalize to the OpenAI shape first, then to the client's wire if it is Anthropic's.
    if (stream && upstream.body !== null) {
      const chunks = openAIChunksFromGemini(upstream.body);
      beginStream(res);
      if (clientWire === 'openai') {
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
    if (clientWire === 'openai') sendJson(res, 200, openai);
    else sendJson(res, 200, openAIToAnthropic(openai, requested));
    return;
  }
  if (wire === 'anthropic') {
    // The client is on the OpenAI wire.
    if (stream && upstream.body !== null) {
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
  if (stream && upstream.body !== null) {
    beginStream(res);
    for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
    res.end();
    return;
  }
  const answer: unknown = await upstream.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
}

/** Serve one client request on either wire, with failover across the ranked candidates. */
async function serve(clientWire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = requirementsFor(body);
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  const candidates = candidatesFor(requires, clientWire);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 502;
  let lastText = '';
  for (const cand of candidates) {
    const pid = cand.probe.provider.id;
    providerCalls.set(pid, (providerCalls.get(pid) ?? 0) + 1);
    let upstream: Response;
    try {
      upstream = await callUpstream(cand, clientWire, body);
    } catch (err) {
      providerFails.set(pid, (providerFails.get(pid) ?? 0) + 1);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (!upstream.ok) {
      providerFails.set(pid, (providerFails.get(pid) ?? 0) + 1);
      lastStatus = upstream.status;
      lastText = (await upstream.text().catch(() => '')).slice(0, 300);
      continue;
    }
    await deliver(upstream, cand, clientWire, body.stream === true, requested, res);
    return;
  }
  const status = lastStatus === 402 || lastStatus === 429 || lastStatus >= 500 ? lastStatus : 502;
  sendError(res, clientWire, status, clientWire === 'anthropic' ? 'api_error' : 'upstream_error', `every provider refused this request (last HTTP ${lastStatus}: ${lastText})`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
