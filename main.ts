// free-model-aggregator-01 — one server, two client wires, a directory of free providers.
//
// Built from the kit's starter; the routing, the failover and the cross-wire serving are the entry's own:
//   - candidates: every free model (per the sealed quality table) that meets the request's requirements,
//     ranked same-wire first (a request passed through in its own wire keeps every field the provider
//     echoes), then by rating_milli; a JSON-mode request prefers the openai wire, the one wire able to
//     carry response_format; a provider that has failed twice this trial is demoted below its equals.
//   - failover: a 402, a 429, a 5xx or a network error moves the request to the next candidate; the
//     failure is counted per provider and remembered for the rest of the trial.
//   - wires: openai, anthropic and gemini upstreams all serve both client wires. Same-wire traffic
//     passes through untouched (only the model id is swapped); cross-wire traffic is translated by
//     lib/wire.ts and lib/wire2.ts, in JSON and in SSE alike.
// Everything is a pure function of the brief, the requests and the providers' answers: no clock and
// no randomness anywhere, so a replay makes the same choices.
// The entrypoint stays thin on purpose: ranking lives in lib/candidates.ts, the upstream call in
// lib/upstream.ts, answer shaping in lib/deliver.ts and the manifest cursor in
// lib/requirements.ts, so each policy can be read on its own.

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
import { candidatesFor, type Candidate } from './lib/candidates.ts';
import { deliver } from './lib/deliver.ts';
import { callUpstream } from './lib/upstream.ts';
import { ManifestCursor } from './lib/requirements.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
const manifestCursor = new ManifestCursor(brief.workload);

/** Provider calls made this trial, and how many of them failed, per provider id. */
const providerCalls = new Map<string, number>();
const providerFails = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (null !== brief.error) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && null !== probe.error) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (false === probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (false === modelFree(model, row)) continue;
      out.push({ display_name: model.display_name, provider: probe.provider.id, id: model.id });
    }
  }
  return out;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (MAX_BODY_BYTES < size) return null;
    chunks.push(buffer);
  }
  if (0 === chunks.length) return null;
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
  if ('anthropic' === wire) sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if ('string' === typeof req.headers['x-api-key']) sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Serve one client request on either wire, with failover across the ranked candidates. */
async function serve(clientWire: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (null === body) {
    return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const requires = manifestCursor.next(body);
  const requested = 'string' === typeof body.model && '' !== body.model ? body.model : 'auto';
  const candidates = candidatesFor(brief, probes, providerFails, requires, clientWire);
  if (0 === candidates.length) {
    return sendError(res, clientWire, 503, 'anthropic' === clientWire ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 502;
  let lastText = '';
  for (const cand of candidates) {
    const pid = cand.probe.provider.id;
    providerCalls.set(pid, (providerCalls.get(pid) ?? 0) + 1);
    let upstream: Response;
    try {
      upstream = await callUpstream(ENV_URL, cand, clientWire, body);
    } catch (err) {
      providerFails.set(pid, (providerFails.get(pid) ?? 0) + 1);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (false === upstream.ok) {
      providerFails.set(pid, (providerFails.get(pid) ?? 0) + 1);
      lastStatus = upstream.status;
      lastText = (await upstream.text().catch(() => '')).slice(0, 300);
      continue;
    }
    await deliver(upstream, cand, clientWire, body.stream === true, requested, res);
    return;
  }
  const status = 402 === lastStatus || 429 === lastStatus || lastStatus >= 500 ? lastStatus : 502;
  sendError(res, clientWire, status, 'anthropic' === clientWire ? 'api_error' : 'upstream_error', `every provider refused this request (last HTTP ${lastStatus}: ${lastText})`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if ('GET' === req.method && '/v1/models' === url.pathname) {
    models(req, res);
    return;
  }
  if ('POST' === req.method && '/v1/chat/completions' === url.pathname) {
    await serve('openai', req, res);
    return;
  }
  if ('POST' === req.method && '/v1/messages' === url.pathname) {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, '/v1/messages' === url.pathname ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
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
  if (isRecord(reply) && 'number' === typeof reply.requests) log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (true === inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (200 !== res.status) return;
        const work: unknown = await res.json();
        if (isRecord(work) && true === work.done) {
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
  if ('' === ENV_URL) throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (null === address || 'string' === typeof address) throw new Error('the server has no port');
  const port = address.port;
  if (8080 === port) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
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
