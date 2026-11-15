// One upstream call on the candidate's own wire, the client's request translated to fit it.
//
// The brief's rule governs every branch: pass the request through in the provider's wire
// untouched — its system prompt, its user text (images included), its tools, its stop sequences,
// its max_tokens, its JSON mode — because the provider echoes what it parsed and the client
// compares that echo with what it sent. So a same-wire call forwards the client's body with
// only the model id swapped for the routed model's; a cross-wire call is rebuilt field by field
// by the translators (lib/wire.ts from Anthropic's shape, lib/wire2.ts to Anthropic's and to
// Gemini's), never by string surgery. Two deliberate exceptions, both invisible to the echo:
// the accept header asks for an event stream when the client asked for one, and a streamed
// openai-wire call asks for the closing usage chunk when the client did not — a streamed answer
// without final usage fails the wire checks, and stream_options is not an echoed field.
// The auth header follows the provider's directory row, not its wire: the wire does not decide
// the header, the brief does (Bearer, x-api-key, x-goog-api-key, or none).

import { authHeaders, isRecord, joinBase, type Json } from './providers.ts';
import { anthropicToOpenAI } from './wire.ts';
import { openAIToAnthropicRequest, openAIToGeminiRequest } from './wire2.ts';
import type { Candidate } from './candidates.ts';

/** One upstream call on the candidate's own wire, the request translated from the client's wire. */
export async function callUpstream(
  envUrl: string,
  cand: Candidate,
  clientWire: 'openai' | 'anthropic',
  body: Json,
): Promise<Response> {
  const provider = cand.probe.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = true === body.stream;
  if (stream) headers.accept = 'text/event-stream';
  if ('openai' === provider.wire) {
    const up = 'openai' === clientWire ? { ...body, model: cand.model.id } : anthropicToOpenAI(body, cand.model.id);
    // A streamed answer must end with a usage chunk; ask for it when the client did not.
    if (stream && !isRecord(up.stream_options)) up.stream_options = { include_usage: true };
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(up) });
  }
  if ('anthropic' === provider.wire) {
    const up = 'anthropic' === clientWire ? { ...body, model: cand.model.id } : openAIToAnthropicRequest(body, cand.model.id);
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(up) });
  }
  const openaiBody = 'openai' === clientWire ? body : anthropicToOpenAI(body, cand.model.id);
  const up = openAIToGeminiRequest(openaiBody, cand.model.id);
  const method = stream ? 'streamGenerateContent' : 'generateContent';
  return fetch(`${base}/v1beta/models/${cand.model.id}:${method}${stream ? '?alt=sse' : ''}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(up),
  });
}
