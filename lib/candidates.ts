// The candidate ranking for one client request: which free models may serve it, in which order.
//
// A candidate is a probed model that is free (per the sealed quality table) and whose capabilities
// meet the request's requirements. The order is the entry's routing policy:
//   1. same wire as the client first — a request passed through in its own wire keeps every field
//      the provider echoes back (system prompt, user text, tools, stop sequences, max_tokens, JSON
//      mode), and the brief warns that a field rewritten or dropped loses the request;
//   2. for a JSON-mode request, the openai wire next — it is the one wire that carries
//      response_format, so a JSON request sent anywhere else echoes the mode as absent;
//   3. a provider that has already failed twice this trial is demoted below its equals — its
//      budget or its failure schedule is likely spent, and calls are the tie-break currency;
//   4. then by the quality table's rating_milli, highest first, with the model id as the final,
//      deterministic tie-break.
// The ranking is a pure function of the brief, the probes and the failure counts: no clock and no
// randomness, so a replay ranks identically.
// Demotion is deliberately shallow: a twice-failed provider sinks below its equals but never off
// the list, because late in a trial it may be the only candidate left that fits the request.

import {
  capabilitiesOf,
  meetsRequirements,
  modelFree,
  ratingOfModel,
  type Brief,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './providers.ts';

export interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Every free model meeting the request, ranked for this client wire (see the header comment). */
export function candidatesFor(
  brief: Brief,
  probes: readonly Probe[],
  providerFails: ReadonlyMap<string, number>,
  requires: Requirement,
  clientWire: 'openai' | 'anthropic',
): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (false === probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (false === modelFree(model, row)) continue;
      if (false === meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ rating: row?.rating_milli ?? 0, model, probe });
    }
  }
  const jsonReq = true === requires.json;
  out.sort((a, b) => {
    const aw = clientWire === a.probe.provider.wire ? 0 : 1;
    const bw = clientWire === b.probe.provider.wire ? 0 : 1;
    if (aw !== bw) return aw - bw;
    if (jsonReq) {
      const aj = 'openai' === a.probe.provider.wire ? 0 : 1;
      const bj = 'openai' === b.probe.provider.wire ? 0 : 1;
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
