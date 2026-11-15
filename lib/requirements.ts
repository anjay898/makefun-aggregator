// What one client request requires of the model that serves it.
//
// The workload manifest is authoritative: the requests arrive in the manifest's order, so the
// requirements of the request being served are the manifest entry at the same position — its
// tools, vision and JSON flags and its minimum context, exactly as the brief sealed them. The
// manifest is consumed one entry per client request, never per upstream attempt: a failover
// retry is the same request and must not drift to the next entry's requirements.
// Outside a trial (or if the manifest runs out) the requirements are read off the request body
// itself: a tools array means tools, an image block in any message means vision, and a
// response_format of json_object or json_schema means JSON mode. That fallback keeps the server
// honest when it is probed by hand; in a graded trial the manifest path is the one that runs.

import { isRecord, type Json, type Requirement, type WorkloadFile } from './providers.ts';

/** Requirements read off a request body, for when no manifest entry names them. */
export function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && ('image' === block.type || 'image_url' === block.type)) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: 'json_object' === format || 'json_schema' === format };
}

/** A cursor over the workload manifest: one entry per client request, in arrival order. */
export class ManifestCursor {
  private index = 0;

  constructor(private readonly workload: WorkloadFile | null) {}

  /** The requirements for the next client request: the manifest's word, or the body's. */
  next(body: Json): Requirement {
    const manifest = this.workload?.manifest ?? [];
    const entry = manifest[this.index];
    this.index += 1;
    if (entry !== undefined) return entry.requires;
    return requirementsFromBody(body);
  }
}
