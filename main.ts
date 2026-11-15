// free-model-aggregator-01: quality-ranked routing with failover over the simulated providers.
//
// Built on the kit starter (main.ts + lib/providers.ts + lib/wire.ts). Changes vs the baseline:
//   - quality: every free candidate that meets a request's requirements is ranked by the quality
//     table's rating_milli (best first); the request goes to the best, not the first in brief order.
//   - failover: a provider call that answers 402, 429 or a 5xx is retried on the next-best candidate
//     that meets the request; the refused provider is marked for the rest of the trial so its
//     per-trial budget is not spent twice. A trial's refused set is the budget-spreading mechanism:
//     providers that refuse are out of budget (or on a failure schedule), so they are skipped.
//   - wires: all three upstream wires are served. The anthropic and gemini providers are reached
//     through lib/upstream.ts, which translates their requests and answers to and from OpenAI
//     shape (including their SSE streams), so the caller only ever deals with one wire. This
//     matters for capacity (10 providers, not 8) and for the best vision model (gemini 1458).
//   - the catalog, --list, register and poll are the starter's, unchanged.

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
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicStreamToOpenAIStream,
  anthropicToOpenAIResponse,
  geminiStreamToOpenAIStream,
  geminiToOpenAIResponse,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
} from './lib/upstream.ts';

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
/** Providers that refused a call this trial (402/429/5xx): out of budget or on a failure schedule. */
const refusedProviders = new Set<string>();

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

/**
 * Every free model that meets the requirements, best rating first, on any wire. Providers that
 * refused this trial are skipped (their per-trial budget is spent). The anthropic and gemini
 * wires are served by lib/upstream.ts, which translates them to and from OpenAI shape.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (refusedProviders.has(probe.provider.id)) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  // Best rating first; ties break by model id, then provider id, so the order is deterministic.
  out.sort((a, b) => b.rating - a.rating || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0) || (a.probe.provider.id < b.probe.provider.id ? -1 : 1));
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

/**
 * One inference call to the candidate's provider, in the provider's own wire, returned as an
 * OpenAI-shaped Response (JSON or SSE) so the caller only ever deals with one wire. A non-2xx is
 * returned as that status with the provider's body, so the caller can fail over.
 */
async function callProvider(candidate: Candidate, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const provider = candidate.probe.provider;
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';

  if (provider.wire === 'openai') {
    // A streaming answer must end with a usage chunk: ask for it, or the client's usage check
    // fails. stream_options is not part of the provider's echo, so adding it is safe.
    const upstreamBody: Json = stream ? { ...body, stream_options: { include_usage: true } } : body;
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
  }

  if (provider.wire === 'anthropic') {
    const upstream = await fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(openAIToAnthropicRequest(body, candidate.model.id)) });
    if (!upstream.ok) return upstream;
    if (stream && upstream.body !== null) {
      const sse = ReadableStream.from(anthropicStreamToOpenAIStream(upstream.body, candidate.model.id)).pipeThrough(new TextEncoderStream());
      return new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } });
    }
    const answer: unknown = await upstream.json();
    return new Response(JSON.stringify(anthropicToOpenAIResponse(isRecord(answer) ? answer : {}, candidate.model.id)), { status: 200, headers: { 'content-type': 'application/json' } });
  }

  // gemini wire
  const { urlModel, payload } = openAIToGeminiRequest(body, candidate.model.id);
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const upstream = await fetch(`${base}/v1beta/models/${encodeURIComponent(urlModel)}:${action}`, { method: 'POST', headers, body: JSON.stringify(payload) });
  if (!upstream.ok) return upstream;
  if (stream && upstream.body !== null) {
    const sse = ReadableStream.from(geminiStreamToOpenAIStream(upstream.body, candidate.model.id)).pipeThrough(new TextEncoderStream());
    return new Response(sse, { status: 200, headers: { 'content-type': 'text/event-stream' } });
  }
  const answer: unknown = await upstream.json();
  return new Response(JSON.stringify(geminiToOpenAIResponse(isRecord(answer) ? answer : {}, candidate.model.id)), { status: 200, headers: { 'content-type': 'application/json' } });
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

/** A refusal (402/429/5xx) marks the provider refused for the rest of the trial. */
function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  let lastStatus = 0;
  let lastText = '';
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, { ...body, model: candidate.model.id });
    } catch (err) {
      refusedProviders.add(candidate.probe.provider.id);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      lastText = (await upstream.text()).slice(0, 500);
      if (isRefusal(upstream.status)) {
        refusedProviders.add(candidate.probe.provider.id);
        continue;
      }
      res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
      res.end(lastText);
      return;
    }
    if (body.stream === true && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      await pipeStream(upstream.body, res);
      return;
    }
    const text = await upstream.text();
    res.writeHead(upstream.status, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
    res.end(text);
    return;
  }
  sendError(res, 'openai', 503, 'no_free_model', `every candidate refused (last HTTP ${lastStatus}): ${lastText.slice(0, 200)}`);
}

async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires);
  if (candidates.length === 0) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  let lastStatus = 0;
  let lastText = '';
  for (const candidate of candidates) {
    const requested = typeof body.model === 'string' && body.model !== '' ? body.model : candidate.model.id;
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, anthropicToOpenAI(body, candidate.model.id));
    } catch (err) {
      refusedProviders.add(candidate.probe.provider.id);
      lastStatus = 502;
      lastText = (err as Error).message;
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      lastText = (await upstream.text()).slice(0, 500);
      if (isRefusal(upstream.status)) {
        refusedProviders.add(candidate.probe.provider.id);
        continue;
      }
      return sendJson(res, upstream.status, anthropicError('api_error', `the provider answered HTTP ${upstream.status}: ${lastText}`));
    }
    if (body.stream === true && upstream.body !== null) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      for await (const event of anthropicEventsFromOpenAI(upstream.body, requested)) res.write(event);
      res.end();
      return;
    }
    const answer: unknown = await upstream.json();
    sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, requested));
    return;
  }
  sendError(res, 'anthropic', 503, 'overloaded_error', `every candidate refused (last HTTP ${lastStatus}): ${lastText.slice(0, 200)}`);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
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
