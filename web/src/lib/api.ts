import { readSSE } from "./sse.ts";
import type {
  AskRequest,
  BenchmarkQuestion,
  ErrorKind,
  ExperimentsIndex,
  Health,
  Meta,
  PipelineEvent,
  PipelineResult,
  RetrieveRequest,
  RetrieveResponse,
} from "./types.ts";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

/**
 * Why a call failed, in terms the interface can act on.
 * - `unreachable`: no HTTP response at all (server down, wrong port, CORS).
 * - `not_found`: a 404, which for experiments and calibration means "not run yet".
 * - `invalid`: the server rejected the request body (4xx).
 * - the pipeline kinds from SPEC 3.7 when the server names one.
 */
export type ApiErrorKind = "unreachable" | "not_found" | "invalid" | "bad_response" | ErrorKind;

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;

  constructor(kind: ApiErrorKind, message: string, status: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

const PIPELINE_KINDS: readonly string[] = ["no_api_key", "budget", "no_index", "internal"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Turns a non-2xx response into an ApiError, reading the `{error: {kind, message}}` body if present. */
async function errorFromResponse(response: Response): Promise<ApiError> {
  let kind: ApiErrorKind = response.status === 404 ? "not_found" : response.status < 500 ? "invalid" : "internal";
  let message = `The API answered ${response.status} ${response.statusText}`.trim() + ".";
  try {
    const body: unknown = await response.json();
    if (isRecord(body)) {
      const error = body.error;
      if (isRecord(error)) {
        if (typeof error.kind === "string" && PIPELINE_KINDS.includes(error.kind)) kind = error.kind as ErrorKind;
        if (typeof error.message === "string" && error.message) message = error.message;
      } else if (typeof body.detail === "string") {
        message = body.detail;
      } else if (Array.isArray(body.detail)) {
        // FastAPI validation errors: [{loc, msg, type}]
        const parts = body.detail
          .map((d) => (isRecord(d) && typeof d.msg === "string" ? d.msg : null))
          .filter((m): m is string => m !== null);
        if (parts.length) message = parts.join("; ");
      }
    }
  } catch {
    // Not JSON; the status line is all there is to report.
  }
  return new ApiError(kind, message, response.status);
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError("unreachable", `No response from the Primum API at ${API_URL}.`);
  }
  if (!response.ok) throw await errorFromResponse(response);
  return response;
}

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await request(path, { signal });
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("bad_response", `The API answered ${path} with something that is not JSON.`, response.status);
  }
}

async function postJSON<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("bad_response", `The API answered ${path} with something that is not JSON.`, response.status);
  }
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return getJSON<Health>("/api/health", signal);
}

export function getMeta(signal?: AbortSignal): Promise<Meta> {
  return getJSON<Meta>("/api/meta", signal);
}

export function retrieve(body: RetrieveRequest, signal?: AbortSignal): Promise<RetrieveResponse> {
  return postJSON<RetrieveResponse>("/api/retrieve", body, signal);
}

/** The default run on the server: the main results directory. */
export const MAIN_RUN = "main";

/** `?run=<name>` for any run but the main one, so main-run URLs stay as they were. */
function runQuery(run: string | undefined): string {
  return run && run !== MAIN_RUN ? `?run=${encodeURIComponent(run)}` : "";
}

export function getExperimentsIndex(signal?: AbortSignal, run?: string): Promise<ExperimentsIndex> {
  return getJSON<ExperimentsIndex>(`/api/experiments${runQuery(run)}`, signal);
}

export type ExperimentName = "rq1_retrieval" | "rq2_confidence" | "rq3_conformal" | "encoder_training" | "pubmedqa_runs";

export function getExperiment(name: ExperimentName, signal?: AbortSignal, run?: string): Promise<unknown> {
  return getJSON<unknown>(`/api/experiments/${name}${runQuery(run)}`, signal);
}

export function getCalibration(signal?: AbortSignal, run?: string): Promise<unknown> {
  return getJSON<unknown>(`/api/calibration${runQuery(run)}`, signal);
}

/** The list endpoints may answer with a bare array or wrap it; accept either. */
function unwrapList(body: unknown, keys: string[]): unknown[] {
  if (Array.isArray(body)) return body;
  if (isRecord(body)) {
    for (const key of keys) {
      const value = body[key];
      if (Array.isArray(value)) return value;
    }
  }
  return [];
}

export async function getBenchmarkQuestions(
  params: { limit: number; offset: number },
  signal?: AbortSignal,
): Promise<BenchmarkQuestion[]> {
  const body = await getJSON<unknown>(`/api/benchmark/questions?limit=${params.limit}&offset=${params.offset}`, signal);
  return unwrapList(body, ["questions", "items", "results"]).flatMap((item) => {
    if (!isRecord(item)) return [];
    const qid = item.qid ?? item.id;
    if ((typeof qid !== "string" && typeof qid !== "number") || typeof item.question !== "string") return [];
    return [{ qid: String(qid), question: item.question }];
  });
}

export interface TraceSummary {
  trace_id: string;
  question: string;
  answered: boolean | null;
  mode: string | null;
}

export async function getTraces(limit: number, signal?: AbortSignal): Promise<TraceSummary[]> {
  const body = await getJSON<unknown>(`/api/traces?limit=${limit}`, signal);
  return unwrapList(body, ["traces", "items", "results"]).flatMap((item) => {
    if (!isRecord(item) || typeof item.trace_id !== "string" || typeof item.question !== "string") return [];
    const confidence = isRecord(item.confidence) ? item.confidence : null;
    const answered =
      typeof item.answered === "boolean"
        ? item.answered
        : confidence && typeof confidence.answered === "boolean"
          ? confidence.answered
          : null;
    return [
      {
        trace_id: item.trace_id,
        question: item.question,
        answered,
        mode: typeof item.mode === "string" ? item.mode : null,
      },
    ];
  });
}

export function getTrace(id: string, signal?: AbortSignal): Promise<PipelineResult> {
  return getJSON<PipelineResult>(`/api/traces/${encodeURIComponent(id)}`, signal);
}

const EVENT_TYPES: readonly string[] = [
  "stage",
  "triage",
  "concepts",
  "retrieval",
  "token",
  "answer",
  "verdicts",
  "confidence",
  "done",
  "error",
];

/**
 * Runs the pipeline for one question and yields its events in the order the
 * server emits them. Throws ApiError if the request never becomes a stream.
 * Unknown event types and frames whose data is not JSON are skipped, so a newer
 * server does not break an older page.
 */
export async function* ask(body: AskRequest, signal?: AbortSignal): AsyncGenerator<PipelineEvent> {
  const response = await request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.body) {
    throw new ApiError("bad_response", "The API accepted the question but sent no event stream.", response.status);
  }
  for await (const message of readSSE(response.body)) {
    if (!EVENT_TYPES.includes(message.event)) continue;
    let data: unknown;
    try {
      data = JSON.parse(message.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    yield { type: message.event, data } as PipelineEvent;
  }
}
