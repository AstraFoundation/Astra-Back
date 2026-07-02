// Shared HTTP plumbing: bearer auth, retry with backoff, error mapping.
// Mirrors clients/python/astra_sdk/_http.py. Uses the global `fetch` (Node 18+)
// — zero runtime dependencies.

const RETRYABLE_STATUS = new Set([429, 502, 503, 504]);

export class AstraApiError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(`[${status}] ${code}: ${message}`);
    this.name = "AstraApiError";
    this.status = status;
    this.code = code;
  }
}

// The hosted Astra origin every deployment lives behind. Baked in so SDK code
// never has to carry a base URL; override with the ASTRA_BASE_URL env var or an
// explicit baseUrl argument (e.g. for self-host / testing).
export const DEFAULT_BASE_URL = "https://astra.kwon5700.kr";

export function resolveBaseUrl(baseUrl?: string): string {
  return (baseUrl || process.env.ASTRA_BASE_URL || DEFAULT_BASE_URL).replace(/\/+$/, "");
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** Parse a non-2xx response into an AstraApiError (reads `detail.code/message`). */
export async function errorFromResponse(resp: Response): Promise<AstraApiError> {
  let raw = "";
  try {
    raw = await resp.text();
  } catch {
    /* body already consumed / unreadable */
  }
  let code = "error";
  let message = raw.slice(0, 500);
  try {
    const body = JSON.parse(raw) as unknown;
    const detail =
      body && typeof body === "object"
        ? (body as Record<string, unknown>).detail
        : undefined;
    if (detail && typeof detail === "object") {
      const d = detail as Record<string, unknown>;
      code = typeof d.code === "string" ? d.code : "error";
      if (typeof d.message === "string") message = d.message;
    } else if (detail !== undefined) {
      message = String(detail);
    }
  } catch {
    /* not JSON — keep the raw text */
  }
  return new AstraApiError(resp.status, code, message || resp.statusText);
}

export interface HttpRequestOptions {
  json?: unknown;
  headers?: Record<string, string>;
}

export interface HttpSessionOptions {
  /** Seconds (matches the Python API). */
  timeout?: number;
  maxAttempts?: number;
  maxBackoff?: number;
}

/**
 * fetch wrapper with bearer auth + bounded retry/backoff.
 *
 * Retries transient failures (network errors, 429/5xx) with exponential
 * backoff capped at `maxBackoff`; gives up after `maxAttempts` and raises the
 * last error. 4xx (except 429) never retries.
 */
export class HttpSession {
  readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;
  private readonly maxAttempts: number;
  private readonly maxBackoff: number;

  constructor(baseUrl: string, apiKey: string, opts: HttpSessionOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.headers = { Authorization: `Bearer ${apiKey}` };
    this.timeoutMs = (opts.timeout ?? 30) * 1000;
    this.maxAttempts = Math.max(1, opts.maxAttempts ?? 3);
    this.maxBackoff = opts.maxBackoff ?? 60;
  }

  async request(
    method: string,
    path: string,
    opts: HttpRequestOptions = {},
  ): Promise<Response> {
    const url = `${this.baseUrl}${path}`;
    const headers: Record<string, string> = { ...this.headers };
    if (opts.json !== undefined) headers["Content-Type"] = "application/json";
    if (opts.headers) Object.assign(headers, opts.headers);

    let lastErr: unknown = null;
    for (let attempt = 0; attempt < this.maxAttempts; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      let resp: Response | undefined;
      try {
        resp = await fetch(url, {
          method,
          headers,
          body: opts.json !== undefined ? JSON.stringify(opts.json) : undefined,
          signal: controller.signal,
        });
      } catch (err) {
        lastErr = err; // network / abort error — retryable
      } finally {
        clearTimeout(timer);
      }

      if (resp) {
        if (resp.status < 400 || resp.status === 304) return resp;
        if (!RETRYABLE_STATUS.has(resp.status)) {
          throw await errorFromResponse(resp);
        }
        lastErr = await errorFromResponse(resp);
      }

      if (attempt < this.maxAttempts - 1) {
        const backoff = Math.min(this.maxBackoff, 2 ** attempt + Math.random());
        await sleep(backoff * 1000);
      }
    }
    if (lastErr instanceof Error) throw lastErr;
    throw new AstraApiError(0, "error", String(lastErr));
  }

  // Symmetry with the Python API; there is no persistent client to tear down.
  close(): void {
    /* no-op: global fetch holds no per-session resources */
  }
}
