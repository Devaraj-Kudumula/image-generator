const DEFAULT_FLASK_BASE =
  process.env.FLASK_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:5002";

export interface FlaskProxyResult<T> {
  ok: boolean;
  status: number;
  data: T;
  error: string;
}

export function getFlaskBaseUrl(): string {
  return DEFAULT_FLASK_BASE;
}

async function requestFlaskApi<T extends Record<string, unknown>>(
  path: string,
  init: RequestInit,
  timeoutMs: number
): Promise<FlaskProxyResult<T>> {
  try {
    const response = await fetch(`${DEFAULT_FLASK_BASE}${path}`, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
    });

    const text = await response.text();
    let data: T & { error?: string } = {} as T & { error?: string };

    if (text) {
      try {
        data = JSON.parse(text) as T & { error?: string };
      } catch {
        return {
          ok: false,
          status: response.status,
          data: {} as T,
          error: text || `Flask request failed: ${response.status}`,
        };
      }
    }

    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        data,
        error: data.error || text || `Flask request failed: ${response.status}`,
      };
    }

    return { ok: true, status: response.status, data, error: "" };
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : "Failed to reach Flask API. Run `npm run dev:flask` or set FLASK_API_URL.";
    return { ok: false, status: 503, data: {} as T, error: message };
  }
}

export async function proxyToFlaskApi<T extends Record<string, unknown>>(
  path: string,
  body: Record<string, unknown>,
  options?: { timeoutMs?: number }
): Promise<FlaskProxyResult<T>> {
  return requestFlaskApi<T>(
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    },
    options?.timeoutMs ?? 300_000
  );
}

export async function getFromFlaskApi<T extends Record<string, unknown>>(
  path: string,
  options?: { timeoutMs?: number }
): Promise<FlaskProxyResult<T>> {
  return requestFlaskApi<T>(path, { method: "GET" }, options?.timeoutMs ?? 30_000);
}

export async function proxyFormToFlaskApi<T extends Record<string, unknown>>(
  path: string,
  form: FormData,
  options?: { timeoutMs?: number }
): Promise<FlaskProxyResult<T>> {
  // Do not set Content-Type; fetch adds the multipart boundary automatically.
  return requestFlaskApi<T>(
    path,
    { method: "POST", body: form },
    options?.timeoutMs ?? 300_000
  );
}
