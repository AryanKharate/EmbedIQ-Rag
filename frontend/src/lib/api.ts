/**
 * API client for the EmbedIQ RAG backend.
 * Base path: /api  (proxied by Vite dev server or nginx in Docker)
 *
 * Automatically injects `Authorization: Bearer <token>` on every request.
 * On a 401, attempts a silent token refresh and retries once before
 * redirecting to /login.
 */

import { getAccessToken, refreshAccessToken, logout } from "@/lib/auth";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getAccessToken();

  const makeHeaders = (t: string | null) => ({
    "Content-Type": "application/json",
    ...(t ? { Authorization: `Bearer ${t}` } : {}),
    ...init?.headers,
  });

  let res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: makeHeaders(token),
  });

  // Attempt silent refresh on 401
  if (res.status === 401) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      res = await fetch(`${BASE}${path}`, {
        ...init,
        headers: makeHeaders(newToken),
      });
    } else {
      logout();
      throw new Error("Session expired. Please log in again.");
    }
  }

  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(
      `API ${init?.method ?? "GET"} ${path} failed (${res.status}): ${text}`,
    );
  }
  // 204 No Content — return empty object
  if (res.status === 204) return {} as T;
  return res.json() as Promise<T>;
}

/** Upload helper — no JSON Content-Type; attach token manually. */
async function uploadRequest<T>(path: string, body: FormData): Promise<T> {
  const token = getAccessToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res = await fetch(`${BASE}${path}`, { method: "POST", headers, body });

  if (res.status === 401) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      headers["Authorization"] = `Bearer ${newToken}`;
      res = await fetch(`${BASE}${path}`, { method: "POST", headers, body });
    } else {
      logout();
      throw new Error("Session expired. Please log in again.");
    }
  }

  if (!res.ok) throw new Error(`Upload failed (${res.status})`);
  return res.json();
}

/* ───────── Types ───────── */

export interface ApiDocument {
  id: string;
  filename: string;
  is_active: boolean;
  created_at: string;
}

export interface Source {
  source: string;
  chunk_index?: number;
  parent_id?: string;
  score?: number;
  image_urls?: string[];
}

export interface StreamCallbacks {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (session_id: string) => void;
  onError?: (err: Error) => void;
}

export interface AuthUser {
  id: number;
  email: string;
  display_name: string;
  is_new?: boolean;
}

export interface AuthResponse {
  access: string;
  refresh: string;
  user: AuthUser;
}

/* ───────── Auth ───────── */

export const authApi = {
  register: (
    email: string,
    password: string,
    display_name: string,
  ): Promise<AuthResponse> =>
    fetch(`${BASE}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, display_name }),
    }).then(async (r) => {
      if (!r.ok) {
        const err = await r
          .json()
          .catch(() => ({ detail: "Registration failed" }));
        throw new Error(err.detail ?? "Registration failed");
      }
      return r.json();
    }),

  login: (email: string, password: string): Promise<AuthResponse> =>
    fetch(`${BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }).then(async (r) => {
      if (!r.ok) {
        const err = await r
          .json()
          .catch(() => ({ detail: "Invalid credentials" }));
        throw new Error(err.detail ?? "Invalid credentials");
      }
      return r.json();
    }),

  googleAuth: (id_token: string): Promise<AuthResponse> =>
    fetch(`${BASE}/auth/google`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id_token }),
    }).then(async (r) => {
      if (!r.ok) {
        const err = await r
          .json()
          .catch(() => ({ detail: "Google sign-in failed" }));
        throw new Error(err.detail ?? "Google sign-in failed");
      }
      return r.json();
    }),
};

/* ───────── Documents ───────── */

export const documentsApi = {
  list: (): Promise<ApiDocument[]> => request("/documents/"),

  upload: (file: File): Promise<ApiDocument> => {
    const form = new FormData();
    form.append("file", file);
    return uploadRequest("/documents/upload", form);
  },

  toggle: (id: string, is_active: boolean): Promise<ApiDocument> =>
    request(`/documents/${id}/toggle`, {
      method: "PATCH",
      body: JSON.stringify({ is_active }),
    }),

  delete: (id: string): Promise<void> =>
    request(`/documents/${id}`, { method: "DELETE" }),
};

/* ───────── Chat / Query ───────── */

export const chatApi = {
  /**
   * Send a question to the RAG backend and receive the answer as a
   * Server-Sent Events stream. Fires callbacks as events arrive:
   *   onSources → immediately, before any text
   *   onToken   → per Gemini chunk (with a small smoothing delay)
   *   onDone    → when the stream ends, carries session_id
   *   onError   → on network / parse failure
   */
  queryStream: async (
    question: string,
    session_id: string | null | undefined,
    callbacks: StreamCallbacks,
  ): Promise<void> => {
    const token = getAccessToken();

    const makeHeaders = (t: string | null) => ({
      "Content-Type": "application/json",
      ...(t ? { Authorization: `Bearer ${t}` } : {}),
    });

    let res = await fetch(`${BASE}/query`, {
      method: "POST",
      headers: makeHeaders(token),
      body: JSON.stringify({ question, session_id: session_id ?? null }),
    });

    // Silent token refresh on 401
    if (res.status === 401) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        res = await fetch(`${BASE}/query`, {
          method: "POST",
          headers: makeHeaders(newToken),
          body: JSON.stringify({ question, session_id: session_id ?? null }),
        });
      } else {
        logout();
        callbacks.onError?.(new Error("Session expired. Please log in again."));
        return;
      }
    }

    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => "Unknown error");
      callbacks.onError?.(new Error(`Query failed (${res.status}): ${text}`));
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // Each character is queued individually so the delay creates true
    // character-by-character rendering (like ChatGPT/Claude), regardless of
    // how large each chunk Gemini sends is.
    const CHAR_DELAY_MS = 5; // tune this: lower = faster, higher = slower
    const charQueue: string[] = [];
    let draining = false;

    const drainQueue = () => {
      if (draining || charQueue.length === 0) return;
      draining = true;
      const next = () => {
        if (charQueue.length === 0) {
          draining = false;
          return;
        }

        // If the user switches tabs, browsers aggressively throttle setTimeout.
        // To prevent the stream from freezing, flush the queue instantly.
        if (
          typeof document !== "undefined" &&
          document.visibilityState === "hidden"
        ) {
          callbacks.onToken(charQueue.join(""));
          charQueue.length = 0;
          draining = false;
          return;
        }

        callbacks.onToken(charQueue.shift()!);
        setTimeout(next, CHAR_DELAY_MS);
      };
      setTimeout(next, CHAR_DELAY_MS);
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? ""; // keep any incomplete trailing chunk

        for (const event of events) {
          const line = event.trim();
          if (!line.startsWith("data: ")) continue;

          try {
            const payload = JSON.parse(line.slice(6));

            if (payload.type === "sources") {
              callbacks.onSources(payload.sources ?? []);
            } else if (payload.type === "token") {
              // Explode the chunk into individual characters
              for (const char of payload.text as string) {
                charQueue.push(char);
              }
              drainQueue();
            } else if (payload.type === "done") {
              // Wait for all queued characters to drain before firing onDone
              const waitForDrain = () =>
                new Promise<void>((resolve) => {
                  const check = () => {
                    if (charQueue.length === 0 && !draining) {
                      resolve();
                    } else {
                      setTimeout(check, CHAR_DELAY_MS * 2);
                    }
                  };
                  check();
                });
              await waitForDrain();
              callbacks.onDone(payload.session_id);
            }
          } catch {
            // Non-JSON line — skip
          }
        }
      }
    } catch (err) {
      callbacks.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  },
};
