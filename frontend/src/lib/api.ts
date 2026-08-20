/**
 * API client for the EmbedIQ RAG backend.
 * Base path: /api  (proxied by Vite dev server or nginx in Docker)
 *
 * Automatically injects `Authorization: Bearer <token>` on every request.
 * On a 401, attempts a silent token refresh and retries once before
 * redirecting to /login.
 */

import {
  getAccessToken,
  refreshAccessToken,
  logout,
  type AuthUser,
} from "@/lib/auth";

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
  page_number?: number | string;
}

export interface StreamCallbacks {
  onSources: (sources: Source[]) => void;
  onToken: (text: string) => void;
  onDone: (session_id: string) => void;
  onError?: (err: Error) => void;
}

export interface QueryOptions {
  /** Per-request override of the server's HyDE default. Omit to use it. */
  useHyde?: boolean;
  /** Per-request override of the server's CRAG default. Omit to use it. */
  useCrag?: boolean;
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

/* ───────── Conversations (backend-persisted history) ───────── */

export interface ConversationSummary {
  id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ConversationDetail {
  id: string;
  turns: ConversationTurn[];
}

export const conversationsApi = {
  list: (): Promise<ConversationSummary[]> => request("/conversations/"),
  get: (id: string): Promise<ConversationDetail> =>
    request(`/conversations/${id}`),
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
    options?: QueryOptions,
  ): Promise<void> => {
    const token = getAccessToken();

    const makeHeaders = (t: string | null) => ({
      "Content-Type": "application/json",
      ...(t ? { Authorization: `Bearer ${t}` } : {}),
    });

    const body = JSON.stringify({
      question,
      session_id: session_id ?? null,
      use_hyde: options?.useHyde ?? null,
      use_crag: options?.useCrag ?? null,
    });

    let res = await fetch(`${BASE}/query`, {
      method: "POST",
      headers: makeHeaders(token),
      body,
    });

    // Silent token refresh on 401
    if (res.status === 401) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        res = await fetch(`${BASE}/query`, {
          method: "POST",
          headers: makeHeaders(newToken),
          body,
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

    // Smooth out however Gemini happens to chunk its output, WITHOUT adding a
    // fixed cost per character.
    //
    // The previous implementation queued every character behind its own
    // `setTimeout(..., 5)`, so displaying an answer took `length x 5ms` no
    // matter how fast the server delivered it — a 1200-character answer spent
    // 6s painting text that had already arrived, several times longer than the
    // actual Gemini stream. It also fired one React state update per character
    // (~200/sec), which the render loop could not keep up with on a long
    // thread, so the backlog grew as the answer got longer.
    //
    // Instead, drain on `requestAnimationFrame` and emit a *proportional*
    // slice each frame: whatever is pending is spread over at most
    // DRAIN_FRAMES frames (~100ms at 60fps). Text still appears progressively
    // rather than in one lump, but the buffer can never lag the network by
    // more than ~100ms, and rendering costs exactly one state update per
    // frame regardless of answer length.
    const DRAIN_FRAMES = 6;
    let pending = "";
    let rafId: number | null = null;

    const canAnimate =
      typeof requestAnimationFrame === "function" &&
      typeof document !== "undefined";

    const flushAll = () => {
      if (!pending) return;
      const text = pending;
      pending = "";
      callbacks.onToken(text);
    };

    const drainFrame = () => {
      rafId = null;
      if (!pending) return;

      // A hidden tab does not fire rAF at all, which would stall the stream
      // until the user came back. Nothing is being painted anyway, so just
      // hand over everything at once.
      if (document.visibilityState === "hidden") {
        flushAll();
        return;
      }

      const take = Math.max(1, Math.ceil(pending.length / DRAIN_FRAMES));
      const text = pending.slice(0, take);
      pending = pending.slice(take);
      callbacks.onToken(text);

      if (pending) rafId = requestAnimationFrame(drainFrame);
    };

    const enqueue = (text: string) => {
      if (!text) return;
      if (!canAnimate) {
        // SSR / non-browser consumer — no frames to align to.
        callbacks.onToken(text);
        return;
      }
      pending += text;
      if (rafId === null) rafId = requestAnimationFrame(drainFrame);
    };

    // Resolves once everything received has been handed to onToken. Bounded by
    // DRAIN_FRAMES frames rather than by the length of the answer.
    const waitForDrain = () =>
      new Promise<void>((resolve) => {
        const check = () => {
          if (!pending) {
            resolve();
            return;
          }
          if (!canAnimate || document.visibilityState === "hidden") {
            flushAll();
            resolve();
            return;
          }
          requestAnimationFrame(check);
        };
        check();
      });

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
              enqueue(payload.text as string);
            } else if (payload.type === "done") {
              // Everything received must reach onToken before onDone, or the
              // final state would drop the tail of the answer.
              await waitForDrain();
              callbacks.onDone(payload.session_id);
            }
          } catch {
            // Non-JSON line — skip
          }
        }
      }
    } catch (err) {
      // Drop anything still queued and cancel the pending frame, so a token
      // callback can't fire after onError has already reset the thread.
      pending = "";
      if (rafId !== null && canAnimate) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      callbacks.onError?.(err instanceof Error ? err : new Error(String(err)));
    }
  },
};
