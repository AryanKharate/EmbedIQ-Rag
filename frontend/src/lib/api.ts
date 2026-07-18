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

export interface QueryResponse {
  answer: string;
  session_id: string;
  sources?: Source[];
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
  query: (
    question: string,
    session_id?: string | null,
  ): Promise<QueryResponse> =>
    request("/query", {
      method: "POST",
      body: JSON.stringify({ question, session_id: session_id ?? null }),
    }),
};
