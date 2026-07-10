/**
 * API client for the EmbedIQ RAG backend.
 * Base path: /api  (proxied by Vite dev server or nginx in Docker)
 */

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
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

/* ───────── Types ───────── */

export interface ApiDocument {
  id: string;
  filename: string;
  is_active: boolean;
  created_at: string;
}

export interface QueryResponse {
  answer: string;
  session_id: string;
}

/* ───────── Documents ───────── */

export const documentsApi = {
  list: (): Promise<ApiDocument[]> => request("/documents/"),

  upload: (file: File): Promise<ApiDocument> => {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/documents/upload`, {
      method: "POST",
      body: form,
    }).then((r) => {
      if (!r.ok) throw new Error(`Upload failed (${r.status})`);
      return r.json();
    });
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
