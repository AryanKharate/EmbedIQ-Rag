import { type ConversationSummary, type Source } from "./api";

export type ChatRole = "user" | "assistant";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  sources?: Source[];
}

export interface ChatThread {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
  /** Linked backend ConversationSession id, once known (set after the first
   *  reply, or immediately for a thread merged in from the backend). */
  sessionId?: string;
  /** True for a stub thread merged in from a backend session whose messages
   *  haven't been fetched yet — see mergeBackendSessions() below. */
  needsFetch?: boolean;
}

const THREADS_KEY = "rag.threads.v1";
// Backend session ids the user explicitly deleted locally. There's no
// DELETE /api/conversations/{id} endpoint (this is a read-only history
// view), so without tracking dismissals, deleting a backend-linked thread
// would just have mergeBackendSessions() re-add it as a stub on next load.
const DISMISSED_SESSIONS_KEY = "rag.dismissed_sessions.v1";

function loadDismissedSessionIds(): Set<string> {
  if (typeof window === "undefined") return new Set();
  try {
    const raw = window.localStorage.getItem(DISMISSED_SESSIONS_KEY);
    if (!raw) return new Set();
    const parsed = JSON.parse(raw);
    return new Set(Array.isArray(parsed) ? parsed : []);
  } catch {
    return new Set();
  }
}

function saveDismissedSessionIds(ids: Set<string>) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(DISMISSED_SESSIONS_KEY, JSON.stringify([...ids]));
}

/** Mark a backend session as locally deleted so mergeBackendSessions()
 *  doesn't bring it back as a stub. Call this whenever a thread with a
 *  sessionId is removed from the sidebar. */
export function dismissSession(sessionId: string) {
  const ids = loadDismissedSessionIds();
  ids.add(sessionId);
  saveDismissedSessionIds(ids);
}

export function loadThreads(): ChatThread[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(THREADS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as ChatThread[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveThreads(threads: ChatThread[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(THREADS_KEY, JSON.stringify(threads));
}

export function createThread(): ChatThread {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

export function deriveTitle(content: string): string {
  const trimmed = content.trim().replace(/\s+/g, " ");
  if (!trimmed) return "New chat";
  return trimmed.length > 42 ? trimmed.slice(0, 42) + "…" : trimmed;
}

/**
 * Add any backend session not already linked to a local thread as a new
 * "stub" thread (empty messages, needsFetch=true) so it shows up in the
 * sidebar. Existing local threads are left untouched and keep their
 * position — backend-only sessions are appended, most-recent first.
 *
 * This is the "hybrid" approach to conversation history: localStorage stays
 * the fast, primary cache (no loading spinner for your usual chats), but a
 * session that only exists in the backend — e.g. read on another device, or
 * after clearing local storage — is no longer permanently invisible.
 *
 * The stub's local id is the session id itself, so /c/$threadId routing
 * needs no separate mapping between a local thread id and a backend session.
 */
export function mergeBackendSessions(
  backendSessions: ConversationSummary[],
): ChatThread[] {
  const local = loadThreads();
  const knownSessionIds = new Set(
    local.map((t) => t.sessionId).filter((id): id is string => Boolean(id)),
  );
  const dismissed = loadDismissedSessionIds();

  const newStubs: ChatThread[] = backendSessions
    .filter((s) => !knownSessionIds.has(s.id) && !dismissed.has(s.id))
    .sort(
      (a, b) =>
        new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
    )
    .map((s) => {
      const updatedAt = new Date(s.updated_at).getTime();
      return {
        id: s.id,
        title: s.title,
        messages: [],
        createdAt: updatedAt,
        updatedAt,
        sessionId: s.id,
        needsFetch: true,
      };
    });

  if (newStubs.length === 0) return local;

  const merged = [...local, ...newStubs];
  saveThreads(merged);
  return merged;
}
