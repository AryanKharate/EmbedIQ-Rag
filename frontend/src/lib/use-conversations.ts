import { useQuery } from "@tanstack/react-query";
import { conversationsApi, type ConversationSummary } from "./api";
import { isAuthenticated } from "./auth";

export type { ConversationSummary };

const CONVERSATIONS_KEY = ["conversations"] as const;

/**
 * Backend-persisted chat sessions for the current user — used to merge
 * sessions the local chat-store (localStorage) doesn't already know about
 * into the sidebar. See chat-store.ts's mergeBackendSessions().
 */
export function useConversations() {
  return useQuery({
    queryKey: CONVERSATIONS_KEY,
    queryFn: conversationsApi.list,
    staleTime: 30_000,
    enabled: isAuthenticated(),
    retry: false,
  });
}
