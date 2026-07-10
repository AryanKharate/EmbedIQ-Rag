import { ArrowUp, FileText, Sparkles } from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
} from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  deriveTitle,
  loadThreads,
  saveThreads,
  type ChatMessage,
  type ChatThread,
} from "@/lib/chat-store";
import { useHydrated } from "@/hooks/use-hydrated";
import { chatApi } from "@/lib/api";

interface Props {
  threadId: string;
}

// Extend ChatThread to persist the backend session_id per conversation
interface RagThread extends ChatThread {
  sessionId?: string;
}

export function ChatView({ threadId }: Props) {
  const hydrated = useHydrated();
  const [thread, setThread] = useState<RagThread | null>(null);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Load the thread from storage
  useEffect(() => {
    if (!hydrated) return;
    const all = loadThreads() as RagThread[];
    const found = all.find((t) => t.id === threadId) ?? null;
    setThread(found);
  }, [hydrated, threadId]);

  // Focus textarea on thread change
  useEffect(() => {
    textareaRef.current?.focus();
  }, [threadId]);

  // Auto-scroll on new messages
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [thread?.messages.length, sending]);

  const persist = useCallback((updated: RagThread) => {
    const all = loadThreads() as RagThread[];
    const idx = all.findIndex((t) => t.id === updated.id);
    const next =
      idx >= 0
        ? all.map((t) => (t.id === updated.id ? updated : t))
        : [updated, ...all];
    saveThreads(next);
    window.dispatchEvent(new Event("rag:threads-updated"));
  }, []);

  const handleSubmit = useCallback(
    async (e?: FormEvent) => {
      e?.preventDefault();
      const text = input.trim();
      if (!text || !thread || sending) return;

      setError(null);

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: text,
        createdAt: Date.now(),
      };

      const withUser: RagThread = {
        ...thread,
        title: thread.messages.length === 0 ? deriveTitle(text) : thread.title,
        messages: [...thread.messages, userMsg],
        updatedAt: Date.now(),
      };
      setThread(withUser);
      persist(withUser);
      setInput("");
      setSending(true);

      try {
        // Call the real RAG backend
        const result = await chatApi.query(text, thread.sessionId ?? null);

        const assistantMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          content: result.answer,
          createdAt: Date.now(),
        };

        const finalThread: RagThread = {
          ...withUser,
          sessionId: result.session_id, // persist session for follow-ups
          messages: [...withUser.messages, assistantMsg],
          updatedAt: Date.now(),
        };
        setThread(finalThread);
        persist(finalThread);
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        // Remove the optimistic user message on error
        setThread(thread);
      } finally {
        setSending(false);
        requestAnimationFrame(() => textareaRef.current?.focus());
      }
    },
    [input, thread, sending, persist],
  );

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit();
    }
  };

  const messages = thread?.messages ?? [];
  const showEmpty = messages.length === 0 && !sending;

  const suggestions = useMemo(
    () => [
      "Summarize my active documents",
      "What are the key takeaways?",
      "Find mentions of pricing",
      "Draft an email based on this",
    ],
    [],
  );

  if (hydrated && !thread) {
    return (
      <div className="flex h-full items-center justify-center p-8 text-center text-sm text-muted-foreground">
        This conversation doesn't exist. Start a new chat from the sidebar.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-3xl px-4 py-8">
          {showEmpty ? (
            <EmptyState
              onPick={(s) => {
                setInput(s);
                textareaRef.current?.focus();
              }}
              suggestions={suggestions}
            />
          ) : (
            <ul className="flex flex-col gap-6">
              {messages.map((m) => (
                <MessageBubble key={m.id} message={m} />
              ))}
              {sending && <TypingBubble />}
            </ul>
          )}

          {error && (
            <div className="mt-4 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          )}
        </div>
      </div>

      <div className="border-t bg-background/60 backdrop-blur">
        <form
          onSubmit={handleSubmit}
          className="mx-auto w-full max-w-3xl px-4 py-4"
        >
          <div className="relative flex items-end rounded-2xl border bg-card shadow-sm focus-within:ring-2 focus-within:ring-ring/40">
            <Textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about your documents…"
              rows={1}
              className="max-h-48 min-h-[52px] resize-none border-0 bg-transparent px-4 py-4 pr-14 text-[15px] leading-6 shadow-none focus-visible:ring-0"
            />
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || sending}
              className="absolute bottom-2 right-2 h-9 w-9 rounded-xl"
              aria-label="Send message"
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          </div>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            Responses are generated from your active documents.
          </p>
        </form>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <li className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Sparkles className="h-4 w-4" />
        </div>
      )}
      <div
        className={cn(
          "max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-[15px] leading-6",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground",
        )}
      >
        {message.content}
      </div>
    </li>
  );
}

function TypingBubble() {
  return (
    <li className="flex gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
        <Sparkles className="h-4 w-4" />
      </div>
      <div className="flex items-center gap-1 rounded-2xl bg-muted px-4 py-3">
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.3s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50 [animation-delay:-0.15s]" />
        <span className="h-2 w-2 animate-bounce rounded-full bg-muted-foreground/50" />
      </div>
    </li>
  );
}

function EmptyState({
  onPick,
  suggestions,
}: {
  onPick: (s: string) => void;
  suggestions: string[];
}) {
  return (
    <div className="flex flex-col items-center pt-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-primary text-primary-foreground shadow-sm">
        <FileText className="h-6 w-6" />
      </div>
      <h1 className="mt-5 text-2xl font-semibold tracking-tight">
        Ask anything about your documents
      </h1>
      <p className="mt-2 max-w-md text-sm text-muted-foreground">
        Toggle documents in the sidebar to include them in the assistant's
        context, then ask a question below.
      </p>
      <div className="mt-8 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2">
        {suggestions.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onPick(s)}
            className="rounded-xl border bg-card px-4 py-3 text-left text-sm text-card-foreground transition hover:bg-accent"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
