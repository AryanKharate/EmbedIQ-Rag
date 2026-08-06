import { ArrowUp, FileText, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
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
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Ref to hold the mutable streaming thread without triggering extra re-renders
  const streamingThreadRef = useRef<RagThread | null>(null);

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

  // Auto-scroll on new messages or streaming content
  useLayoutEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [thread?.messages.length, sending, isStreaming]);

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

      // Placeholder assistant message — content fills in as tokens arrive
      const assistantMsgId = crypto.randomUUID();
      const assistantPlaceholder: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        sources: [],
      };

      try {
        await chatApi.queryStream(text, thread.sessionId ?? null, {
          onSources: (sources) => {
            // Sources arrive before any text — insert the assistant bubble immediately
            const withAssistant: RagThread = {
              ...withUser,
              messages: [
                ...withUser.messages,
                { ...assistantPlaceholder, sources },
              ],
            };
            streamingThreadRef.current = withAssistant;
            setThread(withAssistant);
            setSending(false); // hide TypingBubble
            setIsStreaming(true);
          },

          onToken: (tokenText) => {
            const base = streamingThreadRef.current ?? withUser;
            const updated: RagThread = {
              ...base,
              messages: base.messages.map((m) =>
                m.id === assistantMsgId
                  ? { ...m, content: m.content + tokenText }
                  : m,
              ),
            };
            streamingThreadRef.current = updated;
            setThread(updated);
          },

          onDone: (session_id) => {
            const final = streamingThreadRef.current ?? withUser;
            const finalThread: RagThread = {
              ...final,
              sessionId: session_id,
              updatedAt: Date.now(),
            };
            streamingThreadRef.current = null;
            setIsStreaming(false);
            setThread(finalThread);
            persist(finalThread);
          },

          onError: (err) => {
            const msg = err.message ?? "Something went wrong.";
            setError(msg);
            setSending(false);
            setIsStreaming(false);
            streamingThreadRef.current = null;
            setThread(thread);
          },
        });
      } catch (err) {
        const msg =
          err instanceof Error ? err.message : "Something went wrong.";
        setError(msg);
        setSending(false);
        setIsStreaming(false);
        streamingThreadRef.current = null;
        setThread(thread);
      } finally {
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
  const showEmpty = messages.length === 0 && !sending && !isStreaming;

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
                <MessageBubble
                  key={m.id}
                  message={m}
                  isStreaming={
                    isStreaming &&
                    m.role === "assistant" &&
                    m.id === messages.at(-1)?.id
                  }
                />
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
              disabled={!input.trim() || sending || isStreaming}
              className="absolute bottom-2 right-2 h-9 w-9 rounded-xl"
              aria-label="Send message"
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          </div>
          <p className="mt-2 text-center text-[11px] text-muted-foreground">
            ⚠️ AI-generated responses may contain inaccuracies. Please verify
            important information before making decisions.
          </p>
        </form>
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  isStreaming = false,
}: {
  message: ChatMessage;
  isStreaming?: boolean;
}) {
  const isUser = message.role === "user";
  return (
    <li className={cn("flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
          <Sparkles className="h-4 w-4" />
        </div>
      )}
      <div className="flex flex-col gap-2 max-w-[85%]">
        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-[15px] leading-6",
            isUser
              ? "bg-primary text-primary-foreground whitespace-pre-wrap"
              : "bg-muted text-foreground prose-bubble",
          )}
        >
          {isUser ? (
            message.content
          ) : (
            <>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  // Headings
                  h1: ({ children }) => (
                    <h1 className="text-xl font-bold mb-3 mt-1 text-foreground">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-lg font-semibold mb-2 mt-4 text-foreground">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-base font-semibold mb-1 mt-3 text-foreground">
                      {children}
                    </h3>
                  ),
                  // Paragraphs
                  p: ({ children }) => (
                    <p className="mb-3 last:mb-0 leading-7">{children}</p>
                  ),
                  // Ordered list — numbered
                  ol: ({ children }) => (
                    <ol className="list-decimal list-outside ml-5 mb-3 space-y-2">
                      {children}
                    </ol>
                  ),
                  // Unordered list — bullets
                  ul: ({ children }) => (
                    <ul className="list-disc list-outside ml-5 mb-3 space-y-1">
                      {children}
                    </ul>
                  ),
                  li: ({ children }) => (
                    <li className="leading-7 pl-1">{children}</li>
                  ),
                  // Bold
                  strong: ({ children }) => (
                    <strong className="font-semibold text-foreground">
                      {children}
                    </strong>
                  ),
                  // Italic
                  em: ({ children }) => (
                    <em className="italic text-muted-foreground">{children}</em>
                  ),
                  // Inline code
                  code: ({ children, className }) => {
                    const isBlock = className?.includes("language-");
                    return isBlock ? (
                      <code
                        className={cn(
                          "block bg-background/80 border border-border/50 rounded-lg px-3 py-2 text-[13px] font-mono overflow-x-auto my-2",
                          className,
                        )}
                      >
                        {children}
                      </code>
                    ) : (
                      <code className="bg-background/80 border border-border/40 rounded px-1.5 py-0.5 text-[13px] font-mono">
                        {children}
                      </code>
                    );
                  },
                  // Code block wrapper
                  pre: ({ children }) => (
                    <pre className="bg-background/80 border border-border/50 rounded-xl p-4 overflow-x-auto my-3 text-[13px] font-mono">
                      {children}
                    </pre>
                  ),
                  // Blockquote
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-primary/40 pl-4 italic text-muted-foreground my-3">
                      {children}
                    </blockquote>
                  ),
                  // Horizontal rule
                  hr: () => <hr className="border-border/40 my-4" />,
                  // Tables
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-3">
                      <table className="w-full text-sm border-collapse">
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-background/60">{children}</thead>
                  ),
                  th: ({ children }) => (
                    <th className="border border-border/40 px-3 py-2 text-left font-semibold">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-border/40 px-3 py-2">
                      {children}
                    </td>
                  ),
                  // Images
                  img: (props) => (
                    <EnlargeableImage
                      src={props.src || ""}
                      alt={props.alt || ""}
                      className="max-w-full rounded-md border border-border/50 shadow-sm object-cover my-3"
                    />
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
              {/* Blinking cursor while this bubble is actively streaming */}
              {isStreaming && (
                <span
                  aria-hidden
                  style={{
                    display: "inline-block",
                    width: "2px",
                    height: "1em",
                    background: "currentColor",
                    marginLeft: "2px",
                    verticalAlign: "text-bottom",
                    animation: "embediq-blink 0.9s step-start infinite",
                  }}
                />
              )}
              <style>{`
              @keyframes embediq-blink {
                0%, 100% { opacity: 1; }
                50%       { opacity: 0; }
              }
            `}</style>
            </>
          )}
        </div>
        {message.sources && message.sources.length > 0 && (
          <div className="mt-2 flex flex-col gap-2">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
              Sources & Images
            </h4>

            {/* Text Sources */}
            <div className="flex flex-wrap gap-2 mb-1">
              {Array.from(
                new Map(
                  message.sources.map((s) => [
                    `${s.source}-${s.page_number}`,
                    s,
                  ]),
                ).values(),
              ).map((s, i) => (
                <span
                  key={i}
                  className="inline-flex items-center rounded-md bg-secondary px-2 py-1 text-xs font-medium text-secondary-foreground ring-1 ring-inset ring-secondary-foreground/10"
                >
                  {s.source}{" "}
                  {s.page_number && s.page_number !== "Unknown"
                    ? `(Page ${s.page_number})`
                    : ""}
                </span>
              ))}
            </div>

            {/* Images */}
            <div className="flex flex-wrap gap-2">
              {message.sources
                .flatMap((s) => s.image_urls || [])
                .filter((v, i, a) => a.indexOf(v) === i)
                .map((url, i) => (
                  <EnlargeableImage
                    key={i}
                    src={`http://localhost:8000${url}`}
                    alt="Source content"
                    className="max-w-[200px] rounded-md border border-border/50 shadow-sm object-cover"
                  />
                ))}
            </div>
          </div>
        )}
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

function EnlargeableImage({
  src,
  alt,
  className,
}: {
  src: string;
  alt?: string;
  className?: string;
}) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <img
          src={src}
          alt={alt || "Image"}
          className={cn(
            "cursor-pointer transition-opacity hover:opacity-90",
            className,
          )}
        />
      </DialogTrigger>
      <DialogContent className="max-w-4xl border-none bg-transparent p-0 shadow-none">
        <DialogTitle className="sr-only">Enlarged Image</DialogTitle>
        <DialogDescription className="sr-only">
          View full size image
        </DialogDescription>
        <img
          src={src}
          alt={alt || "Enlarged Image"}
          className="h-auto w-full max-h-[85vh] object-contain rounded-md"
        />
      </DialogContent>
    </Dialog>
  );
}
