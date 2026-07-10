import { Link, useNavigate, useRouterState } from "@tanstack/react-router";
import { FileText, MessageSquarePlus, Trash2, Upload, Loader2 } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import {
  createThread,
  loadThreads,
  saveThreads,
  type ChatThread,
} from "@/lib/chat-store";
import { useHydrated } from "@/hooks/use-hydrated";
import {
  useDocuments,
  useUploadDocument,
  useToggleDocument,
  useDeleteDocument,
} from "@/lib/use-documents";

export function AppSidebar() {
  const hydrated = useHydrated();
  const navigate = useNavigate();
  const currentPath = useRouterState({ select: (r) => r.location.pathname });
  const activeThreadId = currentPath.startsWith("/c/")
    ? currentPath.slice(3)
    : null;

  const [threads, setThreads] = useState<ChatThread[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── Chat threads (still localStorage) ──
  useEffect(() => {
    if (!hydrated) return;
    setThreads(loadThreads());
    const onUpdate = () => setThreads(loadThreads());
    window.addEventListener("rag:threads-updated", onUpdate);
    return () => window.removeEventListener("rag:threads-updated", onUpdate);
  }, [hydrated]);

  // ── Documents (real API) ──
  const { data: docs = [], isLoading: docsLoading } = useDocuments();
  const uploadMutation = useUploadDocument();
  const toggleMutation = useToggleDocument();
  const deleteMutation = useDeleteDocument();

  const handleNewChat = useCallback(() => {
    const t = createThread();
    const next = [t, ...loadThreads()];
    saveThreads(next);
    setThreads(next);
    window.dispatchEvent(new Event("rag:threads-updated"));
    navigate({ to: "/c/$threadId", params: { threadId: t.id } });
  }, [navigate]);

  const handleDeleteThread = useCallback(
    (id: string) => {
      const next = loadThreads().filter((t) => t.id !== id);
      saveThreads(next);
      setThreads(next);
      window.dispatchEvent(new Event("rag:threads-updated"));
      if (activeThreadId === id) {
        navigate({ to: "/" });
      }
    },
    [activeThreadId, navigate],
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    uploadMutation.mutate(file);
    e.target.value = "";
  };

  const activeDocCount = docs.filter((d) => d.is_active).length;

  return (
    <Sidebar>
      <SidebarHeader className="gap-3 px-3 pt-3">
        <div className="flex items-center gap-2 px-1">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
            <span className="text-sm font-semibold">R</span>
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold leading-none">Ragbot</span>
            <span className="text-xs text-muted-foreground">
              Chat with your documents
            </span>
          </div>
        </div>
        <Button
          onClick={handleNewChat}
          variant="outline"
          className="w-full justify-start gap-2"
        >
          <MessageSquarePlus className="h-4 w-4" />
          New chat
        </Button>
      </SidebarHeader>

      <SidebarContent className="px-1">
        {/* ── Conversations ── */}
        <SidebarGroup>
          <SidebarGroupLabel>Conversations</SidebarGroupLabel>
          <SidebarGroupContent>
            {threads.length === 0 ? (
              <p className="px-2 py-6 text-center text-xs text-muted-foreground">
                No conversations yet.
              </p>
            ) : (
              <SidebarMenu>
                {threads.map((t) => {
                  const isActive = t.id === activeThreadId;
                  return (
                    <SidebarMenuItem key={t.id}>
                      <div
                        className={cn(
                          "group/thread flex items-center gap-1 rounded-md",
                          isActive && "bg-sidebar-accent",
                        )}
                      >
                        <Link
                          to="/c/$threadId"
                          params={{ threadId: t.id }}
                          className={cn(
                            "flex-1 truncate rounded-md px-2 py-2 text-sm outline-none transition-colors",
                            isActive
                              ? "text-sidebar-accent-foreground"
                              : "text-sidebar-foreground hover:bg-sidebar-accent/60",
                          )}
                        >
                          {t.title}
                        </Link>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.preventDefault();
                            handleDeleteThread(t.id);
                          }}
                          aria-label="Delete conversation"
                          className="mr-1 rounded p-1 text-muted-foreground opacity-0 transition hover:bg-destructive/10 hover:text-destructive group-hover/thread:opacity-100"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            )}
          </SidebarGroupContent>
        </SidebarGroup>

        {/* ── Documents ── */}
        <SidebarGroup>
          <SidebarGroupLabel className="flex items-center justify-between pr-2">
            <span>Documents</span>
            {!docsLoading && (
              <span className="text-[10px] font-normal text-muted-foreground">
                {activeDocCount}/{docs.length} active
              </span>
            )}
          </SidebarGroupLabel>
          <SidebarGroupContent>
            {/* Upload button */}
            <div className="mb-1 px-1">
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf,.md"
                className="hidden"
                onChange={handleFileChange}
              />
              <Button
                variant="ghost"
                size="sm"
                className="w-full justify-start gap-2 text-xs text-muted-foreground hover:text-foreground"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploadMutation.isPending}
              >
                {uploadMutation.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Upload className="h-3.5 w-3.5" />
                )}
                {uploadMutation.isPending ? "Uploading…" : "Upload document"}
              </Button>
            </div>

            {docsLoading ? (
              <div className="space-y-1 px-2">
                {[1, 2, 3].map((i) => (
                  <div
                    key={i}
                    className="h-10 animate-pulse rounded-md bg-muted"
                  />
                ))}
              </div>
            ) : docs.length === 0 ? (
              <p className="px-2 py-4 text-center text-xs text-muted-foreground">
                No documents yet. Upload one above.
              </p>
            ) : (
              <ul className="flex flex-col gap-0.5 px-1">
                {docs.map((doc) => (
                  <li
                    key={doc.id}
                    className={cn(
                      "group/doc flex items-center gap-2 rounded-md px-2 py-2 text-sm transition-colors",
                      doc.is_active
                        ? "text-sidebar-foreground"
                        : "text-muted-foreground",
                    )}
                  >
                    <FileText
                      className={cn(
                        "h-4 w-4 shrink-0",
                        doc.is_active
                          ? "text-primary"
                          : "text-muted-foreground/60",
                      )}
                    />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs font-medium">
                        {doc.filename}
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        {new Date(doc.created_at).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <Switch
                        checked={doc.is_active}
                        onCheckedChange={(v) =>
                          toggleMutation.mutate({ id: doc.id, is_active: v })
                        }
                        aria-label={`Toggle ${doc.filename}`}
                        disabled={toggleMutation.isPending}
                      />
                      <button
                        type="button"
                        onClick={() => deleteMutation.mutate(doc.id)}
                        aria-label={`Delete ${doc.filename}`}
                        disabled={deleteMutation.isPending}
                        className="rounded p-1 text-muted-foreground opacity-0 transition hover:bg-destructive/10 hover:text-destructive group-hover/doc:opacity-100"
                      >
                        <Trash2 className="h-3 w-3" />
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="px-3 pb-3">
        <div className="rounded-md border border-dashed border-sidebar-border px-3 py-2 text-[11px] leading-relaxed text-muted-foreground">
          Toggle documents to control what the assistant can reference.
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
