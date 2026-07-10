import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { MessageSquarePlus } from "lucide-react";
import { useEffect } from "react";

import { Button } from "@/components/ui/button";
import { createThread, loadThreads, saveThreads } from "@/lib/chat-store";
import { useHydrated } from "@/hooks/use-hydrated";

export const Route = createFileRoute("/")({
  component: Index,
});

function Index() {
  const hydrated = useHydrated();
  const navigate = useNavigate();

  // If there is already a most-recent thread, jump to it.
  useEffect(() => {
    if (!hydrated) return;
    const threads = loadThreads();
    if (threads.length > 0) {
      navigate({
        to: "/c/$threadId",
        params: { threadId: threads[0].id },
        replace: true,
      });
    }
  }, [hydrated, navigate]);

  const startNewChat = () => {
    const t = createThread();
    const next = [t, ...loadThreads()];
    saveThreads(next);
    window.dispatchEvent(new Event("rag:threads-updated"));
    navigate({ to: "/c/$threadId", params: { threadId: t.id } });
  };

  return (
    <div className="flex h-full items-center justify-center px-6">
      <div className="max-w-md text-center">
        <h1 className="text-3xl font-semibold tracking-tight">
          Chat with your documents
        </h1>
        <p className="mt-3 text-sm text-muted-foreground">
          Start a new conversation and toggle which documents the assistant
          should reference from the sidebar.
        </p>
        <Button onClick={startNewChat} size="lg" className="mt-6 gap-2">
          <MessageSquarePlus className="h-4 w-4" />
          Start a new chat
        </Button>
      </div>
    </div>
  );
}
