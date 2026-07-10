import { createFileRoute } from "@tanstack/react-router";

import { ChatView } from "@/components/chat-view";

export const Route = createFileRoute("/c/$threadId")({
  component: ThreadPage,
});

function ThreadPage() {
  const { threadId } = Route.useParams();
  return <ChatView key={threadId} threadId={threadId} />;
}
