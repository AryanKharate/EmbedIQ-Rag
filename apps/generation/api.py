"""
apps/generation/api.py

Django-Ninja router exposing the single RAG endpoint with conversation support:
    POST /api/query  →  {"answer": "...", "session_id": "...", "sources": [...]}

Pass session_id from a previous response to continue a conversation thread.
Omit session_id (or pass null) to start a fresh session.

Authentication: JWT Bearer token required. All queries and sessions are
scoped to the authenticated user.
"""

import logging

from ninja import NinjaAPI, Schema
from django.http import StreamingHttpResponse

from apps.conversations.services import get_or_create_session, get_history
from apps.generation.services import ask_stream
from apps.retrieval.api import router as document_router
from apps.accounts.auth import jwt_auth

logger = logging.getLogger(__name__)

api = NinjaAPI(
    title="EmbedIQ API",
    description="Conversational Retrieval-Augmented Generation over your document corpus.",
    version="0.3.0",
)

api.add_router("/documents", document_router)


class QueryIn(Schema):
    question: str
    session_id: str | None = None  # omit to start a new conversation


@api.post(
    "/query", auth=jwt_auth, summary="Ask a question over your documents (SSE stream)"
)
def query_endpoint(request, payload: QueryIn):
    """
    Runs the full conversational RAG pipeline and streams the response as
    Server-Sent Events (SSE). Clients receive three event types:

      data: {"type": "sources", "sources": [...]}   — emitted first
      data: {"type": "token",   "text": "..."}       — one per Gemini chunk
      data: {"type": "done",    "session_id": "..."}  — final event

    Pipeline steps (scoped to the authenticated user):
    1. Load or create a conversation session
    2. Fetch the last MAX_HISTORY_TURNS messages as context
    3. Rewrite the query using history
    4. Embed → retrieve top-K chunks from Qdrant (user-scoped)
    5. Build multi-turn prompt
    6. Stream answer tokens from Gemini
    7. Persist both turns to Postgres after stream completes
    """
    user = request.auth

    # --- Input validation ---
    question = (payload.question or "").strip()
    if not question:
        return api.create_response(
            request, {"detail": "Question must not be empty."}, status=422
        )
    if len(question) > 4000:
        return api.create_response(
            request,
            {"detail": f"Question too long ({len(question)} chars, max 4000)."},
            status=422,
        )

    logger.info(
        "Received streaming query for user: %s, session: %s",
        user.id,
        payload.session_id,
    )

    # 1. Session (scoped to user)
    session = get_or_create_session(payload.session_id, user=user)

    # 2. History
    history = get_history(session)

    # 3–7. Streaming RAG pipeline — yields SSE events
    def event_stream():
        yield from ask_stream(
            query=question,
            session=session,
            history=history,
            user_id=str(user.id),
        )

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # tell nginx not to buffer this response
    return response
