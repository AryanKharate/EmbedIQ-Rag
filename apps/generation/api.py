"""
apps/generation/api.py

Django-Ninja router exposing the single RAG endpoint with conversation support:
    POST /api/query  →  {"answer": "...", "session_id": "...", "sources": [...]}

Pass session_id from a previous response to continue a conversation thread.
Omit session_id (or pass null) to start a fresh session.

Authentication: JWT Bearer token required. All queries and sessions are
scoped to the authenticated user.
"""
from ninja import NinjaAPI, Schema
import logging

logger = logging.getLogger(__name__)

from apps.conversations.services import get_or_create_session, get_history, save_turn
from apps.generation.services import ask
from apps.retrieval.api import router as document_router
from apps.accounts.auth import jwt_auth

api = NinjaAPI(
    title="EmbedIQ API",
    description="Conversational Retrieval-Augmented Generation over your document corpus.",
    version="0.3.0",
)

api.add_router("/documents", document_router)


class QueryIn(Schema):
    question: str
    session_id: str | None = None  # omit to start a new conversation


class QueryOut(Schema):
    answer: str
    session_id: str  # always returned — pass it back to continue the thread
    sources: list[dict] = []


@api.post("/query", response=QueryOut, auth=jwt_auth, summary="Ask a question over your documents")
def query_endpoint(request, payload: QueryIn) -> QueryOut:
    """
    Runs the full conversational RAG pipeline scoped to the authenticated user:
    1. Load or create a conversation session (owned by this user)
    2. Fetch the last MAX_HISTORY_TURNS messages as context
    3. Rewrite the query using history (fixes vague follow-ups)
    4. Embed the rewritten query → retrieve top-K chunks from Qdrant (user-scoped)
    5. Build a multi-turn prompt (history + new context + question)
    6. Generate the answer with Gemini
    7. Persist both the user question and assistant answer to Postgres
    """
    user = request.auth

    # --- Input validation ---
    question = (payload.question or "").strip()
    if not question:
        return api.create_response(request, {"detail": "Question must not be empty."}, status=422)
    if len(question) > 4000:
        return api.create_response(
            request, {"detail": f"Question too long ({len(question)} chars, max 4000)."}, status=422
        )

    logger.info("Received query for user: %s, session: %s", user.id, payload.session_id)

    # 1. Session (scoped to user)
    session = get_or_create_session(payload.session_id, user=user)

    # 2. History
    history = get_history(session)

    # 3–6. Full conversational RAG (scoped to user's vectors)
    answer, sources = ask(question, history=history, user_id=str(user.id))

    # 7. Persist this exchange
    save_turn(session, "user", question)
    save_turn(session, "assistant", answer)

    logger.info("Successfully generated answer for user: %s, session: %s", user.id, session.id)

    return QueryOut(answer=answer, session_id=str(session.id), sources=sources)
