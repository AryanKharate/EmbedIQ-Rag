"""
apps/generation/api.py

Django-Ninja router exposing the single RAG endpoint with conversation support:
    POST /api/query  →  {"answer": "...", "session_id": "..."}

Pass session_id from a previous response to continue a conversation thread.
Omit session_id (or pass null) to start a fresh session.
"""
from ninja import NinjaAPI, Schema
import logging

logger = logging.getLogger(__name__)

from apps.conversations.services import get_or_create_session, get_history, save_turn
from apps.generation.services import ask
from apps.retrieval.api import router as document_router

api = NinjaAPI(
    title="EmbedIQ API",
    description="Conversational Retrieval-Augmented Generation over your document corpus.",
    version="0.2.0",
)

api.add_router("/documents", document_router)


class QueryIn(Schema):
    question: str
    session_id: str | None = None  # omit to start a new conversation


class QueryOut(Schema):
    answer: str
    session_id: str  # always returned — pass it back to continue the thread
    sources: list[dict] = []


@api.post("/query", response=QueryOut, summary="Ask a question over the corpus")
def query_endpoint(request, payload: QueryIn) -> QueryOut:
    """
    Runs the full conversational RAG pipeline:
    1. Load or create a conversation session
    2. Fetch the last MAX_HISTORY_TURNS messages as context
    3. Rewrite the query using history (fixes vague follow-ups)
    4. Embed the rewritten query → retrieve top-K chunks from Qdrant
    5. Build a multi-turn prompt (history + new context + question)
    6. Generate the answer with Gemini
    7. Persist both the user question and assistant answer to Postgres
    """
    # --- Input validation ---
    question = (payload.question or "").strip()
    if not question:
        return api.create_response(request, {"detail": "Question must not be empty."}, status=422)
    if len(question) > 4000:
        return api.create_response(
            request, {"detail": f"Question too long ({len(question)} chars, max 4000)."}, status=422
        )

    logger.info("Received query for session: %s", payload.session_id)

    # 1. Session
    session = get_or_create_session(payload.session_id)

    # 2. History
    history = get_history(session)

    # 3–6. Full conversational RAG
    answer, sources = ask(question, history=history)

    # 7. Persist this exchange
    save_turn(session, "user", question)
    save_turn(session, "assistant", answer)

    logger.info("Successfully generated answer for session: %s", session.id)

    return QueryOut(answer=answer, session_id=str(session.id), sources=sources)
