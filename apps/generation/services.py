"""
apps/generation/services.py

Prompt building + LLM generation with full multi-turn conversation support.
Calls apps.retrieval.services for query rewriting and vector search so
generation stays decoupled from Qdrant / embedding details.
"""

import json
import logging
import time
from collections import defaultdict

from django.conf import settings
from django.db.models import Q
from google.genai import types
from google.genai.errors import ServerError
from langsmith import trace, traceable
from tenacity import (
    Retrying,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from apps.retrieval.services import prepare_query, search_chunks, search_and_rerank
from config.genai_client import GENERATE_HTTP_OPTIONS, client as _genai_client
from config.timing import stage, timed

logger = logging.getLogger(__name__)


def build_contents(
    query: str, chunks: list, history: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Build the multi-turn contents list for the Gemini API.

    Gemini requires strict alternating user/model turns.
    History is already stored as user/assistant; we map assistant → model here.
    The new user turn appends the retrieved context chunks + current question.
    """
    role_map = {"user": "user", "assistant": "model"}

    # Previous turns (alternating user / model)
    contents = [
        {"role": role_map.get(t["role"], t["role"]), "parts": [{"text": t["content"]}]}
        for t in history
    ]

    # New user turn: stuffed context + question
    seen_texts = set()
    context_chunks = []
    sources = []

    from apps.retrieval.models import DocumentImage

    # Fetch images for every (document, page) pair referenced by these chunks
    # in a single query, instead of one query per chunk.
    page_keys = {
        (c.payload.get("document_id"), c.payload.get("page_number"))
        for c in chunks
        if c.payload.get("document_id") and c.payload.get("page_number")
    }
    images_by_page: dict[tuple[str, int], list[str]] = defaultdict(list)
    if page_keys:
        page_filter = Q()
        for doc_id, page_num in page_keys:
            page_filter |= Q(document_id=doc_id, page_number=page_num)
        for img in DocumentImage.objects.filter(page_filter):
            if img.image:
                images_by_page[(str(img.document_id), img.page_number)].append(
                    img.image.url
                )

    for c in chunks:
        # Prefer parent_text (for new chunks), fallback to text (for old chunks)
        chunk_text = c.payload.get("parent_text") or c.payload.get("text")
        source = c.payload.get("source", "unknown")
        page_num = c.payload.get("page_number")

        # Deduplicate to prevent stuffing the exact same parent context multiple times
        if chunk_text not in seen_texts:
            seen_texts.add(chunk_text)
            context_chunks.append(f"[Source: {source}, Page: {page_num}]\n{chunk_text}")

            doc_id = c.payload.get("document_id")
            image_urls = images_by_page.get((doc_id, page_num), [])

            sources.append(
                {
                    "source": source,
                    "chunk_index": c.payload.get("chunk_index"),
                    "parent_id": c.payload.get("parent_id"),
                    "score": getattr(c, "score", None),
                    "image_urls": image_urls,
                    "page_number": page_num,
                }
            )

    context_text = "\n\n---\n\n".join(context_chunks)

    contents.append(
        {
            "role": "user",
            "parts": [{"text": f"Context:\n{context_text}\n\nQuestion: {query}"}],
        }
    )
    return contents, sources


SYSTEM_INSTRUCTION = (
    "You are a strict, accurate assistant that answers questions using ONLY "
    "the document context provided in each turn. "
    "STRICT RULE: If the answer to the question is not explicitly stated in the provided context, "
    "you MUST refuse to answer and state '(no answer exists)'. "
    "Do NOT use outside knowledge. Do NOT attempt to guess, extrapolate, or deduce the answer if the information is missing. "
    "Never fabricate information. "
    "When providing facts or information from the context, you MUST include in-line citations in the format (Source: <source>, Page: <page_number>). "
    "You may reference previous conversation turns to give coherent answers."
)


def _generation_config(instruction: str) -> types.GenerateContentConfig:
    """Config for the user-facing answer call. http_options bounds the request
    so a hung call can't sit on the critical path indefinitely — this one gets
    the loosest budget of the three, since it *is* the payload."""
    return types.GenerateContentConfig(
        system_instruction=instruction,
        http_options=GENERATE_HTTP_OPTIONS,
    )


@retry(
    wait=wait_random_exponential(
        multiplier=1, max=settings.GEMINI_GENERATE_RETRY_MAX_WAIT
    ),
    stop=stop_after_attempt(settings.GEMINI_GENERATE_MAX_ATTEMPTS),
    retry=retry_if_exception_type(ServerError),
)
def _generate_answer(gen_model: str, contents: list[dict], instruction: str):
    """Single non-streaming Gemini call — safe to retry as a whole since
    nothing has been returned to the caller until it succeeds."""
    return _genai_client.models.generate_content(
        model=gen_model,
        contents=contents,
        config=_generation_config(instruction),
    )


def _start_stream_with_retry(gen_model: str, contents: list[dict], instruction: str):
    """
    Open a Gemini streaming response, retrying only the connection / first-chunk
    step on a transient ServerError. Once a chunk has been produced it may
    already be on its way to the client over SSE, so a failure *after* that
    point is deliberately NOT retried here (that would risk silently
    re-sending duplicate tokens) — see ask_stream()'s except block, which
    reports a clean interruption instead.

    Returns (stream, first_chunk_or_None) — the caller must yield
    first_chunk before continuing to iterate `stream`.
    """
    retryer = Retrying(
        wait=wait_random_exponential(
            multiplier=1, max=settings.GEMINI_GENERATE_RETRY_MAX_WAIT
        ),
        stop=stop_after_attempt(settings.GEMINI_GENERATE_MAX_ATTEMPTS),
        retry=retry_if_exception_type(ServerError),
        reraise=True,
    )

    def _open():
        stream = _genai_client.models.generate_content_stream(
            model=gen_model,
            contents=contents,
            config=_generation_config(instruction),
        )
        first_chunk = next(stream, None)
        return stream, first_chunk

    return retryer(_open)


def _plan_and_retrieve(
    query: str, history: list[dict], user_id: str | None
) -> tuple[list, str]:
    """
    Everything between the user's question and a ranked list of chunks.

    Shared by ask() and ask_stream(), which had drifted into two copies of the
    same block. prepare_query() collapses the query rewrite and the HyDE
    passage into a single Gemini call, and hands the passage straight to
    search so it isn't generated twice.
    """
    with stage("prepare_query"):
        prepared = prepare_query(query, history)

    with stage("retrieve") as s:
        if settings.CRAG_ENABLED:
            from apps.retrieval.crag import corrective_retrieve

            search_fn = search_and_rerank if settings.RERANKER_ENABLED else search_chunks

            # The pre-generated HyDE passage belongs to the *initial* query
            # only. CRAG's correction path searches a different, rewritten
            # query, so it must generate its own (search_chunks does that when
            # hypothetical is None).
            def _scoped_search(q, **kwargs):
                kwargs.setdefault(
                    "hypothetical",
                    prepared.hypothetical if q == prepared.search_query else None,
                )
                return search_fn(q, user_id=user_id, **kwargs)

            chunks, crag_status = corrective_retrieve(
                prepared.search_query, _scoped_search
            )
        else:
            search_fn = search_and_rerank if settings.RERANKER_ENABLED else search_chunks
            chunks = search_fn(
                prepared.search_query,
                user_id=user_id,
                hypothetical=prepared.hypothetical,
            )
            crag_status = "ok"
        s["chunks"] = len(chunks)
        s["crag_status"] = crag_status

    return chunks, crag_status


@traceable(run_type="chain")
@timed("ask")
def ask(
    query: str,
    history: list[dict] | None = None,
    model: str | None = None,
    user_id: str | None = None,
) -> tuple[str, list[dict]]:
    """
    Full conversational RAG pipeline:
      1. Rewrite the query using conversation history (so vague follow-ups work)
      2. Embed the rewritten query and search Qdrant (scoped to user_id if provided)
      3. Build a multi-turn contents list (history + new context + question)
      4. Generate the answer with Gemini using a system instruction

    Returns the answer text. Handles blocked or empty model responses gracefully.
    """
    history = history or []
    gen_model = model or settings.GEN_MODEL

    # Steps 1–2: plan the query (rewrite + HyDE in one call) and retrieve
    chunks, crag_status = _plan_and_retrieve(query, history, user_id)

    if crag_status == "insufficient":
        return (
            "I don't have enough relevant information in the knowledge base to answer that confidently.",
            [],
        )

    if not chunks:
        return "No relevant chunks found in the collection.", []

    # Step 3: build multi-turn contents list
    contents, sources = build_contents(query, chunks, history)

    # Step 4: generate with system instruction passed via config
    instruction = SYSTEM_INSTRUCTION
    if crag_status == "corrected":
        instruction += (
            "\nNote: initial retrieval was weak; a corrected search was used."
        )

    with stage("generate", model=gen_model) as s:
        response = _generate_answer(gen_model, contents, instruction)
        if getattr(response, "usage_metadata", None):
            s["input_tokens"] = getattr(
                response.usage_metadata, "prompt_token_count", None
            )
            s["output_tokens"] = getattr(
                response.usage_metadata, "candidates_token_count", None
            )

    # Log token usage if available
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        logger.info(
            "Token usage: input=%s, output=%s, total=%s",
            getattr(usage, "prompt_token_count", "?"),
            getattr(usage, "candidates_token_count", "?"),
            getattr(usage, "total_token_count", "?"),
        )

    # Handle blocked or empty responses
    if response.text is None or response.text.strip() == "":
        # Check if the response was blocked by safety filters
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            logger.warning(
                "Response blocked by safety filters: %s", response.prompt_feedback
            )
            return (
                "I'm unable to answer that question due to content safety restrictions.",
                sources,
            )
        logger.warning("Model returned empty response for query: %s...", query[:50])
        return (
            "I was unable to generate a response. Please try rephrasing your question.",
            sources,
        )

    return response.text, sources


def ask_stream(
    query: str,
    session,
    history: list[dict] | None = None,
    model: str | None = None,
    user_id: str | None = None,
):
    """
    Streaming version of the full conversational RAG pipeline.
    Yields SSE-formatted strings:
      1. data: {"type": "sources", "sources": [...]}\n\n   (before any text)
      2. data: {"type": "token",   "text": "..."}\n\n    (one per Gemini chunk)
      3. data: {"type": "done",    "session_id": "..."}\n\n  (final event)

    After the stream ends it saves both turns to Postgres (same as ask()).
    """
    from apps.conversations.services import save_turn

    history = history or []
    gen_model = model or settings.GEN_MODEL

    # ask() is traced via @traceable, but this is a generator (yields SSE
    # chunks as they arrive), so it can't use the decorator the same way —
    # trace() as a context manager lets tracing span the yields directly.
    with trace(
        "ask_stream",
        run_type="chain",
        inputs={"query": query, "user_id": user_id, "model": gen_model},
    ) as rt:
        # Steps 1–2: plan the query (rewrite + HyDE in one call) and retrieve
        chunks, crag_status = _plan_and_retrieve(query, history, user_id)

        rt.metadata["crag_status"] = crag_status

        if crag_status == "insufficient":
            msg = "I don't have enough relevant information in the knowledge base to answer that confidently."
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': str(session.id)})}\n\n"
            save_turn(session, "user", query)
            save_turn(session, "assistant", msg)
            rt.end(outputs={"answer": msg, "sources": []})
            return

        if not chunks:
            msg = "No relevant chunks found in the collection."
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'text': msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': str(session.id)})}\n\n"
            save_turn(session, "user", query)
            save_turn(session, "assistant", msg)
            rt.end(outputs={"answer": msg, "sources": []})
            return

        # Step 3: build multi-turn contents list
        contents, sources = build_contents(query, chunks, history)

        # Emit sources immediately so the frontend can show cards before text starts
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        # Step 4: stream from Gemini
        instruction = SYSTEM_INSTRUCTION
        if crag_status == "corrected":
            instruction += (
                "\nNote: initial retrieval was weak; a corrected search was used."
            )

        full_answer_parts: list[str] = []

        # _start_stream_with_retry() calls the wrap_gemini()-wrapped
        # generate_content_stream(), which traces this whole call as one
        # "llm" run (with token usage) once the generator below is fully
        # consumed — no manual tracing needed here.
        with stage("generate", model=gen_model) as s:
            started = time.perf_counter()
            try:
                stream, first_chunk = _start_stream_with_retry(
                    gen_model, contents, instruction
                )

                def _iter_chunks():
                    if first_chunk is not None:
                        yield first_chunk
                    yield from stream

                for chunk in _iter_chunks():
                    if chunk.text:
                        # Time-to-first-token — the latency the user actually feels,
                        # as opposed to the full generation time.
                        if "ttft_ms" not in s:
                            s["ttft_ms"] = round((time.perf_counter() - started) * 1000, 1)
                        full_answer_parts.append(chunk.text)
                        yield f"data: {json.dumps({'type': 'token', 'text': chunk.text})}\n\n"
            except Exception as exc:
                logger.error("Streaming generation error: %s", exc)
                error_msg = "Streaming was interrupted. Please try again."
                yield f"data: {json.dumps({'type': 'token', 'text': error_msg})}\n\n"
                full_answer_parts.append(error_msg)
                s["error"] = type(exc).__name__

        full_answer = "".join(full_answer_parts)
        if not full_answer:
            full_answer = (
                "I was unable to generate a response. Please try rephrasing your question."
            )

        # Persist both turns to DB
        save_turn(session, "user", query)
        save_turn(session, "assistant", full_answer)

        rt.end(outputs={"answer": full_answer, "sources": sources})

        # Final done event carries the session_id the frontend needs for follow-ups
        yield f"data: {json.dumps({'type': 'done', 'session_id': str(session.id)})}\n\n"
