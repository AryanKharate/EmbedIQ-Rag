"""
apps/generation/services.py

Prompt building + LLM generation with full multi-turn conversation support.
Calls apps.retrieval.services for query rewriting and vector search so
generation stays decoupled from Qdrant / embedding details.
"""
import logging
import time

from django.conf import settings
from google import genai
from google.genai import types

from apps.retrieval.services import rewrite_query, search_chunks, search_and_rerank

logger = logging.getLogger(__name__)

# --- Shared client (module-level singleton) ---
_genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)


def build_contents(query: str, chunks: list, history: list[dict]) -> list[dict]:
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
    
    for c in chunks:
        # Prefer parent_text (for new chunks), fallback to text (for old chunks)
        chunk_text = c.payload.get("parent_text") or c.payload.get("text")
        source = c.payload.get("source", "unknown")
        
        # Deduplicate to prevent stuffing the exact same parent context multiple times
        if chunk_text not in seen_texts:
            seen_texts.add(chunk_text)
            context_chunks.append(
                f"[Source: {source}]\n{chunk_text}"
            )
            
            # Fetch images for this document and page
            doc_id = c.payload.get("document_id")
            page_num = c.payload.get("page_number")
            image_urls = []
            if doc_id and page_num:
                images = DocumentImage.objects.filter(document_id=doc_id, page_number=page_num)
                image_urls = [img.image.url for img in images if img.image]

            sources.append({
                "source": source,
                "chunk_index": c.payload.get("chunk_index"),
                "parent_id": c.payload.get("parent_id"),
                "score": getattr(c, "score", None),
                "image_urls": image_urls,
            })
            
    context_text = "\n\n---\n\n".join(context_chunks)

    contents.append({
        "role": "user",
        "parts": [{"text": f"Context:\n{context_text}\n\nQuestion: {query}"}],
    })
    return contents, sources


SYSTEM_INSTRUCTION = (
    "You are a strict, accurate assistant that answers questions using ONLY "
    "the document context provided in each turn. "
    "STRICT RULE: If the answer to the question is not explicitly stated in the provided context, "
    "you MUST refuse to answer and state '(no answer exists)'. "
    "Do NOT use outside knowledge. Do NOT attempt to guess, extrapolate, or deduce the answer if the information is missing. "
    "Never fabricate information. "
    "You may reference previous conversation turns to give coherent answers."
)


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

    # Step 1: context-aware retrieval — rewrite before embedding
    t0 = time.time()
    search_query = rewrite_query(query, history)
    t_rewrite = time.time() - t0

    # Step 2: retrieve relevant chunks, with optional cross-encoder reranking and CRAG
    t0 = time.time()
    if settings.CRAG_ENABLED:
        from apps.retrieval.crag import corrective_retrieve
        search_fn = search_and_rerank if settings.RERANKER_ENABLED else search_chunks
        # Wrap search_fn to pass user_id
        def _scoped_search(q, **kwargs):
            return search_fn(q, user_id=user_id, **kwargs)
        chunks, crag_status = corrective_retrieve(search_query, _scoped_search)

        if crag_status == "insufficient":
            return "I don't have enough relevant information in the knowledge base to answer that confidently."
    else:
        if settings.RERANKER_ENABLED:
            chunks = search_and_rerank(search_query, user_id=user_id)
        else:
            chunks = search_chunks(search_query, user_id=user_id)
        crag_status = "ok"
    t_retrieve = time.time() - t0

    if not chunks:
        return "No relevant chunks found in the collection.", []

    # Step 3: build multi-turn contents list
    contents, sources = build_contents(query, chunks, history)

    # Step 4: generate with system instruction passed via config
    instruction = SYSTEM_INSTRUCTION
    if crag_status == "corrected":
        instruction += "\nNote: initial retrieval was weak; a corrected search was used."

    t0 = time.time()
    response = _genai_client.models.generate_content(
        model=gen_model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=instruction,
        ),
    )
    t_generate = time.time() - t0

    # Log stage timing for performance diagnostics
    logger.info(
        "Pipeline timing: rewrite=%.2fs, retrieve=%.2fs, generate=%.2fs",
        t_rewrite, t_retrieve, t_generate,
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
            logger.warning("Response blocked by safety filters: %s", response.prompt_feedback)
            return "I'm unable to answer that question due to content safety restrictions.", sources
        logger.warning("Model returned empty response for query: %s...", query[:50])
        return "I was unable to generate a response. Please try rephrasing your question.", sources

    return response.text, sources
