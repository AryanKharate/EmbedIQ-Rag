"""
apps/retrieval/services.py

Embedding, vector-search, and query-rewriting functions.
Clients are initialized once at module load using Django settings so they
are shared across all requests (no reconnect overhead).

v3 Changes:
  - Smart rewrite routing: skip LLM rewrite for self-contained questions
  - Embedding output validation
  - Added embed_sparse() using FastEmbed BM25 (in-process, no extra server)
  - Qdrant prefetch + server-side RRF fusion (dense + sparse in one round-trip)

v4 (latency):
  - prepare_query() merges the query rewrite and the HyDE hypothetical answer into
    ONE Gemini call. They used to be two serial utility calls (~0.96s + ~1.53s)
    that both had to finish before the first embedding could start.
  - HYDE_MODE="ensemble" embeds the raw query and the hypothetical answer
    concurrently instead of back-to-back — they are independent, and each is a
    ~0.55s network round-trip.
  - Every utility call fails open: a timeout degrades retrieval quality for one
    request instead of adding a retry backoff to the critical path.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from django.conf import settings
from fastembed import SparseTextEmbedding
from google.genai.errors import ServerError
from langsmith import traceable
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue, SparseVector
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from config.genai_client import (
    EMBED_HTTP_OPTIONS,
    client as _genai_client,
    utility_generate,
)
from config.timing import copy_ctx, stage
from . import vector_store

logger = logging.getLogger(__name__)

# BM25 model — loaded once, shared across all requests
_bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Pronoun/reference patterns that indicate a follow-up needing rewrite
_REFERENTIAL_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|"
    r"he|she|him|her|his|hers|the same|above|previous|"
    r"mentioned|said|earlier)\b",
    re.IGNORECASE,
)


@traceable(run_type="embedding", metadata={"ls_provider": "google_genai", "ls_model_name": "gemini-embedding-001"})
@retry(
    wait=wait_random_exponential(multiplier=1, max=15),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ServerError),
)
def embed_query(text: str) -> list[float]:
    """Embed the user query with gemini-embedding-001 (RETRIEVAL_QUERY task type)."""
    with stage("embed_dense"):
        result = _genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config={
                "task_type": "RETRIEVAL_QUERY",
                "output_dimensionality": settings.EMBED_DIM,
                "http_options": EMBED_HTTP_OPTIONS,
            },
        )
    # Validate embedding output
    if not result.embeddings or len(result.embeddings) == 0:
        raise ValueError("Embedding API returned no embeddings")
    vec = result.embeddings[0].values
    if len(vec) != settings.EMBED_DIM:
        raise ValueError(f"Expected embedding dim {settings.EMBED_DIM}, got {len(vec)}")
    return vec


@traceable(run_type="embedding", metadata={"ls_provider": "fastembed", "ls_model_name": "Qdrant/bm25"})
def embed_sparse(text: str) -> SparseVector:
    """
    Generate a BM25 sparse vector for the query using FastEmbed.

    Runs in-process — no extra server or API call required.
    Mirrors the same model used during ingestion so sparse scores are
    computed in the same vector space.
    """
    with stage("embed_sparse"):
        result = list(_bm25_model.embed([text]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


@traceable(run_type="chain")
def get_dense_query_vector(
    query: str, hypothetical: str | None = None, use_hyde: bool | None = None
) -> list[float]:
    """
    Returns the dense embedding for the query, applying HyDE if enabled.

    hypothetical: a HyDE passage already produced upstream by prepare_query(),
        which generates it in the same call as the query rewrite. Passing it in
        avoids a second utility round-trip. When None (HyDE-only path, or the
        CRAG correction path re-searching a rewritten query) one is generated
        here, as before.
    use_hyde: per-request override of settings.USE_HYDE. None falls back to
        the server default.
    """
    effective_hyde = settings.USE_HYDE if use_hyde is None else use_hyde
    if not effective_hyde:
        return embed_query(query)

    if hypothetical is None:
        hypothetical = _hypothetical_or_none(query)

    # HyDE failed (timeout / empty). Fall back to embedding the raw query —
    # slightly weaker retrieval beats failing the request.
    if not hypothetical:
        return embed_query(query)

    if settings.HYDE_MODE == "replace":
        return embed_query(hypothetical)

    # ensemble: average raw + HyDE vectors.
    # The two embed calls are independent ~0.55s network round-trips, so they
    # run concurrently rather than back-to-back. copy_ctx() carries the trace ID
    # (and the LangSmith run contextvar) into the workers.
    with stage("embed_dense_pair"):
        with ThreadPoolExecutor(max_workers=2) as executor:
            hyde_future = executor.submit(copy_ctx().run, embed_query, hypothetical)
            raw_future = executor.submit(copy_ctx().run, embed_query, query)
            hyde_vec = hyde_future.result()
            raw_vec = raw_future.result()

    return [(a + b) / 2 for a, b in zip(raw_vec, hyde_vec)]


def _hypothetical_or_none(query: str) -> str | None:
    """Generate a HyDE passage, returning None instead of raising. Retrieval
    quality is a nice-to-have; a failed utility call must not take the request
    down with it."""
    from .hyde import generate_hypothetical_answer

    try:
        return generate_hypothetical_answer(query)
    except Exception as exc:
        logger.warning("HyDE generation failed (%s), embedding raw query", exc)
        return None


@traceable(run_type="retriever")
def search_chunks(
    query: str,
    top_k: int | None = None,
    user_id: str | None = None,
    hypothetical: str | None = None,
    use_hyde: bool | None = None,
) -> list:
    """
    Hybrid search: dense (semantic) + sparse (BM25 keyword) with RRF fusion.

    Uses Qdrant's Universal Query API with two prefetch branches:
      - "sparse": BM25 keyword search — strong at exact term matches
      - "dense":  Gemini semantic search — strong at paraphrase/concept queries

    Both branches run in a single network round-trip. Qdrant fuses the
    ranked lists using Reciprocal Rank Fusion (RRF) server-side and returns
    the top_k best results.

    Prefetch limit = top_k * 4 so RRF has enough candidates from each branch
    before truncating to the final top_k.

    user_id: when provided, restricts results to vectors uploaded by that user.
    hypothetical: optional pre-generated HyDE passage (see prepare_query()).
    use_hyde: per-request override of settings.USE_HYDE. None falls back to
        the server default.
    """
    k = top_k if top_k is not None else settings.TOP_K
    prefetch_limit = k * 4

    dense_vec = get_dense_query_vector(
        query, hypothetical=hypothetical, use_hyde=use_hyde
    )
    sparse_vec = embed_sparse(query)

    must_conditions = [FieldCondition(key="is_active", match=MatchValue(value=True))]
    if user_id:
        must_conditions.append(
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        )

    query_filter = Filter(must=must_conditions)

    with stage("qdrant_search", limit=prefetch_limit) as s:
        hits = vector_store.hybrid_search(
            dense_vec, sparse_vec, query_filter, prefetch_limit
        )
        s["hits"] = len(hits)

    # Deduplicate by parent_id (or parent_text fallback) to avoid redundant contexts
    seen_parents = set()
    deduped_hits = []
    for hit in hits:
        parent_key = hit.payload.get("parent_id") or hit.payload.get("parent_text")
        if parent_key not in seen_parents:
            seen_parents.add(parent_key)
            deduped_hits.append(hit)
            if len(deduped_hits) >= k:
                break

    return deduped_hits


@traceable(run_type="retriever")
def search_and_rerank(
    query: str,
    top_k: int | None = None,
    user_id: str | None = None,
    hypothetical: str | None = None,
    use_hyde: bool | None = None,
) -> list:
    """
    Full retrieval pipeline with cross-encoder reranking.

    Step 1 — Hybrid search (dense + sparse RRF):
        Fetches RERANK_CANDIDATE_LIMIT results from Qdrant.
        A wider pool gives the reranker enough candidates to re-order.

    Step 2 — BGE cross-encoder rerank:
        The reranker reads (query, passage) pairs jointly and scores them
        by actual textual relevance, not just vector proximity.
        Returns the top TOP_K results.

    The output format is identical to search_chunks() — a list of ScoredPoint
    objects — so generation/services.py.build_contents() needs no changes.

    user_id: when provided, restricts search to that user's vectors.
    """
    from .reranker import reranker

    with stage("search_and_rerank") as s:
        candidates = search_chunks(
            query,
            top_k=settings.RERANK_CANDIDATE_LIMIT,
            user_id=user_id,
            hypothetical=hypothetical,
            use_hyde=use_hyde,
        )
        reranked = reranker.rerank(
            query, candidates, top_k=settings.RERANK_CANDIDATE_LIMIT
        )
        s["candidates"] = len(candidates)

    # We deduplicate again just in case the reranker reordered things such that
    # lower-ranked children from a higher-ranked parent get pushed down, though
    # search_chunks already did a first-pass dedup.
    k = top_k if top_k is not None else settings.TOP_K
    seen_parents = set()
    deduped_hits = []
    for hit in reranked:
        parent_key = hit.payload.get("parent_id") or hit.payload.get("parent_text")
        if parent_key not in seen_parents:
            seen_parents.add(parent_key)
            deduped_hits.append(hit)
            if len(deduped_hits) >= k:
                break
    return deduped_hits


def _looks_referential(question: str) -> bool:
    """
    Quick heuristic to decide if a follow-up question is referential
    (uses pronouns/references to previous context) and needs rewriting,
    or is already self-contained.

    This avoids an unnecessary LLM rewrite call for standalone questions.
    """
    # Very short questions are likely follow-ups ("what about X?")
    if len(question) < 30:
        return True
    # Check for referential pronouns / phrases
    return bool(_REFERENTIAL_PATTERN.search(question))


@retry(
    wait=wait_random_exponential(
        multiplier=0.5, max=settings.GEMINI_UTILITY_RETRY_MAX_WAIT
    ),
    stop=stop_after_attempt(settings.GEMINI_UTILITY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(ServerError),
)
def _call_rewrite_llm(prompt: str):
    """The actual Gemini call behind rewrite_query() — kept as its own
    function purely so the early-return (no history / self-contained
    question) paths in rewrite_query() never touch the network. Tracing and
    token usage for this call come for free from the wrap_gemini() client."""
    return utility_generate(prompt)


def _format_history(history: list[dict], turns: int = 6) -> str:
    """Render the last `turns` messages as `Role: content` lines."""
    return "\n".join(
        f"{t['role'].capitalize()}: {t['content']}" for t in history[-turns:]
    )


def rewrite_query(original_question: str, history: list[dict]) -> str:
    """
    Use Gemini Flash to rewrite a follow-up question into a fully self-contained
    search query using the conversation history.

    This is the key that makes retrieval work on vague follow-ups like
    "what about its speed?" — it gets rewritten to "How fast is DNS resolution?"
    before hitting Qdrant, so the embedding matches the right chunks.

    Optimizations:
    - Returns the original question unchanged when there is no history.
    - Skips the LLM call when the question looks self-contained (no referential
      pronouns), saving one Gemini call on most standalone questions.
    """
    if not history:
        return original_question

    # Skip LLM rewrite if the question looks self-contained
    if not _looks_referential(original_question):
        logger.debug(
            "Skipping rewrite — question looks self-contained: %s...",
            original_question[:50],
        )
        return original_question

    # Use up to the last 6 turns (3 exchanges) to keep the rewriting prompt small
    prompt = (
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a fully standalone question that captures all necessary context. "
        "Output ONLY the rewritten question, nothing else.\n\n"
        f"Conversation:\n{_format_history(history)}\n\n"
        f"Follow-up: {original_question}\n\n"
        "Standalone question:"
    )
    try:
        with stage("rewrite_llm"):
            response = _call_rewrite_llm(prompt)
    except Exception as exc:
        # Fail open: searching on the raw follow-up retrieves something, which
        # beats making the user wait out a retry budget for nothing.
        logger.warning("Rewrite failed (%s), using original question", exc)
        return original_question

    rewritten = (response.text or "").strip()
    # Fall back to original if rewrite is empty
    if not rewritten:
        logger.warning("Rewrite returned empty result, using original question")
        return original_question
    return rewritten


# --- Merged pre-retrieval planning ------------------------------------------


@dataclass(frozen=True)
class PreparedQuery:
    """What retrieval needs before it can start: the string to search on, and
    (when HyDE is on) the hypothetical passage to embed alongside it."""

    search_query: str
    hypothetical: str | None = None


class _QueryPlan(BaseModel):
    """Structured output of the merged rewrite + HyDE call."""

    standalone_question: str
    hypothetical_answer: str


_PLAN_PROMPT = (
    "You prepare search queries for a document retrieval system.\n\n"
    "Given the conversation and the follow-up question below, produce:\n"
    "1. standalone_question — the follow-up rewritten as a fully self-contained "
    "question that captures all necessary context from the conversation.\n"
    "2. hypothetical_answer — a short, plausible answer to that standalone "
    "question, written as if it appeared in a reference document. Do not hedge "
    "or say you're unsure; write the answer directly. It is embedded for "
    "similarity search, so it does not need to be factually correct.\n\n"
    "Conversation:\n{history}\n\n"
    "Follow-up: {question}"
)


@retry(
    wait=wait_random_exponential(
        multiplier=0.5, max=settings.GEMINI_UTILITY_RETRY_MAX_WAIT
    ),
    stop=stop_after_attempt(settings.GEMINI_UTILITY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(ServerError),
)
def _call_plan_llm(original_question: str, history: list[dict]) -> _QueryPlan:
    """One Gemini call producing both the rewrite and the HyDE passage."""
    response = utility_generate(
        _PLAN_PROMPT.format(
            history=_format_history(history), question=original_question
        ),
        temperature=0.3,
        response_mime_type="application/json",
        response_schema=_QueryPlan,
    )
    return _QueryPlan.model_validate_json(response.text)


@traceable(run_type="chain")
def prepare_query(
    original_question: str, history: list[dict], use_hyde: bool | None = None
) -> PreparedQuery:
    """
    Do all the pre-retrieval LLM work in as few round-trips as possible.

    The old path ran rewrite_query() and then generate_hypothetical_answer() as
    two serial utility calls in front of retrieval — ~2.5s before the first
    embedding could even start. When both are needed they are now a single call
    (plan_query), since the model needs the same conversation history for each.

    Falls back to the individual calls when only one is needed, and to the raw
    question when the merged call fails.

    use_hyde: per-request override of settings.USE_HYDE. None falls back to
        the server default.
    """
    needs_rewrite = bool(history) and _looks_referential(original_question)
    needs_hyde = settings.USE_HYDE if use_hyde is None else use_hyde

    if needs_rewrite and needs_hyde:
        try:
            with stage("query_plan", mode=settings.HYDE_MODE):
                plan = _call_plan_llm(original_question, history)
        except Exception as exc:
            logger.warning(
                "Merged query plan failed (%s), falling back to raw question", exc
            )
            return PreparedQuery(search_query=original_question)

        question = (plan.standalone_question or "").strip() or original_question
        return PreparedQuery(
            search_query=question,
            hypothetical=(plan.hypothetical_answer or "").strip() or None,
        )

    if needs_rewrite:
        return PreparedQuery(search_query=rewrite_query(original_question, history))

    if needs_hyde:
        # No rewrite needed, so the HyDE call stands alone. Generated here
        # rather than inside search so a failure degrades instead of raising.
        return PreparedQuery(
            search_query=original_question,
            hypothetical=_hypothetical_or_none(original_question),
        )

    return PreparedQuery(search_query=original_question)
