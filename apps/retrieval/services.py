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
"""

import logging
import re

from django.conf import settings
from fastembed import SparseTextEmbedding
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

logger = logging.getLogger(__name__)

# --- Shared clients (module-level singletons) ---
_genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
_qdrant_client = QdrantClient(url=settings.QDRANT_URL)

# BM25 model — loaded once, shared across all requests
_bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Pronoun/reference patterns that indicate a follow-up needing rewrite
_REFERENTIAL_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|"
    r"he|she|him|her|his|hers|the same|above|previous|"
    r"mentioned|said|earlier)\b",
    re.IGNORECASE,
)


def embed_query(text: str) -> list[float]:
    """Embed the user query with gemini-embedding-001 (RETRIEVAL_QUERY task type)."""
    result = _genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={
            "task_type": "RETRIEVAL_QUERY",
            "output_dimensionality": settings.EMBED_DIM,
        },
    )
    # Validate embedding output
    if not result.embeddings or len(result.embeddings) == 0:
        raise ValueError("Embedding API returned no embeddings")
    vec = result.embeddings[0].values
    if len(vec) != settings.EMBED_DIM:
        raise ValueError(f"Expected embedding dim {settings.EMBED_DIM}, got {len(vec)}")
    return vec


def embed_sparse(text: str) -> SparseVector:
    """
    Generate a BM25 sparse vector for the query using FastEmbed.

    Runs in-process — no extra server or API call required.
    Mirrors the same model used during ingestion so sparse scores are
    computed in the same vector space.
    """
    result = list(_bm25_model.embed([text]))[0]
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


def get_dense_query_vector(query: str) -> list[float]:
    """
    Returns the dense embedding for the query, applying HyDE if enabled.
    """
    if not settings.USE_HYDE:
        return embed_query(query)

    from .hyde import generate_hypothetical_answer

    hypothetical = generate_hypothetical_answer(query)
    hyde_vec = embed_query(hypothetical)

    if settings.HYDE_MODE == "replace":
        return hyde_vec

    # ensemble: average raw + HyDE vectors
    raw_vec = embed_query(query)
    return [(a + b) / 2 for a, b in zip(raw_vec, hyde_vec)]


def search_chunks(
    query: str, top_k: int | None = None, user_id: str | None = None
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
    """
    k = top_k if top_k is not None else settings.TOP_K
    prefetch_limit = k * 4

    dense_vec = get_dense_query_vector(query)
    sparse_vec = embed_sparse(query)

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    must_conditions = [FieldCondition(key="is_active", match=MatchValue(value=True))]
    if user_id:
        must_conditions.append(
            FieldCondition(key="user_id", match=MatchValue(value=user_id))
        )

    query_filter = Filter(must=must_conditions)

    hits = _qdrant_client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=prefetch_limit,
                filter=query_filter,
            ),
            Prefetch(
                query=dense_vec,
                using="dense",
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=prefetch_limit,  # fetch more before dedup
    ).points

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


def search_and_rerank(
    query: str, top_k: int | None = None, user_id: str | None = None
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

    candidates = search_chunks(
        query, top_k=settings.RERANK_CANDIDATE_LIMIT, user_id=user_id
    )
    reranked = reranker.rerank(query, candidates, top_k=settings.RERANK_CANDIDATE_LIMIT)

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
    recent = history[-6:]
    formatted = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in recent)
    prompt = (
        "Given the conversation below and a follow-up question, rewrite the "
        "follow-up as a fully standalone question that captures all necessary context. "
        "Output ONLY the rewritten question, nothing else.\n\n"
        f"Conversation:\n{formatted}\n\n"
        f"Follow-up: {original_question}\n\n"
        "Standalone question:"
    )
    response = _genai_client.models.generate_content(
        model="gemini-2.5-flash",  # fast + cheap — only used for rewriting
        contents=prompt,
    )
    rewritten = (response.text or "").strip()
    # Fall back to original if rewrite is empty
    if not rewritten:
        logger.warning("Rewrite returned empty result, using original question")
        return original_question
    return rewritten
