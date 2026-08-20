import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.conf import settings
from google.genai.errors import ServerError
from langsmith import traceable
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


from config.genai_client import utility_generate
from config.timing import copy_ctx, stage

logger = logging.getLogger(__name__)


class RelevanceGrade(BaseModel):
    relevant: bool
    confidence: float


@retry(
    wait=wait_random_exponential(
        multiplier=0.5, max=settings.GEMINI_UTILITY_RETRY_MAX_WAIT
    ),
    stop=stop_after_attempt(settings.GEMINI_UTILITY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(ServerError),
)
def grade_chunk(query: str, chunk_text: str) -> RelevanceGrade:
    """
    Uses an LLM to judge the relevance of a chunk to the user's query.
    Returns a structured Pydantic object indicating relevance and confidence.

    Only retries on ServerError (5xx). Client errors (4xx) are raised
    immediately since they will never succeed on retry.
    """
    prompt = f"Query: {query}\n\nRetrieved text:\n{chunk_text}\n\nIs this text relevant to answering the query?"

    response = utility_generate(
        prompt,
        temperature=0,
        response_mime_type="application/json",
        response_schema=RelevanceGrade,
    )

    return RelevanceGrade.model_validate_json(response.text)


@traceable(run_type="chain")
def grade_all_concurrent(query: str, chunk_texts: list[str]) -> list[RelevanceGrade]:
    """
    Grades multiple chunks in parallel using ThreadPoolExecutor.

    Concurrency is capped at settings.CRAG_MAX_GRADE_WORKERS to prevent
    unbounded thread creation when multiple web requests invoke CRAG
    simultaneously.
    Failures in individual grades are logged and treated as "not relevant"
    rather than failing the entire batch.
    """
    if not chunk_texts:
        return []

    results: list[RelevanceGrade | None] = [None] * len(chunk_texts)
    with stage("crag_grade", chunks=len(chunk_texts)):
        with ThreadPoolExecutor(max_workers=settings.CRAG_MAX_GRADE_WORKERS) as executor:
            # ThreadPoolExecutor does not propagate context, so each grade runs
            # inside a copy of the caller's context — otherwise worker threads
            # would log with a missing trace ID.
            future_to_idx = {
                executor.submit(copy_ctx().run, grade_chunk, query, text): idx
                for idx, text in enumerate(chunk_texts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(
                        "CRAG grading failed for chunk %d, treating as not relevant",
                        idx,
                    )
                    results[idx] = RelevanceGrade(relevant=False, confidence=0.0)

    return results  # type: ignore[return-value]


@retry(
    wait=wait_random_exponential(
        multiplier=0.5, max=settings.GEMINI_UTILITY_RETRY_MAX_WAIT
    ),
    stop=stop_after_attempt(settings.GEMINI_UTILITY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(ServerError),
)
def rewrite_for_search(query: str) -> str:
    """
    Rewrites a query that failed to retrieve good results, making it
    broader or more explicit for document search.

    Fails open to the original query: a failed rewrite here shouldn't crash
    the whole request when the retry search can just run on the un-rewritten
    query instead (same fail-open contract as rewrite_query() and
    generate_hypothetical_answer()).
    """
    prompt = (
        "The following query didn't retrieve good results. Rewrite it to be clearer "
        f"and more specific for a document search: {query}"
    )
    try:
        with stage("crag_rewrite"):
            response = utility_generate(prompt, temperature=0.3)
    except Exception as exc:
        logger.warning("CRAG rewrite failed (%s), using original query", exc)
        return query

    return (response.text or "").strip() or query


def _deduplicate_by_parent(chunks: list) -> tuple[list, list[str]]:
    """
    Deduplicate chunks by parent_text before grading so the same parent
    context is not graded (and paid for) multiple times.

    Returns:
        - deduped_chunks: list of unique-parent chunks
        - deduped_texts: corresponding parent/text strings for grading
    """
    seen_parents: set[str] = set()
    deduped_chunks = []
    deduped_texts = []
    for c in chunks:
        text = c.payload.get("parent_text") or c.payload.get("text")
        if text not in seen_parents:
            seen_parents.add(text)
            deduped_chunks.append(c)
            deduped_texts.append(text)
    return deduped_chunks, deduped_texts


@traceable(run_type="chain")
def corrective_retrieve(query: str, search_fn) -> tuple[list, str]:
    """
    CRAG Orchestrator:
    1. Search
    2. Deduplicate by parent text
    3. Grade
    4. If enough relevant, return chunks, 'ok'
    5. If not, rewrite query and retry search
    6. Grade new chunks (against original query)
    7. If enough relevant, return chunks, 'corrected'
    8. If still not enough, return [], 'insufficient'
    """
    min_relevant = settings.CRAG_MIN_RELEVANT_CHUNKS
    threshold = settings.CRAG_CONFIDENCE_THRESHOLD

    with stage("crag") as s:
        # 1. Initial Retrieval
        chunks = search_fn(query)

        # 2. Deduplicate parents before grading to avoid wasting LLM calls
        deduped_chunks, deduped_texts = _deduplicate_by_parent(chunks)

        # 3. Grading
        grades = grade_all_concurrent(query, deduped_texts)

        relevant_chunks = [
            c
            for c, g in zip(deduped_chunks, grades)
            if g.relevant and g.confidence >= threshold
        ]

        if len(relevant_chunks) >= min_relevant:
            s["status"] = "ok"
            s["relevant"] = len(relevant_chunks)
            return relevant_chunks, "ok"

        # 4. Correction Phase
        rewritten_query = rewrite_for_search(query)
        retry_chunks = search_fn(rewritten_query)

        retry_deduped, retry_texts = _deduplicate_by_parent(retry_chunks)

        # Grade the retry chunks against the ORIGINAL query to ensure it answers the user's need
        retry_grades = grade_all_concurrent(query, retry_texts)

        retry_relevant = [
            c
            for c, g in zip(retry_deduped, retry_grades)
            if g.relevant and g.confidence >= threshold
        ]

        # Use the same threshold for both initial and retry paths
        if len(retry_relevant) >= min_relevant:
            s["status"] = "corrected"
            s["relevant"] = len(retry_relevant)
            return retry_relevant, "corrected"

        # 5. Hard Abstention
        s["status"] = "insufficient"
        s["relevant"] = 0
        return [], "insufficient"
