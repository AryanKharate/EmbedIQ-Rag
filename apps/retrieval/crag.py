import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.conf import settings
from google import genai
from google.genai import types
from google.genai.errors import ServerError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

# Shared module-level client
_client = genai.Client(api_key=settings.GEMINI_API_KEY)

# Maximum concurrent grading threads across all requests
_MAX_GRADE_WORKERS = 4


class RelevanceGrade(BaseModel):
    relevant: bool
    confidence: float


@retry(
    wait=wait_random_exponential(multiplier=1, max=15),
    stop=stop_after_attempt(3),
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

    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=RelevanceGrade,
        ),
    )
    return RelevanceGrade.model_validate_json(response.text)


def grade_all_concurrent(query: str, chunk_texts: list[str]) -> list[RelevanceGrade]:
    """
    Grades multiple chunks in parallel using ThreadPoolExecutor.

    Concurrency is capped at _MAX_GRADE_WORKERS to prevent unbounded thread
    creation when multiple web requests invoke CRAG simultaneously.
    Failures in individual grades are logged and treated as "not relevant"
    rather than failing the entire batch.
    """
    if not chunk_texts:
        return []

    results: list[RelevanceGrade | None] = [None] * len(chunk_texts)
    with ThreadPoolExecutor(max_workers=_MAX_GRADE_WORKERS) as executor:
        future_to_idx = {
            executor.submit(grade_chunk, query, text): idx
            for idx, text in enumerate(chunk_texts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                logger.exception("CRAG grading failed for chunk %d, treating as not relevant", idx)
                results[idx] = RelevanceGrade(relevant=False, confidence=0.0)

    return results  # type: ignore[return-value]


@retry(
    wait=wait_random_exponential(multiplier=1, max=15),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ServerError),
)
def rewrite_for_search(query: str) -> str:
    """
    Rewrites a query that failed to retrieve good results, making it
    broader or more explicit for document search.
    """
    prompt = (
        "The following query didn't retrieve good results. Rewrite it to be clearer "
        f"and more specific for a document search: {query}"
    )
    response = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text.strip()


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

    # 1. Initial Retrieval
    chunks = search_fn(query)

    # 2. Deduplicate parents before grading to avoid wasting LLM calls
    deduped_chunks, deduped_texts = _deduplicate_by_parent(chunks)

    # 3. Grading
    grades = grade_all_concurrent(query, deduped_texts)

    relevant_chunks = [
        c for c, g in zip(deduped_chunks, grades)
        if g.relevant and g.confidence >= threshold
    ]

    if len(relevant_chunks) >= min_relevant:
        return relevant_chunks, "ok"

    # 4. Correction Phase
    rewritten_query = rewrite_for_search(query)
    retry_chunks = search_fn(rewritten_query)

    retry_deduped, retry_texts = _deduplicate_by_parent(retry_chunks)

    # Grade the retry chunks against the ORIGINAL query to ensure it answers the user's need
    retry_grades = grade_all_concurrent(query, retry_texts)

    retry_relevant = [
        c for c, g in zip(retry_deduped, retry_grades)
        if g.relevant and g.confidence >= threshold
    ]

    # Use the same threshold for both initial and retry paths
    if len(retry_relevant) >= min_relevant:
        return retry_relevant, "corrected"

    # 5. Hard Abstention
    return [], "insufficient"
