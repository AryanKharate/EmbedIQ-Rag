"""
apps/retrieval/hyde.py

HyDE — generate a hypothetical answer and embed *that* instead of the raw
query, so the embedding matches the shape of the target documents.

The standalone call below is only used when there is nothing to merge it with
(no rewrite needed, or the merged plan call failed). When a rewrite is also
required, apps/retrieval/services.prepare_query() produces the rewrite and the
hypothetical answer in a *single* Gemini call instead of two serial ones —
back-to-back utility calls were ~2.5s of the pre-retrieval budget.
"""

import logging

from django.conf import settings
from google.genai.errors import ServerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from config.genai_client import utility_generate
from config.timing import stage

logger = logging.getLogger(__name__)

HYDE_PROMPT = (
    "Write a short, plausible answer to this question, as if it appeared "
    "in a reference document. Do not hedge or say you're unsure — just write "
    "the answer directly.\n\nQuestion: {query}"
)


@retry(
    wait=wait_random_exponential(
        multiplier=0.5, max=settings.GEMINI_UTILITY_RETRY_MAX_WAIT
    ),
    stop=stop_after_attempt(settings.GEMINI_UTILITY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(ServerError),
)
def generate_hypothetical_answer(query: str) -> str:
    """
    Generates a plausible, hypothetical answer to the query as if it were
    pulled from a reference document. We embed this answer instead of the query
    so the embedding matches the structure and style of the target documents.

    Raises on failure — callers in services.py fail open to the raw query
    rather than failing the whole request over a retrieval-quality nicety.
    """
    # Timed inside the retry wrapper, so each attempt reports its own duration.
    with stage("hyde", mode=settings.HYDE_MODE):
        response = utility_generate(
            HYDE_PROMPT.format(query=query),
            temperature=0.3,
            max_output_tokens=settings.HYDE_MAX_OUTPUT_TOKENS,
        )

    return (response.text or "").strip()
