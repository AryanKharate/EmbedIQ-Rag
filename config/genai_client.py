"""
config/genai_client.py

One shared Gemini client for the whole request path.

Previously apps/generation/services.py, apps/retrieval/services.py,
apps/retrieval/hyde.py and apps/retrieval/crag.py each built their own
``genai.Client``. Four clients means four independent httpx connection pools,
so the rewrite → HyDE → embed → generate chain — four calls to the same host,
back to back — could not reuse a single TLS connection and paid a fresh
handshake in each pool.

It also owns the request timeouts. Every Gemini call in the query path goes
through a ``*_HTTP_OPTIONS`` below; without one, a hung call has no ceiling and
tenacity's exponential backoff (up to 15s x 3 attempts) turns a single slow
response into a half-minute request. The client-level default is the
generation timeout, so a call site that forgets to pass http_options is still
bounded rather than unbounded.

Timeouts are in **milliseconds** (google-genai's HttpOptions.timeout unit).

Note: apps/retrieval/ingest_service.py deliberately keeps its own client —
batch document embedding runs for tens of seconds and needs a completely
different timeout budget from anything on the query path.
"""

import logging

from django.conf import settings
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from langsmith.wrappers import wrap_gemini

logger = logging.getLogger(__name__)

# Per-call option objects, built once — they are immutable config.
#
# UTILITY:  query rewrite, HyDE, CRAG grading. These sit in front of retrieval,
#           so every second they burn is a second before the user sees a token.
#           Kept tight; callers fail open when they trip.
# EMBED:    query embedding. Similar position, slightly looser (a 3072-dim
#           response is ~60KB of JSON).
# GENERATE: the user-facing answer. Loosest, because this one is the payload —
#           it covers opening the stream and receiving the first chunk.
UTILITY_HTTP_OPTIONS = types.HttpOptions(timeout=settings.GEMINI_UTILITY_TIMEOUT_MS)
EMBED_HTTP_OPTIONS = types.HttpOptions(timeout=settings.GEMINI_EMBED_TIMEOUT_MS)
GENERATE_HTTP_OPTIONS = types.HttpOptions(timeout=settings.GEMINI_GENERATE_TIMEOUT_MS)

# wrap_gemini traces every generate_content/generate_content_stream call on this
# client as its own "llm" run in LangSmith, with token usage attached — including
# the streaming path, where the trace closes once the returned generator is fully
# consumed. embed_content is untouched by it.
client = wrap_gemini(
    genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options=GENERATE_HTTP_OPTIONS,
    )
)


# --- Utility calls (rewrite / HyDE / CRAG grading) ---------------------------

# Not every Gemini model lets thinking be switched off, and the ones that don't
# reject the request outright. Rather than have every utility call 400 (and
# silently fail open to "no rewrite, no HyDE") against such a model, the first
# rejection flips this flag and the call is retried without it. One wasted
# request per process, then the correct behaviour forever.
_thinking_supported = True


def utility_generate_config(**overrides) -> types.GenerateContentConfig:
    """
    Shared config for the pre-retrieval utility calls.

    Two latency knobs live here:
      - http_options: a hard request timeout, so a hung call can't sit on the
        critical path indefinitely.
      - thinking_config: these calls are mechanical text transforms, not
        reasoning tasks. Letting the model emit thinking tokens first delays
        the response with no benefit to a query rewrite or a HyDE passage.
        Set UTILITY_THINKING_BUDGET to -1 to leave the model default in place.
    """
    config = {"http_options": UTILITY_HTTP_OPTIONS, **overrides}
    if settings.UTILITY_THINKING_BUDGET >= 0 and _thinking_supported:
        config["thinking_config"] = types.ThinkingConfig(
            thinking_budget=settings.UTILITY_THINKING_BUDGET
        )
    return types.GenerateContentConfig(**config)


def _rejected_thinking_config(exc: ClientError) -> bool:
    """True if this 400 looks like the model refusing thinking_config."""
    return "thinking" in str(exc).lower()


def utility_generate(contents, *, model: str | None = None, **config_kwargs):
    """
    Run a utility generate_content call with the shared timeout + thinking
    settings, transparently retrying once without thinking_config if the model
    turns out not to support disabling it.
    """
    global _thinking_supported

    target = model or settings.UTILITY_MODEL
    try:
        return client.models.generate_content(
            model=target,
            contents=contents,
            config=utility_generate_config(**config_kwargs),
        )
    except ClientError as exc:
        if not (_thinking_supported and _rejected_thinking_config(exc)):
            raise
        logger.warning(
            "Model %s rejected thinking_config (%s); disabling it for this process",
            target,
            exc,
        )
        _thinking_supported = False
        return client.models.generate_content(
            model=target,
            contents=contents,
            config=utility_generate_config(**config_kwargs),
        )
