"""
tests/test_query_planning.py

Unit tests for the pre-retrieval latency work in apps/retrieval/services.py:

  - prepare_query() collapses the query rewrite and the HyDE passage into ONE
    Gemini call when both are needed (they used to be two serial calls).
  - get_dense_query_vector() embeds the raw query and the HyDE passage
    concurrently in "ensemble" mode instead of back-to-back.
  - Every utility call fails open, so a timeout degrades retrieval for one
    request rather than failing it.

All Gemini calls are mocked — these are plain unit tests, no network, no DB.
"""

import threading
import time
from unittest.mock import patch

import pytest
from django.conf import settings
from django.test import override_settings
from google.genai.errors import ServerError

from apps.retrieval import services
from apps.retrieval.services import PreparedQuery, _QueryPlan

HISTORY = [
    {"role": "user", "content": "What is DNS?"},
    {"role": "assistant", "content": "DNS maps hostnames to IPs."},
]
FOLLOW_UP = "what about its speed?"  # referential -> needs a rewrite
STANDALONE = "Explain the full DNS resolution procedure for a cold cache lookup"


def _plan(question="How fast is DNS resolution?", answer="DNS resolves in ~20ms."):
    return _QueryPlan(standalone_question=question, hypothetical_answer=answer)


# --- prepare_query: call-count is the whole point ----------------------------


@override_settings(USE_HYDE=True)
def test_rewrite_and_hyde_are_a_single_call():
    """The core latency fix: two serial utility round-trips become one."""
    with patch.object(services, "_call_plan_llm", return_value=_plan()) as plan_llm:
        with patch.object(services, "_call_rewrite_llm") as rewrite_llm:
            with patch(
                "apps.retrieval.hyde.generate_hypothetical_answer"
            ) as hyde_llm:
                prepared = services.prepare_query(FOLLOW_UP, HISTORY)

    assert plan_llm.call_count == 1
    assert rewrite_llm.call_count == 0, "separate rewrite call should be gone"
    assert hyde_llm.call_count == 0, "separate HyDE call should be gone"
    assert prepared == PreparedQuery(
        search_query="How fast is DNS resolution?",
        hypothetical="DNS resolves in ~20ms.",
    )


@override_settings(USE_HYDE=False)
def test_rewrite_only_when_hyde_disabled():
    with patch.object(services, "_call_plan_llm") as plan_llm:
        with patch.object(services, "_call_rewrite_llm") as rewrite_llm:
            rewrite_llm.return_value.text = "How fast is DNS resolution?"
            prepared = services.prepare_query(FOLLOW_UP, HISTORY)

    assert plan_llm.call_count == 0
    assert rewrite_llm.call_count == 1
    assert prepared.search_query == "How fast is DNS resolution?"
    assert prepared.hypothetical is None


@override_settings(USE_HYDE=True)
def test_hyde_only_when_question_is_self_contained():
    """No history / self-contained question -> no rewrite, so no merged call."""
    with patch.object(services, "_call_plan_llm") as plan_llm:
        with patch(
            "apps.retrieval.hyde.generate_hypothetical_answer",
            return_value="A resolver walks the delegation chain.",
        ):
            prepared = services.prepare_query(STANDALONE, HISTORY)

    assert plan_llm.call_count == 0
    assert prepared.search_query == STANDALONE
    assert prepared.hypothetical == "A resolver walks the delegation chain."


@override_settings(USE_HYDE=False)
def test_no_llm_calls_at_all_on_first_turn():
    with patch.object(services, "_call_plan_llm") as plan_llm:
        with patch.object(services, "_call_rewrite_llm") as rewrite_llm:
            prepared = services.prepare_query(STANDALONE, [])

    assert plan_llm.call_count == 0
    assert rewrite_llm.call_count == 0
    assert prepared == PreparedQuery(search_query=STANDALONE, hypothetical=None)


# --- fail-open ---------------------------------------------------------------


@override_settings(USE_HYDE=True)
def test_plan_failure_falls_back_to_raw_question():
    """A timeout must degrade retrieval, not fail the request."""
    with patch.object(services, "_call_plan_llm", side_effect=TimeoutError("timeout")):
        prepared = services.prepare_query(FOLLOW_UP, HISTORY)

    assert prepared == PreparedQuery(search_query=FOLLOW_UP, hypothetical=None)


@override_settings(USE_HYDE=False)
def test_rewrite_failure_falls_back_to_raw_question():
    error = ServerError(503, {"error": {"message": "unavailable"}})
    with patch.object(services, "_call_rewrite_llm", side_effect=error):
        assert services.rewrite_query(FOLLOW_UP, HISTORY) == FOLLOW_UP


@override_settings(USE_HYDE=True, HYDE_MODE="replace")
def test_hyde_failure_embeds_raw_query():
    with patch(
        "apps.retrieval.hyde.generate_hypothetical_answer",
        side_effect=RuntimeError("boom"),
    ):
        with patch.object(services, "embed_query", return_value=[1.0]) as embed:
            vec = services.get_dense_query_vector("what is dns")

    assert vec == [1.0]
    embed.assert_called_once_with("what is dns")


# --- concurrent embedding ----------------------------------------------------


@override_settings(USE_HYDE=True, HYDE_MODE="ensemble")
def test_ensemble_embeds_concurrently_and_averages():
    """Both embeds must be in flight at once — serial execution would double
    the ~0.55s round-trip on every query."""
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def slow_embed(text):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return [0.0, 2.0] if text == "hypothetical" else [2.0, 4.0]

    with patch.object(services, "embed_query", side_effect=slow_embed):
        started = time.perf_counter()
        vec = services.get_dense_query_vector("query", hypothetical="hypothetical")
        elapsed = time.perf_counter() - started

    assert peak == 2, "the two embed calls ran back-to-back, not concurrently"
    assert elapsed < 0.09, f"took {elapsed:.3f}s — looks serial"
    assert vec == [1.0, 3.0]  # element-wise mean of the two vectors


@override_settings(USE_HYDE=True, HYDE_MODE="replace")
def test_replace_mode_embeds_only_the_hypothetical():
    with patch.object(services, "embed_query", return_value=[1.0]) as embed:
        services.get_dense_query_vector("query", hypothetical="hypothetical")

    embed.assert_called_once_with("hypothetical")


@override_settings(USE_HYDE=True)
def test_prepared_hypothetical_is_not_regenerated_during_search():
    """search_chunks() must reuse the passage prepare_query() already paid for."""
    with patch(
        "apps.retrieval.hyde.generate_hypothetical_answer"
    ) as hyde_llm:
        with patch.object(services, "embed_query", return_value=[1.0]) as embed:
            services.get_dense_query_vector("query", hypothetical="already made")

    assert hyde_llm.call_count == 0
    embed.assert_any_call("already made")


# --- retry budget ------------------------------------------------------------


def test_utility_calls_retry_at_most_twice():
    """Utility retries are capped so a 5xx can't add a long backoff in front
    of retrieval."""
    error = ServerError(503, {"error": {"message": "unavailable"}})
    with patch.object(services, "utility_generate", side_effect=error) as generate:
        with pytest.raises(ServerError):
            services._call_rewrite_llm.retry_with(reraise=True)("prompt")

    assert generate.call_count == settings.GEMINI_UTILITY_MAX_ATTEMPTS == 2
