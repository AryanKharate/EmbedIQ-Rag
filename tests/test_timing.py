"""
tests/test_timing.py

Unit tests for the timing instrumentation added in config/timing.py and
config/middleware.py. These are plain unit tests (no DB, no live server) —
they exercise the context managers and middleware directly.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from django.http import HttpResponse, StreamingHttpResponse
from django.test import RequestFactory, override_settings

from config.middleware import RequestTimingMiddleware
from config.timing import copy_ctx, stage, trace_id_var


def test_stage_logs_duration_and_fields(caplog):
    with caplog.at_level(logging.INFO, logger="embediq.timing"):
        with stage("my_stage", foo="bar") as s:
            s["extra"] = 1

    records = [r for r in caplog.records if r.name == "embediq.timing"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "stage" in message
    assert "name=my_stage" in message
    assert "foo=bar" in message
    assert "extra=1" in message
    assert "duration_ms=" in message


def test_stage_logs_and_reraises_on_exception(caplog):
    with caplog.at_level(logging.INFO, logger="embediq.timing"):
        with pytest.raises(ValueError):
            with stage("failing_stage"):
                raise ValueError("boom")

    records = [r for r in caplog.records if r.name == "embediq.timing"]
    assert len(records) == 1
    assert "error=ValueError" in records[0].getMessage()


@override_settings(TIMING_ENABLED=False)
def test_stage_is_noop_when_disabled(caplog):
    with caplog.at_level(logging.INFO, logger="embediq.timing"):
        with stage("disabled_stage") as s:
            s["foo"] = "bar"

    records = [r for r in caplog.records if r.name == "embediq.timing"]
    assert records == []


def test_trace_id_propagates_into_thread_pool_via_copy_ctx():
    token = trace_id_var.set("test-trace-123")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            seen = executor.submit(copy_ctx().run, trace_id_var.get).result()
        assert seen == "test-trace-123"

        # Without copy_ctx, a plain submit does NOT see the parent's contextvar —
        # this is the bug the crag.py fix addresses.
        with ThreadPoolExecutor(max_workers=1) as executor:
            unset = executor.submit(trace_id_var.get).result()
        assert unset == "-"
    finally:
        trace_id_var.reset(token)


@override_settings(TIMING_SKIP_PATHS=("/health/",))
def test_request_timing_middleware_skips_configured_paths(caplog):
    calls = []

    def get_response(request):
        calls.append(request.path)
        return HttpResponse("ok")

    middleware = RequestTimingMiddleware(get_response)
    request = RequestFactory().get("/health/live")

    with caplog.at_level(logging.INFO, logger="embediq.timing"):
        response = middleware(request)

    assert response.status_code == 200
    assert calls == ["/health/live"]
    records = [r for r in caplog.records if r.name == "embediq.timing"]
    assert records == []


def test_request_timing_middleware_logs_plain_response(caplog):
    def get_response(request):
        return HttpResponse("ok")

    middleware = RequestTimingMiddleware(get_response)
    request = RequestFactory().get("/api/documents/")

    with caplog.at_level(logging.INFO, logger="embediq.timing"):
        middleware(request)

    records = [r for r in caplog.records if r.name == "embediq.timing"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "request" in message
    assert "path=/api/documents/" in message
    assert "status=200" in message
    assert "duration_ms=" in message


def test_request_timing_middleware_times_streaming_response_after_drain(caplog):
    """
    Regression test: a StreamingHttpResponse must not be timed at construction —
    get_response() returns immediately, before the SSE generator body runs. The
    'request' event must only be logged once the stream is fully consumed, and
    ttft_ms must be captured on the first chunk.
    """

    def event_stream():
        yield b"chunk-1"
        yield b"chunk-2"

    def get_response(request):
        return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

    middleware = RequestTimingMiddleware(get_response)
    request = RequestFactory().post("/api/query")

    with caplog.at_level(logging.INFO, logger="embediq.timing"):
        response = middleware(request)
        # No 'request' event yet — the generator hasn't been consumed.
        records = [r for r in caplog.records if r.name == "embediq.timing"]
        assert records == []

        chunks = list(response.streaming_content)

    assert chunks == [b"chunk-1", b"chunk-2"]
    records = [r for r in caplog.records if r.name == "embediq.timing"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "streamed=1" in message
    assert "ttft_ms=" in message
    assert "duration_ms=" in message
