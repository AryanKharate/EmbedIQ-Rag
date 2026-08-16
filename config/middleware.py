"""
config/middleware.py

Request tracing and timing middleware.

RequestTraceMiddleware assigns a unique trace ID to every request and injects
it into the logging context so all log lines from a single request can be
correlated. The trace ID is also returned as an X-Request-ID response header.

RequestTimingMiddleware measures how long each request took and logs it to the
timing log (see config/timing.py). Streaming responses are handled specially —
see the note on _timed_stream below.
"""

import logging
import time
import uuid

from django.conf import settings
from django.http import StreamingHttpResponse

from config.timing import log_event, request_ctx_var, trace_id_var

logger = logging.getLogger(__name__)


class RequestTraceMiddleware:
    """
    Assigns a unique trace_id to every request.

    - Reads X-Request-ID from incoming headers (for upstream load balancers)
    - Falls back to generating a new UUID4
    - Stores it in a ContextVar so the log filter can access it
    - Returns it as X-Request-ID response header

    A ContextVar (not thread-local) is used so the value is reset per request
    rather than lingering on a recycled worker thread, and so it can be carried
    into thread pools and async contexts.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Prefer upstream trace ID, generate one if absent
        trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = trace_id_var.set(trace_id)

        try:
            response = self.get_response(request)
            response["X-Request-ID"] = trace_id
            return response
        finally:
            trace_id_var.reset(token)

    @classmethod
    def get_trace_id(cls) -> str:
        return trace_id_var.get()


class RequestTimingMiddleware:
    """
    Logs one `request` timing event per HTTP request.

    Sits after RequestTraceMiddleware so the trace ID is already set.

    Streaming responses (the SSE /api/query endpoint) need special handling:
    get_response() returns as soon as the StreamingHttpResponse object is
    *constructed*, long before the generator body runs. Timing only that would
    report a few milliseconds for a multi-second query. Instead the streaming
    content is wrapped so the clock stops when the generator is exhausted, and
    time-to-first-token is recorded on the first chunk.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.skip_paths = tuple(getattr(settings, "TIMING_SKIP_PATHS", ()))
        self.slow_ms = getattr(settings, "SLOW_REQUEST_MS", 3000)

    def __call__(self, request):
        if not getattr(settings, "TIMING_ENABLED", True) or request.path.startswith(
            self.skip_paths
        ):
            return self.get_response(request)

        ctx_token = request_ctx_var.set({"path": request.path})
        started = time.perf_counter()
        try:
            response = self.get_response(request)
        except Exception as exc:
            self._log(request, started, status=500, error=type(exc).__name__)
            raise
        finally:
            request_ctx_var.reset(ctx_token)

        # The user is only resolved once the view (and its auth) has run.
        user = self._resolve_user(request)

        if isinstance(response, StreamingHttpResponse):
            response.streaming_content = self._timed_stream(
                response.streaming_content, request, started, response, user
            )
        else:
            self._log(request, started, status=response.status_code, user=user)

        return response

    def _timed_stream(self, stream, request, started, response, user):
        """Wrap a streaming body, logging once the stream is fully consumed."""
        ttft_ms = None
        try:
            for chunk in stream:
                if ttft_ms is None:
                    ttft_ms = round((time.perf_counter() - started) * 1000, 1)
                yield chunk
        finally:
            # Runs on normal completion *and* on client disconnect, so a
            # cancelled stream is still accounted for.
            self._log(
                request,
                started,
                status=response.status_code,
                user=user,
                ttft_ms=ttft_ms,
                streamed=1,
            )

    @staticmethod
    def _resolve_user(request):
        """User id from ninja's request.auth, falling back to the session user."""
        auth = getattr(request, "auth", None)
        if auth is not None and getattr(auth, "id", None) is not None:
            return auth.id
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_authenticated", False):
            return user.id
        return None

    def _log(self, request, started, status, user=None, **fields):
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        level = logging.WARNING if duration_ms > self.slow_ms else logging.INFO
        log_event(
            "request",
            level=level,
            method=request.method,
            path=request.path,
            status=status,
            user=user,
            duration_ms=duration_ms,
            **fields,
        )


class TraceIdFilter(logging.Filter):
    """
    Logging filter that injects trace_id into every log record.

    Use %(trace_id)s in your log format string to include it.
    """

    def filter(self, record):
        record.trace_id = trace_id_var.get()
        return True
