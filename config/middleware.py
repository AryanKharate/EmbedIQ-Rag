"""
config/middleware.py

Request tracing middleware — assigns a unique trace ID to every request
and injects it into the logging context so all log lines from a single
request can be correlated.

The trace ID is also returned as an X-Request-ID response header.
"""
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


class RequestTraceMiddleware:
    """
    Assigns a unique trace_id to every request.

    - Reads X-Request-ID from incoming headers (for upstream load balancers)
    - Falls back to generating a new UUID4
    - Stores it in thread-local storage so the log filter can access it
    - Returns it as X-Request-ID response header
    """

    _local = threading.local()

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Prefer upstream trace ID, generate one if absent
        trace_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        self._local.trace_id = trace_id

        response = self.get_response(request)
        response["X-Request-ID"] = trace_id
        return response

    @classmethod
    def get_trace_id(cls) -> str:
        return getattr(cls._local, "trace_id", "-")


class TraceIdFilter(logging.Filter):
    """
    Logging filter that injects trace_id into every log record.

    Use %(trace_id)s in your log format string to include it.
    """

    def filter(self, record):
        record.trace_id = RequestTraceMiddleware.get_trace_id()
        return True
