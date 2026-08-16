"""
config/timing.py

Timing instrumentation shared by every entry point and pipeline stage.

Two event kinds are emitted on the ``embediq.timing`` logger, both as
key=value tails so they can be read by eye and parsed with awk/cut:

    request method=POST path=/api/query status=200 user=1 duration_ms=6213 ttft_ms=1402
    stage name=retrieve duration_ms=287 user=1 chunks=12

They land in ``logs/timing.log`` (dedicated, rotating) as well as the console
and ``logs/embediq.log``, so timing lines stay inline with the surrounding
request trace. See the LOGGING config in config/settings.py.

The trace ID lives here rather than in the middleware because it is a
ContextVar: unlike thread-local storage it is reset per request, and it can be
carried into ThreadPoolExecutor workers via copy_ctx() (used by CRAG's
concurrent grader).
"""

import contextvars
import functools
import logging
import time
from contextlib import contextmanager

from django.conf import settings

timing_logger = logging.getLogger("embediq.timing")

# Trace ID for the in-flight request (or CLI run). "-" when outside one.
trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id", default="-"
)

# Per-request bag (user, path, ...) so stage lines can carry the user without
# threading it through every function signature in the pipeline.
request_ctx_var: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "request_ctx", default={}
)


def get_trace_id() -> str:
    """Current trace ID, or '-' when running outside a request."""
    return trace_id_var.get()


def copy_ctx() -> contextvars.Context:
    """
    Snapshot of the current context, for propagating the trace ID into worker
    threads. ThreadPoolExecutor does not copy context automatically, so submit
    work as ``executor.submit(copy_ctx().run, fn, *args)``.
    """
    return contextvars.copy_context()


def _format_fields(fields: dict) -> str:
    """Render a dict as a ``k=v k=v`` tail, skipping None values."""
    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    """Emit a single timing event line. Used for both 'request' and 'stage'."""
    timing_logger.log(level, "%s %s", event, _format_fields(fields))


@contextmanager
def stage(name: str, **fields):
    """
    Time a block of work and log its duration.

    Yields a mutable dict; anything added to it is included in the log line, so
    counts only known after the work runs can still be reported::

        with stage("retrieve") as s:
            chunks = search_chunks(query)
            s["chunks"] = len(chunks)

    On exception the duration is still logged (with ``error=<ExcType>``) and the
    exception re-raised — a Gemini call that times out should still report how
    long it burned.
    """
    extra = dict(fields)

    if not getattr(settings, "TIMING_ENABLED", True):
        yield extra
        return

    ctx = request_ctx_var.get()
    started = time.perf_counter()
    error = None
    try:
        yield extra
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        # Merged rather than splatted so a caller-supplied field (e.g. an
        # error it handled itself) can't collide with the built-in keys.
        fields = {
            "name": name,
            "duration_ms": duration_ms,
            "user": ctx.get("user"),
            "error": error,
        }
        fields.update(extra)
        log_event("stage", **fields)


def timed(name: str | None = None):
    """Decorator form of stage(), using the function name when none is given."""

    def decorator(fn):
        stage_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with stage(stage_name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator