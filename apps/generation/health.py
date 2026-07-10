"""
apps/generation/health.py

Health check endpoints for container orchestration and load balancers.

  /health/live   — always returns 200 (process is alive)
  /health/ready  — returns 200 if Postgres + Qdrant are reachable, 503 otherwise
"""
import logging

from django.conf import settings
from django.db import connection
from django.http import JsonResponse
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def liveness(request):
    """Liveness probe — always 200 if the process is running."""
    return JsonResponse({"status": "alive"})


def readiness(request):
    """
    Readiness probe — checks Postgres and Qdrant connectivity.
    Returns 200 if both are reachable, 503 with details otherwise.
    """
    checks = {}

    # Check Postgres
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"
        logger.error("Readiness check failed — Postgres: %s", e)

    # Check Qdrant
    try:
        client = QdrantClient(url=settings.QDRANT_URL, timeout=5)
        client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        logger.error("Readiness check failed — Qdrant: %s", e)

    all_ok = all(v == "ok" for v in checks.values())
    status_code = 200 if all_ok else 503

    return JsonResponse(
        {"status": "ready" if all_ok else "not_ready", "checks": checks},
        status=status_code,
    )
