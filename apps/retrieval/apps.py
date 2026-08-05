import os

from django.apps import AppConfig


class RetrievalConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.retrieval"
    label = "retrieval"

    def ready(self):
        """Warm-load the BGE reranker at container startup."""
        if os.environ.get("RERANKER_ENABLED", "true").lower() == "true":
            # Import the singleton and trigger model load so the
            # first incoming request doesn't pay the download/load penalty.
            from .reranker import reranker

            reranker._load()
