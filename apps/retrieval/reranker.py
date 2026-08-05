"""
apps/retrieval/reranker.py

BGE cross-encoder reranker — loaded once as a thread-safe module-level singleton.

The model is lazily loaded on first use (protected by a lock to prevent
concurrent-first-request races). AppConfig.ready() explicitly triggers
the load at startup so the first request never pays the model-load penalty.

Input:  list of ScoredPoint objects (as returned by search_chunks)
Output: same ScoredPoints, re-sorted by reranker score, truncated to top_k
"""

import logging
import threading

import torch
from django.conf import settings

logger = logging.getLogger(__name__)


class _Reranker:
    """Thread-safe lazy singleton — model loads on first access, shared for all requests."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None  # loaded on first use
        return cls._instance

    def _load(self):
        with self._lock:
            if self._model is not None:
                return
            from FlagEmbedding import FlagReranker

            model_name = settings.RERANKER_MODEL
            use_fp16 = torch.cuda.is_available()
            logger.info(
                "Loading BGE reranker: %s (fp16=%s, device=%s)",
                model_name,
                use_fp16,
                "cuda" if use_fp16 else "cpu",
            )
            self._model = FlagReranker(model_name, use_fp16=use_fp16)
            logger.info("BGE reranker ready.")

    def rerank(self, query: str, candidates: list, top_k: int | None = None) -> list:
        """
        Rerank a list of ScoredPoint objects by cross-encoder relevance score.

        Args:
            query:      The search query (rewritten form).
            candidates: list of qdrant_client ScoredPoint objects returned by
                        search_chunks().  Each has .payload["text"].
            top_k:      How many to return after reranking.
                        Defaults to settings.TOP_K.

        Returns:
            Subset of candidates, re-sorted by rerank score descending.
            Each returned ScoredPoint has its .score replaced with the
            reranker's cross-encoder score for downstream thresholding.
        """
        self._load()
        k = top_k if top_k is not None else settings.TOP_K

        if not candidates:
            return []

        # Extract text from each ScoredPoint payload
        texts = [c.payload.get("text", "") for c in candidates]
        pairs = [[query, text] for text in texts]

        scores = self._model.compute_score(pairs, normalize=True)

        # compute_score returns a single float for a single pair; normalise to list
        if isinstance(scores, float):
            scores = [scores]

        # Sort candidates by rerank score descending, return top_k
        # Preserve the reranker score on each point for downstream use
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = []
        for point, rerank_score in scored[:k]:
            point.score = rerank_score
            result.append(point)
        return result


# Module-level singleton — import this everywhere
reranker = _Reranker()
