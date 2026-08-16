"""
apps/retrieval/vector_store.py

Single place that owns the Qdrant client and every raw Qdrant operation:
collection lifecycle, point writes, filtered/hybrid search, and deletes.

This module is intentionally Qdrant-only — it knows nothing about Gemini,
embeddings, or documents. Callers (apps.retrieval.services, ingest_service,
apps.generation.health, scripts/squad_eval.py) pass already-computed vectors
and filters in; this is just the one place that talks to Qdrant.

collection_name defaults to settings.COLLECTION_NAME, read at call time (not
import time), so callers can override it per call — e.g. scripts/squad_eval.py
points every call at an isolated "squad_eval" collection instead.
"""

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

# Shared module-level singleton — every function below reuses this connection.
_client = QdrantClient(url=settings.QDRANT_URL)


def ensure_collection(collection_name: str | None = None) -> None:
    """Create the named collection (dense + sparse schema) if it doesn't exist."""
    name = collection_name or settings.COLLECTION_NAME
    if not _client.collection_exists(name):
        _client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=settings.EMBED_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
        )


def recreate_collection(collection_name: str) -> None:
    """Drop (if present) and recreate a collection from scratch — used by
    eval scripts that need a clean slate on every run."""
    if _client.collection_exists(collection_name):
        _client.delete_collection(collection_name)
    ensure_collection(collection_name)


def delete_collection(collection_name: str) -> None:
    _client.delete_collection(collection_name)


def hybrid_search(
    dense_vec: list[float],
    sparse_vec: SparseVector,
    query_filter: Filter,
    limit: int,
    collection_name: str | None = None,
) -> list:
    """
    Dense + sparse hybrid search with server-side RRF fusion, via Qdrant's
    Universal Query API. Both branches run in a single round-trip.
    """
    name = collection_name or settings.COLLECTION_NAME
    return _client.query_points(
        collection_name=name,
        prefetch=[
            Prefetch(query=sparse_vec, using="sparse", limit=limit, filter=query_filter),
            Prefetch(query=dense_vec, using="dense", limit=limit, filter=query_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=limit,
    ).points


def upsert_points(points: list[PointStruct], collection_name: str | None = None) -> None:
    name = collection_name or settings.COLLECTION_NAME
    _client.upsert(collection_name=name, points=points, wait=True)


def _document_filter(doc_id) -> Filter:
    return Filter(
        must=[FieldCondition(key="document_id", match=MatchValue(value=str(doc_id)))]
    )


def set_document_active(
    doc_id, is_active: bool, collection_name: str | None = None
) -> None:
    """Update the is_active flag on every chunk vector belonging to doc_id."""
    name = collection_name or settings.COLLECTION_NAME
    _client.set_payload(
        collection_name=name,
        payload={"is_active": is_active},
        points=_document_filter(doc_id),
    )


def delete_document_vectors(doc_id, collection_name: str | None = None) -> None:
    """Delete every chunk vector belonging to doc_id from Qdrant."""
    name = collection_name or settings.COLLECTION_NAME
    _client.delete(collection_name=name, points_selector=_document_filter(doc_id))


def ping() -> None:
    """Raises if Qdrant is unreachable — used by the /health/ready probe."""
    _client.get_collections()
