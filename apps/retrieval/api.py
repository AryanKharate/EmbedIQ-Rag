import uuid
import logging

from ninja import Router, Schema, File
from ninja.files import UploadedFile
from django.shortcuts import get_object_or_404
from django.conf import settings

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .models import Document
from .ingest_service import ingest_document
from apps.accounts.auth import jwt_auth

logger = logging.getLogger(__name__)

router = Router(tags=["Documents"])

_qdrant_client = QdrantClient(url=settings.QDRANT_URL)


class DocumentSchema(Schema):
    id: uuid.UUID
    filename: str
    is_active: bool
    created_at: str


class DocumentStatusUpdate(Schema):
    is_active: bool


@router.get(
    "/",
    response=list[DocumentSchema],
    auth=jwt_auth,
    summary="List all ingested documents",
)
def list_documents(request):
    """Returns all documents owned by the authenticated user."""
    user = request.auth
    logger.info("Fetching documents for user: %s", user.id)
    docs = Document.objects.filter(owner=user).order_by("-created_at")
    logger.info("Retrieved %d documents for user %s", len(docs), user.id)
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "is_active": d.is_active,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.post(
    "/upload",
    response=DocumentSchema,
    auth=jwt_auth,
    summary="Upload and ingest a document (.txt, .pdf, or .md)",
)
def upload_document(request, file: UploadedFile = File(...)):
    """
    Ingests a document directly via API.
    The document is owned by the authenticated user.
    Supports .txt, .pdf, and .md files.
    """
    user = request.auth
    logger.info("Starting upload for document: %s (user=%s)", file.name, user.id)
    file_bytes = file.read()
    doc = ingest_document(filename=file.name, file_bytes=file_bytes, owner=user)
    logger.info("Successfully uploaded and ingested document: %s", doc.id)
    return {
        "id": doc.id,
        "filename": doc.filename,
        "is_active": doc.is_active,
        "created_at": doc.created_at.isoformat(),
    }


@router.patch(
    "/{doc_id}/toggle",
    response=DocumentSchema,
    auth=jwt_auth,
    summary="Toggle document active status",
)
def toggle_document(request, doc_id: uuid.UUID, payload: DocumentStatusUpdate):
    """
    Sets a document as active or inactive.
    Only the owning user can toggle their own documents.
    """
    user = request.auth
    logger.info(
        "Toggling document %s active status to %s (user=%s)",
        doc_id,
        payload.is_active,
        user.id,
    )
    doc = get_object_or_404(Document, id=doc_id, owner=user)
    doc.is_active = payload.is_active
    doc.save()

    # Update Qdrant payload to match
    _qdrant_client.set_payload(
        collection_name=settings.COLLECTION_NAME,
        payload={"is_active": doc.is_active},
        points=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=str(doc.id)))
            ]
        ),
    )

    logger.info("Successfully toggled document %s", doc_id)

    return {
        "id": doc.id,
        "filename": doc.filename,
        "is_active": doc.is_active,
        "created_at": doc.created_at.isoformat(),
    }


@router.delete("/{doc_id}", auth=jwt_auth, summary="Delete a document and its vectors")
def delete_document(request, doc_id: uuid.UUID):
    """
    Deletes the document from the database and removes all its chunk vectors from Qdrant.
    Only the owning user can delete their own documents.
    """
    user = request.auth
    logger.info("Deleting document %s (user=%s)", doc_id, user.id)
    doc = get_object_or_404(Document, id=doc_id, owner=user)

    # Delete from Qdrant
    _qdrant_client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(key="document_id", match=MatchValue(value=str(doc.id)))
            ]
        ),
    )

    # Delete from Postgres
    doc.delete()

    logger.info("Successfully deleted document %s", doc_id)
    return {"success": True, "message": f"Document {doc.filename} deleted."}
