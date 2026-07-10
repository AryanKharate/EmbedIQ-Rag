import uuid
import logging

logger = logging.getLogger(__name__)
from ninja import Router, Schema, File
from ninja.files import UploadedFile
from django.shortcuts import get_object_or_404

from .models import Document
from .ingest_service import ingest_document
from qdrant_client import QdrantClient
from django.conf import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue

router = Router(tags=["Documents"])

_qdrant_client = QdrantClient(url=settings.QDRANT_URL)


class DocumentSchema(Schema):
    id: uuid.UUID
    filename: str
    is_active: bool
    created_at: str


class DocumentStatusUpdate(Schema):
    is_active: bool


@router.get("/", response=list[DocumentSchema], summary="List all ingested documents")
def list_documents(request):
    """Returns a list of all documents currently ingested and tracked."""
    logger.info("Fetching list of all documents")
    docs = Document.objects.all().order_by("-created_at")
    logger.info(f"Retrieved {len(docs)} documents")
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "is_active": d.is_active,
            "created_at": d.created_at.isoformat(),
        }
        for d in docs
    ]


@router.post("/upload", response=DocumentSchema, summary="Upload and ingest a document (.txt or .pdf)")
def upload_document(request, file: UploadedFile = File(...)):
    """
    Ingests a document directly via API.
    Supports .txt and .pdf files.
    Extracts text, chunks it, embeds it, and upserts to Qdrant.
    """
    logger.info(f"Starting upload for document: {file.name}")
    file_bytes = file.read()
    doc = ingest_document(filename=file.name, file_bytes=file_bytes)
    logger.info(f"Successfully uploaded and ingested document: {doc.id}")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "is_active": doc.is_active,
        "created_at": doc.created_at.isoformat(),
    }


@router.patch("/{doc_id}/toggle", response=DocumentSchema, summary="Toggle document active status")
def toggle_document(request, doc_id: uuid.UUID, payload: DocumentStatusUpdate):
    """
    Sets a document as active or inactive.
    Inactive documents will be skipped during Qdrant vector searches.
    """
    logger.info(f"Toggling document {doc_id} active status to {payload.is_active}")
    doc = get_object_or_404(Document, id=doc_id)
    doc.is_active = payload.is_active
    doc.save()

    # Update Qdrant payload to match
    _qdrant_client.set_payload(
        collection_name=settings.COLLECTION_NAME,
        payload={"is_active": doc.is_active},
        points=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=str(doc.id)))]
        ),
    )

    logger.info(f"Successfully toggled document {doc_id}")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "is_active": doc.is_active,
        "created_at": doc.created_at.isoformat(),
    }


@router.delete("/{doc_id}", summary="Delete a document and its vectors")
def delete_document(request, doc_id: uuid.UUID):
    """
    Deletes the document from the database and removes all its chunk vectors from Qdrant.
    """
    logger.info(f"Deleting document {doc_id} and its vectors")
    doc = get_object_or_404(Document, id=doc_id)
    
    # Delete from Qdrant
    _qdrant_client.delete(
        collection_name=settings.COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=str(doc.id)))]
        )
    )
    
    # Delete from Postgres
    doc.delete()
    
    logger.info(f"Successfully deleted document {doc_id}")
    return {"success": True, "message": f"Document {doc.filename} deleted."}
