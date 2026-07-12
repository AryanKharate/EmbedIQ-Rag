import hashlib
import logging
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings
from fastembed import SparseTextEmbedding
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from django.core.files.base import ContentFile

from .models import Document, DocumentImage

logger = logging.getLogger(__name__)

# Shared models for embedding
_genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
_qdrant_client = QdrantClient(url=settings.QDRANT_URL)
_bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# --- Configuration ---
_QDRANT_UPSERT_BATCH_SIZE = 200
_ALLOWED_EXTENSIONS = {".txt", ".pdf", ".md"}
_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def ensure_collection() -> None:
    if not _qdrant_client.collection_exists(settings.COLLECTION_NAME):
        _qdrant_client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=settings.EMBED_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(modifier=Modifier.IDF)
            },
        )


def _chunk_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """
    Split text into chunks of approximately chunk_size characters.

    Uses paragraph and sentence boundaries to avoid mid-word cuts.
    When overlap > 0, consecutive chunks share that many trailing/leading
    characters to prevent facts from being lost at boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                # Carry overlap from end of previous chunk
                if overlap > 0:
                    current_chunk = current_chunk[-overlap:].strip()
                else:
                    current_chunk = ""

            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk = (current_chunk + " " + sentence).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            if overlap > 0:
                                current_chunk = current_chunk[-overlap:].strip() + " " + sentence
                                current_chunk = current_chunk.strip()
                            else:
                                current_chunk = sentence
                        else:
                            current_chunk = sentence
            else:
                current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _embed_dense_batch(chunks: list[str]) -> list[list[float]]:
    # Gemini API supports a maximum of 100 items per batch embed request.
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        result = _genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config={
                "task_type": "RETRIEVAL_DOCUMENT",
                "output_dimensionality": settings.EMBED_DIM,
            },
        )
        all_embeddings.extend([embedding.values for embedding in result.embeddings])
        
    return all_embeddings


def _embed_sparse_batch(texts: list[str]) -> list[SparseVector]:
    """
    Generate BM25 sparse vectors for a batch of texts at once.

    The fastembed model's .embed() returns a generator, so we consume it
    in one pass instead of calling embed([text]) per child.
    """
    results = list(_bm25_model.embed(texts))
    return [
        SparseVector(
            indices=r.indices.tolist(),
            values=r.values.tolist(),
        )
        for r in results
    ]


def _validate_upload(filename: str, file_bytes: bytes) -> None:
    """Basic upload validation: extension allowlist and file size limit."""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}"
        )
    if len(file_bytes) > _MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum: {_MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )


def extract_text_from_bytes(db_doc: Document, file_bytes: bytes) -> list[dict]:
    """Extract text by page and save extracted images."""
    pages_data = []
    if db_doc.filename.lower().endswith(".pdf"):
        # Use PyMuPDF to extract text from PDF bytes
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                
                # Extract images on this page
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    # Use Pixmap instead of extract_image to properly apply colorspaces and decode arrays (fixes inverted colors)
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to RGB if it's CMYK or has weird components
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                        
                    image_bytes = pix.tobytes("png")
                    image_ext = "png"
                    image_filename = f"{db_doc.filename}_p{page_num}_i{img_index}.{image_ext}"
                    
                    # Save image to DocumentImage model
                    doc_img = DocumentImage(document=db_doc, page_number=page_num)
                    doc_img.image.save(image_filename, ContentFile(image_bytes))
                    doc_img.save()

                pages_data.append({"page_number": page_num, "text": text})
    else:
        # Default to utf-8 text, page 1
        pages_data.append({"page_number": 1, "text": file_bytes.decode("utf-8")})
    
    return pages_data


def ingest_document(filename: str, file_bytes: bytes) -> Document:
    """Parse, chunk, embed, and upsert a document into Qdrant, and save to Postgres."""
    # Validate upload before any processing
    _validate_upload(filename, file_bytes)

    ensure_collection()
    
    # Create DB record
    db_doc = Document.objects.create(filename=filename, is_active=True)
    
    # Extract text and images per page
    pages_data = extract_text_from_bytes(db_doc, file_bytes)

    # 1. Chunk into large parent contexts per page
    parent_chunks_with_metadata = []
    for page_data in pages_data:
        p_chunks = _chunk_text(page_data["text"], settings.PARENT_CHUNK_SIZE)
        for c in p_chunks:
            parent_chunks_with_metadata.append({
                "page_number": page_data["page_number"],
                "text": c
            })
            
    if not parent_chunks_with_metadata:
        return db_doc

    child_chunks_flat = []
    child_to_parent_map = []
    child_to_parent_idx = []
    child_to_page_num = []
    
    # 2. Sub-chunk into small child vectors with overlap
    child_overlap = max(0, settings.CHILD_CHUNK_SIZE // 10)  # ~10% overlap
    for parent_idx, parent_obj in enumerate(parent_chunks_with_metadata):
        children = _chunk_text(parent_obj["text"], settings.CHILD_CHUNK_SIZE, overlap=child_overlap)
        for child in children:
            child_chunks_flat.append(child)
            child_to_parent_map.append(parent_obj["text"])
            child_to_parent_idx.append(parent_idx)
            child_to_page_num.append(parent_obj["page_number"])

    # 3. Embed dense (we only embed the children)
    dense_vectors = _embed_dense_batch(child_chunks_flat)

    # 4. Embed sparse in batch (instead of one-at-a-time)
    sparse_vectors = _embed_sparse_batch(child_chunks_flat)

    # 5. Prepare and upsert Qdrant points in bounded batches
    doc_id_str = str(db_doc.id)
    points: list[PointStruct] = []

    for idx, (child, dense_vec, sparse_vec, parent, parent_idx, page_num) in enumerate(
        zip(child_chunks_flat, dense_vectors, sparse_vectors, child_to_parent_map, child_to_parent_idx, child_to_page_num)
    ):
        # Deterministic point ID: hash of (document_id, parent_index, child_index)
        id_seed = f"{doc_id_str}:{parent_idx}:{idx}"
        point_id = str(uuid.UUID(hashlib.sha256(id_seed.encode()).hexdigest()[:32]))

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
                payload={
                    "text": child,
                    "parent_text": parent,  # The LLM will see this!
                    "parent_id": f"{doc_id_str}:{parent_idx}",
                    "source": filename,
                    "chunk_index": idx,
                    "document_id": doc_id_str,
                    "page_number": page_num,
                    "is_active": True,
                },
            )
        )

        # Upsert in bounded batches to avoid memory/request-size limits
        if len(points) >= _QDRANT_UPSERT_BATCH_SIZE:
            _qdrant_client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=points,
                wait=True,
            )
            logger.info("Upserted batch of %d points for %s", len(points), filename)
            points = []

    # Upsert remaining points
    if points:
        _qdrant_client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points,
            wait=True,
        )
        logger.info("Upserted final batch of %d points for %s", len(points), filename)
    
    return db_doc
