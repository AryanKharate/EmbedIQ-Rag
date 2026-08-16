import csv
import hashlib
import io
import logging
import re
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from django.conf import settings
from fastembed import SparseTextEmbedding
from google import genai
from google.genai.errors import ServerError
from qdrant_client.models import PointStruct, SparseVector
from django.core.files.base import ContentFile
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from config.timing import stage
from . import vector_store

from .models import Document, DocumentImage, DocumentTable

logger = logging.getLogger(__name__)

# Shared models for embedding
_genai_client = genai.Client(api_key=settings.GEMINI_API_KEY)
_bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# Regex: matches a GitHub-style Markdown table.
# Requires: a header row, a separator row (|---|...|), and one or more data rows.
_MD_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\s*\n\s*\|[-|: \t]+\|\s*\n(?:\|[^\n]+\|\s*\n?)+)",
    re.MULTILINE,
)


def ensure_collection() -> None:
    vector_store.ensure_collection()


def set_document_active(doc_id, is_active: bool) -> None:
    """Update the is_active flag on every chunk vector belonging to doc_id."""
    vector_store.set_document_active(doc_id, is_active)


def delete_document_vectors(doc_id) -> None:
    """Delete every chunk vector belonging to doc_id from Qdrant."""
    vector_store.delete_document_vectors(doc_id)


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
                                current_chunk = (
                                    current_chunk[-overlap:].strip() + " " + sentence
                                )
                                current_chunk = current_chunk.strip()
                            else:
                                current_chunk = sentence
                        else:
                            current_chunk = sentence
            else:
                current_chunk = (
                    (current_chunk + "\n\n" + para).strip() if current_chunk else para
                )

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


@retry(
    wait=wait_random_exponential(multiplier=1, max=15),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ServerError),
)
def _embed_dense_one_batch(batch: list[str]):
    """One Gemini embed_content call — retried on its own so a transient 5xx
    doesn't force re-embedding batches that already succeeded."""
    return _genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch,
        config={
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": settings.EMBED_DIM,
        },
    )


def _embed_dense_batch(chunks: list[str]) -> list[list[float]]:
    # Gemini API supports a maximum of 100 items per batch embed request.
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        result = _embed_dense_one_batch(batch)
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
    if ext not in settings.ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(sorted(settings.ALLOWED_UPLOAD_EXTENSIONS))}"
        )
    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(
            f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). "
            f"Maximum: {settings.MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f} MB."
        )


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _rows_to_markdown(header: list, rows: list[list]) -> str:
    """Serialize a table (header + data rows) to a GitHub-flavoured Markdown string."""

    def fmt(row: list) -> str:
        return "| " + " | ".join(str(c or "").strip() for c in row) + " |"

    separator = "| " + " | ".join("---" for _ in header) + " |"
    lines = [fmt(header), separator] + [fmt(r) for r in rows]
    return "\n".join(lines)


def _rows_to_csv(header: list, rows: list[list]) -> str:
    """Serialize a table (header + data rows) to a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([str(c or "").strip() for c in header])
    for row in rows:
        writer.writerow([str(c or "").strip() for c in row])
    return buf.getvalue()


def _split_rows_into_batches(
    header: list, rows: list[list], batch_size: int
) -> list[tuple[list, list[list]]]:
    """
    Split data rows into batches of at most batch_size rows.
    The header is returned with every batch so each chunk is self-contained.
    """
    if not rows:
        return [(header, [])]
    return [(header, rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]


def _extract_tables_from_page(page) -> list[dict]:
    """
    Detect and extract tables from a PyMuPDF page using find_tables().

    Large tables are split into row batches of TABLE_ROW_BATCH_SIZE rows, with
    the header row repeated in every batch so each chunk is self-contained.

    Returns a list of dicts:
        markdown     — GitHub Markdown string for this batch
        csv          — CSV string for this batch (optional downstream use)
        row_count    — number of data rows in this batch (excluding header)
        bbox         — (x0, y0, x1, y1) bounding box of the source table on the page
    """
    row_batch_size = settings.TABLE_ROW_BATCH_SIZE
    tables_out: list[dict] = []

    try:
        table_finder = page.find_tables()
    except Exception as exc:
        logger.warning(
            "find_tables() failed — skipping table extraction for page: %s", exc
        )
        return []

    for table in table_finder:
        try:
            data = table.extract()
        except Exception as exc:
            logger.warning("table.extract() failed — skipping table: %s", exc)
            continue

        if not data or len(data) < 2:
            # Need at least a header row + one data row to be meaningful
            continue

        header = data[0]
        rows = data[1:]

        for batch_header, batch_rows in _split_rows_into_batches(
            header, rows, row_batch_size
        ):
            md = _rows_to_markdown(batch_header, batch_rows)
            csv_str = _rows_to_csv(batch_header, batch_rows)
            tables_out.append(
                {
                    "markdown": md,
                    "csv": csv_str,
                    "row_count": len(batch_rows),
                    "bbox": table.bbox,  # used to mask table regions from raw text
                }
            )

    return tables_out


def _split_text_and_md_tables(text: str) -> list[dict]:
    """
    Split a plain-text or Markdown document into alternating segments of
    'text' and 'table' type, detected via regex.

    This prevents _chunk_text() from splitting Markdown tables mid-row in
    .txt and .md files.

    Each segment dict:
        type     — 'text' | 'table'
        content  — raw string (text body or full Markdown table)
        csv      — CSV string (table segments only)
        row_count — number of data rows (table segments only)
    """
    segments: list[dict] = []
    last_end = 0

    for match in _MD_TABLE_RE.finditer(text):
        # Plain text before this table
        before = text[last_end : match.start()]
        if before.strip():
            segments.append({"type": "text", "content": before})

        table_text = match.group(0).strip()

        # Re-parse the markdown table into header + data rows
        lines = [ln for ln in table_text.splitlines() if ln.strip().startswith("|")]
        # Filter out the separator line (|---|---|)
        data_lines = [ln for ln in lines if not re.match(r"^\|[-|: \t]+\|", ln.strip())]
        parsed_rows = [
            [cell.strip() for cell in ln.strip().strip("|").split("|")]
            for ln in data_lines
        ]
        header = parsed_rows[0] if parsed_rows else []
        data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []

        if header:
            # Split into row batches the same way PDF tables are
            # (settings.TABLE_ROW_BATCH_SIZE), so a large Markdown table in a
            # .txt/.md upload doesn't become one unbounded parent chunk.
            for batch_header, batch_rows in _split_rows_into_batches(
                header, data_rows, settings.TABLE_ROW_BATCH_SIZE
            ):
                segments.append(
                    {
                        "type": "table",
                        "content": _rows_to_markdown(batch_header, batch_rows),
                        "csv": _rows_to_csv(batch_header, batch_rows),
                        "row_count": len(batch_rows),
                    }
                )
        else:
            # Couldn't parse a header row — keep the raw matched text as one
            # unsplit segment rather than dropping the table entirely.
            segments.append(
                {
                    "type": "table",
                    "content": table_text,
                    "csv": "",
                    "row_count": len(data_rows),
                }
            )
        last_end = match.end()

    # Any remaining plain text after the last table
    remaining = text[last_end:]
    if remaining.strip():
        segments.append({"type": "text", "content": remaining})

    return segments


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_text_from_bytes(db_doc: Document, file_bytes: bytes) -> list[dict]:
    """
    Extract text and tables per page, save images and DocumentTable records.

    Returns pages_data: list of dicts with keys:
        page_number — 1-indexed
        text        — plain text from the page (table regions excluded for PDFs)
        tables      — list of table dicts (markdown, csv, row_count, bbox)
    """
    pages_data = []

    if db_doc.filename.lower().endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                # ── Table extraction ─────────────────────────────────────────
                table_dicts = _extract_tables_from_page(page)
                table_bboxes = [fitz.Rect(t["bbox"]) for t in table_dicts]

                # ── Text extraction (table regions masked out) ───────────────
                # Use text blocks so we can filter by bbox, avoiding
                # double-indexing text that belongs to a detected table.
                blocks = page.get_text("blocks")
                text_parts = []
                for block in blocks:
                    block_rect = fitz.Rect(block[:4])
                    if any(block_rect.intersects(tb) for tb in table_bboxes):
                        continue  # this block is inside a table — skip it
                    text_parts.append(block[4])  # block[4] is the text string
                text = "\n".join(text_parts)

                # ── Image extraction (unchanged) ─────────────────────────────
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    try:
                        image_bytes = pix.tobytes("png")
                    except Exception as exc:
                        logger.warning(
                            "Failed to save image xref %d directly: %s. Converting to RGB.",
                            xref,
                            exc,
                        )
                        try:
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                            image_bytes = pix_rgb.tobytes("png")
                        except Exception as inner_exc:
                            logger.error(
                                "Failed to convert image xref %d to RGB: %s",
                                xref,
                                inner_exc,
                            )
                            continue
                    image_filename = f"{db_doc.filename}_p{page_num}_i{img_index}.png"
                    doc_img = DocumentImage(document=db_doc, page_number=page_num)
                    doc_img.image.save(image_filename, ContentFile(image_bytes))
                    doc_img.save()

                # ── Save DocumentTable records ────────────────────────────────
                for tbl in table_dicts:
                    DocumentTable.objects.create(
                        document=db_doc,
                        page_number=page_num,
                        row_count=tbl["row_count"],
                        markdown=tbl["markdown"],
                    )

                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": text,
                        "tables": table_dicts,
                    }
                )

    else:
        # .txt / .md — use regex-based Markdown table detection
        raw_text = file_bytes.decode("utf-8")
        segments = _split_text_and_md_tables(raw_text)

        # Rejoin plain-text segments (tables are handled separately)
        text_only = "\n\n".join(s["content"] for s in segments if s["type"] == "text")
        tables = [
            {
                "markdown": s["content"],
                "csv": s.get("csv", ""),
                "row_count": s.get("row_count", 0),
                "bbox": None,
            }
            for s in segments
            if s["type"] == "table"
        ]

        # Save DocumentTable records
        for tbl in tables:
            DocumentTable.objects.create(
                document=db_doc,
                page_number=1,
                row_count=tbl["row_count"],
                markdown=tbl["markdown"],
            )

        pages_data.append({"page_number": 1, "text": text_only, "tables": tables})

    return pages_data


# ---------------------------------------------------------------------------
# Main ingest entry point
# ---------------------------------------------------------------------------


def ingest_document(filename: str, file_bytes: bytes, owner=None) -> Document:
    """Parse, chunk, embed, and upsert a document into Qdrant, and save to Postgres.

    Thin timing wrapper around _ingest_document so the whole ingestion reports a
    total duration alongside its per-phase stages.

    Args:
        filename: Original filename of the document.
        file_bytes: Raw bytes of the document.
        owner: Django User instance that owns this document. If None, the document
               is not scoped to any user (legacy/admin ingestion path).
    """
    with stage("ingest", filename=filename, bytes=len(file_bytes)):
        return _ingest_document(filename, file_bytes, owner=owner)


def _ingest_document(filename: str, file_bytes: bytes, owner=None) -> Document:
    # Validate upload before any processing
    _validate_upload(filename, file_bytes)

    ensure_collection()

    # Create DB record
    db_doc = Document.objects.create(filename=filename, is_active=True, owner=owner)

    # Extract text, tables, and images per page
    with stage("ingest_extract") as s:
        pages_data = extract_text_from_bytes(db_doc, file_bytes)
        s["pages"] = len(pages_data)

    # ── 1. Collect parent chunks (text + tables) ────────────────────────────
    with stage("ingest_chunk") as s:
        parent_chunks_with_metadata: list[dict] = []

        for page_data in pages_data:
            # Text chunks
            p_chunks = _chunk_text(page_data["text"], settings.PARENT_CHUNK_SIZE)
            for c in p_chunks:
                parent_chunks_with_metadata.append(
                    {
                        "page_number": page_data["page_number"],
                        "text": c,
                        "chunk_type": "text",
                        "table_row_count": None,
                        "table_csv": None,
                    }
                )

            # Table chunks — each markdown batch is already a bounded parent
            for tbl in page_data.get("tables", []):
                parent_chunks_with_metadata.append(
                    {
                        "page_number": page_data["page_number"],
                        "text": tbl["markdown"],
                        "chunk_type": "table",
                        "table_row_count": tbl["row_count"],
                        "table_csv": tbl.get("csv") or None,
                    }
                )

        s["parents"] = len(parent_chunks_with_metadata)

        if not parent_chunks_with_metadata:
            return db_doc

        # ── 2. Sub-chunk parents into small child vectors ───────────────────
        child_chunks_flat: list[str] = []
        child_meta: list[dict] = []

        child_overlap = max(0, settings.CHILD_CHUNK_SIZE // 10)  # ~10% overlap

        for parent_idx, parent_obj in enumerate(parent_chunks_with_metadata):
            children = _chunk_text(
                parent_obj["text"], settings.CHILD_CHUNK_SIZE, overlap=child_overlap
            )
            for child in children:
                child_chunks_flat.append(child)
                child_meta.append(
                    {
                        "parent_text": parent_obj["text"],
                        "parent_idx": parent_idx,
                        "page_number": parent_obj["page_number"],
                        "chunk_type": parent_obj["chunk_type"],
                        "table_row_count": parent_obj["table_row_count"],
                        "table_csv": parent_obj["table_csv"],
                    }
                )

        s["children"] = len(child_chunks_flat)

    # ── 3. Embed dense (children only) ─────────────────────────────────────
    with stage("ingest_embed_dense", children=len(child_chunks_flat)):
        dense_vectors = _embed_dense_batch(child_chunks_flat)

    # ── 4. Embed sparse in batch ────────────────────────────────────────────
    with stage("ingest_embed_sparse", children=len(child_chunks_flat)):
        sparse_vectors = _embed_sparse_batch(child_chunks_flat)

    # ── 5. Build and upsert Qdrant points in bounded batches ───────────────
    doc_id_str = str(db_doc.id)
    points: list[PointStruct] = []

    for idx, (child, dense_vec, sparse_vec, meta) in enumerate(
        zip(child_chunks_flat, dense_vectors, sparse_vectors, child_meta)
    ):
        # Deterministic point ID: hash of (document_id, parent_index, child_index)
        id_seed = f"{doc_id_str}:{meta['parent_idx']}:{idx}"
        point_id = str(uuid.UUID(hashlib.sha256(id_seed.encode()).hexdigest()[:32]))

        payload: dict = {
            "text": child,
            "parent_text": meta["parent_text"],  # The LLM will see this!
            "parent_id": f"{doc_id_str}:{meta['parent_idx']}",
            "source": filename,
            "chunk_index": idx,
            "document_id": doc_id_str,
            "user_id": str(owner.id) if owner else None,  # per-user isolation
            "page_number": meta["page_number"],
            "is_active": True,
            "chunk_type": meta["chunk_type"],  # "text" | "table"
        }

        # Table-specific optional fields
        if meta["chunk_type"] == "table":
            payload["table_row_count"] = meta["table_row_count"]
            if meta["table_csv"] is not None:
                payload["table_csv"] = meta["table_csv"]

        points.append(
            PointStruct(
                id=point_id,
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload=payload,
            )
        )

        if len(points) >= settings.QDRANT_UPSERT_BATCH_SIZE:
            with stage("ingest_upsert", points=len(points)):
                vector_store.upsert_points(points)
            logger.info("Upserted batch of %d points for %s", len(points), filename)
            points = []

    # Upsert remaining points
    if points:
        with stage("ingest_upsert", points=len(points)):
            vector_store.upsert_points(points)
        logger.info("Upserted final batch of %d points for %s", len(points), filename)

    return db_doc
