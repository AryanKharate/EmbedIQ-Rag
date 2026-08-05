"""
apps/retrieval/management/commands/ingest.py

Django management command that ingests one or more text files into Qdrant.

Improvements over v1:
  - Named vector collection schema: "dense" (Gemini) + "sparse" (BM25/FastEmbed)
  - Sentence-aware chunking (splits on paragraph/sentence boundaries, no mid-word cuts)
  - Batch Gemini embedding (one API call per document instead of one per chunk)
  - SparseVectorParams with Modifier.IDF for accurate BM25 scoring at scale

Usage:
    python manage.py ingest doc1.txt
    python manage.py ingest doc1.txt doc2.txt --chunk-size 800
"""

from pathlib import Path
import logging

from django.core.management.base import BaseCommand, CommandError
from apps.retrieval.ingest_service import ingest_document

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Chunk, embed (dense + sparse), and upsert text/pdf files into Qdrant."

    def add_arguments(self, parser):
        parser.add_argument(
            "files",
            nargs="+",
            type=Path,
            metavar="FILE",
            help="One or more .txt or .pdf files to ingest.",
        )
        # Chunk sizes are configured via PARENT_CHUNK_SIZE and CHILD_CHUNK_SIZE
        # environment variables (see config/settings.py).

    def handle(self, *args, **options):
        files: list[Path] = options["files"]

        logger.info(f"Starting ingestion process for files: {[str(f) for f in files]}")

        for file_path in files:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                raise CommandError(f"File not found: {file_path}")

            self.stdout.write(f"Ingesting {file_path} ...")
            logger.info(f"Ingesting {file_path} ...")
            file_bytes = file_path.read_bytes()

            doc = ingest_document(
                filename=file_path.name,
                file_bytes=file_bytes,
            )
            self.stdout.write(
                self.style.SUCCESS(f"  Upserted {doc.filename} (ID: {doc.id})")
            )
            logger.info(f"Upserted {doc.filename} (ID: {doc.id})")

        self.stdout.write(self.style.SUCCESS("Done."))
        logger.info("Ingestion process completed successfully.")
