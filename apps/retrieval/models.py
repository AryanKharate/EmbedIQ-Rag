import uuid
from django.contrib.auth.models import User
from django.db import models


class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="documents",
        null=True,  # nullable during migration; will be non-null for all new docs
        blank=True,
    )
    filename = models.CharField(max_length=255)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.filename


class DocumentImage(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="images"
    )
    page_number = models.IntegerField()
    image = models.ImageField(upload_to="document_images/")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.document.filename} - Page {self.page_number}"


class DocumentTable(models.Model):
    """Tracks every table extracted from a document during ingestion."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="tables"
    )
    page_number = models.IntegerField()
    row_count = models.IntegerField(
        help_text="Number of data rows in this batch (excluding header)."
    )
    markdown = models.TextField(
        help_text="Full GitHub-flavoured Markdown representation of the table batch."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.document.filename} - Page {self.page_number} ({self.row_count} rows)"
