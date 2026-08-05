"""
apps/conversations/models.py

Two Django models that persist multi-turn conversation history in Postgres:

  ConversationSession  — one row per chat thread (keyed by UUID)
  ConversationTurn     — one row per message (user or assistant)
"""

import uuid

from django.contrib.auth.models import User
from django.db import models


class ConversationSession(models.Model):
    """A single chat session. Created automatically on the first question."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="sessions",
        null=True,  # nullable for safe migration of existing sessions
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.id} (started {self.created_at:%Y-%m-%d %H:%M})"


class ConversationTurn(models.Model):
    """One user or assistant message within a session."""

    ROLE_CHOICES = [("user", "User"), ("assistant", "Assistant")]

    session = models.ForeignKey(
        ConversationSession,
        on_delete=models.CASCADE,
        related_name="turns",
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}"
