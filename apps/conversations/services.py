"""
apps/conversations/services.py

CRUD helpers for conversation sessions and turns.
All database access goes through here so the API and generation layers
stay decoupled from Django ORM details.
"""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth.models import User

from .models import ConversationSession, ConversationTurn


def get_or_create_session(
    session_id: str | None, user: User | None = None
) -> ConversationSession:
    """
    Return an existing session by UUID, or create a new one.
    If session_id is None or not found, a fresh session is returned.
    The session is scoped to the given user when provided.
    """
    if session_id:
        try:
            qs = ConversationSession.objects.filter(pk=session_id)
            if user is not None:
                qs = qs.filter(user=user)
            return qs.get()
        except ConversationSession.DoesNotExist:
            pass
    return ConversationSession.objects.create(user=user)


def get_history(
    session: ConversationSession,
    max_turns: int | None = None,
) -> list[dict]:
    """
    Return the last `max_turns` turns for this session as a list of dicts:
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    max_turns defaults to settings.MAX_HISTORY_TURNS.
    """
    limit = (
        max_turns
        if max_turns is not None
        else getattr(settings, "MAX_HISTORY_TURNS", 10)
    )
    turns = session.turns.order_by("-created_at")[:limit]
    # Reverse so they are in chronological order
    return [{"role": t.role, "content": t.content} for t in reversed(list(turns))]


def save_turn(
    session: ConversationSession, role: str, content: str
) -> ConversationTurn:
    """Persist a single user or assistant message."""
    return ConversationTurn.objects.create(session=session, role=role, content=content)
