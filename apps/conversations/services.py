"""
apps/conversations/services.py

CRUD helpers for conversation sessions and turns.
All database access goes through here so the API and generation layers
stay decoupled from Django ORM details.
"""

from __future__ import annotations

from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Count, Max

from .models import ConversationSession, ConversationTurn

_TITLE_MAX_LEN = 42  # matches frontend/src/lib/chat-store.ts's deriveTitle()


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


def _derive_title(content: str) -> str:
    """Mirrors frontend/src/lib/chat-store.ts's deriveTitle() so a session's
    title looks the same whether it was created locally or read back from
    the backend."""
    trimmed = " ".join(content.strip().split())
    if not trimmed:
        return "New chat"
    if len(trimmed) > _TITLE_MAX_LEN:
        return trimmed[:_TITLE_MAX_LEN] + "…"
    return trimmed


def list_sessions(user: User) -> list[dict]:
    """
    Return this user's chat sessions as {id, title, updated_at, message_count}
    dicts, most recently active first. Sessions with no turns yet (e.g. an
    interrupted request that never reached save_turn) are excluded.

    title is derived from the first user turn — same truncation the frontend
    already applies locally, so a session looks the same whether it was
    created on this device or read back from the backend.
    """
    sessions = (
        ConversationSession.objects.filter(user=user)
        .annotate(last_turn_at=Max("turns__created_at"), turn_count=Count("turns"))
        .filter(turn_count__gt=0)
        .order_by("-last_turn_at")
    )

    result = []
    for s in sessions:
        first_user_turn = s.turns.filter(role="user").order_by("created_at").first()
        title = _derive_title(first_user_turn.content) if first_user_turn else "New chat"
        result.append(
            {
                "id": str(s.id),
                "title": title,
                "updated_at": s.last_turn_at,
                "message_count": s.turn_count,
            }
        )
    return result


def get_session_for_user(
    session_id: str, user: User
) -> ConversationSession | None:
    """Return the session with its turns prefetched, scoped to user, or None
    if it doesn't exist / isn't owned by this user."""
    try:
        return ConversationSession.objects.prefetch_related("turns").get(
            pk=session_id, user=user
        )
    except ConversationSession.DoesNotExist:
        return None
