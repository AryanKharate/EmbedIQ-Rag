"""
apps/conversations/api.py

Read-only endpoints for browsing a user's persisted chat history:

  GET /api/conversations/       — list this user's sessions
  GET /api/conversations/{id}   — full turn history for one session

Before these existed, /api/query was the only way to touch a
ConversationSession — sessions were fully persisted in Postgres but had no
way to be read back, so they were invisible the moment a client's local
cache (browser localStorage) didn't know about them. These endpoints make
that already-persisted history reachable.
"""

import logging
import uuid

from ninja import Router, Schema
from ninja.errors import HttpError

from apps.accounts.auth import jwt_auth

from .services import get_session_for_user, list_sessions

logger = logging.getLogger(__name__)

router = Router(tags=["Conversations"])


class ConversationSummary(Schema):
    id: str
    title: str
    updated_at: str
    message_count: int


class TurnOut(Schema):
    role: str
    content: str
    created_at: str


class ConversationDetail(Schema):
    id: str
    turns: list[TurnOut]


@router.get(
    "/",
    response=list[ConversationSummary],
    auth=jwt_auth,
    summary="List this user's chat sessions",
)
def list_conversations(request):
    user = request.auth
    sessions = list_sessions(user)
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "updated_at": s["updated_at"].isoformat(),
            "message_count": s["message_count"],
        }
        for s in sessions
    ]


@router.get(
    "/{session_id}",
    response=ConversationDetail,
    auth=jwt_auth,
    summary="Get a chat session's full turn history",
)
def get_conversation(request, session_id: uuid.UUID):
    user = request.auth
    session = get_session_for_user(str(session_id), user)
    if session is None:
        raise HttpError(404, "Conversation not found.")
    return {
        "id": str(session.id),
        "turns": [
            {
                "role": t.role,
                "content": t.content,
                "created_at": t.created_at.isoformat(),
            }
            for t in session.turns.all()
        ],
    }
