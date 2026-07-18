"""
apps/accounts/api.py

Authentication router — mounted at /api/auth/

Endpoints:
  POST /api/auth/register  — create account with email + password
  POST /api/auth/login     — returns access + refresh JWT tokens
  POST /api/auth/refresh   — exchange refresh token for new access token
  POST /api/auth/google    — verify Google ID token, create/get user, return JWTs
  GET  /api/auth/me        — return current authenticated user info
"""
import logging

from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from ninja import Router
from ninja.errors import HttpError
from rest_framework_simplejwt.tokens import RefreshToken

from .auth import jwt_auth
from .schemas import (
    AccessOut,
    GoogleAuthIn,
    LoginIn,
    RefreshIn,
    RegisterIn,
    TokenWithUserOut,
    UserOut,
)

logger = logging.getLogger(__name__)
router = Router(tags=["Auth"])


def _make_tokens(user: User) -> dict:
    """Generate a fresh access + refresh token pair for a user."""
    refresh = RefreshToken.for_user(user)
    return {
        "access": str(refresh.access_token),
        "refresh": str(refresh),
    }


def _user_out(user: User, is_new: bool = False) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "display_name": user.get_full_name() or user.username,
        "is_new": is_new,
    }


# ─── Register ────────────────────────────────────────────────────────────────

@router.post("/register", response=TokenWithUserOut, summary="Register a new account")
def register(request, payload: RegisterIn):
    """Create a new user with email + password and return JWT tokens."""
    email = payload.email.strip().lower()
    if not email or not payload.password:
        raise HttpError(400, "Email and password are required.")
    if User.objects.filter(username=email).exists():
        raise HttpError(409, "An account with this email already exists.")

    display_name = payload.display_name.strip() or email.split("@")[0]
    first, *rest = display_name.split(" ", 1)
    user = User.objects.create_user(
        username=email,
        email=email,
        password=payload.password,
        first_name=first,
        last_name=rest[0] if rest else "",
    )
    tokens = _make_tokens(user)
    logger.info("Registered new user: %s (id=%s)", email, user.id)
    return {**tokens, "user": _user_out(user, is_new=True)}


# ─── Login ───────────────────────────────────────────────────────────────────

@router.post("/login", response=TokenWithUserOut, summary="Login with email + password")
def login(request, payload: LoginIn):
    """Authenticate with email + password and return JWT tokens."""
    email = payload.email.strip().lower()
    user = authenticate(request, username=email, password=payload.password)
    if user is None:
        raise HttpError(401, "Invalid email or password.")
    tokens = _make_tokens(user)
    logger.info("User logged in: %s (id=%s)", email, user.id)
    return {**tokens, "user": _user_out(user)}


# ─── Refresh ─────────────────────────────────────────────────────────────────

@router.post("/refresh", response=AccessOut, summary="Refresh access token")
def refresh_token(request, payload: RefreshIn):
    """Exchange a valid refresh token for a new access token."""
    from rest_framework_simplejwt.tokens import RefreshToken as RT
    from rest_framework_simplejwt.exceptions import TokenError
    try:
        token = RT(payload.refresh)
        return {"access": str(token.access_token)}
    except TokenError as e:
        raise HttpError(401, f"Invalid or expired refresh token: {e}")


# ─── Google OAuth ─────────────────────────────────────────────────────────────

@router.post("/google", response=TokenWithUserOut, summary="Sign in with Google")
def google_auth(request, payload: GoogleAuthIn):
    """
    Verify a Google ID token (from the frontend Google Sign-In SDK),
    then create or retrieve the corresponding user account and return JWTs.
    """
    client_id = getattr(settings, "GOOGLE_CLIENT_ID", "")
    if not client_id:
        raise HttpError(500, "Google OAuth is not configured on this server.")

    try:
        id_info = google_id_token.verify_oauth2_token(
            payload.id_token,
            google_requests.Request(),
            client_id,
        )
    except ValueError as e:
        logger.warning("Google ID token verification failed: %s", e)
        raise HttpError(401, "Invalid Google ID token.")

    email = id_info.get("email", "").strip().lower()
    if not email:
        raise HttpError(400, "Google account has no email address.")

    given_name = id_info.get("given_name", "")
    family_name = id_info.get("family_name", "")

    user, created = User.objects.get_or_create(
        username=email,
        defaults={
            "email": email,
            "first_name": given_name,
            "last_name": family_name,
            "is_active": True,
        },
    )
    # Update name if it changed on Google side
    if not created and (user.first_name != given_name or user.last_name != family_name):
        user.first_name = given_name
        user.last_name = family_name
        user.save(update_fields=["first_name", "last_name"])

    tokens = _make_tokens(user)
    logger.info("Google sign-in: %s (id=%s, new=%s)", email, user.id, created)
    return {**tokens, "user": _user_out(user, is_new=created)}


# ─── Me ──────────────────────────────────────────────────────────────────────

@router.get("/me", response=UserOut, auth=jwt_auth, summary="Get current user")
def me(request):
    """Return the authenticated user's profile."""
    user: User = request.auth
    return _user_out(user)
