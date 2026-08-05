"""
apps/accounts/schemas.py

Pydantic/Ninja schemas for all authentication endpoints.
"""

from ninja import Schema


class RegisterIn(Schema):
    email: str
    password: str
    display_name: str = ""


class LoginIn(Schema):
    email: str
    password: str


class GoogleAuthIn(Schema):
    """Client sends the Google ID token obtained from the Google Sign-In popup."""

    id_token: str


class TokenOut(Schema):
    access: str
    refresh: str


class UserOut(Schema):
    id: int
    email: str
    display_name: str
    is_new: bool = False  # True on first Google sign-in (so frontend can show welcome)


class TokenWithUserOut(Schema):
    access: str
    refresh: str
    user: UserOut


class RefreshIn(Schema):
    refresh: str


class AccessOut(Schema):
    access: str
