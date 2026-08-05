"""
apps/accounts/auth.py

Custom Ninja HttpBearer authenticator using djangorestframework-simplejwt.
Validates the JWT in the Authorization header and returns the User object,
which becomes `request.auth` in every protected endpoint.
"""

import logging

from django.contrib.auth.models import User
from ninja.security import HttpBearer
from rest_framework_simplejwt.exceptions import InvalidToken, TokenError
from rest_framework_simplejwt.tokens import AccessToken

logger = logging.getLogger(__name__)


class JWTAuth(HttpBearer):
    """
    Validate a Bearer JWT token and return the corresponding User.
    If invalid, returns None (Ninja will respond with 401).
    """

    def authenticate(self, request, token: str):
        try:
            validated = AccessToken(token)
            user_id = validated["user_id"]
            user = User.objects.get(pk=user_id)
            return user
        except (InvalidToken, TokenError) as e:
            logger.debug("JWT validation failed: %s", e)
            return None
        except User.DoesNotExist:
            logger.warning(
                "JWT references non-existent user_id=%s", validated.get("user_id")
            )
            return None
        except Exception as e:
            logger.error("Unexpected auth error: %s", e)
            return None


# Singleton — reuse across all routers
jwt_auth = JWTAuth()
