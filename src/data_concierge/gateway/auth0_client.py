"""Auth0 OAuth2 / OIDC helper.

Thin wrapper around Auth0's ``/authorize``, ``/oauth/token``, and ``/userinfo``
endpoints used by the social-login flow. Settings are read from environment
variables so Auth0 can be enabled without touching ``core/config.py``.

Environment variables:
    AUTH0_DOMAIN           e.g. "tenant.us.auth0.com"
    AUTH0_CLIENT_ID        OAuth2 client id
    AUTH0_CLIENT_SECRET    OAuth2 client secret
    AUTH0_CALLBACK_URL     Callback URL registered in Auth0 (e.g.
                           "http://localhost:8501/api/v1/auth/auth0/callback")
    AUTH0_ENABLED          "true" to enable the Auth0 UI + routes
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode

import httpx

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)


def is_enabled() -> bool:
    return os.environ.get("AUTH0_ENABLED", "").lower() in ("1", "true", "yes")


def settings() -> dict[str, str]:
    return {
        "domain": os.environ.get("AUTH0_DOMAIN", "").strip(),
        "client_id": os.environ.get("AUTH0_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("AUTH0_CLIENT_SECRET", "").strip(),
        "callback_url": os.environ.get("AUTH0_CALLBACK_URL", "").strip(),
    }


def authorize_url(state: str, connection: str | None = None) -> str:
    """Build the /authorize URL to redirect users to."""
    s = settings()
    if not s["domain"] or not s["client_id"] or not s["callback_url"]:
        raise RuntimeError("Auth0 is not configured")
    params = {
        "response_type": "code",
        "client_id": s["client_id"],
        "redirect_uri": s["callback_url"],
        "scope": "openid profile email",
        "state": state,
    }
    if connection:
        params["connection"] = connection
    return f"https://{s['domain']}/authorize?{urlencode(params)}"


async def exchange_code(code: str) -> dict[str, Any]:
    """Exchange an authorization code for an access token."""
    s = settings()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"https://{s['domain']}/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": s["client_id"],
                "client_secret": s["client_secret"],
                "code": code,
                "redirect_uri": s["callback_url"],
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        resp.raise_for_status()
        return resp.json()


async def get_userinfo(access_token: str) -> dict[str, Any]:
    """Fetch the user profile (email, name, picture, sub) from Auth0."""
    s = settings()
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"https://{s['domain']}/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        resp.raise_for_status()
        return resp.json()
