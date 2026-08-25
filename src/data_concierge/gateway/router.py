"""FastAPI router for the Gateway layer."""

import asyncio
import os
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.core.models import (
    QueryTier,
)
from data_concierge.data_layer.storage import GCSStorage, storage
from data_concierge.gateway import approved_members as approved_members_store
from data_concierge.gateway import auth0_client, query_logs
from data_concierge.gateway import chats as chats_store
from data_concierge.gateway import ckan_sites as ckan_sites_store
from data_concierge.gateway import feedback as feedback_store
from data_concierge.gateway import roles as roles_store
from data_concierge.gateway.github_publisher import build_blob_url, load_github_settings
from data_concierge.gateway.intent_classifier import IntentClassifier, intent_classifier
from data_concierge.gateway.match_verifier import (
    llm_gate_available,
    verify_match_with_llm,
)
from data_concierge.gateway.session import Session, SessionManager, session_manager
from data_concierge.gateway.verified_notebooks import (
    ReviewStatus,
    VerifiedAnswer,
    approve_notebook,
    approve_quick_answer,
    collapse_answer_submission_as_duplicate,
    collapse_notebook_submission_as_duplicate,
    dedupe_verified_library,
    extract_keywords,
    find_verified_answer_by_question,
    find_verified_notebook_by_question,
    get_all_submissions,
    get_answer_submission,
    get_answer_submissions,
    get_pending_submissions,
    get_stats,
    get_submission,
    get_verified_answer,
    get_verified_answers,
    get_verified_notebook,
    get_verified_notebook_by_submission,
    get_verified_notebooks,
    increment_answer_usage,
    increment_usage,
    reject_notebook,
    reject_quick_answer,
    search_verified_answers,
    search_verified_notebooks,
    submit_notebook,
    submit_quick_answer,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["concierge"])

# Storage prefix for generated notebooks
_NOTEBOOKS_PREFIX = "notebooks"

# =============================================================================
# User Authentication & Management
# =============================================================================

_USERS_KEY = "users.json"
# token -> {"user": username/email, "auth_type": "password"|"auth0", "email": str|None}
_user_tokens: dict[str, dict[str, Any]] = {}
# Auth0 OAuth state -> next_url (CSRF protection + post-login redirect)
_auth0_states: dict[str, str] = {}


def _load_users() -> dict[str, str]:
    """Load users from persistent storage. Returns {username: hashed_password}."""
    import hashlib

    data = storage.read_json(_USERS_KEY)
    if data and data.get("users"):
        return data["users"]
    # Seed with default user from env var (backward compat)
    default_pw = os.environ.get("USER_PASSWORD", "datHere@123")
    default_hash = hashlib.sha256(default_pw.encode()).hexdigest()
    users = {"user": default_hash}
    storage.write_json(_USERS_KEY, {"users": users})
    return users


def _save_users(users: dict[str, str]) -> None:
    """Save users to persistent storage."""
    storage.write_json(_USERS_KEY, {"users": users})


def _hash_password(password: str) -> str:
    import hashlib

    return hashlib.sha256(password.encode()).hexdigest()


def _get_user_token(request: Request) -> str | None:
    """Extract user auth token from cookie or Authorization header."""
    # Check cookie first
    token = request.cookies.get("user_token")
    if token and token in _user_tokens:
        return token
    # Check Authorization header (for API clients)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token in _user_tokens:
            return token
    return None


def get_current_user(request: Request) -> dict[str, Any] | None:
    """Return the user info dict for the current request, or None if not logged in."""
    token = _get_user_token(request)
    if not token:
        return None
    return _user_tokens.get(token)


def require_auth(request: Request) -> str:
    """Dependency that requires a valid user auth token. Returns the token."""
    token = _get_user_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required to process new queries. Verified notebooks are available without login.",
        )
    return token


def require_admin(request: Request) -> dict[str, Any]:
    """Dependency that requires the caller to be a logged-in admin.

    Returns the user info dict. Raises 401 if not logged in, 403 if not admin.
    Checks both the user ID (e.g. GitHub username) and email against the admin
    store, since admin roles may be stored under either identity.
    """
    token = _get_user_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login required",
        )
    user_info = _user_tokens[token]
    user_id = user_info.get("user", "")
    email = user_info.get("email", "")
    if not (roles_store.is_admin(user_id) or (email and roles_store.is_admin(email))):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user_info


def _reviewer_identity(admin: dict[str, Any]) -> str:
    """Resolve the reviewing admin's GitHub identity for the audit trail (#112).

    Prefers the authenticated user id (e.g. the GitHub/Auth0 username), then
    falls back to the email, then to a generic ``admin`` label. This is the
    server-trusted identity recorded in GitHub commit messages — derived from
    the session, not from any client-supplied ``reviewed_by`` field.
    """
    return admin.get("user") or admin.get("email") or "admin"


def _create_session_token(
    user: str,
    auth_type: str,
    email: str | None = None,
    display_name: str | None = None,
) -> str:
    """Mint a new session token attributing it to a user identity."""
    token = secrets.token_hex(32)
    _user_tokens[token] = {
        "user": user,
        "auth_type": auth_type,
        "email": email,
        "display_name": display_name or user,
    }
    return token


def _cookie_secure() -> bool:
    """Whether the session cookie should carry the ``Secure`` flag (issue #92).

    Enabled outside local development so the token is only ever sent over
    HTTPS. Disabled in ``development`` so login still works over plain HTTP on
    localhost.
    """
    return settings.environment != "development"


# =============================================================================
# Request/Response Models
# =============================================================================


class ConversationTurn(BaseModel):
    """One prior chat turn, sent by the UI so follow-ups can be understood."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., max_length=8000)


class QueryRequest(BaseModel):
    """Request model for submitting a query."""

    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    session_id: str | None = Field(default=None, description="Optional session ID for context")
    include_notebook: bool = Field(default=True, description="Include downloadable notebook")
    include_visualization: bool = Field(
        default=True, description="Include visualization if applicable"
    )
    data_source: str = Field(
        default="data_commons",
        description="Data source to use (data_commons, ckan, wprdc, census_mcp)",
    )
    concierge_mode: str = Field(
        default="analyze",
        description="Deprecated — queries always run in analysis mode. Kept for API backward compatibility.",
    )
    # Chat context (optional). When present, a low-latency classifier decides
    # whether this message is a new question (rewritten to be self-contained)
    # or a request to revise the previous answer's notebook, which routes to
    # the notebook-editing path instead of a fresh analysis.
    conversation: list[ConversationTurn] | None = Field(
        default=None,
        max_length=12,
        description="Recent chat turns, oldest first, for follow-up understanding",
    )
    previous_query_id: str | None = Field(
        default=None,
        max_length=64,
        description="query_id of the most recent answer in this chat that has a notebook",
    )


class QueryResponse(BaseModel):
    """Response model for query results."""

    query_id: str
    answer: str
    confidence: float
    confidence_level: str
    # A factor is ``None`` when it could not be computed; ``confidence_unavailable``
    # says why, keyed by the same factor name (issue #132).
    confidence_breakdown: dict[str, float | None] | None = None
    confidence_unavailable: dict[str, str] = Field(
        default_factory=dict,
        description="Factor name -> why it could not be computed",
    )
    confidence_measured_weight: float | None = Field(
        default=None,
        description="Fraction of the total factor weight actually measured",
    )
    confidence_explanation: str | None = Field(
        default=None,
        description="Plain-language note on what the score does not cover",
    )
    verification_pending: bool = Field(
        default=False,
        description="Notebook verification was scheduled; poll "
        "/notebooks/{query_id}/verification for the updated score",
    )
    tier: str
    intent: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    visualization: dict[str, Any] | None = None
    notebook_url: str | None = None
    notebook: dict[str, Any] | None = None
    # Quick answer fields — populated for TIER_1 factual lookups
    is_quick_answer: bool = Field(
        default=False, description="True when response is a quick one-line answer (no notebook)"
    )
    quick_answer: str | None = Field(default=None, description="Concise one-line factual answer")
    source_links: list[dict[str, str]] = Field(
        default_factory=list,
        description="Direct links to source data pages [{name, url, description}]",
    )
    clarification_needed: bool = False
    clarification_question: str | None = None
    alternatives: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(
        default_factory=list,
        description="Alternative questions to try when the analysis fails or has no answer",
    )
    # Notebook revision (chat follow-up editing). When true, this response's
    # notebook is an edited version of the notebook from
    # ``revised_from_query_id``; the original is kept untouched for audit.
    is_revision: bool = False
    revised_from_query_id: str | None = None
    processing_time_ms: int


class ClassificationResponse(BaseModel):
    """Response model for query classification."""

    query: str
    intent: str
    intent_confidence: float
    tier: str
    tier_confidence: float
    combined_confidence: float


class SessionResponse(BaseModel):
    """Response model for session operations."""

    session_id: str
    created: bool = False
    message: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    active_sessions: int


class NotebookSubmitRequest(BaseModel):
    """Request model for submitting a notebook for admin review."""

    query: str = Field(..., min_length=1, description="Original user query")
    answer: str = Field(default="", description="Generated answer text")
    notebook_json: dict[str, Any] = Field(..., description="Full notebook JSON content")
    filename: str = Field(default="", description="Notebook filename")
    submitted_by: str = Field(default="anonymous", description="User identifier")
    data_source: str = Field(default="", description="Data source used")
    confidence: float = Field(default=0.0, description="Confidence score")
    tags: list[str] | None = Field(default=None, description="Optional tags")
    agent_log: list[dict[str, Any]] | None = Field(default=None, description="Agent execution log")
    query_id: str | None = Field(
        default=None, description="Original query ID (for fetching stored logs)"
    )


class NotebookReviewRequest(BaseModel):
    """Request model for reviewing (approve/reject) a notebook or answer.

    ``admin_notes`` is the mandatory reason for the decision (#112). It is
    required for every approve/reject action — notebooks *and* answers, since
    all four review endpoints share this model — and is written into the
    GitHub commit message so the audit trail records *why* an artifact was
    approved or rejected, alongside the reviewer's GitHub identity.
    """

    reviewed_by: str = Field(default="admin", description="Admin identifier")
    admin_notes: str = Field(
        ...,
        min_length=1,
        description="Reason for the review decision (required, #112)",
    )


class VerifiedNotebookSearchRequest(BaseModel):
    """Request model for searching verified notebooks."""

    query: str = Field(..., min_length=1, description="Search query")
    threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum similarity")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results")


class QuickAnswerSubmitRequest(BaseModel):
    """Request model for submitting a quick answer for admin review."""

    query: str = Field(..., min_length=1, description="Original user query")
    answer: str = Field(..., min_length=1, description="One-line factual answer")
    source_links: list[dict[str, str]] = Field(
        ..., description="Direct links to source data pages [{name, url, description}]"
    )
    submitted_by: str = Field(default="anonymous", description="User identifier")
    data_source: str = Field(default="", description="Data source used")
    confidence: float = Field(default=0.0, description="Confidence score")
    tags: list[str] | None = Field(default=None, description="Optional tags")
    variable: str = Field(default="", description="Statistical variable queried")
    place: str = Field(default="", description="Geographic place queried")
    date: str = Field(default="", description="Date of the observation")
    value: str = Field(default="", description="The data value")


class VerifiedAnswerSearchRequest(BaseModel):
    """Request model for searching verified quick answers."""

    query: str = Field(..., min_length=1, description="Search query")
    threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Minimum similarity")
    max_results: int = Field(default=5, ge=1, le=50, description="Max results")


class LoginRequest(BaseModel):
    """Request model for user login."""

    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


class AddUserRequest(BaseModel):
    """Request model for adding a user."""

    username: str = Field(..., min_length=1, max_length=50, description="Username")
    password: str = Field(..., min_length=4, max_length=100, description="Password")


class ConciergeRequest(BaseModel):
    """Request model for concierge-only queries."""

    query: str = Field(
        ..., min_length=1, max_length=2000, description="The user's question about data"
    )
    session_id: str | None = Field(default=None, description="Optional session ID for context")


class DataSourceRecommendation(BaseModel):
    """A single data source recommendation."""

    source_id: str
    source_name: str
    description: str
    relevance_score: float
    portal_url: str | None = None
    api_url: str | None = None
    recommended_datasets: list[dict[str, Any]] = Field(default_factory=list)
    access_instructions: str | None = None


class ConciergeResponse(BaseModel):
    """Response model for concierge queries."""

    query_id: str
    query: str
    answer: str = Field(..., description="Conversational response with recommendations")
    recommendations: list[DataSourceRecommendation] = Field(default_factory=list)
    offer_analysis: bool = Field(default=False, description="Whether deeper analysis is available")
    processing_time_ms: int


# =============================================================================
# Dependencies
# =============================================================================


def get_session_manager() -> SessionManager:
    """Dependency to get session manager."""
    return session_manager


def get_intent_classifier() -> IntentClassifier:
    """Dependency to get intent classifier."""
    return intent_classifier


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check(
    sessions: SessionManager = Depends(get_session_manager),
) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        active_sessions=await sessions.get_active_session_count(),
    )


# =============================================================================
# Auth Endpoints
# =============================================================================


@router.post("/auth/login")
async def login(request: LoginRequest) -> JSONResponse:
    """Log in with username and password. Returns a session token via cookie."""
    users = _load_users()
    pw_hash = _hash_password(request.password)
    if users.get(request.username) != pw_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = _create_session_token(user=request.username, auth_type="password")
    response = JSONResponse(
        content={"message": "Login successful", "authenticated": True, "username": request.username}
    )
    response.set_cookie(
        "user_token",
        token,
        httponly=True,
        samesite="strict",
        secure=_cookie_secure(),
        max_age=86400,
    )
    return response


@router.post("/auth/logout")
async def logout(request: Request) -> JSONResponse:
    """Log out and invalidate session token."""
    token = _get_user_token(request)
    if token:
        _user_tokens.pop(token, None)
    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie("user_token")
    return response


@router.get("/auth/status")
async def auth_status(request: Request) -> dict[str, Any]:
    """Check if the current user is authenticated."""
    user = get_current_user(request)
    if not user:
        return {"authenticated": False, "auth0_enabled": auth0_client.is_enabled()}
    user_id = user.get("user", "")
    email = user.get("email", "")
    return {
        "authenticated": True,
        "username": user.get("display_name") or user_id,
        "auth_type": user.get("auth_type"),
        "email": email,
        "is_admin": roles_store.is_admin(user_id) or (bool(email) and roles_store.is_admin(email)),
        "auth0_enabled": auth0_client.is_enabled(),
    }


# =============================================================================
# Auth0 Social Login
# =============================================================================


@router.get("/auth/auth0/login")
async def auth0_login(connection: str | None = None, next: str = "/") -> RedirectResponse:
    """Redirect user to Auth0's hosted login page."""
    if not auth0_client.is_enabled():
        raise HTTPException(status_code=400, detail="Auth0 is not enabled on this server")
    state = secrets.token_hex(16)
    _auth0_states[state] = next or "/"
    try:
        url = auth0_client.authorize_url(state=state, connection=connection)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return RedirectResponse(url=url, status_code=302)


@router.get("/auth/auth0/callback")
async def auth0_callback(
    code: str | None = None, state: str | None = None, error: str | None = None
) -> RedirectResponse:
    """Handle the Auth0 redirect. Validates the approved-members gate."""
    if error:
        return RedirectResponse(url=f"/auth/pending?reason={error}", status_code=302)
    if not code or not state or state not in _auth0_states:
        raise HTTPException(status_code=400, detail="Invalid Auth0 callback")
    next_url = _auth0_states.pop(state, "/")

    try:
        token_data = await auth0_client.exchange_code(code)
        access_token = token_data.get("access_token")
        if not access_token:
            raise RuntimeError("No access token returned from Auth0")
        profile = await auth0_client.get_userinfo(access_token)
    except Exception as e:
        logger.error("Auth0 login failed", error=str(e))
        raise HTTPException(status_code=502, detail="Auth0 login failed") from e

    email = (profile.get("email") or "").lower()
    # Auth0 GitHub connection provides 'nickname' as the GitHub username
    nickname = profile.get("nickname") or ""
    name = profile.get("name") or nickname or email or "user"

    if not email:
        return RedirectResponse(url="/auth/pending?reason=no_email", status_code=302)

    if not approved_members_store.is_approved(email):
        approved_members_store.add_pending_request(email, name)
        logger.info("Auth0 login rejected: not approved", email=email)
        return RedirectResponse(url=f"/auth/pending?email={email}", status_code=302)

    # Prefer GitHub username (nickname) as the primary user identity
    display_name = nickname or name

    # Update the approved member entry with the GitHub username so the admin
    # panel can show it alongside the email.
    if nickname:
        approved_members_store.add_member(email, display_name=nickname)
        # Also update the admin role entry if this user is an admin
        if roles_store.is_admin(email):
            roles_store.grant_role(email, roles_store.ADMIN_ROLE, display_name=nickname)

    token = _create_session_token(
        user=nickname or email,
        auth_type="auth0",
        email=email,
        display_name=display_name,
    )
    logger.info("Auth0 login success", email=email, nickname=nickname, name=name)

    response = RedirectResponse(url=next_url, status_code=302)
    response.set_cookie(
        "user_token",
        token,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
        max_age=86400,
    )
    return response


# =============================================================================
# Approved Members Management (admin)
# =============================================================================


class ApprovedMemberRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=200)


@router.get("/admin/approved-members")
async def list_approved_members(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    members = approved_members_store.list_members()
    return {"count": len(members), "members": members}


@router.post("/admin/approved-members")
async def add_approved_member(
    request: ApprovedMemberRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    entry = approved_members_store.add_member(request.email)
    return {"message": "Member added", "member": entry}


@router.delete("/admin/approved-members/{email}")
async def delete_approved_member(
    email: str,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    ok = approved_members_store.remove_member(email)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Member '{email}' not found")
    return {"message": f"Member '{email}' removed"}


@router.get("/admin/pending-requests")
async def list_pending_requests(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """List pending access requests from unapproved Auth0 users."""
    requests = approved_members_store.list_pending_requests()
    return {"count": len(requests), "requests": requests}


@router.post("/admin/pending-requests/{email}/approve")
async def approve_pending_request(
    email: str,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Approve a pending request — adds the email to the approved list."""
    # Carry the display name from the pending request to the approved member
    pending = approved_members_store.list_pending_requests()
    display_name = ""
    for req in pending:
        if req.get("email", "").strip().lower() == email.strip().lower():
            display_name = req.get("name", "")
            break
    entry = approved_members_store.add_member(
        email,
        added_by=_admin.get("display_name") or _admin.get("user", "admin"),
        display_name=display_name,
    )
    approved_members_store.remove_pending_request(email)
    return {"message": f"'{email}' approved", "member": entry}


@router.delete("/admin/pending-requests/{email}")
async def dismiss_pending_request(
    email: str,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Dismiss a pending request without approving."""
    ok = approved_members_store.remove_pending_request(email)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Pending request for '{email}' not found")
    return {"message": f"Pending request for '{email}' dismissed"}


# =============================================================================
# Admin Role Management (RBAC)
# =============================================================================


class GrantAdminRequest(BaseModel):
    user_id: str = Field(
        ..., min_length=1, max_length=200, description="Username or email to grant admin"
    )


@router.get("/admin/roles")
async def list_admin_roles(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """List all users with the admin role."""
    admins = roles_store.list_admins()
    return {"count": len(admins), "admins": admins}


@router.post("/admin/roles")
async def grant_admin_role(
    request: GrantAdminRequest,
    admin_user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Grant admin role to a user."""
    entry = roles_store.grant_role(
        request.user_id,
        roles_store.ADMIN_ROLE,
        granted_by=admin_user.get("user", "admin"),
    )
    return {"message": f"Admin role granted to '{request.user_id}'", "entry": entry}


@router.delete("/admin/roles/{user_id:path}")
async def revoke_admin_role(
    user_id: str,
    admin_user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Revoke admin role from a user."""
    # Prevent removing yourself
    if user_id.strip().lower() == admin_user.get("user", "").strip().lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot revoke your own admin access",
        )
    ok = roles_store.revoke_role(user_id, roles_store.ADMIN_ROLE)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' does not have admin role",
        )
    return {"message": f"Admin role revoked from '{user_id}'"}


# =============================================================================
# CKAN Sites Registry (admin)
# =============================================================================


class AddCkanSiteRequest(BaseModel):
    """Request body for adding a new CKAN site to the registry."""

    url: str = Field(..., min_length=4, max_length=400, description="CKAN portal URL")
    name: str = Field(..., min_length=1, max_length=200, description="Display name")
    site_id: str | None = Field(
        default=None,
        description="Optional stable ID; auto-generated from the name if omitted",
        max_length=80,
    )
    organization: str | None = Field(
        default=None,
        description="Optional CKAN organization slug to use as a default filter",
        max_length=200,
    )
    description: str = Field(default="", max_length=2000, description="Portal description")
    quality_score: float = Field(default=0.85, ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list, description="Search keywords")


class UpdateCkanSiteRequest(BaseModel):
    """Partial update payload for an existing CKAN site."""

    url: str | None = None
    name: str | None = None
    organization: str | None = None
    description: str | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    keywords: list[str] | None = None


@router.get("/admin/ckan-sites")
async def list_ckan_sites(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """List every CKAN site the agent is configured to query."""
    sites = ckan_sites_store.list_sites()
    return {"count": len(sites), "sites": sites}


@router.post("/admin/ckan-sites")
async def add_ckan_site(
    request: AddCkanSiteRequest,
    admin_user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Register a new CKAN portal that the agent should check during queries."""
    try:
        entry = ckan_sites_store.add_site(
            url=request.url,
            name=request.name,
            site_id=request.site_id,
            organization=request.organization,
            description=request.description,
            quality_score=request.quality_score,
            keywords=request.keywords,
            added_by=admin_user.get("user", "admin"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return {"message": f"CKAN site '{entry['id']}' added", "site": entry}


@router.put("/admin/ckan-sites/{site_id}")
async def update_ckan_site(
    site_id: str,
    request: UpdateCkanSiteRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Update an existing CKAN site entry."""
    updates = request.model_dump(exclude_none=True)
    entry = ckan_sites_store.update_site(site_id, updates)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CKAN site '{site_id}' not found",
        )
    return {"message": f"CKAN site '{site_id}' updated", "site": entry}


@router.delete("/admin/ckan-sites/{site_id}")
async def delete_ckan_site(
    site_id: str,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Remove a CKAN site from the registry."""
    if not ckan_sites_store.remove_site(site_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CKAN site '{site_id}' not found",
        )
    return {"message": f"CKAN site '{site_id}' removed"}


# Public read-only endpoint — lets the UI populate a picker without admin auth.
@router.get("/ckan-sites")
async def list_ckan_sites_public() -> dict[str, Any]:
    """Public list of CKAN sites (metadata only, no admin audit fields)."""
    sites = ckan_sites_store.list_sites()
    public = [
        {
            "id": s.get("id"),
            "url": s.get("url"),
            "name": s.get("name"),
            "organization": s.get("organization"),
            "description": s.get("description", ""),
            "quality_score": s.get("quality_score", 0.85),
        }
        for s in sites
    ]
    return {"count": len(public), "sites": public}


# =============================================================================
# Chat Session Persistence
# =============================================================================


@router.get("/chats")
async def list_user_chats(
    request: Request,
    _token: str = Depends(require_auth),
) -> dict[str, Any]:
    """Return all saved chats for the current authenticated user.

    Exact-duplicate conversations are collapsed in a non-destructive read-time
    view (no client UI, nothing deleted from storage).
    """
    user_info = get_current_user(request) or {}
    user_id = user_info.get("user", "")
    return {"chats": chats_store.deduped_chats_view(user_id)}


@router.put("/chats/{chat_id}")
async def upsert_user_chat(
    chat_id: str,
    request: Request,
    _token: str = Depends(require_auth),
) -> dict[str, Any]:
    """Create or update a chat for the current authenticated user."""
    user_info = get_current_user(request) or {}
    user_id = user_info.get("user", "")
    body = await request.json()
    chat = await chats_store.save_chat(user_id, chat_id, body)
    return {"chat": chat}


@router.delete("/chats/{chat_id}")
async def delete_user_chat(
    chat_id: str,
    request: Request,
    _token: str = Depends(require_auth),
) -> dict[str, Any]:
    """Delete a chat for the current authenticated user."""
    user_info = get_current_user(request) or {}
    user_id = user_info.get("user", "")
    if not await chats_store.delete_chat(user_id, chat_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found",
        )
    return {"message": f"Chat '{chat_id}' deleted"}


# =============================================================================
# Query Logs (admin)
# =============================================================================


class QueryLogRequest(BaseModel):
    """Request body for POST /query-log — used for client-side verified-cache hits."""

    query: str = Field(..., min_length=1, max_length=2000)
    source: str = Field(
        default="verified_cache",
        description="'generated', 'verified_cache', or 'error'",
    )
    query_id: str | None = None
    confidence: float | None = None
    similarity_score: float | None = None
    verified_query: str | None = None
    notebook_url: str | None = None
    data_source: str | None = None
    is_quick_answer: bool = False
    had_notebook: bool = False


@router.post("/query-log")
async def post_query_log(request: Request, body: QueryLogRequest) -> dict[str, Any]:
    """Record a query-log entry — typically called by the frontend when a
    verified notebook is used so we can track cache hits alongside generated
    answers. The user is resolved from their session cookie; anonymous
    (logged-out) queries are attributed to "anonymous".
    """
    user = get_current_user(request)
    entry = query_logs.append_log(
        user=(user or {}).get("user") or "anonymous",
        auth_type=(user or {}).get("auth_type") or "anonymous",
        query=body.query,
        source=body.source,
        query_id=body.query_id,
        confidence=body.confidence,
        similarity_score=body.similarity_score,
        verified_query=body.verified_query,
        notebook_url=body.notebook_url,
        data_source=body.data_source,
        is_quick_answer=body.is_quick_answer,
        had_notebook=body.had_notebook,
    )
    return {"message": "logged", "id": entry["id"]}


@router.get("/admin/query-logs")
async def admin_list_query_logs(
    limit: int = 200,
    user: str | None = None,
    source: str | None = None,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    logs = query_logs.list_logs(limit=limit, user=user, source=source)
    return {"count": len(logs), "logs": logs, "summary": query_logs.summary()}


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback — a 👍/👎 on one assistant answer."""

    rating: str = Field(..., description="'up' or 'down'")
    query: str = Field(default="", max_length=2000)
    answer: str | None = Field(default=None, max_length=4000)
    query_id: str | None = None
    source: str | None = Field(default=None, description="'generated' | 'verified_cache'")
    note: str | None = Field(default=None, max_length=2000)


@router.post("/feedback")
async def post_feedback(request: Request, body: FeedbackRequest) -> dict[str, Any]:
    """Record 👍/👎 feedback on an answer. Open to anyone viewing an answer
    (attributed to the session user, else 'anonymous') — same trust model as
    POST /query-log."""
    if body.rating not in ("up", "down"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="rating must be 'up' or 'down'"
        )
    user = get_current_user(request)
    entry = feedback_store.append_feedback(
        rating=body.rating,
        query=body.query,
        answer_preview=body.answer,
        user=(user or {}).get("user") or "anonymous",
        auth_type=(user or {}).get("auth_type") or "anonymous",
        query_id=body.query_id,
        source=body.source,
        note=body.note,
    )
    return {"message": "feedback recorded", "id": entry["id"]}


@router.get("/admin/feedback")
async def admin_list_feedback(
    limit: int = 200,
    rating: str | None = None,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    items = feedback_store.list_feedback(limit=limit, rating=rating)
    return {"count": len(items), "feedback": items, "summary": feedback_store.summary()}


@router.get("/admin/notebook-reviews")
async def admin_list_notebook_reviews(
    limit: int = 100,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Notebook verification + adversarial review results (admin only).

    One record per generated notebook: whether it executed, whether its
    output reconciles with the answer, the adversarial method-review
    findings, and how the combined verdict moved the confidence score.
    """
    from data_concierge.gateway.notebook_verification import (
        list_verifications,
        review_summary,
    )

    # Blocking storage I/O (one read per record) — keep it off the event loop.
    records = await asyncio.to_thread(list_verifications, 10000)
    return {
        "count": len(records),
        "reviews": records[: max(0, limit)],
        "summary": review_summary(records),
        "enabled": settings.notebook_verification_enabled,
        "review_enabled": settings.notebook_review_enabled,
    }


# =============================================================================
# User Management (admin-only)
# =============================================================================


@router.get("/admin/users")
async def list_users(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """List all users (admin endpoint). Passwords are not returned."""
    users = _load_users()
    return {
        "count": len(users),
        "users": [{"username": u} for u in sorted(users.keys())],
    }


@router.post("/admin/users")
async def add_user(
    request: AddUserRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Add a new user (admin endpoint)."""
    users = _load_users()
    if request.username in users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User '{request.username}' already exists",
        )
    users[request.username] = _hash_password(request.password)
    _save_users(users)
    logger.info("User added", username=request.username)
    return {"message": f"User '{request.username}' created", "username": request.username}


@router.delete("/admin/users/{username}")
async def delete_user(
    username: str,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Delete a user (admin endpoint)."""
    users = _load_users()
    if username not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found",
        )
    del users[username]
    _save_users(users)
    logger.info("User deleted", username=username)
    return {"message": f"User '{username}' deleted"}


@router.put("/admin/users/{username}/password")
async def reset_user_password(
    username: str,
    request: AddUserRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Reset a user's password (admin endpoint)."""
    users = _load_users()
    if username not in users:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{username}' not found",
        )
    users[username] = _hash_password(request.password)
    _save_users(users)
    logger.info("Password reset", username=username)
    return {"message": f"Password reset for '{username}'"}


@router.post("/session", response_model=SessionResponse)
async def create_session(
    user_id: str | None = None,
    sessions: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Create a new session."""
    session = await sessions.create_session(user_id=user_id)
    return SessionResponse(
        session_id=session.session_id,
        created=True,
        message="Session created successfully",
    )


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    sessions: SessionManager = Depends(get_session_manager),
) -> SessionResponse:
    """Get session information."""
    session = await sessions.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired",
        )
    return SessionResponse(
        session_id=session.session_id,
        created=False,
        message="Session is active",
    )


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    sessions: SessionManager = Depends(get_session_manager),
) -> dict[str, str]:
    """Delete a session."""
    if await sessions.delete_session(session_id):
        return {"message": "Session deleted successfully"}
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )


@router.post("/classify", response_model=ClassificationResponse)
async def classify_query(
    request: QueryRequest,
    classifier: IntentClassifier = Depends(get_intent_classifier),
) -> ClassificationResponse:
    """Classify a query without processing it.

    Useful for understanding how the system will route a query.
    """
    result = classifier.classify(request.query)
    return ClassificationResponse(
        query=request.query,
        intent=result["intent"].value,
        intent_confidence=result["intent_confidence"],
        tier=result["tier"].value,
        tier_confidence=result["tier_confidence"],
        combined_confidence=result["combined_confidence"],
    )


@router.post("/concierge", response_model=ConciergeResponse)
async def concierge_endpoint(
    request: ConciergeRequest,
    _token: str = Depends(require_auth),
    sessions: SessionManager = Depends(get_session_manager),
) -> ConciergeResponse:
    """Data Concierge endpoint - get recommendations for data sources.

    This endpoint acts as a helpful data concierge that:
    1. Understands what data you're looking for
    2. Recommends relevant data sources and datasets
    3. Provides direct links to data portals
    4. Offers to perform deeper analysis if needed

    Use this when you want to:
    - Find out where to get specific data
    - Discover which agencies have the data you need
    - Get links to data portals and APIs
    - Understand what data is available

    For actual data analysis, use the /query endpoint with concierge_mode='analyze'.
    """
    from data_concierge.agents.supervisor import process_concierge_query

    start_time = time.time()
    query_id = str(uuid.uuid4())

    logger.info("Concierge query", query_id=query_id, query=request.query[:100])

    try:
        # Get session if provided
        session = None
        if request.session_id:
            session = await sessions.get_session(request.session_id)
            if session:
                await sessions.update_session(request.session_id, query=request.query)

        # Process through concierge
        result = await process_concierge_query(request.query, session)

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ConciergeResponse(
            query_id=query_id,
            query=request.query,
            answer=result.get("answer", ""),
            recommendations=[],  # Will be populated by the concierge agent
            offer_analysis=result.get("offer_analysis", False),
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error("Concierge error", error=str(e), query_id=query_id)
        processing_time_ms = int((time.time() - start_time) * 1000)

        return ConciergeResponse(
            query_id=query_id,
            query=request.query,
            answer=f"I apologize, but I encountered an error while searching for data sources: {str(e)}",
            offer_analysis=False,
            processing_time_ms=processing_time_ms,
        )


# Minimum similarity score for server-side verified cache hits. Kept as the
# strict keyword-only bar used when the LLM verification gate is unavailable.
_VERIFIED_SIMILARITY_THRESHOLD = 0.50


async def _check_verified_answer(query: str) -> dict[str, Any] | None:
    """Check if a verified quick answer matches the query.

    Two-stage matching (issue #64):

    * **Stage 1 — keyword candidates.** Cheap Jaccard/coverage retrieval with a
      wide net (``verified_match_candidate_threshold``) surfaces a handful of
      topically-similar answers.
    * **Stage 2 — LLM gate.** A low-latency model judges whether each candidate
      *actually* answers the new query, rejecting temporal/geographic/variable
      mismatches that keyword overlap lets through. The first candidate that
      passes with sufficient confidence wins.

    If the LLM gate is unavailable (disabled, no API key, or circuit breaker
    open) it falls back to strict keyword-only matching at
    ``verified_match_fallback_threshold``.

    Returns a dict with answer details on a confirmed match, else None.
    """
    candidates = search_verified_answers(
        query=query,
        threshold=settings.verified_match_candidate_threshold,
        max_results=settings.verified_match_max_candidates,
    )
    if not candidates:
        return None

    gate_active = llm_gate_available()

    for candidate in candidates:
        if gate_active:
            verdict = await verify_match_with_llm(
                user_query=query,
                verified_query=candidate.query,
                verified_answer=candidate.answer,
            )
            if verdict is None:
                # Gate errored mid-loop; switch to keyword fallback for the rest.
                gate_active = False
            elif (
                verdict["is_match"]
                and verdict["confidence"] >= settings.verified_match_confidence_threshold
            ):
                return _build_answer_match(candidate, verdict["confidence"], verdict["reason"])
            else:
                continue  # rejected by the gate — try the next candidate

        # Keyword-only fallback path (gate disabled or errored).
        if not gate_active and (
            candidate.similarity_score >= settings.verified_match_fallback_threshold
        ):
            return _build_answer_match(candidate, candidate.similarity_score, None)

    return None


def _build_answer_match(candidate: Any, similarity: float, reason: str | None) -> dict[str, Any]:
    """Materialise a confirmed verified-answer match (fetch links, bump usage)."""
    full = get_verified_answer(candidate.answer_id)
    increment_answer_usage(candidate.answer_id)
    result: dict[str, Any] = {
        "answer_id": candidate.answer_id,
        "answer": candidate.answer,
        "original_query": candidate.query,
        "similarity": similarity,
        "source_links": full.source_links if full else candidate.source_links,
    }
    if reason:
        result["match_reason"] = reason
    return result


async def _check_verified_notebook(query: str) -> dict[str, Any] | None:
    """Check if a verified notebook matches the query.

    Uses the same two-stage (keyword candidates + LLM gate) strategy as
    :func:`_check_verified_answer`. See its docstring for details.

    Returns a dict with notebook details on a confirmed match, else None.
    """
    candidates = search_verified_notebooks(
        query=query,
        threshold=settings.verified_match_candidate_threshold,
        max_results=settings.verified_match_max_candidates,
    )
    if not candidates:
        return None

    gate_active = llm_gate_available()

    for candidate in candidates:
        if gate_active:
            verdict = await verify_match_with_llm(
                user_query=query,
                verified_query=candidate.query,
                verified_answer=candidate.answer,
            )
            if verdict is None:
                gate_active = False
            elif (
                verdict["is_match"]
                and verdict["confidence"] >= settings.verified_match_confidence_threshold
            ):
                return _build_notebook_match(candidate, verdict["confidence"], verdict["reason"])
            else:
                continue

        if not gate_active and (
            candidate.similarity_score >= settings.verified_match_fallback_threshold
        ):
            return _build_notebook_match(candidate, candidate.similarity_score, None)

    return None


def _build_notebook_match(candidate: Any, similarity: float, reason: str | None) -> dict[str, Any]:
    """Materialise a confirmed verified-notebook match (bump usage)."""
    increment_usage(candidate.notebook_id)
    result: dict[str, Any] = {
        "notebook_id": candidate.notebook_id,
        "answer": candidate.answer,
        "original_query": candidate.query,
        "similarity": similarity,
    }
    if reason:
        result["match_reason"] = reason
    return result


@router.post("/query", response_model=QueryResponse)
async def process_query_endpoint(
    request: QueryRequest,
    http_request: Request,
    _token: str = Depends(require_auth),
    sessions: SessionManager = Depends(get_session_manager),
) -> QueryResponse:
    """Process a data query through the AI Concierge system.

    This is the main endpoint that:
    1. Checks verified notebooks/answers for a fast cached response
    2. If no cache hit, runs the full analysis pipeline
    3. Returns answer with citations and optional notebook
    """
    from data_concierge.agents.supervisor import process_query

    start_time = time.time()
    query_id = str(uuid.uuid4())

    logger.info(
        "Processing query",
        query_id=query_id,
        query=request.query[:100],
        data_source=request.data_source,
    )

    try:
        # ── Follow-up understanding (chat context) ───────────────────
        # Only runs when the UI sent conversation turns, so first messages
        # and API callers without context pay nothing. A revision request
        # routes to the notebook-editing path; a context-dependent question
        # ("what about Ohio?") is rewritten to be self-contained before the
        # cache checks and the pipeline see it.
        effective_query = request.query
        if request.conversation:
            from data_concierge.gateway.followup import MODE_REVISE, classify_followup

            followup_decision = await classify_followup(
                request.query,
                [turn.model_dump() for turn in request.conversation],
                has_notebook=bool(request.previous_query_id),
            )
            if followup_decision.mode == MODE_REVISE and request.previous_query_id:
                revision_response = await _process_revision(
                    request,
                    http_request,
                    query_id=query_id,
                    instruction=followup_decision.instruction,
                    start_time=start_time,
                )
                if revision_response is not None:
                    return revision_response
                # Revision was not possible (missing notebook, editor error):
                # degrade to a fresh analysis of the self-contained rewrite.
            if followup_decision.rewritten_query:
                effective_query = followup_decision.rewritten_query
                if effective_query != request.query:
                    logger.info(
                        "Follow-up rewritten to standalone query",
                        query_id=query_id,
                        original=request.query[:100],
                        rewritten=effective_query[:100],
                    )

        # ── Fast path: check verified answers first ──────────────────
        verified_answer_match = await _check_verified_answer(effective_query)
        if verified_answer_match:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Returning verified answer (server-side cache hit)",
                query_id=query_id,
                answer_id=verified_answer_match["answer_id"],
                similarity=verified_answer_match["similarity"],
            )
            _user_info = get_current_user(http_request) or {}
            query_logs.append_log(
                user=_user_info.get("display_name") or _user_info.get("user") or "unknown",
                auth_type=_user_info.get("auth_type") or "password",
                query=request.query,
                source="verified_cache",
                query_id=query_id,
                confidence=verified_answer_match["similarity"],
                similarity_score=verified_answer_match["similarity"],
                verified_query=verified_answer_match["original_query"],
                processing_time_ms=processing_time_ms,
                is_quick_answer=True,
                data_source=request.data_source,
            )
            return QueryResponse(
                query_id=query_id,
                answer=verified_answer_match["answer"],
                confidence=verified_answer_match["similarity"],
                confidence_level="verified",
                tier="tier_1",
                is_quick_answer=True,
                quick_answer=verified_answer_match["answer"],
                source_links=verified_answer_match.get("source_links", []),
                processing_time_ms=processing_time_ms,
            )

        # ── Fast path: check verified notebooks ─────────────────────
        verified_nb_match = await _check_verified_notebook(effective_query)
        if verified_nb_match:
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "Returning verified notebook (server-side cache hit)",
                query_id=query_id,
                notebook_id=verified_nb_match["notebook_id"],
                similarity=verified_nb_match["similarity"],
            )
            _user_info = get_current_user(http_request) or {}
            query_logs.append_log(
                user=_user_info.get("display_name") or _user_info.get("user") or "unknown",
                auth_type=_user_info.get("auth_type") or "password",
                query=request.query,
                source="verified_cache",
                query_id=query_id,
                confidence=verified_nb_match["similarity"],
                similarity_score=verified_nb_match["similarity"],
                verified_query=verified_nb_match["original_query"],
                notebook_url=f"/api/v1/verified-notebooks/{verified_nb_match['notebook_id']}/download",
                processing_time_ms=processing_time_ms,
                had_notebook=True,
                data_source=request.data_source,
            )
            return QueryResponse(
                query_id=query_id,
                answer=verified_nb_match["answer"],
                confidence=verified_nb_match["similarity"],
                confidence_level="verified",
                tier="tier_1",
                notebook_url=f"/api/v1/verified-notebooks/{verified_nb_match['notebook_id']}/download",
                processing_time_ms=processing_time_ms,
            )

        # ── Full analysis pipeline ──────────────────────────────────
        # Get or create session
        session: Session | None = None
        if request.session_id:
            session = await sessions.get_session(request.session_id)
            if session:
                await sessions.update_session(request.session_id, query=request.query)

        # Always run in analysis mode
        # (retry + fallback for transient 529 errors is handled inside llm_agent)
        # The whole pipeline runs under an admin-configurable timeout so a
        # stuck analysis returns a friendly answer instead of letting the
        # platform (Cloud Run) kill the connection with a generic error.
        from data_concierge.gateway.runtime_settings import get_query_timeout_seconds

        # One analysis budget for the whole request: whatever the follow-up
        # classifier and a declined/failed revision attempt already spent
        # comes out of it, so the total stays under the platform timeout and
        # the friendly timeout answer always beats a raw Cloud Run 504.
        remaining_budget = max(30, get_query_timeout_seconds() - int(time.time() - start_time))
        final_state = await asyncio.wait_for(
            process_query(
                effective_query,
                session,
                request.data_source,
            ),
            timeout=remaining_budget,
        )

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Extract results from state
        answer = final_state.get("answer") or ""
        suggested_questions: list[str] = []
        if not answer.strip():
            # Never return a blank/generic answer — recommend alternatives.
            answer = (
                "I wasn't able to find an answer to that question in the data I have "
                "access to. It may help to rephrase it with a specific place, time "
                "period, or metric — or try one of the suggestions below."
            )
            suggested_questions = _alternative_question_suggestions(request.query)
        elif final_state.get("error"):
            # The pipeline recovered with a fallback answer but the analysis
            # itself failed — offer alternative questions alongside it.
            suggested_questions = _alternative_question_suggestions(request.query)
        confidence = final_state.get("confidence")
        tier = final_state.get("tier", QueryTier.TIER_1)
        intent = final_state.get("intent")
        citations = final_state.get("citations", [])
        visualization = final_state.get("visualization")
        notebook = final_state.get("notebook")

        # Format confidence
        confidence_score = 0.0
        confidence_level = "unknown"
        confidence_breakdown = None
        confidence_unavailable: dict[str, str] = {}
        confidence_measured_weight: float | None = None
        confidence_explanation: str | None = None

        if confidence:
            confidence_score = confidence.final_score
            confidence_level = confidence.level.value
            confidence_unavailable = dict(confidence.unavailable)
            confidence_measured_weight = confidence.measured_weight
            confidence_explanation = confidence.explanation
            confidence_breakdown = {
                # New factor names (primary)
                "answer_grounding": confidence.answer_grounding,
                "data_retrieval_quality": confidence.data_retrieval_quality,
                "source_metadata_quality": confidence.source_metadata_quality,
                "query_answer_alignment": confidence.query_answer_alignment,
                "computation_complexity": confidence.computation_complexity,
                # Legacy aliases for backward compatibility
                "query_interpretation": confidence.query_interpretation,
                "source_authority": confidence.source_authority,
                "retrieval_match": confidence.retrieval_match,
                "data_recency": confidence.data_recency,
                "computation_reliability": confidence.computation_reliability,
            }

        # Format citations
        citations_list = []
        for citation in citations:
            citations_list.append(
                {
                    "source": citation.source.name,
                    "dataset": citation.dataset_title,
                    "url": citation.url,
                    "access_date": citation.access_date,
                    "footnote": citation.footnote_text,
                }
            )

        # Format visualization
        viz_dict = None
        if visualization and request.include_visualization:
            viz_dict = {
                "chart_type": visualization.chart_type,
                "spec": visualization.vega_lite_spec,
                "alt_text": visualization.alt_text,
            }

        # Check if this is a quick answer (TIER_1 factual lookup)
        is_quick_answer = final_state.get("quick_answer_mode", False)
        quick_answer_text = final_state.get("quick_answer") or None
        raw_source_links = final_state.get("source_links", [])

        # Validate source links have required keys
        source_links = []
        for link in raw_source_links:
            if isinstance(link, dict) and link.get("url"):
                source_links.append(
                    {
                        "name": link.get("name", "Source"),
                        "url": link["url"],
                        "description": link.get("description", ""),
                    }
                )

        # For quick answers: use quick_answer as the main answer if available
        if is_quick_answer and quick_answer_text:
            answer = quick_answer_text

        # Save notebook to storage (skip for quick answers)
        notebook_url = None
        notebook_data = None
        if notebook and request.include_notebook and not is_quick_answer:
            notebook_key = f"{_NOTEBOOKS_PREFIX}/{query_id}.ipynb"
            storage.write_json(notebook_key, notebook.notebook_json)
            logger.info("Notebook saved", key=notebook_key, filename=notebook.filename)
            notebook_url = f"/api/v1/notebooks/{query_id}"
            notebook_data = notebook.notebook_json

        # Save agent execution log alongside the notebook
        agent_log = final_state.get("agent_log", [])
        if agent_log:
            log_key = f"{_NOTEBOOKS_PREFIX}/{query_id}_log.json"
            storage.write_json(log_key, {"query": effective_query, "log": agent_log})
            logger.info("Agent log saved", key=log_key, entries=len(agent_log))

        # ── Auto-submit for review when confidence > 50% ───────────
        _user_info = get_current_user(http_request) or {}
        _submitted_by = _user_info.get("display_name") or _user_info.get("user") or "auto"

        # Aggregate token usage from agent log (needed for auto-submit and query log).
        # Prompt-side cost includes cache creation/read tokens per the evidence standard.
        total_input_tokens = 0
        total_output_tokens = 0
        for log_entry in agent_log:
            tokens = log_entry.get("tokens")
            if tokens:
                total_input_tokens += (
                    tokens.get("input", 0)
                    + tokens.get("cache_creation_input", 0)
                    + tokens.get("cache_read_input", 0)
                )
                total_output_tokens += tokens.get("output", 0)

        if is_quick_answer and quick_answer_text and confidence_score > 0.50:
            try:
                submit_quick_answer(
                    query=effective_query,
                    answer=quick_answer_text,
                    source_links=source_links,
                    submitted_by=_submitted_by,
                    data_source=request.data_source,
                    confidence=confidence_score,
                    input_tokens=total_input_tokens or None,
                    output_tokens=total_output_tokens or None,
                )
                logger.info(
                    "Auto-submitted quick answer for review",
                    query_id=query_id,
                    confidence=confidence_score,
                )
            except Exception as e:
                logger.warning("Auto-submit quick answer failed (non-blocking)", error=str(e))
        elif notebook_data and confidence_score > 0.50:
            try:
                submit_notebook(
                    query=effective_query,
                    answer=answer,
                    notebook_json=notebook_data,
                    filename=f"auto_{query_id}.ipynb",
                    submitted_by=_submitted_by,
                    data_source=request.data_source,
                    confidence=confidence_score,
                    input_tokens=total_input_tokens or None,
                    output_tokens=total_output_tokens or None,
                    query_id=query_id,
                )
                logger.info(
                    "Auto-submitted notebook for review",
                    query_id=query_id,
                    confidence=confidence_score,
                )
            except Exception as e:
                logger.warning("Auto-submit notebook failed (non-blocking)", error=str(e))

        if is_quick_answer:
            logger.info(
                "Returning quick answer",
                query_id=query_id,
                quick_answer=quick_answer_text,
                source_link_count=len(source_links),
            )

        # Persist per-user query log
        query_logs.append_log(
            user=_user_info.get("display_name") or _user_info.get("user") or "unknown",
            auth_type=_user_info.get("auth_type") or "password",
            query=request.query,
            source="generated",
            query_id=query_id,
            confidence=confidence_score,
            confidence_level=confidence_level,
            tier=tier.value if tier else None,
            intent=intent.value if intent else None,
            processing_time_ms=processing_time_ms,
            had_notebook=notebook_url is not None,
            is_quick_answer=is_quick_answer,
            notebook_url=notebook_url,
            data_source=request.data_source,
            input_tokens=total_input_tokens or None,
            output_tokens=total_output_tokens or None,
        )

        # Execute the generated notebook and check it re-derives the answer
        # (#131). Runs after this response is sent; the client polls
        # /notebooks/{query_id}/verification for the updated score. No-op
        # while notebook_verification_enabled is False.
        verification_pending = False
        try:
            from data_concierge.gateway.notebook_verification import schedule_verification

            verification_pending = schedule_verification(
                query_id,
                notebook_data,
                answer,
                confidence,
                query=effective_query,
                data_source=request.data_source,
            )
        except Exception as e:  # noqa: BLE001 - scheduling must never fail a query
            logger.warning("Could not schedule notebook verification", error=str(e))

        return QueryResponse(
            query_id=query_id,
            answer=answer,
            confidence=confidence_score,
            confidence_level=confidence_level,
            confidence_breakdown=confidence_breakdown,
            confidence_unavailable=confidence_unavailable,
            confidence_measured_weight=confidence_measured_weight,
            confidence_explanation=confidence_explanation,
            verification_pending=verification_pending,
            tier=tier.value if tier else "tier_1",
            intent=intent.value if intent else None,
            citations=citations_list,
            visualization=viz_dict,
            notebook_url=notebook_url,
            notebook=notebook_data,
            is_quick_answer=is_quick_answer,
            quick_answer=quick_answer_text,
            source_links=source_links,
            clarification_needed=final_state.get("needs_clarification", False),
            clarification_question=final_state.get("clarification_question"),
            suggested_questions=suggested_questions,
            processing_time_ms=processing_time_ms,
        )

    except TimeoutError:
        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.error(
            "Query processing timed out",
            query_id=query_id,
            timeout_ms=processing_time_ms,
        )
        _log_query_error(http_request, request, query_id, processing_time_ms, "timeout")
        return _graceful_error_response(
            query_id=query_id,
            answer=(
                "This question is taking longer to analyze than expected, so I stopped "
                "the search rather than keep you waiting. A more specific question "
                "usually finishes much faster — try narrowing it to one place, one "
                "time period, or one metric, or pick a suggestion below."
            ),
            query=request.query,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error("Query processing failed", error=str(e), query_id=query_id)
        processing_time_ms = int((time.time() - start_time) * 1000)
        _log_query_error(http_request, request, query_id, processing_time_ms, str(e))

        err_str = str(e)
        if "529" in err_str or "overloaded" in err_str.lower():
            user_msg = (
                "The AI service is temporarily overloaded. Please try again in a "
                "minute or two — your question is fine, this is on our side."
            )
        else:
            # Never surface raw exception text to the user — the details are in
            # the server logs and the admin query log.
            user_msg = (
                "Something went wrong on our side while analyzing your question, so I "
                "couldn't finish this one. Please try again in a moment — or try a "
                "different angle with one of the suggestions below."
            )

        return _graceful_error_response(
            query_id=query_id,
            answer=user_msg,
            query=request.query,
            processing_time_ms=processing_time_ms,
        )


def _alternative_question_suggestions(query: str) -> list[str]:
    """Follow-up questions to recommend when a query fails or has no answer.

    Each entry is shown as a clickable chip in the chat UI and re-submitted as
    a query, so they must be self-contained questions.
    """
    return [
        "What datasets are available on this topic?",
        "What data sources can I ask questions about?",
        "What are some example questions that work well?",
    ]


def _log_query_error(
    http_request: Request,
    request: "QueryRequest",
    query_id: str,
    processing_time_ms: int,
    error: str,
) -> None:
    """Record a failed query in the per-user query log (admin-visible)."""
    _user_info = get_current_user(http_request) or {}
    query_logs.append_log(
        user=_user_info.get("display_name") or _user_info.get("user") or "unknown",
        auth_type=_user_info.get("auth_type") or "password",
        query=request.query,
        source="error",
        query_id=query_id,
        processing_time_ms=processing_time_ms,
        data_source=request.data_source,
        error=error,
    )


def _graceful_error_response(
    query_id: str,
    answer: str,
    query: str,
    processing_time_ms: int,
) -> QueryResponse:
    """A friendly, suggestion-bearing response for any failed query."""
    return QueryResponse(
        query_id=query_id,
        answer=answer,
        confidence=0.0,
        confidence_level="error",
        tier="tier_1",
        suggested_questions=_alternative_question_suggestions(query),
        processing_time_ms=processing_time_ms,
    )


# query_ids are minted as uuid4 by this module; anything else in
# ``previous_query_id`` is either stale client state or path mischief, and in
# both cases the revision path must decline (it builds storage keys from it).
_SAFE_QUERY_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


async def _process_revision(
    request: QueryRequest,
    http_request: Request,
    *,
    query_id: str,
    instruction: str,
    start_time: float,
) -> QueryResponse | None:
    """Edit the previous answer's notebook per the user's follow-up.

    Returns a full ``QueryResponse`` when the revision path handled the
    message (successfully or with a friendly decline), or ``None`` when it
    could not run at all — the caller then degrades to a fresh analysis, so
    a revision request never hits a dead end.
    """
    # The editor's data tools are the LLM graph's CKAN/MCP toolset; for a
    # deterministic-graph source (Data Commons) they would silently fall
    # back to the WPRDC portal and edit a federal-statistics notebook
    # against Pittsburgh open data. Decline instead — the caller degrades
    # to a fresh analysis of the rewritten question.
    from data_concierge.agents.supervisor import is_llm_graph_source

    if not is_llm_graph_source(request.data_source):
        logger.info(
            "Revision not supported for deterministic-graph source",
            data_source=request.data_source,
        )
        return None

    prev_id = (request.previous_query_id or "").strip()
    if not _SAFE_QUERY_ID.match(prev_id):
        logger.warning("Revision requested with unusable previous_query_id", prev_id=prev_id[:80])
        return None

    prev_notebook = storage.read_json(f"{_NOTEBOOKS_PREFIX}/{prev_id}.ipynb")
    if not prev_notebook or not prev_notebook.get("cells"):
        logger.info("Revision requested but previous notebook not found", prev_id=prev_id)
        return None

    conversation = [turn.model_dump() for turn in (request.conversation or [])]

    # The question the notebook answers comes from its stored agent log —
    # authoritative even when the latest chat turns were about something
    # else (a verified cache hit, a quick answer). The conversation text
    # itself gives the editor the surrounding answers in correct order, so
    # no separately extracted "previous answer" is passed: pairing the last
    # assistant turn with an older notebook mismatches them.
    original_query = ""
    prev_log = storage.read_json(f"{_NOTEBOOKS_PREFIX}/{prev_id}_log.json")
    if isinstance(prev_log, dict):
        original_query = str(prev_log.get("query") or "")
    if not original_query:
        original_query = next(
            (t["content"] for t in reversed(conversation) if t["role"] == "user"), ""
        )

    from data_concierge.agents.notebook_editor import edit_notebook
    from data_concierge.gateway.followup import render_conversation
    from data_concierge.gateway.runtime_settings import get_query_timeout_seconds

    # The edit shares the request's single analysis budget: it gets what is
    # left of it (the follow-up classifier already spent some), and if it
    # fails the fresh-analysis fallback gets the remainder — never a second
    # full budget, which would blow past the platform (Cloud Run) timeout
    # and surface a raw 504 instead of the friendly timeout answer.
    remaining = max(30, get_query_timeout_seconds() - int(time.time() - start_time))
    edit_result = await asyncio.wait_for(
        edit_notebook(
            prev_notebook,
            instruction=instruction or request.query,
            query=original_query or request.query,
            previous_answer="",
            data_source=request.data_source,
            conversation_text=render_conversation(conversation),
        ),
        timeout=remaining,
    )

    processing_time_ms = int((time.time() - start_time) * 1000)

    if edit_result.error:
        logger.warning(
            "Notebook edit failed; degrading to fresh analysis",
            prev_id=prev_id,
            error=edit_result.error,
        )
        return None

    _user_info = get_current_user(http_request) or {}
    user_name = _user_info.get("display_name") or _user_info.get("user") or "unknown"

    if edit_result.notebook is None:
        # The editor made no changes but explained why — a legitimate chat
        # answer ("that dataset doesn't exist on this portal"), not an error.
        query_logs.append_log(
            user=user_name,
            auth_type=_user_info.get("auth_type") or "password",
            query=request.query,
            source="revision",
            query_id=query_id,
            processing_time_ms=processing_time_ms,
            data_source=request.data_source,
        )
        return QueryResponse(
            query_id=query_id,
            answer=edit_result.answer,
            confidence=0.0,
            confidence_level="unknown",
            tier="tier_2",
            revised_from_query_id=prev_id,
            suggested_questions=_alternative_question_suggestions(request.query),
            processing_time_ms=processing_time_ms,
        )

    answer = edit_result.answer
    notebook_json = edit_result.notebook

    storage.write_json(f"{_NOTEBOOKS_PREFIX}/{query_id}.ipynb", notebook_json)
    if edit_result.agent_log:
        storage.write_json(
            f"{_NOTEBOOKS_PREFIX}/{query_id}_log.json",
            {
                "query": request.query,
                "revised_from": prev_id,
                "instruction": instruction,
                "log": edit_result.agent_log,
            },
        )

    # Confidence for an edited notebook starts from the editor's own (thin)
    # signals; the heavyweight evidence arrives asynchronously when the
    # edited notebook is re-executed and adversarially reviewed, exactly as
    # for a fresh generation. Prior assistant answers count as grounding
    # context: a revision answer legitimately restates numbers the original
    # (already grounded and verified) run derived, and the editor's own tool
    # output alone would score those carried-over numbers as fabricated.
    from data_concierge.core.confidence import confidence_calculator

    prior_answer_texts = [
        t["content"][:4000] for t in conversation if t["role"] == "assistant" and t["content"]
    ]
    confidence = confidence_calculator.calculate_from_signals(
        tool_signals=edit_result.tool_signals,
        final_answer=answer,
        tool_results=edit_result.tool_result_texts + prior_answer_texts,
        data_source=request.data_source,
    )

    from data_concierge.gateway.notebook_verification import schedule_verification

    verification_pending = False
    try:
        verification_pending = schedule_verification(
            query_id,
            notebook_json,
            answer,
            confidence,
            query=f"{original_query or request.query} — revision: {instruction}"[:500],
            data_source=request.data_source,
        )
    except Exception as e:  # noqa: BLE001 - scheduling must never fail the revision
        logger.warning("Could not schedule verification for revision", error=str(e))

    total_input = 0
    total_output = 0
    for entry in edit_result.agent_log:
        tokens = entry.get("tokens")
        if tokens:
            total_input += (
                tokens.get("input", 0)
                + tokens.get("cache_creation_input", 0)
                + tokens.get("cache_read_input", 0)
            )
            total_output += tokens.get("output", 0)

    query_logs.append_log(
        user=user_name,
        auth_type=_user_info.get("auth_type") or "password",
        query=request.query,
        source="revision",
        query_id=query_id,
        confidence=confidence.final_score,
        confidence_level=confidence.level.value,
        tier="tier_2",
        processing_time_ms=processing_time_ms,
        had_notebook=True,
        notebook_url=f"/api/v1/notebooks/{query_id}",
        data_source=request.data_source,
        input_tokens=total_input or None,
        output_tokens=total_output or None,
    )

    logger.info(
        "Notebook revision complete",
        query_id=query_id,
        revised_from=prev_id,
        edits=edit_result.edits_applied,
        confidence=confidence.final_score,
    )

    return QueryResponse(
        query_id=query_id,
        answer=answer,
        confidence=confidence.final_score,
        confidence_level=confidence.level.value,
        confidence_breakdown={
            "answer_grounding": confidence.answer_grounding,
            "data_retrieval_quality": confidence.data_retrieval_quality,
            "source_metadata_quality": confidence.source_metadata_quality,
            "query_answer_alignment": confidence.query_answer_alignment,
            "computation_complexity": confidence.computation_complexity,
            "query_interpretation": confidence.query_interpretation,
            "source_authority": confidence.source_authority,
            "retrieval_match": confidence.retrieval_match,
            "data_recency": confidence.data_recency,
            "computation_reliability": confidence.computation_reliability,
        },
        confidence_unavailable=dict(confidence.unavailable),
        confidence_measured_weight=confidence.measured_weight,
        confidence_explanation=confidence.explanation,
        verification_pending=verification_pending,
        tier="tier_2",
        notebook_url=f"/api/v1/notebooks/{query_id}",
        # Same contract as the fresh-analysis path: the inline notebook JSON
        # honors the request flag (the stored copy is always available at
        # notebook_url either way).
        notebook=notebook_json if request.include_notebook else None,
        is_revision=True,
        revised_from_query_id=prev_id,
        processing_time_ms=processing_time_ms,
    )


@router.get("/sources")
async def list_data_sources() -> dict[str, Any]:
    """List all available data sources and their details.

    This endpoint provides comprehensive information about all data sources
    that the Data Concierge can recommend, including:
    - Source descriptions and URLs
    - Available datasets within each source
    - Data categories covered
    - Quality scores and update frequencies
    """
    from data_concierge.core.data_source_registry import get_data_source_registry

    registry = get_data_source_registry()
    sources = registry.get_enabled_sources()

    source_list = []
    for source in sources:
        source_dict = {
            "id": source.id,
            "name": source.name,
            "description": source.description,
            "base_url": source.base_url,
            "portal_url": source.portal_url,
            "api_url": source.api_url,
            "status": "active" if source.is_enabled else "inactive",
            "quality_score": source.quality_score,
            "update_frequency": source.update_frequency,
            "requires_api_key": source.requires_api_key,
            "categories": [c.value for c in source.categories],
            "keywords": source.keywords[:10],  # Limit keywords in response
            "datasets": [
                {
                    "name": ds.name,
                    "description": ds.description,
                    "url": ds.url,
                    "update_frequency": ds.update_frequency,
                }
                for ds in source.datasets[:5]  # Limit datasets in response
            ],
            "example_queries": source.example_queries,
            "access_instructions": source.access_instructions,
        }
        source_list.append(source_dict)

    return {
        "count": len(source_list),
        "sources": source_list,
    }


@router.get("/sources/{source_id}")
async def get_data_source(source_id: str) -> dict[str, Any]:
    """Get detailed information about a specific data source.

    This provides comprehensive details about a single source including
    all available datasets, access instructions, and example queries.
    """
    from data_concierge.core.data_source_registry import get_data_source_registry

    registry = get_data_source_registry()
    source = registry.get_source(source_id)

    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data source '{source_id}' not found",
        )

    return {
        "id": source.id,
        "name": source.name,
        "description": source.description,
        "base_url": source.base_url,
        "portal_url": source.portal_url,
        "api_url": source.api_url,
        "status": "active" if source.is_enabled else "inactive",
        "quality_score": source.quality_score,
        "update_frequency": source.update_frequency,
        "requires_api_key": source.requires_api_key,
        "source_type": source.source_type,
        "categories": [c.value for c in source.categories],
        "keywords": source.keywords,
        "datasets": [
            {
                "name": ds.name,
                "description": ds.description,
                "url": ds.url,
                "categories": [c.value for c in ds.categories],
                "keywords": ds.keywords,
                "update_frequency": ds.update_frequency,
                "geographic_coverage": ds.geographic_coverage,
                "temporal_coverage": ds.temporal_coverage,
            }
            for ds in source.datasets
        ],
        "example_queries": source.example_queries,
        "access_instructions": source.access_instructions,
    }


@router.get("/sources/search/{keywords}")
async def search_data_sources(keywords: str) -> dict[str, Any]:
    """Search for data sources matching given keywords.

    Keywords should be comma-separated (e.g., "unemployment,jobs,labor").
    Returns sources ranked by relevance.
    """
    from data_concierge.core.data_source_registry import get_data_source_registry

    registry = get_data_source_registry()

    # Parse keywords
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]

    if not keyword_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one keyword is required",
        )

    # Search
    results = registry.search_by_keywords(keyword_list)

    return {
        "keywords": keyword_list,
        "count": len(results),
        "sources": [
            {
                "id": source.id,
                "name": source.name,
                "description": source.description,
                "relevance_score": score,
                "portal_url": source.portal_url,
                "categories": [c.value for c in source.categories],
            }
            for source, score in results
        ],
    }


@router.get("/notebooks")
async def list_notebooks() -> dict[str, Any]:
    """List all generated notebooks.

    Returns a list of available notebooks with metadata.
    """
    keys = storage.list_keys(_NOTEBOOKS_PREFIX, suffix=".ipynb")
    notebooks = []
    for key in keys:
        # key format: notebooks/<query_id>.ipynb
        filename = key.rsplit("/", 1)[-1]
        query_id = filename.replace(".ipynb", "")
        notebooks.append(
            {
                "query_id": query_id,
                "filename": filename,
                "download_url": f"/api/v1/notebooks/{query_id}",
            }
        )

    return {
        "count": len(notebooks),
        "notebooks": notebooks,
    }


# =============================================================================
# Notebook Submission & Admin Review Endpoints
# =============================================================================


@router.post("/notebooks/submit")
async def submit_notebook_endpoint(
    request: NotebookSubmitRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Submit a notebook for admin review.

    After a query generates a notebook, users can submit it for
    admin evaluation. If approved, the notebook becomes a verified
    resource that can be served to users with similar questions.
    """
    user = get_current_user(http_request) or {}
    submitted_by = (
        user.get("display_name") or user.get("user") or request.submitted_by or "anonymous"
    )
    submitter_email = user.get("email")
    submitter_auth_type = user.get("auth_type")

    # Aggregate tokens from agent log if available (cache tokens count
    # toward the prompt side per the evidence standard)
    submit_input_tokens = 0
    submit_output_tokens = 0
    if request.agent_log:
        for entry in request.agent_log:
            tok = entry.get("tokens")
            if tok:
                submit_input_tokens += (
                    tok.get("input", 0)
                    + tok.get("cache_creation_input", 0)
                    + tok.get("cache_read_input", 0)
                )
                submit_output_tokens += tok.get("output", 0)

    submission = submit_notebook(
        query=request.query,
        answer=request.answer,
        notebook_json=request.notebook_json,
        filename=request.filename,
        submitted_by=submitted_by,
        submitter_email=submitter_email,
        submitter_auth_type=submitter_auth_type,
        data_source=request.data_source,
        confidence=request.confidence,
        tags=request.tags,
        input_tokens=submit_input_tokens or None,
        output_tokens=submit_output_tokens or None,
        query_id=request.query_id,
    )

    # Save agent execution log alongside the submission
    agent_log = request.agent_log
    if not agent_log and request.query_id:
        stored = storage.read_json(f"{_NOTEBOOKS_PREFIX}/{request.query_id}_log.json")
        if stored:
            agent_log = stored.get("log", [])
    if agent_log:
        log_key = f"verified_notebooks/submissions/{submission.submission_id}_log.json"
        storage.write_json(log_key, {"query": request.query, "log": agent_log})

    # Notify admins (fire-and-forget)
    try:
        from data_concierge.gateway.admin_notifications import notify_new_submission

        await notify_new_submission(
            kind="notebook",
            query=request.query,
            submitted_by=submitted_by,
            submitter_email=submitter_email,
            submission_id=submission.submission_id,
        )
    except Exception as e:
        logger.warning("Admin notification failed (non-blocking)", error=str(e))

    # Publish draft to GitHub (fire-and-forget, don't block the response)
    github_result = None
    try:
        from data_concierge.gateway.github_publisher import publish_draft

        github_result = await publish_draft(
            submission.submission_id, request.query, request.notebook_json
        )
    except Exception as e:
        logger.warning("GitHub draft publish failed (non-blocking)", error=str(e))

    return {
        "message": "Notebook submitted for admin review",
        "submission_id": submission.submission_id,
        "status": submission.status.value,
        "github_draft": github_result,
    }


@router.get("/notebooks/submissions")
async def list_submissions(
    status_filter: str | None = None,
) -> dict[str, Any]:
    """List notebook submissions.

    Args:
        status_filter: Filter by status ('pending', 'approved', 'rejected').
                       If None, returns all submissions.
    """
    if status_filter == "pending":
        submissions = get_pending_submissions()
    else:
        submissions = get_all_submissions()
        if status_filter:
            submissions = [s for s in submissions if s.status.value == status_filter]

    return {
        "count": len(submissions),
        "submissions": [s.model_dump() for s in submissions],
    }


@router.get("/notebooks/submissions/{submission_id}")
async def get_submission_endpoint(submission_id: str) -> dict[str, Any]:
    """Get a specific notebook submission."""
    submission = get_submission(submission_id)
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission {submission_id} not found",
        )
    payload = submission.model_dump()
    # If this submission has already been approved into the verified library,
    # surface a clickable link to the published GitHub artifact (#111).
    verified = get_verified_notebook_by_submission(submission_id)
    payload["github_url"] = build_blob_url(verified.github_path) if verified else None
    return payload


@router.get("/notebooks/submissions/{submission_id}/logs")
async def get_submission_logs(submission_id: str) -> dict[str, Any]:
    """Get agent execution logs for a submission."""
    log_key = f"verified_notebooks/submissions/{submission_id}_log.json"
    log_data = storage.read_json(log_key)
    if not log_data:
        return {"query": "", "log": []}
    return log_data


@router.post("/notebooks/submissions/{submission_id}/approve")
async def approve_notebook_endpoint(
    submission_id: str,
    request: NotebookReviewRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Approve a notebook submission, making it verified.

    Once approved, the notebook is available for search and reuse
    when similar questions are asked.

    Ordering (write-through SSOT): we publish to GitHub *first* and only
    mark the notebook verified locally if either GitHub succeeded or
    GitHub publishing is disabled. If GitHub is enabled and the publish
    fails, the submission stays PENDING and the caller gets HTTP 502 so
    they can retry once the underlying issue (outage, expired token,
    rate limit) is resolved. This avoids the previous failure mode where
    a notebook could be locally "verified" with no record in GitHub.

    Provenance (issue #46 step 4): we enrich the notebook's
    ``metadata.data_concierge`` namespace with the verification fields
    (``submission_id``, ``confidence``, ``verified_by``, ``verified_at``)
    BEFORE handing it to ``publish_verified``, and we pass the same
    enriched copy to ``approve_notebook`` so the local blob matches what
    GitHub got. The single ``verified_at`` timestamp is reused as the
    local index's ``github_synced_at``, so the embedded metadata and the
    index entry stay coherent — that coherence is what makes
    "rebuild the index from GitHub" recoverable in steps 5+.
    """
    from datetime import UTC
    from datetime import datetime as _dt

    from data_concierge.gateway.github_publisher import (
        GitHubPublishError,
        publish_verified,
    )
    from data_concierge.gateway.verified_notebooks import (
        add_verification_metadata,
    )

    # 1. Peek the submission (read-only) so we can validate without mutating.
    submission = get_submission(submission_id)
    if submission is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission {submission_id} not found",
        )
    if submission.status != ReviewStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission {submission_id} not found or already reviewed",
        )

    # Capture WHO approved this from the authenticated admin session rather
    # than the client-supplied ``reviewed_by`` (which the UI hardcodes to
    # "admin"). This is the verifier identity recorded in provenance (#73).
    reviewer = _reviewer_identity(_admin)

    # Forward dedup: keep exactly one verified entry per question. If this
    # question is already in the verified library, collapse the submission into
    # that entry instead of publishing/creating a second copy. Skipping the
    # GitHub publish here is what makes this safe — no orphaned repo file.
    existing_nb = find_verified_notebook_by_question(submission.query)
    if existing_nb is not None:
        collapse_notebook_submission_as_duplicate(
            submission_id=submission_id,
            existing_notebook_id=existing_nb.notebook_id,
            reviewed_by=reviewer,
            admin_notes=request.admin_notes,
        )
        logger.info(
            "Approved submission collapsed into existing verified notebook",
            submission_id=submission_id,
            notebook_id=existing_nb.notebook_id,
        )
        return {
            "message": (
                "Notebook approved; collapsed into existing verified notebook "
                "for the same question (no duplicate created)."
            ),
            "notebook_id": existing_nb.notebook_id,
            "submission_id": submission_id,
            "deduplicated": True,
            "github_verified": None,
        }

    # 2. Pre-compute the verification timestamp ONCE so the embedded
    # metadata's ``verified_at`` and the local index's ``github_synced_at``
    # are the same instant. Enrich the notebook's metadata.data_concierge
    # namespace with the provenance fields before anything goes to GitHub.
    verified_at = _dt.now(UTC).isoformat().replace("+00:00", "Z")
    enriched_notebook = add_verification_metadata(
        submission.notebook_json,
        submission_id=submission_id,
        confidence=submission.confidence,
        verified_by=reviewer,
        verified_at=verified_at,
        data_source=submission.data_source,
    )

    # 2b. Typed Standards evidence embedding (spec §8.8.2), gated + best-effort.
    # When enabled, sign + (optionally) timestamp/log the package, embed the
    # commitment view into the published notebook, and stage the canonical
    # package JSON as a sibling so the embedded packageUrl resolves. Any failure
    # here must NOT block approval, so it is fully wrapped.
    evidence_sibling_files: dict[str, bytes] | None = None
    if settings.evidence_embed_enabled and settings.evidence_signing_enabled:
        try:
            import json as _json

            from data_concierge.gateway.evidence import (
                build_evidence_package,
                commitment_endpoint_url,
                embed_commitment_view,
                evidence_commitment_storage_key,
                evidence_package_storage_key,
                package_endpoint_url,
            )
            from data_concierge.gateway.evidence_attestation import obtain_attestations
            from data_concierge.gateway.evidence_signing import get_active_signer
            from data_concierge.gateway.github_publisher import (
                _notebook_filename,
                load_github_settings,
            )

            _gh = load_github_settings()
            _filename = _notebook_filename(submission_id, submission.query)
            _verified_path = f"{_gh.get('verified_folder')}/{_filename}"
            _pkg_path = _verified_path.removesuffix(".ipynb") + ".package.json"

            _log = storage.read_json(f"verified_notebooks/submissions/{submission_id}_log.json")
            _agent_log = (_log or {}).get("log", [])
            _pkg = build_evidence_package(
                notebook_json=enriched_notebook,
                answer=submission.answer,
                query=submission.query,
                agent_log=_agent_log,
                title=submission.query,
                signer=get_active_signer(),
            )
            _hash = _pkg.envelope_hash
            # packageUrl points at our JSON+CORS package endpoint — the verifier
            # rejects GitHub-raw's text/plain content-type — not the git sibling.
            _pkg.commitment_view["packageUrl"] = package_endpoint_url(_hash)
            _attest = await obtain_attestations(_hash, _pkg.signature)
            _pkg.commitment_view.update(_attest)

            # Serve the commitment + canonical package from our host for the
            # verifier (keyed by hash); embed the commitment into the published
            # notebook and commit the canonical package to git for provenance.
            storage.write_json(evidence_package_storage_key(_hash), _pkg.package)
            storage.write_json(evidence_commitment_storage_key(_hash), _pkg.commitment_view)
            enriched_notebook = embed_commitment_view(enriched_notebook, _pkg)
            evidence_sibling_files = {
                _pkg_path: _json.dumps(_pkg.package, indent=2).encode("utf-8")
            }
            logger.info(
                "Embedded Typed Standards commitment view",
                submission_id=submission_id,
                signed=_pkg.signed,
                package_hash=_hash[:12],
                commitment_url=commitment_endpoint_url(_hash),
            )
        except Exception as _ev_err:
            logger.warning(
                "Evidence embedding failed (non-blocking)",
                submission_id=submission_id,
                error=str(_ev_err),
            )

    # 3. Publish the enriched copy to GitHub. publish_verified() returns
    # None when GitHub is disabled and raises GitHubPublishError on real
    # failures (HTTP, network, auth).
    # Pass sibling_files only when present so that with evidence embedding off
    # (the default) the call is identical to before — no behavior change.
    _publish_kwargs: dict[str, Any] = {}
    if evidence_sibling_files:
        _publish_kwargs["sibling_files"] = evidence_sibling_files
    try:
        github_result = await publish_verified(
            submission_id,
            "pending",  # placeholder; the local notebook_id is generated below
            submission.query,
            enriched_notebook,
            reason=request.admin_notes,
            reviewer=reviewer,
            **_publish_kwargs,
        )
    except GitHubPublishError as e:
        logger.error(
            "GitHub publish failed; aborting approval to preserve write-through SSOT",
            submission_id=submission_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"GitHub publish failed: {e}. The submission is unchanged; "
                "retry once the GitHub-side issue is resolved."
            ),
        ) from e

    # 4. Commit local state. Reuse ``verified_at`` as ``github_synced_at``
    # so the embedded metadata and the index entry agree on the moment of
    # approval. When GitHub is disabled, github_path/github_synced_at stay
    # None (no GitHub commit happened) but the local copy still carries
    # the verified_at provenance.
    github_path = github_result.get("path") if github_result else None
    github_synced_at = verified_at if github_result else None

    verified = approve_notebook(
        submission_id=submission_id,
        reviewed_by=reviewer,
        admin_notes=request.admin_notes,
        github_path=github_path,
        github_synced_at=github_synced_at,
        notebook_json=enriched_notebook,
    )
    if not verified:
        # Race: submission was approved/rejected between the peek and the
        # commit. If we already pushed to GitHub, the verified-folder file
        # is now orphaned (no local index entry points at it) — log loudly
        # so an admin can reconcile via the existing GitHub tooling.
        if github_result:
            logger.error(
                "Submission state changed during approval; GitHub file may be orphaned",
                submission_id=submission_id,
                github_path=github_result.get("path"),
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Submission {submission_id} was modified by another request "
                "during approval; reload and retry."
            ),
        )

    return {
        "message": "Notebook approved and verified",
        "notebook_id": verified.notebook_id,
        "submission_id": submission_id,
        "github_verified": github_result,
    }


@router.post("/notebooks/submissions/{submission_id}/reject")
async def reject_notebook_endpoint(
    submission_id: str,
    request: NotebookReviewRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Reject a notebook submission."""
    # Record the authenticated admin as the rejecter, not the client-supplied
    # ``reviewed_by`` (which the UI hardcodes to "admin") — #73.
    reviewer = _reviewer_identity(_admin)
    submission = reject_notebook(
        submission_id=submission_id,
        reviewed_by=reviewer,
        admin_notes=request.admin_notes,
    )
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission {submission_id} not found or already reviewed",
        )

    # Remove draft from GitHub
    try:
        from data_concierge.gateway.github_publisher import remove_draft

        await remove_draft(
            submission_id,
            submission.query,
            reason=request.admin_notes,
            reviewer=reviewer,
        )
    except Exception as e:
        logger.warning("GitHub draft removal failed (non-blocking)", error=str(e))

    return {
        "message": "Notebook rejected",
        "submission_id": submission_id,
    }


@router.get("/verified-notebooks")
async def list_verified_notebooks() -> dict[str, Any]:
    """List all verified notebooks."""
    from data_concierge.gateway.evidence import commitment_endpoint_url

    notebooks = get_verified_notebooks()
    # Load GitHub settings once so every row's blob URL is built without
    # re-reading the settings store in the loop (#111).
    gh_settings = load_github_settings()

    def _verify_url(nb: Any) -> str | None:
        if not nb.evidence_package_hash:
            return None
        return "https://typedstandards.org/verify?url=" + commitment_endpoint_url(
            nb.evidence_package_hash
        )

    return {
        "count": len(notebooks),
        "notebooks": [
            {
                "notebook_id": nb.notebook_id,
                # Expose the originating submission so the admin UI can fetch
                # this notebook's agent logs (#71) and surface verifier
                # provenance (#73) without a second round-trip.
                "submission_id": nb.submission_id,
                "query": nb.query,
                "answer": nb.answer[:200] + "..." if len(nb.answer) > 200 else nb.answer,
                "tags": nb.tags,
                "data_source": nb.data_source,
                "verified_at": nb.verified_at,
                "verified_by": nb.verified_by,
                "admin_notes": nb.admin_notes,
                "submitted_by": nb.submitted_by or "",
                "confidence": nb.confidence,
                "input_tokens": nb.input_tokens,
                "output_tokens": nb.output_tokens,
                "usage_count": nb.usage_count,
                "download_url": f"/api/v1/verified-notebooks/{nb.notebook_id}/download",
                "github_path": nb.github_path,
                "github_synced_at": nb.github_synced_at,
                "github_url": build_blob_url(nb.github_path, gh_settings),
                "evidence_package_hash": nb.evidence_package_hash,
                "evidence_verify_url": _verify_url(nb),
            }
            for nb in notebooks
        ],
    }


async def _refresh_verified_from_github(notebook_id: str) -> Any:
    """Pull the latest notebook content from GitHub (if configured) and
    update the local index/blob. Returns the refreshed notebook, or the
    cached one when no fresher copy is available.
    """
    from datetime import datetime as _dt

    from data_concierge.gateway.github_publisher import fetch_notebook
    from data_concierge.gateway.verified_notebooks import (
        update_verified_notebook as _update_verified,
    )

    notebook = get_verified_notebook(notebook_id)
    if not notebook or not notebook.github_path:
        return notebook

    fresh = await fetch_notebook(notebook.github_path)
    if fresh is None:
        return notebook

    _update_verified(
        notebook_id,
        notebook_json=fresh,
        github_synced_at=_dt.utcnow().isoformat() + "Z",
    )
    return get_verified_notebook(notebook_id)


@router.get("/verified-notebooks/{notebook_id}")
async def get_verified_notebook_endpoint(notebook_id: str) -> dict[str, Any]:
    """Get a specific verified notebook.

    When the notebook has a GitHub path recorded, the latest content is
    pulled from the repo so admin edits made directly on GitHub are
    reflected here without re-approval.
    """
    notebook = await _refresh_verified_from_github(notebook_id)
    if not notebook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verified notebook {notebook_id} not found",
        )
    payload = notebook.model_dump()
    payload["github_url"] = build_blob_url(notebook.github_path)
    # Surface the Typed Standards verify link once a package has been minted.
    if notebook.evidence_package_hash:
        from data_concierge.gateway.evidence import commitment_endpoint_url

        cu = commitment_endpoint_url(notebook.evidence_package_hash)
        payload["evidence_commitment_url"] = cu
        payload["evidence_verify_url"] = f"https://typedstandards.org/verify?url={cu}"
    return payload


def _evidence_summary(
    notebook_id: str, pkg: Any, attestations: dict, *, reused: bool
) -> dict[str, Any]:
    from data_concierge.gateway.evidence import commitment_endpoint_url

    commitment_url = commitment_endpoint_url(pkg.envelope_hash) if pkg.signed else None
    return {
        "notebook_id": notebook_id,
        "signed": pkg.signed,
        "reused": reused,
        "packageHash": pkg.envelope_hash,
        "attestations": list(attestations.keys()),
        "commitmentUrl": commitment_url,
        "verifyUrl": (
            f"https://typedstandards.org/verify?url={commitment_url}" if commitment_url else None
        ),
    }


async def _build_and_store_evidence(
    notebook: Any, *, force: bool = False
) -> tuple[dict[str, Any], Any]:
    """Build a verified notebook's evidence package and, when signed, store the
    commitment + canonical package for the public serving endpoints.

    The package id + ``createdAt`` are derived deterministically from the
    notebook, so the envelope hash is a stable content address: rebuilding the
    same notebook yields the same hash. Already-minted notebooks are reused
    (skipping a rebuild — and, crucially, a duplicate FreeTSA/Rekor round-trip)
    unless ``force``. When signing is off, nothing is stored. Returns
    ``(summary, pkg)``.
    """
    import uuid as _uuid

    from data_concierge.gateway.evidence import (
        EvidencePackage,
        SignatureEnvelope,
        build_evidence_package,
        evidence_commitment_storage_key,
        evidence_package_storage_key,
        package_endpoint_url,
    )
    from data_concierge.gateway.evidence_attestation import obtain_attestations
    from data_concierge.gateway.evidence_signing import get_active_signer
    from data_concierge.gateway.verified_notebooks import update_verified_notebook

    # Reuse an already-minted, still-stored package (idempotent; avoids a
    # duplicate public Rekor entry on re-runs).
    existing = getattr(notebook, "evidence_package_hash", None)
    if existing and not force:
        commitment = storage.read_json(evidence_commitment_storage_key(existing))
        package = storage.read_json(evidence_package_storage_key(existing))
        if commitment and package:
            sig = commitment.get("signature", {})
            pkg = EvidencePackage(
                package=package,
                signature=SignatureEnvelope(
                    signature=sig.get("signature", ""),
                    publicKey=sig.get("publicKey", ""),
                    algorithm=sig.get("algorithm", "Ed25519ph"),
                    kid=sig.get("kid", ""),
                ),
                envelope_hash=existing,
                content_hash=package.get("contentHash", {}).get("sha256", ""),
                commitment_view=commitment,
            )
            return _evidence_summary(notebook.notebook_id, pkg, {}, reused=True), pkg

    agent_log: list[dict[str, Any]] = []
    if notebook.submission_id:
        stored = storage.read_json(
            f"verified_notebooks/submissions/{notebook.submission_id}_log.json"
        )
        if stored:
            agent_log = stored.get("log", [])

    pkg = build_evidence_package(
        notebook_json=notebook.notebook_json,
        answer=notebook.answer,
        query=notebook.query,
        agent_log=agent_log,
        title=notebook.query,
        signer=get_active_signer(),
        # Deterministic so the hash is a stable content address per notebook.
        package_id=str(
            _uuid.uuid5(_uuid.NAMESPACE_URL, f"datHere-evidence:{notebook.notebook_id}")
        ),
        created_at=notebook.verified_at,
    )
    pkg_hash = pkg.envelope_hash
    pkg.commitment_view["packageUrl"] = package_endpoint_url(pkg_hash)
    attestations = await obtain_attestations(pkg_hash, pkg.signature)
    pkg.commitment_view.update(attestations)

    if pkg.signed:
        storage.write_json(evidence_package_storage_key(pkg_hash), pkg.package)
        storage.write_json(evidence_commitment_storage_key(pkg_hash), pkg.commitment_view)
        update_verified_notebook(notebook.notebook_id, evidence_package_hash=pkg_hash)

    summary = _evidence_summary(notebook.notebook_id, pkg, attestations, reused=False)
    summary["hasAgentLog"] = bool(agent_log)
    return summary, pkg


@router.get("/verified-notebooks/{notebook_id}/evidence-package")
async def get_verified_notebook_evidence_package(
    notebook_id: str,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Build (and, when signed, publish) the ``datHere`` evidence package.

    Maps the verified notebook + its captured ``agent_log`` into the A–G
    evidence package and commitment view (Typed Standards spec §8.7/§8.8).

    When signing is configured, this also **stores** the commitment + canonical
    package keyed by the package hash, making them servable at
    ``/api/evidence/{hash}/commitment`` and ``/api/evidence/{hash}/package`` —
    so an admin can mint a verifiable sample from any existing verified notebook
    without a fresh approval. When signing is off, nothing is stored and the
    commitment view is marked ``dev-unsigned``.
    """
    from data_concierge.gateway.evidence import embed_commitment_view

    notebook = await _refresh_verified_from_github(notebook_id)
    if not notebook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verified notebook {notebook_id} not found",
        )

    summary, pkg = await _build_and_store_evidence(notebook)
    return {
        **summary,
        "contentHash": pkg.content_hash,
        "package": pkg.package,
        "commitmentView": pkg.commitment_view,
        "notebookWithEvidence": embed_commitment_view(notebook.notebook_json, pkg),
    }


@router.post("/verified-notebooks/backfill-evidence")
async def backfill_evidence_packages(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Build + store an evidence package for **every** verified notebook.

    Deploying does not retro-generate packages — embedding/storing only runs on
    a new approval or the per-notebook preview above. This backfills the whole
    already-approved library at once so each notebook becomes verifiable at its
    ``/api/evidence/{hash}/commitment`` URL. Idempotent (content-addressed);
    only signed packages are stored. Does NOT rewrite the GitHub ``.ipynb``
    artifacts — it just makes the commitment/package servable.
    """
    results: list[dict[str, Any]] = []
    for notebook in get_verified_notebooks():
        try:
            summary, _ = await _build_and_store_evidence(notebook)
            results.append(summary)
        except Exception as exc:  # one bad notebook shouldn't abort the batch
            logger.warning(
                "Backfill failed for notebook",
                notebook_id=notebook.notebook_id,
                error=str(exc),
            )
            results.append(
                {"notebook_id": notebook.notebook_id, "error": str(exc), "signed": False}
            )
    stored = sum(1 for r in results if r.get("commitmentUrl"))
    return {"count": len(results), "stored": stored, "results": results}


@router.post("/verified-notebooks/dedupe")
async def dedupe_verified_library_endpoint(
    dry_run: bool = False,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Collapse duplicate verified notebooks/answers that share a question.

    The verified library should hold one entry per question; duplicates
    accumulate when several submissions for the same question are approved.
    This keeps the most-used (then oldest) survivor, merges the duplicates'
    usage counts into it, removes the redundant entries, and deletes orphaned
    notebook blobs. GitHub-published entries are never deleted (that would
    orphan repo files) — such groups are reported under
    ``needs_manual_github_reconcile``. Pass ``?dry_run=true`` to preview.
    Idempotent and safe to re-run (e.g. after a GitHub sync).
    """
    summary = dedupe_verified_library(dry_run=dry_run)
    summary["message"] = (
        f"{'Would remove' if dry_run else 'Removed'} {summary['removed_count']} "
        "duplicate verified entr"
        f"{'y' if summary['removed_count'] == 1 else 'ies'}."
    )
    return summary


@router.get("/verified-notebooks/{notebook_id}/download")
async def download_verified_notebook(notebook_id: str):
    """Download a verified notebook file.

    If the notebook is tracked on GitHub, the latest version is pulled
    from the repo before serving — making the GitHub repo the source of
    truth for verified notebook content.
    """
    await _refresh_verified_from_github(notebook_id)

    notebook_key = f"verified_notebooks/verified/{notebook_id}.ipynb"

    if isinstance(storage, GCSStorage):
        # For GCS: download to temp file then serve
        tmp_path = storage.download_to_tmp(notebook_key)
        if not tmp_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Verified notebook file {notebook_id} not found",
            )
        increment_usage(notebook_id)
        return FileResponse(
            path=tmp_path,
            media_type="application/x-ipynb+json",
            filename=f"verified_{notebook_id}.ipynb",
        )

    # Local storage: serve directly
    local_path = storage.full_path(notebook_key)
    if not Path(local_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verified notebook file {notebook_id} not found",
        )
    increment_usage(notebook_id)
    return FileResponse(
        path=local_path,
        media_type="application/x-ipynb+json",
        filename=f"verified_{notebook_id}.ipynb",
    )


@router.post("/verified-notebooks/search")
async def search_verified_notebooks_endpoint(
    request: VerifiedNotebookSearchRequest,
) -> dict[str, Any]:
    """Search verified notebooks for similar queries.

    This is used to find pre-verified notebooks that match a user's
    question. Results are ranked by keyword similarity.
    """
    results = search_verified_notebooks(
        query=request.query,
        threshold=request.threshold,
        max_results=request.max_results,
    )
    return {
        "query": request.query,
        "count": len(results),
        "results": [r.model_dump() for r in results],
    }


@router.get("/notebooks/admin/stats")
async def notebook_admin_stats(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get statistics about notebook submissions, verified notebooks, and quick answers."""
    return get_stats()


@router.post("/verified-notebooks/sync-from-github")
async def sync_verified_from_github(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Re-sync all verified notebooks from the GitHub repo.

    Treats the GitHub repo as the source of truth: for every verified
    notebook with a recorded ``github_path`` (or one that matches a
    file in the verified folder by filename), pull the latest content
    and overwrite the locally cached copy. Notebooks that no longer
    exist on GitHub are reported but left in place.
    """
    from data_concierge.gateway.verified_notebooks import sync_all_from_github

    result = await sync_all_from_github()
    if result.get("skipped"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub publishing is not enabled. Configure it under Settings → GitHub.",
        )
    return {
        "message": f"Synced {len(result['updated'])} verified notebook(s) from GitHub",
        **result,
    }


@router.post("/verified-notebooks/bootstrap-from-github")
async def bootstrap_verified_from_github(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Rebuild the verified-notebook index from the GitHub repo (disaster recovery).

    Unlike ``/verified-notebooks/sync-from-github`` (which iterates the
    local index and refreshes each entry — a no-op when the local index
    is empty), this endpoint iterates the GitHub ``verified/`` folder
    and reconstructs the local index from each notebook's embedded
    ``metadata.data_concierge`` provenance.

    Use it on a fresh deployment / cold-start / corrupted-index recovery:
    GitHub is the source of truth, local is rebuilt from there.

    Admin-only. Returns the bootstrap report (checked / created / updated
    / failed / skipped_no_metadata / skipped_bad_metadata /
    skipped_duplicate_submission_id / orphaned_locally).

    Returns 400 when GitHub publishing is disabled. Returns 502 when
    GitHub cannot be reached for the listing (auth, network, 5xx) —
    distinct from "I checked and there's nothing to rebuild".
    """
    from data_concierge.gateway.github_publisher import GitHubPublishError
    from data_concierge.gateway.verified_notebooks import (
        bootstrap_index_from_github,
    )

    try:
        result = await bootstrap_index_from_github()
    except GitHubPublishError as e:
        logger.error(
            "Bootstrap failed before inspecting GitHub",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(f"Could not reach GitHub to bootstrap: {e}. Verify the token and try again."),
        ) from e
    if result.get("skipped"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub publishing is not enabled. Configure it under Settings → GitHub.",
        )
    return {
        "message": (
            f"Bootstrapped {result['checked']} verified notebook(s) from GitHub; "
            f"created {len(result['created'])}, updated {len(result['updated'])}, "
            f"skipped {len(result['skipped_no_metadata'])} (no metadata), "
            f"skipped {len(result.get('skipped_bad_metadata', []))} (bad metadata), "
            f"skipped {len(result.get('skipped_duplicate_submission_id', []))} (duplicate submission_id), "
            f"{len(result['failed'])} fetch failure(s), "
            f"{len(result['orphaned_locally'])} local-only entries."
        ),
        **result,
    }


class _SyncOnePayload(BaseModel):
    """Request body for POST /verified-notebooks/sync-one-from-github."""

    path: str


@router.post("/verified-notebooks/sync-one-from-github")
async def sync_one_verified_from_github(
    payload: _SyncOnePayload,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Sync a single verified notebook from GitHub by path.

    Per-path counterpart to ``/verified-notebooks/bootstrap-from-github``.
    Fetches the notebook at ``path``, validates its embedded
    ``metadata.data_concierge`` provenance, and creates or refreshes the
    matching local index entry.

    Step 7 (webhook handler) will call this for each file in a push
    event's added/modified list. Admins can also call it directly to
    force-resync a single file when they've manually edited it on
    GitHub.

    Admin-only. Returns the per-file result
    (status / notebook_id / path / reason).

    Returns 400 when GitHub publishing is disabled. Returns 502 when
    the GitHub fetch fails (this endpoint does NOT delete local entries
    on 404 — that's a separate operation tied to webhook DELETE events).
    """
    from data_concierge.gateway.verified_notebooks import sync_one_from_github

    result = await sync_one_from_github(payload.path)
    if result["status"] == "skipped_disabled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub publishing is not enabled. Configure it under Settings → GitHub.",
        )
    if result["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not fetch {payload.path} from GitHub. "
                "Verify the file exists at that path and the token has access."
            ),
        )
    return result


@router.post("/verified-answers/bootstrap-from-github")
async def bootstrap_verified_answers_from_github(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Rebuild the verified-answer index from the GitHub repo (issue #46 step 9).

    Counterpart to ``/verified-notebooks/bootstrap-from-github`` for
    quick answers. Iterates the GitHub ``verified_answers_folder`` and
    reconstructs the local index from each ``<answer_id>.json`` file.

    Use it on a fresh deployment / cold-start / corrupted-index
    recovery: GitHub is the source of truth, local is rebuilt from
    there.

    Admin-only. Returns the bootstrap report (checked / created /
    updated / failed / skipped_bad_metadata /
    skipped_duplicate_answer_id / orphaned_locally).

    Returns 400 when GitHub publishing is disabled. Returns 502 when
    GitHub cannot be reached for the listing (auth, network, 5xx) —
    distinct from "I checked and there's nothing to rebuild".
    """
    from data_concierge.gateway.github_publisher import GitHubPublishError
    from data_concierge.gateway.verified_notebooks import (
        bootstrap_answers_from_github,
    )

    try:
        result = await bootstrap_answers_from_github()
    except GitHubPublishError as e:
        logger.error(
            "Bootstrap-answers failed before inspecting GitHub",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not reach GitHub to bootstrap answers: {e}. Verify the token and try again."
            ),
        ) from e
    if result.get("skipped"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub publishing is not enabled. Configure it under Settings → GitHub.",
        )
    return {
        "message": (
            f"Bootstrapped {result['checked']} verified answer(s) from GitHub; "
            f"created {len(result['created'])}, updated {len(result['updated'])}, "
            f"skipped {len(result['skipped_bad_metadata'])} (bad metadata), "
            f"skipped {len(result['skipped_duplicate_answer_id'])} (duplicate answer_id), "
            f"{len(result['failed'])} fetch failure(s), "
            f"{len(result['orphaned_locally'])} local-only entries."
        ),
        **result,
    }


@router.post("/verified-answers/sync-one-from-github")
async def sync_one_verified_answer_from_github(
    payload: _SyncOnePayload,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Sync a single verified answer from GitHub by path (issue #46 step 9).

    Per-path counterpart to
    ``/verified-answers/bootstrap-from-github``. Fetches the
    ``<answer_id>.json`` file at ``path`` and creates or refreshes
    the matching local index entry.

    PR C's webhook handler will call this for each file in a push
    event's added / modified list. Admins can also call it directly
    to force-resync a single answer when they've manually edited
    its JSON on GitHub.

    Admin-only. Returns the per-file result
    (status / answer_id / path / reason).

    Returns 400 when GitHub publishing is disabled. Returns 502 when
    the GitHub fetch fails.
    """
    from data_concierge.gateway.verified_notebooks import (
        sync_one_answer_from_github,
    )

    result = await sync_one_answer_from_github(payload.path)
    if result["status"] == "skipped_disabled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub publishing is not enabled. Configure it under Settings → GitHub.",
        )
    if result["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not fetch {payload.path} from GitHub. "
                "Verify the file exists at that path and the token has access."
            ),
        )
    return result


@router.post("/notebooks/reconcile-drafts")
async def reconcile_drafts_endpoint(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Drop drafts that are also present in the verified folder on GitHub.

    The publish_verified() flow creates the verified file and then deletes
    the draft. If the delete failed (transient race, 5xx, draft never
    existed), the notebook can end up in both folders. This endpoint
    detects and cleans up those duplicates. Safe to call repeatedly.

    Admin-only. Returns the reconcile report
    (checked / duplicates_found / cleaned / failed).

    Returns 400 when GitHub publishing is disabled. Returns 502 when
    reconcile cannot read GitHub at all (auth, network, GitHub 5xx) —
    distinct from "I checked and there's nothing to do".
    """
    from data_concierge.gateway.github_publisher import (
        GitHubPublishError,
        reconcile_drafts_verified,
    )

    try:
        result = await reconcile_drafts_verified()
    except GitHubPublishError as e:
        logger.error(
            "Reconcile failed before inspecting GitHub",
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"Could not reach GitHub to reconcile drafts: {e}. Verify the token and try again."
            ),
        ) from e
    if result.get("disabled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="GitHub publishing is not enabled. Configure it under Settings → GitHub.",
        )
    return {
        "message": (
            f"Reconciled {result['checked']} draft(s); "
            f"cleaned {len(result['cleaned'])} duplicate(s); "
            f"{len(result['failed'])} failure(s)."
        ),
        **result,
    }


# =============================================================================
# GitHub Publishing Settings Endpoints
# =============================================================================


class GitHubSettingsRequest(BaseModel):
    """Request model for updating GitHub publishing settings.

    Note: ``enabled`` is intentionally absent. Publishing is active
    whenever ``repo`` and ``token`` are both set; admins explicitly
    pause/resume via ``POST /settings/github/pause``. See
    ``github_publisher.is_publishing_active``.
    """

    token: str = Field(default="", description="GitHub personal access token")
    repo: str = Field(
        default="", description="GitHub repo (owner/name)"
    )
    branch: str = Field(default="main", description="Target branch")
    drafts_folder: str = Field(default="drafts", description="Folder for draft notebooks")
    verified_folder: str = Field(default="verified", description="Folder for verified notebooks")
    verified_answers_folder: str = Field(
        default="verified-answers",
        description=(
            "Folder for verified quick answers (issue #46 step 9). Each "
            "answer is published as ``<folder>/<answer_id>.json``."
        ),
    )
    webhook_secret: str = Field(
        default="",
        description=(
            "Shared secret for verifying inbound GitHub webhook deliveries. "
            "Leave blank to keep the existing secret unchanged; submit a new "
            "non-empty value to replace it."
        ),
    )


class GitHubPauseRequest(BaseModel):
    """Request model for ``POST /settings/github/pause``.

    Set ``paused=true`` to halt all GitHub I/O without erasing the
    token; ``paused=false`` to resume. Pause/resume requires the
    integration to be configured (repo + token) — calling pause on an
    unconfigured integration returns 400.
    """

    paused: bool = Field(description="True to pause publishing, False to resume")


class LandingPageSettingsRequest(BaseModel):
    """Request model for ``POST /settings/landing`` (#109).

    All fields optional — only the provided ones are updated. Lets admins
    re-skin the public landing page (title/logo/call-to-action above the
    prompt; "Try asking"/"Powered by"/sample questions below it).
    """

    title: str | None = Field(default=None, description="Headline title")
    show_beta_badge: bool | None = Field(
        default=None, description="Show the Beta badge next to the title"
    )
    logo_url: str | None = Field(default=None, description="Logo image URL or path")
    call_to_action: str | None = Field(
        default=None, description="Lead paragraph shown above the prompt"
    )
    search_placeholder: str | None = Field(
        default=None, description="Placeholder text inside the prompt field"
    )
    try_asking_label: str | None = Field(
        default=None, description="Heading above the sample questions"
    )
    powered_by_label: str | None = Field(default=None, description="'Powered by' link label")
    powered_by_url: str | None = Field(default=None, description="'Powered by' link destination")
    sample_questions: list[str] | None = Field(
        default=None, description="Admin-curated sample questions"
    )


@router.get("/settings/landing")
async def get_landing_settings() -> dict[str, Any]:
    """Get the public landing-page configuration (#109).

    Public (no auth) so the landing page itself can render from it.
    Returns defaults merged with any admin-saved overrides.
    """
    from data_concierge.gateway.landing_page import load_landing_settings

    return load_landing_settings()


@router.post("/settings/landing")
async def update_landing_settings(
    request: LandingPageSettingsRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Update landing-page configuration (admin only, #109)."""
    from data_concierge.gateway.landing_page import save_landing_settings

    try:
        settings = save_landing_settings(request.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return {"message": "Landing-page settings saved", "settings": settings}


class SystemPromptSettingsRequest(BaseModel):
    """Request model for ``POST /settings/system-prompt`` (admin only).

    Both fields optional — only provided templates are updated. A blank string
    resets that template to the shipped default. Non-empty templates are
    validated (must render with only the allowed placeholders) before saving.
    """

    ckan_template: str | None = Field(
        default=None, description="System prompt for CKAN / open-data sources"
    )
    mcp_template: str | None = Field(
        default=None, description="System prompt for MCP-backed sources"
    )
    notebook_header_template: str | None = Field(
        default=None, description="Markdown template for the notebook title/how-to cell"
    )
    notebook_results_template: str | None = Field(
        default=None, description="Markdown template for the notebook results cell"
    )
    notebook_review_template: str | None = Field(
        default=None,
        description="System prompt for the adversarial notebook method review",
    )


@router.get("/settings/system-prompt")
async def get_system_prompt_settings(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get the effective LLM system-prompt templates + defaults (admin only)."""
    from data_concierge.gateway.system_prompt import load_system_prompt_settings

    return load_system_prompt_settings()


@router.post("/settings/system-prompt")
async def update_system_prompt_settings(
    request: SystemPromptSettingsRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Update the LLM system-prompt templates (admin only)."""
    from data_concierge.gateway.system_prompt import save_system_prompt_settings

    try:
        settings = save_system_prompt_settings(**request.model_dump(exclude_none=True))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return {"message": "System prompt saved", "settings": settings}


class RuntimeSettingsRequest(BaseModel):
    """Request model for ``POST /settings/runtime`` (admin only).

    ``query_timeout_seconds``: hard limit for a single analysis run. ``None``
    leaves it unchanged; ``0`` resets it to the default.
    """

    query_timeout_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Analysis timeout in seconds (0 resets to the default)",
    )


@router.get("/settings/runtime")
async def get_runtime_settings(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get the effective runtime settings (admin only)."""
    from data_concierge.gateway.runtime_settings import load_runtime_settings

    return load_runtime_settings()


@router.post("/settings/runtime")
async def update_runtime_settings(
    request: RuntimeSettingsRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Update the runtime settings (admin only)."""
    from data_concierge.gateway.runtime_settings import save_runtime_settings

    try:
        saved = save_runtime_settings(query_timeout_seconds=request.query_timeout_seconds)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return {"message": "Runtime settings saved", "settings": saved}


@router.get("/settings/github")
async def get_github_settings(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Get current GitHub publishing settings.

    Returns derived status fields (``configured``, ``paused``,
    ``active``) so the UI can render the right state without
    re-deriving the rules. The token and webhook secret are never
    echoed back — only set/unset flags and a masked token preview.
    """
    from data_concierge.gateway.github_publisher import (
        is_publishing_active,
        load_github_settings,
    )

    settings = load_github_settings()
    # Mask the token for display
    token = settings.get("token", "")
    masked = f"{'*' * 20}...{token[-4:]}" if len(token) > 4 else ("****" if token else "")
    # The webhook secret is never echoed back — admins only need to know
    # whether one is configured so they can decide whether to set it.
    webhook_secret = settings.get("webhook_secret", "") or ""
    configured = bool(token) and bool(settings.get("repo"))
    paused = bool(settings.get("paused", False))
    return {
        "configured": configured,
        "paused": paused,
        "active": is_publishing_active(settings),
        "token_set": bool(token),
        "token_masked": masked,
        "repo": settings.get("repo", ""),
        "branch": settings.get("branch", "main"),
        "drafts_folder": settings.get("drafts_folder", "drafts"),
        "verified_folder": settings.get("verified_folder", "verified"),
        "verified_answers_folder": settings.get("verified_answers_folder", "verified-answers"),
        "webhook_secret_set": bool(webhook_secret),
    }


@router.post("/settings/github")
async def update_github_settings(
    request: GitHubSettingsRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Update GitHub publishing settings.

    Both ``token`` and ``webhook_secret`` follow a "blank = preserve"
    convention: pass a non-empty string to set/rotate the secret, or
    omit / pass an empty string to keep the existing value. This
    matches the GET response which never echoes back the raw secrets —
    admins can't see what's currently there, so a blank submit MUST
    not clear it.

    Does NOT touch ``paused`` — use ``POST /settings/github/pause`` for
    that. Saving the config form is a routine config edit and should
    not silently flip publishing state.
    """
    from data_concierge.gateway.github_publisher import (
        is_publishing_active,
        load_github_settings,
        save_github_settings,
    )

    settings = load_github_settings()
    settings["repo"] = request.repo
    settings["branch"] = request.branch
    settings["drafts_folder"] = request.drafts_folder
    settings["verified_folder"] = request.verified_folder
    settings["verified_answers_folder"] = request.verified_answers_folder

    # Only update token if a new one is provided (non-empty)
    if request.token:
        settings["token"] = request.token
    # Same blank-preserves rule for the webhook secret.
    if request.webhook_secret:
        settings["webhook_secret"] = request.webhook_secret

    save_github_settings(settings)
    return {
        "message": "GitHub settings updated",
        "configured": bool(settings.get("token")) and bool(settings.get("repo")),
        "paused": bool(settings.get("paused", False)),
        "active": is_publishing_active(settings),
        "webhook_secret_set": bool(settings.get("webhook_secret")),
    }


@router.post("/settings/github/pause")
async def pause_github_publishing(
    request: GitHubPauseRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Pause or resume GitHub publishing.

    Pause is an explicit operator kill-switch that halts all GitHub
    I/O (publish, fetch, sync) without erasing the token. Requires
    the integration to be configured (both ``repo`` and ``token``) —
    pausing an unconfigured integration is meaningless and returns
    400 with a hint to finish configuring first.

    The split between this endpoint and ``POST /settings/github``
    is intentional: routine config edits MUST NOT silently flip
    publishing state, and the UI surfaces pause/resume as a
    deliberate action with consequences (verified library will drift
    from GitHub until resumed).
    """
    from data_concierge.gateway.github_publisher import (
        is_publishing_active,
        load_github_settings,
        save_github_settings,
    )

    settings = load_github_settings()
    configured = bool(settings.get("token")) and bool(settings.get("repo"))
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Cannot pause/resume: GitHub publishing is not configured. "
                "Set both repo and token before pausing."
            ),
        )
    settings["paused"] = bool(request.paused)
    save_github_settings(settings)
    logger.info(
        "GitHub publishing pause state changed",
        paused=settings["paused"],
    )
    return {
        "message": "Publishing paused" if settings["paused"] else "Publishing resumed",
        "paused": settings["paused"],
        "active": is_publishing_active(settings),
    }


@router.post("/settings/github/test")
async def test_github_connection(
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Test the GitHub connection with current settings."""
    from data_concierge.gateway.github_publisher import test_connection

    return await test_connection()


# =============================================================================
# Quick Answer Submission & Admin Review Endpoints
# =============================================================================


@router.post("/answers/submit")
async def submit_quick_answer_endpoint(
    request: QuickAnswerSubmitRequest,
    http_request: Request,
) -> dict[str, Any]:
    """Submit a quick answer for admin review.

    Quick answers are concise one-line factual answers with direct source
    links, used for simple TIER_1 lookups. They go through the same admin
    vetting pipeline as notebooks to ensure accuracy.
    """
    user = get_current_user(http_request) or {}
    submitted_by = (
        user.get("display_name") or user.get("user") or request.submitted_by or "anonymous"
    )
    submitter_email = user.get("email")
    submitter_auth_type = user.get("auth_type")

    submission = submit_quick_answer(
        query=request.query,
        answer=request.answer,
        source_links=request.source_links,
        submitted_by=submitted_by,
        submitter_email=submitter_email,
        submitter_auth_type=submitter_auth_type,
        data_source=request.data_source,
        confidence=request.confidence,
        tags=request.tags,
        variable=request.variable,
        place=request.place,
        date=request.date,
        value=request.value,
    )

    try:
        from data_concierge.gateway.admin_notifications import notify_new_submission

        await notify_new_submission(
            kind="quick_answer",
            query=request.query,
            submitted_by=submitted_by,
            submitter_email=submitter_email,
            submission_id=submission.submission_id,
        )
    except Exception as e:
        logger.warning("Admin notification failed (non-blocking)", error=str(e))

    return {
        "message": "Quick answer submitted for admin review",
        "submission_id": submission.submission_id,
        "status": submission.status.value,
    }


@router.get("/answers/submissions")
async def list_answer_submissions(
    status_filter: str | None = None,
) -> dict[str, Any]:
    """List quick answer submissions.

    Args:
        status_filter: Filter by status ('pending', 'approved', 'rejected').
                       If None, returns all submissions.
    """
    submissions = get_answer_submissions(status_filter)
    return {
        "count": len(submissions),
        "submissions": [s.model_dump() for s in submissions],
    }


@router.get("/answers/submissions/{submission_id}")
async def get_answer_submission_endpoint(submission_id: str) -> dict[str, Any]:
    """Get a specific quick answer submission."""
    submission = get_answer_submission(submission_id)
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Answer submission {submission_id} not found",
        )
    return submission.model_dump()


@router.post("/answers/submissions/{submission_id}/approve")
async def approve_answer_endpoint(
    submission_id: str,
    request: NotebookReviewRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Approve a quick answer submission, making it verified.

    Once approved, the answer is available for search and reuse
    when similar questions are asked.

    **Write-through SSOT (issue #46 step 9):** when GitHub publishing is
    enabled, the answer is pushed to ``verified-answers/<answer_id>.json``
    FIRST, then committed to the local index. If the GitHub publish
    fails, this endpoint returns 502 and the submission stays PENDING —
    so a transient GitHub failure can never leave a "verified locally,
    missing on GitHub" inconsistency. ``verified_at`` is pre-computed
    ONCE and reused as the local index's ``github_synced_at`` so the
    two stay coherent (the same anchor that makes
    ``bootstrap_answers_from_github`` recoverable in the next PR).
    When GitHub publishing is disabled, the behavior collapses to a
    local-only commit, same as before.
    """
    from datetime import UTC
    from datetime import datetime as _dt

    from data_concierge.gateway.github_publisher import (
        GitHubPublishError,
        publish_verified_answer,
    )

    # 1. Peek the submission (read-only) so we can validate without mutating.
    sub_data = get_answer_submission(submission_id)
    if sub_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Answer submission {submission_id} not found",
        )
    if sub_data.status != ReviewStatus.PENDING:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Answer submission {submission_id} not found or already reviewed",
        )

    # Forward dedup: keep exactly one verified answer per question. If this
    # question is already verified, collapse the submission into that answer
    # instead of publishing/creating a second copy (no orphaned GitHub file).
    existing_ans = find_verified_answer_by_question(sub_data.query)
    if existing_ans is not None:
        collapse_answer_submission_as_duplicate(
            submission_id=submission_id,
            existing_answer_id=existing_ans.answer_id,
            reviewed_by=_reviewer_identity(_admin),
            admin_notes=request.admin_notes,
        )
        logger.info(
            "Approved answer submission collapsed into existing verified answer",
            submission_id=submission_id,
            answer_id=existing_ans.answer_id,
        )
        return {
            "message": (
                "Answer approved; collapsed into existing verified answer "
                "for the same question (no duplicate created)."
            ),
            "answer_id": existing_ans.answer_id,
            "submission_id": submission_id,
            "deduplicated": True,
            "github_verified": None,
        }

    # 2. Pre-compute the verification timestamp ONCE so the published
    # file's verified_at and the local index's github_synced_at refer
    # to the same instant. The published file IS the provenance, so
    # we need to build the answer payload up front (before the local
    # commit) — that means we generate the answer_id here too, then
    # pass it down to approve_quick_answer below.
    verified_at = _dt.now(UTC).isoformat().replace("+00:00", "Z")
    keywords = extract_keywords(sub_data.query + " " + sub_data.answer)
    answer_id = str(uuid.uuid4())
    # Capture WHO verified this from the authenticated admin session rather
    # than the client-supplied ``reviewed_by`` (the UI hardcodes "admin") — #73.
    reviewer = _reviewer_identity(_admin)
    # Build the EXACT payload that will land in the local index, so
    # bootstrap can round-trip it back without divergence. usage_count
    # stays at 0 — it's local-only operational state and is explicitly
    # NOT written to GitHub (each increment would otherwise dirty the
    # repo history).
    answer_payload = VerifiedAnswer(
        answer_id=answer_id,
        submission_id=submission_id,
        query=sub_data.query,
        answer=sub_data.answer,
        source_links=sub_data.source_links,
        verified_at=verified_at,
        verified_by=reviewer,
        admin_notes=request.admin_notes,
        tags=sub_data.tags,
        data_source=sub_data.data_source,
        confidence=sub_data.confidence,
        submitted_by=sub_data.submitted_by,
        input_tokens=sub_data.input_tokens,
        output_tokens=sub_data.output_tokens,
        variable=sub_data.variable,
        place=sub_data.place,
        date=sub_data.date,
        value=sub_data.value,
        keywords=keywords,
        # github_path/synced_at are filled in below after publish.
    ).model_dump()

    # 3. Publish to GitHub. publish_verified_answer returns None when
    # GitHub is disabled; raises GitHubPublishError on real failures
    # (HTTP, network, auth).
    try:
        github_result = await publish_verified_answer(
            answer_id=answer_id,
            query=sub_data.query,
            answer_json=answer_payload,
            reason=request.admin_notes,
            reviewer=reviewer,
        )
    except GitHubPublishError as e:
        logger.error(
            "GitHub publish failed; aborting answer approval to preserve write-through SSOT",
            submission_id=submission_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"GitHub publish failed: {e}. The submission is unchanged; "
                "retry once the GitHub-side issue is resolved."
            ),
        ) from e

    # 4. Commit local state. Reuse verified_at as github_synced_at so
    # the published file and the index entry agree on the moment of
    # approval. When GitHub is disabled, github_path/synced_at stay
    # None but the local copy still carries the verified_at provenance.
    github_path = github_result.get("path") if github_result else None
    github_synced_at = verified_at if github_result else None

    # Stamp the published-coords on the payload we already built (so
    # the next read sees the same payload that's on GitHub) and let
    # approve_quick_answer perform the index commit with the matching
    # github fields and pre-computed verified_at.
    verified = approve_quick_answer(
        submission_id=submission_id,
        reviewed_by=reviewer,
        admin_notes=request.admin_notes,
        answer_id=answer_id,
        github_path=github_path,
        github_synced_at=github_synced_at,
        verified_at=verified_at,
    )
    if not verified:
        # Race: submission was approved/rejected between the peek and
        # the commit. If we already pushed to GitHub, that file is now
        # orphaned (no local index entry points at this answer_id) —
        # log loudly so an admin can reconcile via the bootstrap path.
        if github_result:
            logger.error(
                "Answer approval race: submission state changed between peek "
                "and commit. GitHub file is published but no local index "
                "entry was created — reconcile via bootstrap.",
                submission_id=submission_id,
                answer_id=answer_id,
                github_path=github_path,
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Answer submission {submission_id} not found or already reviewed",
        )
    return {
        "message": "Quick answer approved and verified",
        "answer_id": verified.answer_id,
        "submission_id": submission_id,
        "github_path": github_path,
        "github_synced_at": github_synced_at,
    }


@router.post("/answers/submissions/{submission_id}/reject")
async def reject_answer_endpoint(
    submission_id: str,
    request: NotebookReviewRequest,
    _admin: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    """Reject a quick answer submission."""
    # Record the authenticated admin as the rejecter, not the client-supplied
    # ``reviewed_by`` (which the UI hardcodes to "admin") — #73.
    reviewer = _reviewer_identity(_admin)
    submission = reject_quick_answer(
        submission_id=submission_id,
        reviewed_by=reviewer,
        admin_notes=request.admin_notes,
    )
    if not submission:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Answer submission {submission_id} not found or already reviewed",
        )
    return {
        "message": "Quick answer rejected",
        "submission_id": submission_id,
    }


@router.get("/verified-answers")
async def list_verified_answers_endpoint() -> dict[str, Any]:
    """List all verified quick answers."""
    answers = get_verified_answers()
    gh_settings = load_github_settings()
    return {
        "count": len(answers),
        "answers": [
            {
                "answer_id": a.answer_id,
                "query": a.query,
                "answer": a.answer,
                "source_links": a.source_links,
                "tags": a.tags,
                "data_source": a.data_source,
                "verified_at": a.verified_at,
                "verified_by": a.verified_by,
                "submitted_by": a.submitted_by or "",
                "confidence": a.confidence,
                "input_tokens": a.input_tokens,
                "output_tokens": a.output_tokens,
                "usage_count": a.usage_count,
                "github_path": a.github_path,
                "github_url": build_blob_url(a.github_path, gh_settings),
            }
            for a in answers
        ],
    }


@router.get("/verified-answers/{answer_id}")
async def get_verified_answer_endpoint(answer_id: str) -> dict[str, Any]:
    """Get a specific verified quick answer."""
    answer = get_verified_answer(answer_id)
    if not answer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verified answer {answer_id} not found",
        )
    # Increment usage count when retrieving
    increment_answer_usage(answer_id)
    payload = answer.model_dump()
    payload["github_url"] = build_blob_url(answer.github_path)
    return payload


@router.post("/verified-answers/search")
async def search_verified_answers_endpoint(
    request: VerifiedAnswerSearchRequest,
) -> dict[str, Any]:
    """Search verified quick answers for similar queries.

    This is used to find pre-verified answers that match a user's
    question. Results are ranked by keyword similarity.
    """
    results = search_verified_answers(
        query=request.query,
        threshold=request.threshold,
        max_results=request.max_results,
    )
    return {
        "query": request.query,
        "count": len(results),
        "results": [r.model_dump() for r in results],
    }


@router.get("/notebooks/{query_id}/verification")
async def get_notebook_verification(query_id: str) -> dict[str, Any]:
    """Notebook verification result for a query (#131).

    The client polls this after a query whose response had
    ``verification_pending``. Returns the updated confidence once the
    notebook has been executed and reconciled against the answer.

    ``status`` is one of:

    * ``disabled``  — verification is switched off for this deployment
    * ``pending``   — scheduled, not finished. May stay this way: Cloud Run
      throttles CPU outside a request, so a background run can stall or be
      lost when the instance scales down. Clients should give up eventually
      rather than poll forever.
    * ``complete``  — a verdict is available
    * ``error``     — the verifier itself failed; confidence is unchanged
    * ``not_found`` — no verification was ever scheduled for this query
    """
    # Records exist whenever EITHER execution or the adversarial review is
    # enabled (review-only mode still schedules the pass).
    if not (settings.notebook_verification_enabled or settings.notebook_review_enabled):
        return {"query_id": query_id, "status": "disabled"}

    from data_concierge.gateway.notebook_verification import get_verification

    result = get_verification(query_id)
    if result is None:
        return {"query_id": query_id, "status": "not_found"}
    return result


# NOTE: This route uses a path parameter and must be defined AFTER all
# static /notebooks/* routes to avoid capturing "submit", "submissions",
# and "admin" as query_id values. The /verification sub-path above must stay
# ABOVE it, or this wildcard swallows it.
@router.get("/notebooks/{query_id}")
async def get_notebook(query_id: str):
    """Get a generated notebook by query ID.

    Returns the notebook JSON content for API consumption.
    """
    notebook_key = f"{_NOTEBOOKS_PREFIX}/{query_id}.ipynb"
    notebook_json = storage.read_json(notebook_key)

    if not notebook_json:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook for query {query_id} not found or expired",
        )

    return JSONResponse(content=notebook_json)


@router.get("/notebooks/{query_id}/download")
async def download_notebook(query_id: str):
    """Download a generated notebook file.

    Returns the notebook as a downloadable .ipynb file.
    """
    notebook_key = f"{_NOTEBOOKS_PREFIX}/{query_id}.ipynb"

    if isinstance(storage, GCSStorage):
        tmp_path = storage.download_to_tmp(notebook_key)
        if not tmp_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Notebook for query {query_id} not found or expired",
            )
        return FileResponse(
            path=tmp_path,
            media_type="application/x-ipynb+json",
            filename=f"data_concierge_{query_id}.ipynb",
        )

    local_path = storage.full_path(notebook_key)
    if not Path(local_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notebook for query {query_id} not found or expired",
        )
    return FileResponse(
        path=local_path,
        media_type="application/x-ipynb+json",
        filename=f"data_concierge_{query_id}.ipynb",
    )


@router.get("/notebooks/{query_id}/logs")
async def get_notebook_logs(query_id: str):
    """Get agent execution logs for a query.

    Returns the raw LLM conversation, tool calls, and timing data
    captured during notebook generation.
    """
    log_key = f"{_NOTEBOOKS_PREFIX}/{query_id}_log.json"
    log_data = storage.read_json(log_key)

    if not log_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Logs for query {query_id} not found",
        )

    return JSONResponse(content=log_data)


# =============================================================================
# Fair Store — structured data dictionary (issue #136)
# =============================================================================
# Reads the column-level dictionary that onboarding already builds (labels,
# types, qsv stats, top values) and exposes it over the API. Until now this
# metadata was only visible to the agent; these endpoints make it a first-
# class, queryable surface — the backend the interactive visual data
# dictionary renders from. Read-only and public: it is verified public
# metadata, not the underlying rows.

# Site ids are lowercase slugs. The value is interpolated into a storage key
# (ckan_onboard/{site}/index.json), so a value containing "/" or ".." would
# read a different index.json under LocalStorage — reject anything that is not
# a plain slug before it reaches the storage layer.
_SITE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _validate_site(site: str) -> str:
    if not site or not _SITE_SLUG_RE.match(site):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid site id; expected a lowercase slug (letters, digits, - and _).",
        )
    return site


@router.get("/fairstore/search")
async def fairstore_search(q: str, site: str = "wprdc", limit: int = 10) -> dict[str, Any]:
    """Search the Fair Store data dictionary by keyword.

    Returns matching datasets with their column and record counts, ranked by
    relevance. Use the resource_id against /fairstore/resource/{id} for the
    full column-level schematic.
    """
    from data_concierge.data_layer.onboard_index import get_onboarded_index

    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="q is required")

    site = _validate_site(site)
    index = get_onboarded_index(site)
    if not index.load(site):
        return {"site": site, "query": query, "count": 0, "results": [], "available": False}

    results = index.search(query, n_results=max(1, min(limit, 50)))
    return {
        "site": site,
        "query": query,
        "count": len(results),
        "available": True,
        "results": [
            {
                "resource_id": r.get("resource_id", ""),
                "resource_name": r.get("resource_name", ""),
                "dataset_id": r.get("dataset_id", ""),
                "dataset_title": r.get("dataset_title", ""),
                "description": r.get("description", ""),
                "record_count": r.get("record_count", 0),
                "column_count": r.get("column_count", 0),
                "score": round(r.get("score", 0.0), 3),
            }
            for r in results
        ],
    }


@router.get("/fairstore/resource/{resource_id}")
async def fairstore_resource(resource_id: str, site: str = "wprdc") -> dict[str, Any]:
    """Full column-level data dictionary for one resource.

    Each column carries its label, type, qsv statistics (min/max/mean/median/
    stddev/nullcount/cardinality) and top values — the schematic that gives a
    dataset context before anyone reads a single row.
    """
    from data_concierge.data_layer.onboard_index import get_onboarded_index

    site = _validate_site(site)
    index = get_onboarded_index(site)
    if not index.load(site):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No onboarded dictionary for site {site!r}",
        )

    detail = index.get_resource_detail(resource_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Resource {resource_id!r} not found in the {site!r} dictionary",
        )
    return detail
