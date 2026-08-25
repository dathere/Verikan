"""
Data Concierge Web UI Server.

A clean Bootstrap-based web interface for the Data Concierge system.
Serves the frontend and proxies API requests to the backend.

Run with:
    python -m data_concierge.ui.web
    # or
    ./run_web.sh
"""

import os
from pathlib import Path

# Load .env into os.environ before any module that reads env vars at import
# time (e.g. auth0_client, approved_members, roles). Pydantic Settings reads
# the file for its own purposes but does not populate os.environ.
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from collections.abc import AsyncGenerator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from fastapi.templating import Jinja2Templates  # noqa: E402

# Paths
UI_DIR = Path(__file__).parent
TEMPLATES_DIR = UI_DIR / "templates"
STATIC_DIR = UI_DIR / "static"

ACCESS_DENIED_HTML = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Access Denied - Verikan</title>
    <link rel="icon" type="image/svg+xml" href="/static/images/favicon.svg?v=20260630-verikan">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;550;600;650;700;750&display=swap" rel="stylesheet">
    <link href="/static/css/style.css?v=20260713-redesign" rel="stylesheet">
    <script>
        (function() {
            var s = localStorage.getItem('dc-theme') || 'light';
            document.documentElement.setAttribute('data-bs-theme', s);
        })();
    </script>
</head>
<body class="d-flex align-items-center justify-content-center p-3" style="min-height:100vh;">
    <div class="card" style="max-width:450px;width:100%;">
        <div class="card-body p-4 text-center">
            <img src="/static/images/logo-light.svg" alt="Verikan" height="44" class="mb-3 dc-logo">
            <i class="bi bi-shield-lock display-3 text-danger mb-3 d-block" aria-hidden="true"></i>
            <h1 class="h5 mb-3">Admin Access Required</h1>
            <p class="text-muted">
                You are signed in as <strong>{username}</strong>, but your account
                does not have admin privileges. Ask an existing admin to grant you access.
            </p>
            <a href="/" class="btn btn-primary mt-3">
                <i class="bi bi-arrow-left me-1" aria-hidden="true"></i>Back to Chat
            </a>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var theme = localStorage.getItem('dc-theme') || 'light';
            var src = theme === 'light' ? '/static/images/logo-light.svg' : '/static/images/logo.png';
            document.querySelectorAll('img.dc-logo').forEach(function(img) { img.src = src; });
        });
    </script>
</body>
</html>
"""

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Web UI app lifespan — drains HTTP connection pools on shutdown.

    Connectors create ``httpx.AsyncClient`` pools lazily and never closed
    them, leaving pools open on Cloud Run shutdown (issue #96). Close them
    all when the app stops.
    """
    yield
    from data_concierge.data_layer.connectors import close_all_clients

    await close_all_clients()


# Create FastAPI app
app = FastAPI(
    title="Verikan Web UI",
    description="Web interface for Verikan — the verified Data Concierge",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# Include the API router from the main app
def include_api_routes():
    """Include the API routes from the gateway and MCP routers."""
    try:
        from data_concierge.gateway.router import router as api_router
        app.include_router(api_router)
    except ImportError as e:
        print(f"Warning: Could not import API router: {e}")
        print("API endpoints will not be available.")

    try:
        from data_concierge.mcp.router import router as mcp_router
        app.include_router(mcp_router)
    except ImportError as e:
        print(f"Warning: Could not import MCP router: {e}")
        print("MCP endpoints will not be available.")

    try:
        from data_concierge.gateway.github_webhook import router as github_webhook_router
        app.include_router(github_webhook_router)
    except ImportError as e:
        print(f"Warning: Could not import GitHub webhook router: {e}")
        print("GitHub webhook endpoint will not be available.")

    try:
        from data_concierge.gateway.evidence_router import router as evidence_router
        app.include_router(evidence_router)
    except ImportError as e:
        print(f"Warning: Could not import evidence router: {e}")
        print("Public evidence endpoints will not be available.")


include_api_routes()


# HTML Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface."""
    from data_concierge.gateway.landing_page import load_landing_settings

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"landing": load_landing_settings()},
    )


def _get_admin_user(request: Request) -> dict | None:
    """Return the user info dict if the caller is a logged-in admin, else None."""
    from data_concierge.gateway.roles import is_admin
    from data_concierge.gateway.router import get_current_user

    user = get_current_user(request)
    if not user:
        return None
    user_id = user.get("user", "")
    email = user.get("email", "")
    if not (is_admin(user_id) or (email and is_admin(email))):
        return None
    return user


@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    """Serve the user documentation / help page (public)."""
    return templates.TemplateResponse(request=request, name="docs.html")


@app.get("/library", response_class=HTMLResponse)
async def library(request: Request):
    """Serve the public Verified Library catalog (no login required)."""
    return templates.TemplateResponse(request=request, name="library.html")


@app.get("/dictionary", response_class=HTMLResponse)
async def dictionary(request: Request):
    """Serve the public Fair Store data dictionary browser (#136)."""
    return templates.TemplateResponse(request=request, name="dictionary.html")


@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Serve the admin panel (requires logged-in admin)."""
    from data_concierge.gateway.router import get_current_user

    user = get_current_user(request)
    if not user:
        # Not logged in at all — redirect to home to log in first
        return RedirectResponse(url="/?next=/admin", status_code=302)
    if not _get_admin_user(request):
        # Logged in but not admin
        username = user.get("user", "unknown")
        html = ACCESS_DENIED_HTML.replace("{username}", username)
        return HTMLResponse(content=html, status_code=403)
    return templates.TemplateResponse(request=request, name="admin.html")


@app.get("/settings/mcp", response_class=HTMLResponse)
async def mcp_settings(request: Request):
    """Serve the MCP server management page (requires logged-in admin)."""
    from data_concierge.gateway.router import get_current_user

    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/?next=/settings/mcp", status_code=302)
    if not _get_admin_user(request):
        username = user.get("user", "unknown")
        html = ACCESS_DENIED_HTML.replace("{username}", username)
        return HTMLResponse(content=html, status_code=403)
    return templates.TemplateResponse(request=request, name="mcp_settings.html")


PENDING_APPROVAL_HTML = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pending Approval - Verikan</title>
    <link rel="icon" type="image/svg+xml" href="/static/images/favicon.svg?v=20260630-verikan">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;550;600;650;700;750&display=swap" rel="stylesheet">
    <link href="/static/css/style.css?v=20260713-redesign" rel="stylesheet">
    <script>
        (function() {
            var s = localStorage.getItem('dc-theme') || 'light';
            document.documentElement.setAttribute('data-bs-theme', s);
        })();
    </script>
</head>
<body class="d-flex align-items-center justify-content-center p-3" style="min-height:100vh;">
    <div class="card" style="max-width:500px;width:100%;">
        <div class="card-body p-4 text-center">
            <img src="/static/images/logo-light.svg" alt="Verikan" height="44" class="mb-3 dc-logo">
            <i class="bi bi-hourglass-split display-3 text-warning mb-3 d-block" aria-hidden="true"></i>
            <h1 class="h5 mb-3">Awaiting Verification</h1>
            <p class="text-muted">
                Thanks for signing in{email_html}.
                Your account isn't on the approved list yet — an administrator
                has been notified and will review your access shortly.
            </p>
            <a href="/" class="btn btn-primary mt-3">
                <i class="bi bi-arrow-left me-1" aria-hidden="true"></i>Back to Chat
            </a>
        </div>
    </div>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            var theme = localStorage.getItem('dc-theme') || 'light';
            var src = theme === 'light' ? '/static/images/logo-light.svg' : '/static/images/logo.png';
            document.querySelectorAll('img.dc-logo').forEach(function(img) { img.src = src; });
        });
    </script>
</body>
</html>
"""


@app.get("/auth/pending", response_class=HTMLResponse)
async def auth_pending(email: str = "", reason: str = ""):
    """Landing page shown when an Auth0-authenticated user is not on the approved list."""
    email_html = f" as <strong>{email}</strong>" if email else ""
    html = PENDING_APPROVAL_HTML.replace("{email_html}", email_html)
    return HTMLResponse(content=html)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "data-concierge-web"}




def main():
    """Run the web server."""
    port = int(os.environ.get("PORT", 8501))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"\n{'='*60}")
    print("Data Concierge Web UI")
    print(f"{'='*60}")
    print(f"\n  Chat Interface: http://localhost:{port}/")
    print(f"  Admin Panel:    http://localhost:{port}/admin")
    print(f"  API Docs:       http://localhost:{port}/docs")
    print(f"\n{'='*60}\n")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
