"""FastAPI application entry point."""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

# Load .env into os.environ before importing any modules that read env vars at
# import time (Auth0 client, approved-members seed, etc.).
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from data_concierge.core.config import settings  # noqa: E402
from data_concierge.core.logging import get_logger  # noqa: E402
from data_concierge.gateway.github_webhook import router as github_webhook_router  # noqa: E402
from data_concierge.gateway.router import router as gateway_router  # noqa: E402
from data_concierge.mcp.router import router as mcp_router  # noqa: E402

logger = get_logger(__name__)


async def _github_sync_loop(interval_hours: float) -> None:
    """Periodically pull verified notebooks from GitHub.

    Skips silently when GitHub publishing is disabled. Errors are logged
    and the loop continues — a transient API failure shouldn't kill the
    background task.
    """
    interval_seconds = max(60.0, interval_hours * 3600.0)
    from data_concierge.gateway.verified_notebooks import sync_all_from_github

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            result = await sync_all_from_github()
            if result.get("skipped"):
                continue
            logger.info(
                "Periodic GitHub sync complete",
                updated=len(result.get("updated", [])),
                failed=len(result.get("failed", [])),
                missing=len(result.get("missing_path", [])),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Periodic GitHub sync failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info(
        "Starting Data Concierge API",
        environment=settings.environment,
        version=settings.app_version,
    )

    # Initialize MCP server registry and auto-connect enabled servers
    from data_concierge.mcp.registry import get_mcp_registry

    mcp_registry = get_mcp_registry()
    logger.info("MCP registry initialized", servers=len(mcp_registry.get_all_servers()))

    connect_results = await mcp_registry.connect_all_auto()
    if connect_results:
        for sid, status in connect_results.items():
            logger.info("MCP auto-connect", server_id=sid, status=status)

    # Background: periodically pull verified notebooks from GitHub so admin
    # edits made directly in the repo flow back into the verified library.
    sync_task: asyncio.Task[None] | None = None
    if settings.github_sync_interval_hours and settings.github_sync_interval_hours > 0:
        sync_task = asyncio.create_task(
            _github_sync_loop(settings.github_sync_interval_hours)
        )
        logger.info(
            "GitHub verified-notebook sync scheduled",
            every_hours=settings.github_sync_interval_hours,
        )

    yield

    # Shutdown
    logger.info("Shutting down Data Concierge API")

    if sync_task is not None:
        sync_task.cancel()
        try:
            await sync_task
        except (asyncio.CancelledError, Exception):
            pass

    # Disconnect all MCP servers
    for server in mcp_registry.get_connected_servers():
        try:
            await mcp_registry.disconnect_server(server.config.id)
        except Exception:
            pass

    # Drain lazily-created HTTP connection pools so shutdown is clean and
    # no pools leak (issue #96).
    from data_concierge.data_layer.connectors import close_all_clients

    await close_all_clients()


app = FastAPI(
    title=settings.app_name,
    description=(
        "An intelligent AI assistant for accessing and understanding "
        "federal statistical data. Provides natural language queries, "
        "reproducible notebooks, and citation-backed answers."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(gateway_router)
app.include_router(mcp_router)
app.include_router(github_webhook_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }
