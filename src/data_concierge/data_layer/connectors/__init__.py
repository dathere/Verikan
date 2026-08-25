"""Data layer connectors for external services.

Also hosts a tiny registry of objects that own long-lived
``httpx.AsyncClient`` connection pools. Connectors create their clients
lazily and (before this) never closed them, so Cloud Run shutdowns were
slow and pools could leak (issue #96). Each connector registers itself
on construction; the FastAPI lifespan shutdown calls
:func:`close_all_clients` to drain every pool cleanly.
"""

from __future__ import annotations

import inspect
import weakref
from typing import Any

from data_concierge.core.logging import get_logger

logger = get_logger(__name__)

# WeakSet so merely registering never keeps an otherwise-dead connector
# alive; long-lived singletons (graph-held agents, lazy connector
# instances) stay referenced elsewhere and so survive to be closed.
_closeables: weakref.WeakSet[Any] = weakref.WeakSet()


def register_closeable(obj: Any) -> None:
    """Register an object exposing a ``close()`` method for shutdown cleanup.

    ``close()`` may be sync or async; :func:`close_all_clients` awaits it
    when it returns an awaitable.
    """
    _closeables.add(obj)


async def close_all_clients() -> int:
    """Close every registered HTTP client. Returns the number closed.

    Best-effort: a failure closing one client is logged and does not stop
    the others. Safe to call multiple times (connectors null out their
    client on close, so a second pass is a no-op).
    """
    closed = 0
    for obj in list(_closeables):
        close = getattr(obj, "close", None)
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
            closed += 1
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            logger.warning(
                "Error closing HTTP client",
                connector=type(obj).__name__,
                error=str(e),
            )
    if closed:
        logger.info("Closed HTTP clients at shutdown", count=closed)
    return closed
