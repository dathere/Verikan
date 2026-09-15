"""Authenticated query event stream with request-scoped cancellation.

The existing JSON endpoint remains the single implementation of query handling.
Each stream owns its worker; aborting the connection cancels that same task on
the same worker, without an instance-local job registry or polling endpoint.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from functools import partial
from typing import Any

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

from data_concierge.core.logging import get_logger
from data_concierge.core.query_progress import ProgressEvent, emit_progress, progress_sink
from data_concierge.gateway.router import (
    QueryRequest,
    QueryResponse,
    get_session_manager,
    process_query_endpoint,
    require_auth,
)
from data_concierge.gateway.session import SessionManager

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["concierge"])
_HEARTBEAT_SECONDS = 15.0


async def query_events(
    operation: Callable[[], Awaitable[QueryResponse]],
) -> AsyncIterator[str]:
    """Yield SSE frames and finish all worker cleanup before closing the stream."""
    events: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=128)

    def publish(event: ProgressEvent) -> None:
        # A slow reader must not block the analysis or grow memory without bound.
        # Terminal events use await put() below and are never discarded.
        with suppress(asyncio.QueueFull):
            events.put_nowait(dict(event))

    async def run_query() -> None:
        with progress_sink(publish):
            try:
                emit_progress("preparing", "Preparing your question")
                result = await operation()
                await events.put({"type": "result", "data": result.model_dump(mode="json")})
            except Exception as exc:
                logger.warning("Query stream failed", error_type=type(exc).__name__)
                await events.put(
                    {
                        "type": "error",
                        "message": "The analysis could not finish. Please try again or narrow your question.",
                    }
                )

    worker = asyncio.create_task(run_query(), name="query-stream-worker")
    try:
        while True:
            try:
                event = await asyncio.wait_for(events.get(), timeout=_HEARTBEAT_SECONDS)
            except TimeoutError:
                # Keep proxies from treating a slow source as an idle connection.
                # Heartbeats do not suggest a made-up analysis stage.
                yield ": keep-alive\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] in {"result", "error"}:
                break
    finally:
        worker.cancel()
        # Starlette cancels its AnyIO task group when it receives disconnect.
        # Shield cleanup from that scope so nested async HTTP/LLM work is awaited.
        with anyio.CancelScope(shield=True), suppress(asyncio.CancelledError):
            await worker


class QueryStreamingResponse(StreamingResponse):
    """Also close the iterator when ASGI 2.4 reports a failed socket send."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # A failed send can leave the async generator suspended at yield.
            # Explicit close guarantees cancellation without waiting for GC.
            with anyio.CancelScope(shield=True):
                close = getattr(self.body_iterator, "aclose", None)
                if close is not None:
                    await close()


@router.post("/query/stream")
async def stream_query_endpoint(
    request: QueryRequest,
    http_request: Request,
    _token: str = Depends(require_auth),
    sessions: SessionManager = Depends(get_session_manager),
) -> StreamingResponse:
    """Stream actual progress and one final QueryResponse; browser abort stops work."""
    operation = partial(process_query_endpoint, request, http_request, _token, sessions)
    return QueryStreamingResponse(
        query_events(operation),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
