"""Offline coverage for stream delivery, task isolation, auth, and cancellation."""

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.graph import END, StateGraph
from starlette.requests import ClientDisconnect

from data_concierge.agents.state import GraphState, create_initial_state
from data_concierge.agents.supervisor import _with_progress
from data_concierge.core.query_progress import emit_progress, emit_tool_progress, progress_sink
from data_concierge.gateway.query_stream import QueryStreamingResponse, query_events, router
from data_concierge.gateway.router import QueryResponse, _create_session_token, _user_tokens


def response(query_id: str = "sample") -> QueryResponse:
    return QueryResponse(
        query_id=query_id,
        answer="A citation-backed answer.",
        confidence=0.8,
        confidence_level="medium",
        tier="tier_2",
        processing_time_ms=25,
    )


def parse_frame(frame: str) -> dict:
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[6:])


async def test_graph_nodes_publish_on_entry_and_preserve_execution_trace() -> None:
    events = []

    async def compute(state: GraphState) -> GraphState:
        # The UI can see the actual node before it starts its work.
        assert events[-1]["stage"] == "computing"
        state["execution_trace"].append({"agent": "offline", "action": "compute"})
        state["answer"] = "Completed"
        return state

    graph = StateGraph(GraphState)
    graph.add_node("compute", _with_progress(compute, "computing", "Calculating results"))
    graph.set_entry_point("compute")
    graph.add_edge("compute", END)
    with progress_sink(events.append):
        result = await graph.compile().ainvoke(create_initial_state("A question"))
    assert result["answer"] == "Completed"
    assert result["execution_trace"] == [{"agent": "offline", "action": "compute"}]
    assert len(events) == 1


async def test_progress_arrives_before_completion_and_preserves_response() -> None:
    finish = asyncio.Event()

    async def operation() -> QueryResponse:
        emit_progress("searching", "Searching datasets")
        await finish.wait()
        return response()

    stream = query_events(operation)
    assert parse_frame(await anext(stream))["stage"] == "preparing"
    assert parse_frame(await anext(stream))["stage"] == "searching"
    finish.set()
    result = parse_frame(await anext(stream))
    assert result == {"type": "result", "data": response().model_dump(mode="json")}
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


async def test_streams_do_not_share_progress_and_restore_caller_context() -> None:
    ready = [asyncio.Event(), asyncio.Event()]
    outer_events = []

    async def collect(index: int) -> list[dict]:
        async def operation() -> QueryResponse:
            ready[index].set()
            await ready[1 - index].wait()
            emit_progress(f"source_{index}", f"Checking source {index}")
            return response(str(index))

        return [parse_frame(frame) async for frame in query_events(operation)]

    with progress_sink(outer_events.append):
        first, second = await asyncio.gather(collect(0), collect(1))
        emit_progress("outer", "Original context")

    assert [event.get("stage") for event in first[:-1]] == ["preparing", "source_0"]
    assert [event.get("stage") for event in second[:-1]] == ["preparing", "source_1"]
    assert first[-1]["data"]["query_id"] == "0"
    assert second[-1]["data"]["query_id"] == "1"
    assert outer_events == [{"type": "progress", "stage": "outer", "message": "Original context"}]
    emit_progress("outside", "No sink attached")
    assert len(outer_events) == 1


async def test_closing_stream_cancels_and_awaits_nested_work() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def child() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Cleanup must get an opportunity to await, including under AnyIO.
            await asyncio.sleep(0)
            cleaned_up.set()

    async def operation() -> QueryResponse:
        await asyncio.gather(child())
        return response()

    stream = query_events(operation)
    await anext(stream)
    await started.wait()
    await stream.aclose()
    assert cleaned_up.is_set()


async def test_cancelled_stream_reader_cancels_worker() -> None:
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def operation() -> QueryResponse:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.set()
        return response()

    async def read() -> None:
        async for _ in query_events(operation):
            pass

    reader = asyncio.create_task(read())
    await started.wait()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    assert cleaned_up.is_set()


@pytest.mark.parametrize("asgi_version", ["2.3", "2.4"])
async def test_client_disconnect_finishes_worker_cleanup(asgi_version: str) -> None:
    disconnect = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def operation() -> QueryResponse:
        emit_progress("searching", "Searching datasets")
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned_up.set()
        return response()

    async def receive() -> dict:
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if b"searching" in message.get("body", b""):
            if asgi_version == "2.4":
                raise OSError("Socket closed")
            disconnect.set()

    stream_response = QueryStreamingResponse(query_events(operation))
    scope = {"type": "http", "asgi": {"spec_version": asgi_version}}
    if asgi_version == "2.4":
        with pytest.raises(ClientDisconnect):
            await stream_response(scope, receive, send)
    else:
        await stream_response(scope, receive, send)
    assert cleaned_up.is_set()


async def test_errors_are_sanitized_and_terminal() -> None:
    async def operation() -> QueryResponse:
        raise RuntimeError("upstream failed: token=do-not-display SELECT private_column")

    events = [parse_frame(frame) async for frame in query_events(operation)]
    assert [event["type"] for event in events] == ["progress", "error"]
    assert "Please try again" in events[-1]["message"]
    assert "do-not-display" not in json.dumps(events)
    assert "SELECT" not in json.dumps(events)


def test_tool_progress_describes_source_without_exposing_tool_names() -> None:
    events = []
    with progress_sink(events.append):
        emit_tool_progress("mcp__census-data-api__get_dataset_info", "U.S. Census Bureau")
        emit_tool_progress("run_sql_query", "City Data Portal")
    assert events[0]["stage"] == "checking_dataset"
    assert "U.S. Census Bureau" in events[0]["message"]
    assert events[1]["stage"] == "querying_data"
    assert "City Data Portal" in events[1]["message"]
    assert "run_sql_query" not in json.dumps(events)


def test_endpoint_requires_auth_before_stream_and_validates_existing_request() -> None:
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        unauthenticated = client.post("/api/v1/query/stream", json={"query": "Population?"})
        assert unauthenticated.status_code == 401
        assert "text/event-stream" not in unauthenticated.headers["content-type"]

        token = _create_session_token("stream-test", "password")
        try:
            invalid = client.post(
                "/api/v1/query/stream",
                headers={"Authorization": f"Bearer {token}"},
                json={"query": ""},
            )
            assert invalid.status_code == 422
        finally:
            _user_tokens.pop(token, None)
