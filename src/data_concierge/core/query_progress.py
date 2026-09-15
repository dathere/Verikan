"""Task-local, optional progress reporting for a running query.

Only publish user-facing stage descriptions here. Tool arguments, SQL, model
messages, and exception text belong in the evidence log, never the stream.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypedDict


class ProgressEvent(TypedDict):
    type: str
    stage: str
    message: str


ProgressSink = Callable[[ProgressEvent], None]
_progress_sink: ContextVar[ProgressSink | None] = ContextVar("query_progress_sink", default=None)


@contextmanager
def progress_sink(sink: ProgressSink) -> Iterator[None]:
    """Attach a sink to this request and its child tasks, then restore it."""
    token = _progress_sink.set(sink)
    try:
        yield
    finally:
        _progress_sink.reset(token)


def emit_progress(stage: str, message: str) -> None:
    """Report a real operation as it starts; ordinary JSON requests are a no-op."""
    sink = _progress_sink.get()
    if sink is not None:
        sink({"type": "progress", "stage": stage, "message": message})


def emit_tool_progress(tool_name: str, source_name: str) -> None:
    """Describe the operation and registered source without exposing arguments."""
    bare_name = tool_name.rsplit("__", 1)[-1].lower()
    source = " ".join(source_name.split())[:120] or "the data source"
    if "search" in bare_name or "find" in bare_name:
        emit_progress("searching", f"Searching datasets in {source}")
    elif any(word in bare_name for word in ("info", "describe", "metadata", "schema")):
        emit_progress("checking_dataset", f"Checking dataset details in {source}")
    elif "list" in bare_name or "catalog" in bare_name:
        emit_progress("searching", f"Exploring available data in {source}")
    elif "sql" in bare_name:
        emit_progress("querying_data", f"Calculating results from {source}")
    else:
        emit_progress("loading_data", f"Loading data from {source}")
