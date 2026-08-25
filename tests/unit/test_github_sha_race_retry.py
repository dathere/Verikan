"""Tests for the GitHub 409 SHA-race retry in `_create_or_update_file`.

When two `publish_verified()` calls target the same path concurrently the
loser sees HTTP 409 because the SHA it fetched before its PUT was
invalidated by the winner. The helper should detect that, refetch the SHA,
and retry once. These tests cover the common cases.

See issue #46 follow-ups, item A in
an earlier production incident.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from data_concierge.gateway import github_publisher
from data_concierge.gateway.github_publisher import (
    GitHubPublishError,
    _create_or_update_file,
    publish_verified,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for httpx.Response used by _create_or_update_file."""

    def __init__(
        self, status_code: int, text: str = "", json_body: dict[str, Any] | None = None
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json = json_body or {}

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if 400 <= self.status_code < 600:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("PUT", "https://api.github.com/x"),
                response=httpx.Response(self.status_code, text=self.text),
            )


class _FakeAsyncClient:
    """Captures PUT/GET calls and returns scripted responses for each."""

    def __init__(
        self,
        sha_sequence: list[str | None],
        put_sequence: list[_FakeResponse],
    ) -> None:
        self._sha_sequence = list(sha_sequence)
        self._put_sequence = list(put_sequence)
        self.put_calls: list[dict[str, Any]] = []
        self.get_calls: list[str] = []

    async def get(self, url: str, **kw: Any) -> _FakeResponse:
        self.get_calls.append(url)
        # Real GitHub returns 200 + a body containing sha when the file exists,
        # or 404 when it doesn't. _get_file_sha only inspects status_code and
        # json()["sha"], so we model both.
        sha = self._sha_sequence.pop(0) if self._sha_sequence else None
        if sha is None:
            return _FakeResponse(404)
        return _FakeResponse(200, json_body={"sha": sha})

    async def put(self, url: str, **kw: Any) -> _FakeResponse:
        self.put_calls.append({"url": url, "body": kw.get("json", {})})
        return self._put_sequence.pop(0)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _create_or_update_file
# ---------------------------------------------------------------------------


class TestCreateOrUpdateFileRetry:
    """The SHA-race retry path on PUT 409."""

    def test_happy_path_no_retry_when_first_put_succeeds(self) -> None:
        """200 on the first PUT — no retry, no second SHA fetch."""
        client = _FakeAsyncClient(
            sha_sequence=[None],  # file doesn't exist yet
            put_sequence=[
                _FakeResponse(201, json_body={"content": {"sha": "abc123"}}),
            ],
        )
        result = _run(
            _create_or_update_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "verified/foo.ipynb",
                b'{"cells": []}',
                "msg",
                "tok",
                "main",
            )
        )
        assert result == {"content": {"sha": "abc123"}}
        assert len(client.put_calls) == 1
        assert len(client.get_calls) == 1  # only the pre-PUT SHA check

    def test_retry_once_on_409_then_succeeds(self) -> None:
        """First PUT 409, second SHA refetch returns the winner's SHA,
        second PUT 200."""
        client = _FakeAsyncClient(
            sha_sequence=["stale-sha", "fresh-sha"],
            put_sequence=[
                _FakeResponse(409, text="expected sha doesn't match"),
                _FakeResponse(200, json_body={"content": {"sha": "fresh-sha"}}),
            ],
        )
        result = _run(
            _create_or_update_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "verified/foo.ipynb",
                b'{"cells": []}',
                "msg",
                "tok",
                "main",
            )
        )
        assert result == {"content": {"sha": "fresh-sha"}}
        # Two PUTs (initial + retry), two GETs (initial SHA + refetch).
        assert len(client.put_calls) == 2
        assert len(client.get_calls) == 2
        # The retry must use the freshly-fetched SHA, not the stale one.
        assert client.put_calls[0]["body"]["sha"] == "stale-sha"
        assert client.put_calls[1]["body"]["sha"] == "fresh-sha"

    def test_retry_once_then_409_again_raises(self) -> None:
        """Two consecutive 409s — give up and raise so callers see the error."""
        client = _FakeAsyncClient(
            sha_sequence=["sha1", "sha2"],
            put_sequence=[
                _FakeResponse(409, text="conflict"),
                _FakeResponse(409, text="still conflicting"),
            ],
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            _run(
                _create_or_update_file(
                    client,  # type: ignore[arg-type]
                    "owner/repo",
                    "verified/foo.ipynb",
                    b'{"cells": []}',
                    "msg",
                    "tok",
                    "main",
                )
            )
        assert exc.value.response.status_code == 409
        # Exactly two attempts — no infinite loop, no third try.
        assert len(client.put_calls) == 2

    def test_no_retry_on_non_409_error(self) -> None:
        """A 401 (bad token) must raise immediately without a retry."""
        client = _FakeAsyncClient(
            sha_sequence=[None],
            put_sequence=[
                _FakeResponse(401, text="Bad credentials"),
                # If this is ever consumed the test would still pass on the
                # status code, but len(client.put_calls) catches the bug.
                _FakeResponse(200, json_body={"content": {"sha": "x"}}),
            ],
        )
        with pytest.raises(httpx.HTTPStatusError) as exc:
            _run(
                _create_or_update_file(
                    client,  # type: ignore[arg-type]
                    "owner/repo",
                    "verified/foo.ipynb",
                    b'{"cells": []}',
                    "msg",
                    "tok",
                    "main",
                )
            )
        assert exc.value.response.status_code == 401
        assert len(client.put_calls) == 1, "401 should NOT trigger a retry; only 409 SHA races do"

    def test_retry_creates_new_file_when_winner_deleted_it(self) -> None:
        """Edge case: between attempts, the winning writer deleted the file.
        The retry's SHA refetch returns None — the PUT should proceed without
        a sha in the body and the create succeeds."""
        client = _FakeAsyncClient(
            sha_sequence=["sha-before-delete", None],
            put_sequence=[
                _FakeResponse(409, text="conflict"),
                _FakeResponse(201, json_body={"content": {"sha": "new-sha"}}),
            ],
        )
        result = _run(
            _create_or_update_file(
                client,  # type: ignore[arg-type]
                "owner/repo",
                "verified/foo.ipynb",
                b'{"cells": []}',
                "msg",
                "tok",
                "main",
            )
        )
        assert result == {"content": {"sha": "new-sha"}}
        # Second PUT must NOT carry a stale sha field.
        assert "sha" not in client.put_calls[1]["body"]


# ---------------------------------------------------------------------------
# publish_verified wraps _create_or_update_file — make sure the retry flows
# end-to-end and a final 409 surfaces as GitHubPublishError (not as a
# generic Exception).
# ---------------------------------------------------------------------------


class TestPublishVerifiedSurfacesFinal409:
    def test_final_409_after_retry_raises_github_publish_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If both attempts 409, publish_verified should convert the
        HTTPStatusError into GitHubPublishError (existing contract from
        the write-through fix)."""
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {
                "enabled": True,
                "token": "tok",
                "repo": "owner/repo",
                "branch": "main",
                "drafts_folder": "drafts",
                "verified_folder": "verified",
            },
        )

        async def _double_409(*a: Any, **kw: Any) -> None:
            raise httpx.HTTPStatusError(
                "HTTP 409",
                request=httpx.Request("PUT", "https://api.github.com/x"),
                response=httpx.Response(409, text="conflict"),
            )

        monkeypatch.setattr(github_publisher, "_create_or_update_file", _double_409)

        with pytest.raises(GitHubPublishError) as exc:
            _run(
                publish_verified(
                    submission_id="s",
                    notebook_id="n",
                    query="q",
                    notebook_json={"cells": []},
                )
            )
        assert "409" in str(exc.value)
