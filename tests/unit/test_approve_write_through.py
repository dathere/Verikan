"""Tests for write-through approval ordering.

Covers the fix for the bug where a notebook could be locally "verified" with
no record on GitHub if the GitHub publish failed after the local commit.

See issue #46 follow-ups, item B in
an earlier production incident.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from data_concierge.gateway import github_publisher, verified_notebooks
from data_concierge.gateway.github_publisher import (
    GitHubPublishError,
    publish_verified,
)
from data_concierge.gateway.verified_notebooks import (
    ReviewStatus,
    approve_notebook,
    get_submission,
    get_verified_notebook,
    submit_notebook,
)

# ---------------------------------------------------------------------------
# Storage fixture: redirect verified_notebooks.storage at a tmp dir so each
# test gets an isolated index. The module-level singleton is replaced for the
# duration of the test.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)


def _make_submission() -> Any:
    return submit_notebook(
        query="What is the unemployment rate in Texas?",
        answer="4.1%",
        notebook_json={"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
        submitted_by="tester",
        data_source="bls",
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# approve_notebook() — github metadata persistence
# ---------------------------------------------------------------------------


class TestApproveNotebookPersistsGithubMetadata:
    """approve_notebook() must atomically persist GitHub provenance fields."""

    def test_persists_github_path_and_synced_at(self, tmp_storage: None) -> None:
        sub = _make_submission()
        verified = approve_notebook(
            submission_id=sub.submission_id,
            reviewed_by="admin",
            admin_notes="looks good",
            github_path="verified/some-notebook_abc12345.ipynb",
            github_synced_at="2026-05-27T10:00:00Z",
        )
        assert verified is not None
        assert verified.github_path == "verified/some-notebook_abc12345.ipynb"
        assert verified.github_synced_at == "2026-05-27T10:00:00Z"

        # Persisted to the index, not just the in-memory object
        fetched = get_verified_notebook(verified.notebook_id)
        assert fetched is not None
        assert fetched.github_path == "verified/some-notebook_abc12345.ipynb"
        assert fetched.github_synced_at == "2026-05-27T10:00:00Z"

    def test_defaults_to_none_when_github_metadata_omitted(self, tmp_storage: None) -> None:
        """Backwards compatible: callers that don't pass GitHub metadata still work."""
        sub = _make_submission()
        verified = approve_notebook(submission_id=sub.submission_id)
        assert verified is not None
        assert verified.github_path is None
        assert verified.github_synced_at is None

    def test_returns_none_for_missing_submission(self, tmp_storage: None) -> None:
        assert approve_notebook(submission_id="does-not-exist") is None

    def test_returns_none_when_already_reviewed(self, tmp_storage: None) -> None:
        sub = _make_submission()
        first = approve_notebook(submission_id=sub.submission_id)
        assert first is not None
        # Second approve should be a no-op and return None — this is the race
        # signal the endpoint relies on to detect TOCTOU between peek and commit.
        second = approve_notebook(submission_id=sub.submission_id)
        assert second is None


# ---------------------------------------------------------------------------
# publish_verified() — disabled vs. failure are distinct signals
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeRequest:
    def __init__(self, url: str = "https://api.github.com/x") -> None:
        self.url = url


class TestPublishVerifiedSignalling:
    """publish_verified() must distinguish "disabled" from "failed"."""

    def test_returns_none_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings",
            lambda: {"enabled": False, "token": ""},
        )

        import asyncio

        result = asyncio.run(
            publish_verified(
                submission_id="sub-1",
                notebook_id="nb-1",
                query="q",
                notebook_json={"cells": []},
            )
        )
        assert result is None

    def test_raises_github_publish_error_on_http_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        async def _boom(*a: Any, **kw: Any) -> None:
            raise httpx.HTTPStatusError(
                "GitHub rate limited",
                request=_FakeRequest(),  # type: ignore[arg-type]
                response=_FakeResponse(429, "rate limit exceeded"),  # type: ignore[arg-type]
            )

        monkeypatch.setattr(github_publisher, "_create_or_update_file", _boom)

        import asyncio

        with pytest.raises(GitHubPublishError) as exc:
            asyncio.run(
                publish_verified(
                    submission_id="sub-1",
                    notebook_id="nb-1",
                    query="q",
                    notebook_json={"cells": []},
                )
            )
        assert "429" in str(exc.value)

    def test_raises_github_publish_error_on_network_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        async def _network_err(*a: Any, **kw: Any) -> None:
            raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(github_publisher, "_create_or_update_file", _network_err)

        import asyncio

        with pytest.raises(GitHubPublishError) as exc:
            asyncio.run(
                publish_verified(
                    submission_id="sub-1",
                    notebook_id="nb-1",
                    query="q",
                    notebook_json={"cells": []},
                )
            )
        assert "connection refused" in str(exc.value)


# ---------------------------------------------------------------------------
# End-to-end: the endpoint must NOT commit local state when GitHub fails.
# We exercise the endpoint function directly (bypassing FastAPI deps).
# ---------------------------------------------------------------------------


class TestApproveEndpointWriteThrough:
    """The approve endpoint is the place the bug actually lived.

    These tests assert the new write-through ordering: if GitHub publishing is
    enabled and fails, no VerifiedNotebook is created and the submission stays
    PENDING. If GitHub is disabled, local approval still proceeds.
    """

    async def test_github_failure_does_not_commit_local_state(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway import router
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        sub = _make_submission()

        async def _failing_publish(*a: Any, **kw: Any) -> None:
            raise GitHubPublishError("simulated GitHub outage")

        # Patch the symbol imported lazily inside the endpoint
        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _failing_publish,
        )

        with pytest.raises(HTTPException) as exc:
            await approve_notebook_endpoint(
                submission_id=sub.submission_id,
                request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 502
        assert "GitHub publish failed" in str(exc.value.detail)

        # Submission must still be PENDING
        peek = get_submission(sub.submission_id)
        assert peek is not None
        assert peek.status == ReviewStatus.PENDING

        # No VerifiedNotebook should exist for this submission
        from data_concierge.gateway.verified_notebooks import get_verified_notebooks

        verified_for_sub = [
            v for v in get_verified_notebooks() if v.submission_id == sub.submission_id
        ]
        assert verified_for_sub == [], (
            "Local VerifiedNotebook was created despite GitHub failure — "
            "write-through ordering is broken"
        )

        # Sanity: the router import in this test file is the FastAPI router
        # instance, just used here to ensure the import path resolves.
        assert router is not None

    async def test_github_success_commits_with_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        sub = _make_submission()

        async def _ok_publish(*a: Any, **kw: Any) -> dict[str, Any]:
            return {
                "path": "verified/q_abc12345.ipynb",
                "filename": "q_abc12345.ipynb",
                "sha": "deadbeef",
            }

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _ok_publish,
        )

        result = await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
            _admin={"user": "admin"},
        )
        assert result["message"] == "Notebook approved and verified"
        notebook_id = result["notebook_id"]

        v = get_verified_notebook(notebook_id)
        assert v is not None
        assert v.github_path == "verified/q_abc12345.ipynb"
        assert v.github_synced_at is not None
        assert v.github_synced_at.endswith("Z")

        peek = get_submission(sub.submission_id)
        assert peek is not None
        assert peek.status == ReviewStatus.APPROVED

    async def test_github_disabled_still_approves_locally(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GitHub is disabled, publish_verified() returns None and the
        endpoint should treat that as success — local-only mode still works."""
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        sub = _make_submission()

        async def _disabled_publish(*a: Any, **kw: Any) -> None:
            return None

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _disabled_publish,
        )

        result = await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
            _admin={"user": "admin"},
        )
        notebook_id = result["notebook_id"]
        v = get_verified_notebook(notebook_id)
        assert v is not None
        assert v.github_path is None
        assert v.github_synced_at is None

    async def test_missing_submission_returns_404(self, tmp_storage: None) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        with pytest.raises(HTTPException) as exc:
            await approve_notebook_endpoint(
                submission_id="nope",
                request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 404

    async def test_race_after_github_success_returns_409(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """TOCTOU: publish_verified succeeded, then approve_notebook returns None.

        Simulates the race where the submission was approved or rejected by
        another request between the endpoint's peek and its commit. The endpoint
        must return 409 (not 200) so the caller knows the local commit didn't
        happen, and must log the GitHub path so an admin can reconcile the now-
        orphaned verified-folder file.
        """
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        sub = _make_submission()

        async def _ok_publish(*a: Any, **kw: Any) -> dict[str, Any]:
            return {
                "path": "verified/race_aaaaaaaa.ipynb",
                "filename": "race_aaaaaaaa.ipynb",
                "sha": "deadbeef",
            }

        # publish_verified is imported lazily inside the endpoint
        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _ok_publish,
        )

        # approve_notebook is imported at module load into the router module.
        # `from data_concierge.gateway import router` returns the APIRouter
        # global with the same name (it shadows the submodule attribute), so
        # grab the real module via sys.modules.
        import sys

        router_mod = sys.modules["data_concierge.gateway.router"]

        def _race_loses(**kw: Any) -> None:
            return None

        monkeypatch.setattr(router_mod, "approve_notebook", _race_loses)

        with pytest.raises(HTTPException) as exc:
            await approve_notebook_endpoint(
                submission_id=sub.submission_id,
                request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
                _admin={"user": "admin"},
            )

        assert exc.value.status_code == 409
        assert "modified by another request" in str(exc.value.detail)

        # The orphan warning must reference the GitHub path so an admin can
        # reconcile the now-orphaned verified-folder file. structlog routes
        # through the root logger to stderr by default in this project, so we
        # capture both streams and look for the message + path.
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "orphan" in combined.lower(), (
            "Expected an 'orphan' warning on TOCTOU 409 so admins can reconcile"
        )
        assert "verified/race_aaaaaaaa.ipynb" in combined, (
            "Orphan warning must include the GitHub path"
        )
