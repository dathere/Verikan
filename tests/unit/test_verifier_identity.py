"""Tests for capturing the real verifier identity and exposing provenance.

These cover three coupled issues:

* **#73** — the verifier/approver/rejecter recorded on a notebook or answer
  must be the authenticated admin from the session, NOT the client-supplied
  ``reviewed_by`` field (the admin UI hardcodes that to "admin"). The endpoints
  derive the identity server-side via ``_reviewer_identity(_admin)``.
* **#71 / #72** — ``list_verified_notebooks`` must expose ``submission_id`` (so
  the admin UI can fetch a verified notebook's retained agent logs) and
  ``admin_notes`` (the reviewer note shown in the in-app viewer).
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.gateway import verified_notebooks
from data_concierge.gateway.verified_notebooks import (
    ReviewStatus,
    get_submission,
    get_verified_notebook,
    submit_notebook,
)


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


async def _disabled_publish(*a: Any, **kw: Any) -> None:
    """Simulate GitHub disabled so approval stays local-only."""
    return None


# ---------------------------------------------------------------------------
# #73 — notebook approve/reject records the authenticated admin identity
# ---------------------------------------------------------------------------


class TestNotebookVerifierIdentity:
    async def test_approve_records_authenticated_admin_not_request(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _disabled_publish,
        )

        sub = _make_submission()
        result = await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            # The UI sends reviewed_by="admin" — it must be ignored in favour
            # of the authenticated session identity.
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="verified vs BLS"),
            _admin={"user": "octocat", "email": "octocat@example.com"},
        )

        verified = get_verified_notebook(result["notebook_id"])
        assert verified is not None
        assert verified.verified_by == "octocat"
        assert verified.admin_notes == "verified vs BLS"

    async def test_reject_records_authenticated_admin_not_request(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            reject_notebook_endpoint,
        )

        async def _noop_remove(*a: Any, **kw: Any) -> None:
            return None

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.remove_draft",
            _noop_remove,
        )

        sub = _make_submission()
        await reject_notebook_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="out of scope"),
            _admin={"user": "octocat", "email": "octocat@example.com"},
        )

        peek = get_submission(sub.submission_id)
        assert peek is not None
        assert peek.status == ReviewStatus.REJECTED
        assert peek.reviewed_by == "octocat"

    async def test_falls_back_to_email_then_admin(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _disabled_publish,
        )

        sub = _make_submission()
        result = await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="ok"),
            # No user id — should fall back to email.
            _admin={"user": "", "email": "reviewer@example.com"},
        )
        verified = get_verified_notebook(result["notebook_id"])
        assert verified is not None
        assert verified.verified_by == "reviewer@example.com"


# ---------------------------------------------------------------------------
# #71 / #72 — list serialization exposes submission_id + admin_notes
# ---------------------------------------------------------------------------


class TestListVerifiedNotebooksSerialization:
    async def test_exposes_submission_id_and_admin_notes(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
            list_verified_notebooks,
        )

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _disabled_publish,
        )

        sub = _make_submission()
        await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="looks good"),
            _admin={"user": "octocat"},
        )

        listing = await list_verified_notebooks()
        assert listing["count"] == 1
        row = listing["notebooks"][0]
        # submission_id lets the UI fetch this notebook's retained agent logs.
        assert row["submission_id"] == sub.submission_id
        assert row["admin_notes"] == "looks good"
        assert row["verified_by"] == "octocat"
