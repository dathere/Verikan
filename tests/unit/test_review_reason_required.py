"""Tests for mandatory review reason + commit-message audit trail (#112).

#112 requires that approving or rejecting a notebook OR an answer always
carries a reason, and that the reason plus the reviewing admin's GitHub
identity are recorded in the GitHub commit message (the bot account stays
the committer).

Coverage:

* ``NotebookReviewRequest`` rejects a missing/empty ``admin_notes`` — the
  single shared model that gates all four review endpoints.
* ``_build_commit_message`` composes the ``Reason:`` / ``Reviewed-by:``
  trailer and degrades gracefully when either is omitted (internal sync
  callers).
* ``_reviewer_identity`` prefers user id, then email, then ``admin``.
* ``publish_verified`` / ``publish_verified_answer`` / ``remove_draft``
  actually send the reason + reviewer in the GitHub commit message body.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx
from pydantic import ValidationError

from data_concierge.gateway import github_publisher
from data_concierge.gateway.github_publisher import (
    _build_commit_message,
    publish_verified,
    publish_verified_answer,
    remove_draft,
)
from data_concierge.gateway.router import NotebookReviewRequest, _reviewer_identity

# ---------------------------------------------------------------------------
# NotebookReviewRequest — admin_notes now required
# ---------------------------------------------------------------------------


class TestReviewRequestRequiresReason:
    def test_missing_admin_notes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NotebookReviewRequest(reviewed_by="admin")  # type: ignore[call-arg]

    def test_empty_admin_notes_rejected(self) -> None:
        with pytest.raises(ValidationError):
            NotebookReviewRequest(reviewed_by="admin", admin_notes="")

    def test_valid_reason_accepted(self) -> None:
        req = NotebookReviewRequest(reviewed_by="admin", admin_notes="looks good")
        assert req.admin_notes == "looks good"


# ---------------------------------------------------------------------------
# _build_commit_message — audit trailer
# ---------------------------------------------------------------------------


class TestBuildCommitMessage:
    def test_includes_reason_and_reviewer(self) -> None:
        msg = _build_commit_message("Verify notebook: q", reason="accurate", reviewer="octocat")
        lines = msg.splitlines()
        assert lines[0] == "Verify notebook: q"
        assert "Reason: accurate" in lines
        assert "Reviewed-by: octocat" in lines
        # Blank line separates summary from the trailer.
        assert lines[1] == ""

    def test_summary_only_when_no_reason_or_reviewer(self) -> None:
        assert _build_commit_message("Summary") == "Summary"

    def test_reason_only(self) -> None:
        msg = _build_commit_message("Summary", reason="why")
        assert msg == "Summary\n\nReason: why"

    def test_reviewer_only(self) -> None:
        msg = _build_commit_message("Summary", reviewer="octocat")
        assert msg == "Summary\n\nReviewed-by: octocat"


# ---------------------------------------------------------------------------
# _reviewer_identity — server-trusted reviewer derivation
# ---------------------------------------------------------------------------


class TestReviewerIdentity:
    def test_prefers_user(self) -> None:
        assert _reviewer_identity({"user": "octocat", "email": "o@x.com"}) == "octocat"

    def test_falls_back_to_email(self) -> None:
        assert _reviewer_identity({"user": "", "email": "o@x.com"}) == "o@x.com"

    def test_falls_back_to_admin(self) -> None:
        assert _reviewer_identity({}) == "admin"


# ---------------------------------------------------------------------------
# Publishers embed the reason + reviewer in the commit message
# ---------------------------------------------------------------------------


def _patch_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings",
        lambda: {
            "token": "tok",
            "repo": "owner/repo",
            "branch": "main",
            "drafts_folder": "drafts",
            "verified_folder": "verified",
            "verified_answers_folder": "verified-answers",
            "webhook_secret": "",
        },
    )


def _commit_message_from_put(route: Any) -> str:
    body = json.loads(route.calls[0].request.content)
    return body["message"]


class TestPublishersRecordAudit:
    @respx.mock
    async def test_publish_verified_records_reason_and_reviewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        # Verified create: GET sha (404 -> create) then PUT.
        respx.get(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified/.+"
        ).mock(return_value=httpx.Response(404))
        put_route = respx.put(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified/.+"
        ).mock(return_value=httpx.Response(201, json={"content": {"sha": "s"}}))
        # Draft lookup returns 404 -> no delete attempted (fine for this test).
        respx.get(url__regex=r"https://api\.github\.com/repos/owner/repo/contents/drafts/.+").mock(
            return_value=httpx.Response(404)
        )

        await publish_verified(
            "sub-123",
            "nb-1",
            "unemployment in texas",
            {"cells": []},
            reason="data verified against BLS",
            reviewer="octocat",
        )

        msg = _commit_message_from_put(put_route)
        assert msg.startswith("Verify notebook: unemployment in texas")
        assert "Reason: data verified against BLS" in msg
        assert "Reviewed-by: octocat" in msg

    @respx.mock
    async def test_publish_verified_answer_records_reason_and_reviewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        respx.get(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified-answers/.+"
        ).mock(return_value=httpx.Response(404))
        put_route = respx.put(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified-answers/.+"
        ).mock(return_value=httpx.Response(201, json={"content": {"sha": "s"}}))

        await publish_verified_answer(
            answer_id="ans-1",
            query="texas unemployment",
            answer_json={"answer_id": "ans-1", "answer": "4.1%"},
            reason="matches BLS series",
            reviewer="octocat",
        )

        msg = _commit_message_from_put(put_route)
        assert msg.startswith("Verify answer: texas unemployment")
        assert "Reason: matches BLS series" in msg
        assert "Reviewed-by: octocat" in msg

    @respx.mock
    async def test_remove_draft_records_reason_and_reviewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        respx.get(url__regex=r"https://api\.github\.com/repos/owner/repo/contents/drafts/.+").mock(
            return_value=httpx.Response(200, json={"sha": "draftsha"})
        )
        del_route = respx.delete(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/drafts/.+"
        ).mock(return_value=httpx.Response(200, json={"commit": {"sha": "s"}}))

        await remove_draft(
            "sub-9",
            "rejected query",
            reason="out of scope",
            reviewer="octocat",
        )

        body = json.loads(del_route.calls[0].request.content)
        msg = body["message"]
        assert msg.startswith("Remove rejected draft: rejected query")
        assert "Reason: out of scope" in msg
        assert "Reviewed-by: octocat" in msg
