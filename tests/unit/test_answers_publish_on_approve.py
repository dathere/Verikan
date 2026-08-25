"""Tests for verified-answer publish-on-approve (issue #46, step 9 PR A).

Mirrors test_approve_write_through.py for notebooks:

* ``publish_verified_answer()`` writes ``verified-answers/<id>.json`` to
  the configured repo, returns None when GitHub is disabled, and raises
  ``GitHubPublishError`` on HTTP/network/auth failure.
* The ``/answers/submissions/<id>/approve`` endpoint follows the same
  write-through SSOT order as the notebook approval: publish to GitHub
  FIRST, then commit local. A GitHub publish failure returns 502 and
  leaves the submission PENDING — no "verified locally, missing on
  GitHub" inconsistency.
* The published file IS the provenance: ``verified_at`` and
  ``github_synced_at`` are the same instant, ``usage_count`` is 0 (NOT
  populated from local operational state), and the on-GitHub payload
  matches the local index entry exactly so bootstrap (PR B) can
  round-trip it without divergence.
* GitHub settings GET/POST expose ``verified_answers_folder`` so admins
  can override the default.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from data_concierge.gateway import github_publisher, verified_notebooks
from data_concierge.gateway.github_publisher import (
    GitHubPublishError,
    publish_verified_answer,
)
from data_concierge.gateway.verified_notebooks import (
    ReviewStatus,
    get_answer_submission,
    get_verified_answer,
    submit_quick_answer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    monkeypatch.setattr(github_publisher, "storage", test_storage)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    verified_answers_folder: str = "verified-answers",
) -> None:
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings",
        lambda: {
            "enabled": enabled,
            "token": "tok" if enabled else "",
            "repo": "owner/repo",
            "branch": "main",
            "drafts_folder": "drafts",
            "verified_folder": "verified",
            "verified_answers_folder": verified_answers_folder,
            "webhook_secret": "",
        },
    )


def _seed_submission() -> Any:
    return submit_quick_answer(
        query="What is the unemployment rate in Texas?",
        answer="4.1%",
        source_links=[
            {"name": "BLS", "url": "https://bls.gov/x", "description": "Source"}
        ],
        submitted_by="tester",
        data_source="bls",
        confidence=0.9,
        variable="unemployment_rate",
        place="Texas",
        date="2026-04",
        value="4.1",
    )


# ---------------------------------------------------------------------------
# publish_verified_answer — unit behavior
# ---------------------------------------------------------------------------


class TestPublishVerifiedAnswer:
    async def test_returns_none_when_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, enabled=False)
        result = await publish_verified_answer(
            answer_id="abc",
            query="q",
            answer_json={"answer_id": "abc", "answer": "x"},
        )
        assert result is None

    @respx.mock
    async def test_publishes_to_configured_folder_and_filename(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        # GitHub Contents API: GET → 404 (file doesn't exist), PUT → 201.
        get_route = respx.get(
            "https://api.github.com/repos/owner/repo/contents/verified-answers/abc.json"
        ).mock(return_value=httpx.Response(404))
        put_route = respx.put(
            "https://api.github.com/repos/owner/repo/contents/verified-answers/abc.json"
        ).mock(
            return_value=httpx.Response(
                201,
                json={"content": {"sha": "fakesha", "path": "verified-answers/abc.json"}},
            )
        )

        result = await publish_verified_answer(
            answer_id="abc",
            query="What is the unemployment rate in Texas?",
            answer_json={"answer_id": "abc", "answer": "4.1%"},
        )

        assert result is not None
        assert result["path"] == "verified-answers/abc.json"
        assert result["filename"] == "abc.json"
        assert result["sha"] == "fakesha"
        assert get_route.called
        assert put_route.called
        # The PUT body must carry the full answer JSON, base64-encoded.
        import base64
        request_body = json.loads(put_route.calls[0].request.content)
        decoded = json.loads(base64.b64decode(request_body["content"]))
        assert decoded == {"answer_id": "abc", "answer": "4.1%"}
        # Commit message uses the (truncated) query.
        assert "Verify answer" in request_body["message"]
        assert "Texas" in request_body["message"]

    @respx.mock
    async def test_respects_overridden_answers_folder(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, verified_answers_folder="my/custom/answers")
        respx.get(
            "https://api.github.com/repos/owner/repo/contents/my/custom/answers/abc.json"
        ).mock(return_value=httpx.Response(404))
        put_route = respx.put(
            "https://api.github.com/repos/owner/repo/contents/my/custom/answers/abc.json"
        ).mock(
            return_value=httpx.Response(
                201, json={"content": {"sha": "s"}}
            )
        )

        result = await publish_verified_answer(
            answer_id="abc", query="q", answer_json={"answer": "x"}
        )
        assert result is not None
        assert result["path"] == "my/custom/answers/abc.json"
        assert put_route.called

    @respx.mock
    async def test_raises_on_github_http_error(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        respx.get(
            "https://api.github.com/repos/owner/repo/contents/verified-answers/abc.json"
        ).mock(return_value=httpx.Response(404))
        respx.put(
            "https://api.github.com/repos/owner/repo/contents/verified-answers/abc.json"
        ).mock(return_value=httpx.Response(401, text="Bad credentials"))

        with pytest.raises(GitHubPublishError) as exc:
            await publish_verified_answer(
                answer_id="abc", query="q", answer_json={"answer": "x"}
            )
        assert "401" in str(exc.value)


# ---------------------------------------------------------------------------
# approve_answer_endpoint — write-through SSOT order
# ---------------------------------------------------------------------------


class TestApproveAnswerEndpointWriteThrough:
    async def test_disabled_github_commits_locally_with_no_github_coords(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_answer_endpoint,
        )

        _patch_settings(monkeypatch, enabled=False)
        sub = _seed_submission()

        result = await approve_answer_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
            _admin={"user": "admin"},
        )
        assert result["github_path"] is None
        assert result["github_synced_at"] is None

        verified = get_verified_answer(result["answer_id"])
        assert verified is not None
        assert verified.github_path is None
        assert verified.github_synced_at is None
        # Local-only fields populated as before.
        assert verified.query == sub.query
        assert verified.answer == sub.answer
        assert verified.verified_by == "admin"
        assert verified.usage_count == 0

    @respx.mock
    async def test_enabled_github_publishes_then_commits_local_with_matching_coords(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_answer_endpoint,
        )

        _patch_settings(monkeypatch)
        sub = _seed_submission()

        # Mock GitHub PUT — we don't know the answer_id yet (generated in
        # the endpoint), so match the folder + .json suffix.
        respx.get(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified-answers/.+\.json"
        ).mock(return_value=httpx.Response(404))
        put_route = respx.put(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified-answers/.+\.json"
        ).mock(
            return_value=httpx.Response(
                201, json={"content": {"sha": "newsha"}}
            )
        )

        result = await approve_answer_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
            _admin={"user": "admin"},
        )

        assert put_route.called
        # Endpoint exposes the GitHub coords in its response body so
        # admins / clients can see where the file landed.
        assert result["github_path"] is not None
        assert result["github_path"].startswith("verified-answers/")
        assert result["github_path"].endswith(".json")
        assert result["github_synced_at"] is not None

        verified = get_verified_answer(result["answer_id"])
        assert verified is not None
        # Coherence anchor: github_synced_at == verified_at (same instant).
        assert verified.github_synced_at == verified.verified_at
        assert verified.github_path == result["github_path"]
        # The on-GitHub payload must equal the local entry (modulo
        # github_path/synced_at — those are set AFTER the PUT). Decode
        # the PUT body and compare against the seeded fields.
        import base64
        request_body = json.loads(put_route.calls[0].request.content)
        published = json.loads(base64.b64decode(request_body["content"]))
        assert published["answer_id"] == verified.answer_id
        assert published["submission_id"] == sub.submission_id
        assert published["query"] == sub.query
        assert published["answer"] == sub.answer
        assert published["confidence"] == sub.confidence
        assert published["data_source"] == sub.data_source
        # The published payload's verified_at equals the local one
        # (the coherence anchor again).
        assert published["verified_at"] == verified.verified_at
        # usage_count is local-only operational state — must NOT be
        # baked into the GitHub file (every increment would otherwise
        # dirty the repo history).
        assert published["usage_count"] == 0

    @respx.mock
    async def test_github_failure_returns_502_and_leaves_submission_pending(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point of write-through: a GitHub failure must NOT
        leave a 'verified locally, missing on GitHub' inconsistency.
        Endpoint returns 502; submission stays PENDING; no
        VerifiedAnswer entry exists."""
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_answer_endpoint,
        )

        _patch_settings(monkeypatch)
        sub = _seed_submission()

        respx.get(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified-answers/.+\.json"
        ).mock(return_value=httpx.Response(404))
        respx.put(
            url__regex=r"https://api\.github\.com/repos/owner/repo/contents/verified-answers/.+\.json"
        ).mock(return_value=httpx.Response(500, text="Internal Error"))

        with pytest.raises(HTTPException) as exc:
            await approve_answer_endpoint(
                submission_id=sub.submission_id,
                request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 502

        # Submission unchanged.
        still = get_answer_submission(sub.submission_id)
        assert still is not None
        assert still.status == ReviewStatus.PENDING

        # No verified-answer entry exists for that submission.
        from data_concierge.gateway.verified_notebooks import (
            get_verified_answers,
        )
        verifieds = get_verified_answers()
        assert not any(v.submission_id == sub.submission_id for v in verifieds)

    async def test_already_reviewed_submission_returns_404(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_answer_endpoint,
        )

        _patch_settings(monkeypatch, enabled=False)
        sub = _seed_submission()

        # First approval succeeds.
        await approve_answer_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
            _admin={"user": "admin"},
        )
        # Second approval on same submission → 404 (status is now APPROVED).
        with pytest.raises(HTTPException) as exc:
            await approve_answer_endpoint(
                submission_id=sub.submission_id,
                request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 404

    def test_approve_route_is_actually_registered_with_fastapi(self) -> None:
        """Regression: an earlier adversarial review caught a missing @router.post decorator
        on approve_answer_endpoint — direct unit calls still passed because
        the function exists, but the route wasn't registered with FastAPI
        and real HTTP clients got 404/405. Assert the route is wired so
        that failure mode can never recur silently.
        """
        from data_concierge.gateway.router import router

        post_paths = {
            r.path for r in router.routes
            if getattr(r, "methods", None) and "POST" in r.methods
        }
        assert (
            "/api/v1/answers/submissions/{submission_id}/approve" in post_paths
        ), (
            "POST /api/v1/answers/submissions/{submission_id}/approve is not "
            "registered with the router — check that "
            "@router.post(...) decorator is on approve_answer_endpoint."
        )


# ---------------------------------------------------------------------------
# GitHub settings expose verified_answers_folder
# ---------------------------------------------------------------------------


class TestSettingsExposeAnswersFolder:
    async def test_get_returns_default_folder(self, tmp_storage: None) -> None:
        from data_concierge.gateway.router import get_github_settings

        body = await get_github_settings(_admin={"user": "admin"})
        assert body["verified_answers_folder"] == "verified-answers"

    async def test_post_updates_folder(self, tmp_storage: None) -> None:
        from data_concierge.gateway.router import (
            GitHubSettingsRequest,
            get_github_settings,
            update_github_settings,
        )

        await update_github_settings(
            request=GitHubSettingsRequest(
                enabled=True,
                repo="owner/repo",
                verified_answers_folder="my/answers",
            ),
            _admin={"user": "admin"},
        )
        body = await get_github_settings(_admin={"user": "admin"})
        assert body["verified_answers_folder"] == "my/answers"
