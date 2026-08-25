"""End-to-end smoke checks for the approve-notebook endpoint.

Stands in for the manual smoke items on the write-through PR:

* Approve with GitHub enabled + a valid token -> 200, github_path /
  github_synced_at populated, submission marked APPROVED.
* Approve with GitHub enabled + a bad token -> 502, submission stays
  PENDING, no VerifiedNotebook is created.

We exercise the real FastAPI router via TestClient and intercept httpx
calls at the transport layer with respx, so the entire stack (route
matching, request validation, dependency injection, settings storage,
publish_verified, _create_or_update_file, _delete_file) runs exactly
as in production -- only the GitHub HTTP responses are faked.
"""

from __future__ import annotations

import re
from typing import Any

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from data_concierge.data_layer.storage import LocalStorage
from data_concierge.gateway import github_publisher, verified_notebooks
from data_concierge.gateway.router import require_admin, router
from data_concierge.gateway.verified_notebooks import (
    ReviewStatus,
    get_submission,
    get_verified_notebooks,
    submit_notebook,
)

_REPO = "owner/test-repo"
_TOKEN = "fake-pat"
_CONTENTS_RE = re.compile(rf"^https://api\.github\.com/repos/{re.escape(_REPO)}/contents/.+$")


@pytest.fixture
def app_client(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient bound to a fresh FastAPI app with isolated storage."""
    test_storage = LocalStorage(tmp_path)
    # Both modules import the storage singleton at module level; both must be
    # patched so the test sees a single isolated tmp filesystem.
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    monkeypatch.setattr(github_publisher, "storage", test_storage)

    # Seed GitHub settings so load_github_settings() returns enabled=True.
    test_storage.write_json(
        "github_settings.json",
        {
            "enabled": True,
            "token": _TOKEN,
            "repo": _REPO,
            "branch": "main",
            "drafts_folder": "drafts",
            "verified_folder": "verified",
        },
    )

    app = FastAPI()
    app.include_router(router)
    # Bypass the real auth chain; the endpoint only uses _admin for the
    # dependency, not in its body, so any dict will do.
    app.dependency_overrides[require_admin] = lambda: {"user": "admin"}
    return TestClient(app)


def _submit() -> str:
    sub = submit_notebook(
        query="Smoke: unemployment rate in Texas?",
        answer="4.1%",
        notebook_json={
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        submitted_by="smoke-tester",
        data_source="bls",
        confidence=0.9,
    )
    return sub.submission_id


@respx.mock
def test_smoke_github_success_populates_metadata(app_client: TestClient) -> None:
    """SMOKE 1: GitHub enabled + valid token, full publish flow exercised.

    Asserts the endpoint returns 200, the VerifiedNotebook ends up in the
    local index with both github_path and github_synced_at populated, the
    submission is marked APPROVED, and the publish flow's draft-cleanup
    step is actually exercised (no longer short-circuiting on a 404).

    publish_verified() makes four GitHub calls we mock here:
      1. GET verified/<path>  (SHA check before create) -> 404, new file
      2. PUT verified/<path>  (create)                  -> 201
      3. GET drafts/<path>    (SHA check inside _delete_file) -> 200 + sha
      4. DELETE drafts/<path> (remove the published draft)    -> 200

    A side-effect router on GET branches by URL so we can return 404 on the
    verified path (the file doesn't exist yet) and 200 on the drafts path
    (the draft was published at submission time).
    """
    submission_id = _submit()

    def _get_handler(request: httpx.Request) -> Response:
        path = request.url.path
        if "/verified/" in path:
            return Response(404)
        if "/drafts/" in path:
            return Response(200, json={"sha": "draft-sha"})
        raise AssertionError(f"unexpected GET to {path}")

    get_route = respx.get(url__regex=_CONTENTS_RE).mock(side_effect=_get_handler)
    put_route = respx.put(url__regex=_CONTENTS_RE).mock(
        return_value=Response(
            201,
            json={"content": {"sha": "deadbeef", "path": "verified/dummy.ipynb"}},
        )
    )
    delete_route = respx.delete(url__regex=_CONTENTS_RE).mock(
        return_value=Response(200)
    )

    resp = app_client.post(
        f"/api/v1/notebooks/submissions/{submission_id}/approve",
        json={"reviewed_by": "smoke-admin", "admin_notes": "ok"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"] == "Notebook approved and verified"
    assert body["github_verified"]["path"].startswith("verified/")
    # Step 3: the publisher now reports whether the draft cleanup happened.
    assert body["github_verified"]["draft_cleanup_pending"] is False

    # Verified entry exists with GitHub metadata committed atomically.
    verified = [v for v in get_verified_notebooks() if v.submission_id == submission_id]
    assert len(verified) == 1
    v = verified[0]
    assert v.github_path is not None
    assert v.github_path.startswith("verified/")
    assert v.github_synced_at is not None
    assert v.github_synced_at.endswith("Z")

    # Submission is now APPROVED.
    sub = get_submission(submission_id)
    assert sub is not None
    assert sub.status == ReviewStatus.APPROVED

    # Inspect recorded URLs explicitly so a regression that skipped the
    # verified-path SHA check (or the DELETE) specifically fails the smoke.
    verified_gets = [c for c in get_route.calls if "/verified/" in str(c.request.url)]
    drafts_gets = [c for c in get_route.calls if "/drafts/" in str(c.request.url)]
    verified_puts = [c for c in put_route.calls if "/verified/" in str(c.request.url)]
    drafts_deletes = [
        c for c in delete_route.calls if "/drafts/" in str(c.request.url)
    ]
    assert verified_gets, "Pre-PUT SHA check on verified path was never called"
    assert verified_puts, "PUT to verified path was never called"
    assert drafts_gets, "Pre-DELETE SHA check on drafts path was never called"
    assert drafts_deletes, (
        "DELETE on drafts path was never called — _delete_file may have "
        "regressed to short-circuit"
    )


@respx.mock
def test_smoke_github_bad_token_returns_502_and_leaves_pending(
    app_client: TestClient,
) -> None:
    """SMOKE 2: GitHub enabled + bad token.

    Asserts the endpoint returns 502 with a clear error message, the
    submission stays PENDING, and no VerifiedNotebook is created. This is
    the case the original bug got wrong -- previously the local commit
    would succeed and the GitHub-side failure would be swallowed.
    """
    submission_id = _submit()

    # SHA check returns 404 (no existing file).
    respx.get(url__regex=_CONTENTS_RE).mock(return_value=Response(404))
    # GitHub rejects the PUT with 401 Bad credentials -- what a revoked or
    # mistyped PAT actually looks like in the wild.
    respx.put(url__regex=_CONTENTS_RE).mock(
        return_value=Response(401, json={"message": "Bad credentials"})
    )

    resp = app_client.post(
        f"/api/v1/notebooks/submissions/{submission_id}/approve",
        json={"reviewed_by": "smoke-admin", "admin_notes": "verified against source"},
    )
    assert resp.status_code == 502, resp.text
    assert "GitHub publish failed" in resp.json()["detail"]
    assert "401" in resp.json()["detail"]

    # Submission must still be PENDING.
    sub = get_submission(submission_id)
    assert sub is not None
    assert sub.status == ReviewStatus.PENDING, (
        f"Submission should stay PENDING on GitHub failure, got {sub.status}"
    )

    # No VerifiedNotebook should exist.
    verified = [v for v in get_verified_notebooks() if v.submission_id == submission_id]
    assert verified == [], (
        "Local VerifiedNotebook was created despite GitHub failure -- "
        "write-through ordering is broken"
    )


# ---------------------------------------------------------------------------
# Step 3: reconcile-drafts endpoint
# ---------------------------------------------------------------------------


@respx.mock
def test_smoke_reconcile_drafts_deletes_duplicates(app_client: TestClient) -> None:
    """End-to-end smoke for POST /api/v1/notebooks/reconcile-drafts.

    Two drafts; one is also in verified. The reconcile endpoint should
    list both folders, find the duplicate, and DELETE the draft. Returns
    a report with the cleaned path.
    """
    contents_base = f"https://api.github.com/repos/{_REPO}/contents"
    drafts_folder_url = re.compile(
        rf"^{re.escape(contents_base)}/drafts(\?|$)"
    )
    verified_folder_url = re.compile(
        rf"^{re.escape(contents_base)}/verified(\?|$)"
    )
    # Per-file SHA check + DELETE — match anything under /contents/drafts/<file>.
    draft_file_re = re.compile(
        rf"^{re.escape(contents_base)}/drafts/.+$"
    )

    respx.get(url__regex=drafts_folder_url).mock(
        return_value=Response(
            200,
            json=[
                {
                    "type": "file",
                    "name": "leaked.ipynb",
                    "path": "drafts/leaked.ipynb",
                    "sha": "d-leaked",
                },
                {
                    "type": "file",
                    "name": "pending.ipynb",
                    "path": "drafts/pending.ipynb",
                    "sha": "d-pending",
                },
            ],
        )
    )
    respx.get(url__regex=verified_folder_url).mock(
        return_value=Response(
            200,
            json=[
                {
                    "type": "file",
                    "name": "leaked.ipynb",
                    "path": "verified/leaked.ipynb",
                    "sha": "v-leaked",
                },
            ],
        )
    )
    # Per-file SHA check inside _delete_file returns the existing draft SHA.
    respx.get(url__regex=draft_file_re).mock(
        return_value=Response(200, json={"sha": "d-leaked"})
    )
    delete_route = respx.delete(url__regex=draft_file_re).mock(
        return_value=Response(200)
    )

    resp = app_client.post("/api/v1/notebooks/reconcile-drafts")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["checked"] == 2
    assert body["duplicates_found"] == 1
    assert body["cleaned"] == ["drafts/leaked.ipynb"]
    assert body["failed"] == []
    assert body["disabled"] is False

    # Confirm we DELETE'd specifically the leaked draft, not the pending one.
    deleted_urls = [str(c.request.url) for c in delete_route.calls]
    assert any("drafts/leaked.ipynb" in u for u in deleted_urls)
    assert not any("drafts/pending.ipynb" in u for u in deleted_urls)


@respx.mock
def test_smoke_reconcile_drafts_returns_400_when_disabled(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If GitHub publishing is off, the endpoint returns 400 with a clear
    message rather than silently doing nothing."""
    # Override the seeded settings to disabled.
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings",
        lambda: {"enabled": False, "token": ""},
    )
    resp = app_client.post("/api/v1/notebooks/reconcile-drafts")
    assert resp.status_code == 400, resp.text
    assert "GitHub publishing is not enabled" in resp.json()["detail"]


@respx.mock
def test_smoke_reconcile_drafts_returns_502_when_listing_fails(
    app_client: TestClient,
) -> None:
    """Copilot PR #78 follow-up: if reconcile can't list GitHub (401 / 5xx),
    the endpoint must return 502 with an actionable error — not a silent
    `checked: 0` no-op."""
    contents_base = f"https://api.github.com/repos/{_REPO}/contents"
    folder_url = re.compile(rf"^{re.escape(contents_base)}/drafts(\?|$)")
    # 401 on the drafts listing — strict mode raises GitHubPublishError
    # before reconcile even looks at verified.
    respx.get(url__regex=folder_url).mock(
        return_value=Response(401, json={"message": "Bad credentials"})
    )

    resp = app_client.post("/api/v1/notebooks/reconcile-drafts")
    assert resp.status_code == 502, resp.text
    assert "Could not reach GitHub" in resp.json()["detail"]
    assert "401" in resp.json()["detail"]
