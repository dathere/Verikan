"""Tests for ``pull_request`` event handling (issue #74).

Companion to ``test_github_webhook.py`` (which covers the ``push`` flow
from issue #46 step 7). Coverage:

* Action filter: ``opened`` / ``reopened`` / ``closed`` recorded;
  ``synchronize`` / ``edited`` / etc. ignored but marked seen.
* Outcome split on ``closed``: ``merged`` vs ``closed_unmerged``.
* Base-branch filter: PR targeting a non-configured base is ignored.
* Audit log: persisted entries grow, cap holds, redelivery is no-op.
* Pause guard: returns 200 ``skipped_paused`` without writing audit.
* Signature, missing pull_request object, JSON parse failures.
* Push annotation: PR # parsed from default merge + squash-merge
  commit subjects; rebase merges surface ``pr_number=None``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from data_concierge.gateway import (
    admin_notifications,
    github_publisher,
    github_webhook,
    verified_notebooks,
)
from data_concierge.gateway.github_webhook import github_webhook as webhook_handler

WEBHOOK_SECRET = "shared-secret-for-tests"


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    monkeypatch.setattr(github_publisher, "storage", test_storage)
    monkeypatch.setattr(github_webhook, "storage", test_storage)


@pytest.fixture
def captured_notifications(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Capture notify_pr_event calls instead of actually sending mail."""
    calls: list[dict[str, Any]] = []

    async def _fake(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(admin_notifications, "notify_pr_event", _fake)
    return calls


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    secret: str = WEBHOOK_SECRET,
    branch: str = "main",
    settings_status: str = "ok",
) -> None:
    fake_settings = {
        "paused": not enabled,
        "token": "tok",
        "repo": "owner/repo",
        "branch": branch,
        "drafts_folder": "drafts",
        "verified_folder": "verified",
        "verified_answers_folder": "verified-answers",
        "webhook_secret": secret,
    }
    monkeypatch.setattr(github_publisher, "load_github_settings", lambda: fake_settings)
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings_with_status",
        lambda: (fake_settings, settings_status),
    )


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _make_request(body: bytes, headers: dict[str, str]) -> Request:
    raw_headers = [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/github/webhook",
        "headers": raw_headers,
        "query_string": b"",
    }

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=_receive)


def _pr_payload(
    *,
    action: str,
    number: int = 42,
    title: str = "Add a verified notebook",
    base_ref: str = "main",
    head_ref: str = "feature/foo",
    merged: bool = False,
    author: str = "octocat",
    html_url: str = "https://github.com/owner/repo/pull/42",
) -> dict[str, Any]:
    return {
        "action": action,
        "pull_request": {
            "number": number,
            "title": title,
            "html_url": html_url,
            "merged": merged,
            "user": {"login": author},
            "base": {"ref": base_ref},
            "head": {"ref": head_ref},
        },
        "repository": {
            "full_name": "owner/repo",
            "html_url": "https://github.com/owner/repo",
        },
    }


def _push_payload(
    *,
    head_message: str | None = None,
    repo_url: str = "https://github.com/owner/repo",
) -> dict[str, Any]:
    head_commit: dict[str, Any] | None = None
    if head_message is not None:
        head_commit = {"message": head_message}
    return {
        "ref": "refs/heads/main",
        "before": "0" * 40,
        "after": "1" * 40,
        "commits": [],
        "head_commit": head_commit,
        "repository": {"full_name": "owner/repo", "html_url": repo_url},
    }


class TestPullRequestAuditAndNotify:
    async def test_opened_pr_is_audited_and_notified(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="opened")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-opened",
        )
        assert result["status"] == "processed"
        assert result["event"] == "pull_request"
        assert result["action"] == "opened"
        assert result["outcome"] == "opened"
        assert result["pr_number"] == 42
        assert result["html_url"] == "https://github.com/owner/repo/pull/42"
        assert result["diff_url"] == "https://github.com/owner/repo/pull/42/files"
        # Audit log entry persisted.
        data = github_webhook.storage.read_json(github_webhook._PR_AUDIT_KEY)
        assert isinstance(data, dict)
        entries = data["entries"]
        assert len(entries) == 1
        assert entries[0]["pr_number"] == 42
        assert entries[0]["outcome"] == "opened"
        # Notification dispatched with diff URL.
        assert len(captured_notifications) == 1
        assert captured_notifications[0]["pr_number"] == 42
        assert (
            captured_notifications[0]["diff_url"] == "https://github.com/owner/repo/pull/42/files"
        )

    async def test_closed_merged_splits_outcome_to_merged(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="closed", merged=True)).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-merged",
        )
        assert result["outcome"] == "merged"
        assert captured_notifications[0]["outcome"] == "merged"

    async def test_closed_unmerged_splits_outcome_to_closed_unmerged(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="closed", merged=False)).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-closed",
        )
        assert result["outcome"] == "closed_unmerged"
        assert captured_notifications[0]["outcome"] == "closed_unmerged"

    async def test_reopened_pr_is_recorded(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="reopened")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-reopened",
        )
        assert result["outcome"] == "reopened"
        assert len(captured_notifications) == 1


class TestPullRequestFilters:
    async def test_uninteresting_action_is_ignored_and_marked_seen(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="synchronize")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-sync",
        )
        assert result["status"] == "ignored"
        assert result["reason"] == "uninteresting action"
        assert result["action"] == "synchronize"
        # No audit, no notification.
        assert captured_notifications == []
        # Marked seen so a redelivery is a fast no-op.
        assert github_webhook._is_delivery_seen("d-pr-sync")

    async def test_pr_against_non_target_base_is_ignored(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch, branch="main")
        body = json.dumps(_pr_payload(action="opened", base_ref="dev")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-wrong-base",
        )
        assert result["status"] == "ignored"
        assert result["reason"] == "base ref mismatch"
        assert result["base_ref"] == "dev"
        assert captured_notifications == []
        assert github_webhook._is_delivery_seen("d-pr-wrong-base")

    async def test_missing_pull_request_object_is_ignored(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps({"action": "opened"}).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-malformed",
        )
        assert result["status"] == "ignored"
        assert result["reason"] == "missing pull_request object"
        assert captured_notifications == []


class TestPullRequestDeduplication:
    async def test_duplicate_pr_delivery_is_skipped(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="opened")).encode("utf-8")
        # First delivery.
        result1 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-dup",
        )
        assert result1["status"] == "processed"
        # Second delivery with same ID.
        result2 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-dup",
        )
        assert result2["status"] == "duplicate_delivery"
        # Only one audit entry, only one notification.
        data = github_webhook.storage.read_json(github_webhook._PR_AUDIT_KEY)
        assert len(data["entries"]) == 1
        assert len(captured_notifications) == 1


class TestPullRequestPauseGuard:
    async def test_pause_returns_skipped_paused_without_audit(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch, enabled=False)
        body = json.dumps(_pr_payload(action="opened")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="pull_request",
            x_github_delivery="d-pr-paused",
        )
        assert result["status"] == "skipped_paused"
        # No audit, no notification while paused.
        try:
            data = github_webhook.storage.read_json(github_webhook._PR_AUDIT_KEY)
        except Exception:
            data = None
        assert not data or not data.get("entries")
        assert captured_notifications == []
        # NOT marked seen — admin pause is steady state; if they resume
        # before GitHub gives up, redelivery should re-evaluate.
        assert not github_webhook._is_delivery_seen("d-pr-paused")

    async def test_fail_closed_settings_raise_503(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
        captured_notifications: list[dict[str, Any]],
    ) -> None:
        _patch_settings(monkeypatch, enabled=False, settings_status="fail_closed")
        body = json.dumps(_pr_payload(action="opened")).encode("utf-8")
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="pull_request",
                x_github_delivery="d-pr-failclosed",
            )
        assert exc.value.status_code == 503


class TestPullRequestSignature:
    async def test_invalid_signature_returns_401(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_pr_payload(action="opened")).encode("utf-8")
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256="sha256=" + "0" * 64,
                x_github_event="pull_request",
                x_github_delivery="d-pr-badsig",
            )
        assert exc.value.status_code == 401


class TestPullRequestBodyParsing:
    """Pin the PR branch's body-validation paths independently of the
    push handler. The two share dedup + pause-guard scaffolding but
    each has its own JSON parse + non-dict error text, so push-side
    tests don't transitively cover them.
    """

    async def test_invalid_json_body_returns_400(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_settings(monkeypatch)
        body = b"{not valid json"
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="pull_request",
                x_github_delivery="d-pr-bad-json",
            )
        assert exc.value.status_code == 400
        assert "Invalid JSON body" in str(exc.value.detail)

    async def test_non_object_json_body_returns_400(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A JSON list parses fine but isn't a dict — the PR handler
        # rejects it with its own error text ("pull_request event
        # payload must be a JSON object"), distinct from the push
        # handler's wording.
        _patch_settings(monkeypatch)
        body = b'["not", "an", "object"]'
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="pull_request",
                x_github_delivery="d-pr-non-object",
            )
        assert exc.value.status_code == 400
        assert "pull_request" in str(exc.value.detail)
        assert "JSON object" in str(exc.value.detail)


class TestPullRequestAuditCap:
    async def test_audit_log_truncates_at_cap(
        self,
        tmp_storage: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Shrink the cap so the test runs fast.
        monkeypatch.setattr(github_webhook, "_MAX_PR_AUDIT", 3)
        # Pre-seed a full-then-some audit log.
        for i in range(5):
            github_webhook._record_pr_event({"pr_number": i, "outcome": "opened"})
        data = github_webhook.storage.read_json(github_webhook._PR_AUDIT_KEY)
        entries = data["entries"]
        assert len(entries) == 3
        assert [e["pr_number"] for e in entries] == [2, 3, 4]


class TestPushAnnotation:
    """Push results gain ``pr_number``/``pr_html_url``/``pr_diff_url``
    when the head commit subject matches GitHub's merge templates."""

    async def test_default_merge_commit_surfaces_pr_number(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(
            _push_payload(
                head_message=(
                    "Merge pull request #128 from owner/feature-branch\n\nAdd a verified notebook"
                )
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="d-push-merge",
        )
        assert result["status"] == "processed"
        assert result["pr_number"] == 128
        assert result["pr_html_url"] == "https://github.com/owner/repo/pull/128"
        assert result["pr_diff_url"] == "https://github.com/owner/repo/pull/128/files"

    async def test_squash_merge_subject_surfaces_pr_number(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload(head_message="Improve the docstring (#7)")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="d-push-squash",
        )
        assert result["pr_number"] == 7
        assert result["pr_html_url"] == "https://github.com/owner/repo/pull/7"

    async def test_rebase_merge_with_no_pr_number_is_none(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload(head_message="A regular commit subject")).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="d-push-rebase",
        )
        assert result["pr_number"] is None
        assert result["pr_html_url"] == ""
        assert result["pr_diff_url"] == ""

    async def test_missing_head_commit_is_safe(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload(head_message=None)).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="d-push-no-head",
        )
        assert result["pr_number"] is None


class TestParseHelpersDirect:
    """Direct coverage of the regex helpers — small, fast, and pins the
    contract independent of the full webhook flow."""

    def test_parse_merge_pr_number(self) -> None:
        assert (
            github_webhook._parse_pr_number_from_commit_message(
                "Merge pull request #99 from owner/foo\n\nbody"
            )
            == 99
        )

    def test_parse_squash_pr_number(self) -> None:
        assert (
            github_webhook._parse_pr_number_from_commit_message("Refactor sync logic (#12)") == 12
        )

    def test_parse_none_for_unrelated_subject(self) -> None:
        assert github_webhook._parse_pr_number_from_commit_message("Fix typo in README") is None

    def test_parse_none_for_empty_or_none(self) -> None:
        assert github_webhook._parse_pr_number_from_commit_message(None) is None
        assert github_webhook._parse_pr_number_from_commit_message("") is None

    def test_pr_urls_returns_empty_when_pr_number_missing(self) -> None:
        html, diff = github_webhook._pr_urls_from_repository(
            {"html_url": "https://github.com/o/r"}, None
        )
        assert html == "" and diff == ""

    def test_pr_urls_returns_empty_when_repo_missing(self) -> None:
        html, diff = github_webhook._pr_urls_from_repository(None, 42)
        assert html == "" and diff == ""

    def test_pr_urls_strips_trailing_slash_from_repo_url(self) -> None:
        html, diff = github_webhook._pr_urls_from_repository(
            {"html_url": "https://github.com/o/r/"}, 42
        )
        assert html == "https://github.com/o/r/pull/42"
        assert diff == "https://github.com/o/r/pull/42/files"
