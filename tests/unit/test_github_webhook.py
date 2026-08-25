"""Tests for the GitHub webhook handler (issue #46, step 7).

The webhook receives push events from the verified-notebook repo and
dispatches each changed path to ``sync_one_from_github`` (add/modify)
or ``delete_local_by_github_path`` (remove). Coverage:

* HMAC signature: valid → processed; invalid/missing → 401.
* Webhook secret not configured → 503.
* Ping event → pong (200).
* Non-push event → ignored (200, not processed).
* Push to non-target ref → ignored (200, not processed).
* Push with added/modified paths in verified/ → upsert dispatched.
* Push with removed paths in verified/ → delete dispatched.
* Push with paths outside verified/ → filtered out.
* Duplicate delivery → skipped (idempotency).
* Add-then-modify-then-remove across commits → last action (delete) wins.
* Invalid JSON body → 400.
* sync_one raising mid-loop → recorded as failed, other paths continue.
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
    github_publisher,
    github_webhook,
    verified_notebooks,
)
from data_concierge.gateway.github_webhook import github_webhook as webhook_handler


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    monkeypatch.setattr(github_publisher, "storage", test_storage)
    monkeypatch.setattr(github_webhook, "storage", test_storage)


WEBHOOK_SECRET = "shared-secret-for-tests"


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    secret: str = WEBHOOK_SECRET,
    branch: str = "main",
    verified_folder: str = "verified",
) -> None:
    fake_settings = {
        "paused": not enabled,
        "token": "tok",
        "repo": "owner/repo",
        "branch": branch,
        "drafts_folder": "drafts",
        "verified_folder": verified_folder,
        "webhook_secret": secret,
    }
    monkeypatch.setattr(
        github_publisher, "load_github_settings", lambda: fake_settings
    )
    # The webhook handler uses the status-aware variant (an earlier adversarial review)
    # so it can distinguish admin pause (HTTP 200) from a settings-read
    # failure (HTTP 503). Tests exercise the "ok" path; fail-closed
    # scenarios are covered by direct unit tests in test_github_settings.
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings_with_status",
        lambda: (fake_settings, "ok"),
    )


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def _make_request(body: bytes, headers: dict[str, str]) -> Request:
    """Build a Starlette Request whose body() returns ``body``."""
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1"))
        for k, v in headers.items()
    ]
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


def _push_payload(
    *,
    ref: str = "refs/heads/main",
    commits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ref": ref,
        "before": "0" * 40,
        "after": "1" * 40,
        "commits": commits or [],
        "head_commit": (commits or [{}])[-1] if commits else None,
        "repository": {"full_name": "owner/repo"},
    }


class TestSignatureVerification:
    async def test_valid_signature_is_accepted(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload()).encode("utf-8")
        req = _make_request(
            body,
            {
                "x-hub-signature-256": _sign(body),
                "x-github-event": "push",
                "x-github-delivery": "delivery-valid-1",
            },
        )
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-valid-1",
        )
        # Empty commits → no upserts/deletes, but status is processed.
        assert result["status"] == "processed"
        assert result["upserts"] == []
        assert result["deletes"] == []

    async def test_invalid_signature_returns_401(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload()).encode("utf-8")
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256="sha256=" + "0" * 64,
                x_github_event="push",
                x_github_delivery="delivery-bad-sig",
            )
        assert exc.value.status_code == 401

    async def test_missing_signature_header_returns_401(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload()).encode("utf-8")
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=None,
                x_github_event="push",
                x_github_delivery="delivery-missing-sig",
            )
        assert exc.value.status_code == 401

    async def test_signature_with_wrong_prefix_returns_401(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SHA-1 signature ('sha1=...') from an old GitHub config must
        be rejected — we ONLY accept SHA-256."""
        _patch_settings(monkeypatch)
        body = json.dumps(_push_payload()).encode("utf-8")
        # Valid SHA-1 HMAC but with the wrong prefix label.
        sha1_sig = "sha1=" + hmac.new(
            WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha1
        ).hexdigest()
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=sha1_sig,
                x_github_event="push",
                x_github_delivery="delivery-sha1",
            )
        assert exc.value.status_code == 401


class TestSecretNotConfigured:
    async def test_returns_503_when_secret_empty(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, secret="")
        body = json.dumps(_push_payload()).encode("utf-8")
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256="anything",
                x_github_event="push",
                x_github_delivery="d",
            )
        assert exc.value.status_code == 503
        assert "not configured" in str(exc.value.detail).lower()


class TestEventRouting:
    async def test_ping_event_returns_pong(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = b'{"zen": "Speak like a human"}'
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="ping",
            x_github_delivery="delivery-ping",
        )
        assert result["status"] == "pong"

    async def test_non_push_event_is_ignored(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``pull_request`` is now actively handled (issue #74) — use an
        # event we genuinely ignore (e.g. ``issues``) to exercise the
        # non-push fallback.
        _patch_settings(monkeypatch)
        body = json.dumps({"action": "opened"}).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="issues",
            x_github_delivery="delivery-issues",
        )
        assert result["status"] == "ignored"
        assert result["event"] == "issues"

    async def test_push_to_non_target_ref_is_ignored(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, branch="main")
        body = json.dumps(
            _push_payload(ref="refs/heads/dev-branch")
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-wrong-ref",
        )
        assert result["status"] == "ignored"
        assert result["reason"] == "ref mismatch"


class TestPushDispatch:
    """sync_one_from_github / delete_local_by_github_path are monkeypatched
    here so the dispatch logic is verified independently of step-6 wiring
    (which already has its own coverage in test_sync_one_from_github.py)."""

    async def test_added_path_in_verified_folder_dispatches_sync_one(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            calls.append(path)
            return {"status": "created", "notebook_id": "nb-x", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )

        body = json.dumps(
            _push_payload(
                commits=[{"added": ["verified/new.ipynb"], "modified": [], "removed": []}]
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-added",
        )
        assert calls == ["verified/new.ipynb"]
        assert result["status"] == "processed"
        assert len(result["upserts"]) == 1
        assert result["deletes"] == []

    async def test_removed_path_dispatches_delete(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        calls: list[str] = []

        def _fake_delete(path: str) -> dict[str, Any]:
            calls.append(path)
            return {"status": "deleted", "notebook_id": "nb-x", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "delete_local_by_github_path", _fake_delete
        )

        body = json.dumps(
            _push_payload(
                commits=[{"added": [], "modified": [], "removed": ["verified/gone.ipynb"]}]
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-removed",
        )
        assert calls == ["verified/gone.ipynb"]
        assert result["upserts"] == []
        assert len(result["deletes"]) == 1

    async def test_paths_outside_verified_folder_are_filtered(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A push to README.md or drafts/foo.ipynb must NOT trigger any
        sync — those aren't verified notebooks."""
        _patch_settings(monkeypatch)
        sync_calls: list[str] = []
        delete_calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            sync_calls.append(path)
            return {"status": "created", "path": path}

        def _fake_delete(path: str) -> dict[str, Any]:
            delete_calls.append(path)
            return {"status": "deleted", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )
        monkeypatch.setattr(
            verified_notebooks, "delete_local_by_github_path", _fake_delete
        )

        body = json.dumps(
            _push_payload(
                commits=[
                    {
                        "added": ["README.md", "drafts/x.ipynb"],
                        "modified": ["docs/site.yml"],
                        "removed": ["scripts/old.py"],
                    }
                ]
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-noverified",
        )
        assert sync_calls == []
        assert delete_calls == []
        assert result["status"] == "processed"
        assert result["upserts"] == []
        assert result["deletes"] == []

    async def test_last_action_per_path_wins(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If a path is added in commit A, modified in commit B, and
        removed in commit C, the final on-GitHub state is REMOVED.
        Only the delete should fire; no upsert for this path."""
        _patch_settings(monkeypatch)
        sync_calls: list[str] = []
        delete_calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            sync_calls.append(path)
            return {"status": "created", "path": path}

        def _fake_delete(path: str) -> dict[str, Any]:
            delete_calls.append(path)
            return {"status": "deleted", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )
        monkeypatch.setattr(
            verified_notebooks, "delete_local_by_github_path", _fake_delete
        )

        body = json.dumps(
            _push_payload(
                commits=[
                    {"added": ["verified/a.ipynb"], "modified": [], "removed": []},
                    {"added": [], "modified": ["verified/a.ipynb"], "removed": []},
                    {"added": [], "modified": [], "removed": ["verified/a.ipynb"]},
                ]
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-collapse",
        )
        assert sync_calls == []
        assert delete_calls == ["verified/a.ipynb"]
        assert len(result["deletes"]) == 1

    async def test_failure_in_one_dispatch_does_not_abort_others(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If sync_one raises for one path, the webhook must record the
        failure but still process the remaining paths in the push."""
        _patch_settings(monkeypatch)
        seen: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            seen.append(path)
            if "bomb" in path:
                raise RuntimeError("boom")
            return {"status": "created", "notebook_id": f"nb-{path}", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )

        body = json.dumps(
            _push_payload(
                commits=[
                    {
                        "added": [
                            "verified/ok-1.ipynb",
                            "verified/bomb.ipynb",
                            "verified/ok-2.ipynb",
                        ],
                        "modified": [],
                        "removed": [],
                    }
                ]
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-failure-mid",
        )
        # All three were attempted.
        assert set(seen) == {
            "verified/ok-1.ipynb",
            "verified/bomb.ipynb",
            "verified/ok-2.ipynb",
        }
        statuses_by_path = {r["path"]: r["status"] for r in result["upserts"]}
        assert statuses_by_path["verified/ok-1.ipynb"] == "created"
        assert statuses_by_path["verified/bomb.ipynb"] == "failed"
        assert statuses_by_path["verified/ok-2.ipynb"] == "created"


class TestIdempotency:
    async def test_duplicate_delivery_is_skipped(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same X-GitHub-Delivery a second time must not re-dispatch."""
        _patch_settings(monkeypatch)
        sync_calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            sync_calls.append(path)
            return {"status": "created", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )

        body = json.dumps(
            _push_payload(
                commits=[{"added": ["verified/x.ipynb"], "modified": [], "removed": []}]
            )
        ).encode("utf-8")
        sig = _sign(body)

        # First delivery: processed.
        req1 = _make_request(body, {})
        r1 = await webhook_handler(
            request=req1,
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery="delivery-idem-1",
        )
        assert r1["status"] == "processed"
        assert sync_calls == ["verified/x.ipynb"]

        # Same delivery again: skipped, NOT re-dispatched.
        req2 = _make_request(body, {})
        r2 = await webhook_handler(
            request=req2,
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery="delivery-idem-1",
        )
        assert r2["status"] == "duplicate_delivery"
        assert sync_calls == ["verified/x.ipynb"], (
            "sync_one must NOT have been called a second time"
        )

    async def test_different_deliveries_are_both_processed(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        sync_calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            sync_calls.append(path)
            return {"status": "created", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )

        for d_id, path in [("delivery-a", "verified/a.ipynb"),
                            ("delivery-b", "verified/b.ipynb")]:
            body = json.dumps(
                _push_payload(
                    commits=[
                        {"added": [path], "modified": [], "removed": []}
                    ]
                )
            ).encode("utf-8")
            req = _make_request(body, {})
            r = await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="push",
                x_github_delivery=d_id,
            )
            assert r["status"] == "processed"

        assert sync_calls == ["verified/a.ipynb", "verified/b.ipynb"]


    async def test_redelivery_after_failed_path_is_retried(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an earlier adversarial review: if any per-path op returned status="failed"
        (transient GitHub fetch error, etc.), the delivery MUST NOT be
        marked seen — otherwise a GitHub redelivery would be skipped as
        a duplicate, leaving the local index permanently stale for the
        failed paths."""
        _patch_settings(monkeypatch)

        attempt = {"n": 0}

        async def _fake_sync(path: str) -> dict[str, Any]:
            attempt["n"] += 1
            # First call: transient failure. Second call: success.
            if attempt["n"] == 1:
                return {"status": "failed", "path": path, "reason": "transient"}
            return {"status": "created", "notebook_id": "nb-x", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )

        body = json.dumps(
            _push_payload(
                commits=[
                    {"added": ["verified/retry.ipynb"], "modified": [], "removed": []}
                ]
            )
        ).encode("utf-8")
        sig = _sign(body)
        delivery_id = "delivery-retry-after-fail"

        # First delivery: one failed path.
        r1 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r1["status"] == "processed"
        assert r1["any_failed"] is True
        assert r1["upserts"][0]["status"] == "failed"

        # Same delivery again: MUST be re-processed, not skipped as
        # duplicate, because the first attempt had a failed path.
        r2 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r2["status"] == "processed", (
            "redelivery after a failed path must be retried, not deduped"
        )
        assert r2["any_failed"] is False
        assert r2["upserts"][0]["status"] == "created"
        assert attempt["n"] == 2, "sync_one must have been called a second time"

        # A THIRD delivery (now that everything succeeded) should be
        # deduped — the success was marked seen.
        r3 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r3["status"] == "duplicate_delivery"
        assert attempt["n"] == 2, "third delivery must NOT call sync_one again"

    async def test_redelivery_after_raised_exception_is_retried(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sync_one that raises is recorded as status="failed" by the
        webhook wrapper; the same retry semantics must apply."""
        _patch_settings(monkeypatch)

        attempt = {"n": 0}

        async def _fake_sync(path: str) -> dict[str, Any]:
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise RuntimeError("transient")
            return {"status": "created", "notebook_id": "nb-y", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_from_github", _fake_sync
        )

        body = json.dumps(
            _push_payload(
                commits=[
                    {"added": ["verified/raise.ipynb"], "modified": [], "removed": []}
                ]
            )
        ).encode("utf-8")
        sig = _sign(body)
        delivery_id = "delivery-retry-after-raise"

        r1 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r1["any_failed"] is True

        r2 = await webhook_handler(
            request=_make_request(body, {}),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r2["status"] == "processed"
        assert r2["any_failed"] is False
        assert attempt["n"] == 2


class TestSettingsFailClosed:
    """Regression coverage for an earlier adversarial review.

    When ``load_github_settings_with_status`` reports ``fail_closed``
    (storage read errored), the webhook MUST return HTTP 503 — not a
    200 with ``skipped_paused`` — so GitHub retries the delivery once
    storage recovers. A 200 acknowledges the delivery and GitHub will
    never replay it.
    """

    async def test_fail_closed_settings_returns_503(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # paused=True + status="fail_closed" mimics the corrupt/unreadable
        # settings file case. Webhook secret is still set so signature
        # verification succeeds and we reach the pause guard.
        fake = {
            "paused": True,
            "token": "",  # empty because env-derived only
            "repo": "owner/repo",
            "branch": "main",
            "drafts_folder": "drafts",
            "verified_folder": "verified",
            "verified_answers_folder": "verified-answers",
            "webhook_secret": WEBHOOK_SECRET,
        }
        monkeypatch.setattr(
            github_publisher, "load_github_settings", lambda: fake
        )
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings_with_status",
            lambda: (fake, "fail_closed"),
        )
        body = json.dumps(_push_payload()).encode("utf-8")
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="push",
                x_github_delivery="delivery-fail-closed-1",
            )
        assert exc.value.status_code == 503

    async def test_intentional_pause_returns_200_skipped(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # paused=True + status="ok" mimics an admin who deliberately
        # paused publishing. This stays HTTP 200 so GitHub does NOT
        # retry — the admin uses "Sync from GitHub" to catch up.
        fake = {
            "paused": True,
            "token": "tok",
            "repo": "owner/repo",
            "branch": "main",
            "drafts_folder": "drafts",
            "verified_folder": "verified",
            "verified_answers_folder": "verified-answers",
            "webhook_secret": WEBHOOK_SECRET,
        }
        monkeypatch.setattr(
            github_publisher, "load_github_settings", lambda: fake
        )
        monkeypatch.setattr(
            github_publisher,
            "load_github_settings_with_status",
            lambda: (fake, "ok"),
        )
        body = json.dumps(_push_payload()).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-paused-1",
        )
        assert result["status"] == "skipped_paused"


class TestMalformedBody:
    async def test_invalid_json_returns_400(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = b"{not json"
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="push",
                x_github_delivery="delivery-badjson",
            )
        assert exc.value.status_code == 400

    async def test_non_dict_payload_returns_400(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        body = b'["a", "list", "not", "an", "object"]'
        req = _make_request(body, {})
        with pytest.raises(HTTPException) as exc:
            await webhook_handler(
                request=req,
                x_hub_signature_256=_sign(body),
                x_github_event="push",
                x_github_delivery="delivery-listpayload",
            )
        assert exc.value.status_code == 400


class TestEndToEndWithRealSyncOne:
    """One smoke test exercising the FULL chain: HMAC verify → branch
    filter → sync_one_from_github writes a real local entry. The
    monkeypatched layer above stubs sync_one to keep dispatch tests
    independent; this test makes sure the real wiring works."""

    async def test_real_push_creates_local_verified_entry(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        from data_concierge.gateway.verified_notebooks import (
            get_verified_notebooks,
        )

        # Provide a real fetch_notebook for the path the webhook will sync.
        nb_json = {
            "cells": [{"cell_type": "markdown", "source": ["# real"],
                       "metadata": {}}],
            "metadata": {
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "data_concierge": {
                    "version": "0.1.0",
                    "generated": "2026-05-27T10:00:00",
                    "query": "real",
                    "data_source": "bls",
                    "submission_id": "sub-real",
                    "confidence": 0.9,
                    "verified_by": "admin",
                    "verified_at": "2026-05-27T20:00:00Z",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            assert path == "verified/real.ipynb"
            return nb_json

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        body = json.dumps(
            _push_payload(
                commits=[
                    {
                        "added": ["verified/real.ipynb"],
                        "modified": [],
                        "removed": [],
                    }
                ]
            )
        ).encode("utf-8")
        req = _make_request(body, {})
        result = await webhook_handler(
            request=req,
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-e2e",
        )
        assert result["status"] == "processed"
        assert len(result["upserts"]) == 1
        assert result["upserts"][0]["status"] == "created"

        verifieds = get_verified_notebooks()
        assert [v.submission_id for v in verifieds] == ["sub-real"]
