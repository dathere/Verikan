"""Tests for webhook dispatch on verified-answer paths (issue #46 step 9 PR C).

PR C extends the step-7 webhook to also recognize the
``verified_answers_folder/`` prefix in push events:

* ADD/MODIFY of ``verified-answers/<id>.json`` → ``sync_one_answer_from_github``
* REMOVE of ``verified-answers/<id>.json`` → ``delete_local_answer_by_github_path``

The existing notebook dispatch (``verified/<id>.ipynb``) is unchanged
and lives in ``test_github_webhook.py``; this file focuses on the new
answer branch + cross-folder mixed pushes + the delete helper.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest
from starlette.requests import Request

from data_concierge.gateway import (
    github_publisher,
    github_webhook,
    verified_notebooks,
)
from data_concierge.gateway.github_webhook import github_webhook as webhook_handler
from data_concierge.gateway.verified_notebooks import (
    approve_quick_answer,
    delete_local_answer_by_github_path,
    get_verified_answer,
    submit_quick_answer,
)

WEBHOOK_SECRET = "shared-secret-for-tests"


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    monkeypatch.setattr(github_publisher, "storage", test_storage)
    monkeypatch.setattr(github_webhook, "storage", test_storage)


def _patch_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    secret: str = WEBHOOK_SECRET,
    branch: str = "main",
    verified_folder: str = "verified",
    verified_answers_folder: str = "verified-answers",
) -> None:
    fake_settings = {
        "paused": not enabled,
        "token": "tok",
        "repo": "owner/repo",
        "branch": branch,
        "drafts_folder": "drafts",
        "verified_folder": verified_folder,
        "verified_answers_folder": verified_answers_folder,
        "webhook_secret": secret,
    }
    monkeypatch.setattr(
        github_publisher, "load_github_settings", lambda: fake_settings
    )
    # The webhook handler uses the status-aware variant (an earlier adversarial review)
    # so it can distinguish admin pause (HTTP 200) from a settings-read
    # failure (HTTP 503). Tests exercise the "ok" path.
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings_with_status",
        lambda: (fake_settings, "ok"),
    )


def _sign(body: bytes, secret: str = WEBHOOK_SECRET) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def _make_request(body: bytes) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/github/webhook",
        "headers": [],
        "query_string": b"",
    }

    async def _receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive=_receive)


def _push_payload(*, commits: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "ref": "refs/heads/main",
        "before": "0" * 40,
        "after": "1" * 40,
        "commits": commits or [],
        "head_commit": (commits or [{}])[-1] if commits else None,
        "repository": {"full_name": "owner/repo"},
    }


def _seed_local_answer(github_path: str) -> Any:
    """Seed a verified answer in the local index with a github_path set —
    needed to test delete dispatch which looks up by github_path."""
    sub = submit_quick_answer(
        query="seed", answer="0.1", source_links=[],
        submitted_by="tester", data_source="bls", confidence=0.5,
    )
    return approve_quick_answer(
        submission_id=sub.submission_id,
        reviewed_by="admin",
        github_path=github_path,
        github_synced_at="2026-05-27T20:00:00Z",
    )


# ---------------------------------------------------------------------------
# delete_local_answer_by_github_path — unit
# ---------------------------------------------------------------------------


class TestDeleteLocalAnswerByGithubPath:
    def test_deletes_local_entry_when_path_matches(
        self, tmp_storage: None
    ) -> None:
        verified = _seed_local_answer("verified-answers/a-1.json")
        assert verified is not None

        result = delete_local_answer_by_github_path("verified-answers/a-1.json")
        assert result["status"] == "deleted"
        assert result["answer_id"] == verified.answer_id
        # Local entry is actually gone.
        assert get_verified_answer(verified.answer_id) is None

    def test_skips_when_no_local_match(self, tmp_storage: None) -> None:
        result = delete_local_answer_by_github_path("verified-answers/missing.json")
        assert result == {
            "status": "skipped_no_local",
            "path": "verified-answers/missing.json",
        }

    def test_does_not_touch_unrelated_entries(self, tmp_storage: None) -> None:
        keep = _seed_local_answer("verified-answers/keep.json")
        target = _seed_local_answer("verified-answers/drop.json")
        assert keep is not None and target is not None

        result = delete_local_answer_by_github_path("verified-answers/drop.json")
        assert result["status"] == "deleted"
        # The unrelated entry survives.
        assert get_verified_answer(keep.answer_id) is not None
        assert get_verified_answer(target.answer_id) is None


# ---------------------------------------------------------------------------
# Webhook dispatch on verified-answers/ prefix
# ---------------------------------------------------------------------------


class TestWebhookAnswerDispatch:
    async def test_added_answer_path_dispatches_to_sync_one_answer(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            calls.append(path)
            return {"status": "created", "answer_id": "a-x", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_answer_from_github", _fake_sync
        )

        body = json.dumps(_push_payload(commits=[{
            "added": ["verified-answers/a-x.json"],
            "modified": [],
            "removed": [],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-ans-added",
        )
        assert result["status"] == "processed"
        assert calls == ["verified-answers/a-x.json"]
        assert len(result["upserts"]) == 1
        assert result["upserts"][0]["answer_id"] == "a-x"

    async def test_modified_answer_path_dispatches_to_sync_one_answer(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        calls: list[str] = []

        async def _fake_sync(path: str) -> dict[str, Any]:
            calls.append(path)
            return {"status": "updated", "answer_id": "a-m", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "sync_one_answer_from_github", _fake_sync
        )

        body = json.dumps(_push_payload(commits=[{
            "added": [],
            "modified": ["verified-answers/a-m.json"],
            "removed": [],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-ans-mod",
        )
        assert calls == ["verified-answers/a-m.json"]
        assert result["upserts"][0]["status"] == "updated"

    async def test_removed_answer_path_dispatches_to_delete_helper(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        calls: list[str] = []

        def _fake_delete(path: str) -> dict[str, Any]:
            calls.append(path)
            return {"status": "deleted", "answer_id": "a-gone", "path": path}

        monkeypatch.setattr(
            verified_notebooks, "delete_local_answer_by_github_path", _fake_delete
        )

        body = json.dumps(_push_payload(commits=[{
            "added": [],
            "modified": [],
            "removed": ["verified-answers/a-gone.json"],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-ans-removed",
        )
        assert calls == ["verified-answers/a-gone.json"]
        assert len(result["deletes"]) == 1
        assert result["deletes"][0]["status"] == "deleted"

    async def test_paths_outside_both_folders_are_filtered(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        sync_nb: list[str] = []
        sync_ans: list[str] = []
        del_nb: list[str] = []
        del_ans: list[str] = []

        async def _fake_sync_nb(path: str) -> dict[str, Any]:
            sync_nb.append(path)
            return {"status": "created", "path": path}

        async def _fake_sync_ans(path: str) -> dict[str, Any]:
            sync_ans.append(path)
            return {"status": "created", "path": path}

        def _fake_del_nb(path: str) -> dict[str, Any]:
            del_nb.append(path)
            return {"status": "deleted", "path": path}

        def _fake_del_ans(path: str) -> dict[str, Any]:
            del_ans.append(path)
            return {"status": "deleted", "path": path}

        monkeypatch.setattr(verified_notebooks, "sync_one_from_github", _fake_sync_nb)
        monkeypatch.setattr(verified_notebooks, "sync_one_answer_from_github", _fake_sync_ans)
        monkeypatch.setattr(verified_notebooks, "delete_local_by_github_path", _fake_del_nb)
        monkeypatch.setattr(verified_notebooks, "delete_local_answer_by_github_path", _fake_del_ans)

        body = json.dumps(_push_payload(commits=[{
            "added": ["README.md", "drafts/x.ipynb", "docs/site.yml"],
            "modified": ["scripts/foo.py"],
            "removed": ["misc/bar.json"],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-ans-noisy",
        )
        assert result["status"] == "processed"
        assert sync_nb == [] and sync_ans == []
        assert del_nb == [] and del_ans == []
        assert result["upserts"] == [] and result["deletes"] == []


# ---------------------------------------------------------------------------
# Mixed pushes — notebooks AND answers in the same event
# ---------------------------------------------------------------------------


class TestWebhookMixedDispatch:
    async def test_mixed_push_dispatches_to_both_helpers(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single push that adds a notebook AND an answer must invoke
        the correct helper for each — no cross-contamination, no
        double-dispatch."""
        _patch_settings(monkeypatch)
        sync_nb: list[str] = []
        sync_ans: list[str] = []

        async def _fake_sync_nb(path: str) -> dict[str, Any]:
            sync_nb.append(path)
            return {"status": "created", "notebook_id": "nb-1", "path": path}

        async def _fake_sync_ans(path: str) -> dict[str, Any]:
            sync_ans.append(path)
            return {"status": "created", "answer_id": "a-1", "path": path}

        monkeypatch.setattr(verified_notebooks, "sync_one_from_github", _fake_sync_nb)
        monkeypatch.setattr(verified_notebooks, "sync_one_answer_from_github", _fake_sync_ans)

        body = json.dumps(_push_payload(commits=[{
            "added": [
                "verified/nb-1.ipynb",
                "verified-answers/a-1.json",
            ],
            "modified": [],
            "removed": [],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-mixed",
        )
        assert result["status"] == "processed"
        assert sync_nb == ["verified/nb-1.ipynb"]
        assert sync_ans == ["verified-answers/a-1.json"]
        # Both upserts reported in the response.
        statuses = {r.get("notebook_id") or r.get("answer_id"): r["path"]
                    for r in result["upserts"]}
        assert "nb-1" in statuses and "a-1" in statuses

    async def test_verified_folder_does_not_match_verified_answers_path(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critical correctness property: ``verified/`` and
        ``verified-answers/`` MUST NOT cross-match. A ``verified-answers/x.json``
        path must dispatch to the answer helper only, never to the
        notebook helper (the slash makes the prefixes disjoint).
        Worth an explicit assertion because both prefixes share the
        word 'verified'."""
        _patch_settings(monkeypatch)
        sync_nb: list[str] = []
        sync_ans: list[str] = []

        async def _fake_sync_nb(path: str) -> dict[str, Any]:
            sync_nb.append(path)
            return {"status": "created", "path": path}

        async def _fake_sync_ans(path: str) -> dict[str, Any]:
            sync_ans.append(path)
            return {"status": "created", "path": path}

        monkeypatch.setattr(verified_notebooks, "sync_one_from_github", _fake_sync_nb)
        monkeypatch.setattr(verified_notebooks, "sync_one_answer_from_github", _fake_sync_ans)

        body = json.dumps(_push_payload(commits=[{
            "added": ["verified-answers/x.json"],
            "modified": [],
            "removed": [],
        }])).encode("utf-8")
        await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-no-cross",
        )
        assert sync_nb == []  # MUST NOT cross-match
        assert sync_ans == ["verified-answers/x.json"]

    async def test_mixed_failure_blocks_marking_seen_for_retry(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The an earlier adversarial review idempotency rule (any failed → don't mark
        seen → redelivery retries) extends across both content types.
        A failing notebook upsert with a successful answer upsert must
        still leave the delivery unmarked so retry re-processes both."""
        _patch_settings(monkeypatch)

        nb_attempts = {"n": 0}

        async def _fake_sync_nb(path: str) -> dict[str, Any]:
            nb_attempts["n"] += 1
            if nb_attempts["n"] == 1:
                return {"status": "failed", "path": path, "reason": "transient"}
            return {"status": "created", "notebook_id": "nb-z", "path": path}

        async def _fake_sync_ans(path: str) -> dict[str, Any]:
            return {"status": "created", "answer_id": "a-z", "path": path}

        monkeypatch.setattr(verified_notebooks, "sync_one_from_github", _fake_sync_nb)
        monkeypatch.setattr(verified_notebooks, "sync_one_answer_from_github", _fake_sync_ans)

        body = json.dumps(_push_payload(commits=[{
            "added": ["verified/nb-z.ipynb", "verified-answers/a-z.json"],
            "modified": [],
            "removed": [],
        }])).encode("utf-8")
        sig = _sign(body)
        delivery_id = "delivery-mixed-retry"

        r1 = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r1["any_failed"] is True

        # Redelivery: must NOT be deduped because notebook failed.
        r2 = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=sig,
            x_github_event="push",
            x_github_delivery=delivery_id,
        )
        assert r2["status"] == "processed"
        assert r2["any_failed"] is False
        assert nb_attempts["n"] == 2  # notebook was retried


# ---------------------------------------------------------------------------
# End-to-end with REAL sync_one_answer and storage
# ---------------------------------------------------------------------------


class TestWebhookAnswerEndToEnd:
    async def test_real_push_creates_local_verified_answer(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercise the FULL chain: HMAC verify → branch filter → answer
        dispatch → sync_one_answer_from_github → real local storage."""
        _patch_settings(monkeypatch)
        from data_concierge.gateway.verified_notebooks import (
            get_verified_answers,
        )

        payload_json = {
            "answer_id": "real-1",
            "submission_id": "sub-real",
            "query": "real",
            "answer": "yes",
            "source_links": [],
            "verified_at": "2026-05-27T22:00:00Z",
            "verified_by": "admin",
            "data_source": "bls",
            "confidence": 0.9,
            "submitted_by": "tester",
            "variable": "",
            "place": "",
            "date": "",
            "value": "",
            "usage_count": 0,
            "keywords": [],
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            assert path == "verified-answers/real-1.json"
            return payload_json

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        body = json.dumps(_push_payload(commits=[{
            "added": ["verified-answers/real-1.json"],
            "modified": [],
            "removed": [],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-ans-e2e",
        )
        assert result["status"] == "processed"
        assert result["upserts"][0]["status"] == "created"
        assert [a.answer_id for a in get_verified_answers()] == ["real-1"]

    async def test_real_push_deletes_local_verified_answer(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Seed an answer with a github_path, then receive a REMOVE
        webhook event for that path — local entry must disappear."""
        _patch_settings(monkeypatch)
        from data_concierge.gateway.verified_notebooks import (
            get_verified_answers,
        )

        seeded = _seed_local_answer("verified-answers/del-1.json")
        assert seeded is not None
        assert [a.answer_id for a in get_verified_answers()] == [seeded.answer_id]

        body = json.dumps(_push_payload(commits=[{
            "added": [],
            "modified": [],
            "removed": ["verified-answers/del-1.json"],
        }])).encode("utf-8")
        result = await webhook_handler(
            request=_make_request(body),
            x_hub_signature_256=_sign(body),
            x_github_event="push",
            x_github_delivery="delivery-ans-del-e2e",
        )
        assert result["status"] == "processed"
        assert result["deletes"][0]["status"] == "deleted"
        assert result["deletes"][0]["answer_id"] == seeded.answer_id
        # Local entry gone.
        assert get_verified_answers() == []


# ---------------------------------------------------------------------------
# Route registration regression (post-an earlier adversarial review lesson)
# ---------------------------------------------------------------------------


class TestWebhookRouteStillRegistered:
    def test_webhook_route_is_registered(self) -> None:
        """Make sure the @router.post('/webhook') decorator wasn't
        accidentally dropped while we extended the handler body."""
        from data_concierge.gateway.github_webhook import router

        post_paths = {
            r.path for r in router.routes
            if getattr(r, "methods", None) and "POST" in r.methods
        }
        assert "/api/v1/github/webhook" in post_paths
