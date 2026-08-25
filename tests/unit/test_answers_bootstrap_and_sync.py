"""Tests for verified-answer bootstrap + per-path sync (issue #46, step 9 PR B).

Mirrors test_bootstrap_from_github.py + test_sync_one_from_github.py for
notebooks, with the structural simplifications answers get:

* The file IS the model (no metadata.data_concierge wrapper).
* ``answer_id`` is the lookup key AND the filename (no separate
  submission_id indirection).
* No on-disk blob — the index entry IS the payload.

Coverage:

* bootstrap rebuilds local from empty when GitHub has files.
* bootstrap update path preserves operational state
  (``usage_count`` / ``admin_notes`` / ``tags`` / ``keywords``) AND
  refreshes GitHub-owned fields.
* bootstrap update with type-corrupt GitHub payload → skipped_bad_metadata,
  local entry untouched (round-trip-via-model guard).
* bootstrap shape guards: non-dict file, missing answer_id, non-string
  answer_id → all skipped_bad_metadata.
* bootstrap duplicate answer_id → first wins, rest reported.
* bootstrap orphaned-locally: local entry with no GitHub counterpart is
  reported, NOT deleted.
* bootstrap GitHub disabled → skipped.
* bootstrap strict listing failure → GitHubPublishError propagates so
  the endpoint can 502.
* sync_one mirrors bootstrap's per-file semantics.
* Endpoints return 400 / 502 / success appropriately.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from data_concierge.gateway import github_publisher, verified_notebooks
from data_concierge.gateway.verified_notebooks import (
    VerifiedAnswer,
    approve_quick_answer,
    bootstrap_answers_from_github,
    get_verified_answer,
    get_verified_answers,
    submit_quick_answer,
    sync_one_answer_from_github,
)


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


def _make_answer_payload(
    *,
    answer_id: str,
    query: str = "q",
    answer: str = "a",
    confidence: float = 0.9,
    verified_by: str = "admin",
    verified_at: str = "2026-05-27T18:30:00Z",
    data_source: str = "bls",
) -> dict[str, Any]:
    """Construct a JSON payload shaped like what PR A publishes."""
    return {
        "answer_id": answer_id,
        "submission_id": f"sub-{answer_id}",
        "query": query,
        "answer": answer,
        "source_links": [
            {"name": "BLS", "url": "https://bls.gov/x", "description": "Source"}
        ],
        "verified_at": verified_at,
        "verified_by": verified_by,
        "data_source": data_source,
        "confidence": confidence,
        "submitted_by": "tester",
        "input_tokens": None,
        "output_tokens": None,
        "variable": "unemployment_rate",
        "place": "Texas",
        "date": "2026-04",
        "value": "4.1",
        "usage_count": 0,
        "keywords": [],
        # admin_notes / tags / github_path / github_synced_at default-None.
    }


def _seed_local_answer(**overrides: Any) -> VerifiedAnswer:
    """Seed a verified answer in the local index via the normal pipeline
    so the index entry has the same shape PR A would produce."""
    sub = submit_quick_answer(
        query=overrides.get("query", "Pre-existing query"),
        answer=overrides.get("answer", "0.5"),
        source_links=overrides.get("source_links", []),
        submitted_by="tester",
        data_source=overrides.get("data_source", "bls"),
        confidence=overrides.get("confidence", 0.5),
    )
    verified = approve_quick_answer(
        submission_id=sub.submission_id,
        reviewed_by="admin",
        answer_id=overrides.get("answer_id"),
        github_path=overrides.get("github_path"),
        github_synced_at=overrides.get("github_synced_at"),
        verified_at=overrides.get("verified_at"),
    )
    assert verified is not None
    return verified


# ---------------------------------------------------------------------------
# Bootstrap — fresh local + GitHub-only state
# ---------------------------------------------------------------------------


class TestBootstrapAnswersFromEmptyLocalIndex:
    async def test_creates_entries_from_github_files(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            assert folder == "verified-answers"
            return [
                {"name": "a-1.json", "path": "verified-answers/a-1.json", "sha": "s1"},
                {"name": "a-2.json", "path": "verified-answers/a-2.json", "sha": "s2"},
            ]

        files = {
            "verified-answers/a-1.json": _make_answer_payload(answer_id="a-1", query="q1", answer="A1"),
            "verified-answers/a-2.json": _make_answer_payload(answer_id="a-2", query="q2", answer="A2"),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return files.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["checked"] == 2
        assert sorted(result["created"]) == ["a-1", "a-2"]
        assert result["updated"] == []
        assert result["failed"] == []
        assert result["skipped_bad_metadata"] == []
        assert result["skipped_duplicate_answer_id"] == []
        assert result["orphaned_locally"] == []

        # Local entries are queryable.
        verifieds = get_verified_answers()
        assert {v.answer_id for v in verifieds} == {"a-1", "a-2"}
        a1 = get_verified_answer("a-1")
        assert a1 is not None
        assert a1.query == "q1"
        assert a1.answer == "A1"
        # github_path / github_synced_at populated from bootstrap.
        assert a1.github_path == "verified-answers/a-1.json"
        assert a1.github_synced_at and a1.github_synced_at.endswith("Z")


# ---------------------------------------------------------------------------
# Bootstrap update path — preserve operational state, refresh GitHub-owned
# ---------------------------------------------------------------------------


class TestBootstrapAnswersPreservesLocalOperationalState:
    async def test_existing_entry_keeps_usage_count_admin_notes_tags(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critical correctness property mirroring the notebook
        equivalent: bootstrap MUST refresh GitHub-owned fields but
        MUST NOT clobber operational state."""
        _patch_settings(monkeypatch)

        seeded = _seed_local_answer(
            answer_id="a-pre",
            query="Old query",
            answer="0.5",
            confidence=0.5,
        )
        # Add operational state.
        verified_notebooks.increment_answer_usage(seeded.answer_id)
        verified_notebooks.increment_answer_usage(seeded.answer_id)
        idx = verified_notebooks._load_index()
        idx["verified_answers"][seeded.answer_id]["admin_notes"] = "reviewed"
        idx["verified_answers"][seeded.answer_id]["tags"] = ["finance"]
        verified_notebooks._save_index(idx)

        # Fresh GitHub version with different query/answer/confidence.
        fresh = _make_answer_payload(
            answer_id=seeded.answer_id,
            query="Refined query",
            answer="0.99",
            confidence=0.99,
            verified_by="admin2",
            verified_at="2026-05-27T20:00:00Z",
            data_source="census",
        )

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return [{"name": f"{seeded.answer_id}.json",
                     "path": f"verified-answers/{seeded.answer_id}.json",
                     "sha": "fresh"}]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return fresh

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["updated"] == [seeded.answer_id]
        assert result["created"] == []

        refreshed = get_verified_answer(seeded.answer_id)
        assert refreshed is not None
        # Operational state PRESERVED.
        assert refreshed.usage_count == 2
        assert refreshed.admin_notes == "reviewed"
        assert refreshed.tags == ["finance"]
        # GitHub-owned fields REFRESHED.
        assert refreshed.query == "Refined query"
        assert refreshed.answer == "0.99"
        assert refreshed.confidence == 0.99
        assert refreshed.verified_by == "admin2"
        assert refreshed.verified_at == "2026-05-27T20:00:00Z"
        assert refreshed.data_source == "census"
        assert refreshed.github_path == f"verified-answers/{seeded.answer_id}.json"
        assert refreshed.github_synced_at and refreshed.github_synced_at.endswith("Z")


# ---------------------------------------------------------------------------
# Bootstrap edge cases — bad metadata / duplicates / orphans / disabled
# ---------------------------------------------------------------------------


class TestBootstrapAnswersEdgeCases:
    async def test_corrupt_file_routes_to_skipped_bad_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All shape corruption cases go to skipped_bad_metadata: file
        not a dict, missing answer_id, non-string answer_id, Pydantic
        validation failure (e.g. non-numeric confidence)."""
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return [
                {"name": "a.json", "path": "verified-answers/a.json", "sha": "1"},
                {"name": "b.json", "path": "verified-answers/b.json", "sha": "2"},
                {"name": "c.json", "path": "verified-answers/c.json", "sha": "3"},
                {"name": "d.json", "path": "verified-answers/d.json", "sha": "4"},
                {"name": "e-good.json", "path": "verified-answers/e-good.json", "sha": "5"},
            ]

        # a: not a dict; b: missing answer_id; c: non-str answer_id;
        # d: confidence is a string; e: well-formed.
        d_bad = _make_answer_payload(answer_id="d-bad")
        d_bad["confidence"] = "very-high"
        files: dict[str, Any] = {
            "verified-answers/a.json": ["not", "a", "dict"],
            "verified-answers/b.json": {"query": "no id here"},
            "verified-answers/c.json": {
                "answer_id": ["unhashable"],
                "query": "q",
                "answer": "a",
            },
            "verified-answers/d.json": d_bad,
            "verified-answers/e-good.json": _make_answer_payload(answer_id="e-good"),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return files.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["checked"] == 5
        # Only e-good landed.
        assert result["created"] == ["e-good"]
        assert set(result["skipped_bad_metadata"]) == {
            "verified-answers/a.json",
            "verified-answers/b.json",
            "verified-answers/c.json",
            "verified-answers/d.json",
        }
        # Nothing leaked to failed / created.
        assert result["failed"] == []

    async def test_bad_metadata_on_update_preserves_existing(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Round-trip-via-model guard: corrupt GitHub payload for an
        existing local entry MUST leave the local entry untouched."""
        _patch_settings(monkeypatch)

        seeded = _seed_local_answer(
            answer_id="a-good", query="Good", answer="0.85", confidence=0.85
        )

        bad = _make_answer_payload(
            answer_id=seeded.answer_id,
            query="Should NOT apply",
            answer="0.7",
            confidence=0.7,
        )
        # verified_by must be a str per the model — list breaks validation.
        bad["verified_by"] = ["not", "a", "str"]

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return [{"name": "a-good.json",
                     "path": "verified-answers/a-good.json",
                     "sha": "1"}]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return bad

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["updated"] == []
        assert result["skipped_bad_metadata"] == ["verified-answers/a-good.json"]

        unchanged = get_verified_answer(seeded.answer_id)
        assert unchanged is not None
        assert unchanged.query == seeded.query
        assert unchanged.answer == seeded.answer
        assert unchanged.confidence == seeded.confidence
        assert unchanged.verified_by == seeded.verified_by
        # And listing still works (would raise ValidationError on read
        # if a corrupt entry had landed).
        listed = get_verified_answers()
        assert any(a.answer_id == seeded.answer_id for a in listed)

    async def test_duplicate_answer_id_first_wins(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return [
                {"name": "a.json", "path": "verified-answers/first.json", "sha": "1"},
                {"name": "b.json", "path": "verified-answers/dup.json", "sha": "2"},
                {"name": "c.json", "path": "verified-answers/other.json", "sha": "3"},
            ]

        files = {
            "verified-answers/first.json": _make_answer_payload(
                answer_id="a-shared", query="first"
            ),
            "verified-answers/dup.json": _make_answer_payload(
                answer_id="a-shared", query="duplicate"
            ),
            "verified-answers/other.json": _make_answer_payload(
                answer_id="a-other", query="other"
            ),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return files.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["checked"] == 3
        assert sorted(result["created"]) == ["a-other", "a-shared"]
        assert result["skipped_duplicate_answer_id"] == ["verified-answers/dup.json"]
        # And exactly ONE local entry per answer_id.
        verifieds = get_verified_answers()
        ids = [v.answer_id for v in verifieds]
        assert ids.count("a-shared") == 1
        assert set(ids) == {"a-shared", "a-other"}

    async def test_local_only_entries_reported_as_orphaned_not_deleted(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        seeded = _seed_local_answer(answer_id="a-local-only")

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return []

        async def _fetch(path: str) -> dict[str, Any] | None:
            return None

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["checked"] == 0
        assert result["orphaned_locally"] == [seeded.answer_id]
        # Local entry survives.
        assert get_verified_answer(seeded.answer_id) is not None

    async def test_fetch_failure_reported_others_succeed(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return [
                {"name": "ok.json", "path": "verified-answers/ok.json", "sha": "1"},
                {"name": "broken.json", "path": "verified-answers/broken.json", "sha": "2"},
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            if path == "verified-answers/broken.json":
                return None
            return _make_answer_payload(answer_id="ok")

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_answers_from_github()
        assert result["created"] == ["ok"]
        assert result["failed"] == ["verified-answers/broken.json"]

    async def test_returns_skipped_when_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, enabled=False)
        result = await bootstrap_answers_from_github()
        assert result["skipped"] is True

    async def test_strict_listing_failure_propagates(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _list_fail(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            assert strict is True
            raise github_publisher.GitHubPublishError(
                "GitHub returned 401 listing verified-answers"
            )

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list_fail)
        with pytest.raises(github_publisher.GitHubPublishError) as exc:
            await bootstrap_answers_from_github()
        assert "401" in str(exc.value)


# ---------------------------------------------------------------------------
# sync_one_answer_from_github
# ---------------------------------------------------------------------------


class TestSyncOneAnswer:
    async def test_creates_when_no_local_match(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_answer_payload(answer_id="a-new", query="q", answer="A")

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_answer_from_github("verified-answers/a-new.json")
        assert result["status"] == "created"
        assert result["answer_id"] == "a-new"
        a = get_verified_answer("a-new")
        assert a is not None
        assert a.query == "q"
        assert a.github_path == "verified-answers/a-new.json"

    async def test_updates_existing_preserves_operational_state(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        seeded = _seed_local_answer(answer_id="a-up", query="old", confidence=0.5)
        verified_notebooks.increment_answer_usage(seeded.answer_id)
        idx = verified_notebooks._load_index()
        idx["verified_answers"][seeded.answer_id]["admin_notes"] = "kept"
        verified_notebooks._save_index(idx)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_answer_payload(
                answer_id=seeded.answer_id, query="new", confidence=0.95
            )

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_answer_from_github(
            f"verified-answers/{seeded.answer_id}.json"
        )
        assert result["status"] == "updated"
        a = get_verified_answer(seeded.answer_id)
        assert a is not None
        assert a.query == "new"
        assert a.confidence == 0.95
        assert a.usage_count == 1
        assert a.admin_notes == "kept"

    async def test_idempotent_double_call(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_answer_payload(answer_id="a-idem")

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        r1 = await sync_one_answer_from_github("verified-answers/a-idem.json")
        assert r1["status"] == "created"
        r2 = await sync_one_answer_from_github("verified-answers/a-idem.json")
        assert r2["status"] == "updated"
        assert r2["answer_id"] == r1["answer_id"]
        assert [a.answer_id for a in get_verified_answers()] == ["a-idem"]

    async def test_disabled_returns_skipped(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, enabled=False)
        result = await sync_one_answer_from_github("verified-answers/x.json")
        assert result == {"status": "skipped_disabled", "path": "verified-answers/x.json"}

    async def test_fetch_failure_returns_failed(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return None

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_answer_from_github("verified-answers/missing.json")
        assert result == {"status": "failed", "path": "verified-answers/missing.json"}

    async def test_shape_corrupt_routes_to_bad_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        cases: list[tuple[str, Any, str]] = [
            (
                "verified-answers/a-not-dict.json",
                ["not", "a", "dict"],
                "not a JSON object",
            ),
            (
                "verified-answers/b-no-id.json",
                {"query": "no id"},
                "missing answer_id",
            ),
            (
                "verified-answers/c-id-not-str.json",
                {"answer_id": ["unhashable"], "query": "q", "answer": "a"},
                "answer_id is not a string",
            ),
        ]
        for path, payload, fragment in cases:
            async def _fetch(p: str, _payload: Any = payload) -> Any:
                return _payload
            monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
            result = await sync_one_answer_from_github(path)
            assert result["status"] == "skipped_bad_metadata"
            assert fragment in result["reason"]

    async def test_type_corrupt_on_update_preserves_existing(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        seeded = _seed_local_answer(answer_id="a-keep", query="Good", confidence=0.85)
        seeded_verified_by = seeded.verified_by

        bad = _make_answer_payload(answer_id=seeded.answer_id, query="bad", confidence=0.7)
        bad["verified_by"] = ["list", "not", "str"]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return bad

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_answer_from_github(
            f"verified-answers/{seeded.answer_id}.json"
        )
        assert result["status"] == "skipped_bad_metadata"

        unchanged = get_verified_answer(seeded.answer_id)
        assert unchanged is not None
        assert unchanged.query == seeded.query
        assert unchanged.verified_by == seeded_verified_by


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


class TestBootstrapAnswersEndpoint:
    async def test_400_when_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            bootstrap_verified_answers_from_github,
        )

        _patch_settings(monkeypatch, enabled=False)
        with pytest.raises(HTTPException) as exc:
            await bootstrap_verified_answers_from_github(_admin={"user": "admin"})
        assert exc.value.status_code == 400

    async def test_502_when_listing_fails(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            bootstrap_verified_answers_from_github,
        )

        _patch_settings(monkeypatch)

        async def _list_fail(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            raise github_publisher.GitHubPublishError("GitHub 401")

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list_fail)
        with pytest.raises(HTTPException) as exc:
            await bootstrap_verified_answers_from_github(_admin={"user": "admin"})
        assert exc.value.status_code == 502
        assert "401" in str(exc.value.detail)

    async def test_success_returns_report(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            bootstrap_verified_answers_from_github,
        )

        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False, suffix: str = ".ipynb") -> list[dict[str, Any]]:
            return [{"name": "n.json", "path": "verified-answers/n.json", "sha": "s"}]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_answer_payload(answer_id="n")

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        body = await bootstrap_verified_answers_from_github(_admin={"user": "admin"})
        assert body["checked"] == 1
        assert body["created"] == ["n"]
        assert "Bootstrapped 1 verified answer" in body["message"]


class TestSyncOneAnswerEndpoint:
    async def test_400_when_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            _SyncOnePayload,
            sync_one_verified_answer_from_github,
        )

        _patch_settings(monkeypatch, enabled=False)
        with pytest.raises(HTTPException) as exc:
            await sync_one_verified_answer_from_github(
                payload=_SyncOnePayload(path="verified-answers/x.json"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 400

    async def test_502_when_fetch_fails(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            _SyncOnePayload,
            sync_one_verified_answer_from_github,
        )

        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return None

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        with pytest.raises(HTTPException) as exc:
            await sync_one_verified_answer_from_github(
                payload=_SyncOnePayload(path="verified-answers/gone.json"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 502

    async def test_success(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            _SyncOnePayload,
            sync_one_verified_answer_from_github,
        )

        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_answer_payload(answer_id="e2e")

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        body = await sync_one_verified_answer_from_github(
            payload=_SyncOnePayload(path="verified-answers/e2e.json"),
            _admin={"user": "admin"},
        )
        assert body["status"] == "created"
        assert body["answer_id"] == "e2e"


class TestListFolderFilesSuffixFilter:
    """an earlier adversarial review: ``_list_folder_files()`` filters by file extension,
    and the default is ``.ipynb``. Bootstrap of verified answers (which
    publishes ``<id>.json``) must pass ``suffix=".json"`` or every real
    answer file gets filtered out and the report shows ``checked: 0``.

    The other answer-bootstrap tests monkeypatch ``_list_folder_files``
    with a hardcoded list, which bypasses the filter entirely. These
    tests go through the REAL helper via respx so the suffix-filter
    contract is actually exercised.
    """

    @respx.mock
    async def test_default_suffix_filters_to_ipynb(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        respx.get(
            "https://api.github.com/repos/owner/repo/contents/verified"
        ).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"type": "file", "name": "nb.ipynb",
                     "path": "verified/nb.ipynb", "sha": "1"},
                    {"type": "file", "name": "readme.md",
                     "path": "verified/readme.md", "sha": "2"},
                    {"type": "file", "name": "ans.json",
                     "path": "verified/ans.json", "sha": "3"},
                ],
            )
        )

        result = await github_publisher._list_folder_files("verified")
        names = [r["name"] for r in result]
        assert names == ["nb.ipynb"], (
            "Default suffix must filter to .ipynb (backwards-compat for "
            "the notebook callers)."
        )

    @respx.mock
    async def test_json_suffix_filters_to_json(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        respx.get(
            "https://api.github.com/repos/owner/repo/contents/verified-answers"
        ).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"type": "file", "name": "a-1.json",
                     "path": "verified-answers/a-1.json", "sha": "1"},
                    {"type": "file", "name": "a-2.json",
                     "path": "verified-answers/a-2.json", "sha": "2"},
                    {"type": "file", "name": "stray.ipynb",
                     "path": "verified-answers/stray.ipynb", "sha": "3"},
                    {"type": "file", "name": "readme.md",
                     "path": "verified-answers/readme.md", "sha": "4"},
                ],
            )
        )

        result = await github_publisher._list_folder_files(
            "verified-answers", suffix=".json"
        )
        names = sorted(r["name"] for r in result)
        assert names == ["a-1.json", "a-2.json"], (
            "suffix='.json' must include .json files and exclude others — "
            "this is the regression for an earlier adversarial review where the default "
            ".ipynb filter dropped every real answer file."
        )

    @respx.mock
    async def test_bootstrap_answers_end_to_end_with_real_lister(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end proof: bootstrap_answers_from_github goes through
        the REAL _list_folder_files and successfully discovers .json
        files. Before the fix, the default .ipynb filter would have
        produced checked: 0 here."""
        import base64
        import json

        _patch_settings(monkeypatch)

        payload = _make_answer_payload(answer_id="real-1", query="real")
        respx.get(
            "https://api.github.com/repos/owner/repo/contents/verified-answers"
        ).mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"type": "file", "name": "real-1.json",
                     "path": "verified-answers/real-1.json", "sha": "s1"},
                ],
            )
        )
        respx.get(
            "https://api.github.com/repos/owner/repo/contents/verified-answers/real-1.json"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "type": "file",
                    "encoding": "base64",
                    "content": base64.b64encode(
                        json.dumps(payload).encode("utf-8")
                    ).decode("ascii"),
                },
            )
        )

        result = await bootstrap_answers_from_github()
        assert result["checked"] == 1, (
            "checked should be 1; if it's 0 the suffix filter is wrong "
            "(an earlier adversarial review regression)."
        )
        assert result["created"] == ["real-1"]


class TestRouteRegistration:
    """Regression for an earlier adversarial review-style accidents: assert both new
    endpoints are wired into FastAPI's route table."""

    def test_both_routes_are_registered(self) -> None:
        from data_concierge.gateway.router import router

        post_paths = {
            r.path for r in router.routes
            if getattr(r, "methods", None) and "POST" in r.methods
        }
        assert "/api/v1/verified-answers/bootstrap-from-github" in post_paths
        assert "/api/v1/verified-answers/sync-one-from-github" in post_paths
