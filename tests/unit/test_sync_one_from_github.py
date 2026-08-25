"""Tests for per-path GitHub sync (issue #46, step 6).

``sync_one_from_github(path)`` is the per-path counterpart to
``bootstrap_index_from_github()``. The webhook handler in step 7 will
call this for each file in a push event's added/modified list.

Mirrors the bootstrap test coverage but for a single path:

* No matching local entry  → ``status="created"``, new VerifiedNotebook
  built from embedded metadata.
* Matching local entry  → ``status="updated"``, refreshes notebook_json
  + GitHub coords + metadata-derived fields, preserves operational state
  (usage_count / admin_notes / tags / keywords).
* Shape-malformed notebook (top-level not dict, metadata not dict, etc.)
  → ``status="skipped_bad_metadata"`` (NOT a crash).
* Missing metadata namespace  → ``status="skipped_no_metadata"``.
* Numeric ``confidence`` or pydantic type validation fails on the
  candidate  → ``status="skipped_bad_metadata"`` with existing local
  entry UNTOUCHED (bad republish must not silently overwrite a good
  cached entry).
* GitHub disabled  → ``status="skipped_disabled"``.
* GitHub fetch returned None  → ``status="failed"``.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.gateway import github_publisher, verified_notebooks
from data_concierge.gateway.verified_notebooks import (
    _VERIFIED_PREFIX,
    get_verified_notebook,
    get_verified_notebooks,
    sync_one_from_github,
)


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    monkeypatch.setattr(github_publisher, "storage", test_storage)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, enabled: bool = True) -> None:
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
        },
    )


def _make_github_notebook(
    *,
    submission_id: str,
    query: str,
    confidence: float = 0.9,
    verified_by: str = "admin",
    verified_at: str = "2026-05-27T18:30:00Z",
    data_source: str = "bls",
    omit_metadata: bool = False,
) -> dict[str, Any]:
    """Construct a notebook_json shaped like what step 4 publishes."""
    nb: dict[str, Any] = {
        "cells": [
            {"cell_type": "markdown", "source": ["# " + query], "metadata": {}}
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    if not omit_metadata:
        nb["metadata"]["data_concierge"] = {
            "version": "0.1.0",
            "generated": "2026-05-27T10:00:00",
            "query": query,
            "data_source": data_source,
            "colab_compatible": True,
            "submission_id": submission_id,
            "confidence": confidence,
            "verified_by": verified_by,
            "verified_at": verified_at,
        }
    return nb


class TestSyncOneCreatesNewEntry:
    async def test_creates_entry_when_no_local_match(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        nb = _make_github_notebook(
            submission_id="sub-new",
            query="What is the unemployment rate in Texas?",
            confidence=0.91,
            verified_by="admin",
            verified_at="2026-05-27T18:30:00Z",
            data_source="bls",
        )

        async def _fetch(path: str) -> dict[str, Any] | None:
            assert path == "verified/new.ipynb"
            return nb

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await sync_one_from_github("verified/new.ipynb")
        assert result["status"] == "created"
        assert result["path"] == "verified/new.ipynb"
        notebook_id = result["notebook_id"]

        created = get_verified_notebook(notebook_id)
        assert created is not None
        assert created.submission_id == "sub-new"
        assert created.query == "What is the unemployment rate in Texas?"
        assert created.confidence == 0.91
        assert created.verified_by == "admin"
        assert created.verified_at == "2026-05-27T18:30:00Z"
        assert created.data_source == "bls"
        assert created.github_path == "verified/new.ipynb"
        assert created.github_synced_at and created.github_synced_at.endswith("Z")
        # filename derived from path basename.
        assert created.filename == "new.ipynb"

    async def test_filename_from_path_with_no_slash(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paths without a slash (rare but possible) still produce a
        sensible filename rather than the empty string."""
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_github_notebook(submission_id="sub-flat", query="q")

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await sync_one_from_github("flat.ipynb")
        assert result["status"] == "created"
        nb = get_verified_notebook(result["notebook_id"])
        assert nb is not None
        assert nb.filename == "flat.ipynb"


class TestSyncOneUpdatesExisting:
    async def test_updates_existing_entry_preserves_operational_state(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Update path mirrors bootstrap: refresh notebook_json + GitHub
        coords + metadata-derived fields; preserve usage_count /
        admin_notes / tags / keywords."""
        _patch_settings(monkeypatch)

        # Seed a local entry, then add operational state to it.
        seed_sub = verified_notebooks.submit_notebook(
            query="Pre-existing query",
            answer="answer",
            notebook_json=_make_github_notebook(
                submission_id="sub-pre", query="Pre-existing query",
                confidence=0.5,
            ),
            submitted_by="tester",
            data_source="bls",
            confidence=0.5,
        )
        existing = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id, reviewed_by="admin"
        )
        assert existing is not None
        existing_nb_id = existing.notebook_id
        verified_notebooks.increment_usage(existing_nb_id)
        verified_notebooks.increment_usage(existing_nb_id)
        idx = verified_notebooks._load_index()
        idx["verified"][existing_nb_id]["admin_notes"] = "reviewed by Joel"
        idx["verified"][existing_nb_id]["tags"] = ["finance", "labor"]
        verified_notebooks._save_index(idx)

        fresh = _make_github_notebook(
            submission_id=seed_sub.submission_id,
            query="Refined query",
            confidence=0.99,
            verified_by="admin2",
            verified_at="2026-05-27T20:00:00Z",
            data_source="census",
        )

        async def _fetch(path: str) -> dict[str, Any] | None:
            return fresh

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await sync_one_from_github("verified/pre.ipynb")
        assert result["status"] == "updated"
        assert result["notebook_id"] == existing_nb_id

        refreshed = get_verified_notebook(existing_nb_id)
        assert refreshed is not None
        # Operational state PRESERVED.
        assert refreshed.usage_count == 2
        assert refreshed.admin_notes == "reviewed by Joel"
        assert refreshed.tags == ["finance", "labor"]
        # GitHub-owned + metadata-derived fields REFRESHED.
        assert refreshed.query == "Refined query"
        assert refreshed.confidence == 0.99
        assert refreshed.verified_by == "admin2"
        assert refreshed.verified_at == "2026-05-27T20:00:00Z"
        assert refreshed.data_source == "census"
        assert refreshed.github_path == "verified/pre.ipynb"
        assert refreshed.github_synced_at and refreshed.github_synced_at.endswith("Z")

    async def test_idempotent_double_call_produces_same_state(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling sync_one twice with the same GitHub file should be
        a no-op the second time — first call creates, second updates,
        but ends up with exactly one local entry."""
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_github_notebook(submission_id="sub-idem", query="q")

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        r1 = await sync_one_from_github("verified/idem.ipynb")
        assert r1["status"] == "created"
        r2 = await sync_one_from_github("verified/idem.ipynb")
        assert r2["status"] == "updated"
        assert r2["notebook_id"] == r1["notebook_id"]

        # Exactly one local entry.
        verifieds = get_verified_notebooks()
        assert [v.submission_id for v in verifieds] == ["sub-idem"]


class TestSyncOneSkips:
    async def test_disabled_returns_skipped(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, enabled=False)
        result = await sync_one_from_github("verified/x.ipynb")
        assert result == {"status": "skipped_disabled", "path": "verified/x.ipynb"}

    async def test_fetch_failure_returns_failed(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return None

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_from_github("verified/missing.ipynb")
        assert result == {"status": "failed", "path": "verified/missing.ipynb"}

    async def test_no_metadata_namespace_routes_to_skipped_no_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_github_notebook(
                submission_id="ignored", query="q", omit_metadata=True
            )

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_from_github("verified/legacy.ipynb")
        assert result["status"] == "skipped_no_metadata"
        assert result["path"] == "verified/legacy.ipynb"
        # And no local entry was created.
        assert get_verified_notebooks() == []

    async def test_shape_malformed_routes_to_skipped_bad_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All the same shape corruption cases the bootstrap guards
        against: top-level non-dict, non-dict metadata, non-dict
        data_concierge, unhashable submission_id."""
        _patch_settings(monkeypatch)

        cases: list[tuple[str, Any, str]] = [
            (
                "verified/a-top-not-dict.ipynb",
                ["not", "a", "dict"],
                "top-level",
            ),
            (
                "verified/b-metadata-not-dict.ipynb",
                {"cells": [], "metadata": "string", "nbformat": 4},
                "metadata is not",
            ),
            (
                "verified/c-dc-not-dict.ipynb",
                {
                    "cells": [],
                    "metadata": {"data_concierge": ["wrong"]},
                    "nbformat": 4,
                },
                "data_concierge is not",
            ),
            (
                "verified/d-sub-id-not-str.ipynb",
                {
                    "cells": [],
                    "metadata": {
                        "data_concierge": {
                            "submission_id": ["a", "b"],
                            "query": "q",
                            "confidence": 0.5,
                        }
                    },
                    "nbformat": 4,
                },
                "submission_id is not",
            ),
        ]

        for path, payload, reason_fragment in cases:
            async def _fetch(p: str, _payload: Any = payload) -> Any:
                return _payload

            monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
            result = await sync_one_from_github(path)
            assert result["status"] == "skipped_bad_metadata", path
            assert result["path"] == path
            assert reason_fragment in result["reason"], (
                f"path={path!r}: reason {result['reason']!r} missing"
                f" fragment {reason_fragment!r}"
            )

        # None of the bad cases created a local entry.
        assert get_verified_notebooks() == []

    async def test_bad_confidence_on_create_routes_to_skipped(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        bad = _make_github_notebook(submission_id="sub-bad", query="q")
        bad["metadata"]["data_concierge"]["confidence"] = "very-high"

        async def _fetch(path: str) -> dict[str, Any] | None:
            return bad

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_from_github("verified/bad.ipynb")
        assert result["status"] == "skipped_bad_metadata"
        assert "confidence" in result["reason"]
        assert get_verified_notebooks() == []

    async def test_bad_metadata_on_update_preserves_existing(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critical property mirroring the bootstrap update path: a
        type-corrupt field (verified_by as list) on an EXISTING entry
        must route to skipped_bad_metadata and leave the local entry
        UNTOUCHED — a bad republish must not silently overwrite a good
        cached entry."""
        _patch_settings(monkeypatch)

        seed_sub = verified_notebooks.submit_notebook(
            query="Good query",
            answer="answer",
            notebook_json=_make_github_notebook(
                submission_id="sub-good", query="Good query", confidence=0.85,
            ),
            submitted_by="tester",
            data_source="bls",
            confidence=0.85,
        )
        existing = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id, reviewed_by="admin"
        )
        assert existing is not None
        seeded_query = existing.query
        seeded_confidence = existing.confidence
        seeded_verified_by = existing.verified_by

        bad = _make_github_notebook(
            submission_id=seed_sub.submission_id,
            query="Should NOT be applied",
            confidence=0.7,
        )
        bad["metadata"]["data_concierge"]["verified_by"] = ["not", "a", "str"]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return bad

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_from_github("verified/good.ipynb")
        assert result["status"] == "skipped_bad_metadata"

        unchanged = get_verified_notebook(existing.notebook_id)
        assert unchanged is not None
        assert unchanged.query == seeded_query
        assert unchanged.confidence == seeded_confidence
        assert unchanged.verified_by == seeded_verified_by
        # And listing must work, not raise ValidationError.
        listed = get_verified_notebooks()
        assert any(v.notebook_id == existing.notebook_id for v in listed)


class TestSyncOneBlob:
    """The on-disk blob (``verified/<id>.ipynb``) is what /download
    serves. Both create and update must refresh it so the user sees
    the latest content."""

    async def test_create_writes_blob(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)
        nb = _make_github_notebook(submission_id="sub-blob", query="q")

        async def _fetch(path: str) -> dict[str, Any] | None:
            return nb

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_from_github("verified/blob.ipynb")
        assert result["status"] == "created"

        blob_key = f"{_VERIFIED_PREFIX}/{result['notebook_id']}.ipynb"
        stored = verified_notebooks.storage.read_json(blob_key)
        assert stored is not None
        assert stored["metadata"]["data_concierge"]["submission_id"] == "sub-blob"

    async def test_update_refreshes_blob(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        # Seed an entry with an OLD notebook_json.
        old_nb = _make_github_notebook(
            submission_id="sub-refresh", query="old", confidence=0.5,
        )
        seed_sub = verified_notebooks.submit_notebook(
            query="old",
            answer="a",
            notebook_json=old_nb,
            submitted_by="tester",
            data_source="bls",
            confidence=0.5,
        )
        existing = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id, reviewed_by="admin"
        )
        assert existing is not None

        fresh_nb = _make_github_notebook(
            submission_id=seed_sub.submission_id,
            query="new",
            confidence=0.95,
        )
        # Add a distinctive marker cell to detect the refresh.
        fresh_nb["cells"].append(
            {"cell_type": "code", "source": ["print('fresh')"], "metadata": {}}
        )

        async def _fetch(path: str) -> dict[str, Any] | None:
            return fresh_nb

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        result = await sync_one_from_github("verified/refresh.ipynb")
        assert result["status"] == "updated"

        blob_key = f"{_VERIFIED_PREFIX}/{existing.notebook_id}.ipynb"
        stored = verified_notebooks.storage.read_json(blob_key)
        assert stored is not None
        # The fresh marker is present.
        sources = [
            "".join(c.get("source", [])) for c in stored.get("cells", [])
        ]
        assert any("fresh" in s for s in sources), (
            "blob was not refreshed — /download would still serve the old content"
        )


class TestSyncOneEndpoint:
    """End-to-end smoke for POST /verified-notebooks/sync-one-from-github."""

    async def test_endpoint_returns_400_when_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            _SyncOnePayload,
            sync_one_verified_from_github,
        )

        _patch_settings(monkeypatch, enabled=False)
        with pytest.raises(HTTPException) as exc:
            await sync_one_verified_from_github(
                payload=_SyncOnePayload(path="verified/x.ipynb"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 400
        assert "GitHub publishing is not enabled" in str(exc.value.detail)

    async def test_endpoint_returns_502_when_fetch_fails(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            _SyncOnePayload,
            sync_one_verified_from_github,
        )

        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return None

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        with pytest.raises(HTTPException) as exc:
            await sync_one_verified_from_github(
                payload=_SyncOnePayload(path="verified/gone.ipynb"),
                _admin={"user": "admin"},
            )
        assert exc.value.status_code == 502
        assert "verified/gone.ipynb" in str(exc.value.detail)

    async def test_endpoint_success_returns_result(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            _SyncOnePayload,
            sync_one_verified_from_github,
        )

        _patch_settings(monkeypatch)

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_github_notebook(submission_id="sub-e2e", query="q")

        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)
        body = await sync_one_verified_from_github(
            payload=_SyncOnePayload(path="verified/e2e.ipynb"),
            _admin={"user": "admin"},
        )
        assert body["status"] == "created"
        assert body["path"] == "verified/e2e.ipynb"
        assert "notebook_id" in body
