"""Tests for the disaster-recovery rebuild (issue #46, step 5).

``bootstrap_index_from_github()`` is the disaster-recovery counterpart to
``sync_all_from_github()``. Where sync iterates the LOCAL index and
refreshes each entry (a no-op when local is empty), bootstrap iterates
the GITHUB ``verified/`` folder and reconstructs the local index from
each notebook's embedded ``metadata.data_concierge`` provenance — the
fields step 4 populates at approval time.

Covers:

* Empty local index + GitHub has notebooks  → new entries created from
  embedded metadata.
* Local entry already exists for the same submission_id → refresh
  notebook_json / github_path / github_synced_at but PRESERVE
  usage_count / admin_notes / tags / keywords.
* Notebook on GitHub with no metadata.data_concierge  → skipped,
  reported for triage (not crashed).
* Fetch failure on one path  → reported, others still process.
* Local entry with no GitHub counterpart  → reported as orphaned
  (NOT deleted).
* Strict-listing failure  → propagates GitHubPublishError so the
  endpoint can return 502.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.gateway import github_publisher, verified_notebooks
from data_concierge.gateway.verified_notebooks import (
    _VERIFIED_PREFIX,
    bootstrap_index_from_github,
    get_verified_notebook,
    get_verified_notebooks,
)


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)
    # github_publisher also reaches for storage (for github_settings.json).
    monkeypatch.setattr(github_publisher, "storage", test_storage)


def _patch_settings(monkeypatch: pytest.MonkeyPatch, enabled: bool = True) -> None:
    """Patch ``load_github_settings`` to return a test config.

    ``enabled`` is the legacy param name kept for callsite compatibility:
    True → token + repo set, paused=False (publishing active);
    False → empty token, paused=True (publishing inactive). The
    underlying schema uses the new "configured + not paused" model
    introduced in commit a0e3e6f.
    """
    monkeypatch.setattr(
        github_publisher,
        "load_github_settings",
        lambda: {
            "paused": not enabled,
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


class TestBootstrapFromEmptyLocalIndex:
    async def test_creates_entries_from_embedded_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            assert folder == "verified"
            return [
                {
                    "name": "tx-unemployment_abcd1234.ipynb",
                    "path": "verified/tx-unemployment_abcd1234.ipynb",
                    "sha": "sha-a",
                },
                {
                    "name": "ca-population_efgh5678.ipynb",
                    "path": "verified/ca-population_efgh5678.ipynb",
                    "sha": "sha-b",
                },
            ]

        notebooks = {
            "verified/tx-unemployment_abcd1234.ipynb": _make_github_notebook(
                submission_id="sub-tx",
                query="Texas unemployment rate?",
                confidence=0.92,
                data_source="bls",
            ),
            "verified/ca-population_efgh5678.ipynb": _make_github_notebook(
                submission_id="sub-ca",
                query="California population?",
                confidence=0.88,
                data_source="census",
            ),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return notebooks.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["skipped"] is False
        assert result["checked"] == 2
        assert len(result["created"]) == 2
        assert result["updated"] == []
        assert result["failed"] == []
        assert result["skipped_no_metadata"] == []
        assert result["orphaned_locally"] == []

        # Both VerifiedNotebook entries exist with reconstructed fields.
        verifieds = get_verified_notebooks()
        by_sub = {v.submission_id: v for v in verifieds}
        assert set(by_sub.keys()) == {"sub-tx", "sub-ca"}
        tx = by_sub["sub-tx"]
        assert tx.query == "Texas unemployment rate?"
        assert tx.verified_by == "admin"
        assert tx.confidence == 0.92
        assert tx.data_source == "bls"
        assert tx.github_path == "verified/tx-unemployment_abcd1234.ipynb"
        assert tx.github_synced_at and tx.github_synced_at.endswith("Z")
        # The on-disk blob got seeded for the new entry.
        blob_key = f"{_VERIFIED_PREFIX}/{tx.notebook_id}.ipynb"
        blob = verified_notebooks.storage.read_json(blob_key)
        assert blob is not None
        assert blob["metadata"]["data_concierge"]["submission_id"] == "sub-tx"


class TestBootstrapPreservesLocalOperationalState:
    async def test_existing_entry_keeps_usage_count_and_admin_notes(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critical correctness property: bootstrap must NOT clobber
        local-only operational metadata for entries that already exist,
        but MUST refresh GitHub-owned metadata fields so admin edits in
        GitHub propagate to the local index."""
        _patch_settings(monkeypatch)

        # Seed a pre-existing local entry with operational state.
        seed_sub = verified_notebooks.submit_notebook(
            query="Pre-existing query",
            answer="answer",
            notebook_json=_make_github_notebook(
                submission_id="sub-pre",
                query="Pre-existing query",
                confidence=0.5,  # old confidence
                verified_at="2026-01-01T00:00:00Z",  # old verified_at
            ),
            submitted_by="tester",
            data_source="bls",
            confidence=0.5,
        )
        existing = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id,
            reviewed_by="admin",
        )
        assert existing is not None
        existing_nb_id = existing.notebook_id

        # Bump usage_count and add admin_notes (local-only ops state).
        verified_notebooks.increment_usage(existing_nb_id)
        verified_notebooks.increment_usage(existing_nb_id)
        idx = verified_notebooks._load_index()
        idx["verified"][existing_nb_id]["admin_notes"] = "reviewed by Joel"
        idx["verified"][existing_nb_id]["tags"] = ["finance", "labor"]
        verified_notebooks._save_index(idx)

        # Now simulate GitHub having a FRESHER version of the same notebook:
        # admin re-published with updated query text, confidence, verified_by,
        # verified_at, and data_source.
        fresh_notebook = _make_github_notebook(
            submission_id=seed_sub.submission_id,
            query="Refined pre-existing query",
            confidence=0.99,  # updated
            verified_by="admin2",  # different reviewer
            verified_at="2026-05-27T18:30:00Z",  # newer
            data_source="census",  # admin retargeted the data source
        )

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {
                    "name": "pre.ipynb",
                    "path": "verified/pre.ipynb",
                    "sha": "fresh",
                }
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return fresh_notebook

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["updated"] == [existing_nb_id]
        assert result["created"] == []

        refreshed = get_verified_notebook(existing_nb_id)
        assert refreshed is not None
        # Operational state PRESERVED.
        assert refreshed.usage_count == 2, "usage_count must survive bootstrap"
        assert refreshed.admin_notes == "reviewed by Joel"
        assert refreshed.tags == ["finance", "labor"]
        # GitHub-owned content REFRESHED — both in the embedded blob and in
        # the index-level fields that get_verified_notebooks/search read from.
        assert (
            refreshed.notebook_json["metadata"]["data_concierge"]["confidence"]
            == 0.99
        )
        assert refreshed.github_path == "verified/pre.ipynb"
        assert refreshed.github_synced_at and refreshed.github_synced_at.endswith("Z")
        # Metadata-derived index fields refreshed from embedded metadata so
        # admin edits in GitHub aren't invisible to search/listing.
        assert refreshed.query == "Refined pre-existing query"
        assert refreshed.confidence == 0.99
        assert refreshed.verified_by == "admin2"
        assert refreshed.verified_at == "2026-05-27T18:30:00Z"
        assert refreshed.data_source == "census"


class TestBootstrapEdgeCases:
    async def test_notebook_without_metadata_is_skipped_not_crashed(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An admin uploaded a raw .ipynb to verified/ before this provenance
        scheme existed. We can't rebuild from nothing — report it for
        manual triage instead of crashing or guessing."""
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {
                    "name": "good.ipynb",
                    "path": "verified/good.ipynb",
                    "sha": "1",
                },
                {
                    "name": "legacy.ipynb",
                    "path": "verified/legacy.ipynb",
                    "sha": "2",
                },
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            if path == "verified/good.ipynb":
                return _make_github_notebook(
                    submission_id="sub-good", query="good"
                )
            return _make_github_notebook(
                submission_id="ignored", query="legacy",
                omit_metadata=True,
            )

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["checked"] == 2
        assert len(result["created"]) == 1
        assert result["skipped_no_metadata"] == ["verified/legacy.ipynb"]
        assert result["failed"] == []

    async def test_fetch_failure_on_one_path_is_reported_others_succeed(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "ok.ipynb", "path": "verified/ok.ipynb", "sha": "1"},
                {"name": "broken.ipynb", "path": "verified/broken.ipynb",
                 "sha": "2"},
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            if path == "verified/broken.ipynb":
                return None  # GitHub fetch failed
            return _make_github_notebook(submission_id="sub-ok", query="ok")

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert len(result["created"]) == 1
        assert result["failed"] == ["verified/broken.ipynb"]

    async def test_local_only_entries_are_reported_as_orphaned_not_deleted(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A VerifiedNotebook exists locally but its file is missing from
        GitHub (e.g. an admin deleted it manually, or it was never pushed).
        Bootstrap must NOT delete the local entry — admins decide."""
        _patch_settings(monkeypatch)

        # Seed a local-only verified notebook.
        seed_sub = verified_notebooks.submit_notebook(
            query="local-only",
            answer="x",
            notebook_json={"cells": [], "metadata": {}, "nbformat": 4,
                          "nbformat_minor": 5},
            submitted_by="tester",
            data_source="bls",
            confidence=0.5,
        )
        local_only = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id, reviewed_by="admin"
        )
        assert local_only is not None

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return []  # GitHub has nothing for this folder

        async def _fetch(path: str) -> dict[str, Any] | None:
            return None

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["checked"] == 0
        assert result["created"] == []
        assert result["updated"] == []
        assert result["orphaned_locally"] == [local_only.notebook_id]
        # Local entry still exists — bootstrap is non-destructive.
        still_there = get_verified_notebook(local_only.notebook_id)
        assert still_there is not None

    async def test_malformed_metadata_does_not_crash_whole_bootstrap(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an earlier adversarial review: a single bad notebook (non-numeric confidence, wrong
        field types) used to crash the whole bootstrap with a 500, throwing
        away progress on every later file. Bad metadata must now route to
        skipped_bad_metadata and the loop must keep going."""
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "a-bad-confidence.ipynb",
                 "path": "verified/a-bad-confidence.ipynb", "sha": "1"},
                {"name": "b-bad-types.ipynb",
                 "path": "verified/b-bad-types.ipynb", "sha": "2"},
                {"name": "c-good.ipynb",
                 "path": "verified/c-good.ipynb", "sha": "3"},
            ]

        def _bad_confidence_notebook() -> dict[str, Any]:
            nb = _make_github_notebook(submission_id="sub-a", query="q-a")
            # confidence is a string that can't be parsed as float.
            nb["metadata"]["data_concierge"]["confidence"] = "very-high"
            return nb

        def _bad_types_notebook() -> dict[str, Any]:
            nb = _make_github_notebook(submission_id="sub-b", query="q-b")
            # verified_by must be a string per the Pydantic model. A list
            # triggers a ValidationError in the VerifiedNotebook constructor.
            nb["metadata"]["data_concierge"]["verified_by"] = ["not", "a", "string"]
            return nb

        notebooks = {
            "verified/a-bad-confidence.ipynb": _bad_confidence_notebook(),
            "verified/b-bad-types.ipynb": _bad_types_notebook(),
            "verified/c-good.ipynb": _make_github_notebook(
                submission_id="sub-c", query="q-c"
            ),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return notebooks.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        # Critical: this MUST NOT raise — the whole point is that one bad
        # file doesn't take down the bootstrap.
        result = await bootstrap_index_from_github()
        assert result["checked"] == 3
        # Good file still got reconstructed.
        assert len(result["created"]) == 1
        # Both bad files are reported in the dedicated bucket.
        assert set(result["skipped_bad_metadata"]) == {
            "verified/a-bad-confidence.ipynb",
            "verified/b-bad-types.ipynb",
        }
        # And not in any other bucket — keep the report unambiguous.
        assert "verified/a-bad-confidence.ipynb" not in result["failed"]
        assert "verified/b-bad-types.ipynb" not in result["skipped_no_metadata"]

        # The good notebook is actually queryable.
        verifieds = get_verified_notebooks()
        assert [v.submission_id for v in verifieds] == ["sub-c"]

    async def test_malformed_shapes_before_pydantic_do_not_crash(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an earlier adversarial review: malformed metadata SHAPES (non-dict metadata,
        non-dict data_concierge, unhashable submission_id, top-level non-dict
        notebook_json) used to crash the bootstrap BEFORE reaching the
        VerifiedNotebook try/except. The shape guards must route every one
        of those into skipped_bad_metadata and keep going."""
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "a-notebook-not-dict.ipynb",
                 "path": "verified/a-notebook-not-dict.ipynb", "sha": "1"},
                {"name": "b-metadata-not-dict.ipynb",
                 "path": "verified/b-metadata-not-dict.ipynb", "sha": "2"},
                {"name": "c-data-concierge-not-dict.ipynb",
                 "path": "verified/c-data-concierge-not-dict.ipynb", "sha": "3"},
                {"name": "d-sub-id-unhashable.ipynb",
                 "path": "verified/d-sub-id-unhashable.ipynb", "sha": "4"},
                {"name": "e-good.ipynb",
                 "path": "verified/e-good.ipynb", "sha": "5"},
            ]

        notebooks: dict[str, Any] = {
            # Top-level isn't a dict — e.g. a corrupted .ipynb that parsed
            # as a list. ``notebook_json.get("metadata")`` used to AttributeError.
            "verified/a-notebook-not-dict.ipynb": ["not", "a", "dict"],
            # metadata field is a string, not a dict. ``metadata.get(...)``
            # used to AttributeError.
            "verified/b-metadata-not-dict.ipynb": {
                "cells": [],
                "metadata": "this should have been a dict",
                "nbformat": 4,
            },
            # data_concierge namespace is a list. ``dc.get(...)`` used to
            # AttributeError. The ``or {}`` fallback only catches falsy.
            "verified/c-data-concierge-not-dict.ipynb": {
                "cells": [],
                "metadata": {"data_concierge": ["wrong", "shape"]},
                "nbformat": 4,
            },
            # submission_id is a list — unhashable, can't go in a set.
            # ``seen_submission_ids.add(sub_id)`` used to TypeError.
            "verified/d-sub-id-unhashable.ipynb": {
                "cells": [],
                "metadata": {
                    "data_concierge": {
                        "submission_id": ["a", "b"],
                        "query": "q",
                        "confidence": 0.5,
                        "verified_by": "admin",
                        "verified_at": "2026-05-27T18:30:00Z",
                    }
                },
                "nbformat": 4,
            },
            "verified/e-good.ipynb": _make_github_notebook(
                submission_id="sub-e", query="q-e"
            ),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return notebooks.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["checked"] == 5
        # The good notebook still made it through.
        assert len(result["created"]) == 1
        # All four shape-malformed files routed to skipped_bad_metadata.
        assert set(result["skipped_bad_metadata"]) == {
            "verified/a-notebook-not-dict.ipynb",
            "verified/b-metadata-not-dict.ipynb",
            "verified/c-data-concierge-not-dict.ipynb",
            "verified/d-sub-id-unhashable.ipynb",
        }
        # None of them ended up in any other bucket.
        assert result["failed"] == []
        assert result["skipped_no_metadata"] == []

        # The good notebook is queryable.
        verifieds = get_verified_notebooks()
        assert [v.submission_id for v in verifieds] == ["sub-e"]

    async def test_missing_metadata_namespace_routes_to_skipped_no_metadata(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: the metadata-genuinely-missing case (None vs wrong-shape)
        still routes to skipped_no_metadata so the two buckets stay
        semantically distinct in the report."""
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "no-metadata-key.ipynb",
                 "path": "verified/no-metadata-key.ipynb", "sha": "1"},
                {"name": "no-dc-namespace.ipynb",
                 "path": "verified/no-dc-namespace.ipynb", "sha": "2"},
                {"name": "empty-sub-id.ipynb",
                 "path": "verified/empty-sub-id.ipynb", "sha": "3"},
            ]

        notebooks = {
            # Bare notebook with no metadata field at all.
            "verified/no-metadata-key.ipynb": {"cells": [], "nbformat": 4},
            # metadata present but no data_concierge namespace.
            "verified/no-dc-namespace.ipynb": {
                "cells": [],
                "metadata": {"kernelspec": {"name": "python3"}},
                "nbformat": 4,
            },
            # data_concierge present but submission_id is "".
            "verified/empty-sub-id.ipynb": {
                "cells": [],
                "metadata": {"data_concierge": {"submission_id": ""}},
                "nbformat": 4,
            },
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return notebooks.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["checked"] == 3
        assert set(result["skipped_no_metadata"]) == {
            "verified/no-metadata-key.ipynb",
            "verified/no-dc-namespace.ipynb",
            "verified/empty-sub-id.ipynb",
        }
        assert result["skipped_bad_metadata"] == []

    async def test_returns_skipped_when_github_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_settings(monkeypatch, enabled=False)
        result = await bootstrap_index_from_github()
        assert result["skipped"] is True
        assert "not active" in result["reason"].lower()

    async def test_strict_listing_failure_propagates(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GitHub auth/network fails, bootstrap must raise so the
        endpoint can return 502 — never silently report 'rebuilt 0'."""
        _patch_settings(monkeypatch)

        async def _list_fail(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            assert strict is True
            raise github_publisher.GitHubPublishError(
                "GitHub returned 401 listing verified"
            )

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list_fail)

        with pytest.raises(github_publisher.GitHubPublishError) as exc:
            await bootstrap_index_from_github()
        assert "401" in str(exc.value)

    async def test_duplicate_submission_id_on_github_routes_to_skipped(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two GitHub files claim the same submission_id and there's no
        local entry yet. Without a guard, both would fall into the create
        branch and produce two VerifiedNotebook rows with the same
        submission_id but different notebook_ids — index corruption.
        The first occurrence wins; the duplicate is reported."""
        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "first.ipynb",
                 "path": "verified/first.ipynb", "sha": "1"},
                {"name": "dup.ipynb",
                 "path": "verified/dup.ipynb", "sha": "2"},
                {"name": "other.ipynb",
                 "path": "verified/other.ipynb", "sha": "3"},
            ]

        notebooks = {
            "verified/first.ipynb": _make_github_notebook(
                submission_id="sub-shared", query="first"
            ),
            "verified/dup.ipynb": _make_github_notebook(
                submission_id="sub-shared", query="duplicate"
            ),
            "verified/other.ipynb": _make_github_notebook(
                submission_id="sub-other", query="other"
            ),
        }

        async def _fetch(path: str) -> dict[str, Any] | None:
            return notebooks.get(path)

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["checked"] == 3
        # First occurrence wins; one unrelated also creates.
        assert len(result["created"]) == 2
        # Duplicate routed to its dedicated bucket — NOT into bad_metadata
        # (the file itself is well-formed) and NOT into created (which
        # would corrupt the index).
        assert result["skipped_duplicate_submission_id"] == ["verified/dup.ipynb"]
        assert "verified/dup.ipynb" not in result["skipped_bad_metadata"]
        assert "verified/dup.ipynb" not in result["created"]

        # Critical invariant: only ONE local entry per submission_id.
        verifieds = get_verified_notebooks()
        sub_ids = [v.submission_id for v in verifieds]
        assert sub_ids.count("sub-shared") == 1, (
            "index must not contain two entries with the same submission_id"
        )
        assert set(sub_ids) == {"sub-shared", "sub-other"}

    async def test_bad_metadata_on_update_preserves_existing_local_entry(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When an existing local entry exists but GitHub's version has
        corrupt metadata (e.g. non-numeric confidence), bootstrap must
        route the path to skipped_bad_metadata and leave the local entry
        UNTOUCHED — a bad republish must not silently overwrite a good
        cached entry."""
        _patch_settings(monkeypatch)

        # Seed a known-good local entry.
        seed_sub = verified_notebooks.submit_notebook(
            query="Good query",
            answer="answer",
            notebook_json=_make_github_notebook(
                submission_id="sub-good",
                query="Good query",
                confidence=0.85,
                verified_at="2026-01-01T00:00:00Z",
            ),
            submitted_by="tester",
            data_source="bls",
            confidence=0.85,
        )
        existing = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id,
            reviewed_by="admin",
        )
        assert existing is not None
        existing_nb_id = existing.notebook_id
        # Snapshot the seeded values — approve_notebook generates its own
        # verified_at/verified_by, so we compare against what was actually
        # written, not what we passed into _make_github_notebook.
        seeded_query = existing.query
        seeded_confidence = existing.confidence
        seeded_verified_by = existing.verified_by
        seeded_verified_at = existing.verified_at
        seeded_data_source = existing.data_source

        # GitHub now has the same submission_id but with bad confidence.
        bad_notebook = _make_github_notebook(
            submission_id=seed_sub.submission_id,
            query="Should NOT be applied",
            confidence=0.5,  # placeholder, we'll corrupt it
            verified_by="evil-admin",
            verified_at="2026-05-27T18:30:00Z",
            data_source="census",
        )
        bad_notebook["metadata"]["data_concierge"]["confidence"] = "very-high"

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "bad.ipynb", "path": "verified/bad.ipynb", "sha": "1"},
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return bad_notebook

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["updated"] == []
        assert result["created"] == []
        assert result["skipped_bad_metadata"] == ["verified/bad.ipynb"]

        # Local entry MUST be unchanged — every field stays at its seeded
        # value, none of the corrupt "Should NOT be applied" fields leak in.
        unchanged = get_verified_notebook(existing_nb_id)
        assert unchanged is not None
        assert unchanged.query == seeded_query
        assert unchanged.confidence == seeded_confidence
        assert unchanged.verified_by == seeded_verified_by
        assert unchanged.verified_at == seeded_verified_at
        assert unchanged.data_source == seeded_data_source

    async def test_type_corrupt_verified_by_on_update_preserves_existing_local_entry(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """an earlier adversarial review: the update path used to validate ``confidence``
        only, then assign raw GitHub values for ``verified_by`` / ``verified_at``
        / ``data_source`` straight into the index. A list-shaped ``verified_by``
        passed the inline checks but produced an index entry that crashed
        the next ``VerifiedNotebook(**data)`` read with ValidationError
        (i.e. the bad republish corrupted the local index, surfacing as a
        500 on a later list/search call). Round-tripping the merged
        candidate through the model catches this before it lands."""
        _patch_settings(monkeypatch)

        # Seed a known-good local entry.
        seed_sub = verified_notebooks.submit_notebook(
            query="Good query",
            answer="answer",
            notebook_json=_make_github_notebook(
                submission_id="sub-good-vb",
                query="Good query",
                confidence=0.85,
            ),
            submitted_by="tester",
            data_source="bls",
            confidence=0.85,
        )
        existing = verified_notebooks.approve_notebook(
            submission_id=seed_sub.submission_id,
            reviewed_by="admin",
        )
        assert existing is not None
        existing_nb_id = existing.notebook_id
        seeded_verified_by = existing.verified_by

        # GitHub has the same submission_id but ``verified_by`` is a list —
        # numerically/structurally invalid for the str-typed field.
        bad_notebook = _make_github_notebook(
            submission_id=seed_sub.submission_id,
            query="Should NOT be applied",
            confidence=0.7,  # numerically fine — only verified_by is corrupt
        )
        bad_notebook["metadata"]["data_concierge"]["verified_by"] = ["not", "a", "str"]

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "vb.ipynb", "path": "verified/vb.ipynb", "sha": "1"},
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return bad_notebook

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        result = await bootstrap_index_from_github()
        assert result["updated"] == []
        assert result["created"] == []
        assert result["skipped_bad_metadata"] == ["verified/vb.ipynb"]

        # Local entry MUST be unchanged and MUST still be readable —
        # the round-trip would have produced a corrupt index entry that
        # this same call would then crash on.
        unchanged = get_verified_notebook(existing_nb_id)
        assert unchanged is not None
        assert unchanged.verified_by == seeded_verified_by
        # And listing must work, not raise ValidationError.
        listed = get_verified_notebooks()
        assert any(v.notebook_id == existing_nb_id for v in listed)


class TestBootstrapEndpoint:
    """End-to-end smoke for POST /verified-notebooks/bootstrap-from-github."""

    async def test_endpoint_returns_502_when_listing_fails(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            bootstrap_verified_from_github,
        )

        _patch_settings(monkeypatch)

        async def _list_fail(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            raise github_publisher.GitHubPublishError(
                "GitHub returned 401 listing verified"
            )

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list_fail)

        with pytest.raises(HTTPException) as exc:
            await bootstrap_verified_from_github(_admin={"user": "admin"})
        assert exc.value.status_code == 502
        assert "Could not reach GitHub" in str(exc.value.detail)
        assert "401" in str(exc.value.detail)

    async def test_endpoint_returns_400_when_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import HTTPException

        from data_concierge.gateway.router import (
            bootstrap_verified_from_github,
        )

        _patch_settings(monkeypatch, enabled=False)

        with pytest.raises(HTTPException) as exc:
            await bootstrap_verified_from_github(_admin={"user": "admin"})
        assert exc.value.status_code == 400
        assert "GitHub publishing is not enabled" in str(exc.value.detail)

    async def test_endpoint_success_returns_report(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data_concierge.gateway.router import (
            bootstrap_verified_from_github,
        )

        _patch_settings(monkeypatch)

        async def _list(folder: str, *, strict: bool = False) -> list[dict[str, Any]]:
            return [
                {"name": "n.ipynb", "path": "verified/n.ipynb", "sha": "s"}
            ]

        async def _fetch(path: str) -> dict[str, Any] | None:
            return _make_github_notebook(submission_id="sub-1", query="q")

        monkeypatch.setattr(github_publisher, "_list_folder_files", _list)
        monkeypatch.setattr(github_publisher, "fetch_notebook", _fetch)

        body = await bootstrap_verified_from_github(_admin={"user": "admin"})
        assert body["checked"] == 1
        assert len(body["created"]) == 1
        assert "Bootstrapped 1 verified notebook" in body["message"]
