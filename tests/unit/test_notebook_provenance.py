"""Tests for the embedded notebook provenance (issue #46, step 4).

When an admin approves a notebook, the verification fields
(``submission_id``, ``confidence``, ``verified_by``, ``verified_at``)
are merged into the notebook's ``metadata.data_concierge`` namespace
before the notebook is published to GitHub and persisted locally. This
makes the .ipynb files on GitHub self-describing — the enabler for the
disaster-recovery rebuild in step 5.

Covers:

* ``add_verification_metadata`` enriches correctly, preserves existing
  generator fields, and doesn't mutate the input.
* ``approve_notebook`` writes the override copy to both the index entry
  and the on-disk blob when one is supplied.
* The approve endpoint reuses a single ``verified_at`` timestamp for
  both the embedded metadata and the index's ``github_synced_at``, so
  the two views stay coherent.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.gateway import verified_notebooks
from data_concierge.gateway.verified_notebooks import (
    _VERIFIED_PREFIX,
    ReviewStatus,
    add_verification_metadata,
    approve_notebook,
    get_submission,
    get_verified_notebook,
    submit_notebook,
)

# ---------------------------------------------------------------------------
# Storage fixture (same shape as the step-1 tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    test_storage = LocalStorage(tmp_path)
    monkeypatch.setattr(verified_notebooks, "storage", test_storage)


def _make_submission() -> Any:
    return submit_notebook(
        query="Provenance: unemployment rate in Texas?",
        answer="4.1%",
        notebook_json={
            "cells": [],
            "metadata": {
                # Mimic what notebook_generator.py emits.
                "kernelspec": {"name": "python3", "display_name": "Python 3"},
                "data_concierge": {
                    "version": "0.1.0",
                    "generated": "2026-05-27T10:00:00",
                    "query": "Provenance: unemployment rate in Texas?",
                    "data_source": "bls",
                    "colab_compatible": True,
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        submitted_by="tester",
        data_source="bls",
        confidence=0.87,
    )


# ---------------------------------------------------------------------------
# add_verification_metadata — pure function behavior
# ---------------------------------------------------------------------------


class TestAddVerificationMetadata:
    def test_adds_all_verification_fields(self) -> None:
        original = {
            "cells": [],
            "metadata": {"data_concierge": {"version": "0.1.0"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        enriched = add_verification_metadata(
            original,
            submission_id="sub-123",
            confidence=0.87,
            verified_by="admin@dathere",
            verified_at="2026-05-27T18:30:00Z",
            data_source="bls",
        )
        dc = enriched["metadata"]["data_concierge"]
        assert dc["submission_id"] == "sub-123"
        assert dc["confidence"] == 0.87
        assert dc["verified_by"] == "admin@dathere"
        assert dc["verified_at"] == "2026-05-27T18:30:00Z"
        # Existing field preserved.
        assert dc["version"] == "0.1.0"

    def test_preserves_existing_data_concierge_namespace_fields(self) -> None:
        """Generator-time fields (version, generated, query) must not be
        overwritten by verification-time enrichment."""
        original = {
            "metadata": {
                "data_concierge": {
                    "version": "0.1.0",
                    "generated": "2026-05-27T10:00:00",
                    "query": "original query",
                    "data_source": "bls",
                    "colab_compatible": True,
                }
            }
        }
        enriched = add_verification_metadata(
            original,
            submission_id="s",
            confidence=0.5,
            verified_by="x",
            verified_at="2026-05-27T18:30:00Z",
            data_source="census",  # different from existing — existing wins
        )
        dc = enriched["metadata"]["data_concierge"]
        assert dc["version"] == "0.1.0"
        assert dc["generated"] == "2026-05-27T10:00:00"
        assert dc["query"] == "original query"
        assert dc["data_source"] == "bls", (
            "Existing data_source from the generator must win — that's the "
            "truth at notebook creation time"
        )
        assert dc["colab_compatible"] is True

    def test_does_not_mutate_input(self) -> None:
        original = {"metadata": {"data_concierge": {"version": "0.1.0"}}}
        original_snapshot = {
            "metadata": {"data_concierge": {"version": "0.1.0"}}
        }
        add_verification_metadata(
            original,
            submission_id="s",
            confidence=0.5,
            verified_by="x",
            verified_at="t",
        )
        assert original == original_snapshot, (
            "add_verification_metadata must return a new dict and leave the "
            "input alone — callers may keep their own reference"
        )

    def test_handles_missing_metadata_namespace(self) -> None:
        """A bare notebook with no metadata at all still gets enriched."""
        original = {"cells": [], "nbformat": 4}
        enriched = add_verification_metadata(
            original,
            submission_id="s",
            confidence=0.5,
            verified_by="x",
            verified_at="t",
            data_source="bls",
        )
        dc = enriched["metadata"]["data_concierge"]
        assert dc["submission_id"] == "s"
        assert dc["data_source"] == "bls"  # set fresh since there was none

    def test_skips_confidence_when_none(self) -> None:
        """confidence=None is a sentinel for 'unknown' — don't write null."""
        original: dict[str, Any] = {"metadata": {}}
        enriched = add_verification_metadata(
            original,
            submission_id="s",
            confidence=None,
            verified_by="x",
            verified_at="t",
        )
        assert "confidence" not in enriched["metadata"]["data_concierge"]


# ---------------------------------------------------------------------------
# approve_notebook with notebook_json override
# ---------------------------------------------------------------------------


class TestApproveNotebookWithOverride:
    def test_override_is_persisted_to_index_and_blob(
        self, tmp_storage: None
    ) -> None:
        sub = _make_submission()
        enriched = add_verification_metadata(
            sub.notebook_json,
            submission_id=sub.submission_id,
            confidence=0.87,
            verified_by="admin",
            verified_at="2026-05-27T18:30:00Z",
        )

        verified = approve_notebook(
            submission_id=sub.submission_id,
            reviewed_by="admin",
            notebook_json=enriched,
        )
        assert verified is not None

        # Index entry has the enriched copy.
        from_index = get_verified_notebook(verified.notebook_id)
        assert from_index is not None
        dc = from_index.notebook_json["metadata"]["data_concierge"]
        assert dc["submission_id"] == sub.submission_id
        assert dc["verified_by"] == "admin"
        assert dc["verified_at"] == "2026-05-27T18:30:00Z"

        # On-disk blob has the enriched copy too (this is what
        # /verified-notebooks/{id}/download serves and what
        # sync_all_from_github / disaster-recovery-rebuild reads).
        blob_key = f"{_VERIFIED_PREFIX}/{verified.notebook_id}.ipynb"
        blob = verified_notebooks.storage.read_json(blob_key)
        assert blob is not None
        assert (
            blob["metadata"]["data_concierge"]["submission_id"]
            == sub.submission_id
        )

    def test_no_override_keeps_submission_notebook_unchanged(
        self, tmp_storage: None
    ) -> None:
        """Backwards-compat path: existing callers that don't pass an
        override still see the submission's original notebook_json."""
        sub = _make_submission()
        verified = approve_notebook(submission_id=sub.submission_id)
        assert verified is not None
        assert verified.notebook_json == sub.notebook_json


# ---------------------------------------------------------------------------
# End-to-end through the approve endpoint
# ---------------------------------------------------------------------------


class TestEndpointEmbedsProvenance:
    async def test_endpoint_embeds_verified_at_matching_github_synced_at(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The single ``verified_at`` instant must show up in both the
        embedded notebook metadata AND the local index's
        ``github_synced_at`` — that coherence is what makes the
        disaster-recovery rebuild (step 5) possible."""
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        sub = _make_submission()

        captured_publish_arg: dict[str, Any] = {}

        async def _ok_publish(
            sub_id: str,
            nb_id: str,
            query: str,
            notebook_json: dict[str, Any],
            reason: str | None = None,
            reviewer: str | None = None,
        ) -> dict[str, Any]:
            captured_publish_arg["notebook_json"] = notebook_json
            return {
                "path": "verified/p_aaaaaaaa.ipynb",
                "filename": "p_aaaaaaaa.ipynb",
                "sha": "deadbeef",
                "draft_cleanup_pending": False,
            }

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _ok_publish,
        )

        result = await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            # The client-supplied reviewed_by is ignored; the verifier identity
            # comes from the authenticated admin session (#73).
            request=NotebookReviewRequest(reviewed_by="not-the-verifier", admin_notes="reviewed for tests"),
            _admin={"user": "admin@dathere"},
        )

        # 1. The notebook handed to publish_verified carries the provenance.
        sent_dc = captured_publish_arg["notebook_json"]["metadata"][
            "data_concierge"
        ]
        assert sent_dc["submission_id"] == sub.submission_id
        assert sent_dc["verified_by"] == "admin@dathere"
        assert sent_dc["verified_at"]  # populated, real timestamp
        assert sent_dc["verified_at"].endswith("Z")
        assert sent_dc["confidence"] == 0.87  # from the submission
        # Generator-time fields still there.
        assert sent_dc["version"] == "0.1.0"
        assert sent_dc["query"] == sub.query

        # 2. The local VerifiedNotebook has the SAME verified_at as the
        # index's github_synced_at.
        v = get_verified_notebook(result["notebook_id"])
        assert v is not None
        local_verified_at = v.notebook_json["metadata"]["data_concierge"][
            "verified_at"
        ]
        assert local_verified_at == sent_dc["verified_at"], (
            "GitHub copy and local copy must carry the same verified_at"
        )
        assert v.github_synced_at == local_verified_at, (
            "github_synced_at in the local index must equal the embedded "
            "verified_at — coherence required for disaster-recovery rebuild"
        )

        # 3. Submission marked APPROVED.
        peek = get_submission(sub.submission_id)
        assert peek is not None
        assert peek.status == ReviewStatus.APPROVED

    async def test_endpoint_embeds_provenance_even_when_github_disabled(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When GitHub is disabled, the local-only approval should still
        carry the verified_at / verified_by provenance — disaster-recovery
        rebuild may run against a local backup eventually."""
        from data_concierge.gateway.router import (
            NotebookReviewRequest,
            approve_notebook_endpoint,
        )

        sub = _make_submission()

        async def _disabled_publish(*a: Any, **kw: Any) -> None:
            return None

        monkeypatch.setattr(
            "data_concierge.gateway.github_publisher.publish_verified",
            _disabled_publish,
        )

        result = await approve_notebook_endpoint(
            submission_id=sub.submission_id,
            request=NotebookReviewRequest(reviewed_by="admin", admin_notes="reviewed for tests"),
            _admin={"user": "admin"},
        )
        v = get_verified_notebook(result["notebook_id"])
        assert v is not None
        dc = v.notebook_json["metadata"]["data_concierge"]
        assert dc["submission_id"] == sub.submission_id
        assert dc["verified_by"] == "admin"
        assert dc["verified_at"]
        # github_synced_at stays None (no GitHub commit), but the in-notebook
        # provenance is set regardless.
        assert v.github_synced_at is None
