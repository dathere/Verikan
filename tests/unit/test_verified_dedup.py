"""Tests for verified-library de-duplication.

The verified library should hold exactly one entry per question. Duplicates
accumulated in production (7 verified answers all for "What is the unemployment
rate in Texas?"). These tests cover the normalizer, the one-shot cleanup
(``dedupe_verified_library``), the by-question finders, and the submission
collapse helpers used by the approve endpoints for forward prevention.
"""

from __future__ import annotations

from typing import Any

import pytest

from data_concierge.gateway import verified_notebooks
from data_concierge.gateway.verified_notebooks import (
    NotebookSubmission,
    QuickAnswerSubmission,
    ReviewStatus,
    VerifiedAnswer,
    VerifiedNotebook,
    collapse_answer_submission_as_duplicate,
    collapse_notebook_submission_as_duplicate,
    dedupe_verified_library,
    find_verified_answer_by_question,
    find_verified_notebook_by_question,
    normalize_question,
)


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    monkeypatch.setattr(verified_notebooks, "storage", LocalStorage(tmp_path))


def _seed_answer(query: str, *, usage: int = 0, verified_at: str = "2026-01-01T00:00:00",
                 github_path: str | None = None) -> VerifiedAnswer:
    va = VerifiedAnswer(
        submission_id="s",
        query=query,
        answer="42",
        usage_count=usage,
        verified_at=verified_at,
        github_path=github_path,
    )
    index = verified_notebooks._load_index()
    index.setdefault("verified_answers", {})[va.answer_id] = va.model_dump()
    verified_notebooks._save_index(index)
    return va


def _seed_notebook(query: str, *, usage: int = 0, verified_at: str = "2026-01-01T00:00:00",
                   github_path: str | None = None) -> VerifiedNotebook:
    nb = VerifiedNotebook(
        submission_id="s",
        query=query,
        notebook_json={"cells": []},
        usage_count=usage,
        verified_at=verified_at,
        github_path=github_path,
    )
    index = verified_notebooks._load_index()
    index.setdefault("verified", {})[nb.notebook_id] = nb.model_dump()
    verified_notebooks._save_index(index)
    # mirror the on-disk blob that approve_notebook writes
    verified_notebooks.storage.write_json(
        f"{verified_notebooks._VERIFIED_PREFIX}/{nb.notebook_id}.ipynb", nb.notebook_json
    )
    return nb


# ---------------------------------------------------------------------------
# normalize_question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b",
    [
        ("What is the unemployment rate in Texas?", "what is the unemployment rate in texas"),
        ("  WHAT   is  GDP? ", "what is gdp"),
        ("Population of Ohio.", "population of ohio"),
        ('"Median income"', "median income"),
    ],
)
def test_normalize_question_canonicalises(a: str, b: str) -> None:
    assert normalize_question(a) == b


def test_normalize_distinguishes_real_differences() -> None:
    # Conservative: does NOT merge genuinely different questions.
    assert normalize_question("what is unemployment") != normalize_question(
        "what is the unemployment rate"
    )


def test_normalize_empty() -> None:
    assert normalize_question("") == ""
    assert normalize_question("   ?  ") == ""


# ---------------------------------------------------------------------------
# dedupe_verified_library
# ---------------------------------------------------------------------------


def test_dedupe_collapses_duplicate_answers_and_merges_usage(tmp_storage: None) -> None:
    survivor = _seed_answer("Unemployment in Texas?", usage=1, verified_at="2026-02-13T00:00:00")
    _seed_answer("unemployment in texas", usage=1, verified_at="2026-02-19T00:00:00")
    _seed_answer("UNEMPLOYMENT IN TEXAS", usage=1, verified_at="2026-02-25T00:00:00")
    _seed_answer("Population of Ohio?", usage=3)  # distinct — untouched

    summary = dedupe_verified_library()
    assert summary["removed_count"] == 2

    answers = {a.query.lower(): a for a in verified_notebooks.get_verified_answers()}
    assert len(answers) == 2  # one unemployment + one ohio
    # Survivor is the OLDEST of the duplicate group, with merged usage (1+1+1).
    kept = verified_notebooks.get_verified_answer(survivor.answer_id)
    assert kept is not None
    assert kept.usage_count == 3


def test_dedupe_dry_run_does_not_mutate(tmp_storage: None) -> None:
    _seed_answer("dup question", verified_at="2026-01-01T00:00:00")
    _seed_answer("DUP question", verified_at="2026-01-02T00:00:00")
    summary = dedupe_verified_library(dry_run=True)
    assert summary["removed_count"] == 1
    assert len(verified_notebooks.get_verified_answers()) == 2  # nothing removed


def test_dedupe_deletes_orphan_notebook_blobs(tmp_storage: None) -> None:
    keep = _seed_notebook("Same question", verified_at="2026-01-01T00:00:00")
    drop = _seed_notebook("same question", verified_at="2026-01-02T00:00:00")
    drop_key = f"{verified_notebooks._VERIFIED_PREFIX}/{drop.notebook_id}.ipynb"
    keep_key = f"{verified_notebooks._VERIFIED_PREFIX}/{keep.notebook_id}.ipynb"
    assert verified_notebooks.storage.exists(drop_key)

    dedupe_verified_library()

    assert not verified_notebooks.storage.exists(drop_key)  # orphan blob removed
    assert verified_notebooks.storage.exists(keep_key)  # survivor blob kept
    assert verified_notebooks.get_verified_notebook(drop.notebook_id) is None
    assert verified_notebooks.get_verified_notebook(keep.notebook_id) is not None


def test_dedupe_never_deletes_github_published_entries(tmp_storage: None) -> None:
    # Two published copies of the same question must NOT be auto-deleted —
    # that would orphan a GitHub file. They're flagged for manual reconcile.
    _seed_answer("Q one", github_path="verified-answers/a.json")
    _seed_answer("q one", github_path="verified-answers/b.json")
    summary = dedupe_verified_library()
    assert summary["removed_count"] == 0
    assert len(verified_notebooks.get_verified_answers()) == 2
    assert summary["needs_manual_github_reconcile"]


def test_dedupe_collapses_local_into_published_survivor(tmp_storage: None) -> None:
    # A published entry + a local-only duplicate: drop the local one, keep
    # the published survivor and merge usage.
    pub = _seed_answer("Topic", usage=2, github_path="verified-answers/p.json")
    _seed_answer("topic", usage=5)  # local-only duplicate
    summary = dedupe_verified_library()
    assert summary["removed_count"] == 1
    kept = verified_notebooks.get_verified_answer(pub.answer_id)
    assert kept is not None and kept.usage_count == 7  # 2 + 5 merged


def test_dedupe_idempotent(tmp_storage: None) -> None:
    _seed_answer("repeat", verified_at="2026-01-01T00:00:00")
    _seed_answer("Repeat", verified_at="2026-01-02T00:00:00")
    assert dedupe_verified_library()["removed_count"] == 1
    assert dedupe_verified_library()["removed_count"] == 0  # nothing left to do


# ---------------------------------------------------------------------------
# finders + collapse helpers (forward prevention used by approve endpoints)
# ---------------------------------------------------------------------------


def test_find_verified_answer_by_question(tmp_storage: None) -> None:
    seeded = _seed_answer("What is GDP?")
    found = find_verified_answer_by_question("  what is gdp  ")
    assert found is not None and found.answer_id == seeded.answer_id
    assert find_verified_answer_by_question("unrelated") is None


def test_find_verified_notebook_prefers_published(tmp_storage: None) -> None:
    _seed_notebook("Crime data", usage=10)  # local, more usage
    pub = _seed_notebook("crime data", usage=1, github_path="verified/c.ipynb")
    found = find_verified_notebook_by_question("Crime Data?")
    assert found is not None and found.notebook_id == pub.notebook_id  # published wins


def test_collapse_notebook_submission_marks_approved_without_new_entry(tmp_storage: None) -> None:
    existing = _seed_notebook("Existing q")
    sub = NotebookSubmission(query="existing q", notebook_json={"cells": []})
    index = verified_notebooks._load_index()
    index["submissions"][sub.submission_id] = sub.model_dump()
    verified_notebooks._save_index(index)

    ok = collapse_notebook_submission_as_duplicate(
        sub.submission_id, existing.notebook_id, reviewed_by="alice", admin_notes="dup"
    )
    assert ok is True
    updated = verified_notebooks.get_submission(sub.submission_id)
    assert updated is not None and updated.status == ReviewStatus.APPROVED
    # No second verified entry was created.
    assert len(verified_notebooks.get_verified_notebooks()) == 1


def test_collapse_answer_submission(tmp_storage: None) -> None:
    existing = _seed_answer("Existing a")
    sub = QuickAnswerSubmission(query="existing a", answer="x")
    index = verified_notebooks._load_index()
    index.setdefault("answer_submissions", {})[sub.submission_id] = sub.model_dump()
    verified_notebooks._save_index(index)

    ok = collapse_answer_submission_as_duplicate(sub.submission_id, existing.answer_id)
    assert ok is True
    updated = verified_notebooks.get_answer_submission(sub.submission_id)
    assert updated is not None and updated.status == ReviewStatus.APPROVED
    assert len(verified_notebooks.get_verified_answers()) == 1
