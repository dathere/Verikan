"""Merging the review signal into notebook verification (#131, third signal).

``gateway/notebook_verification`` combines execution/reconciliation with the
adversarial review into the single ``notebook_verification`` confidence
factor, persists both verdicts per query, and serves them to the admin
Notebook Reviews pane. The merge must follow the #132 rules: an unavailable
side never enters as a stand-in constant, and when neither side ran the
factor is unavailable with a reason.
"""

import asyncio

import pytest

from data_concierge.agents.notebook_reviewer import ReviewFinding, ReviewVerdict
from data_concierge.core.models import ConfidenceScore
from data_concierge.data_layer.storage import LocalStorage
from data_concierge.gateway import notebook_verification as nv


def make_nb(*sources: str) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "source": s, "metadata": {}, "outputs": []} for s in sources
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def confidence() -> ConfidenceScore:
    return ConfidenceScore.compute_from_signals(
        answer_grounding=0.8,
        data_retrieval_quality=0.7,
        source_metadata_quality=0.9,
        query_answer_alignment=None,
        computation_complexity=0.6,
        unavailable={
            "query_answer_alignment": "not scored yet",
            "notebook_verification": ConfidenceScore.NOTEBOOK_PENDING_REASON,
        },
    )


class TestCombine:
    def test_both_measured_is_weighted_70_30(self) -> None:
        score, reason = nv.combine_scores(1.0, 0.5)
        assert score == pytest.approx(0.85)
        assert reason is None

    def test_execution_alone_carries_the_factor(self) -> None:
        score, reason = nv.combine_scores(0.8, None, None, "review offline")
        assert score == 0.8
        assert reason is None

    def test_review_alone_carries_the_factor(self) -> None:
        score, reason = nv.combine_scores(None, 0.9, "kernel missing", None)
        assert score == 0.9
        assert reason is None

    def test_neither_measured_is_unavailable_with_reasons(self) -> None:
        score, reason = nv.combine_scores(None, None, "kernel missing", "review offline")
        assert score is None
        assert "kernel missing" in reason
        assert "review offline" in reason

    def test_critical_finding_caps_the_combined_score(self) -> None:
        """A hardcoded answer executes cleanly and reconciles perfectly, so
        clean mechanical signals must not outvote a critical review finding."""
        capped, _ = nv.combine_scores(1.0, 0.1, review_has_critical=True)
        assert capped == pytest.approx(0.1)
        uncapped, _ = nv.combine_scores(1.0, 0.1, review_has_critical=False)
        assert uncapped == pytest.approx(0.73)

    def test_confidence_folds_in_the_combined_score(self) -> None:
        before = confidence()
        after = before.with_notebook_verification(0.85, None, {"review_findings": 1})
        assert after.notebook_verification == 0.85
        assert "notebook_verification" not in after.unavailable
        assert after.final_score != before.final_score


class TestScheduledMerge:
    """The async pass runs execution and review together and stores both."""

    @staticmethod
    def _reviewed_verdict(*findings: ReviewFinding) -> ReviewVerdict:
        from data_concierge.agents.notebook_reviewer import score_from_findings

        return ReviewVerdict(
            reviewed=True,
            summary="stubbed",
            findings=list(findings),
            model="stub-model",
            score=score_from_findings(list(findings)),
        )

    async def _run_scheduled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        review: ReviewVerdict,
        notebook: dict,
        answer: str,
        query_id: str,
    ) -> dict:
        monkeypatch.setattr(nv.settings, "notebook_verification_enabled", True)
        monkeypatch.setattr(nv, "storage", LocalStorage(tmp_path))

        async def fake_review(nb: dict, query: str, ans: str) -> ReviewVerdict:
            return review

        monkeypatch.setattr(nv, "review_notebook", fake_review)

        assert nv.schedule_verification(
            query_id, notebook, answer, confidence(), query="test question", data_source="wprdc"
        )
        for _ in range(200):
            await asyncio.sleep(0.1)
            if not nv._in_flight:
                break
        stored = nv.get_verification(query_id)
        assert stored is not None
        return stored

    async def test_clean_review_and_clean_execution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        stored = await self._run_scheduled(
            monkeypatch,
            tmp_path,
            self._reviewed_verdict(),
            make_nb("print(4.1)"),
            "The rate was 4.1 percent.",
            "q-merge-clean",
        )
        assert stored["status"] == "complete"
        assert stored["verdict"]["executed"] is True
        assert stored["review"]["reviewed"] is True
        assert stored["combined_score"] == pytest.approx(1.0)
        assert stored["confidence_breakdown"]["notebook_verification"] == pytest.approx(1.0)
        assert stored["query"] == "test question"
        assert stored["data_source"] == "wprdc"
        assert stored["confidence_before"] == pytest.approx(confidence().final_score)

    async def test_critical_finding_caps_despite_perfect_execution(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        critical = ReviewFinding(
            severity="critical", title="Hardcoded answer", detail="4.1 is a literal"
        )
        stored = await self._run_scheduled(
            monkeypatch,
            tmp_path,
            self._reviewed_verdict(critical),
            make_nb("print(4.1)"),
            "The rate was 4.1 percent.",
            "q-merge-critical",
        )
        assert stored["verdict"]["executed"] is True
        assert stored["combined_score"] == pytest.approx(0.1)
        assert stored["confidence"] < stored["confidence_before"]

    async def test_unavailable_review_leaves_execution_verdict_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        unavailable = ReviewVerdict(reviewed=False, score=None, reason="review offline")
        stored = await self._run_scheduled(
            monkeypatch,
            tmp_path,
            unavailable,
            make_nb("print(4.1)"),
            "The rate was 4.1 percent.",
            "q-merge-noreview",
        )
        assert stored["review"]["reviewed"] is False
        assert stored["combined_score"] == pytest.approx(1.0)

    async def test_review_only_mode_skips_execution_but_still_reviews(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Disabling execution (the heavyweight subprocess) must not silently
        disable the API-only adversarial review with it."""
        monkeypatch.setattr(nv.settings, "notebook_verification_enabled", False)
        monkeypatch.setattr(nv.settings, "notebook_review_enabled", True)
        monkeypatch.setattr(nv, "storage", LocalStorage(tmp_path))

        async def fake_review(nb: dict, query: str, ans: str) -> ReviewVerdict:
            return self._reviewed_verdict()

        monkeypatch.setattr(nv, "review_notebook", fake_review)

        assert nv.schedule_verification(
            "q-review-only", make_nb("print(1)"), "answer", confidence(), query="q"
        )
        for _ in range(200):
            await asyncio.sleep(0.05)
            if not nv._in_flight:
                break

        stored = nv.get_verification("q-review-only")
        assert stored is not None
        assert stored["status"] == "complete"
        # Execution reported unavailable-with-reason, not run and not zeroed.
        assert stored["verdict"]["executed"] is False
        assert stored["verdict"]["score"] is None
        assert "disabled" in (stored["verdict"]["reason"] or "")
        # The review alone carries the factor.
        assert stored["combined_score"] == pytest.approx(1.0)


class TestListing:
    """The admin pane's data source: list + summarize stored records."""

    def _seed(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        store = LocalStorage(tmp_path)
        monkeypatch.setattr(nv, "storage", store)
        records = [
            {
                "query_id": "a",
                "status": "complete",
                "completed_at": "2026-08-19T10:00:00",
                "verdict": {"executed": True},
                "review": {
                    "reviewed": True,
                    "findings": [{"severity": "high", "title": "x", "detail": "y"}],
                },
                "combined_score": 0.6,
            },
            {
                "query_id": "b",
                "status": "complete",
                "completed_at": "2026-08-19T12:00:00",
                "verdict": {"executed": False},
                "review": {"reviewed": True, "findings": []},
                "combined_score": 0.2,
            },
            {
                "query_id": "c",
                "status": "pending",
                "completed_at": "2026-08-19T11:00:00",
                "verdict": None,
                "review": None,
                "combined_score": None,
            },
        ]
        for r in records:
            store.write_json(f"notebook_verification/{r['query_id']}.json", r)

    def test_newest_first_and_limit(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._seed(tmp_path, monkeypatch)
        listed = nv.list_verifications(limit=2)
        assert [r["query_id"] for r in listed] == ["b", "c"]
        assert len(nv.list_verifications(limit=100)) == 3

    def test_summary_counts(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._seed(tmp_path, monkeypatch)
        summary = nv.review_summary(nv.list_verifications(limit=100))
        assert summary["total"] == 3
        assert summary["complete"] == 2
        assert summary["pending"] == 1
        assert summary["executed_ok"] == 1
        assert summary["reviewed"] == 2
        assert summary["findings_by_severity"] == {"high": 1}
        assert summary["avg_combined_score"] == pytest.approx(0.4)

    def test_empty_prefix_is_empty_not_an_error(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(nv, "storage", LocalStorage(tmp_path))
        assert nv.list_verifications() == []
        assert nv.review_summary([])["total"] == 0
