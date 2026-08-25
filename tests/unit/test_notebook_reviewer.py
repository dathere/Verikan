"""Adversarial notebook method review (#131, third signal).

The reviewer's *scoring* is deterministic — a fixed deduction schedule over
severity-rated findings — precisely so it can be pinned here without an API
call. The API path itself is only tested for its unavailability behaviour:
when the review cannot run it must report ``score=None`` with a reason, never
a fabricated number (#132 rules).
"""

import pytest

from data_concierge.agents.notebook_reviewer import (
    ReviewFinding,
    ReviewVerdict,
    _parse_review_input,
    render_notebook_for_review,
    review_notebook,
    score_from_findings,
)


def find(severity: str) -> ReviewFinding:
    return ReviewFinding(severity=severity, title="t", detail="d")


def make_nb(*sources: str) -> dict:
    return {
        "cells": [
            {"cell_type": "code", "source": s, "metadata": {}, "outputs": []} for s in sources
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class TestScoring:
    """The deduction schedule is the contract the confidence factor rests on."""

    def test_clean_review_is_perfect(self) -> None:
        assert score_from_findings([]) == 1.0

    def test_single_critical_sinks_the_score(self) -> None:
        """A hardcoded answer passes execution AND reconciliation — the
        review is the only signal that can catch it, so one critical finding
        must speak decisively."""
        assert score_from_findings([find("critical")]) == pytest.approx(0.1)

    def test_severity_ladder(self) -> None:
        assert score_from_findings([find("high")]) == pytest.approx(0.6)
        assert score_from_findings([find("medium")]) == pytest.approx(0.85)
        assert score_from_findings([find("low")]) == pytest.approx(0.95)

    def test_deductions_accumulate_and_floor_at_zero(self) -> None:
        assert score_from_findings([find("medium")] * 2 + [find("low")]) == pytest.approx(0.65)
        assert score_from_findings([find("critical"), find("high")]) == 0.0

    def test_unknown_severity_costs_nothing(self) -> None:
        # _parse_review_input clamps severities, so this only happens for
        # findings constructed in code; they must not silently sink a score.
        assert score_from_findings([find("bogus")]) == 1.0


class TestParsing:
    """The model's tool input is coerced, never trusted."""

    def test_valid_findings_survive(self) -> None:
        summary, findings = _parse_review_input(
            {
                "summary": "one hardcoded number",
                "findings": [
                    {
                        "severity": "critical",
                        "title": "Hardcoded answer",
                        "detail": "4.1 is a literal",
                        "cell_index": 3,
                    }
                ],
            }
        )
        assert summary == "one hardcoded number"
        assert findings[0].severity == "critical"
        assert findings[0].cell_index == 3

    def test_unknown_severity_clamps_to_medium(self) -> None:
        _, findings = _parse_review_input(
            {"summary": "s", "findings": [{"severity": "fatal", "title": "x", "detail": "y"}]}
        )
        assert findings[0].severity == "medium"

    def test_garbage_entries_are_dropped(self) -> None:
        _, findings = _parse_review_input(
            {
                "summary": "s",
                "findings": ["not a dict", {"severity": "low"}, None],
            }
        )
        assert findings == []

    def test_finding_cap(self) -> None:
        many = [{"severity": "low", "title": f"t{i}", "detail": "d"} for i in range(50)]
        _, findings = _parse_review_input({"summary": "s", "findings": many})
        assert len(findings) == 12

    def test_non_integer_cell_index_is_dropped(self) -> None:
        _, findings = _parse_review_input(
            {
                "summary": "s",
                "findings": [{"severity": "low", "title": "t", "detail": "d", "cell_index": "3"}],
            }
        )
        assert findings[0].cell_index is None


class TestRendering:
    def test_cells_are_indexed_and_typed(self) -> None:
        nb = make_nb("print(1)", "print(2)")
        nb["cells"].insert(0, {"cell_type": "markdown", "source": "# Title", "metadata": {}})
        text = render_notebook_for_review(nb)
        assert "--- cell 0 (markdown) ---" in text
        assert "--- cell 1 (code) ---" in text
        assert "--- cell 2 (code) ---" in text

    def test_list_form_source_is_joined(self) -> None:
        nb = {"cells": [{"cell_type": "code", "source": ["a = 1\n", "print(a)\n"]}]}
        assert "a = 1\nprint(a)" in render_notebook_for_review(nb)

    def test_oversized_cell_is_truncated(self) -> None:
        text = render_notebook_for_review(make_nb("x" * 50000))
        assert "[truncated" in text

    def test_oversized_notebook_is_capped(self) -> None:
        text = render_notebook_for_review(make_nb(*["y" * 3900] * 100))
        assert "remaining cells omitted" in text


class TestUnavailability:
    """A review that cannot run reports unavailable — never a stand-in score."""

    async def test_disabled_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from data_concierge.agents import notebook_reviewer as nr

        monkeypatch.setattr(nr.settings, "notebook_review_enabled", False)
        verdict = await review_notebook(make_nb("print(1)"), "q", "a")
        assert verdict.reviewed is False
        assert verdict.score is None
        assert verdict.reason

    async def test_no_api_key_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import SecretStr

        from data_concierge.agents import notebook_reviewer as nr

        monkeypatch.setattr(nr.settings, "notebook_review_enabled", True)
        monkeypatch.setattr(nr.settings, "anthropic_api_key", SecretStr(""))
        verdict = await review_notebook(make_nb("print(1)"), "q", "a")
        assert verdict.score is None
        assert "not configured" in (verdict.reason or "")

    async def test_empty_notebook_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from pydantic import SecretStr

        from data_concierge.agents import notebook_reviewer as nr

        monkeypatch.setattr(nr.settings, "notebook_review_enabled", True)
        monkeypatch.setattr(nr.settings, "anthropic_api_key", SecretStr("sk-test"))
        verdict = await review_notebook({"cells": []}, "q", "a")
        assert verdict.score is None
        assert "no cells" in (verdict.reason or "")


class TestVerdictShape:
    def test_severity_counts(self) -> None:
        verdict = ReviewVerdict(
            reviewed=True,
            findings=[find("critical"), find("low"), find("low")],
        )
        counts = verdict.severity_counts
        assert counts["critical"] == 1
        assert counts["low"] == 2
        assert counts["high"] == 0

    def test_signals_are_compact(self) -> None:
        verdict = ReviewVerdict(reviewed=True, findings=[find("high")], model="m")
        signals = verdict.as_signals()
        assert signals["review_ran"] is True
        assert signals["review_findings"] == 1
        assert signals["review_severities"] == {"high": 1}
