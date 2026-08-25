"""Tests for two-stage verified-answer matching (issue #64).

The keyword-only matcher let temporal mismatches through — a query for
"crime in 2023" matched a verified answer for "crime in 2025" because the
shared topic terms dominated the Jaccard/coverage score. These tests cover:

* ``extract_keywords`` now preserving 4-digit years (so they surface as
  candidate-retrieval tokens instead of being silently dropped);
* the LLM verification gate helpers (response parsing, circuit breaker,
  availability toggle);
* the router two-stage flow — the gate rejecting a temporal mismatch, accepting
  a true match, and the keyword-only fallback when the gate is unavailable.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from data_concierge.gateway import match_verifier, verified_notebooks
from data_concierge.gateway.match_verifier import (
    _CircuitBreaker,
    _parse_response,
    llm_gate_available,
)
from data_concierge.gateway.verified_notebooks import (
    VerifiedAnswer,
    VerifiedNotebook,
    extract_keywords,
)

# gateway/__init__.py rebinds the name ``router`` to the APIRouter *instance*,
# shadowing the submodule. Pull the actual module out of sys.modules so we can
# monkeypatch its module-level helpers.
router = importlib.import_module("data_concierge.gateway.router")


# ---------------------------------------------------------------------------
# Storage fixture: isolate the verified index per test.
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_storage(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from data_concierge.data_layer.storage import LocalStorage

    monkeypatch.setattr(verified_notebooks, "storage", LocalStorage(tmp_path))


def _seed_answer(query: str, answer: str) -> VerifiedAnswer:
    va = VerifiedAnswer(
        submission_id="sub-1",
        query=query,
        answer=answer,
        keywords=extract_keywords(query),
    )
    index = verified_notebooks._load_index()
    index.setdefault("verified_answers", {})[va.answer_id] = va.model_dump()
    verified_notebooks._save_index(index)
    return va


def _seed_notebook(query: str, answer: str) -> VerifiedNotebook:
    nb = VerifiedNotebook(
        submission_id="sub-nb-1",
        query=query,
        answer=answer,
        notebook_json={"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5},
        keywords=extract_keywords(query),
    )
    index = verified_notebooks._load_index()
    index.setdefault("verified", {})[nb.notebook_id] = nb.model_dump()
    verified_notebooks._save_index(index)
    return nb


# ---------------------------------------------------------------------------
# extract_keywords — year capture (issue #64 root cause)
# ---------------------------------------------------------------------------


class TestExtractKeywordsYears:
    def test_captures_four_digit_year(self) -> None:
        assert "2023" in extract_keywords("crime in Pittsburgh in 2023")

    def test_distinguishes_years(self) -> None:
        kw_2023 = extract_keywords("crime in 2023")
        kw_2025 = extract_keywords("crime in 2025")
        assert "2023" in kw_2023 and "2023" not in kw_2025
        assert "2025" in kw_2025 and "2025" not in kw_2023

    def test_ignores_non_year_digits(self) -> None:
        # Three- and five-digit numbers are not year tokens.
        kws = extract_keywords("permits 123 and 99999 issued")
        assert "123" not in kws
        assert "99999" not in kws

    def test_still_extracts_words(self) -> None:
        kws = extract_keywords("unemployment rate in Texas")
        assert "unemployment" in kws
        assert "texas" in kws


# ---------------------------------------------------------------------------
# LLM gate helpers
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_plain_json(self) -> None:
        out = _parse_response('{"is_match": true, "confidence": 0.9, "reason": "ok"}')
        assert out == {"is_match": True, "confidence": 0.9, "reason": "ok"}

    def test_json_with_prose_and_fences(self) -> None:
        raw = 'Here is my verdict:\n```json\n{"is_match": false, "confidence": 0.1, "reason": "year differs"}\n```'
        out = _parse_response(raw)
        assert out is not None
        assert out["is_match"] is False
        assert out["reason"] == "year differs"

    def test_confidence_clamped(self) -> None:
        out = _parse_response('{"is_match": true, "confidence": 5, "reason": "x"}')
        assert out is not None
        assert out["confidence"] == 1.0

    def test_missing_is_match_returns_none(self) -> None:
        assert _parse_response('{"confidence": 0.9}') is None

    def test_garbage_returns_none(self) -> None:
        assert _parse_response("not json at all") is None
        assert _parse_response("") is None


class TestCircuitBreaker:
    def test_opens_after_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            match_verifier.settings, "verified_match_circuit_breaker_threshold", 2
        )
        cb = _CircuitBreaker()
        assert not cb.is_open
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open

    def test_success_resets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            match_verifier.settings, "verified_match_circuit_breaker_threshold", 2
        )
        cb = _CircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open
        cb.record_success()
        assert not cb.is_open


class TestGateAvailability:
    def test_disabled_by_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(match_verifier.settings, "verified_match_llm_enabled", False)
        assert llm_gate_available() is False

    def test_open_breaker_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(match_verifier.settings, "verified_match_llm_enabled", True)
        monkeypatch.setattr(match_verifier, "ANTHROPIC_AVAILABLE", True)
        monkeypatch.setattr(
            match_verifier.settings.anthropic_api_key, "get_secret_value", lambda: "key"
        )
        monkeypatch.setattr(match_verifier._breaker, "_consecutive_failures", 99)
        assert llm_gate_available() is False


# ---------------------------------------------------------------------------
# Router two-stage flow (the #64 scenario)
# ---------------------------------------------------------------------------


def _force_gate(monkeypatch: pytest.MonkeyPatch, active: bool) -> None:
    monkeypatch.setattr(router, "llm_gate_available", lambda: active)


class TestRouterTwoStageAnswer:
    async def test_temporal_mismatch_rejected_by_gate(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Verified answer is for 2025; user asks about 2023. Keyword retrieval
        # still surfaces it as a candidate, but the gate must reject it.
        _seed_answer("How much crime in Pittsburgh in 2025?", "Crime fell 5% in 2025.")
        _force_gate(monkeypatch, True)

        async def fake_verify(**kwargs: Any) -> dict[str, Any]:
            return {"is_match": False, "confidence": 0.0, "reason": "year mismatch"}

        monkeypatch.setattr(router, "verify_match_with_llm", fake_verify)

        result = await router._check_verified_answer("How much crime in Pittsburgh in 2023?")
        assert result is None

    async def test_true_match_accepted(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        va = _seed_answer("Unemployment rate in Texas in 2023?", "4.1% in 2023.")
        _force_gate(monkeypatch, True)

        async def fake_verify(**kwargs: Any) -> dict[str, Any]:
            return {"is_match": True, "confidence": 0.95, "reason": "exact match"}

        monkeypatch.setattr(router, "verify_match_with_llm", fake_verify)

        result = await router._check_verified_answer("What was the Texas unemployment rate in 2023?")
        assert result is not None
        assert result["answer_id"] == va.answer_id
        assert result["similarity"] == 0.95
        assert result["match_reason"] == "exact match"

    async def test_low_confidence_rejected(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_answer("Unemployment rate in Texas in 2023?", "4.1% in 2023.")
        _force_gate(monkeypatch, True)
        monkeypatch.setattr(
            router.settings, "verified_match_confidence_threshold", 0.80
        )

        async def fake_verify(**kwargs: Any) -> dict[str, Any]:
            return {"is_match": True, "confidence": 0.5, "reason": "weak"}

        monkeypatch.setattr(router, "verify_match_with_llm", fake_verify)

        result = await router._check_verified_answer("Texas unemployment rate 2023?")
        assert result is None

    async def test_keyword_fallback_when_gate_unavailable(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Identical query → very high keyword score → fallback accepts it.
        va = _seed_answer("Unemployment rate in Texas in 2023?", "4.1% in 2023.")
        _force_gate(monkeypatch, False)
        monkeypatch.setattr(router.settings, "verified_match_fallback_threshold", 0.75)

        result = await router._check_verified_answer("Unemployment rate in Texas in 2023?")
        assert result is not None
        assert result["answer_id"] == va.answer_id
        assert "match_reason" not in result

    async def test_no_candidates_returns_none(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_gate(monkeypatch, True)
        result = await router._check_verified_answer("totally unrelated topic xyzzy")
        assert result is None


class TestRouterTwoStageNotebook:
    async def test_temporal_mismatch_rejected(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_notebook(
            "How many building permits in Pittsburgh in 2025?", "1,200 permits in 2025."
        )
        _force_gate(monkeypatch, True)

        async def fake_verify(**kwargs: Any) -> dict[str, Any]:
            return {"is_match": False, "confidence": 0.0, "reason": "year mismatch"}

        monkeypatch.setattr(router, "verify_match_with_llm", fake_verify)

        result = await router._check_verified_notebook(
            "How many building permits in Pittsburgh in 2023?"
        )
        assert result is None

    async def test_true_match_accepted(
        self, tmp_storage: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nb = _seed_notebook(
            "How many building permits in Pittsburgh in 2023?", "1,200 permits in 2023."
        )
        _force_gate(monkeypatch, True)

        async def fake_verify(**kwargs: Any) -> dict[str, Any]:
            return {"is_match": True, "confidence": 0.9, "reason": "same year"}

        monkeypatch.setattr(router, "verify_match_with_llm", fake_verify)

        result = await router._check_verified_notebook(
            "Building permits issued in Pittsburgh during 2023?"
        )
        assert result is not None
        assert result["notebook_id"] == nb.notebook_id
        assert result["match_reason"] == "same year"
