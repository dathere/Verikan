"""Provenance must distinguish data coverage from access, update and review dates."""

from data_concierge.core.models import Citation, DataSource, ToolCallSignals
from data_concierge.gateway.answer_metadata import (
    AnswerEvidence,
    attach_notebook_evidence,
    evidence_from_saved,
    evidence_from_state,
)
from data_concierge.gateway.verified_notebooks import VerifiedAnswer, VerifiedNotebook


def test_observation_dates_determine_coverage_not_question_or_vintage() -> None:
    evidence = evidence_from_state(
        {
            "query": "What happened in 2026?",
            "retrieved_data": {
                "observations": [{"date": "2022-04"}, {"date": "2020-01"}],
                "data_vintage": "2025-09-01",
            },
            "citations": [
                {
                    "source": "Census",
                    "url": "https://data.census.gov/",
                    "access_date": "2026-09-15",
                }
            ],
        },
        original_query="What happened in 2026?",
    )
    assert evidence.data_period is not None
    assert evidence.data_period.model_dump() == {"start": "2020-01", "end": "2022-04"}
    assert evidence.retrieved_at == "2026-09-15"
    assert evidence.source_updated_at == "2025-09-01"
    assert evidence.verified_at is None
    assert evidence.verification_status == "unreviewed"


def test_llm_resource_modified_signal_does_not_claim_data_coverage() -> None:
    source = DataSource(id="test", name="Open data", url="https://example.com/data")
    evidence = evidence_from_state(
        {
            "citations": [
                Citation(
                    source=source,
                    dataset_title="Population data",
                    url=source.url,
                    access_date="2026-09-15",
                )
            ],
            "tool_call_signals": ToolCallSignals(resource_metadata_modified="2025-02-03T12:00:00Z"),
        },
        original_query="Population in 2026",
    )
    assert evidence.source_updated_at == "2025-02-03T12:00:00Z"
    assert evidence.data_period is None
    assert evidence.sources[0].description == "Population data"
    assert evidence.sources[0].name == "Open data"


def test_legacy_notebook_never_uses_generation_or_default_review_date() -> None:
    notebook = VerifiedNotebook(
        submission_id="old-submission",
        query="The 2023 unemployment rate",
        notebook_json={
            "metadata": {"data_concierge": {"generated": "2026-09-15", "query": "The 2023 rate"}},
            "cells": [],
        },
    )
    evidence = evidence_from_saved(notebook, reviewed=True)
    assert evidence.verification_status == "reviewed"
    assert evidence.verified_at is None  # The model's default factory is not provenance.
    assert evidence.data_period is None
    assert evidence.retrieved_at is None
    assert evidence.source_updated_at is None
    assert evidence.sources == []


def test_notebook_round_trip_keeps_original_retrieval_date_on_reuse() -> None:
    generated = evidence_from_state(
        {
            "retrieved_data": {"observations": [{"date": "2024"}]},
            "citations": [{"url": "https://example.com", "access_date": "2025-01-03"}],
        },
        original_query="Population",
    )
    notebook = {"cells": [], "metadata": {"data_concierge": {"query": "Population"}}}
    attach_notebook_evidence(notebook, generated)
    saved = evidence_from_saved(
        {"notebook_json": notebook, "query": "Population", "verified_at": "2025-02-04"},
        reviewed=True,
    )
    assert saved.retrieved_at == "2025-01-03"
    assert saved.verified_at == "2025-02-04"
    assert saved.data_period == generated.data_period
    assert saved.sources == generated.sources


def test_saved_quick_answer_uses_explicit_observation_date() -> None:
    answer = VerifiedAnswer(
        submission_id="submission",
        query="Population now",
        answer="12,000 people",
        date="2024",
        verified_at="2026-08-01",
        source_links=[{"name": "Census", "url": "https://data.census.gov/"}],
    )
    evidence = evidence_from_saved(answer, reviewed=True)
    assert evidence.data_period is not None
    assert evidence.data_period.start == evidence.data_period.end == "2024"
    assert evidence.verified_at == "2026-08-01"
    assert evidence.retrieved_at is None
    assert evidence.sources[0].name == "Census"


def test_sidecar_restores_quick_answer_provenance_without_self_attesting_review() -> None:
    evidence = evidence_from_saved(
        {"query": "Population", "verified_at": "2026-08-01"},
        reviewed=True,
        stored_evidence={
            "data_period": {"start": "2023", "end": "2024"},
            "retrieved_at": "2025-01-01",
            "verified_at": "2099-01-01",
            "verification_status": "pending",
        },
    )
    assert evidence.retrieved_at == "2025-01-01"
    assert evidence.verified_at == "2026-08-01"
    assert evidence.verification_status == "reviewed"
    assert evidence.data_period is not None and evidence.data_period.end == "2024"


def test_unreviewed_notebook_cannot_claim_review_in_its_metadata() -> None:
    evidence = evidence_from_saved(
        {"query": "Population"},
        stored_evidence={"verification_status": "reviewed", "verified_at": "2025-01-01"},
    )
    assert evidence.verification_status == "unknown"
    assert evidence.verified_at is None


def test_sources_are_deduplicated_and_nonweb_or_credential_urls_are_excluded() -> None:
    evidence = evidence_from_state(
        {
            "source_links": [
                {"url": "https://example.com/data", "name": "Data"},
                {"url": "javascript:alert(1)", "name": "Unsafe"},
                {"url": "https://user:password@example.com/data"},
                {"url": "file:///tmp/data"},
                {"url": "https://[invalid"},
            ],
            "retrieved_data": {
                "source_info": [{"url": "https://example.com/data", "name": "Data"}]
            },
        },
        original_query="Data",
    )
    assert [source.url for source in evidence.sources] == ["https://example.com/data"]


def test_invalid_dates_are_unknown_and_reversed_period_is_not_displayed() -> None:
    evidence = evidence_from_saved(
        {},
        stored_evidence={
            "retrieved_at": "recently",
            "source_updated_at": "2025-99-99",
            "data_period": {"start": "2025", "end": "2023"},
        },
    )
    assert evidence.data_period is None
    assert evidence.retrieved_at is None
    assert evidence.source_updated_at is None
    assert evidence == AnswerEvidence()
