"""Verified Notebooks Store.

Manages the lifecycle of notebook submissions:
- Users submit notebooks for admin review after analysis
- Admins approve or reject notebooks
- Approved notebooks become "verified" and are searchable
- New queries are matched against verified notebooks for reuse

Storage is backed by the unified storage layer (GCS in production, local in dev).
"""

import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from data_concierge.core.logging import get_logger
from data_concierge.data_layer.storage import storage

logger = get_logger(__name__)

# Storage key prefixes
_STORE_PREFIX = "verified_notebooks"
_SUBMISSIONS_PREFIX = f"{_STORE_PREFIX}/submissions"
_VERIFIED_PREFIX = f"{_STORE_PREFIX}/verified"
_INDEX_KEY = f"{_STORE_PREFIX}/index.json"


# =============================================================================
# Models
# =============================================================================


class ReviewStatus(str, Enum):
    """Status of a notebook submission."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class NotebookSubmission(BaseModel):
    """A notebook submitted for admin review."""

    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(..., description="Original user query")
    answer: str = Field(default="", description="Generated answer text")
    notebook_json: dict[str, Any] = Field(..., description="Full notebook content")
    filename: str = Field(default="", description="Original notebook filename")
    status: ReviewStatus = Field(default=ReviewStatus.PENDING)
    submitted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    submitted_by: str = Field(default="anonymous", description="User who submitted")
    submitter_email: str | None = Field(default=None, description="Email of submitter (if known)")
    submitter_auth_type: str | None = Field(default=None, description="Auth type: password|auth0")
    reviewed_at: str | None = Field(default=None)
    reviewed_by: str | None = Field(default=None)
    admin_notes: str | None = Field(default=None, description="Admin review notes")
    tags: list[str] = Field(default_factory=list, description="Searchable tags")
    data_source: str = Field(default="", description="Data source used")
    confidence: float = Field(default=0.0, description="Original confidence score")
    input_tokens: int | None = Field(default=None, description="LLM input tokens used")
    output_tokens: int | None = Field(default=None, description="LLM output tokens used")
    # Links the submission to its notebook_verification/{query_id}.json record
    # (execution + adversarial method review), so the admin review modal can
    # show the verdict next to the agent logs. Optional: older submissions
    # predate it, and their auto-submit filenames (auto_{query_id}.ipynb)
    # remain the fallback.
    query_id: str | None = Field(default=None, description="Originating query ID")


class VerifiedNotebook(BaseModel):
    """An approved, verified notebook available for reuse."""

    notebook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = Field(..., description="Original submission ID")
    query: str = Field(..., description="Original query")
    answer: str = Field(default="", description="Generated answer")
    notebook_json: dict[str, Any] = Field(..., description="Notebook content")
    filename: str = Field(default="")
    verified_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    verified_by: str = Field(default="admin")
    admin_notes: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    data_source: str = Field(default="")
    confidence: float = Field(default=0.0)
    submitted_by: str = Field(default="", description="User who originally submitted")
    input_tokens: int | None = Field(default=None, description="LLM input tokens used")
    output_tokens: int | None = Field(default=None, description="LLM output tokens used")
    usage_count: int = Field(default=0, description="Times this notebook was served")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    github_path: str | None = Field(
        default=None,
        description="Path of this notebook in the GitHub repo when GitHub is the source of truth",
    )
    github_synced_at: str | None = Field(
        default=None,
        description="ISO timestamp of the last successful sync from GitHub",
    )
    evidence_package_hash: str | None = Field(
        default=None,
        description=(
            "SHA-256 hash of this notebook's Typed Standards evidence package, "
            "once minted. The content-address for its /api/evidence/{hash}/* URLs."
        ),
    )


class QuickAnswerSubmission(BaseModel):
    """A quick answer submitted for admin review.

    Unlike notebook submissions, quick answers are concise one-line factual
    answers with direct source links, used for simple TIER_1 lookups.
    """

    submission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="One-line factual answer")
    source_links: list[dict[str, str]] = Field(
        default_factory=list,
        description="Direct links to data source pages [{name, url, description}]",
    )
    status: ReviewStatus = Field(default=ReviewStatus.PENDING)
    submitted_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    submitted_by: str = Field(default="anonymous")
    submitter_email: str | None = Field(default=None, description="Email of submitter (if known)")
    submitter_auth_type: str | None = Field(default=None, description="Auth type: password|auth0")
    reviewed_at: str | None = Field(default=None)
    reviewed_by: str | None = Field(default=None)
    admin_notes: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    data_source: str = Field(default="")
    confidence: float = Field(default=0.0)
    input_tokens: int | None = Field(default=None, description="LLM input tokens used")
    output_tokens: int | None = Field(default=None, description="LLM output tokens used")
    variable: str = Field(default="", description="Statistical variable queried")
    place: str = Field(default="", description="Geographic place queried")
    date: str = Field(default="", description="Date of the observation")
    value: str = Field(default="", description="The data value")


class VerifiedAnswer(BaseModel):
    """An approved, verified quick answer available for reuse."""

    answer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    submission_id: str = Field(..., description="Original submission ID")
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Verified one-line answer")
    source_links: list[dict[str, str]] = Field(
        default_factory=list,
        description="Verified direct links to data source pages",
    )
    verified_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    verified_by: str = Field(default="admin")
    admin_notes: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    data_source: str = Field(default="")
    confidence: float = Field(default=0.0)
    submitted_by: str = Field(default="", description="User who originally submitted")
    input_tokens: int | None = Field(default=None, description="LLM input tokens used")
    output_tokens: int | None = Field(default=None, description="LLM output tokens used")
    variable: str = Field(default="")
    place: str = Field(default="")
    date: str = Field(default="")
    value: str = Field(default="")
    usage_count: int = Field(default=0, description="Times this answer was served")
    keywords: list[str] = Field(default_factory=list, description="Extracted keywords")
    github_path: str | None = Field(
        default=None,
        description=(
            "Path of this answer's JSON file in the GitHub repo when "
            "GitHub is the source of truth (issue #46 step 9)."
        ),
    )
    github_synced_at: str | None = Field(
        default=None,
        description=(
            "ISO timestamp of the last successful sync of this answer "
            "from GitHub. Equals verified_at on the approve path so the "
            "local index and the published file agree on the moment of "
            "approval."
        ),
    )


class SearchResult(BaseModel):
    """A search result from verified notebooks."""

    notebook_id: str
    query: str
    answer: str
    similarity_score: float
    tags: list[str]
    verified_at: str
    verified_by: str
    usage_count: int
    download_url: str


class AnswerSearchResult(BaseModel):
    """A search result from verified quick answers."""

    answer_id: str
    query: str
    answer: str
    source_links: list[dict[str, str]]
    similarity_score: float
    tags: list[str]
    verified_at: str
    verified_by: str
    usage_count: int


# =============================================================================
# Keyword Extraction & Similarity
# =============================================================================

# Common stop words to filter out
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "don",
        "now",
        "and",
        "but",
        "or",
        "if",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "it",
        "its",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "they",
        "them",
        "his",
        "her",
        "about",
        "up",
        "down",
    }
)


def add_verification_metadata(
    notebook_json: dict[str, Any],
    *,
    submission_id: str,
    confidence: float | None,
    verified_by: str,
    verified_at: str,
    data_source: str = "",
) -> dict[str, Any]:
    """Return a deep-enough copy of ``notebook_json`` with verification
    provenance merged into its ``metadata.data_concierge`` namespace.

    The generator (``agents/notebook_generator.py``) already seeds that
    namespace with ``version``, ``generated``, ``query``, ``data_source``,
    and ``colab_compatible`` at notebook creation time. This helper adds
    the verification-time fields once an admin approves the notebook:

    * ``submission_id``  — provenance back to the original submission
    * ``confidence``     — the pipeline's confidence at submission time
    * ``verified_by``    — the admin who approved
    * ``verified_at``    — ISO timestamp of approval (must match the
      ``github_synced_at`` recorded in the local index so the two stay
      coherent for audit / disaster-recovery rebuild).

    Notebooks published to GitHub are therefore self-describing: a future
    "rebuild local index from the GitHub repo" pass can reconstruct
    every VerifiedNotebook entry by reading ``metadata.data_concierge``
    from each ``.ipynb`` in the verified folder. That capability is what
    makes the SSOT design (issue #46) actually recoverable.

    Doesn't mutate the input. Pre-existing ``data_concierge`` fields are
    preserved; the new fields are merged in. If ``data_source`` is passed
    and the existing metadata has none, it's set; otherwise the existing
    value wins (the generator's snapshot at creation time is the truth).
    """
    enriched = dict(notebook_json)
    metadata = dict(enriched.get("metadata") or {})
    dc = dict(metadata.get("data_concierge") or {})

    dc["submission_id"] = submission_id
    if confidence is not None:
        dc["confidence"] = confidence
    dc["verified_by"] = verified_by
    dc["verified_at"] = verified_at
    if data_source and not dc.get("data_source"):
        dc["data_source"] = data_source

    metadata["data_concierge"] = dc
    enriched["metadata"] = metadata
    return enriched


def normalize_question(query: str) -> str:
    """Return a canonical form of ``query`` for duplicate detection.

    Two questions that differ only in casing, surrounding whitespace, internal
    spacing, or trailing punctuation collapse to the same string. This is
    deliberately a *conservative* normalizer (exact match after light
    canonicalisation) — NOT the lossy :func:`extract_keywords`, which strips
    stop words and short tokens and would wrongly merge genuinely different
    questions ("what is unemployment" vs "what is the unemployment rate").

    Used as the dedup key for the verified library so the same question can't
    occupy multiple verified entries (issue: 7 verified answers all for
    "What is the unemployment rate in Texas?").
    """
    s = (query or "").strip().lower()
    s = re.sub(r"\s+", " ", s)  # collapse internal whitespace
    s = s.strip(" \t\r\n.?!\"'`")  # drop surrounding punctuation/quotes
    return s


def extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text.

    Uses simple tokenization and stop word removal. Four-digit years (e.g.
    ``2023``) are preserved as tokens so that temporal constraints surface in
    keyword candidate retrieval — the old ``[a-z]+`` pattern silently dropped
    them, letting a "crime in 2023" query match a "crime in 2025" answer
    (issue #64).
    """
    # Lowercase and tokenize: alphabetic words plus standalone 4-digit years.
    words = re.findall(r"[a-z]+|\b(?:19|20)\d{2}\b", text.lower())
    # Remove stop words and short words (4-digit years pass the length check).
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    # Return unique keywords preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)
    return unique


def compute_similarity(query_keywords: list[str], doc_keywords: list[str]) -> float:
    """Compute keyword-based similarity between a query and document.

    Uses a combination of Jaccard similarity and term overlap scoring.
    """
    if not query_keywords or not doc_keywords:
        return 0.0

    query_set = set(query_keywords)
    doc_set = set(doc_keywords)

    # Jaccard similarity
    intersection = query_set & doc_set
    union = query_set | doc_set
    jaccard = len(intersection) / len(union) if union else 0.0

    # Weighted overlap: fraction of query terms found in document
    query_coverage = len(intersection) / len(query_set) if query_set else 0.0

    # Combined score (weight query coverage more)
    score = 0.4 * jaccard + 0.6 * query_coverage
    return round(score, 4)


# =============================================================================
# Store Operations (using storage backend)
# =============================================================================


def _load_index() -> dict[str, Any]:
    """Load the notebook index from storage."""
    data = storage.read_json(_INDEX_KEY)
    if data:
        # Ensure quick answer sections exist (backward compat)
        if "answer_submissions" not in data:
            data["answer_submissions"] = {}
        if "verified_answers" not in data:
            data["verified_answers"] = {}
        return data
    return {"submissions": {}, "verified": {}, "answer_submissions": {}, "verified_answers": {}}


def _save_index(index: dict[str, Any]) -> None:
    """Save the notebook index to storage."""
    storage.write_json(_INDEX_KEY, index)


def submit_notebook(
    query: str,
    answer: str,
    notebook_json: dict[str, Any],
    filename: str = "",
    submitted_by: str = "anonymous",
    submitter_email: str | None = None,
    submitter_auth_type: str | None = None,
    data_source: str = "",
    confidence: float = 0.0,
    tags: list[str] | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    query_id: str | None = None,
) -> NotebookSubmission:
    """Submit a notebook for admin review.

    Args:
        query: The original user query
        answer: The generated answer
        notebook_json: Full notebook JSON content
        filename: Original notebook filename
        submitted_by: User identifier
        data_source: Data source used
        confidence: Original confidence score
        tags: Optional tags for categorization
        query_id: Originating query ID, linking the submission to its
            verification/review record

    Returns:
        The created submission
    """
    submission = NotebookSubmission(
        query=query,
        answer=answer,
        notebook_json=notebook_json,
        filename=filename,
        submitted_by=submitted_by,
        submitter_email=submitter_email,
        submitter_auth_type=submitter_auth_type,
        data_source=data_source,
        confidence=confidence,
        tags=tags or extract_keywords(query)[:10],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        query_id=query_id,
    )

    # Save notebook file
    notebook_key = f"{_SUBMISSIONS_PREFIX}/{submission.submission_id}.ipynb"
    storage.write_json(notebook_key, notebook_json)

    # Update index
    index = _load_index()
    index["submissions"][submission.submission_id] = submission.model_dump()
    _save_index(index)

    logger.info(
        "Notebook submitted for review",
        submission_id=submission.submission_id,
        query=query[:100],
    )
    return submission


def get_pending_submissions() -> list[NotebookSubmission]:
    """Get all pending notebook submissions."""
    index = _load_index()
    submissions = []
    for data in index.get("submissions", {}).values():
        sub = NotebookSubmission(**data)
        if sub.status == ReviewStatus.PENDING:
            submissions.append(sub)
    # Sort by submission time (newest first)
    submissions.sort(key=lambda s: s.submitted_at, reverse=True)
    return submissions


def get_all_submissions() -> list[NotebookSubmission]:
    """Get all notebook submissions regardless of status."""
    index = _load_index()
    submissions = []
    for data in index.get("submissions", {}).values():
        submissions.append(NotebookSubmission(**data))
    submissions.sort(key=lambda s: s.submitted_at, reverse=True)
    return submissions


def get_submission(submission_id: str) -> NotebookSubmission | None:
    """Get a specific submission by ID."""
    index = _load_index()
    data = index.get("submissions", {}).get(submission_id)
    if data:
        return NotebookSubmission(**data)
    return None


def approve_notebook(
    submission_id: str,
    reviewed_by: str = "admin",
    admin_notes: str | None = None,
    github_path: str | None = None,
    github_synced_at: str | None = None,
    notebook_json: dict[str, Any] | None = None,
) -> VerifiedNotebook | None:
    """Approve a notebook submission, making it verified.

    Args:
        submission_id: The submission to approve
        reviewed_by: Admin identifier
        admin_notes: Optional review notes
        github_path: If GitHub publishing succeeded, the verified-folder path
            of the notebook in the GitHub repo. Persisted on the
            VerifiedNotebook so local state and the GitHub commit are
            recorded together (write-through SSOT).
        github_synced_at: ISO timestamp of the GitHub commit. Should be set
            together with ``github_path``.
        notebook_json: Optional override for the notebook content stored on
            the VerifiedNotebook and on disk. Typically the caller passes
            the result of :func:`add_verification_metadata` so the local
            copy carries the same provenance fields as the copy published
            to GitHub. When ``None`` (default), the submission's original
            ``notebook_json`` is used unchanged — backwards compatible.

    Returns:
        The created verified notebook, or None if submission not found
    """
    index = _load_index()
    sub_data = index.get("submissions", {}).get(submission_id)
    if not sub_data:
        return None

    submission = NotebookSubmission(**sub_data)
    if submission.status != ReviewStatus.PENDING:
        logger.warning(
            "Submission already reviewed",
            submission_id=submission_id,
            status=submission.status,
        )
        return None

    # Use the caller's enriched notebook_json when provided so local + GitHub
    # stay coherent. Otherwise fall back to the submission's original copy.
    final_notebook_json = notebook_json if notebook_json is not None else submission.notebook_json

    # Create verified notebook
    keywords = extract_keywords(submission.query + " " + submission.answer)
    verified = VerifiedNotebook(
        submission_id=submission_id,
        query=submission.query,
        answer=submission.answer,
        notebook_json=final_notebook_json,
        filename=submission.filename,
        verified_by=reviewed_by,
        admin_notes=admin_notes,
        tags=submission.tags,
        data_source=submission.data_source,
        confidence=submission.confidence,
        submitted_by=submission.submitted_by,
        input_tokens=submission.input_tokens,
        output_tokens=submission.output_tokens,
        keywords=keywords,
        github_path=github_path,
        github_synced_at=github_synced_at,
    )

    # Save verified notebook file (use the enriched copy if provided so the
    # on-disk blob matches what was published to GitHub).
    verified_key = f"{_VERIFIED_PREFIX}/{verified.notebook_id}.ipynb"
    storage.write_json(verified_key, final_notebook_json)

    # Update submission status
    submission.status = ReviewStatus.APPROVED
    submission.reviewed_at = datetime.now(UTC).isoformat()
    submission.reviewed_by = reviewed_by
    submission.admin_notes = admin_notes
    index["submissions"][submission_id] = submission.model_dump()

    # Add to verified index
    index["verified"][verified.notebook_id] = verified.model_dump()
    _save_index(index)

    logger.info(
        "Notebook approved",
        submission_id=submission_id,
        notebook_id=verified.notebook_id,
        github_path=github_path,
    )
    return verified


def reject_notebook(
    submission_id: str,
    reviewed_by: str = "admin",
    admin_notes: str | None = None,
) -> NotebookSubmission | None:
    """Reject a notebook submission.

    Args:
        submission_id: The submission to reject
        reviewed_by: Admin identifier
        admin_notes: Reason for rejection

    Returns:
        The updated submission, or None if not found
    """
    index = _load_index()
    sub_data = index.get("submissions", {}).get(submission_id)
    if not sub_data:
        return None

    submission = NotebookSubmission(**sub_data)
    if submission.status != ReviewStatus.PENDING:
        return None

    submission.status = ReviewStatus.REJECTED
    submission.reviewed_at = datetime.now(UTC).isoformat()
    submission.reviewed_by = reviewed_by
    submission.admin_notes = admin_notes

    index["submissions"][submission_id] = submission.model_dump()
    _save_index(index)

    logger.info(
        "Notebook rejected",
        submission_id=submission_id,
    )
    return submission


def get_verified_notebooks() -> list[VerifiedNotebook]:
    """Get all verified notebooks."""
    index = _load_index()
    notebooks = []
    for data in index.get("verified", {}).values():
        notebooks.append(VerifiedNotebook(**data))
    notebooks.sort(key=lambda n: n.verified_at, reverse=True)
    return notebooks


def get_verified_notebook(notebook_id: str) -> VerifiedNotebook | None:
    """Get a specific verified notebook by ID."""
    index = _load_index()
    data = index.get("verified", {}).get(notebook_id)
    if data:
        return VerifiedNotebook(**data)
    return None


def get_verified_notebook_by_submission(submission_id: str) -> VerifiedNotebook | None:
    """Return the verified notebook produced from ``submission_id``, if any.

    Used by the admin UI to surface a clickable GitHub link on a submission
    once it has been approved (#111). Returns ``None`` for submissions that
    are still pending or were rejected (no verified entry exists).
    """
    index = _load_index()
    for data in index.get("verified", {}).values():
        if data.get("submission_id") == submission_id:
            return VerifiedNotebook(**data)
    return None


def search_verified_notebooks(
    query: str,
    threshold: float = 0.25,
    max_results: int = 5,
) -> list[SearchResult]:
    """Search verified notebooks for similar queries.

    Args:
        query: The search query
        threshold: Minimum similarity score (0-1)
        max_results: Maximum number of results

    Returns:
        List of search results sorted by similarity
    """
    query_keywords = extract_keywords(query)
    if not query_keywords:
        return []

    index = _load_index()
    results = []

    for notebook_id, data in index.get("verified", {}).items():
        notebook = VerifiedNotebook(**data)
        doc_keywords = notebook.keywords or extract_keywords(notebook.query)

        score = compute_similarity(query_keywords, doc_keywords)
        if score >= threshold:
            results.append(
                SearchResult(
                    notebook_id=notebook_id,
                    query=notebook.query,
                    answer=notebook.answer,
                    similarity_score=score,
                    tags=notebook.tags,
                    verified_at=notebook.verified_at,
                    verified_by=notebook.verified_by,
                    usage_count=notebook.usage_count,
                    download_url=f"/api/v1/verified-notebooks/{notebook_id}/download",
                )
            )

    # Sort by similarity (highest first)
    results.sort(key=lambda r: r.similarity_score, reverse=True)
    return results[:max_results]


def update_verified_notebook(
    notebook_id: str,
    *,
    notebook_json: dict[str, Any] | None = None,
    github_path: str | None = None,
    github_synced_at: str | None = None,
    evidence_package_hash: str | None = None,
) -> bool:
    """Patch fields on a verified notebook entry.

    Used by the GitHub-sync flow to record where a notebook lives on
    GitHub and to refresh the locally cached notebook content with
    edits made directly in the GitHub repo.
    """
    index = _load_index()
    entry = index.get("verified", {}).get(notebook_id)
    if not entry:
        return False

    if notebook_json is not None:
        entry["notebook_json"] = notebook_json
        # Keep the file blob in sync with the index for download endpoints
        # that read from storage rather than from the index.
        verified_key = f"{_VERIFIED_PREFIX}/{notebook_id}.ipynb"
        try:
            storage.write_json(verified_key, notebook_json)
        except Exception as e:
            logger.warning(
                "Failed to refresh local verified notebook blob",
                notebook_id=notebook_id,
                error=str(e),
            )
    if github_path is not None:
        entry["github_path"] = github_path
    if github_synced_at is not None:
        entry["github_synced_at"] = github_synced_at
    if evidence_package_hash is not None:
        entry["evidence_package_hash"] = evidence_package_hash

    index["verified"][notebook_id] = entry
    _save_index(index)
    return True


async def sync_all_from_github() -> dict[str, Any]:
    """Pull the latest content for every verified notebook from GitHub.

    Treats the configured GitHub repo as the source of truth and refreshes
    each verified notebook's locally cached content if a matching file
    exists on GitHub. Returns a summary dict.

    When GitHub publishing is disabled or unconfigured, returns
    ``{"skipped": True, ...}`` without raising.
    """
    # Imported here to avoid a circular import at module load time.
    from data_concierge.gateway.github_publisher import (
        fetch_notebook,
        is_publishing_active,
        list_verified_files,
        load_github_settings,
    )

    gh_settings = load_github_settings()
    if not is_publishing_active(gh_settings):
        return {
            "skipped": True,
            "reason": "GitHub publishing not active (not configured or paused)",
            "updated": [],
            "failed": [],
            "missing_path": [],
            "total_in_repo": 0,
        }

    repo_files = await list_verified_files()
    by_name: dict[str, str] = {f["name"]: f["path"] for f in repo_files if f.get("name")}

    notebooks = get_verified_notebooks()
    updated: list[str] = []
    missing: list[str] = []
    failed: list[str] = []
    now = datetime.utcnow().isoformat() + "Z"

    for nb in notebooks:
        path = nb.github_path
        if not path and nb.filename and nb.filename in by_name:
            path = by_name[nb.filename]
        if not path:
            missing.append(nb.notebook_id)
            continue

        fresh = await fetch_notebook(path)
        if fresh is None:
            failed.append(nb.notebook_id)
            continue

        ok = update_verified_notebook(
            nb.notebook_id,
            notebook_json=fresh,
            github_path=path,
            github_synced_at=now,
        )
        if ok:
            updated.append(nb.notebook_id)
        else:
            failed.append(nb.notebook_id)

    return {
        "skipped": False,
        "updated": updated,
        "failed": failed,
        "missing_path": missing,
        "total_in_repo": len(repo_files),
    }


async def bootstrap_index_from_github() -> dict[str, Any]:
    """Rebuild the local verified-notebook index from the GitHub repo.

    This is the disaster-recovery counterpart to :func:`sync_all_from_github`.
    Where ``sync_all_from_github`` iterates the LOCAL index and refreshes
    each entry from GitHub (a no-op when the local index is empty), this
    function iterates the GITHUB ``verified/`` folder and ensures every
    notebook there has a local index entry — reconstructing entries from
    each notebook's embedded ``metadata.data_concierge`` provenance (step 4
    populates these fields at approval time).

    Use cases:

    * Cold start on a fresh Cloud Run instance with no ``GCS_BUCKET``
      configured — local storage is ephemeral, the index is empty, and
      everything must be rebuilt from GitHub.
    * Recovery after a corrupted or wiped ``index.json``.
    * Onboarding: copy the GitHub repo into a new environment and
      reconstruct the local view from it.

    Per-file outcomes are non-destructive for fields the local index owns:

    * Notebook on GitHub with a matching local entry (same ``submission_id``
      from embedded metadata): the cached ``notebook_json`` is refreshed,
      ``github_path`` / ``github_synced_at`` are recorded, and the
      metadata-derived fields (``query``, ``confidence``, ``verified_by``,
      ``verified_at``, ``data_source``) are refreshed from the embedded
      metadata so admin edits in GitHub propagate to the local index that
      ``get_verified_notebooks()`` / ``search_verified_notebooks()`` read
      from. Local-only operational fields (``usage_count``, ``admin_notes``,
      ``tags``, ``keywords``) are preserved. Reported under ``updated``.
      If the embedded metadata fails validation (e.g. non-numeric
      ``confidence``), the existing local entry is left UNTOUCHED and the
      path is reported under ``skipped_bad_metadata`` instead — a bad
      republish MUST NOT silently overwrite a good cached entry.
    * Notebook on GitHub with no matching local entry: a new
      ``VerifiedNotebook`` is created from the embedded metadata. A fresh
      ``notebook_id`` is generated (the previous notebook_id is not in
      the notebook itself, by design — it's a local-only handle).
      Reported under ``created``.
    * Notebook on GitHub with no ``metadata.data_concierge`` namespace
      (e.g. uploaded manually before this provenance scheme existed):
      cannot rebuild, reported under ``skipped_no_metadata`` for manual
      triage.
    * Notebook with embedded metadata that's malformed (non-numeric
      ``confidence``, wrong field types, Pydantic validation fails):
      reported under ``skipped_bad_metadata`` so admins know which files
      to fix. The bootstrap continues with the remaining files — one
      bad notebook MUST NOT crash the entire recovery.
    * Two GitHub files sharing the same ``submission_id``: the first
      occurrence is processed normally; subsequent duplicates are
      reported under ``skipped_duplicate_submission_id`` so admins can
      deduplicate. Without this guard, a duplicate with no matching
      local entry would create a second ``VerifiedNotebook`` row with a
      different ``notebook_id`` but the same ``submission_id`` —
      corrupting the index and breaking orphan detection.
    * Notebook with embedded metadata but the fetch failed: reported
      under ``failed``.
    * Local index entry with no corresponding GitHub file: NOT touched;
      reported under ``orphaned_locally`` so admins can investigate.

    Raises :class:`GitHubPublishError` when the underlying GitHub
    listing fails (strict mode — see :func:`_list_folder_files`). The
    endpoint maps that to 502 so admins see real failures instead of a
    misleading "checked: 0" report.
    """
    # Imported here to avoid a circular import at module load time.
    from data_concierge.gateway.github_publisher import (
        _list_folder_files,
        fetch_notebook,
        is_publishing_active,
        load_github_settings,
    )

    gh_settings = load_github_settings()
    if not is_publishing_active(gh_settings):
        return {
            "skipped": True,
            "reason": "GitHub publishing not active (not configured or paused)",
            "checked": 0,
            "created": [],
            "updated": [],
            "failed": [],
            "skipped_no_metadata": [],
            "skipped_bad_metadata": [],
            "skipped_duplicate_submission_id": [],
            "orphaned_locally": [],
        }

    verified_folder = gh_settings["verified_folder"]
    # Strict listing — admin disaster-recovery must surface real failures.
    repo_files = await _list_folder_files(verified_folder, strict=True)

    # Build a submission_id → existing notebook_id map from the local index
    # so we can decide create-vs-update without rescanning per file.
    index = _load_index()
    local_verified = index.get("verified", {})
    by_submission_id: dict[str, str] = {}
    for nb_id, entry in local_verified.items():
        sub_id = entry.get("submission_id")
        if sub_id:
            by_submission_id[sub_id] = nb_id

    created: list[str] = []
    updated: list[str] = []
    failed: list[str] = []
    skipped_no_metadata: list[str] = []
    skipped_bad_metadata: list[str] = []
    skipped_duplicate_submission_id: list[str] = []
    sync_ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    seen_submission_ids: set[str] = set()

    for f in repo_files:
        path = f.get("path") or f"{verified_folder}/{f.get('name')}"
        notebook_json = await fetch_notebook(path)
        if notebook_json is None:
            failed.append(path)
            continue

        # Defensive type guards: a malformed .ipynb (top-level isn't a dict,
        # metadata or data_concierge is a string/list, submission_id is
        # unhashable, etc.) used to crash the loop with AttributeError /
        # TypeError BEFORE reaching the VerifiedNotebook try/except below.
        # Validate the shape up front and route corruption to
        # ``skipped_bad_metadata`` (distinct from ``skipped_no_metadata``
        # which means the namespace genuinely isn't there).
        if not isinstance(notebook_json, dict):
            skipped_bad_metadata.append(path)
            continue
        metadata = notebook_json.get("metadata")
        if metadata is None:
            skipped_no_metadata.append(path)
            continue
        if not isinstance(metadata, dict):
            skipped_bad_metadata.append(path)
            continue
        dc = metadata.get("data_concierge")
        if dc is None:
            skipped_no_metadata.append(path)
            continue
        if not isinstance(dc, dict):
            skipped_bad_metadata.append(path)
            continue
        sub_id = dc.get("submission_id")
        if sub_id is None or sub_id == "":
            # Namespace exists but no submission_id at all — closer to
            # "no metadata" than "bad metadata" from the admin's view.
            skipped_no_metadata.append(path)
            continue
        if not isinstance(sub_id, str):
            # A list/dict here is corrupt: it can't be hashed (set.add
            # would TypeError) and it can't index the by_submission_id map.
            skipped_bad_metadata.append(path)
            continue

        # Guard against two GitHub files claiming the same submission_id.
        # Without this, a duplicate where neither path has a local entry
        # yet would fall into the create branch twice and produce two
        # ``VerifiedNotebook`` rows with the same submission_id — index
        # corruption that breaks orphan detection and search.
        if sub_id in seen_submission_ids:
            skipped_duplicate_submission_id.append(path)
            continue
        seen_submission_ids.add(sub_id)
        existing_nb_id = by_submission_id.get(sub_id)

        if existing_nb_id:
            # Build a candidate merged entry, then round-trip it through the
            # ``VerifiedNotebook`` model BEFORE writing anything. This catches
            # type-corrupt fields the explicit-field validation misses
            # (e.g. ``verified_by`` as a list, ``verified_at`` as an int).
            # Without this, the entry would land in the index and the next
            # ``VerifiedNotebook(**data)`` read in get/search would raise a
            # ValidationError as a 500 instead of a graceful skip.
            #
            # On failure: route to ``skipped_bad_metadata`` and leave the
            # existing local entry UNTOUCHED — a bad republish MUST NOT
            # silently overwrite a good cached entry / blob.
            try:
                raw_confidence = dc.get("confidence")
                try:
                    confidence_val = float(raw_confidence) if raw_confidence is not None else 0.0
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"confidence={raw_confidence!r} is not numeric") from exc

                # Refresh GitHub-owned fields AND the metadata-derived ones so
                # admin edits in GitHub propagate back to the local index that
                # search/listing reads from. Preserve local-only operational
                # state (usage_count, admin_notes, tags, keywords) by starting
                # from the existing entry and overlaying changes.
                candidate = dict(local_verified[existing_nb_id])
                candidate["notebook_json"] = notebook_json
                candidate["github_path"] = path
                candidate["github_synced_at"] = sync_ts
                candidate["query"] = dc.get("query") or ""
                candidate["confidence"] = confidence_val
                candidate["verified_by"] = dc.get("verified_by", "admin")
                candidate["verified_at"] = dc.get("verified_at", sync_ts)
                candidate["data_source"] = dc.get("data_source", "")
                # Round-trip through the model to surface ANY type error now
                # (verified_by must be str, verified_at must be str, etc.)
                # instead of when get_verified_notebook is called later.
                validated_entry = VerifiedNotebook(**candidate).model_dump()
            except Exception as e:
                logger.warning(
                    "bootstrap: failed to validate metadata for existing"
                    " entry — leaving local entry untouched",
                    path=path,
                    submission_id=sub_id,
                    notebook_id=existing_nb_id,
                    error=str(e),
                )
                skipped_bad_metadata.append(path)
                continue

            local_verified[existing_nb_id] = validated_entry
            # Refresh the on-disk blob so /download serves the latest content.
            blob_key = f"{_VERIFIED_PREFIX}/{existing_nb_id}.ipynb"
            try:
                storage.write_json(blob_key, notebook_json)
            except Exception as e:
                logger.warning(
                    "bootstrap: failed to refresh verified blob",
                    notebook_id=existing_nb_id,
                    error=str(e),
                )
            updated.append(existing_nb_id)
            continue

        # New entry — reconstruct from embedded metadata.
        #
        # All field normalization and the VerifiedNotebook construction are
        # wrapped in a single try/except. A malformed metadata.data_concierge
        # (non-numeric confidence, wrong field type, Pydantic validation
        # failure, etc.) must NOT crash the whole bootstrap — one bad file
        # would otherwise wipe out the progress of every later file in the
        # iteration. Route exceptions into a dedicated bucket so admins can
        # see exactly which files need manual triage.
        try:
            query = dc.get("query") or ""
            keywords = extract_keywords(query)
            raw_confidence = dc.get("confidence")
            try:
                confidence_val = float(raw_confidence) if raw_confidence is not None else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"confidence={raw_confidence!r} is not numeric") from exc
            new = VerifiedNotebook(
                submission_id=sub_id,
                query=query,
                answer="",  # not embedded; admins can fill in via existing edit paths
                notebook_json=notebook_json,
                filename=f.get("name", ""),
                verified_by=dc.get("verified_by", "admin"),
                verified_at=dc.get("verified_at", sync_ts),
                data_source=dc.get("data_source", ""),
                confidence=confidence_val,
                keywords=keywords,
                github_path=path,
                github_synced_at=sync_ts,
            )
        except Exception as e:
            logger.warning(
                "bootstrap: failed to reconstruct VerifiedNotebook from metadata",
                path=path,
                submission_id=sub_id,
                error=str(e),
            )
            skipped_bad_metadata.append(path)
            continue

        local_verified[new.notebook_id] = new.model_dump()
        # Seed the on-disk blob so the new entry is fully usable immediately.
        blob_key = f"{_VERIFIED_PREFIX}/{new.notebook_id}.ipynb"
        try:
            storage.write_json(blob_key, notebook_json)
        except Exception as e:
            logger.warning(
                "bootstrap: failed to seed verified blob for new entry",
                notebook_id=new.notebook_id,
                error=str(e),
            )
        created.append(new.notebook_id)

    # Detect local entries with no GitHub counterpart so admins can decide
    # whether to keep, republish, or hard-delete them. NOT destructive.
    orphaned_locally: list[str] = []
    for sub_id, nb_id in by_submission_id.items():
        if sub_id not in seen_submission_ids:
            orphaned_locally.append(nb_id)

    index["verified"] = local_verified
    _save_index(index)

    logger.info(
        "Bootstrapped index from GitHub",
        checked=len(repo_files),
        created=len(created),
        updated=len(updated),
        failed=len(failed),
        skipped_no_metadata=len(skipped_no_metadata),
        skipped_bad_metadata=len(skipped_bad_metadata),
        skipped_duplicate_submission_id=len(skipped_duplicate_submission_id),
        orphaned_locally=len(orphaned_locally),
    )
    return {
        "skipped": False,
        "checked": len(repo_files),
        "created": created,
        "updated": updated,
        "failed": failed,
        "skipped_no_metadata": skipped_no_metadata,
        "skipped_bad_metadata": skipped_bad_metadata,
        "skipped_duplicate_submission_id": skipped_duplicate_submission_id,
        "orphaned_locally": orphaned_locally,
    }


async def sync_one_from_github(path: str) -> dict[str, Any]:
    """Sync a single verified notebook from GitHub by path (upsert).

    Per-path counterpart to :func:`bootstrap_index_from_github`. Fetches
    the notebook at ``path``, validates its embedded
    ``metadata.data_concierge`` provenance, and creates or refreshes the
    corresponding local index entry. The webhook handler (step 7) will
    call this for each file in a push event's added/modified list.
    Removed paths are handled separately at the webhook layer — this
    function does NOT delete.

    Reuses the same validation pattern as the bootstrap update path:
    shape guards before any ``.get()`` so a malformed ``.ipynb`` can't
    crash with ``AttributeError`` / ``TypeError``; numeric ``confidence``
    validation up front; the merged candidate dict is round-tripped
    through :class:`VerifiedNotebook` BEFORE writing — any pydantic
    ``ValidationError`` leaves the existing local entry UNTOUCHED so a
    bad republish can't corrupt the index.

    Returns a dict describing the per-file outcome:

    * ``{"status": "skipped_disabled", "path": ...}`` — GitHub publishing
      not active (not configured or paused).
    * ``{"status": "failed", "path": ...}`` — GitHub fetch returned
      ``None`` (network failure, auth failure, or file genuinely missing;
      step 6 does NOT distinguish — step 7's webhook only invokes this
      for paths the event listed as present).
    * ``{"status": "skipped_no_metadata", "path": ...}`` — file fetched
      but has no ``metadata.data_concierge`` namespace; cannot rebuild.
    * ``{"status": "skipped_bad_metadata", "path": ..., "reason": ...}``
      — namespace exists but is shape-corrupt or fails Pydantic
      validation.
    * ``{"status": "created", "notebook_id": ..., "path": ...}`` — no
      matching local entry; a new :class:`VerifiedNotebook` was created.
    * ``{"status": "updated", "notebook_id": ..., "path": ...}`` —
      existing local entry refreshed in place.

    Edge case (NOT handled here): if a local entry has the same
    ``github_path`` but a different ``submission_id`` (admin replaced
    the file with one carrying different metadata), this function will
    create a new entry rather than overwrite the old one. The orphaned
    entry will surface in the next bootstrap as ``orphaned_locally`` so
    admins can decide what to do.
    """
    # Imported here to avoid a circular import at module load time.
    from data_concierge.gateway.github_publisher import (
        fetch_notebook,
        is_publishing_active,
        load_github_settings,
    )

    gh_settings = load_github_settings()
    if not is_publishing_active(gh_settings):
        return {"status": "skipped_disabled", "path": path}

    notebook_json = await fetch_notebook(path)
    if notebook_json is None:
        logger.warning("sync_one: GitHub fetch failed or path not found", path=path)
        return {"status": "failed", "path": path}

    # Defensive shape guards — see bootstrap_index_from_github for why
    # each isinstance check is necessary. A corrupt .ipynb must NOT crash.
    if not isinstance(notebook_json, dict):
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "notebook_json top-level is not a dict",
        }
    metadata = notebook_json.get("metadata")
    if metadata is None:
        return {"status": "skipped_no_metadata", "path": path}
    if not isinstance(metadata, dict):
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "metadata is not a dict",
        }
    dc = metadata.get("data_concierge")
    if dc is None:
        return {"status": "skipped_no_metadata", "path": path}
    if not isinstance(dc, dict):
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "metadata.data_concierge is not a dict",
        }
    sub_id = dc.get("submission_id")
    if sub_id is None or sub_id == "":
        return {"status": "skipped_no_metadata", "path": path}
    if not isinstance(sub_id, str):
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "submission_id is not a string",
        }

    # Look up existing local entry by submission_id.
    index = _load_index()
    local_verified = index.get("verified", {})
    existing_nb_id: str | None = None
    for nb_id, entry in local_verified.items():
        if entry.get("submission_id") == sub_id:
            existing_nb_id = nb_id
            break

    sync_ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    if existing_nb_id:
        # Update path — mirrors bootstrap's update branch: build a
        # candidate dict, round-trip through VerifiedNotebook so any
        # type-corrupt field surfaces NOW (not on a later read), then
        # write entry + blob.
        try:
            raw_confidence = dc.get("confidence")
            try:
                confidence_val = float(raw_confidence) if raw_confidence is not None else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"confidence={raw_confidence!r} is not numeric") from exc

            # Preserve local-only operational state (usage_count,
            # admin_notes, tags, keywords) by starting from the existing
            # entry and overlaying GitHub-owned fields.
            candidate = dict(local_verified[existing_nb_id])
            candidate["notebook_json"] = notebook_json
            candidate["github_path"] = path
            candidate["github_synced_at"] = sync_ts
            candidate["query"] = dc.get("query") or ""
            candidate["confidence"] = confidence_val
            candidate["verified_by"] = dc.get("verified_by", "admin")
            candidate["verified_at"] = dc.get("verified_at", sync_ts)
            candidate["data_source"] = dc.get("data_source", "")
            validated_entry = VerifiedNotebook(**candidate).model_dump()
        except Exception as e:
            logger.warning(
                "sync_one: failed to validate metadata for existing"
                " entry — leaving local entry untouched",
                path=path,
                submission_id=sub_id,
                notebook_id=existing_nb_id,
                error=str(e),
            )
            return {
                "status": "skipped_bad_metadata",
                "path": path,
                "reason": str(e),
            }

        local_verified[existing_nb_id] = validated_entry
        blob_key = f"{_VERIFIED_PREFIX}/{existing_nb_id}.ipynb"
        try:
            storage.write_json(blob_key, notebook_json)
        except Exception as e:
            logger.warning(
                "sync_one: failed to refresh verified blob",
                notebook_id=existing_nb_id,
                error=str(e),
            )

        index["verified"] = local_verified
        _save_index(index)
        logger.info(
            "Synced one notebook from GitHub (updated)",
            path=path,
            notebook_id=existing_nb_id,
            submission_id=sub_id,
        )
        return {
            "status": "updated",
            "notebook_id": existing_nb_id,
            "path": path,
        }

    # Create path — mirrors bootstrap's create branch. Wrapped in a
    # single try/except so corrupt embedded metadata routes to
    # skipped_bad_metadata instead of raising at the call site.
    try:
        query = dc.get("query") or ""
        keywords = extract_keywords(query)
        raw_confidence = dc.get("confidence")
        try:
            confidence_val = float(raw_confidence) if raw_confidence is not None else 0.0
        except (TypeError, ValueError) as exc:
            raise ValueError(f"confidence={raw_confidence!r} is not numeric") from exc
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        new = VerifiedNotebook(
            submission_id=sub_id,
            query=query,
            answer="",  # not embedded; admins can fill in via edit paths
            notebook_json=notebook_json,
            filename=filename,
            verified_by=dc.get("verified_by", "admin"),
            verified_at=dc.get("verified_at", sync_ts),
            data_source=dc.get("data_source", ""),
            confidence=confidence_val,
            keywords=keywords,
            github_path=path,
            github_synced_at=sync_ts,
        )
    except Exception as e:
        logger.warning(
            "sync_one: failed to reconstruct VerifiedNotebook from metadata",
            path=path,
            submission_id=sub_id,
            error=str(e),
        )
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": str(e),
        }

    local_verified[new.notebook_id] = new.model_dump()
    blob_key = f"{_VERIFIED_PREFIX}/{new.notebook_id}.ipynb"
    try:
        storage.write_json(blob_key, notebook_json)
    except Exception as e:
        logger.warning(
            "sync_one: failed to seed verified blob for new entry",
            notebook_id=new.notebook_id,
            error=str(e),
        )

    index["verified"] = local_verified
    _save_index(index)
    logger.info(
        "Synced one notebook from GitHub (created)",
        path=path,
        notebook_id=new.notebook_id,
        submission_id=sub_id,
    )
    return {"status": "created", "notebook_id": new.notebook_id, "path": path}


def delete_local_by_github_path(path: str) -> dict[str, Any]:
    """Remove the local index entry whose ``github_path`` equals ``path``.

    Counterpart to :func:`sync_one_from_github` for webhook DELETE events
    (file removed from the GitHub repo). Bootstrap is deliberately
    non-destructive (orphaned entries are reported, never deleted) but
    the live webhook path needs an active deletion verb: when admin
    deletes a notebook in GitHub, the local cache must drop it too.

    Returns:

    * ``{"status": "deleted", "notebook_id": ..., "path": ...}`` — local
      entry was found and removed (both index entry and on-disk blob).
    * ``{"status": "skipped_no_local", "path": ...}`` — no local entry
      pointed at this path; nothing to do.

    Does NOT touch local-only entries whose ``github_path`` was never
    set; those are out-of-scope for webhook events (they live entirely
    in local storage and never had a GitHub counterpart).
    """
    index = _load_index()
    local_verified = index.get("verified", {})
    target_nb_id: str | None = None
    for nb_id, entry in local_verified.items():
        if entry.get("github_path") == path:
            target_nb_id = nb_id
            break

    if target_nb_id is None:
        logger.info("delete_local_by_github_path: no local entry for path", path=path)
        return {"status": "skipped_no_local", "path": path}

    # Drop the on-disk blob first (best-effort). If this fails the index
    # removal still proceeds — leaving a stale blob is preferable to a
    # zombie index entry that points at a file the user can't download.
    blob_key = f"{_VERIFIED_PREFIX}/{target_nb_id}.ipynb"
    try:
        storage.delete(blob_key)
    except Exception as e:
        logger.warning(
            "delete_local_by_github_path: failed to delete blob",
            notebook_id=target_nb_id,
            error=str(e),
        )

    del local_verified[target_nb_id]
    index["verified"] = local_verified
    _save_index(index)
    logger.info(
        "Deleted local verified-notebook entry (github webhook DELETE)",
        path=path,
        notebook_id=target_nb_id,
    )
    return {"status": "deleted", "notebook_id": target_nb_id, "path": path}


async def bootstrap_answers_from_github() -> dict[str, Any]:
    """Rebuild the local verified-answers index from the GitHub repo.

    Step-9 counterpart to :func:`bootstrap_index_from_github`. Iterates
    the GitHub ``verified_answers_folder`` and ensures every
    ``<answer_id>.json`` file there has a local index entry, rebuilding
    entries from the file content itself (the file IS the
    :class:`VerifiedAnswer` — no ``metadata.data_concierge`` wrapper
    like notebooks needed, since the filename === the durable identity).

    Use cases — identical to the notebook bootstrap:

    * Cold start on a fresh Cloud Run instance with no ``GCS_BUCKET``
      configured — local storage is ephemeral, the index is empty, and
      everything must be rebuilt from GitHub.
    * Recovery after a corrupted or wiped ``index.json``.
    * Onboarding: copy the GitHub repo into a new environment and
      reconstruct the local view from it.

    Per-file outcomes are non-destructive for fields the local index owns:

    * File on GitHub with a matching local entry (same ``answer_id``):
      ``query`` / ``answer`` / ``source_links`` / ``verified_at`` /
      ``verified_by`` / ``data_source`` / ``confidence`` /
      ``variable`` / ``place`` / ``date`` / ``value`` /
      ``input_tokens`` / ``output_tokens`` / ``submission_id`` /
      ``submitted_by`` / ``github_path`` / ``github_synced_at`` are
      refreshed from the GitHub file so admin edits propagate.
      Local-only operational state (``usage_count``, ``admin_notes``,
      ``tags``, ``keywords``) is preserved. Reported under ``updated``.
      The merged candidate is round-tripped through ``VerifiedAnswer``
      BEFORE writing — a type-corrupt field routes to
      ``skipped_bad_metadata`` and leaves the local entry UNTOUCHED.
    * File on GitHub with no matching local entry: a new
      ``VerifiedAnswer`` is created from the file. Reported under
      ``created``.
    * File on GitHub that's shape-corrupt (not a JSON dict, missing
      ``answer_id``, fails Pydantic validation): reported under
      ``skipped_bad_metadata`` so admins know which files to fix.
      Bootstrap keeps going — one bad file MUST NOT crash the recovery.
    * Two GitHub files claiming the same ``answer_id`` (extremely
      unusual since the filename IS the id, but possible via manual
      duplication or a copy-then-rename mistake): the first occurrence
      wins; subsequent duplicates go to
      ``skipped_duplicate_answer_id``. Without this guard, both files
      would land in the create branch and overwrite each other.
    * File fetched returned ``None``: reported under ``failed``.
    * Local entry with no corresponding GitHub file: reported under
      ``orphaned_locally`` — NOT deleted (bootstrap is non-destructive).

    Raises :class:`GitHubPublishError` when the underlying GitHub
    listing fails (strict mode), same as the notebook bootstrap. The
    endpoint maps that to 502 so admins see real failures instead of
    a misleading "checked: 0" report.
    """
    # Imported here to avoid a circular import at module load time.
    from data_concierge.gateway.github_publisher import (
        _list_folder_files,
        fetch_notebook,
        is_publishing_active,
        load_github_settings,
    )

    gh_settings = load_github_settings()
    if not is_publishing_active(gh_settings):
        return {
            "skipped": True,
            "reason": "GitHub publishing not active (not configured or paused)",
            "checked": 0,
            "created": [],
            "updated": [],
            "failed": [],
            "skipped_bad_metadata": [],
            "skipped_duplicate_answer_id": [],
            "orphaned_locally": [],
        }

    answers_folder = gh_settings.get("verified_answers_folder", "verified-answers")
    # Strict listing — admin disaster-recovery must surface real failures.
    # ``suffix=".json"`` is required: the default filters to ``.ipynb`` for
    # notebook callers, which would silently drop every answer file and
    # report ``checked: 0`` (an earlier adversarial review).
    repo_files = await _list_folder_files(answers_folder, strict=True, suffix=".json")

    index = _load_index()
    local_answers = index.get("verified_answers", {})
    # ``answer_id`` IS the local index key — no extra map needed.

    created: list[str] = []
    updated: list[str] = []
    failed: list[str] = []
    skipped_bad_metadata: list[str] = []
    skipped_duplicate_answer_id: list[str] = []
    sync_ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    seen_answer_ids: set[str] = set()

    for f in repo_files:
        path = f.get("path") or f"{answers_folder}/{f.get('name')}"
        # fetch_notebook is a misnomer in this context but is the
        # generic JSON fetcher used by the notebook bootstrap too —
        # it does Content-API + base64-decode + json.loads.
        file_json = await fetch_notebook(path)
        if file_json is None:
            failed.append(path)
            continue

        # Defensive shape guards — without these, corruption would
        # raise AttributeError / TypeError before reaching the
        # VerifiedAnswer try/except below and crash the whole loop.
        if not isinstance(file_json, dict):
            skipped_bad_metadata.append(path)
            continue
        answer_id = file_json.get("answer_id")
        if answer_id is None or answer_id == "":
            skipped_bad_metadata.append(path)
            continue
        if not isinstance(answer_id, str):
            # Unhashable list/dict — would TypeError on set.add below.
            skipped_bad_metadata.append(path)
            continue

        if answer_id in seen_answer_ids:
            skipped_duplicate_answer_id.append(path)
            continue
        seen_answer_ids.add(answer_id)

        existing_entry = local_answers.get(answer_id)
        if existing_entry is not None:
            # Update path: refresh GitHub-owned fields, preserve
            # local-only operational state. Round-trip through the
            # model so type-corrupt fields route to skipped_bad_metadata
            # instead of corrupting the index (same pattern used for
            # notebooks).
            try:
                candidate = dict(existing_entry)
                # Refresh every field that lives on GitHub.
                for k in (
                    "query",
                    "answer",
                    "source_links",
                    "verified_at",
                    "verified_by",
                    "data_source",
                    "confidence",
                    "submission_id",
                    "submitted_by",
                    "input_tokens",
                    "output_tokens",
                    "variable",
                    "place",
                    "date",
                    "value",
                ):
                    if k in file_json:
                        candidate[k] = file_json[k]
                # github_path is the literal repo path we listed under;
                # github_synced_at marks this bootstrap moment.
                candidate["github_path"] = path
                candidate["github_synced_at"] = sync_ts
                # answer_id stays the same (it's the lookup key).
                candidate["answer_id"] = answer_id
                validated_entry = VerifiedAnswer(**candidate).model_dump()
            except Exception as e:
                logger.warning(
                    "bootstrap_answers: failed to validate metadata for "
                    "existing entry — leaving local entry untouched",
                    path=path,
                    answer_id=answer_id,
                    error=str(e),
                )
                skipped_bad_metadata.append(path)
                continue
            local_answers[answer_id] = validated_entry
            updated.append(answer_id)
            continue

        # Create path: reconstruct VerifiedAnswer from the file content.
        try:
            candidate = dict(file_json)
            candidate["answer_id"] = answer_id  # explicit, in case missing
            candidate["github_path"] = path
            candidate["github_synced_at"] = sync_ts
            new = VerifiedAnswer(**candidate)
        except Exception as e:
            logger.warning(
                "bootstrap_answers: failed to reconstruct VerifiedAnswer",
                path=path,
                answer_id=answer_id,
                error=str(e),
            )
            skipped_bad_metadata.append(path)
            continue
        local_answers[new.answer_id] = new.model_dump()
        created.append(new.answer_id)

    # Detect local entries with no GitHub counterpart — admins decide
    # whether to keep, republish, or hard-delete them. NOT destructive.
    orphaned_locally: list[str] = []
    for nb_id in local_answers:
        if nb_id not in seen_answer_ids:
            orphaned_locally.append(nb_id)

    index["verified_answers"] = local_answers
    _save_index(index)

    logger.info(
        "Bootstrapped answers index from GitHub",
        checked=len(repo_files),
        created=len(created),
        updated=len(updated),
        failed=len(failed),
        skipped_bad_metadata=len(skipped_bad_metadata),
        skipped_duplicate_answer_id=len(skipped_duplicate_answer_id),
        orphaned_locally=len(orphaned_locally),
    )
    return {
        "skipped": False,
        "checked": len(repo_files),
        "created": created,
        "updated": updated,
        "failed": failed,
        "skipped_bad_metadata": skipped_bad_metadata,
        "skipped_duplicate_answer_id": skipped_duplicate_answer_id,
        "orphaned_locally": orphaned_locally,
    }


async def sync_one_answer_from_github(path: str) -> dict[str, Any]:
    """Sync a single verified answer from GitHub by path (upsert).

    Per-path counterpart to :func:`bootstrap_answers_from_github`,
    mirroring :func:`sync_one_from_github` for notebooks. The webhook
    handler will call this for each file in a push event's added /
    modified list (PR C). Removed paths get a sibling helper.

    Returns a dict describing the per-file outcome:

    * ``{"status": "skipped_disabled", "path": ...}`` — GitHub
      publishing not active (not configured or paused).
    * ``{"status": "failed", "path": ...}`` — fetch returned ``None``.
    * ``{"status": "skipped_bad_metadata", "path": ..., "reason": ...}``
      — file shape-corrupt or fails Pydantic validation.
    * ``{"status": "created", "answer_id": ..., "path": ...}``.
    * ``{"status": "updated", "answer_id": ..., "path": ...}``.

    Type-corrupt update: round-tripped through ``VerifiedAnswer``
    BEFORE writing, so a bad republish leaves the existing local
    entry UNTOUCHED.
    """
    from data_concierge.gateway.github_publisher import (
        fetch_notebook,
        is_publishing_active,
        load_github_settings,
    )

    gh_settings = load_github_settings()
    if not is_publishing_active(gh_settings):
        return {"status": "skipped_disabled", "path": path}

    file_json = await fetch_notebook(path)
    if file_json is None:
        logger.warning("sync_one_answer: GitHub fetch failed or path not found", path=path)
        return {"status": "failed", "path": path}

    if not isinstance(file_json, dict):
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "file content is not a JSON object",
        }
    answer_id = file_json.get("answer_id")
    if answer_id is None or answer_id == "":
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "missing answer_id",
        }
    if not isinstance(answer_id, str):
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": "answer_id is not a string",
        }

    sync_ts = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    index = _load_index()
    local_answers = index.get("verified_answers", {})
    existing_entry = local_answers.get(answer_id)

    if existing_entry is not None:
        try:
            candidate = dict(existing_entry)
            for k in (
                "query",
                "answer",
                "source_links",
                "verified_at",
                "verified_by",
                "data_source",
                "confidence",
                "submission_id",
                "submitted_by",
                "input_tokens",
                "output_tokens",
                "variable",
                "place",
                "date",
                "value",
            ):
                if k in file_json:
                    candidate[k] = file_json[k]
            candidate["github_path"] = path
            candidate["github_synced_at"] = sync_ts
            candidate["answer_id"] = answer_id
            validated_entry = VerifiedAnswer(**candidate).model_dump()
        except Exception as e:
            logger.warning(
                "sync_one_answer: failed to validate metadata for existing "
                "entry — leaving local entry untouched",
                path=path,
                answer_id=answer_id,
                error=str(e),
            )
            return {
                "status": "skipped_bad_metadata",
                "path": path,
                "reason": str(e),
            }
        local_answers[answer_id] = validated_entry
        index["verified_answers"] = local_answers
        _save_index(index)
        logger.info(
            "Synced one answer from GitHub (updated)",
            path=path,
            answer_id=answer_id,
        )
        return {"status": "updated", "answer_id": answer_id, "path": path}

    # Create path
    try:
        candidate = dict(file_json)
        candidate["answer_id"] = answer_id
        candidate["github_path"] = path
        candidate["github_synced_at"] = sync_ts
        new = VerifiedAnswer(**candidate)
    except Exception as e:
        logger.warning(
            "sync_one_answer: failed to reconstruct VerifiedAnswer",
            path=path,
            answer_id=answer_id,
            error=str(e),
        )
        return {
            "status": "skipped_bad_metadata",
            "path": path,
            "reason": str(e),
        }
    local_answers[new.answer_id] = new.model_dump()
    index["verified_answers"] = local_answers
    _save_index(index)
    logger.info(
        "Synced one answer from GitHub (created)",
        path=path,
        answer_id=new.answer_id,
    )
    return {"status": "created", "answer_id": new.answer_id, "path": path}


def delete_local_answer_by_github_path(path: str) -> dict[str, Any]:
    """Remove the local verified-answer entry whose ``github_path`` equals
    ``path``.

    Counterpart to :func:`delete_local_by_github_path` for the answer
    side. Wired up by the webhook handler (issue #46 step 9 PR C) to
    handle REMOVE events on verified-answers/*.json files: when an
    admin deletes an answer file in GitHub, the local cache must drop
    it too.

    Returns:

    * ``{"status": "deleted", "answer_id": ..., "path": ...}`` — local
      entry was found and removed from the index. No on-disk blob to
      delete (unlike notebooks, an answer IS its index entry).
    * ``{"status": "skipped_no_local", "path": ...}`` — no local entry
      pointed at this path; nothing to do.

    Does NOT touch local-only entries whose ``github_path`` was never
    set (e.g. answers approved while GitHub publishing was disabled);
    those are out-of-scope for webhook events.
    """
    index = _load_index()
    local_answers = index.get("verified_answers", {})
    target_answer_id: str | None = None
    for answer_id, entry in local_answers.items():
        if entry.get("github_path") == path:
            target_answer_id = answer_id
            break

    if target_answer_id is None:
        logger.info(
            "delete_local_answer_by_github_path: no local entry for path",
            path=path,
        )
        return {"status": "skipped_no_local", "path": path}

    del local_answers[target_answer_id]
    index["verified_answers"] = local_answers
    _save_index(index)
    logger.info(
        "Deleted local verified-answer entry (github webhook DELETE)",
        path=path,
        answer_id=target_answer_id,
    )
    return {"status": "deleted", "answer_id": target_answer_id, "path": path}


def increment_usage(notebook_id: str) -> None:
    """Increment the usage count for a verified notebook."""
    index = _load_index()
    if notebook_id in index.get("verified", {}):
        index["verified"][notebook_id]["usage_count"] = (
            index["verified"][notebook_id].get("usage_count", 0) + 1
        )
        _save_index(index)


def get_stats() -> dict[str, Any]:
    """Get statistics about the notebook store and quick answers."""
    index = _load_index()
    submissions = index.get("submissions", {})
    verified = index.get("verified", {})
    answer_submissions = index.get("answer_submissions", {})
    verified_answers = index.get("verified_answers", {})

    status_counts = Counter()
    for sub in submissions.values():
        status_counts[sub.get("status", "unknown")] += 1

    answer_status_counts = Counter()
    for sub in answer_submissions.values():
        answer_status_counts[sub.get("status", "unknown")] += 1

    return {
        "total_submissions": len(submissions),
        "pending": status_counts.get("pending", 0),
        "approved": status_counts.get("approved", 0),
        "rejected": status_counts.get("rejected", 0),
        "total_verified": len(verified),
        "total_usage": sum(v.get("usage_count", 0) for v in verified.values()),
        "total_answer_submissions": len(answer_submissions),
        "answer_pending": answer_status_counts.get("pending", 0),
        "answer_approved": answer_status_counts.get("approved", 0),
        "answer_rejected": answer_status_counts.get("rejected", 0),
        "total_verified_answers": len(verified_answers),
        "total_answer_usage": sum(v.get("usage_count", 0) for v in verified_answers.values()),
    }


# =============================================================================
# Quick Answer Store Operations
# =============================================================================


def submit_quick_answer(
    query: str,
    answer: str,
    source_links: list[dict[str, str]],
    submitted_by: str = "anonymous",
    submitter_email: str | None = None,
    submitter_auth_type: str | None = None,
    data_source: str = "",
    confidence: float = 0.0,
    tags: list[str] | None = None,
    variable: str = "",
    place: str = "",
    date: str = "",
    value: str = "",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> QuickAnswerSubmission:
    """Submit a quick answer for admin review.

    Args:
        query: The original user query
        answer: The one-line factual answer
        source_links: Direct links to data source pages
        submitted_by: User identifier
        data_source: Data source used
        confidence: Original confidence score
        tags: Optional tags for categorization
        variable: Statistical variable queried
        place: Geographic place queried
        date: Date of the observation
        value: The data value

    Returns:
        The created submission
    """
    submission = QuickAnswerSubmission(
        query=query,
        answer=answer,
        source_links=source_links,
        submitted_by=submitted_by,
        submitter_email=submitter_email,
        submitter_auth_type=submitter_auth_type,
        data_source=data_source,
        confidence=confidence,
        tags=tags or extract_keywords(query)[:10],
        variable=variable,
        place=place,
        date=date,
        value=value,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    # Update index
    index = _load_index()
    index["answer_submissions"][submission.submission_id] = submission.model_dump()
    _save_index(index)

    logger.info(
        "Quick answer submitted for review",
        submission_id=submission.submission_id,
        query=query[:100],
    )
    return submission


def get_answer_submissions(
    status_filter: str | None = None,
) -> list[QuickAnswerSubmission]:
    """Get quick answer submissions, optionally filtered by status."""
    index = _load_index()
    submissions = []
    for data in index.get("answer_submissions", {}).values():
        sub = QuickAnswerSubmission(**data)
        if status_filter is None or sub.status.value == status_filter:
            submissions.append(sub)
    submissions.sort(key=lambda s: s.submitted_at, reverse=True)
    return submissions


def get_answer_submission(submission_id: str) -> QuickAnswerSubmission | None:
    """Get a specific quick answer submission by ID."""
    index = _load_index()
    data = index.get("answer_submissions", {}).get(submission_id)
    if data:
        return QuickAnswerSubmission(**data)
    return None


def approve_quick_answer(
    submission_id: str,
    reviewed_by: str = "admin",
    admin_notes: str | None = None,
    answer_id: str | None = None,
    github_path: str | None = None,
    github_synced_at: str | None = None,
    verified_at: str | None = None,
) -> VerifiedAnswer | None:
    """Approve a quick answer submission, making it verified.

    Args:
        submission_id: The submission to approve
        reviewed_by: Admin identifier
        admin_notes: Optional review notes
        answer_id: Pre-generated answer_id to use instead of the model's
            ``default_factory``. The router pre-generates this BEFORE
            publishing to GitHub so the on-GitHub filename, the
            published file's embedded ``answer_id``, and the local
            index key are all the same UUID — the coherence anchor
            that makes bootstrap recoverable in PR B.
        github_path: Path of the published answer in the GitHub repo
            (set on the write-through path by the router; ``None`` when
            GitHub publishing is disabled).
        github_synced_at: ISO timestamp of the GitHub publish; the
            router pre-computes this ONCE and reuses it as the
            ``verified_at`` value below so the index entry and the
            published file agree on the moment of approval (issue #46
            step 9 — same coherence anchor as notebooks step 4).
        verified_at: Pre-computed verification timestamp. When provided,
            stored on the ``VerifiedAnswer`` instead of "now" so it
            matches ``github_synced_at`` exactly.

    Returns:
        The created verified answer, or None if submission not found
    """
    index = _load_index()
    sub_data = index.get("answer_submissions", {}).get(submission_id)
    if not sub_data:
        return None

    submission = QuickAnswerSubmission(**sub_data)
    if submission.status != ReviewStatus.PENDING:
        logger.warning(
            "Answer submission already reviewed",
            submission_id=submission_id,
            status=submission.status,
        )
        return None

    keywords = extract_keywords(submission.query + " " + submission.answer)
    verified_kwargs: dict[str, Any] = {
        "submission_id": submission_id,
        "query": submission.query,
        "answer": submission.answer,
        "source_links": submission.source_links,
        "verified_by": reviewed_by,
        "admin_notes": admin_notes,
        "tags": submission.tags,
        "data_source": submission.data_source,
        "confidence": submission.confidence,
        "submitted_by": submission.submitted_by,
        "input_tokens": submission.input_tokens,
        "output_tokens": submission.output_tokens,
        "variable": submission.variable,
        "place": submission.place,
        "date": submission.date,
        "value": submission.value,
        "keywords": keywords,
        "github_path": github_path,
        "github_synced_at": github_synced_at,
    }
    if answer_id is not None:
        verified_kwargs["answer_id"] = answer_id
    if verified_at is not None:
        verified_kwargs["verified_at"] = verified_at
    verified = VerifiedAnswer(**verified_kwargs)

    # Update submission status
    submission.status = ReviewStatus.APPROVED
    submission.reviewed_at = datetime.now(UTC).isoformat()
    submission.reviewed_by = reviewed_by
    submission.admin_notes = admin_notes
    index["answer_submissions"][submission_id] = submission.model_dump()

    # Add to verified answers index
    index["verified_answers"][verified.answer_id] = verified.model_dump()
    _save_index(index)

    logger.info(
        "Quick answer approved",
        submission_id=submission_id,
        answer_id=verified.answer_id,
        github_path=github_path,
    )
    return verified


def reject_quick_answer(
    submission_id: str,
    reviewed_by: str = "admin",
    admin_notes: str | None = None,
) -> QuickAnswerSubmission | None:
    """Reject a quick answer submission.

    Args:
        submission_id: The submission to reject
        reviewed_by: Admin identifier
        admin_notes: Reason for rejection

    Returns:
        The updated submission, or None if not found
    """
    index = _load_index()
    sub_data = index.get("answer_submissions", {}).get(submission_id)
    if not sub_data:
        return None

    submission = QuickAnswerSubmission(**sub_data)
    if submission.status != ReviewStatus.PENDING:
        return None

    submission.status = ReviewStatus.REJECTED
    submission.reviewed_at = datetime.now(UTC).isoformat()
    submission.reviewed_by = reviewed_by
    submission.admin_notes = admin_notes

    index["answer_submissions"][submission_id] = submission.model_dump()
    _save_index(index)

    logger.info("Quick answer rejected", submission_id=submission_id)
    return submission


def get_verified_answers() -> list[VerifiedAnswer]:
    """Get all verified quick answers."""
    index = _load_index()
    answers = []
    for data in index.get("verified_answers", {}).values():
        answers.append(VerifiedAnswer(**data))
    answers.sort(key=lambda a: a.verified_at, reverse=True)
    return answers


def get_verified_answer(answer_id: str) -> VerifiedAnswer | None:
    """Get a specific verified answer by ID."""
    index = _load_index()
    data = index.get("verified_answers", {}).get(answer_id)
    if data:
        return VerifiedAnswer(**data)
    return None


def search_verified_answers(
    query: str,
    threshold: float = 0.25,
    max_results: int = 5,
) -> list[AnswerSearchResult]:
    """Search verified quick answers for similar queries.

    Args:
        query: The search query
        threshold: Minimum similarity score (0-1)
        max_results: Maximum number of results

    Returns:
        List of search results sorted by similarity
    """
    query_keywords = extract_keywords(query)
    if not query_keywords:
        return []

    index = _load_index()
    results = []

    for answer_id, data in index.get("verified_answers", {}).items():
        answer = VerifiedAnswer(**data)
        doc_keywords = answer.keywords or extract_keywords(answer.query)

        score = compute_similarity(query_keywords, doc_keywords)
        if score >= threshold:
            results.append(
                AnswerSearchResult(
                    answer_id=answer_id,
                    query=answer.query,
                    answer=answer.answer,
                    source_links=answer.source_links,
                    similarity_score=score,
                    tags=answer.tags,
                    verified_at=answer.verified_at,
                    verified_by=answer.verified_by,
                    usage_count=answer.usage_count,
                )
            )

    results.sort(key=lambda r: r.similarity_score, reverse=True)
    return results[:max_results]


def increment_answer_usage(answer_id: str) -> None:
    """Increment the usage count for a verified quick answer."""
    index = _load_index()
    if answer_id in index.get("verified_answers", {}):
        index["verified_answers"][answer_id]["usage_count"] = (
            index["verified_answers"][answer_id].get("usage_count", 0) + 1
        )
        _save_index(index)


# =============================================================================
# Verified-library de-duplication
# =============================================================================
# The verified library should hold exactly ONE entry per question. Duplicates
# accumulate when several users submit the same question and an admin approves
# more than one (e.g. 7 verified answers all for "What is the unemployment rate
# in Texas?"). Two mechanisms keep the library clean:
#   1. Forward prevention (router): before publishing/creating a new verified
#      entry, the approve endpoints look for an existing entry with the same
#      normalized question and collapse into it instead.
#   2. One-time / on-demand cleanup: ``dedupe_verified_library()`` collapses any
#      duplicates already in the index, merging usage counts into the survivor.
# Both use ``normalize_question()`` as the (conservative) dedup key.


def _dup_rank_key(entry: dict[str, Any], id_field: str) -> tuple[int, str, str]:
    """Survivor preference among entries sharing a normalized question.

    ``min()`` over this key picks the survivor: higher usage first (negated so
    smaller-is-better), then EARLIER ``verified_at`` (older = better-tested),
    then lexicographic id for a stable final tie-break.
    """
    return (
        -(entry.get("usage_count") or 0),
        entry.get("verified_at") or "",
        entry.get(id_field) or "",
    )


def find_verified_notebook_by_question(query: str) -> VerifiedNotebook | None:
    """Return an existing verified notebook whose question matches ``query``
    (after :func:`normalize_question`), or ``None``. Prefers a GitHub-published
    entry, then the most-used, then the oldest — the safe survivor to reuse."""
    nq = normalize_question(query)
    if not nq:
        return None
    index = _load_index()
    matches = [
        d
        for d in index.get("verified", {}).values()
        if normalize_question(d.get("query") or "") == nq
    ]
    if not matches:
        return None
    matches.sort(key=lambda d: (0 if d.get("github_path") else 1, *_dup_rank_key(d, "notebook_id")))
    return VerifiedNotebook(**matches[0])


def find_verified_answer_by_question(query: str) -> VerifiedAnswer | None:
    """Return an existing verified quick answer matching ``query`` (after
    :func:`normalize_question`), or ``None``. Same survivor preference as
    :func:`find_verified_notebook_by_question`."""
    nq = normalize_question(query)
    if not nq:
        return None
    index = _load_index()
    matches = [
        d
        for d in index.get("verified_answers", {}).values()
        if normalize_question(d.get("query") or "") == nq
    ]
    if not matches:
        return None
    matches.sort(key=lambda d: (0 if d.get("github_path") else 1, *_dup_rank_key(d, "answer_id")))
    return VerifiedAnswer(**matches[0])


def collapse_notebook_submission_as_duplicate(
    submission_id: str,
    existing_notebook_id: str,
    reviewed_by: str = "admin",
    admin_notes: str | None = None,
) -> bool:
    """Mark a pending notebook submission APPROVED as a duplicate of an existing
    verified notebook WITHOUT creating a second verified entry. Keeps the
    library at one entry per question. Returns ``True`` if collapsed."""
    index = _load_index()
    sub_data = index.get("submissions", {}).get(submission_id)
    if not sub_data:
        return False
    submission = NotebookSubmission(**sub_data)
    if submission.status != ReviewStatus.PENDING:
        return False
    note = f"Auto-collapsed: duplicate of verified notebook {existing_notebook_id}"
    submission.status = ReviewStatus.APPROVED
    submission.reviewed_at = datetime.now(UTC).isoformat()
    submission.reviewed_by = reviewed_by
    submission.admin_notes = f"{admin_notes} — {note}" if admin_notes else note
    index["submissions"][submission_id] = submission.model_dump()
    _save_index(index)
    logger.info(
        "Collapsed duplicate notebook submission",
        submission_id=submission_id,
        existing_notebook_id=existing_notebook_id,
    )
    return True


def collapse_answer_submission_as_duplicate(
    submission_id: str,
    existing_answer_id: str,
    reviewed_by: str = "admin",
    admin_notes: str | None = None,
) -> bool:
    """Quick-answer counterpart of
    :func:`collapse_notebook_submission_as_duplicate`."""
    index = _load_index()
    sub_data = index.get("answer_submissions", {}).get(submission_id)
    if not sub_data:
        return False
    submission = QuickAnswerSubmission(**sub_data)
    if submission.status != ReviewStatus.PENDING:
        return False
    note = f"Auto-collapsed: duplicate of verified answer {existing_answer_id}"
    submission.status = ReviewStatus.APPROVED
    submission.reviewed_at = datetime.now(UTC).isoformat()
    submission.reviewed_by = reviewed_by
    submission.admin_notes = f"{admin_notes} — {note}" if admin_notes else note
    index["answer_submissions"][submission_id] = submission.model_dump()
    _save_index(index)
    logger.info(
        "Collapsed duplicate answer submission",
        submission_id=submission_id,
        existing_answer_id=existing_answer_id,
    )
    return True


def _dedupe_section(
    index: dict[str, Any],
    section: str,
    id_field: str,
    has_blob: bool,
    dry_run: bool,
) -> tuple[list[str], list[str]]:
    """Collapse duplicate entries in ``index[section]`` by normalized question.

    Returns ``(removed_ids, needs_manual_reconcile_questions)``.

    Safety: GitHub-published entries (``github_path`` set) are NEVER deleted —
    removing one would orphan its file in the repo. So when a duplicate group
    contains published entries, only the LOCAL-ONLY copies are removed; their
    usage counts are merged into the chosen survivor. A group with more than one
    published copy is reported for manual GitHub reconciliation instead.
    """
    entries: dict[str, Any] = index.get(section, {})
    groups: dict[str, list[str]] = {}
    for eid, e in entries.items():
        nq = normalize_question(e.get("query") or "")
        if not nq:
            continue
        groups.setdefault(nq, []).append(eid)

    removed: list[str] = []
    needs_reconcile: list[str] = []
    for nq, ids in groups.items():
        if len(ids) < 2:
            continue
        group = [entries[i] for i in ids]
        published = [e for e in group if e.get("github_path")]
        local_only = [e for e in group if not e.get("github_path")]
        if published:
            survivor = min(published, key=lambda e: _dup_rank_key(e, id_field))
            losers = local_only  # only local-only copies are safe to drop
            if len(published) > 1:
                needs_reconcile.append(nq)
        else:
            survivor = min(local_only, key=lambda e: _dup_rank_key(e, id_field))
            losers = [e for e in local_only if e is not survivor]
        if not losers:
            continue
        merged_usage = (survivor.get("usage_count") or 0) + sum(
            (e.get("usage_count") or 0) for e in losers
        )
        for e in losers:
            lid = e.get(id_field)
            removed.append(lid)
            if dry_run:
                continue
            entries.pop(lid, None)
            if has_blob:
                try:
                    storage.delete(f"{_VERIFIED_PREFIX}/{lid}.ipynb")
                except Exception as ex:  # blob may already be gone — non-fatal
                    logger.warning(
                        "Failed to delete duplicate notebook blob",
                        notebook_id=lid,
                        error=str(ex),
                    )
        if not dry_run:
            survivor["usage_count"] = merged_usage
    return removed, needs_reconcile


def dedupe_verified_library(dry_run: bool = False) -> dict[str, Any]:
    """Collapse duplicate verified notebooks and answers that share a question.

    Keeps one survivor per normalized question (most-used, then oldest), merges
    the duplicates' usage counts into it, removes the redundant index entries,
    and deletes orphaned notebook blob files. Idempotent and safe to re-run
    (e.g. after a GitHub sync). With ``dry_run=True`` it reports what WOULD be
    removed without mutating anything.
    """
    index = _load_index()
    nb_removed, nb_reconcile = _dedupe_section(
        index, "verified", "notebook_id", has_blob=True, dry_run=dry_run
    )
    ans_removed, ans_reconcile = _dedupe_section(
        index, "verified_answers", "answer_id", has_blob=False, dry_run=dry_run
    )
    if not dry_run and (nb_removed or ans_removed):
        _save_index(index)

    summary = {
        "dry_run": dry_run,
        "notebooks_removed": nb_removed,
        "answers_removed": ans_removed,
        "removed_count": len(nb_removed) + len(ans_removed),
        "needs_manual_github_reconcile": nb_reconcile + ans_reconcile,
    }
    logger.info(
        "Verified library dedup",
        dry_run=dry_run,
        removed_count=summary["removed_count"],
        notebooks_removed=len(nb_removed),
        answers_removed=len(ans_removed),
        needs_reconcile=len(summary["needs_manual_github_reconcile"]),
    )
    return summary
