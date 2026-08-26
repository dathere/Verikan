"""Tests for the Typed Standards `datHere` evidence-package builder.

Validates the A–G envelope mapping (spec §8.7), RFC 8785 JCS canonicalization
(§8.2), content/envelope hashing (§8.2/§8.3), the commitment-view field set
(§8.8.1), and the notebook-embedded serialization + cell-0 table (§8.8.2/.4).
"""

import pytest

from data_concierge.gateway.evidence import (
    CONTENT_PROFILE,
    ENVIRONMENT_NS,
    EVIDENCE_NS,
    NOTEBOOK_NS,
    PRODUCER_PROFILE,
    UnsignedSigner,
    build_evidence_package,
    canonicalize_jcs,
    embed_commitment_view,
)


@pytest.fixture
def agent_log() -> list[dict]:
    return [
        {
            "type": "session_start",
            "timestamp": "2026-06-16T00:00:00+00:00",
            "query": "What is X?",
            "data_source": "wprdc",
            "portal_url": "https://data.wprdc.org",
            "model": "claude-sonnet-4-6",
            "tools_available": ["search_datasets", "run_sql_query"],
            "system_prompt": "SYSTEM PROMPT",
            "system_prompt_sha256": "abc123",
        },
        {
            "type": "llm_response",
            "timestamp": "2026-06-16T00:00:01+00:00",
            "iteration": 1,
            "message_id": "msg_1",
            "model": "claude-sonnet-4-6",
            "stop_reason": "tool_use",
            "texts": ["let me look that up"],
            "tool_calls": [],
            "tokens": {
                "input": 100,
                "output": 50,
                "cache_creation_input": 10,
                "cache_read_input": 5,
            },
            "duration_ms": 1200,
        },
        {
            "type": "tool_execution",
            "timestamp": "2026-06-16T00:00:02+00:00",
            "iteration": 1,
            "tool": "run_sql_query",
            "tool_use_id": "tu1",
            "source": "ckan:wprdc",
            "operation_type": "query",
            "status": "success",
            "input": {"sql": "SELECT 1"},
            "result": "ok",
            "result_chars": 2,
            "duration_ms": 300,
        },
        {
            "type": "summary",
            "timestamp": "2026-06-16T00:00:03+00:00",
            "total_iterations": 2,
            "total_tool_calls": 1,
            "total_elapsed_ms": 5000,
            "model": "claude-sonnet-4-6",
            "data_source": "wprdc",
            "token_totals": {
                "input": 100,
                "output": 50,
                "cache_creation_input": 10,
                "cache_read_input": 5,
            },
        },
    ]


@pytest.fixture
def notebook() -> dict:
    return {
        "cells": [{"cell_type": "code", "source": ["print(1)"], "metadata": {}, "outputs": []}],
        "metadata": {"kernelspec": {"name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class TestHostValidation:
    def test_valid_host_and_port(self) -> None:
        from data_concierge.gateway.evidence import _validated_evidence_host

        assert _validated_evidence_host("data-concierge.dathere.com") == "data-concierge.dathere.com"
        assert _validated_evidence_host("localhost:8000") == "localhost:8000"

    def test_invalid_host_falls_back_to_default(self) -> None:
        from data_concierge.gateway.evidence import (
            _DEFAULT_EVIDENCE_HOST,
            _validated_evidence_host,
        )

        # A quote (the injection char) and other junk must not pass through.
        assert _validated_evidence_host("evil'host") == _DEFAULT_EVIDENCE_HOST
        assert _validated_evidence_host("a/b") == _DEFAULT_EVIDENCE_HOST
        assert _validated_evidence_host("https://x.com") == _DEFAULT_EVIDENCE_HOST
        assert _validated_evidence_host("") == _DEFAULT_EVIDENCE_HOST
        assert _validated_evidence_host(None) == _DEFAULT_EVIDENCE_HOST


class TestJCS:
    def test_keys_sorted_and_compact(self) -> None:
        assert canonicalize_jcs({"b": 1, "a": 2}) == b'{"a":2,"b":1}'

    def test_integral_float_emits_as_integer(self) -> None:
        assert canonicalize_jcs({"t": 0.0}) == b'{"t":0}'

    def test_fractional_float(self) -> None:
        assert canonicalize_jcs({"t": 0.7}) == b'{"t":0.7}'

    def test_literals_and_arrays(self) -> None:
        assert canonicalize_jcs([1, True, False, None, "x"]) == b'[1,true,false,null,"x"]'

    def test_string_escaping(self) -> None:
        assert canonicalize_jcs('a"b\\c\n') == b'"a\\"b\\\\c\\n"'

    def test_unicode_is_utf8_not_escaped(self) -> None:
        assert canonicalize_jcs("café") == '"café"'.encode()

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            canonicalize_jcs(float("nan"))


class TestBuildPackage:
    def test_required_top_level_fields(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="Answer.", query="What is X?", agent_log=agent_log
        )
        for field in (
            "metadata",
            "prompt",
            "queries",
            "dataSources",
            "cost",
            "skillMetadata",
            "output",
            "trace",
            "summary",
            "contentProfile",
            "producerProfile",
            "contentHash",
            "contentCanonicalization",
            "type",
            "signer",
            "extensions",
        ):
            assert field in pkg.package, f"missing required field {field}"

    def test_dathere_profile_labels(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="Answer.", query="Q", agent_log=agent_log
        )
        assert pkg.package["contentProfile"] == CONTENT_PROFILE
        assert pkg.package["producerProfile"] == PRODUCER_PROFILE
        assert pkg.package["metadata"]["contentProfile"] == CONTENT_PROFILE

    def test_prompt_is_full_text_verbatim(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="What is X?", agent_log=agent_log
        )
        assert pkg.package["prompt"]["visibility"] == "full_text"
        assert pkg.package["prompt"]["text"] == "What is X?"

    def test_system_prompt_from_agent_log(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        assert pkg.package["skillMetadata"]["skillText"] == "SYSTEM PROMPT"

    def test_token_accounting_includes_cache(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        cost = pkg.package["cost"]
        assert cost["promptTokens"] == 115  # 100 + 10 + 5
        assert cost["completionTokens"] == 50
        assert cost["totalTokens"] == 165

    def test_environment_required_fields(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        env = pkg.package["extensions"][ENVIRONMENT_NS]
        for field in ("modelVersion", "temperature", "mcpServers", "toolDefinitions", "host"):
            assert field in env

    def test_notebook_extension_present(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        ext = pkg.package["extensions"][NOTEBOOK_NS]
        assert ext["format"] == "jupyter-v4.5"
        assert ext["provenance"] == "skeleton"
        assert ext["notebook"]["nbformat"] == 4

    def test_queries_carry_source_and_operation(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        assert len(pkg.package["queries"]) == 1
        q = pkg.package["queries"][0]
        assert q["source"] == "ckan:wprdc"
        assert q["operationType"] == "query"
        assert pkg.package["dataSources"] == [{"sourceId": "ckan:wprdc"}]

    def test_summary_present_and_bounded(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook,
            answer="The answer is 42. " * 40,
            query="Q",
            agent_log=agent_log,
        )
        assert pkg.package["summary"]
        assert len(pkg.package["summary"]) <= 281

    def test_content_hash_matches_field(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        assert pkg.package["contentHash"]["sha256"] == pkg.content_hash

    def test_deterministic_hash_with_fixed_id_and_time(self, notebook, agent_log) -> None:
        # Fixed packageId + createdAt make the envelope hash a stable content
        # address (the property the backfill's idempotency relies on).
        kw = {"notebook_json": notebook, "answer": "A", "query": "Q", "agent_log": agent_log}
        a = build_evidence_package(**kw, package_id="pid-1", created_at="2026-01-01T00:00:00Z")
        b = build_evidence_package(**kw, package_id="pid-1", created_at="2026-01-01T00:00:00Z")
        assert a.envelope_hash == b.envelope_hash
        c = build_evidence_package(**kw, package_id="pid-2", created_at="2026-01-01T00:00:00Z")
        assert c.envelope_hash != a.envelope_hash

    def test_content_hash_stable_across_builds(self, notebook, agent_log) -> None:
        # packageId/createdAt differ per build, but the content hash fingerprints
        # only the notebook, so it must be stable.
        a = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        b = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        assert a.content_hash == b.content_hash
        assert a.package["metadata"]["packageId"] != b.package["metadata"]["packageId"]

    def test_unsigned_by_default(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        assert not pkg.signed
        assert pkg.signature.kid == "dev-unsigned"
        assert pkg.commitment_view["_status"] == "dev-unsigned"

    def test_signing_key_id_matches_envelope_kid(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook,
            answer="A",
            query="Q",
            agent_log=agent_log,
            signer=UnsignedSigner(),
        )
        assert pkg.package["metadata"]["signingKeyId"] == pkg.signature.kid


class TestCommitmentView:
    def test_field_set(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook,
            answer="A",
            query="Q",
            agent_log=agent_log,
            package_url="https://example/nb.ipynb",
            title="My analysis",
        )
        cv = pkg.commitment_view
        for field in (
            "evidenceProtocolVersion",
            "packageHash",
            "packageUrl",
            "captureMethod",
            "contentProfile",
            "signature",
            "signer",
            "signerIdentity",
            "trustRegistryUrl",
            "subjectTitle",
            "subjectSummary",
            "attestations",
        ):
            assert field in cv
        assert cv["packageHash"] == pkg.envelope_hash
        assert cv["packageUrl"] == "https://example/nb.ipynb"
        assert cv["subjectTitle"] == "My analysis"

    def test_signer_claim_mirrors_the_package(self, notebook, agent_log) -> None:
        """§8.8.1: the view carries ``signer`` (the claim) as well as
        ``signerIdentity`` (the informational block).

        A verifier following a lifecycle chain reads
        ``commitment.signer.identifier``. Emitting only ``signerIdentity``
        leaves that empty, so a withdrawal or supersession would not resolve —
        and nothing fails until the first such attestation is published.
        """
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        cv = pkg.commitment_view
        assert cv["signer"] == pkg.package["signer"]
        assert cv["signer"]["identifier"].startswith("platform:")
        assert cv["signer"]["identifier"] == cv["signerIdentity"]["identifier"]

    def test_signer_mirror_does_not_change_the_package_hash(
        self, notebook, agent_log
    ) -> None:
        """The commitment view is built after the envelope hash and is not
        hashed, so mirroring the signer into it must not move the content
        address of any package."""
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        from data_concierge.gateway.evidence import _sha256_hex, canonicalize_jcs

        assert pkg.envelope_hash == _sha256_hex(canonicalize_jcs(pkg.package))


class TestEmbedding:
    def test_embeds_namespace_and_preserves_siblings(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        emb = embed_commitment_view(notebook, pkg)
        assert emb["metadata"][EVIDENCE_NS]["packageHash"] == pkg.envelope_hash
        assert emb["metadata"]["kernelspec"] == {"name": "python3"}
        assert emb["metadata"]["language_info"] == {"name": "python"}

    def test_cell0_table_prepended(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        emb = embed_commitment_view(notebook, pkg)
        assert emb["cells"][0]["cell_type"] == "markdown"
        assert emb["cells"][0]["metadata"][EVIDENCE_NS]["role"] == "commitment-table"
        assert emb["cells"][1]["source"] == ["print(1)"]  # original cell intact, shifted

    def test_idempotent_reembed(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        emb = embed_commitment_view(embed_commitment_view(notebook, pkg), pkg)
        tables = [
            c
            for c in emb["cells"]
            if c.get("metadata", {}).get(EVIDENCE_NS, {}).get("role") == "commitment-table"
        ]
        assert len(tables) == 1

    def test_does_not_mutate_input(self, notebook, agent_log) -> None:
        pkg = build_evidence_package(
            notebook_json=notebook, answer="A", query="Q", agent_log=agent_log
        )
        embed_commitment_view(notebook, pkg)
        assert notebook["cells"][0]["source"] == ["print(1)"]
        assert EVIDENCE_NS not in notebook["metadata"]


class TestCanonicalPackagePublish:
    """The cross-host verification invariant (spec §8.8): a reader fetches the
    canonical package at packageUrl, recomputes its hash, and compares to the
    commitment view's packageHash embedded in the notebook."""

    def test_published_package_rehashes_to_commitment(self, notebook, agent_log) -> None:
        import hashlib

        from data_concierge.gateway.evidence import canonicalize_jcs

        pkg = build_evidence_package(
            notebook_json=notebook,
            answer="A",
            query="Q",
            agent_log=agent_log,
            package_url="https://raw.githubusercontent.com/dathere/x/main/v/q.package.json",
        )
        emb = embed_commitment_view(notebook, pkg)
        cv = emb["metadata"][EVIDENCE_NS]
        # pkg.package is exactly what we publish as the sibling .package.json
        rehash = hashlib.sha256(canonicalize_jcs(pkg.package)).hexdigest()
        assert rehash == cv["packageHash"] == pkg.envelope_hash
        assert cv["packageUrl"].endswith("q.package.json")

    def test_build_raw_url(self) -> None:
        from data_concierge.gateway.github_publisher import build_raw_url

        url = build_raw_url("verified/q.ipynb", {"repo": "dathere/x", "branch": "main"})
        assert url == "https://raw.githubusercontent.com/dathere/x/main/verified/q.ipynb"
        assert build_raw_url(None, {"repo": "dathere/x"}) is None
        assert build_raw_url("v/q.ipynb", {}) is None  # no repo configured
