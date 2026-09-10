"""Guards on the embedded text that Pinecone indexes.

These are retrieval-quality regressions, not style preferences. The embedded
text was originally capped at 8000 chars (median 4963) and retrieval collapsed:
every similarity score fell in 0.17-0.28 and short records became the top hit
for unrelated queries, because one vector over a long blob averages the topic
away. Re-benchmarked on 35 labeled queries over the 110-dataset WPRDC corpus,
tightening the budget raised precision@1 from 69% to 89%.

A change that loosens these budgets must come with a fresh benchmark.
"""

from __future__ import annotations

from data_concierge.data_layer.pinecone_upload import (
    MAX_TEXT_CHARS,
    build_record,
    build_text,
)

WIDE_RESOURCE = {
    "resource_id": "r1",
    "dataset_title": "Pittsburgh 311 Data",
    "dataset_description": "Service requests submitted to the city. " * 200,
    "qsv_tags": ["311", "potholes", "identifier", "float", "boolean"],
    "dataset_tags": ["service requests", "public works"],
    "columns": [
        {
            "name": f"COLUMN_{i}",
            "qsv_description": "a column description that is quite long " * 10,
            "top_values": [{"value": f"value_{i}_{j}", "count": 5} for j in range(5)],
        }
        for i in range(200)
    ],
}


class TestLengthDiscipline:
    def test_total_budget_is_small(self):
        """8000 was the measured failure; the budget must stay tight."""
        assert MAX_TEXT_CHARS <= 1000

    def test_a_very_wide_table_still_fits_the_budget(self):
        text = build_text(WIDE_RESOURCE)
        assert len(text) <= MAX_TEXT_CHARS

    def test_title_comes_first(self):
        """If any cap ever binds, the most discriminative string must survive."""
        text = build_text(WIDE_RESOURCE)
        assert text.startswith("Pittsburgh 311 Data")

    def test_columns_cannot_crowd_out_the_topic(self):
        """On a 200-column table the topic block must still be the single
        largest section — columns are what diluted the original 8000-char text."""
        text = build_text(WIDE_RESOURCE)
        topic = text.split("Fields:")[0]
        fields = text.split("Fields:")[1].split("Values:")[0] if "Fields:" in text else ""
        values = text.split("Values:")[1] if "Values:" in text else ""
        assert len(topic) > len(fields)
        assert len(topic) > len(values)


class TestTagHandling:
    def test_both_tag_sources_are_merged(self):
        """qsv_tags OR dataset_tags discarded curator tags on 84 of 110 records."""
        text = build_text(WIDE_RESOURCE).lower()
        assert "potholes" in text          # from qsv_tags
        assert "service requests" in text  # from dataset_tags

    def test_schema_words_are_not_treated_as_topics(self):
        """'identifier'/'float'/'boolean' describe every dataset equally, so they
        pull records together instead of apart."""
        topics = ""
        for line in build_text(WIDE_RESOURCE).splitlines():
            if line.startswith("Topics:"):
                topics = line.lower()
        assert topics, "no Topics line emitted"
        for noise in ("identifier", "float", "boolean"):
            assert noise not in topics


class TestContentSignal:
    def test_distinctive_values_survive(self):
        """A dataset must be findable by what is IN it: 'potholes' appears only
        as a 311 request-type value, never in a title or description."""
        resource = {
            "resource_id": "r2",
            "dataset_title": "Service Requests",
            "dataset_description": "Requests submitted by residents.",
            "columns": [
                {"name": "REQUEST_TYPE",
                 "top_values": [{"value": "Potholes", "count": 90},
                                {"value": "Weeds/Debris", "count": 40}]},
            ],
        }
        text = build_text(resource).lower()
        assert "potholes" in text

    def test_generic_field_names_are_dropped(self):
        """Names every dataset has add no signal and cost budget."""
        resource = {
            "resource_id": "r3",
            "dataset_title": "Some Dataset",
            "columns": [
                {"name": "id"}, {"name": "address"}, {"name": "latitude"},
                {"name": "LANDSLIDE_RISK"},
            ],
        }
        text = build_text(resource)
        fields = next((ln for ln in text.splitlines() if ln.startswith("Fields:")), "")
        assert "LANDSLIDE_RISK" in fields
        assert "latitude" not in fields.lower()

    def test_numeric_values_are_not_embedded_as_topics(self):
        resource = {
            "resource_id": "r4",
            "dataset_title": "Counts",
            "columns": [{"name": "AMOUNT", "top_values": [{"value": "12345", "count": 3}]}],
        }
        assert "12345" not in build_text(resource)


class TestRecordContract:
    def test_record_still_carries_the_search_fields(self):
        """The query side asks for these by name; renaming one silently empties
        the result metadata."""
        rec = build_record(WIDE_RESOURCE, site_id="wprdc")
        assert rec is not None
        for field in ("text", "site_id", "resource_id", "dataset_title",
                      "has_temporal", "has_geographic", "record_count"):
            assert field in rec
        assert rec["_id"] == "wprdc:r1"
        assert len(rec["text"]) <= MAX_TEXT_CHARS


class TestPortalScoping:
    """One namespace holds every portal's records, so an unscoped search hands
    the agent resource IDs from a portal it is not querying — which then 404."""

    def _store(self):
        from data_concierge.data_layer.connectors.pinecone_store import PineconeVectorStore

        return PineconeVectorStore.__new__(PineconeVectorStore)

    def test_non_legacy_portal_matches_strictly(self):
        f = self._store().portal_filter("wprdc")
        assert f == {"site_id": {"$eq": "wprdc"}}

    def test_legacy_portal_also_matches_untagged_records(self):
        """Records written before tagging existed carry no site_id. The portal
        that owns them matches those too, so no one-way backfill is needed."""
        f = self._store().portal_filter("ckan")
        assert f == {
            "$or": [{"site_id": {"$eq": "ckan"}}, {"site_id": {"$exists": False}}]
        }

    def test_untagged_records_are_claimed_by_exactly_one_portal(self):
        """A new portal must never inherit the legacy corpus."""
        store = self._store()
        for other in ("wprdc", "data-pa-gov", "anything-else"):
            assert "$or" not in store.portal_filter(other)
            assert store.portal_filter(other)["site_id"]["$eq"] == other

    def test_no_site_id_means_no_scoping(self):
        assert self._store().portal_filter(None) is None
        assert self._store().portal_filter("") is None

    def test_search_fields_include_site_id(self):
        """Scoping is server-side, but the field must come back so a result can
        be attributed — filtering on a field the server never returns is a
        silent no-op."""
        import inspect

        from data_concierge.data_layer.connectors import pinecone_store

        src = inspect.getsource(pinecone_store.PineconeVectorStore._pinecone_search)
        assert '"site_id"' in src

    def test_agent_resolves_a_site_id_for_the_semantic_tool(self):
        import inspect

        from data_concierge.agents import llm_agent

        src = inspect.getsource(llm_agent.LLMAnalysisAgent._execute_tool)
        # The semantic tool must be handed the resolved portal, not called bare.
        assert "_tool_semantic_search(tool_input, effective_site_id)" in src
