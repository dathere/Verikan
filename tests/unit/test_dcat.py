"""Tests for DCAT portal support: catalog parsing, registry, dispatch, indexing."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from data_concierge.data_layer.connectors.dcat import (
    DCATClient,
    UnsafeMCPTarget,
    clear_catalog_cache,
    parse_catalog,
    short_id,
)

# A miniature DCAT-US 1.1 catalog in the exact shape data.pa.gov serves.
SAMPLE_CATALOG = {
    "@context": "https://project-open-data.cio.gov/v1.1/schema/catalog.jsonld",
    "@type": "dcat:Catalog",
    "conformsTo": "https://project-open-data.cio.gov/v1.1/schema",
    "dataset": [
        {
            "@type": "dcat:Dataset",
            "identifier": "https://data.example.gov/api/views/aaaa-1111",
            "title": "Opioid Overdose Deaths by County",
            "description": "Annual counts of overdose deaths for each county.",
            "keyword": ["opioid", "overdose", "health"],
            "theme": ["Health"],
            "modified": "2026-02-28",
            "issued": "2022-11-25",
            "landingPage": "https://data.example.gov/d/aaaa-1111",
            "license": "https://creativecommons.org/publicdomain/zero/1.0/",
            "publisher": {"@type": "org:Organization", "name": "data.example.gov"},
            "contactPoint": {"fn": "Data Team", "hasEmail": "mailto:x@example.gov"},
            "distribution": [
                {
                    "@type": "dcat:Distribution",
                    "mediaType": "text/csv",
                    "downloadURL": "https://data.example.gov/api/views/aaaa-1111/rows.csv",
                },
                {
                    "@type": "dcat:Distribution",
                    "mediaType": "application/json",
                    "downloadURL": "https://data.example.gov/api/views/aaaa-1111/rows.json",
                },
            ],
        },
        {
            "@type": "dcat:Dataset",
            "identifier": "https://data.example.gov/api/views/bbbb-2222",
            "title": "Bridge Conditions",
            "description": "Structural ratings for state bridges. Mentions opioid nowhere.",
            "keyword": ["transportation", "bridges"],
            "distribution": [
                {
                    "@type": "dcat:Distribution",
                    "mediaType": "application/pdf",
                    "downloadURL": "https://data.example.gov/api/views/bbbb-2222/report.pdf",
                }
            ],
        },
    ],
}


class TestCatalogParsing:
    def test_parses_dcat_us_catalog(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "https://data.example.gov/data.json")
        assert len(catalog.datasets) == 2
        ds = catalog.datasets[0]
        assert ds.id == "aaaa-1111"
        assert ds.title == "Opioid Overdose Deaths by County"
        assert ds.keywords == ["opioid", "overdose", "health"]
        assert ds.themes == ["Health"]
        assert ds.publisher == "data.example.gov"
        assert ds.license.startswith("https://creativecommons.org")
        assert len(ds.distributions) == 2

    def test_identifies_tabular_distributions(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "u")
        assert len(catalog.datasets[0].tabular_distributions) == 1
        # A PDF-only dataset has nothing loadable as rows.
        assert catalog.datasets[1].tabular_distributions == []

    def test_parses_jsonld_graph_shape(self):
        body = {
            "@graph": [
                {"@type": "dcat:Catalog", "title": "ignored"},
                {
                    "@type": "dcat:Dataset",
                    "identifier": "urn:x:1",
                    "title": "Graph Dataset",
                    "dcat:distribution": [],
                },
            ]
        }
        catalog = parse_catalog(body, "u")
        assert [d.title for d in catalog.datasets] == ["Graph Dataset"]

    def test_parses_bare_array_shape(self):
        body = [
            {
                "@type": "dcat:Dataset",
                "identifier": "https://data.example.gov/api/views/zzz",
                "title": "Bare",
            }
        ]
        assert [d.id for d in parse_catalog(body, "u").datasets] == ["zzz"]

    def test_drops_entries_with_no_identity(self):
        body = {"dataset": [{"@type": "dcat:Dataset", "description": "orphan"}]}
        assert parse_catalog(body, "u").datasets == []

    def test_handles_missing_optional_fields(self):
        body = {"dataset": [{"identifier": "abc", "title": "Minimal"}]}
        ds = parse_catalog(body, "u").datasets[0]
        assert ds.keywords == [] and ds.distributions == [] and ds.license == ""


class TestShortId:
    def test_derives_trailing_segment_of_url_identifier(self):
        assert short_id("https://data.pa.gov/api/views/23n7-cwjw") == "23n7-cwjw"

    def test_passes_through_non_url_identifier(self):
        # Only http(s) identifiers are path-like. An opaque identifier is kept
        # whole — splitting it on "/" would invent an ID the portal never used.
        assert short_id("urn:uuid:abc-123") == "urn:uuid:abc-123"
        assert short_id("x/y/zzz") == "x/y/zzz"

    def test_falls_back_to_title_slug(self):
        assert short_id("", "My Great Dataset") == "my-great-dataset"

    def test_handles_trailing_slash(self):
        assert short_id("https://x.gov/d/abc/") == "abc"


class TestSearch:
    def test_title_match_outranks_description_match(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "u")
        hits = catalog.search("opioid", limit=5)
        assert hits[0][0].id == "aaaa-1111"
        # The bridge dataset mentions "opioid" only in prose, so it ranks lower.
        if len(hits) > 1:
            assert hits[0][1] > hits[1][1]

    def test_no_match_returns_empty(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "u")
        assert catalog.search("zzzznotarealterm", limit=5) == []

    def test_empty_query_browses_catalog(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "u")
        assert len(catalog.search("", limit=1)) == 1

    def test_tabular_only_filters_non_loadable(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "u")
        hits = catalog.search("bridges", limit=5, tabular_only=True)
        assert hits == []

    def test_get_by_id_identifier_and_title(self):
        catalog = parse_catalog(SAMPLE_CATALOG, "u")
        assert catalog.get("aaaa-1111").title.startswith("Opioid")
        assert catalog.get("https://data.example.gov/api/views/aaaa-1111") is not None
        assert catalog.get("Bridge Conditions").id == "bbbb-2222"
        assert catalog.get("nope") is None


def _mock_client(handler) -> DCATClient:
    client = DCATClient("https://data.example.gov", catalog_url="https://data.example.gov/data.json")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    # The mock host does not resolve, so skip the outbound safety check that
    # would otherwise reject it before the transport is reached.
    client._safe_url = lambda url: url  # type: ignore[method-assign]
    return client


class TestLoadDistribution:
    """The row ceiling is the load-bearing behaviour: a DCAT distribution has
    no server-side limit, so a runaway read is only prevented on our side."""

    def test_stops_at_max_rows_and_reports_truncation(self):
        body = "a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(5000))

        def handler(request):
            return httpx.Response(200, text=body)

        client = _mock_client(handler)
        result = asyncio.run(
            client.load_distribution("https://data.example.gov/f.csv", max_rows=10)
        )
        assert result["row_count"] == 10
        assert result["truncated"] is True
        assert result["columns"] == ["a", "b"]
        assert result["rows"][0] == {"a": "0", "b": "0"}
        asyncio.run(client.close())

    def test_short_file_is_not_marked_truncated(self):
        def handler(request):
            return httpx.Response(200, text="a,b\n1,2\n3,4\n")

        client = _mock_client(handler)
        result = asyncio.run(
            client.load_distribution("https://data.example.gov/f.csv", max_rows=100)
        )
        assert result["row_count"] == 2
        assert result["truncated"] is False
        asyncio.run(client.close())

    def test_quoted_newlines_do_not_split_rows(self):
        body = 'name,note\n"Smith, J","line one\nline two"\n'

        def handler(request):
            return httpx.Response(200, text=body)

        client = _mock_client(handler)
        result = asyncio.run(
            client.load_distribution("https://data.example.gov/f.csv", max_rows=10)
        )
        assert result["row_count"] == 1
        assert result["rows"][0]["name"] == "Smith, J"
        assert "line one\nline two" == result["rows"][0]["note"]
        asyncio.run(client.close())

    def test_ragged_rows_do_not_raise(self):
        def handler(request):
            return httpx.Response(200, text="a,b\n1\n1,2,3\n")

        client = _mock_client(handler)
        result = asyncio.run(
            client.load_distribution("https://data.example.gov/f.csv", max_rows=10)
        )
        assert result["rows"][0] == {"a": "1", "b": None}
        assert result["rows"][1]["col_2"] == "3"
        asyncio.run(client.close())

    def test_byte_cap_stops_a_huge_file(self):
        def handler(request):
            return httpx.Response(200, text="a\n" + "x\n" * 100000)

        client = _mock_client(handler)
        result = asyncio.run(
            client.load_distribution(
                "https://data.example.gov/f.csv", max_rows=10**9, max_bytes=500
            )
        )
        assert result["hit_byte_cap"] is True
        assert result["truncated"] is True
        asyncio.run(client.close())


class TestOutboundSafety:
    """Catalogs are third-party documents naming arbitrary hosts, so every
    outbound URL is checked before it is fetched."""

    def test_rejects_cloud_metadata_endpoint(self):
        client = DCATClient("https://data.example.gov")
        with pytest.raises(UnsafeMCPTarget):
            client._safe_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_loopback(self):
        client = DCATClient("https://data.example.gov")
        with pytest.raises(UnsafeMCPTarget):
            client._safe_url("http://127.0.0.1:8080/data.json")

    def test_rejects_non_http_scheme(self):
        client = DCATClient("https://data.example.gov")
        with pytest.raises(UnsafeMCPTarget):
            client._safe_url("file:///etc/passwd")


class TestCatalogCaching:
    def test_catalog_is_fetched_once_and_reused(self):
        clear_catalog_cache()
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json=SAMPLE_CATALOG)

        client = _mock_client(handler)

        async def run():
            a = await client.fetch_catalog()
            b = await client.fetch_catalog()
            return a, b

        a, b = asyncio.run(run())
        assert calls["n"] == 1
        assert a is b
        clear_catalog_cache()
        asyncio.run(client.close())

    def test_concurrent_callers_share_one_fetch(self):
        clear_catalog_cache()
        calls = {"n": 0}

        def handler(request):
            calls["n"] += 1
            return httpx.Response(200, json=SAMPLE_CATALOG)

        client = _mock_client(handler)

        async def run():
            await asyncio.gather(*(client.fetch_catalog() for _ in range(5)))

        asyncio.run(run())
        assert calls["n"] == 1
        clear_catalog_cache()
        asyncio.run(client.close())


class TestRegistry:
    """Portal-type handling in the admin-managed registry."""

    def test_missing_portal_type_means_ckan(self):
        from data_concierge.gateway.ckan_sites import normalize_portal_type

        # Entries persisted before DCAT existed carry no portal_type; treating
        # them as anything but CKAN would change how they are queried.
        assert normalize_portal_type(None) == "ckan"
        assert normalize_portal_type("") == "ckan"
        assert normalize_portal_type("socrata") == "ckan"
        assert normalize_portal_type("DCAT") == "dcat"

    def test_pa_portal_is_registered_as_dcat(self):
        from data_concierge.gateway import ckan_sites

        site = ckan_sites.get_site("data-pa-gov")
        assert site is not None
        assert ckan_sites.normalize_portal_type(site["portal_type"]) == "dcat"
        assert site["catalog_url"] == "https://data.pa.gov/data.json"

    def test_portal_type_resolves_by_url(self):
        from data_concierge.gateway import ckan_sites

        assert ckan_sites.portal_type_for_url("https://data.pa.gov") == "dcat"
        assert ckan_sites.portal_type_for_url("https://data.pa.gov/") == "dcat"
        assert ckan_sites.portal_type_for_url("https://data.wprdc.org") == "ckan"
        # An unregistered URL must fail closed to CKAN, not to DCAT.
        assert ckan_sites.portal_type_for_url("https://unknown.example") == "ckan"

    def test_deleted_default_is_not_resurrected(self, tmp_path):
        """A default an admin removed must stay removed across reloads."""
        from data_concierge.data_layer.storage import storage
        from data_concierge.gateway import ckan_sites

        original_root = storage.root
        storage.root = tmp_path
        try:
            assert ckan_sites.get_site("data-pa-gov") is not None
            assert ckan_sites.remove_site("data-pa-gov") is True
            assert ckan_sites.get_site("data-pa-gov") is None
            # Reload: the seed-merge must not add it back.
            assert ckan_sites.get_site("data-pa-gov") is None
        finally:
            storage.root = original_root

    def test_new_default_reaches_a_legacy_install(self, tmp_path):
        """An install predating the PA default gets it on the next load."""
        from data_concierge.data_layer.storage import storage
        from data_concierge.gateway import ckan_sites

        original_root = storage.root
        storage.root = tmp_path
        try:
            storage.write_json(
                "ckan_sites.json",
                {"sites": [{"id": "wprdc", "url": "https://data.wprdc.org", "name": "W"}]},
            )
            ids = [s["id"] for s in ckan_sites.list_sites()]
            assert "data-pa-gov" in ids
            assert "wprdc" in ids
        finally:
            storage.root = original_root

    def test_add_site_persists_portal_type_and_catalog_url(self, tmp_path):
        from data_concierge.data_layer.storage import storage
        from data_concierge.gateway import ckan_sites

        original_root = storage.root
        storage.root = tmp_path
        try:
            entry = ckan_sites.add_site(
                url="https://data.ny.gov",
                name="NY Open Data",
                portal_type="dcat",
                catalog_url="https://data.ny.gov/data.json",
            )
            assert entry["portal_type"] == "dcat"
            cfg = ckan_sites.get_portal_config(entry["id"])
            assert cfg["portal_type"] == "dcat"
            assert cfg["catalog_url"] == "https://data.ny.gov/data.json"
        finally:
            storage.root = original_root


class TestAgentDispatch:
    def test_sql_is_refused_on_a_dcat_portal(self):
        """A DCAT catalog has no query engine; the refusal must name the
        alternative rather than surfacing a transport error."""
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        agent = LLMAnalysisAgent()
        result = asyncio.run(
            agent._execute_tool(
                "run_sql_query", {"sql": "SELECT 1"}, "https://data.pa.gov"
            )
        )
        assert "not available on a DCAT portal" in result
        assert "load_resource_data" in result

    def test_ckan_portal_still_routes_to_ckan_tools(self):
        from data_concierge.agents.llm_agent import _portal_type_for_url

        assert _portal_type_for_url("https://data.wprdc.org") == "ckan"
        assert _portal_type_for_url("https://data.pa.gov") == "dcat"


class TestDcatCodegen:
    """Generated notebook cells must parse — a SyntaxError fails verification
    for the whole notebook."""

    @pytest.mark.parametrize(
        "tool_name,tool_input",
        [
            ("search_datasets", {"query": "opioid deaths", "rows": 5}),
            ("get_dataset_info", {"dataset_id": "aaaa-1111"}),
            ("load_resource_data", {"resource_id": "aaaa-1111", "limit": 50}),
            (
                "load_resource_data",
                {"resource_id": "https://data.pa.gov/x/rows.csv", "limit": 10},
            ),
        ],
    )
    def test_generated_cell_compiles(self, tool_name, tool_input):
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        code = LLMAnalysisAgent._code_for_tool(tool_name, tool_input, "https://data.pa.gov")
        compile(code, f"<{tool_name}>", "exec")

    def test_semantic_search_cell_does_not_call_the_ckan_api(self):
        """Regression: the semantic-search stand-in emitted CKAN's
        package_search, which 404s on a DCAT portal and failed the notebook."""
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        code = LLMAnalysisAgent._code_for_tool(
            "semantic_search_resources",
            {"query": "opioid deaths", "n_results": 5},
            "https://data.pa.gov",
        )
        assert "package_search" not in code
        assert "/api/3/action" not in code
        assert "data.pa.gov/data.json" in code
        compile(code, "<sem>", "exec")

    def test_ckan_portal_keeps_the_ckan_stand_in(self):
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        code = LLMAnalysisAgent._code_for_tool(
            "semantic_search_resources",
            {"query": "311 requests", "n_results": 5},
            "https://data.wprdc.org",
        )
        assert "package_search" in code

    def test_quotes_in_query_do_not_break_the_cell(self):
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        code = LLMAnalysisAgent._code_for_tool(
            "search_datasets",
            {"query": "it's a \"quoted\" query\\with backslash"},
            "https://data.pa.gov",
        )
        compile(code, "<search>", "exec")

    def test_row_limit_is_carried_into_the_cell(self):
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        code = LLMAnalysisAgent._code_for_tool(
            "load_resource_data", {"resource_id": "abc", "limit": 250}, "https://data.pa.gov"
        )
        # The notebook must reproduce the same slice the agent read.
        assert "nrows=250" in code


class TestPineconeRecords:
    """The record shape is a contract with the query side of the store."""

    SAMPLE_INDEX = {
        "datasets": [
            {
                "dataset_id": "aaaa-1111",
                "dataset_title": "Overdose Deaths Health Care Cost Containment Council",
                "dataset_description": "Deaths by county.",
                "organization": "data.example.gov",
                "tags": ["opioid", "health"],
                "resources": [
                    {
                        "resource_id": "aaaa-1111",
                        "resource_name": "Overdose Deaths",
                        "format": "CSV",
                        "row_count": 68,
                        "columns": [
                            {"name": "County Name", "stats": {}},
                            {
                                "name": "Time Period",
                                "stats": {"min": "2016", "max": "2020"},
                            },
                            {"name": "Count of Deaths", "stats": {}},
                        ],
                    }
                ],
            }
        ]
    }

    def test_record_carries_every_field_the_search_reads(self):
        from data_concierge.data_layer.pinecone_upload import build_records

        records = build_records(self.SAMPLE_INDEX, site_id="pa")
        assert len(records) == 1
        rec = records[0]
        # These names are exactly what _pinecone_search requests in `fields`.
        for field in (
            "resource_id", "resource_name", "dataset_id", "dataset_title",
            "format", "record_count", "column_count", "ai_tags",
            "has_temporal", "has_geographic", "has_demographic", "has_financial",
            "temporal_min", "temporal_max", "text", "description",
        ):
            assert field in rec, f"missing {field}"
        assert rec["_id"] == "pa:aaaa-1111"
        assert rec["record_count"] == 68
        assert rec["column_count"] == 3

    def test_facets_come_from_columns_not_the_title(self):
        """"Cost Containment Council" in a title must not mark a health
        dataset as financial — a facet true for everything cannot filter."""
        from data_concierge.data_layer.pinecone_upload import build_records

        rec = build_records(self.SAMPLE_INDEX, site_id="pa")[0]
        assert rec["has_financial"] is False
        assert rec["has_temporal"] is True
        assert rec["has_geographic"] is True
        assert rec["has_demographic"] is False

    def test_temporal_range_comes_from_qsv_stats(self):
        from data_concierge.data_layer.pinecone_upload import build_records

        rec = build_records(self.SAMPLE_INDEX, site_id="pa")[0]
        assert rec["temporal_min"] == "2016"
        assert rec["temporal_max"] == "2020"

    def test_ids_are_namespaced_by_site(self):
        """Two portals with the same resource ID must not overwrite each
        other in a shared index."""
        from data_concierge.data_layer.pinecone_upload import build_records

        a = build_records(self.SAMPLE_INDEX, site_id="pa")[0]["_id"]
        b = build_records(self.SAMPLE_INDEX, site_id="ny")[0]["_id"]
        assert a != b

    def test_resource_without_identity_is_skipped(self):
        from data_concierge.data_layer.pinecone_upload import build_record

        assert build_record({"columns": []}, site_id="pa") is None

    def test_embedded_text_is_capped(self):
        from data_concierge.data_layer.pinecone_upload import MAX_TEXT_CHARS, build_record

        wide = {
            "resource_id": "x",
            "dataset_title": "T",
            "dataset_description": "d" * 50000,
            "columns": [{"name": f"col_{i}", "qsv_description": "y" * 500} for i in range(300)],
        }
        rec = build_record(wide, site_id="pa")
        assert rec is not None
        assert len(rec["text"]) <= MAX_TEXT_CHARS

    def test_upload_dry_run_writes_nothing(self):
        from data_concierge.data_layer.pinecone_upload import upload_index

        summary = upload_index(self.SAMPLE_INDEX, site_id="pa", dry_run=True)
        assert summary["dry_run"] is True
        assert summary["upserted"] == 0
        assert summary["records_built"] == 1
        assert "sample" in summary


class TestQsvProfilingShared:
    def test_merge_columns_without_portal_fields(self):
        """A DCAT portal supplies no data dictionary, so columns must be
        buildable from qsv output alone."""
        from data_concierge.data_layer.qsv_profiling import merge_columns

        cols = merge_columns(
            None,
            None,
            {"County": {"type": "String", "min": "Adams"}},
            {"County": [{"value": "Adams", "count": 5}]},
        )
        assert len(cols) == 1
        assert cols[0]["name"] == "County"
        assert cols[0]["stats"]["type"] == "String"
        assert cols[0]["top_values"][0]["value"] == "Adams"

    def test_merge_columns_still_uses_ckan_field_ids(self):
        from data_concierge.data_layer.qsv_profiling import merge_columns

        cols = merge_columns([{"id": "amount", "type": "numeric"}], None, None, None)
        assert cols[0]["name"] == "amount"
        assert cols[0]["ckan_type"] == "numeric"


def test_catalog_json_round_trips_through_parse():
    """Guards the parse path against a JSON document with unusual nesting."""
    body = json.loads(json.dumps(SAMPLE_CATALOG))
    assert len(parse_catalog(body, "u").datasets) == 2


class TestRetrievalSignals:
    """A DCAT load must feed the confidence signals, and must not invent a
    dataset total the catalog never reported."""

    def _extract(self, result_text: str) -> tuple[int, list[int]]:
        """Mirror the signal extraction in llm_agent.process for one load."""
        import re

        total_rows_loaded = 0
        record_counts: list[int] = []
        total_match = re.search(r"Total records:\s*([\d,]+)", result_text)
        if total_match:
            count = int(total_match.group(1).replace(",", ""))
            record_counts.append(count)
            total_rows_loaded += min(count, 100)
        else:
            loaded_match = re.search(r"Loaded:\s*([\d,]+)\s+rows", result_text)
            if loaded_match:
                loaded = int(loaded_match.group(1).replace(",", ""))
                total_rows_loaded += loaded
                if "NOTE: truncated" not in result_text:
                    record_counts.append(loaded)
        return total_rows_loaded, record_counts

    def test_complete_read_counts_as_a_known_total(self):
        text = (
            "Dataset: X\nDistribution: u\nLoaded: 68 rows, 11 columns\n"
            "Complete: the full distribution fit within the read limit."
        )
        rows, counts = self._extract(text)
        assert rows == 68
        assert counts == [68]

    def test_truncated_read_counts_rows_but_claims_no_total(self):
        text = (
            "Dataset: X\nDistribution: u\nLoaded: 500 rows, 11 columns\n"
            "NOTE: truncated at the 500-row read limit — this is the first slice"
        )
        rows, counts = self._extract(text)
        assert rows == 500
        # The file's true size is unknown; recording 500 as the total would
        # report a floor as a measurement.
        assert counts == []

    def test_ckan_format_still_parses(self):
        rows, counts = self._extract("Resource: r\nTotal records: 12,345\nLoaded: 100")
        assert counts == [12345]
        assert rows == 100

    def test_dcat_load_output_shape_matches_the_extractor(self):
        """Guards the coupling: the tool's wording is what the signal parses."""
        import asyncio

        import httpx

        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        def handler(request):
            return httpx.Response(200, text="a,b\n1,2\n3,4\n")

        client = _mock_client(handler)
        agent = LLMAnalysisAgent()
        out = asyncio.run(
            agent._tool_dcat_load(
                client, {"resource_id": "https://data.example.gov/f.csv", "limit": 100}
            )
        )
        rows, counts = self._extract(out)
        assert rows == 2
        assert counts == [2]
        asyncio.run(client.close())
