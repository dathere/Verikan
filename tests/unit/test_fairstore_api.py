"""Tests for the Fair Store dictionary API (issue #136)."""

import pytest
from httpx import ASGITransport, AsyncClient

from data_concierge.api.main import app


async def _get(path: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


class TestFairStoreSearch:
    async def test_requires_query(self) -> None:
        resp = await _get("/api/v1/fairstore/search?site=wprdc")
        assert resp.status_code == 422  # missing required q

    async def test_empty_query_rejected(self) -> None:
        resp = await _get("/api/v1/fairstore/search?q=%20&site=wprdc")
        assert resp.status_code == 400

    async def test_search_returns_shape(self) -> None:
        """Against the real onboarded index if present; graceful if not."""
        resp = await _get("/api/v1/fairstore/search?q=census&site=wprdc")
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "census"
        assert "results" in body and "available" in body
        if body["available"] and body["results"]:
            r = body["results"][0]
            assert {"resource_id", "column_count", "record_count", "score"} <= set(r)

    async def test_unknown_site_is_graceful(self) -> None:
        resp = await _get("/api/v1/fairstore/search?q=x&site=does-not-exist")
        assert resp.status_code == 200
        assert resp.json()["available"] is False


class TestFairStoreResource:
    async def test_missing_resource_is_404(self) -> None:
        resp = await _get("/api/v1/fairstore/resource/nonexistent-uuid?site=wprdc")
        assert resp.status_code == 404

    async def test_real_resource_carries_column_detail(self) -> None:
        """Find a real resource via search, then fetch its dictionary."""
        s = await _get("/api/v1/fairstore/search?q=census&site=wprdc")
        body = s.json()
        if not (body["available"] and body["results"]):
            pytest.skip("no onboarded data available in this environment")
        rid = body["results"][0]["resource_id"]
        resp = await _get(f"/api/v1/fairstore/resource/{rid}?site=wprdc")
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["resource_id"] == rid
        assert "columns_detail" in detail
        if detail["columns_detail"]:
            col = detail["columns_detail"][0]
            assert {"name", "type", "stats", "top_values"} <= set(col)


class TestSecretScrubbing:
    """Onboarded descriptions carry the qsv describegpt API key (issue found
    building #136). It must never reach the API, the UI, or the agent."""

    def test_scrubber_redacts_provider_keys(self) -> None:
        from data_concierge.data_layer.onboard_index import _scrub_secrets

        cases = [
            "run with --api-key sk-or-v1-deadbeef0123456789 now",
            "openrouter sk-or-v1-abc123def456ghi789 embedded",
            "anthropic sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA key",
            "--api-key=secrettoken12345",
        ]
        for text in cases:
            out = _scrub_secrets(text)
            assert "sk-or-v1" not in out
            assert "sk-ant" not in out
            assert "secrettoken" not in out
            assert "[redacted]" in out

    def test_scrubber_leaves_clean_text_untouched(self) -> None:
        from data_concierge.data_layer.onboard_index import _scrub_secrets

        clean = "This dataset contains 311 service requests for Pittsburgh."
        assert _scrub_secrets(clean) == clean

    def test_scrubber_handles_empty(self) -> None:
        from data_concierge.data_layer.onboard_index import _scrub_secrets

        assert _scrub_secrets("") == ""
        assert _scrub_secrets(None) is None

    async def test_api_descriptions_carry_no_key(self) -> None:
        """The served search/detail payloads must be key-free."""
        s = await _get("/api/v1/fairstore/search?q=census&site=wprdc")
        body = s.json()
        if not (body.get("available") and body.get("results")):
            import pytest

            pytest.skip("no onboarded data in this environment")
        blob = str(body)
        assert "sk-or-v1" not in blob and "--api-key" not in blob
        rid = body["results"][0]["resource_id"]
        d = await _get(f"/api/v1/fairstore/resource/{rid}?site=wprdc")
        detail_blob = str(d.json())
        assert "sk-or-v1" not in detail_blob
        assert "--api-key sk-" not in detail_blob


class TestSiteValidation:
    """Path-traversal via the site param (adversarial review finding)."""

    @staticmethod
    async def _get_search(site: str):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(f"/api/v1/fairstore/search?q=x&site={site}")

    async def test_traversal_site_rejected(self) -> None:
        for bad in ["../verified_notebooks", "..%2f..%2fetc", "a/b", "a..b", "../../etc"]:
            resp = await self._get_search(bad)
            assert resp.status_code == 400, f"{bad!r} should be rejected"

    async def test_normal_site_accepted(self) -> None:
        resp = await self._get_search("wprdc")
        assert resp.status_code == 200

    async def test_resource_endpoint_validates_site(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/v1/fairstore/resource/x?site=../secrets")
        assert resp.status_code == 400
