"""Tests for MCP endpoint containment (issue #135)."""

import pytest

from data_concierge.mcp.guards import (
    UnsafeMCPTarget,
    pin_sse_endpoint,
    validate_server_url,
)


class TestUrlValidation:
    def test_public_https_is_allowed(self) -> None:
        assert validate_server_url("https://example.com/mcp")

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/computeMetadata/v1/",
            "http://metadata.google.internal/",
            "http://metadata/",
            "http://100.100.100.100/",
        ],
    )
    def test_cloud_metadata_is_refused(self, url: str) -> None:
        """The highest-value SSRF target: it hands out instance credentials."""
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1:8080/mcp",
            "http://localhost:8080/mcp",
            "http://10.0.0.5/mcp",
            "http://192.168.1.10/mcp",
            "http://172.16.0.1/mcp",
            "http://[::1]/mcp",
            "http://0.0.0.0/mcp",
        ],
    )
    def test_internal_addresses_are_refused(self, url: str) -> None:
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url(url)

    @pytest.mark.parametrize(
        "url", ["file:///etc/passwd", "gopher://x/", "ftp://x/", "javascript:alert(1)"]
    )
    def test_non_http_schemes_are_refused(self, url: str) -> None:
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url(url)

    def test_empty_and_hostless(self) -> None:
        for bad in ("", "   ", "http://"):
            with pytest.raises(UnsafeMCPTarget):
                validate_server_url(bad)

    def test_localhost_allowed_only_when_opted_in(self) -> None:
        """Local dev needs it; nothing reachable should set it."""
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url("http://localhost:8080/mcp")
        assert validate_server_url("http://localhost:8080/mcp", allow_private=True)

    def test_metadata_refused_even_with_private_allowed(self) -> None:
        """allow_private is for localhost, not for credential endpoints."""
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url("http://169.254.169.254/", allow_private=True)

    def test_ships_locked_down(self) -> None:
        from data_concierge.core.config import settings

        assert settings.mcp_allow_private_urls is False


class TestSseEndpointPinning:
    CFG = "https://mcp.example.com:8443/sse"

    def test_relative_path_joins_configured_origin(self) -> None:
        assert (
            pin_sse_endpoint(self.CFG, "/messages/?id=1")
            == "https://mcp.example.com:8443/messages/?id=1"
        )

    def test_path_without_leading_slash(self) -> None:
        assert pin_sse_endpoint(self.CFG, "messages") == "https://mcp.example.com:8443/messages"

    def test_same_origin_absolute_is_kept(self) -> None:
        same = "https://mcp.example.com:8443/messages/?id=1"
        assert pin_sse_endpoint(self.CFG, same) == same

    def test_foreign_origin_is_stripped(self) -> None:
        """A hostile server must not redirect our JSON-RPC bodies elsewhere."""
        pinned = pin_sse_endpoint(self.CFG, "https://attacker.example/collect?id=1")
        assert pinned.startswith("https://mcp.example.com:8443/")
        assert "attacker.example" not in pinned
        assert pinned == "https://mcp.example.com:8443/collect?id=1"

    def test_metadata_redirect_is_stripped(self) -> None:
        pinned = pin_sse_endpoint(self.CFG, "http://169.254.169.254/latest/meta-data/")
        assert "169.254.169.254" not in pinned

    def test_empty_endpoint_is_refused(self) -> None:
        with pytest.raises(UnsafeMCPTarget):
            pin_sse_endpoint(self.CFG, "")


class TestSolrFilterEscaping:
    def test_organization_is_quoted_and_escaped(self) -> None:
        """A model-supplied org must not restructure the Solr filter."""
        import inspect

        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        src = inspect.getsource(LLMAnalysisAgent._tool_search)
        assert 'organization:"' in src, "fq value must be quoted"
        assert 'f"organization:{org}"' not in src, "unquoted interpolation is back"


class TestGuardHardening:
    """Fixes from the adversarial review of the SSRF guards."""

    def test_model_validator_rejects_metadata_url(self) -> None:
        """URL validation is a model invariant, not a per-endpoint bolt-on."""
        import pytest
        from pydantic import ValidationError

        from data_concierge.mcp.models import MCPServerConfig, MCPTransportType

        with pytest.raises(ValidationError):
            MCPServerConfig(
                id="x", name="x", transport=MCPTransportType.STREAMABLE_HTTP,
                url="http://169.254.169.254/latest/meta-data/",
            )

    def test_model_validator_rejects_on_update_construction(self) -> None:
        """The update path rebuilds the model, so it is covered too."""
        import pytest
        from pydantic import ValidationError

        from data_concierge.mcp.models import MCPServerConfig, MCPTransportType

        good = MCPServerConfig(
            id="x", name="x", transport=MCPTransportType.STREAMABLE_HTTP,
            url="https://example.com/mcp",
        )
        data = good.model_dump()
        data["url"] = "http://127.0.0.1:8080/mcp"  # what a malicious PUT would set
        with pytest.raises(ValidationError):
            MCPServerConfig(**data)

    def test_syntactic_check_skips_dns(self) -> None:
        """resolve=False must not hit the network (public hostname passes)."""
        # A hostname that would need DNS; with resolve=False it is accepted
        # syntactically without a lookup.
        assert validate_server_url("https://mcp.example.com/x", resolve=False)

    def test_trailing_dot_metadata_host_blocked(self) -> None:
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url("http://metadata.google.internal./", resolve=False)

    def test_allow_private_still_blocks_metadata_literal(self) -> None:
        with pytest.raises(UnsafeMCPTarget):
            validate_server_url("http://169.254.169.254/", allow_private=True)

    def test_allow_private_permits_loopback_literal(self) -> None:
        assert validate_server_url("http://127.0.0.1:8080/x", allow_private=True)
