"""Tests for MCP server integration."""


from data_concierge.mcp.models import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPToolCallResult,
    MCPToolInfo,
    MCPTransportType,
)
from data_concierge.mcp.registry import MCPServerRegistry


class TestMCPModels:
    """Test MCP Pydantic models."""

    def test_server_config_defaults(self):
        config = MCPServerConfig(id="test", name="Test Server")
        assert config.transport == MCPTransportType.STDIO
        assert config.enabled is True
        assert config.auto_connect is True
        assert config.categories == []
        assert config.keywords == []

    def test_server_config_stdio(self):
        config = MCPServerConfig(
            id="census",
            name="Census MCP",
            transport=MCPTransportType.STDIO,
            command="docker",
            args=["compose", "run", "--rm", "census-mcp"],
            categories=["demographics"],
        )
        assert config.command == "docker"
        assert len(config.args) == 4
        assert "demographics" in config.categories

    def test_server_config_sse(self):
        config = MCPServerConfig(
            id="remote",
            name="Remote Server",
            transport=MCPTransportType.SSE,
            url="http://localhost:3000",
        )
        assert config.transport == MCPTransportType.SSE
        assert config.url == "http://localhost:3000"

    def test_server_state_defaults(self):
        config = MCPServerConfig(id="test", name="Test")
        state = MCPServerState(config=config)
        assert state.status == MCPServerStatus.DISCONNECTED
        assert state.tools == []
        assert state.prompts == []
        assert state.server_name is None

    def test_tool_info(self):
        tool = MCPToolInfo(
            name="list-datasets",
            description="List Census datasets",
            input_schema={"type": "object", "properties": {}},
        )
        assert tool.name == "list-datasets"
        assert "Census" in tool.description

    def test_tool_call_result_text(self):
        result = MCPToolCallResult(
            content=[
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": "World"},
            ]
        )
        assert result.text == "Hello\nWorld"
        assert result.is_error is False

    def test_tool_call_result_error(self):
        result = MCPToolCallResult(
            content=[{"type": "text", "text": "Failed"}],
            is_error=True,
        )
        assert result.is_error is True
        assert result.text == "Failed"


class TestMCPRegistry:
    """Test MCP server registry."""

    def test_registry_init_with_defaults(self):
        registry = MCPServerRegistry()
        servers = registry.get_all_servers()
        # Servers are loaded from configs/mcp_servers.json (Census Data API + FBI).
        assert any(s.config.id == "census-data-api" for s in servers)

    def test_register_server(self):
        registry = MCPServerRegistry()
        config = MCPServerConfig(
            id="test-server",
            name="Test MCP Server",
            command="echo",
            args=["hello"],
        )
        state = registry.register_server(config, persist=False)
        assert state.status == MCPServerStatus.DISCONNECTED
        assert registry.get_server("test-server") is not None

    def test_unregister_server(self):
        registry = MCPServerRegistry()
        config = MCPServerConfig(id="temp", name="Temporary")
        registry.register_server(config, persist=False)
        assert registry.get_server("temp") is not None

        result = registry.unregister_server("temp")
        assert result is True
        assert registry.get_server("temp") is None

    def test_unregister_nonexistent(self):
        registry = MCPServerRegistry()
        result = registry.unregister_server("nonexistent")
        assert result is False

    def test_get_enabled_servers(self):
        registry = MCPServerRegistry()
        enabled = registry.get_enabled_servers()
        for s in enabled:
            assert s.config.enabled is True

    def test_get_all_tools_empty_when_disconnected(self):
        registry = MCPServerRegistry()
        tools = registry.get_all_tools()
        assert tools == []  # Nothing connected

    def test_find_tools_for_query_empty_when_disconnected(self):
        registry = MCPServerRegistry()
        tools = registry.find_tools_for_query("population of Texas")
        assert tools == []

    def test_update_server_config(self):
        registry = MCPServerRegistry()
        config = MCPServerConfig(id="update-test", name="Original Name")
        registry.register_server(config, persist=False)

        state = registry.update_server_config(
            "update-test", {"name": "Updated Name"}
        )
        assert state is not None
        assert state.config.name == "Updated Name"

        # Clean up: remove test server so it doesn't persist to disk
        registry.unregister_server("update-test")

    def test_update_nonexistent_server(self):
        registry = MCPServerRegistry()
        result = registry.update_server_config("nope", {"name": "Foo"})
        assert result is None

    def test_hosted_mcp_servers_loaded(self):
        # The Census Data API and FBI Crime Data MCP servers are loaded from
        # configs/mcp_servers.json as remote HTTP-based servers (no stdio default).
        registry = MCPServerRegistry()
        census = registry.get_server("census-data-api")
        assert census is not None
        assert census.config.transport == MCPTransportType.STREAMABLE_HTTP
        assert "census" in census.config.keywords
        # The retired Cloud Run census-mcp default is no longer registered.
        assert registry.get_server("census-mcp") is None


class TestMCPConnector:
    """Test MCP data connector."""

    def test_connector_import(self):
        from data_concierge.mcp.connector import MCPDataConnector, get_mcp_connector

        connector = get_mcp_connector()
        assert isinstance(connector, MCPDataConnector)

    def test_get_available_tools_empty(self):
        from data_concierge.mcp.connector import MCPDataConnector

        connector = MCPDataConnector()
        tools = connector.get_available_tools()
        assert tools == []  # Nothing connected

    def test_get_tool_definitions_for_llm_empty(self):
        from data_concierge.mcp.connector import MCPDataConnector

        connector = MCPDataConnector()
        defs = connector.get_tool_definitions_for_llm()
        assert defs == []

    def test_generate_census_code_fetch(self):
        from data_concierge.mcp.connector import MCPDataConnector

        connector = MCPDataConnector()
        code = connector.generate_reproducible_code(
            "census-mcp",
            "fetch-aggregate-data",
            {
                "dataset": "acs/acs5",
                "year": 2022,
                "get": {"variables": ["B01003_001E", "NAME"]},
                "for": "state:*",
            },
            "some result data",
        )
        assert "census" in code.lower() or "Census" in code
        assert "acs" in code.lower()
        assert "requests" in code

    def test_generate_census_code_fips(self):
        from data_concierge.mcp.connector import MCPDataConnector

        connector = MCPDataConnector()
        code = connector.generate_reproducible_code(
            "census-mcp",
            "resolve-geography-fips",
            {"geography_name": "Pittsburgh"},
            "FIPS: 4261000",
        )
        assert "Pittsburgh" in code
        assert "FIPS" in code


class TestMCPRouter:
    """Test MCP API router registration."""

    def test_router_has_endpoints(self):
        from data_concierge.mcp.router import router

        paths = [r.path for r in router.routes]
        assert "/api/v1/mcp/servers" in paths
        assert "/api/v1/mcp/tools" in paths


class TestLLMAgentMCPIntegration:
    """Test that the LLM agent properly integrates MCP tools."""

    def test_portal_configs_include_hosted_mcp_sources(self):
        from data_concierge.agents.llm_agent import PORTAL_CONFIGS

        assert "census-data-api" in PORTAL_CONFIGS
        cfg = PORTAL_CONFIGS["census-data-api"]
        assert "Census" in cfg["name"]
        assert cfg["quality_score"] == 0.95
        assert "fbi-crime-data" in PORTAL_CONFIGS

    def test_llm_graph_sources_include_hosted_mcp_sources(self):
        from data_concierge.agents.supervisor import LLM_GRAPH_SOURCES

        assert "census-data-api" in LLM_GRAPH_SOURCES
        assert "fbi-crime-data" in LLM_GRAPH_SOURCES

    def test_llm_agent_get_mcp_tools(self):
        from data_concierge.agents.llm_agent import LLMAnalysisAgent

        agent = LLMAnalysisAgent()
        # When no servers are connected, should return empty list
        tools = agent._get_mcp_tools()
        assert isinstance(tools, list)
        assert len(tools) == 0
