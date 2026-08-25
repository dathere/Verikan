"""MCP (Model Context Protocol) integration for Data Concierge.

Provides infrastructure for connecting to MCP servers, managing their lifecycle,
and exposing their tools to the agent pipeline.
"""

from data_concierge.mcp.models import MCPServerConfig, MCPServerStatus, MCPToolInfo
from data_concierge.mcp.registry import MCPServerRegistry, get_mcp_registry

__all__ = [
    "MCPServerConfig",
    "MCPServerStatus",
    "MCPToolInfo",
    "MCPServerRegistry",
    "get_mcp_registry",
]
