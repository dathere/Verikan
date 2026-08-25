"""Pydantic models for MCP server configuration and state."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MCPTransportType(StrEnum):
    """Transport type for MCP server communication."""

    STDIO = "stdio"
    SSE = "sse"
    # Modern MCP "Streamable HTTP" transport: a single endpoint (e.g. /mcp) that
    # accepts JSON-RPC over POST and returns the response in the HTTP body
    # (plain JSON, or a single SSE "message" event). No endpoint handshake.
    STREAMABLE_HTTP = "streamable_http"


class MCPServerStatus(StrEnum):
    """Connection status of an MCP server."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MCPToolInfo(BaseModel):
    """Information about a tool exposed by an MCP server."""

    name: str = Field(..., description="Tool name")
    description: str = Field(default="", description="Tool description")
    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for tool parameters"
    )


class MCPPromptInfo(BaseModel):
    """Information about a prompt exposed by an MCP server."""

    name: str = Field(..., description="Prompt name")
    description: str = Field(default="", description="Prompt description")
    arguments: list[dict[str, Any]] = Field(
        default_factory=list, description="Prompt arguments"
    )


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server connection."""

    id: str = Field(..., description="Unique server identifier")
    name: str = Field(..., description="Human-readable server name")
    description: str = Field(default="", description="Server description")

    # Transport configuration
    transport: MCPTransportType = Field(
        default=MCPTransportType.STDIO, description="Transport type"
    )

    # For stdio transport
    command: str = Field(default="", description="Command to start the server process")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the process"
    )
    working_dir: str = Field(
        default="", description="Working directory for the server process (cwd)"
    )

    # For SSE transport
    url: str = Field(default="", description="Server URL for SSE transport")

    # Metadata
    enabled: bool = Field(default=True, description="Whether the server is enabled")
    auto_connect: bool = Field(
        default=True, description="Auto-connect on startup"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="Creation timestamp"
    )

    # Data source integration
    categories: list[str] = Field(
        default_factory=list, description="Data categories this server covers"
    )
    keywords: list[str] = Field(
        default_factory=list, description="Keywords for matching queries"
    )

    @field_validator("url")
    @classmethod
    def _url_must_be_safe(cls, v: str) -> str:
        """Reject SSRF-dangerous URLs as a model invariant (#135).

        This runs on *every* construction — API create/update, and loading
        configs from disk — so validation cannot be bypassed by editing a
        server after registration or by seeding configs/mcp_servers.json.

        Only the cheap syntactic checks run here (scheme, literal-IP,
        metadata host): no DNS, so model construction stays fast and works
        offline. The full resolving check runs at the API and connect
        boundaries, which is where a rebinding hostname is caught.
        """
        if not v:
            return v
        # Imported lazily to avoid a models -> guards -> ... import cycle.
        from data_concierge.core.config import settings
        from data_concierge.mcp.guards import UnsafeMCPTarget, validate_server_url

        try:
            return validate_server_url(
                v, allow_private=settings.mcp_allow_private_urls, resolve=False
            )
        except UnsafeMCPTarget as e:
            raise ValueError(f"unsafe MCP server URL: {e}") from None


class MCPServerState(BaseModel):
    """Runtime state of an MCP server connection."""

    config: MCPServerConfig
    status: MCPServerStatus = MCPServerStatus.DISCONNECTED
    tools: list[MCPToolInfo] = Field(default_factory=list)
    prompts: list[MCPPromptInfo] = Field(default_factory=list)
    last_connected: str | None = None
    last_error: str | None = None
    server_name: str | None = Field(
        default=None, description="Server name reported during initialization"
    )
    server_version: str | None = Field(
        default=None, description="Server version reported during initialization"
    )


class MCPToolCallResult(BaseModel):
    """Result from calling an MCP tool."""

    content: list[dict[str, Any]] = Field(
        default_factory=list, description="Response content blocks"
    )
    is_error: bool = Field(default=False, description="Whether the call resulted in an error")

    @property
    def text(self) -> str:
        """Extract text content from the result."""
        parts = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
