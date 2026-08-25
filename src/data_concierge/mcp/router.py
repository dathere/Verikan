"""FastAPI router for MCP server management endpoints."""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from data_concierge.core.config import settings
from data_concierge.core.logging import get_logger
from data_concierge.gateway.router import require_admin
from data_concierge.mcp.models import MCPServerConfig, MCPServerStatus, MCPTransportType
from data_concierge.mcp.registry import get_mcp_registry

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/mcp", tags=["mcp"], dependencies=[Depends(require_admin)])


# =============================================================================
# Request/Response Models
# =============================================================================


class AddMCPServerRequest(BaseModel):
    """Request to register a new MCP server."""

    name: str = Field(..., min_length=1, max_length=200, description="Server name")
    description: str = Field(default="", description="Server description")
    transport: str = Field(default="stdio", description="Transport type: 'stdio' or 'sse'")

    # stdio fields
    command: str = Field(default="", description="Command to start the server")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    working_dir: str = Field(default="", description="Working directory for the server process")

    # SSE fields
    url: str = Field(default="", description="Server URL for SSE transport")

    # Metadata
    auto_connect: bool = Field(default=False, description="Auto-connect on startup")
    categories: list[str] = Field(default_factory=list, description="Data categories")
    keywords: list[str] = Field(default_factory=list, description="Search keywords")


class UpdateMCPServerRequest(BaseModel):
    """Request to update an MCP server configuration."""

    name: str | None = None
    description: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    working_dir: str | None = None
    url: str | None = None
    enabled: bool | None = None
    auto_connect: bool | None = None
    categories: list[str] | None = None
    keywords: list[str] | None = None


class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool."""

    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")


class MCPPromptRequest(BaseModel):
    """Request to get an MCP prompt."""

    prompt_name: str = Field(..., description="Name of the prompt")
    arguments: dict[str, str] = Field(default_factory=dict, description="Prompt arguments")


# =============================================================================
# Server Management Endpoints
# =============================================================================


@router.get("/servers")
async def list_mcp_servers() -> dict[str, Any]:
    """List all configured MCP servers."""
    registry = get_mcp_registry()
    servers = registry.get_all_servers()

    return {
        "count": len(servers),
        "servers": [
            {
                "id": s.config.id,
                "name": s.config.name,
                "description": s.config.description,
                "transport": s.config.transport.value,
                "command": s.config.command,
                "args": s.config.args,
                "working_dir": s.config.working_dir,
                "url": s.config.url,
                "enabled": s.config.enabled,
                "auto_connect": s.config.auto_connect,
                "status": s.status.value,
                "tools": [
                    {"name": t.name, "description": t.description}
                    for t in s.tools
                ],
                "prompts": [
                    {"name": p.name, "description": p.description}
                    for p in s.prompts
                ],
                "server_name": s.server_name,
                "server_version": s.server_version,
                "last_connected": s.last_connected,
                "last_error": s.last_error,
                "categories": s.config.categories,
                "keywords": s.config.keywords,
                "created_at": s.config.created_at,
            }
            for s in servers
        ],
    }


@router.post("/servers")
async def add_mcp_server(request: AddMCPServerRequest) -> dict[str, Any]:
    """Register a new MCP server."""
    registry = get_mcp_registry()

    # Generate ID from name
    server_id = request.name.lower().replace(" ", "-").replace("_", "-")
    server_id = "".join(c for c in server_id if c.isalnum() or c == "-")

    # Ensure unique
    if registry.get_server(server_id):
        server_id = f"{server_id}-{uuid.uuid4().hex[:6]}"

    # Map the requested transport properly. This used to be a two-way pick
    # between SSE and STDIO, so "streamable_http" silently registered as a
    # stdio server and then failed to connect.
    try:
        transport = MCPTransportType(request.transport)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown transport {request.transport!r}. Use one of: "
                + ", ".join(t.value for t in MCPTransportType)
            ),
        ) from None

    # Validate transport-specific fields
    if transport == MCPTransportType.STDIO and not request.command:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Command is required for stdio transport",
        )
    if transport in (MCPTransportType.SSE, MCPTransportType.STREAMABLE_HTTP) and not request.url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL is required for {transport.value} transport",
        )

    # Refuse targets that would turn the agent into an SSRF proxy (#135) —
    # cloud metadata endpoints above all, since those hand out instance
    # credentials to anything that asks.
    if request.url:
        from data_concierge.mcp.guards import UnsafeMCPTarget, validate_server_url

        try:
            validate_server_url(
                request.url, allow_private=settings.mcp_allow_private_urls
            )
        except UnsafeMCPTarget as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Refusing to register this URL: {e}",
            ) from None

    config = MCPServerConfig(
        id=server_id,
        name=request.name,
        description=request.description,
        transport=transport,
        command=request.command,
        args=request.args,
        env=request.env,
        working_dir=request.working_dir,
        url=request.url,
        enabled=True,
        auto_connect=request.auto_connect,
        categories=request.categories,
        keywords=request.keywords,
    )

    state = registry.register_server(config)

    return {
        "message": f"MCP server '{request.name}' registered",
        "server_id": server_id,
        "status": state.status.value,
    }


@router.get("/servers/{server_id}")
async def get_mcp_server(server_id: str) -> dict[str, Any]:
    """Get detailed info about an MCP server."""
    registry = get_mcp_registry()
    state = registry.get_server(server_id)

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_id}' not found",
        )

    return {
        "id": state.config.id,
        "name": state.config.name,
        "description": state.config.description,
        "transport": state.config.transport.value,
        "command": state.config.command,
        "args": state.config.args,
        "env": dict.fromkeys(state.config.env, "***"),  # Mask env values
        "working_dir": state.config.working_dir,
        "url": state.config.url,
        "enabled": state.config.enabled,
        "auto_connect": state.config.auto_connect,
        "status": state.status.value,
        "tools": [t.model_dump() for t in state.tools],
        "prompts": [p.model_dump() for p in state.prompts],
        "server_name": state.server_name,
        "server_version": state.server_version,
        "last_connected": state.last_connected,
        "last_error": state.last_error,
        "categories": state.config.categories,
        "keywords": state.config.keywords,
    }


@router.put("/servers/{server_id}")
async def update_mcp_server(
    server_id: str, request: UpdateMCPServerRequest
) -> dict[str, Any]:
    """Update an MCP server's configuration."""
    registry = get_mcp_registry()

    updates = request.model_dump(exclude_none=True)
    state = registry.update_server_config(server_id, updates)

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_id}' not found",
        )

    return {
        "message": f"MCP server '{server_id}' updated",
        "server_id": server_id,
    }


@router.delete("/servers/{server_id}")
async def remove_mcp_server(server_id: str) -> dict[str, Any]:
    """Remove an MCP server."""
    registry = get_mcp_registry()

    if not registry.unregister_server(server_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_id}' not found",
        )

    return {"message": f"MCP server '{server_id}' removed"}


# =============================================================================
# Connection Management
# =============================================================================


@router.post("/servers/{server_id}/connect")
async def connect_mcp_server(server_id: str) -> dict[str, Any]:
    """Connect to an MCP server."""
    registry = get_mcp_registry()

    try:
        state = await registry.connect_server(server_id)
        return {
            "message": f"Connected to '{state.config.name}'",
            "server_id": server_id,
            "status": state.status.value,
            "server_name": state.server_name,
            "server_version": state.server_version,
            "tools": [
                {"name": t.name, "description": t.description}
                for t in state.tools
            ],
            "prompts": [
                {"name": p.name, "description": p.description}
                for p in state.prompts
            ],
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e)
        ) from e


@router.post("/servers/{server_id}/disconnect")
async def disconnect_mcp_server(server_id: str) -> dict[str, Any]:
    """Disconnect from an MCP server."""
    registry = get_mcp_registry()

    try:
        state = await registry.disconnect_server(server_id)
        return {
            "message": f"Disconnected from '{state.config.name}'",
            "server_id": server_id,
            "status": state.status.value,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        ) from e


# =============================================================================
# Tool & Prompt Interaction
# =============================================================================


@router.get("/servers/{server_id}/tools")
async def list_server_tools(server_id: str) -> dict[str, Any]:
    """List tools available on an MCP server."""
    registry = get_mcp_registry()
    state = registry.get_server(server_id)

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"MCP server '{server_id}' not found",
        )

    if state.status != MCPServerStatus.CONNECTED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Server '{server_id}' is not connected (status: {state.status.value})",
        )

    return {
        "server_id": server_id,
        "count": len(state.tools),
        "tools": [t.model_dump() for t in state.tools],
    }


@router.post("/servers/{server_id}/tools/call")
async def call_server_tool(
    server_id: str, request: MCPToolCallRequest
) -> dict[str, Any]:
    """Call a tool on an MCP server."""
    registry = get_mcp_registry()

    try:
        result = await registry.call_tool(
            server_id, request.tool_name, request.arguments
        )
        return {
            "server_id": server_id,
            "tool_name": request.tool_name,
            "is_error": result.is_error,
            "content": result.content,
            "text": result.text,
        }
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tool call failed: {e}",
        ) from e


@router.post("/servers/{server_id}/prompts/get")
async def get_server_prompt(
    server_id: str, request: MCPPromptRequest
) -> dict[str, Any]:
    """Get a prompt from an MCP server."""
    registry = get_mcp_registry()

    try:
        result = await registry.get_prompt(
            server_id, request.prompt_name, request.arguments or None
        )
        return {
            "server_id": server_id,
            "prompt_name": request.prompt_name,
            "messages": result.get("messages", []),
        }
    except ConnectionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(e)
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt get failed: {e}",
        ) from e


# =============================================================================
# Discovery
# =============================================================================


@router.get("/tools")
async def list_all_tools() -> dict[str, Any]:
    """List all available tools across all connected MCP servers."""
    registry = get_mcp_registry()
    tools = registry.get_all_tools()

    return {
        "count": len(tools),
        "tools": [
            {
                "server_id": t["server_id"],
                "server_name": t["server_name"],
                "name": t["tool"].name,
                "description": t["tool"].description,
                "input_schema": t["tool"].input_schema,
            }
            for t in tools
        ],
    }
