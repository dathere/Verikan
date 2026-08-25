"""MCP Server Registry for managing configured MCP servers.

Handles server registration, connection lifecycle, and persistence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from data_concierge.core.logging import get_logger
from data_concierge.mcp.client import MCPClient
from data_concierge.mcp.models import (
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPToolCallResult,
)

logger = get_logger(__name__)

# Persistence file for MCP server configs
MCP_CONFIG_DIR = Path.cwd() / "configs"
MCP_CONFIG_FILE = MCP_CONFIG_DIR / "mcp_servers.json"


class MCPServerRegistry:
    """Registry for managing MCP server connections.

    Handles server configuration, connection lifecycle, tool discovery,
    and persistence of server configs.
    """

    def __init__(self) -> None:
        self._servers: dict[str, MCPServerState] = {}
        self._clients: dict[str, MCPClient] = {}
        self._load_configs()
        self._register_default_servers()
        logger.info(
            "MCPServerRegistry initialized",
            server_count=len(self._servers),
        )

    def _register_default_servers(self) -> None:
        """Register built-in default MCP servers.

        There are no built-in defaults: all MCP servers (Census Data API, FBI
        Crime Data, etc.) are loaded from ``configs/mcp_servers.json`` via
        ``_load_configs``. Kept as a no-op hook so callers/tests stay valid.
        """
        return

    def _load_configs(self) -> None:
        """Load saved MCP server configurations from disk."""
        if not MCP_CONFIG_FILE.exists():
            return

        try:
            with open(MCP_CONFIG_FILE) as f:
                configs = json.load(f)

            for config_data in configs:
                config = MCPServerConfig(**config_data)
                self._servers[config.id] = MCPServerState(config=config)
                logger.debug("Loaded MCP server config", server_id=config.id)

        except Exception as e:
            logger.warning("Failed to load MCP configs", error=str(e))

    def _save_configs(self) -> None:
        """Save MCP server configurations to disk."""
        MCP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

        configs = [
            state.config.model_dump() for state in self._servers.values()
        ]

        try:
            with open(MCP_CONFIG_FILE, "w") as f:
                json.dump(configs, f, indent=2, default=str)
            logger.debug("Saved MCP configs", count=len(configs))
        except Exception as e:
            logger.warning("Failed to save MCP configs", error=str(e))

    def register_server(self, config: MCPServerConfig, persist: bool = True) -> MCPServerState:
        """Register a new MCP server.

        Args:
            config: Server configuration
            persist: Whether to save to disk

        Returns:
            Server state
        """
        state = MCPServerState(config=config)
        self._servers[config.id] = state

        if persist:
            self._save_configs()

        logger.info("Registered MCP server", server_id=config.id, name=config.name)
        return state

    def unregister_server(self, server_id: str) -> bool:
        """Remove an MCP server.

        Args:
            server_id: Server to remove

        Returns:
            True if removed
        """
        if server_id not in self._servers:
            return False

        # Disconnect if connected
        if server_id in self._clients:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._clients[server_id].disconnect())
                else:
                    loop.run_until_complete(self._clients[server_id].disconnect())
            except Exception:
                pass
            del self._clients[server_id]

        del self._servers[server_id]
        self._save_configs()

        logger.info("Unregistered MCP server", server_id=server_id)
        return True

    def get_server(self, server_id: str) -> MCPServerState | None:
        """Get a server's state by ID."""
        return self._servers.get(server_id)

    def get_all_servers(self) -> list[MCPServerState]:
        """Get all registered servers."""
        return list(self._servers.values())

    def get_enabled_servers(self) -> list[MCPServerState]:
        """Get all enabled servers."""
        return [s for s in self._servers.values() if s.config.enabled]

    def get_connected_servers(self) -> list[MCPServerState]:
        """Get all connected servers."""
        return [
            s for s in self._servers.values()
            if s.status == MCPServerStatus.CONNECTED
        ]

    async def connect_server(self, server_id: str) -> MCPServerState:
        """Connect to an MCP server.

        Args:
            server_id: Server to connect to

        Returns:
            Updated server state

        Raises:
            ValueError: If server not found
            RuntimeError: If connection fails
        """
        state = self._servers.get(server_id)
        if not state:
            raise ValueError(f"MCP server '{server_id}' not found")

        # Full resolving SSRF check at the last moment before we connect
        # (#135). The model validator is syntactic only; this catches a
        # hostname that resolves to an internal address — including DNS
        # rebinding, where the name resolved safely at registration and
        # points somewhere dangerous now.
        if state.config.url:
            from data_concierge.core.config import settings
            from data_concierge.mcp.guards import UnsafeMCPTarget, validate_server_url

            try:
                validate_server_url(
                    state.config.url, allow_private=settings.mcp_allow_private_urls
                )
            except UnsafeMCPTarget as e:
                state.status = MCPServerStatus.ERROR
                state.last_error = f"refused unsafe URL: {e}"
                raise RuntimeError(f"Refusing to connect to unsafe MCP URL: {e}") from None

        client = MCPClient(state.config)
        try:
            await client.connect()

            # Update state
            state.status = MCPServerStatus.CONNECTED
            state.tools = client.tools
            state.prompts = client.prompts
            state.server_name = client.server_name
            state.server_version = client.server_version
            state.last_connected = datetime.now().isoformat()
            state.last_error = None

            self._clients[server_id] = client

            return state

        except Exception as e:
            state.status = MCPServerStatus.ERROR
            state.last_error = str(e)
            raise RuntimeError(f"Failed to connect to '{server_id}': {e}") from e

    async def disconnect_server(self, server_id: str) -> MCPServerState:
        """Disconnect from an MCP server."""
        state = self._servers.get(server_id)
        if not state:
            raise ValueError(f"MCP server '{server_id}' not found")

        client = self._clients.get(server_id)
        if client:
            await client.disconnect()
            del self._clients[server_id]

        state.status = MCPServerStatus.DISCONNECTED
        state.tools = []
        state.prompts = []
        return state

    async def call_tool(
        self, server_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        """Call a tool on an MCP server.

        Args:
            server_id: Server that has the tool
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        client = self._clients.get(server_id)
        if not client:
            raise ConnectionError(f"MCP server '{server_id}' is not connected")

        return await client.call_tool(tool_name, arguments)

    async def get_prompt(
        self, server_id: str, prompt_name: str, arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Get a prompt from an MCP server."""
        client = self._clients.get(server_id)
        if not client:
            raise ConnectionError(f"MCP server '{server_id}' is not connected")

        return await client.get_prompt(prompt_name, arguments)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all available tools across all connected servers.

        Returns:
            List of tools with server info
        """
        all_tools = []
        for state in self._servers.values():
            if state.status == MCPServerStatus.CONNECTED:
                for tool in state.tools:
                    all_tools.append({
                        "server_id": state.config.id,
                        "server_name": state.config.name,
                        "tool": tool,
                    })
        return all_tools

    def find_tools_for_query(self, query: str) -> list[dict[str, Any]]:
        """Find MCP tools relevant to a query.

        Args:
            query: User's query text

        Returns:
            List of matching tools with server info, sorted by relevance
        """
        query_lower = query.lower()
        results = []

        for state in self._servers.values():
            if state.status != MCPServerStatus.CONNECTED:
                continue
            if not state.config.enabled:
                continue

            # Score based on keywords
            score = 0.0
            for keyword in state.config.keywords:
                if keyword.lower() in query_lower:
                    score += 1.0

            # Check tool descriptions
            for tool in state.tools:
                tool_score = score
                if tool.description:
                    for word in query_lower.split():
                        if len(word) > 3 and word in tool.description.lower():
                            tool_score += 0.5

                if tool_score > 0:
                    results.append({
                        "server_id": state.config.id,
                        "server_name": state.config.name,
                        "tool": tool,
                        "score": tool_score,
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def update_server_config(
        self, server_id: str, updates: dict[str, Any]
    ) -> MCPServerState | None:
        """Update a server's configuration.

        Args:
            server_id: Server to update
            updates: Fields to update

        Returns:
            Updated state, or None if not found
        """
        state = self._servers.get(server_id)
        if not state:
            return None

        config_data = state.config.model_dump()
        config_data.update(updates)
        state.config = MCPServerConfig(**config_data)

        self._save_configs()
        return state

    async def connect_all_auto(self) -> dict[str, str]:
        """Connect to all servers marked for auto-connect.

        Returns:
            Dict of server_id -> status message
        """
        results = {}
        for state in self._servers.values():
            if state.config.enabled and state.config.auto_connect:
                try:
                    await self.connect_server(state.config.id)
                    results[state.config.id] = "connected"
                except Exception as e:
                    results[state.config.id] = f"error: {e}"
        return results


# Singleton instance
_registry: MCPServerRegistry | None = None


def get_mcp_registry() -> MCPServerRegistry:
    """Get or create the MCP server registry singleton."""
    global _registry
    if _registry is None:
        _registry = MCPServerRegistry()
    return _registry
