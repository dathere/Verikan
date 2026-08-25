"""MCP client for communicating with MCP servers via stdio or SSE transport.

Implements the Model Context Protocol (MCP) JSON-RPC communication layer.
"""

import asyncio
import json
from typing import Any

import httpx

from data_concierge.core.logging import get_logger
from data_concierge.mcp.models import (
    MCPPromptInfo,
    MCPServerConfig,
    MCPServerStatus,
    MCPToolCallResult,
    MCPToolInfo,
    MCPTransportType,
)

logger = get_logger(__name__)


class MCPClient:
    """Client for communicating with an MCP server.

    Supports both stdio (subprocess) and SSE (HTTP) transports.
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.status = MCPServerStatus.DISCONNECTED
        self._request_id = 0
        self._process: asyncio.subprocess.Process | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._server_name: str | None = None
        self._server_version: str | None = None
        self._tools: list[MCPToolInfo] = []
        self._prompts: list[MCPPromptInfo] = []
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None
        self._last_stderr_lines: list[str] = []
        # SSE transport state
        self._sse_task: asyncio.Task[None] | None = None
        self._sse_endpoint: str | None = None
        self._sse_connected: asyncio.Event = asyncio.Event()
        self._sse_pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        # Streamable HTTP transport state
        self._mcp_session_id: str | None = None

    @property
    def server_name(self) -> str | None:
        return self._server_name

    @property
    def server_version(self) -> str | None:
        return self._server_version

    @property
    def tools(self) -> list[MCPToolInfo]:
        return self._tools

    @property
    def prompts(self) -> list[MCPPromptInfo]:
        return self._prompts

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def connect(self) -> None:
        """Establish connection to the MCP server."""
        self.status = MCPServerStatus.CONNECTING
        logger.info(
            "Connecting to MCP server",
            server_id=self.config.id,
            transport=self.config.transport.value,
        )

        try:
            if self.config.transport == MCPTransportType.STDIO:
                await self._connect_stdio()
            elif self.config.transport == MCPTransportType.STREAMABLE_HTTP:
                await self._connect_streamable_http()
            else:
                await self._connect_sse()

            # Initialize the connection
            await self._initialize()

            # Discover tools and prompts
            self._tools = await self._list_tools()
            self._prompts = await self._list_prompts()

            self.status = MCPServerStatus.CONNECTED
            logger.info(
                "MCP server connected",
                server_id=self.config.id,
                server_name=self._server_name,
                tools=len(self._tools),
                prompts=len(self._prompts),
            )

        except Exception as e:
            self.status = MCPServerStatus.ERROR
            logger.error(
                "Failed to connect to MCP server",
                server_id=self.config.id,
                error=str(e),
            )
            raise

    async def _connect_stdio(self) -> None:
        """Start MCP server subprocess and connect via stdio."""
        cmd = self.config.command
        args = list(self.config.args)

        env = None

        # For Docker commands, ensure -T (no TTY) and inject -e flags for env vars
        if cmd in ("docker", "docker-compose") and "run" in args:
            run_idx = args.index("run")
            insert_at = run_idx + 1

            # Skip past existing flags like --rm, -T that are already there
            while insert_at < len(args) and args[insert_at].startswith("-"):
                if args[insert_at] in ("-e",) and insert_at + 1 < len(args):
                    insert_at += 2  # skip -e and its value
                else:
                    insert_at += 1

            # Ensure -T is present (disable TTY allocation for stdio JSON-RPC)
            if "-T" not in args:
                args.insert(run_idx + 1, "-T")
                insert_at += 1

            # Inject -e flags so env vars reach the container
            if self.config.env:
                for key, value in self.config.env.items():
                    args.insert(insert_at, "-e")
                    args.insert(insert_at + 1, f"{key}={value}")
                    insert_at += 2

        if self.config.env:
            import os

            env = {**os.environ, **self.config.env}

        # Use working_dir as cwd if specified
        cwd = self.config.working_dir if self.config.working_dir else None

        logger.debug(
            "Starting MCP process", command=cmd, args=args, cwd=cwd,
        )

        self._process = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )

        # Start background stderr drainer to prevent pipe buffer deadlock.
        # Without this, if the subprocess writes >64KB to stderr (common with
        # Docker build/migration output), it blocks and never reaches the MCP
        # server connection phase.
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """Continuously read stderr to prevent pipe buffer deadlock."""
        if not self._process or not self._process.stderr:
            return
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    # Keep last 50 lines for error context
                    self._last_stderr_lines.append(text)
                    if len(self._last_stderr_lines) > 50:
                        self._last_stderr_lines.pop(0)
                    logger.info("MCP stderr", server_id=self.config.id, line=text[:500])
        except Exception:
            pass

    async def _connect_sse(self) -> None:
        """Connect to MCP server via SSE/HTTP transport (classic SSE pattern).

        Classic SSE MCP transport:
          1. GET <url>  Accept: text/event-stream  →  server pushes SSE events
          2. First SSE event is  event: endpoint / data: /mcp/message?sessionId=xxx
          3. Client POSTs JSON-RPC to that endpoint; responses arrive via SSE stream.
        """
        self._sse_connected = asyncio.Event()
        self._sse_pending = {}
        self._sse_endpoint = None
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=10.0),
        )
        self._sse_task = asyncio.create_task(self._run_sse_reader())
        # Wait until the server sends the endpoint event (or fails)
        try:
            await asyncio.wait_for(self._sse_connected.wait(), timeout=15.0)
        except TimeoutError as e:
            raise RuntimeError("Timed out waiting for SSE endpoint event") from e
        if not self._sse_endpoint:
            raise RuntimeError("SSE connection failed — no endpoint received")

    async def _run_sse_reader(self) -> None:
        """Background task: reads the SSE stream and routes responses to waiting futures."""
        try:
            async with self._http_client.stream(
                "GET",
                self.config.url,
                headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
            ) as response:
                response.raise_for_status()
                event_type: str | None = None
                data_lines: list[str] = []
                async for raw_line in response.aiter_lines():
                    line = raw_line.rstrip("\r")
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif line == "":
                        data = "\n".join(data_lines)
                        data_lines = []
                        if event_type == "endpoint":
                            self._sse_endpoint = data.strip()
                            self._sse_connected.set()
                        elif event_type == "message" and data:
                            try:
                                msg = json.loads(data)
                                req_id = msg.get("id")
                                if req_id is not None and req_id in self._sse_pending:
                                    fut = self._sse_pending[req_id]
                                    if not fut.done():
                                        fut.set_result(msg)
                            except json.JSONDecodeError:
                                pass
                        event_type = None
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("SSE reader terminated", server_id=self.config.id, error=str(e))
            # Unblock connect() if it's still waiting
            if not self._sse_connected.is_set():
                self._sse_connected.set()
            # Fail any in-flight requests
            for fut in self._sse_pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError(f"SSE stream closed: {e}"))

    async def disconnect(self) -> None:
        """Close the connection to the MCP server."""
        logger.info("Disconnecting from MCP server", server_id=self.config.id)

        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            self._stderr_task = None

        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            self._sse_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (TimeoutError, ProcessLookupError):
                self._process.kill()
            self._process = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._sse_endpoint = None
        self._sse_pending = {}
        self.status = MCPServerStatus.DISCONNECTED
        self._tools = []
        self._prompts = []

    async def _send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and return the result."""
        request_id = self._next_id()
        message: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            message["params"] = params

        if self.config.transport == MCPTransportType.STDIO:
            return await self._send_stdio(message)
        elif self.config.transport == MCPTransportType.STREAMABLE_HTTP:
            return await self._send_streamable_http(message)
        else:
            return await self._send_sse(message)

    async def _send_stdio(self, message: dict[str, Any]) -> Any:
        """Send request via stdio transport."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            raise ConnectionError("MCP process not running")

        payload = json.dumps(message)
        # MCP SDK 1.x uses newline-delimited JSON over stdio
        wire = payload + "\n"

        async with self._write_lock:
            self._process.stdin.write(wire.encode())
            await self._process.stdin.drain()

        async with self._read_lock:
            response = await self._read_stdio_response()

        if "error" in response:
            error = response["error"]
            raise RuntimeError(
                f"MCP error ({error.get('code', 'unknown')}): {error.get('message', 'Unknown error')}"
            )

        return response.get("result")

    async def _read_stdio_response(self) -> dict[str, Any]:
        """Read a JSON-RPC response from stdio using newline-delimited JSON.

        MCP SDK 1.x uses newline-delimited JSON: each message is a single
        JSON object followed by a newline character.
        """
        if not self._process or not self._process.stdout:
            raise ConnectionError("MCP process not running")

        # Read lines until we get a valid JSON-RPC response.
        # Skip any non-JSON output (e.g., Docker startup messages, blank lines).
        # Use a longer timeout for the first read (container startup can be slow).
        is_first_read = True
        while True:
            timeout = 180.0 if is_first_read else 60.0
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=timeout
                )
            except TimeoutError:
                # Use captured stderr lines for context
                stderr_context = "\n".join(self._last_stderr_lines[-10:])
                detail = f"Timed out waiting for MCP server response ({timeout:.0f}s)"
                if stderr_context:
                    detail += f"\nRecent stderr:\n{stderr_context}"
                raise TimeoutError(detail) from None
            is_first_read = False
            if not line:
                # Process exited — use captured stderr for error context
                stderr_context = "\n".join(self._last_stderr_lines[-10:])
                detail = stderr_context or "MCP process closed stdout unexpectedly"
                raise ConnectionError(detail)

            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            # Try to parse as JSON
            try:
                response = json.loads(line_str)
            except json.JSONDecodeError:
                # Not JSON — skip (Docker output, log lines, etc.)
                logger.debug("Skipping non-JSON stdio output", line=line_str[:200])
                continue

            # Skip notifications (no "id" field) and read again
            if "id" not in response:
                continue

            return response

    async def _send_sse(self, message: dict[str, Any]) -> Any:
        """Send a JSON-RPC request via SSE transport and wait for the response on the SSE stream."""
        if not self._http_client or not self._sse_endpoint:
            raise ConnectionError("SSE connection not established")

        req_id: int = message["id"]
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._sse_pending[req_id] = future

        try:
            # Resolve the message endpoint the server advertised, pinned to
            # its own origin (#135). The server picks the path; it does not
            # get to pick the host, or it could redirect our traffic — and
            # whatever the agent has in flight — somewhere else.
            from data_concierge.mcp.guards import pin_sse_endpoint

            endpoint = pin_sse_endpoint(str(self.config.url), self._sse_endpoint)

            post_resp = await self._http_client.post(
                endpoint,
                json=message,
                headers={"Content-Type": "application/json"},
                timeout=30.0,
            )
            post_resp.raise_for_status()
            # Server returns 202 Accepted — actual result arrives via SSE stream

            result_msg = await asyncio.wait_for(future, timeout=60.0)
        finally:
            self._sse_pending.pop(req_id, None)

        if "error" in result_msg:
            error = result_msg["error"]
            raise RuntimeError(
                f"MCP error ({error.get('code', 'unknown')}): {error.get('message', 'Unknown error')}"
            )

        return result_msg.get("result")

    async def _connect_streamable_http(self) -> None:
        """Connect via the modern Streamable HTTP transport.

        Unlike classic SSE there is no endpoint handshake: each JSON-RPC
        message is POSTed to ``self.config.url`` and the response comes back in
        the HTTP body (plain JSON, or a single SSE ``message`` event). Stateless
        servers issue no session id; if a server returns ``Mcp-Session-Id`` we
        echo it on subsequent requests.
        """
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._mcp_session_id = None

    async def _send_streamable_http(self, message: dict[str, Any]) -> Any:
        """Send a JSON-RPC message over Streamable HTTP and return the result."""
        if not self._http_client:
            raise ConnectionError("HTTP connection not established")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._mcp_session_id:
            headers["Mcp-Session-Id"] = self._mcp_session_id

        resp = await self._http_client.post(
            str(self.config.url), json=message, headers=headers, timeout=60.0
        )

        # Capture a server-issued session id for subsequent requests.
        session_id = resp.headers.get("mcp-session-id")
        if session_id:
            self._mcp_session_id = session_id

        # Notifications (no "id") get 202 Accepted with an empty body.
        if resp.status_code == 202 or not resp.content:
            return None
        resp.raise_for_status()

        data = self._parse_http_body(resp.headers.get("content-type", ""), resp.text)
        if data is None:
            return None
        if "error" in data:
            error = data["error"]
            raise RuntimeError(
                f"MCP error ({error.get('code', 'unknown')}): {error.get('message', 'Unknown error')}"
            )
        return data.get("result")

    @staticmethod
    def _parse_http_body(content_type: str, text: str) -> dict[str, Any] | None:
        """Parse a Streamable HTTP body (plain JSON or a single SSE ``message`` event)."""
        if "text/event-stream" in content_type:
            # Streamed response: pull the JSON-RPC object out of the data: lines.
            result: dict[str, Any] | None = None
            for raw in text.splitlines():
                line = raw.strip()
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    try:
                        msg = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and ("result" in msg or "error" in msg):
                        result = msg
            return result
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    async def _initialize(self) -> None:
        """Send initialize request to the MCP server."""
        result = await self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": False},
                },
                "clientInfo": {
                    "name": "data-concierge",
                    "version": "0.1.0",
                },
            },
        )

        server_info = result.get("serverInfo", {})
        self._server_name = server_info.get("name")
        self._server_version = server_info.get("version")

        # Send initialized notification
        notification: dict[str, Any] = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        if self.config.transport == MCPTransportType.STDIO:
            payload = json.dumps(notification) + "\n"
            if self._process and self._process.stdin:
                async with self._write_lock:
                    self._process.stdin.write(payload.encode())
                    await self._process.stdin.drain()
        elif self.config.transport == MCPTransportType.STREAMABLE_HTTP:
            # Streamable HTTP: POST the notification (server replies 202, no body)
            if self._http_client:
                try:
                    await self._send_streamable_http(notification)
                except Exception:
                    pass  # Notification failure is non-fatal
        else:
            # SSE: fire-and-forget POST (no id, no response expected)
            if self._http_client and self._sse_endpoint:
                from data_concierge.mcp.guards import pin_sse_endpoint

                endpoint = pin_sse_endpoint(str(self.config.url), self._sse_endpoint)
                try:
                    await self._http_client.post(
                        endpoint,
                        json=notification,
                        headers={"Content-Type": "application/json"},
                        timeout=10.0,
                    )
                except Exception:
                    pass  # Notification failure is non-fatal

    async def _list_tools(self) -> list[MCPToolInfo]:
        """List available tools from the MCP server."""
        try:
            result = await self._send_request("tools/list", {})
            tools = []
            for tool_data in result.get("tools", []):
                tools.append(
                    MCPToolInfo(
                        name=tool_data.get("name", ""),
                        description=tool_data.get("description", ""),
                        input_schema=tool_data.get("inputSchema", {}),
                    )
                )
            return tools
        except Exception as e:
            logger.warning("Failed to list tools", server_id=self.config.id, error=str(e))
            return []

    async def _list_prompts(self) -> list[MCPPromptInfo]:
        """List available prompts from the MCP server."""
        try:
            result = await self._send_request("prompts/list", {})
            prompts = []
            for prompt_data in result.get("prompts", []):
                prompts.append(
                    MCPPromptInfo(
                        name=prompt_data.get("name", ""),
                        description=prompt_data.get("description", ""),
                        arguments=prompt_data.get("arguments", []),
                    )
                )
            return prompts
        except Exception as e:
            logger.warning("Failed to list prompts", server_id=self.config.id, error=str(e))
            return []

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolCallResult:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool call result
        """
        logger.info(
            "Calling MCP tool",
            server_id=self.config.id,
            tool=tool_name,
        )

        result = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

        return MCPToolCallResult(
            content=result.get("content", []),
            is_error=result.get("isError", False),
        )

    async def get_prompt(
        self, prompt_name: str, arguments: dict[str, str] | None = None
    ) -> dict[str, Any]:
        """Get a prompt from the MCP server.

        Args:
            prompt_name: Name of the prompt
            arguments: Prompt arguments

        Returns:
            Prompt messages
        """
        params: dict[str, Any] = {"name": prompt_name}
        if arguments:
            params["arguments"] = arguments

        result = await self._send_request("prompts/get", params)
        return result
