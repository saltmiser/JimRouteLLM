"""
RouteLLM MCP Registry
=====================
Manages connections to LAN Docker MCP servers and local stdio MCP processes,
caches tool definitions, and maps tools to semantic domain clusters.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
import httpx

logger = logging.getLogger("routellm.mcp_registry")


# Domain to MCP Server mapping
DOMAIN_TO_SERVERS: Dict[str, List[str]] = {
    "shipping": ["shipstation-v1", "shipstation-v2"],
    "ecommerce": ["shopify", "recharge-v1w-mcp"],
    "roasting_inventory": ["cropster-v1-mcp", "listingmirror-v3"],
    "digital_assets": ["canto-v1-mcp"],
    "web_search": ["searxng-mcp", "searxng", "searxng-server"],
    "browser": ["playwright", "playwright-mcp", "browser"],
}

SERVER_TO_DOMAIN: Dict[str, str] = {}
for domain, srvs in DOMAIN_TO_SERVERS.items():
    for srv in srvs:
        SERVER_TO_DOMAIN[srv] = domain


def mcp_tool_to_openai(tool: Dict[str, Any], server_name: str = "") -> Dict[str, Any]:
    """Convert MCP tool schema (name, description, inputSchema) to OpenAI function format."""
    name = tool.get("name", "")
    description = tool.get("description", "")
    input_schema = tool.get("inputSchema", {}) or {"type": "object", "properties": {}}
    
    # Ensure properties is present
    if "properties" not in input_schema and input_schema.get("type") == "object":
        input_schema["properties"] = {}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
        "_server": server_name,
        "_domain": SERVER_TO_DOMAIN.get(server_name, "custom"),
    }


def clean_openai_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Strip internal routing metadata before sending tool schema to LiteLLM / model."""
    if "function" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["function"].get("name", ""),
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {"type": "object", "properties": {}}),
            }
        }
    return tool


class MCPRegistry:
    """Manages discovery, schema caching, and lifecycle for all MCP servers."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcpServers.json")
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self.cached_tools: Dict[str, List[Dict[str, Any]]] = {}  # server_name -> raw MCP tools
        self.openai_tools: Dict[str, List[Dict[str, Any]]] = {}  # server_name -> OpenAI format tools
        self.tool_to_server: Dict[str, str] = {}  # tool_name -> server_name
        self.tool_to_domain: Dict[str, str] = {}  # tool_name -> domain
        self.last_sync_time: float = 0.0
        self.sync_lock = asyncio.Lock()
        self.is_initialized: bool = False
        self.sync_errors: Dict[str, str] = {}

    def load_config(self) -> bool:
        """Load server definitions from mcpServers.json or Cline/Claude fallback configs."""
        candidate_paths = [
            self.config_path,
            os.path.expanduser("~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"),
            os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json"),
        ]
        
        found_path = None
        for path in candidate_paths:
            if path and os.path.exists(path):
                found_path = path
                break

        if not found_path:
            logger.warning(f"MCP configuration file not found in candidate paths: {candidate_paths}")
            return False

        try:
            with open(found_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.server_configs = data.get("mcpServers", {})
            logger.info(f"Loaded {len(self.server_configs)} MCP server definitions from {found_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {found_path}: {e}")
            return False

    async def _fetch_http_tools(self, server_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Connect to an HTTP/SSE MCP server, perform initialize handshake, and fetch tools/list."""
        url = config.get("url", "")
        headers_cfg = config.get("headers", {})
        
        req_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        req_headers.update(headers_cfg)

        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1. Initialize
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "routellm-proxy", "version": "1.0"},
                },
            }
            init_resp = await client.post(url, headers=req_headers, json=init_payload)
            if init_resp.status_code not in (200, 201):
                raise RuntimeError(f"Initialize failed with status {init_resp.status_code}: {init_resp.text[:120]}")
            
            session_id = init_resp.headers.get("mcp-session-id")
            tools_headers = dict(req_headers)
            if session_id:
                tools_headers["mcp-session-id"] = session_id

            # 2. tools/list
            tools_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }
            tools_resp = await client.post(url, headers=tools_headers, json=tools_payload)
            if tools_resp.status_code not in (200, 201):
                raise RuntimeError(f"tools/list failed with status {tools_resp.status_code}: {tools_resp.text[:120]}")

            # Parse SSE or raw JSON response
            raw_text = tools_resp.text
            tools: List[Dict[str, Any]] = []
            
            # Check for SSE format
            for line in raw_text.splitlines():
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                        tools = payload.get("result", {}).get("tools", [])
                        break
                    except Exception:
                        continue
            
            # Fallback to direct JSON body if not SSE formatted
            if not tools:
                try:
                    payload = json.loads(raw_text)
                    tools = payload.get("result", {}).get("tools", [])
                except Exception:
                    pass

            return tools

    async def _fetch_stdio_tools(self, server_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Start a local stdio MCP subprocess, run initialize handshake, and fetch tools/list."""
        command = config.get("command")
        args = config.get("args", [])
        env_vars = config.get("env", {})

        full_env = os.environ.copy()
        full_env.update(env_vars)

        cmd = [command] + args
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )

        try:
            # 1. Initialize
            init_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "routellm-proxy", "version": "1.0"},
                },
            }) + "\n"
            proc.stdin.write(init_req.encode("utf-8"))
            await proc.stdin.drain()

            # Read initialize response
            init_line = await asyncio.wait_for(proc.stdout.readline(), timeout=15.0)
            if not init_line:
                raise RuntimeError("Subprocess closed stdout during initialize")

            # Send notifications/initialized per MCP specification
            notif = json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }) + "\n"
            proc.stdin.write(notif.encode("utf-8"))
            await proc.stdin.drain()

            # 2. tools/list
            tools_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {},
            }) + "\n"
            proc.stdin.write(tools_req.encode("utf-8"))
            await proc.stdin.drain()

            tools_line = await asyncio.wait_for(proc.stdout.readline(), timeout=15.0)
            if not tools_line:
                raise RuntimeError("Subprocess closed stdout during tools/list")

            payload = json.loads(tools_line.decode("utf-8"))
            tools = payload.get("result", {}).get("tools", [])
            return tools

        finally:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def _execute_http_tool(self, server_name: str, config: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on an HTTP/SSE MCP server."""
        url = config.get("url", "")
        headers_cfg = config.get("headers", {})
        req_headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        req_headers.update(headers_cfg)

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Initialize
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "routellm-proxy", "version": "1.0"},
                },
            }
            init_resp = await client.post(url, headers=req_headers, json=init_payload)
            session_id = init_resp.headers.get("mcp-session-id")
            call_headers = dict(req_headers)
            if session_id:
                call_headers["mcp-session-id"] = session_id

            # 2. tools/call
            call_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }
            resp = await client.post(url, headers=call_headers, json=call_payload)
            raw_text = resp.text
            for line in raw_text.splitlines():
                if line.startswith("data: "):
                    try:
                        return json.loads(line[6:])
                    except Exception:
                        pass
            try:
                return json.loads(raw_text)
            except Exception:
                return {"result": {"content": [{"type": "text", "text": raw_text}]}}

    async def _execute_stdio_tool(self, server_name: str, config: Dict[str, Any], tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on a local stdio MCP subprocess."""
        command = config.get("command")
        args = config.get("args", [])
        env_vars = config.get("env", {})
        full_env = os.environ.copy()
        full_env.update(env_vars)

        proc = await asyncio.create_subprocess_exec(
            command, *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=full_env,
        )
        try:
            # 1. Initialize
            init_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "routellm-proxy", "version": "1.0"},
                },
            }) + "\n"
            proc.stdin.write(init_req.encode("utf-8"))
            await proc.stdin.drain()
            await asyncio.wait_for(proc.stdout.readline(), timeout=15.0)

            # Send notifications/initialized per MCP specification
            notif = json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }) + "\n"
            proc.stdin.write(notif.encode("utf-8"))
            await proc.stdin.drain()

            # 2. tools/call
            call_req = json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
            }) + "\n"
            proc.stdin.write(call_req.encode("utf-8"))
            await proc.stdin.drain()

            res_line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
            return json.loads(res_line.decode("utf-8"))
        finally:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call against the appropriate MCP server."""
        server_name = self.get_server_for_tool(tool_name)
        if not server_name:
            raise ValueError(f"Unknown tool '{tool_name}' - not found in MCP registry")
        config = self.server_configs.get(server_name)
        if not config:
            raise ValueError(f"No configuration found for server '{server_name}'")

        # Smart normalization for Shopify ID arguments
        if server_name == "shopify":
            for id_key in ("orderId", "id"):
                if id_key in arguments and isinstance(arguments[id_key], (str, int)):
                    val = str(arguments[id_key]).strip()
                    if val and not val.startswith("gid://"):
                        clean_name = val.lstrip("#")
                        try:
                            search_res = await self.execute_tool("get-orders", {"query": f"name:{clean_name}", "limit": 1})
                            content_str = search_res.get("result", {}).get("content", [{}])[0].get("text", "{}")
                            order_data = json.loads(content_str)
                            orders = order_data.get("orders", [])
                            if orders:
                                real_gid = orders[0]["id"]
                                logger.info(f"Resolved Shopify order '{val}' -> '{real_gid}'")
                                arguments[id_key] = real_gid
                        except Exception as ex:
                            logger.warning(f"Could not auto-resolve Shopify order GID for '{val}': {ex}")

        if "url" in config:
            return await self._execute_http_tool(server_name, config, tool_name, arguments)
        elif "command" in config:
            return await self._execute_stdio_tool(server_name, config, tool_name, arguments)
        else:
            raise ValueError(f"Invalid server configuration for '{server_name}'")

    async def sync_server(self, server_name: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Sync tools for a single server configuration."""
        if "url" in config:
            return await self._fetch_http_tools(server_name, config)
        elif "command" in config:
            return await self._fetch_stdio_tools(server_name, config)
        else:
            raise ValueError(f"Server '{server_name}' has neither 'url' nor 'command' specified")

    async def sync_all_servers(self) -> Dict[str, int]:
        """Synchronize tools from all configured MCP servers in parallel."""
        async with self.sync_lock:
            if not self.server_configs:
                self.load_config()

            tasks = []
            server_names = []
            for name, cfg in self.server_configs.items():
                tasks.append(self.sync_server(name, cfg))
                server_names.append(name)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            counts: Dict[str, int] = {}
            new_cached_tools: Dict[str, List[Dict[str, Any]]] = {}
            new_openai_tools: Dict[str, List[Dict[str, Any]]] = {}
            new_tool_to_server: Dict[str, str] = {}
            new_tool_to_domain: Dict[str, str] = {}
            new_errors: Dict[str, str] = {}

            for name, res in zip(server_names, results):
                if isinstance(res, Exception):
                    logger.warning(f"Failed to sync MCP server '{name}': {res}")
                    new_errors[name] = str(res)
                    # Preserve existing cached tools if available
                    if name in self.cached_tools:
                        new_cached_tools[name] = self.cached_tools[name]
                        new_openai_tools[name] = self.openai_tools[name]
                else:
                    new_cached_tools[name] = res
                    openai_list = [mcp_tool_to_openai(t, name) for t in res]
                    new_openai_tools[name] = openai_list
                    counts[name] = len(res)
                    domain = SERVER_TO_DOMAIN.get(name, "custom")
                    for t in res:
                        t_name = t.get("name", "")
                        if t_name:
                            new_tool_to_server[t_name] = name
                            new_tool_to_domain[t_name] = domain
                    logger.info(f"✅ Synced {len(res)} tools from MCP server '{name}'")

            self.cached_tools = new_cached_tools
            self.openai_tools = new_openai_tools
            self.tool_to_server = new_tool_to_server
            self.tool_to_domain = new_tool_to_domain
            self.sync_errors = new_errors
            self.last_sync_time = time.time()
            self.is_initialized = True

            total_tools = sum(len(ts) for ts in self.cached_tools.values())
            logger.info(f"MCP Registry sync complete: {total_tools} tools active across {len(self.cached_tools)} servers")
            return counts

    def get_tools_for_domains(self, domains: List[str]) -> List[Dict[str, Any]]:
        """Return combined, deduplicated OpenAI function tools for active domain keys."""
        matched_tools: List[Dict[str, Any]] = []
        seen_names = set()

        for domain in domains:
            servers = DOMAIN_TO_SERVERS.get(domain, [])
            for srv in servers:
                for tool in self.openai_tools.get(srv, []):
                    tool_name = tool["function"]["name"]
                    if tool_name not in seen_names:
                        seen_names.add(tool_name)
                        matched_tools.append(tool)

        return matched_tools

    def get_all_openai_tools(self) -> List[Dict[str, Any]]:
        """Return all cached tools in OpenAI format across all servers."""
        all_tools: List[Dict[str, Any]] = []
        seen_names = set()
        for srv, tools in self.openai_tools.items():
            for t in tools:
                name = t["function"]["name"]
                if name not in seen_names:
                    seen_names.add(name)
                    all_tools.append(t)
        return all_tools

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Find origin server name for a given tool name."""
        return self.tool_to_server.get(tool_name)

    def get_domain_for_tool(self, tool_name: str) -> Optional[str]:
        """Find semantic domain for a given tool name."""
        return self.tool_to_domain.get(tool_name)

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics and health status for all registered MCP servers."""
        server_stats = {}
        for name in self.server_configs:
            tool_count = len(self.cached_tools.get(name, []))
            status = "online" if name in self.cached_tools and tool_count > 0 else "error"
            server_stats[name] = {
                "status": status,
                "tool_count": tool_count,
                "domain": SERVER_TO_DOMAIN.get(name, "custom"),
                "error": self.sync_errors.get(name),
            }

        domain_counts = {}
        for domain, srvs in DOMAIN_TO_SERVERS.items():
            domain_counts[domain] = sum(len(self.cached_tools.get(s, [])) for s in srvs)

        return {
            "total_servers": len(self.server_configs),
            "online_servers": sum(1 for s in server_stats.values() if s["status"] == "online"),
            "total_tools": sum(len(ts) for ts in self.cached_tools.values()),
            "last_sync_time": self.last_sync_time,
            "domains": domain_counts,
            "servers": server_stats,
        }


# Global singleton instance
mcp_registry = MCPRegistry()
