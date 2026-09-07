"""
JimRouteLLM Unified MCP Gateway
================================
Aggregates all custom MCP servers (SearXNG, LAN Docker containers, and built-in cluster tools)
into a single unified MCP interface. Exposes tools to agent clients (Open Interpreter, Codex CLI)
via both stdio and HTTP/SSE transports.
"""

import sys
import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from jimroutellm_proxy.config import settings
from jimroutellm_proxy.mcp_registry import mcp_registry
from jimroutellm_proxy.nodes import node_manager

logger = logging.getLogger("jimroutellm.mcp_gateway")

# Built-in custom tools exposed by the Gateway
BUILTIN_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "cluster_node_status",
        "description": "Inspect the real-time operational health, active models, hardware roles, and connectivity of all 4 LAN inference nodes and the Google Gemini cloud tier in the JimRouteLLM cluster.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "verbose": {
                    "type": "boolean",
                    "description": "Whether to return detailed capability lists for each node",
                    "default": False
                }
            }
        }
    }
]


async def handle_cluster_node_status(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Return live status of the LAN inference nodes and router configuration."""
    verbose = arguments.get("verbose", False)
    nodes_info = []
    for node in node_manager.get_all_nodes():
        item = {
            "id": node.id,
            "name": node.name,
            "base_url": node.base_url,
            "primary_model": node.primary_model,
            "active_models": node.active_models,
            "is_online": node.is_online,
        }
        if verbose:
            item["capabilities"] = list(node.capabilities)
            item["priority"] = node.priority
        nodes_info.append(item)

    report = {
        "gateway": "JimRouteLLM Unified MCP Gateway",
        "router_threshold": settings.routing_threshold,
        "local_easy_model": settings.local_easy_model,
        "local_hard_model": settings.local_hard_model,
        "cloud_provider": "Google AI Studio (Gemini)" if settings.is_gemini_configured() else "None",
        "lan_nodes": nodes_info,
        "active_mcp_servers": list(mcp_registry.server_configs.keys()),
    }
    return report


async def get_all_gateway_tools() -> List[Dict[str, Any]]:
    """Collect all tools: built-in custom tools + all tools from synced MCP servers."""
    if not mcp_registry.is_initialized:
        mcp_registry.load_config()
        await mcp_registry.sync_all_servers()

    tools: List[Dict[str, Any]] = list(BUILTIN_TOOLS)
    seen_names = {t["name"] for t in tools}

    for srv_name, raw_tools in mcp_registry.cached_tools.items():
        for t in raw_tools:
            t_name = t.get("name")
            if t_name and t_name not in seen_names:
                seen_names.add(t_name)
                tools.append({
                    "name": t_name,
                    "description": t.get("description", f"Tool from {srv_name} MCP server"),
                    "inputSchema": t.get("inputSchema", {"type": "object", "properties": {}}),
                })

    return tools


async def dispatch_mcp_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool call execution to built-in handlers or external MCP server processes."""
    if tool_name == "cluster_node_status":
        return await handle_cluster_node_status(arguments)

    # Check if tool is managed by mcp_registry
    if mcp_registry.get_server_for_tool(tool_name):
        res = await mcp_registry.execute_tool(tool_name, arguments)
        # Extract content if standard MCP structure
        if isinstance(res, dict) and "result" in res:
            res = res["result"]
        return res

    raise ValueError(f"Unknown tool: '{tool_name}' in JimRouteLLM MCP Gateway")


async def process_jsonrpc_request(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a standard JSON-RPC 2.0 MCP request and return the response."""
    req_id = req.get("id")
    method = req.get("method")

    if method == "notifications/initialized":
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False}
                },
                "serverInfo": {
                    "name": "jimroutellm-mcp-gateway",
                    "version": "1.0.0"
                }
            }
        }

    if method == "tools/list":
        tools = await get_all_gateway_tools()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": tools
            }
        }

    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            output = await dispatch_mcp_call(tool_name, arguments)
            if isinstance(output, (dict, list)):
                text_content = json.dumps(output, indent=2)
            else:
                text_content = str(output)

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": text_content}
                    ]
                }
            }
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "isError": True,
                "result": {
                    "content": [
                        {"type": "text", "text": f"Error executing {tool_name}: {str(e)}"}
                    ]
                }
            }

    if method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {}
        }

    # Unknown method fallback
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }

    return None


async def run_stdio_server():
    """Run MCP Gateway over stdio for CLI clients."""
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    # Initial sync in background
    mcp_registry.load_config()
    await mcp_registry.sync_all_servers()

    while True:
        line = await reader.readline()
        if not line:
            break
        raw = line.decode("utf-8").strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            resp = await process_jsonrpc_request(req)
            if resp:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"JimRouteLLM MCP Gateway error: {e}\n")
            sys.stderr.flush()


def main():
    try:
        asyncio.run(run_stdio_server())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
