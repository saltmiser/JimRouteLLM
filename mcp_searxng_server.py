"""
SearXNG MCP Server Bridge
=========================
Exposes local SearXNG instance (http://127.0.0.1:8888) as a standard MCP tool.
"""

import sys
import json
import requests

SEARXNG_URL = "http://127.0.0.1:8888"

TOOLS_SCHEMA = [
    {
        "name": "searxng_search",
        "description": "Perform privacy-respecting multi-engine web search using local SearXNG instance. Returns relevant webpage titles, URLs, and snippet summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default: 5)",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]

def handle_search(query: str, num_results: int = 5) -> dict:
    try:
        resp = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=8.0
        )
        if resp.status_code != 200:
            return {"error": f"SearXNG returned status code {resp.status_code}"}
        
        data = resp.json()
        raw_results = data.get("results", [])[:num_results]
        formatted = [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "content": r.get("content")
            }
            for r in raw_results
        ]
        return {"results": formatted}
    except Exception as e:
        return {"error": f"Failed to query SearXNG: {str(e)}"}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req.get("method")

            if method == "notifications/initialized":
                continue

            if method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "searxng-mcp-server", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS_SCHEMA}}
            elif method == "tools/call":
                params = req.get("params", {})
                tool_name = params.get("name")
                args = params.get("arguments", {})

                if tool_name == "searxng_search":
                    result = handle_search(args.get("query", ""), args.get("num_results", 5))
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(result, indent=2)}]
                        }
                    }
                else:
                    resp = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}}
            else:
                if req_id is not None:
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": {}}
                else:
                    continue

            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stderr.write(f"Error processing MCP message: {e}\n")

if __name__ == "__main__":
    main()
