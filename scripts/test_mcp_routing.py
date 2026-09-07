import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jimroutellm_proxy.config import settings
from jimroutellm_proxy.mcp_registry import mcp_registry
from jimroutellm_proxy.mcp_classifier import mcp_classifier

async def main():
    print("=" * 65)
    print("  JimRouteLLM MCP & SearXNG Live Verification Suite")
    print("=" * 65)

    # 1. Load MCP config
    print("\n[1] Loading MCP Servers Configuration...")
    mcp_registry.load_config()
    print(f"    Configured Servers: {list(mcp_registry.server_configs.keys())}")

    # 2. Sync Schemas via stdio MCP bridge
    print("\n[2] Synchronizing MCP Tool Schemas from Local SearXNG...")
    await mcp_registry.sync_all_servers()
    all_tools = mcp_registry.get_all_openai_tools()
    print(f"    Discovered Tools: {[t['function']['name'] for t in all_tools]}")
    for tool in all_tools:
        fn = tool.get("function", {})
        print(f"    • Tool: {fn.get('name')}")
        print(f"      Desc: {fn.get('description')[:70]}...")

    # 3. Test MCP Domain Classifier
    print("\n[3] Testing Prompt Intent & Dynamic Tool Pruning:")
    test_cases = [
        ("Search the web for latest AMD Ryzen AI Linux drivers", True, "web_search"),
        ("Write a Python script to compute Fibonacci numbers with memoization", False, "none"),
        ("Browse online and look up current news regarding space exploration", True, "web_search"),
    ]

    for prompt, should_include_search, expected_domain in test_cases:
        matched_domains = mcp_classifier.classify_domains(prompt)
        filtered_tools, total, kept, pruned = mcp_classifier.filter_tools(
            incoming_tools=all_tools,
            active_domains=matched_domains,
            prompt=prompt
        )
        has_search_tool = any(t["function"]["name"] == "searxng_search" for t in filtered_tools)
        print(f"\n    • Prompt: '{prompt[:55]}...'")
        print(f"      Matched Domains: {matched_domains}")
        print(f"      Filtered Tools:  {[t['function']['name'] for t in filtered_tools]} (Kept: {kept}, Pruned: {pruned})")
        print(f"      Search Included: {has_search_tool} {'[MATCH]' if has_search_tool == should_include_search else '[FAIL]'}")
        assert has_search_tool == should_include_search, f"Tool filtering mismatch for: {prompt}"

    # 4. Test Live Tool Execution via MCP Bridge
    print("\n[4] Testing Live SearXNG Search Execution via MCP Bridge:")
    from mcp_searxng_server import handle_search
    query = "HP ZBook Power G11 Linux Ubuntu"
    res = handle_search(query, num_results=3)
    results = res.get("results", [])
    print(f"    Query: '{query}'")
    print(f"    Results Found: {len(results)}")
    for idx, r in enumerate(results, 1):
        print(f"    {idx}. {r.get('title')}")
        print(f"       URL: {r.get('url')}")
        print(f"       Snippet: {r.get('content')[:80]}...")

    print("\n" + "=" * 65)
    print("  [✔] ALL MCP & SearXNG ROUTING TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    asyncio.run(main())
