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

    # 3. Test MCP Domain Classifier & Dynamic Tool Pruning
    print("\n[3] Testing Prompt Intent & Dynamic Tool Pruning:")
    
    # Simulate realistic Open Interpreter tool payload (MCP tools + core agent tools)
    sample_agent_tools = [
        {"type": "function", "function": {"name": "exec_command", "description": "Execute shell command"}},
        {"type": "function", "function": {"name": "write_stdin", "description": "Write to stdin"}},
    ]
    full_tool_catalog = sample_agent_tools + all_tools

    test_cases = [
        (
            "Search the web for latest AMD Ryzen AI Linux drivers",
            ["web_search"],
            {"searxng_search", "exec_command", "write_stdin"},
            "browser_navigate"
        ),
        (
            "Write a Python script to compute Fibonacci numbers with memoization",
            [],
            {"exec_command", "write_stdin"},
            "searxng_search"
        ),
        (
            "Use browser_navigate to visit http://example.com and click the submit button",
            ["browser"],
            {"browser_navigate", "browser_click", "exec_command", "write_stdin"},
            "searxng_search"
        ),
    ]

    for prompt, expected_domains, must_include, must_not_include in test_cases:
        matched_domains = mcp_classifier.classify_domains(prompt)
        filtered_tools, total, kept, pruned = mcp_classifier.filter_tools(
            incoming_tools=full_tool_catalog,
            active_domains=matched_domains,
            prompt=prompt
        )
        tool_names = {t["function"]["name"] for t in filtered_tools}
        
        has_required = must_include.issubset(tool_names)
        excludes_unwanted = must_not_include not in tool_names
        
        print(f"\n    • Prompt: '{prompt[:60]}...'")
        print(f"      Matched Domains: {matched_domains} (Expected: {expected_domains})")
        print(f"      Tools: Kept {len(filtered_tools)} / Pruned {pruned} from {total}")
        print(f"      Required Tools Included: {has_required} [MATCH]")
        print(f"      Excluded Unwanted Tools: {excludes_unwanted} [MATCH]")
        assert has_required, f"Missing required tools in: {tool_names}"
        assert excludes_unwanted, f"Unwanted tool '{must_not_include}' present in: {tool_names}"

    # Test Multi-Turn Continuity
    print("\n[3.1] Testing Multi-Turn Conversation Continuity:")
    conversation = [
        {"role": "user", "content": "Navigate to https://example.com"},
        {"role": "assistant", "content": "Navigating now.", "tool_calls": [{"function": {"name": "browser_navigate"}}]},
        {"role": "user", "content": "What is the heading text?"} # Short prompt without explicit keyword
    ]
    conv_domains = mcp_classifier.classify_conversation(conversation)
    print(f"    • Conversation Active Domains: {conv_domains}")
    assert "browser" in conv_domains, f"Expected 'browser' domain in multi-turn conversation, got: {conv_domains}"
    print("    • Multi-turn tool continuity preserved: [MATCH]")

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
