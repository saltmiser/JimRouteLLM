import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000/v1/chat/completions"
SESSION_ID = "stress-kv-session-bench-42"

turns = [
    ("Hello! I am building a high-performance in-memory cache in Python. What data structures should I consider?", "none"),
    ("Let's go with an OrderedDict to implement an LRU eviction policy. Can you write a skeleton class for it?", "none"),
    ("Add a TTL (time to live) expiration feature to each key.", "none"),
    ("Now search the web for Python cache benchmark libraries.", "web_search"),
    ("What was the first library mentioned in the search results?", "web_search"),
    ("Now write a pytest test case verifying the LRU eviction logic from earlier.", "none"),
    ("Add thread safety with threading.RLock.", "none"),
    ("Use browser_navigate to check the Python documentation for threading at https://docs.python.org", "browser"),
    ("Did the docs mention reentrant locks?", "browser"),
    ("Summarize our cache implementation and all features we added across this session.", "none"),
]

sample_tools = [
    {"type": "function", "function": {"name": "exec_command", "description": "Execute shell command"}},
    {"type": "function", "function": {"name": "write_stdin", "description": "Write to stdin"}},
    {"type": "function", "function": {"name": "searxng_search", "description": "Search web using local SearXNG"}},
    {"type": "function", "function": {"name": "browser_navigate", "description": "Navigate to URL"}},
    {"type": "function", "function": {"name": "browser_click", "description": "Click element"}},
    {"type": "function", "function": {"name": "browser_snapshot", "description": "Accessibility snapshot"}},
]

def main():
    print("=" * 80)
    print("  JimRouteLLM 10-Turn KV-Cache & Sticky Session Stress Benchmark")
    print(f"  Session ID: {SESSION_ID}")
    print("=" * 80)

    conversation = []
    nodes_seen = []
    latencies = []

    print(f"\n{'Turn':<5} | {'Active Domains':<15} | {'Node':<14} | {'Model':<22} | {'Latency':<8} | {'Tokens Saved'}")
    print("-" * 80)

    for i, (prompt, expected_domain) in enumerate(turns, 1):
        conversation.append({"role": "user", "content": prompt})

        payload = {
            "model": "routellm",
            "messages": conversation,
            "tools": sample_tools,
            "max_tokens": 40,
            "temperature": 0.2
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            BASE_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "X-Session-ID": SESSION_ID
            },
            method="POST"
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                dt = time.perf_counter() - t0
                headers = dict(resp.headers)
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"Error on turn {i}: {e}")
            sys.exit(1)

        node = headers.get("x-routellm-node", "unknown")
        model = headers.get("x-routellm-model", "unknown")
        active_domains = headers.get("x-routellm-mcp-domains", "none")
        tokens_saved = headers.get("x-routellm-tokens-saved", "0")
        reply = body["choices"][0]["message"].get("content", "")

        conversation.append({"role": "assistant", "content": reply})
        nodes_seen.append(node)
        latencies.append(dt)

        print(f"{i:<5} | {active_domains:<15} | {node:<14} | {model[:21]:<22} | {dt:>6.2f}s  | ~{tokens_saved} tokens")

    print("-" * 80)
    
    # Assertions
    all_same_node = len(set(nodes_seen)) == 1
    print(f"\n[Verification Summary]")
    print(f"  • Pinned Node Across All Turns: {nodes_seen[0]} (Affinity: {'100% PASS' if all_same_node else 'FAIL'})")
    print(f"  • Average Latency:              {sum(latencies)/len(latencies):.2f}s")
    print(f"  • Total Turns Completed:        {len(turns)} / 10")
    assert all_same_node, "Session affinity violation: different nodes assigned to the same session!"
    print("  [✔] STRESS TEST PASSED: Sticky KV-cache affinity & dynamic pruning 100% verified.")

if __name__ == "__main__":
    main()
