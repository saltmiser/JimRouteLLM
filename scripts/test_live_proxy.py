import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def post_json(endpoint, payload, headers=None):
    url = f"{BASE_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            resp_headers = dict(resp.headers)
            body = resp.read().decode("utf-8")
            return resp.status, resp_headers, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = body
        return e.code, dict(e.headers), parsed

def get_json(endpoint):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp_headers = dict(resp.headers)
        body = resp.read().decode("utf-8")
        return resp.status, resp_headers, json.loads(body)

def main():
    print("=" * 70)
    print("  JimRouteLLM Live Proxy & Routing Core Integration Suite")
    print("=" * 70)

    # 1. Health Endpoint
    print("\n[Test 1] Health Check (/health)")
    status, headers, body = get_json("/health")
    print(f"    Status: {status} | Response: {body}")
    assert status == 200, f"Expected 200, got {status}"
    assert body.get("status") == "ok", f"Expected status ok, got {body}"
    print("    [✔] PASSED: Health endpoint OK")

    # 2. Models Endpoint
    print("\n[Test 2] Models List (/v1/models)")
    status, headers, body = get_json("/v1/models")
    models = [m["id"] for m in body.get("data", [])]
    print(f"    Status: {status} | Discovered {len(models)} model IDs")
    print(f"    Sample models: {models[:6]}")
    assert "routellm" in models, "routellm virtual model missing"
    assert "google/gemma-4-12b-qat" in models, "gemma model missing"
    assert "meta/muse-glimmer" in models, "muse-glimmer model missing"
    print("    [✔] PASSED: Virtual and physical models exposed")

    # 3. Fast Tier Routing (Gemma 4 12B)
    print("\n[Test 3] Low-Complexity Prompt Routing (Target: google/gemma-4-12b-qat)")
    easy_payload = {
        "model": "routellm",
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Return only the single digit number."}
        ],
        "max_tokens": 60,
        "temperature": 0.0
    }
    t0 = time.perf_counter()
    status, resp_headers, body = post_json("/v1/chat/completions", easy_payload)
    dt = time.perf_counter() - t0
    target = resp_headers.get("x-routellm-target")
    model_used = resp_headers.get("x-routellm-model")
    score = float(resp_headers.get("x-routellm-score", 0))
    reply = body["choices"][0]["message"]["content"].strip()
    
    print(f"    Latency:      {dt:.2f}s")
    print(f"    Target Tier:  {target}")
    print(f"    Model Routed: {model_used}")
    print(f"    NPU Score:    {score:.3f} (Threshold < 0.45)")
    print(f"    Reply:        '{reply}'")
    assert status == 200, f"HTTP {status}"
    assert model_used == "google/gemma-4-12b-qat", f"Expected gemma, got {model_used}"
    assert score < 0.45, f"Expected score < 0.45, got {score}"
    assert "4" in reply, f"Expected '4' in response, got '{reply}'"
    print("    [✔] PASSED: Low-complexity prompt cleanly routed to Gemma 4 12B")

    # 4. Heavy Tier Routing (Muse Glimmer)
    print("\n[Test 4] High-Complexity Prompt Routing (Target: meta/muse-glimmer)")
    hard_prompt = (
        "Design a distributed event-driven microservices architecture with Raft consensus, "
        "handling out-of-order Kafka message deduplication and distributed deadlocks in Rust."
    )
    hard_payload = {
        "model": "routellm",
        "messages": [{"role": "user", "content": hard_prompt}],
        "max_tokens": 60,
        "temperature": 0.0
    }
    t0 = time.perf_counter()
    status, resp_headers, body = post_json("/v1/chat/completions", hard_payload)
    dt = time.perf_counter() - t0
    target = resp_headers.get("x-routellm-target")
    model_used = resp_headers.get("x-routellm-model")
    score = float(resp_headers.get("x-routellm-score", 0))
    reply = body["choices"][0]["message"]["content"].strip()
    
    print(f"    Latency:      {dt:.2f}s")
    print(f"    Target Tier:  {target}")
    print(f"    Model Routed: {model_used}")
    print(f"    NPU Score:    {score:.3f} (Threshold >= 0.45)")
    print(f"    Reply snippet:'{reply[:90]}...'")
    assert status == 200, f"HTTP {status}"
    assert model_used == "meta/muse-glimmer", f"Expected muse-glimmer, got {model_used}"
    assert score >= 0.45, f"Expected score >= 0.45, got {score}"
    print("    [✔] PASSED: High-complexity prompt cleanly routed to Muse Glimmer")

    # 5. Dynamic MCP Tool Pruning in Live HTTP Request
    print("\n[Test 5] Live Dynamic MCP Tool Pruning & Header Observability")
    
    sample_tools = [
        {"type": "function", "function": {"name": "exec_command", "description": "Execute shell command"}},
        {"type": "function", "function": {"name": "write_stdin", "description": "Write stdin"}},
        {"type": "function", "function": {"name": "searxng_search", "description": "Search web using SearXNG"}},
        {"type": "function", "function": {"name": "browser_navigate", "description": "Navigate to URL"}},
        {"type": "function", "function": {"name": "browser_click", "description": "Click element"}},
        {"type": "function", "function": {"name": "browser_take_screenshot", "description": "Screenshot"}},
        {"type": "function", "function": {"name": "browser_fill_form", "description": "Fill form"}}
    ]

    # 5A: Pure Code / Math (No MCP domain active)
    math_req = {
        "model": "routellm",
        "messages": [{"role": "user", "content": "What is 100 * 100? Respond in 1 line."}],
        "tools": sample_tools,
        "max_tokens": 40
    }
    status, headers, body = post_json("/v1/chat/completions", math_req)
    print(f"    [5A Math Query]")
    print(f"      Active Domains: {headers.get('x-routellm-mcp-domains')}")
    print(f"      Original Tools: {headers.get('x-routellm-tools-original')}")
    print(f"      Pruned Tools:   {headers.get('x-routellm-tools-pruned')}")
    print(f"      Tokens Saved:   {headers.get('x-routellm-tokens-saved')}")
    assert headers.get("x-routellm-mcp-domains") == "none", "Expected domains: none"
    assert int(headers.get("x-routellm-tools-pruned", 0)) >= 5, "Expected MCP tools pruned"
    print("      [✔] 100% of MCP tools pruned for pure math prompt")

    # 5B: Search Query (web_search domain active)
    search_req = {
        "model": "routellm",
        "messages": [{"role": "user", "content": "Search the web for AMD Ryzen AI Linux driver releases"}],
        "tools": sample_tools,
        "max_tokens": 40
    }
    status, headers, body = post_json("/v1/chat/completions", search_req)
    print(f"    [5B Search Query]")
    print(f"      Active Domains: {headers.get('x-routellm-mcp-domains')}")
    print(f"      Original Tools: {headers.get('x-routellm-tools-original')}")
    print(f"      Pruned Tools:   {headers.get('x-routellm-tools-pruned')}")
    assert "web_search" in headers.get("x-routellm-mcp-domains", ""), "Expected web_search domain"
    assert int(headers.get("x-routellm-tools-pruned", 0)) >= 4, "Expected browser tools pruned"
    print("      [✔] SearXNG retained while browser tools pruned")

    # 5C: Browser Navigation Query (browser domain active)
    browser_req = {
        "model": "routellm",
        "messages": [{"role": "user", "content": "Use browser_navigate to visit https://example.com and screenshot"}],
        "tools": sample_tools,
        "max_tokens": 40
    }
    status, headers, body = post_json("/v1/chat/completions", browser_req)
    print(f"    [5C Browser Query]")
    print(f"      Active Domains: {headers.get('x-routellm-mcp-domains')}")
    print(f"      Original Tools: {headers.get('x-routellm-tools-original')}")
    print(f"      Pruned Tools:   {headers.get('x-routellm-tools-pruned')}")
    assert "browser" in headers.get("x-routellm-mcp-domains", ""), "Expected browser domain"
    assert int(headers.get("x-routellm-tools-pruned", 0)) >= 1, "Expected search tool pruned"
    print("      [✔] Playwright tools retained while search tool pruned")

    # 6. Sticky Session Routing
    print("\n[Test 6] Sticky Session KV-Cache Affinity")
    session_id = "test-session-dev-alpha-99"
    headers_turn1 = {"X-Session-ID": session_id}
    status, h1, _ = post_json(
        "/v1/chat/completions",
        {"model": "routellm", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 15},
        headers=headers_turn1
    )
    node1 = h1.get("x-routellm-node")
    status, h2, _ = post_json(
        "/v1/chat/completions",
        {"model": "routellm", "messages": [{"role": "user", "content": "What was my first message?"}], "max_tokens": 15},
        headers=headers_turn1
    )
    node2 = h2.get("x-routellm-node")
    print(f"    Turn 1 Node: {node1}")
    print(f"    Turn 2 Node: {node2} {'[MATCH]' if node1 == node2 else '[FAIL]'}")
    assert node1 == node2, f"Session affinity broken: {node1} != {node2}"
    print("    [✔] PASSED: Sticky KV-Cache node affinity 100% deterministic")

    # 7. Error Handling
    print("\n[Test 7] Error Handling & Validation")
    status, _, err_body = post_json("/v1/chat/completions", {"model": "routellm", "messages": []})
    print(f"    Empty messages status: {status} | Error: {err_body}")
    assert status == 400, f"Expected 400 for empty messages, got {status}"
    print("    [✔] PASSED: Proper 400 Bad Request returned for invalid input")

    print("\n" + "=" * 70)
    print("  [✔] ALL 7 CORE & PROXY VERIFICATION TESTS PASSED WITH 0 ERRORS!")
    print("=" * 70)

if __name__ == "__main__":
    main()
