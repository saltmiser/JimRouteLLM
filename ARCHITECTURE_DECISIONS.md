# Architectural Decision Record (ADR): Dynamic Negative Tool Pruning

**Status**: Accepted & Implemented  
**Date**: September 7, 2026  
**Component**: JimRouteLLM Proxy (`jimroutellm_proxy`) & Open Interpreter Client (`~/.openinterpreter/config.toml`)  
**Authors**: saltmiser (System Architect) & Gemini 3.8 Flash (Pair Programmer)  

---

## 1. Context & Motivation

When deploying autonomous coding agents (such as Open Interpreter or Codex CLI) backed by local, on-device Large Language Models (e.g., `google/gemma-4-12b-qat` and `meta/muse-glimmer` on an HP ZBook Power G11 workstation), two primary challenges arise:

1. **Context Window & TTFT Latency**: Local models running on consumer workstation GPUs and NPUs are sensitive to prompt token size. Every additional 1,000 tokens of prompt context increases Time-To-First-Token (TTFT) and places heavy strain on KV-cache memory.
2. **Tool Specialization**: Real-world developer agents need access to specialized tools via the Model Context Protocol (MCP)—including local multi-engine web search ([SearXNG](http://127.0.0.1:8888)), headless browser automation ([Playwright Chrome](https://github.com/microsoft/playwright)), file manipulation, and terminal execution.

How these tools are registered, exposed, and filtered between the agent client (Open Interpreter) and the inference routing proxy (JimRouteLLM) determines whether the system is responsive (<1s latency) or stalls in unrecoverable retry loops.

---

## 2. Evolution of the Architecture

The architecture went through three distinct design phases before converging on the optimal paradigm.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                ARCHITECTURAL EVOLUTION                                 │
│                                                                                        │
│  Phase 1: Monolithic Client Injection                                                  │
│  [Open Interpreter] ──(Dumps 47 tools / 12K tokens on every turn)──> [Local LLM]      │
│  Outcome: 18-40s TTFT, prompt bloat, model confusion on simple queries.                │
│                                                                                        │
│  Phase 2: Invisible Server-Side Injection                                              │
│  [Open Interpreter] ──(0 tools)──> [Proxy Injects & Executes Tools] <──> [Local LLM]  │
│  Outcome: Broken agent approvals, client-server desync, browser infinite loops.        │
│                                                                                        │
│  Phase 3: Dynamic Negative Tool Pruning (CURRENT)                                      │
│  [Open Interpreter] ──(All 47 tools)──> [Proxy Strips Unused Tools] ──> [Local LLM]   │
│  Outcome: <1s TTFT, full client sovereignty, multi-tool chaining, zero server bloat.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: The Monolithic Client-Side Tool Dump

#### Description
Both `@playwright/mcp` and SearXNG were registered directly in Open Interpreter's `~/.openinterpreter/config.toml`. 

#### The Failure Mode
Open Interpreter naively included the full JSON Schema definitions for every single registered tool in every `/v1/chat/completions` request. This introduced:
- **30+ Playwright browser schemas** (navigation, element clicking, drag-and-drop, accessibility trees, console listeners).
- **SearXNG web search schema**.
- **Core agent schemas** (bash execution, python scripting).

Total overhead: **~4,800 to 12,500 prompt tokens** per request.

#### Impact on Local Models
- Simple non-tool queries (e.g., `"Calculate 42 * 42"`) took **18 to 40 seconds** just to emit the first token.
- Local 12B models frequently hallucinated nonexistent parameters or attempted to call browser tools when asked basic math or programming questions.
- KV-cache allocation was exhausted rapidly during multi-turn chats.

---

### Phase 2: The "Invisible Server-Side Injection" Hypothesis (The Misstep)

#### Description
To eliminate client prompt bloat, we tested removing MCP servers entirely from Open Interpreter's `config.toml`. Instead, the `JimRouteLLM` proxy attempted to:
1. Inspect the incoming prompt.
2. Invisibly inject server-side tool schemas (e.g., `searxng_search` or `browser_navigate`) into the request forwarded to the LLM.
3. Intercept the LLM's `tool_calls` response before it reached Open Interpreter.
4. Execute the tool internally inside the proxy server (e.g., querying SearXNG or driving Playwright via Python).
5. Append the tool output to the message history and perform a recursive followup completion.
6. Synthesize the final answer and stream it to Open Interpreter as plain text chunks.

#### Why It Failed
While this worked for isolated single-turn queries, it completely broke autonomous coding agent workflows:

1. **Stolen Agent Flow**: Open Interpreter lost visibility into tool calls. Users could not approve commands, view live execution progress bars, or enforce filesystem sandboxing.
2. **Streaming Desynchronization**: Open Interpreter's streaming parser received synthesized responses that didn't match its expected turn-by-turn state machine.
3. **Infinite Browser Loops**: When the model emitted browser automation calls, Open Interpreter complained of `unsupported call: browser_navigate` or entered an infinite retry loop attempting shell commands like `which agent-browser`.
4. **Massive DOM Payloads**: Server-side Playwright execution returned massive raw DOM accessibility trees into the proxy's local LLM context, triggering server-side timeouts and stalled socket connections.

---

### Phase 3: The Epiphany — Dynamic Negative Tool Pruning

#### The Breakthrough Realization
The fundamental error in Phase 2 was attempting to make the proxy act as an *agent*. An LLM router should not execute agent tools—it should optimize payloads and route them.

The correct mental model was articulated as:

> *"We should tell Open Interpreter about all MCP tools enabled in the JimRouteLLM proxy, and then think of this differently: the proxy server removes tools that a specific prompt isn't using yet. Instead of the proxy server trying to invisibly add MCP servers, the client thinks all MCP tools are active all the time, and the proxy server removes unused MCP servers on a per-turn basis!"*

#### How Dynamic Negative Pruning Works

1. **Client Sovereignty (`~/.openinterpreter/config.toml`)**:
   - Open Interpreter registers all external MCP servers natively:
     ```toml
     [mcp_servers.playwright]
     command = "npx"
     args = ["-y", "@playwright/mcp", "--headless", "--executable-path", "/usr/local/bin/google-chrome", "--caps", "vision"]

     [mcp_servers.searxng]
     command = "python3"
     args = ["/home/jac-jim/src/jac-jim/mgmt/JimRouteLLM/mcp_searxng_server.py"]
     ```
   - Open Interpreter's native client retains full authority over:
     - User approvals and permission escalations.
     - Filesystem and network sandboxing.
     - Live UI rendering and execution feedback.
     - Native multi-tool chaining (combining bash commands with MCP calls).

2. **Full Catalog Request**:
   - On every request, Open Interpreter sends its full catalog of registered tools (47+ tools).

3. **Sub-Millisecond Domain Classification**:
   - JimRouteLLM's `MCPClassifier` inspects the conversation (recent user prompts and assistant tool calls) in **<0.2ms** using compiled regexes, domain token sets, and keyword heuristics.
   - It identifies active domain clusters (`web_search`, `browser`, `shipping`, `ecommerce`, etc.).

4. **Dynamic Negative Filtering (`filter_tools`)**:
   - The proxy prunes away every tool schema belonging to inactive domains:
     - **Math / Coding / Factoid**: Active domains = `['none']`. Strips all 31 MCP tools. Model receives 0 MCP overhead (saves **~4,800 tokens**).
     - **Web Search**: Active domains = `['web_search']`. Strips all 30 Playwright tools; retains only `searxng_search`.
     - **Browser Automation**: Active domains = `['browser']`. Strips SearXNG; retains only Playwright tools (ranked by prompt relevance).
   - **Core Agent Invariance**: Client-native tools (`exec_command`, `bash`, `python`, `file_editor`) are recognized by `is_core_agent_tool()` and are **never** pruned.

5. **Pass-Through Proxy Simplicity**:
   - The proxy forward-prunes the payload and streams the LLM's response directly back to Open Interpreter.
   - The proxy contains **zero server-side tool execution loops** and **zero synthetic message injections**.

---

## 3. Empirical Verification: Session `01a07d69-979e-7513-8d69-5c0b4fbbfa40`

The power of this architecture was validated in a live multi-turn agent session (`codex-tui` / Open Interpreter session `01a07d69-979e-7513-8d69-5c0b4fbbfa40`):

### Turn 1: Chained Shell + SearXNG Execution
- **User Prompt**: `"Check the current date, and then search the web for an update on the Iran war."`
- **Proxy Behavior**: Classified `active_domains: ['web_search']`. Pruned 30 Playwright tools.
- **Step 1 (Bash Tool)**: Model emitted `exec_command(cmd="date")`. Open Interpreter executed `/bin/bash -lc date` in **0.004s**, returning `Mon Sep 7 15:48:27 EDT 2026`.
- **Step 2 (SearXNG MCP Tool)**: Model incorporated the date and emitted `searxng_search(query="Iran war update September 2026")`. Open Interpreter executed `mcp_searxng_server.py` in **1.09s**, receiving 5 real-time news articles.
- **Step 3 (Synthesis)**: Delivered a structured 5-bullet analysis of current developments.

### Turn 2: Conversational Follow-up with Zero Context Bloat
- **User Prompt**: `"What will the United Arab Emirates do, were they attacked recently?"`
- **Proxy Behavior**: Maintained continuity, stripped Playwright tools, preserved fast **1.95s TTFT**.
- **Execution**: Model emitted `searxng_search(query="UAE attacked recently September 2026 Iran war")`. Executed in **1.09s**, yielding immediate analysis of recent Artesh drone/missile strikes and diplomatic statements.

---

## 4. Architectural Comparison

| Dimension | Monolithic Client (Phase 1) | Server Injection (Phase 2) | Dynamic Negative Pruning (Phase 3) |
| :--- | :--- | :--- | :--- |
| **Client Configuration** | All MCP servers configured | Zero MCP servers configured | **All MCP servers configured** |
| **Prompt Token Bloat** | +5,000 to +12,000 tokens/turn | 0 tokens (client) | **Pruned to <200 tokens on non-tool turns** |
| **Time-To-First-Token (TTFT)** | 18s – 40s (stalls) | Stalled on large DOMs | **<1s (math/code), ~1.9s (tool turns)** |
| **Agent Shell/Tool Chaining** | Worked, but extremely slow | Broken (proxy hijacked loop) | **Flawless (bash + MCP freely intermixed)** |
| **Approval / Sandbox Support** | Client-enforced | Bypassed / Broken | **100% Client-enforced & Native** |
| **Proxy Code Complexity** | Low | High (150+ lines of agent loops) | **Minimal (pure router & schema filter)** |
| **Tool Execution Locus** | Client-side | Server-side | **Client-side** |

---

## 5. Summary & Design Principles

1. **The Client is the Agent**: Tool execution, approval prompts, sandboxing, and UI rendering belong exclusively to the client harness (Open Interpreter).
2. **The Proxy is the Optimizer**: The proxy's role is strictly routing, caching, and payload optimization (negative schema pruning).
3. **Negative Pruning Scales**: As more MCP domains are added (e.g., Shopify, ShipStation, Cropster, Canto), the client registers them all. The LLM only ever receives the tiny subset of schemas relevant to the user's immediate intent.
