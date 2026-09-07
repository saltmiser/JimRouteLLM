# JimRouteLLM: Intelligent Local/LAN LLM Router & NPU-Accelerated Classifier

**JimRouteLLM** is an OpenAI-compatible proxy and routing engine designed for local developer tooling, autonomous coding agents (e.g., Open Interpreter), and heterogeneous LAN clusters. 

It pairs an **AMD XDNA 1 NPU-accelerated ModernBERT classifier** with tiered routing across dual local models supporting native **Thinking** and **Multimodal Vision**, integrated with **SearXNG Web Search** and **Playwright Headless Chrome** over the Model Context Protocol (MCP).

[**Installation Guide**](INSTALL.md) • [**Hardware Profile (HP ZBook)**](DEVICES.md) • [**Testing & Benchmarks**](TESTING.md) • [**Citation**](CITATION.cff)

![AI Generated](https://img.shields.io/badge/Code-100%25%20AI%20Generated-7952n2?style=flat-square&logo=openai&logoColor=white)
---

## Architecture Overview

```
                                      [ Client / Open Interpreter ]
                                                    │
                                         POST /v1/chat/completions
                                                    ▼
                                     ┌──────────────────────────────┐
                                     │         JimRouteLLM          │
                                     │  • Sticky KV-Cache Hashing   │
                                     │  • MCP Domain Tool Filter    │
                                     └──────────────┬───────────────┘
                                                    │
                                       AMD Ryzen AI XDNA 1 NPU
                                     [ ModernBERT-Large INT8/ONNX ]
                                      Evaluates Prompt Complexity
                                                    │
                                          Score >= 0.45 (Hard)?
                                           /                 \
                                       [ YES ]             [ NO ]
                                          │                   │
                                          ▼                   ▼
                     ┌───────────────────────────────┐   ┌───────────────────────────────┐
                     │      `meta/muse-glimmer`      │   │    `google/gemma-4-12b-qat`   │
                     │  • 28B Heavy Reasoning Model  │   │  • 12B Fast Thinking Model    │
                     │  • Multimodal Vision Input    │   │  • Multimodal Vision Input    │
                     │  • Deep Code/Math & Arch      │   │  • Fast Generation / Easy     │
                     │  • 131K Context, Single Slot  │   │  • 256K Context, Single Slot  │
                     │  • AMD GPU (Vulkan/ROCm)      │   │  • NVIDIA Ada GPU (CUDA)      │
                     └───────────────────────────────┘   └───────────────────────────────┘
                                           \               /
                                            ▼             ▼
                                  ┌──────────────────────────────┐
                                  │      Local LM Studio API     │
                                  │    http://127.0.0.1:1234/v1  │
                                  └──────────────────────────────┘
```

---

## Core Features

### 1. Hardware-Accelerated Routing via AMD XDNA 1 NPU
- **Device**: `/dev/accel/accel0` (`RyzenAI-npu1`) via ONNX Runtime & `pyxrt`.
- **Classifier**: `answerdotai/ModernBERT-large` (395M parameters, 8k context window).
- **Latency**: Sub-40ms prompt complexity scoring offloaded entirely from the CPU and GPU.

### 2. Dual Thinking & Multimodal Vision Endpoints
Both loaded endpoints support native reasoning (`reasoning_content` chain-of-thought tokens) and image inputs:
- **`google/gemma-4-12b-qat`** (7.15 GB, 256K context, 1 slot):
  Primary endpoint for general tasks, rapid code generation, and standard vision questions.
- **`meta/muse-glimmer`** (18.16 GB, 131K context, 1 slot):
  Heavyweight reasoning engine for complex distributed architectures, formal proofs, and deep vision analysis.
- **Single-Slot Memory Efficiency**: Models are pinned to a single active slot to prevent DDR5 memory bandwidth contention during KV cache lookups.

### 3. Sticky KV-Cache Preservation
- Standard round-robin proxies invalidate prompt caches on local inference engines.
- JimRouteLLM implements consistent hashing over `X-Session-ID`, `X-Conversation-ID`, or user identity, keeping multi-turn conversations sticky to the same model instance.

### 4. Integrated Model Context Protocol (MCP) Scaffolding
- **SearXNG Web Search** (`mcp_searxng_server.py`):
  - Connects to local SearXNG (`http://127.0.0.1:8888`) with live Google Search & Google CSE querying.
  - Exposes `searxng_search` via MCP and as a standalone CLI tool in `$PATH`.
- **Playwright Headless Chrome** (`@playwright/mcp`):
  - Automates host Google Chrome (`/usr/local/bin/google-chrome`) headlessly with `--caps vision`.
  - Exposes 30 browser automation tools (navigate, click, type, accessibility snapshot trees, screenshots).
- **Dynamic MCP Tool Pruning & Latency Optimization** (`mcp_classifier.py`):
  - **The Problem**: Coding assistant clients (e.g. Open Interpreter) naively inject all registered MCP tool schemas (30+ Playwright tools + SearXNG) into the prompt on every turn, adding 12,000+ tokens of schema overhead that stalls local single-slot models with ~18s TTFT delays.
  - **The Solution**: JimRouteLLM intercepts the payload, inspects user intent and conversation history, identifies active domains (`web_search`, `browser`, etc.), and dynamically prunes inactive MCP tool schemas while preserving core agent tools (`exec_command`, `write_stdin`, etc.).
  - **Observability**: Returns detailed response headers:
    - `X-RouteLLM-MCP-Domains`: Active domains (e.g. `web_search`, `browser`, or `none`).
    - `X-RouteLLM-Tools-Original`: Initial tool schema count.
    - `X-RouteLLM-Tools-Pruned`: Count of stripped schemas.
    - `X-RouteLLM-Tokens-Saved`: Estimated prompt tokens saved.
  - **Performance**: Pruned prompt overhead by over 98% (from ~12,500 down to ~150 tokens), cutting Time-To-First-Token from ~18s to <1s and slashing turnaround time on local hardware from ~38s down to **9 seconds**.

---

## Open Interpreter Integration

Open Interpreter is configured with all available MCP servers in `~/.openinterpreter/config.toml`. While Open Interpreter considers all tools active at all times, **JimRouteLLM acts as a dynamic negative tool pruner**: it inspects each prompt in <0.2ms and automatically strips out irrelevant tool schemas before sending the prompt to the LLM, preserving fast Time-To-First-Token (<1s) while allowing Open Interpreter to natively execute the requested tool calls:

```toml
model = "routellm"
model_provider = "jimroutellm"

[model_providers.jimroutellm]
name = "JimRouteLLM"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "chat"

[mcp_servers.playwright]
command = "npx"
args = ["-y", "@playwright/mcp", "--headless", "--executable-path", "/usr/local/bin/google-chrome", "--caps", "vision"]

[mcp_servers.searxng]
command = "python3"
args = ["/home/jac-jim/src/jac-jim/mgmt/JimRouteLLM/mcp_searxng_server.py"]
```

To run an interactive session:
```bash
interpreter
```

To run a non-interactive task:
```bash
interpreter exec "Use searxng_search to find the latest Linux kernel release and summarize it."
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and adjust as needed:

```bash
# Proxy Server
PROXY_HOST=0.0.0.0
PROXY_PORT=8000

# Routing Engine & NPU Classifier
ROUTER_TYPE=modernbert
ROUTING_THRESHOLD=0.45
MODERNBERT_MODEL_ID=answerdotai/ModernBERT-large
MODERNBERT_USE_ONNX=true
NPU_ENABLED=true

# Local Models (LM Studio @ 127.0.0.1:1234)
LOCAL_EASY_MODEL=google/gemma-4-12b-qat
LOCAL_HARD_MODEL=meta/muse-glimmer
LOCAL_VISION_MODEL=meta/muse-glimmer
LOCAL_LM_STUDIO_URL=http://127.0.0.1:1234/v1

# MCP Scaffolding
ENABLE_MCP_ROUTING=true
MCP_ROUTING_MODE=filter
MCP_SERVERS_FILE=mcpServers.json
```

---

## Verification & Testing

JimRouteLLM includes automated verification suites:

1. **Routing & Unit Test Suite**:
   ```bash
   ./venv/bin/python scripts/test_proxy.py
   ```
   Validates LAN node assignments, NPU classifier accuracy, complexity thresholding, and multi-turn sticky hashing.

2. **MCP & Search Verification Suite**:
   ```bash
   ./venv/bin/python scripts/test_mcp_routing.py
   ```
   Verifies tool discovery from SearXNG and Playwright, domain classification, and live search execution.

3. **Live Proxy & Core Integration Suite**:
   ```bash
   ./venv/bin/python scripts/test_live_proxy.py
   ```
   Tests live HTTP endpoints against the running server (`http://127.0.0.1:8000`), validating tier routing (Gemma vs. Muse), live MCP dynamic tool schema pruning, response observability headers, sticky session caching, and error handling.

4. **Multi-Turn KV-Cache & Sticky Session Benchmark**:
   ```bash
   ./venv/bin/python scripts/test_kv_cache_multiturn.py
   ```
   Simulates a 10-turn sequential developer session, verifying 100% deterministic node affinity to maintain local hardware prompt KV-cache acceleration while transitioning across coding, search, and browser domains.

---

## Starting the Server

To launch the proxy server:
```bash
./scripts/start.sh
```
The server will start at `http://0.0.0.0:8000` with the standard OpenAI API endpoints:
- `GET  /v1/models`
- `POST /v1/chat/completions`
- `GET  /health`
