# JimRouteLLM: Intelligent Local/LAN LLM Router & NPU-Accelerated Classifier

**JimRouteLLM** is an OpenAI-compatible proxy and routing engine designed for local developer tooling, autonomous coding agents (e.g., Open Interpreter), and heterogeneous LAN clusters. 

It pairs an **AMD XDNA 1 NPU-accelerated ModernBERT classifier** with tiered routing across dual local models supporting native **Thinking** and **Multimodal Vision**, integrated with **SearXNG Web Search** and **Playwright Headless Chrome** over the Model Context Protocol (MCP).

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
- **Dynamic MCP Tool Pruning** (`mcp_classifier.py`):
  - Protects local models from naive MCP client catalog dumping (which can inject 12,000+ tokens of unused JSON schemas into the prompt, delaying TTFT by 15-20s).

---

## Open Interpreter Integration

Open Interpreter is pre-configured to utilize JimRouteLLM and both MCP servers in `~/.openinterpreter/config.toml`:

```toml
model = "routellm"
model_provider = "jimroutellm"

[model_providers.jimroutellm]
name = "JimRouteLLM"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "chat"

[mcp_servers.searxng]
command = "/home/jac-jim/src/jac-jim/mgmt/JimRouteLLM/venv/bin/python"
args = ["/home/jac-jim/src/jac-jim/mgmt/JimRouteLLM/mcp_searxng_server.py"]

[mcp_servers.playwright]
command = "npx"
args = ["-y", "@playwright/mcp", "--headless", "--executable-path", "/usr/local/bin/google-chrome", "--caps", "vision"]
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
