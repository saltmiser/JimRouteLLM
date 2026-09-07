# JimRouteLLM: Intelligent Local/LAN LLM Router & AVX-512 INT8 Classifier

**JimRouteLLM** is an OpenAI-compatible proxy and routing engine designed for local developer tooling, autonomous coding agents (e.g., Open Interpreter), and heterogeneous LAN clusters. 

It pairs an **INT8-quantized ModernBERT classifier** (accelerated by AMD Zen 4 AVX-512 VNNI with XDNA 1 NPU hardware device probe) with tiered routing across dual local models supporting native **Thinking** and **Multimodal Vision**, integrated with **SearXNG Web Search** and **Playwright Headless Chrome** over the Model Context Protocol (MCP).

[**Installation Guide**](INSTALL.md) • [**Hardware Profile (HP ZBook)**](DEVICES.md) • [**Testing & Benchmarks**](TESTING.md) • [**Architecture Decisions**](ARCHITECTURE_DECISIONS.md) • [**Citation**](CITATION.cff)

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
                                     │  • Sticky KV-Cache Affinity  │
                                     │  • MCP Domain Tool Filter    │
                                     │  • Head-Tail Middle Truncator│
                                     └──────────────┬───────────────┘
                                                    │
                                     ModernBERT-Large INT8 Engine
                                    [ AVX-512 VNNI / XDNA 1 Device ]
                                      Evaluates Prompt Complexity
                                                    │
                                          Score >= 0.28 (Hard)?
                                           /                 \
                                       [ YES ]             [ NO ]
                                          │                   │
                                          ▼                   ▼
                     ┌───────────────────────────────┐   ┌───────────────────────────────┐
                     │  `google/gemma-4-26b-a4b-qat` │   │     `google/gemma-4-e2b`      │
                     │  • 26B Heavy Reasoning Model  │   │  • High-Speed Model (72 tps)  │
                     │  • Multimodal Vision Input    │   │  • Multimodal Vision Input    │
                     │  • Deep Code/Math & Arch      │   │  • Fast Generation / Easy     │
                     │  • 128K Context, Single Slot  │   │  • 128K Context, Single Slot  │
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

### 1. High-Performance ModernBERT Classifier & Dual Device Execution
- **Quantized / FP16 Engines**:
  - `CLASSIFIER_DEVICE=cuda`: Executes in PyTorch FP16 directly on the NVIDIA RTX 2000 Ada laptop GPU (`CUDA-Ada`). Yields ultra-low latency (**~8–11ms** inference, 116ms at 2,048 tokens, 808ms at 8,192 tokens) while consuming only ~972 MiB of VRAM.
  - `CLASSIFIER_DEVICE=cpu`: Executes the INT8 quantized ONNX graph (`models/modernbert_large_int8.onnx`, 379.4 MB) via `CPUExecutionProvider` on the Zen 4 CPU utilizing AVX-512 VNNI vector instructions (yielding 18–35ms prompt scoring and 0 MB VRAM).
- **Hardware Profile**: Device `/dev/accel/accel0` (`RyzenAI-npu1`, device ID `0x1502`) managed via `pyxrt` and `amdxdna`.
- **Head-Tail Middle Truncation**: Prompts exceeding `CLASSIFIER_MAX_TOKENS` (default: 2048) are dynamically truncated from the center, strictly preserving both the Head ($N/2$ tokens of system framing, agent personas, and task definitions) and the Tail ($N/2$ tokens of final constraints, inputs, and latest user questions).

### 2. Dual Thinking & Multimodal Vision Endpoints
Both loaded endpoints support native reasoning (`reasoning_content` chain-of-thought tokens) and image inputs (`ALL_MODELS_SUPPORT_VISION=true`):
- **`google/gemma-4-e2b`** (Q6_K running at 72 tokens/sec):
  Primary endpoint for low-complexity queries, standard questions, quick formatting, and rapid vision tasks.
- **`google/gemma-4-26b-a4b-qat`** (Heavyweight MoE reasoning engine):
  Heavyweight reasoning engine for complex distributed architectures, algorithm design, formal proofs, and multi-file codebases.
- **Single-Slot Memory Efficiency**: Models are pinned to single active slots to prevent DDR5 memory bandwidth contention during KV cache lookups.

### 3. Sticky KV-Cache Preservation
- Standard round-robin proxies invalidate prompt caches on local inference engines.
- JimRouteLLM implements consistent hashing over `X-Session-ID`, `X-Conversation-ID`, or user identity, keeping multi-turn conversations sticky to the same model instance and node. Once a session engages the heavy model, sticky session affinity preserves the heavy KV-cache across follow-up turns to eliminate context re-ingestion latency.

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

Open Interpreter is configured with all available MCP servers in `~/.openinterpreter/config.toml`. While Open Interpreter considers all tools active at all times, **JimRouteLLM acts as a dynamic negative tool pruner**: it inspects each prompt in <0.2ms and automatically strips out irrelevant tool schemas before sending the prompt to the LLM, preserving fast Time-To-First-Token (<1s) while allowing Open Interpreter to natively execute the requested tool calls.

*(For the complete design journey, failure analysis of server-side tool injection, and empirical session traces, see the [**Architecture Decision Record**](ARCHITECTURE_DECISIONS.md).)*

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

# Routing Engine & ModernBERT INT8 Classifier
ROUTER_TYPE=modernbert
ROUTING_THRESHOLD=0.28
MODERNBERT_MODEL_ID=answerdotai/ModernBERT-large
MODERNBERT_USE_ONNX=true
NPU_ENABLED=true
CLASSIFIER_MAX_TOKENS=2048

# Local Target Models (LM Studio @ 127.0.0.1:1234)
LOCAL_EASY_MODEL=google/gemma-4-e2b
LOCAL_HARD_MODEL=google/gemma-4-26b-a4b-qat
LOCAL_VISION_MODEL=google/gemma-4-26b-a4b-qat
LOCAL_LM_STUDIO_URL=http://127.0.0.1:1234/v1
ALL_MODELS_SUPPORT_VISION=true

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
   Validates LAN node assignments, ModernBERT INT8 scoring, complexity thresholding, and multi-turn sticky hashing.

2. **MCP & Search Verification Suite**:
   ```bash
   ./venv/bin/python scripts/test_mcp_routing.py
   ```
   Verifies tool discovery from SearXNG and Playwright, domain classification, and live search execution.

3. **Live Proxy & Core Integration Suite**:
   ```bash
   ./venv/bin/python scripts/test_live_proxy.py
   ```
   Tests live HTTP endpoints against the running server (`http://127.0.0.1:8000`), validating tier routing (Gemma 4 E2B vs. Gemma 4 26B-A4B), live MCP dynamic tool schema pruning, response observability headers, sticky session caching, and error handling.

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
