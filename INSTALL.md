# JimRouteLLM: Installation & Setup Guide

This guide walks through configuring and running **JimRouteLLM** on local Linux hardware, including setting up the NPU-accelerated classifier, local LLM backends (LM Studio), SearXNG web search, Playwright headless browser automation, and Open Interpreter integration.

For details regarding the host hardware profile (AMD Ryzen AI NPU + dual GPUs), see [DEVICES.md](DEVICES.md).

---

## 1. Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 24.04 / 26.04 recommended)
- **Python**: Version 3.10 or newer (with `python3-venv` and `python3-pip`)
- **Node.js**: v18+ (for `@playwright/mcp`)
- **Browser**: Google Chrome installed at `/usr/local/bin/google-chrome` or `/usr/bin/google-chrome`
- **LM Studio**: Running locally at `http://127.0.0.1:1234` with OpenAI-compatible API enabled

---

## 2. Clone & Environment Setup

Clone the repository and create an isolated Python virtual environment:

```bash
cd /home/jac-jim/src/jac-jim/mgmt/JimRouteLLM

# Create Python virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Key Python Dependencies
- `fastapi`, `uvicorn`: High-throughput async REST proxy engine
- `litellm`: Multi-provider backend dispatch
- `onnxruntime` or `onnxruntime-vitisai`: NPU / ONNX acceleration for ModernBERT
- `pydantic-settings`: Type-safe configuration loading
- `requests`, `httpx`: Async network transport

---

## 3. Environment Configuration (`.env`)

Copy `.env.example` to create your active `.env`:

```bash
cp .env.example .env
```

Ensure the configuration matches your local endpoints:

```bash
# Server Host & Port
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

# MCP Dynamic Tool Pruning
ENABLE_MCP_ROUTING=true
MCP_ROUTING_MODE=filter
MCP_SERVERS_FILE=mcpServers.json
MCP_MAX_LOCAL_TOOLS=16
```

---

## 4. MCP Servers Configuration

`mcpServers.json` defines the external Model Context Protocol services:

```json
{
  "mcpServers": {
    "searxng": {
      "command": "/home/jac-jim/src/jac-jim/mgmt/JimRouteLLM/venv/bin/python",
      "args": ["/home/jac-jim/src/jac-jim/mgmt/JimRouteLLM/mcp_searxng_server.py"]
    },
    "playwright": {
      "command": "npx",
      "args": [
        "-y",
        "@playwright/mcp",
        "--headless",
        "--executable-path",
        "/usr/local/bin/google-chrome",
        "--caps",
        "vision"
      ]
    }
  }
}
```

Ensure the local SearXNG daemon is running at `http://127.0.0.1:8888`.

---

## 5. Verification & Testing

Before starting the server, run the automated verification suites:

```bash
# 1. Verify routing decisions, LAN node configs, and NPU classification:
./venv/bin/python scripts/test_proxy.py

# 2. Verify MCP tool discovery, schema synchronization, and intent pruning:
./venv/bin/python scripts/test_mcp_routing.py
```

Both tests should exit with code 0.

---

## 6. Starting the Proxy Server

To launch JimRouteLLM:

```bash
./scripts/start.sh
```

Or run via `uvicorn` directly:

```bash
./venv/bin/uvicorn jimroutellm_proxy.server:app --host 0.0.0.0 --port 8000
```

Verify that the health check responds:
```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","timestamp":...}
```

---

## 7. Open Interpreter Client Integration

Configure Open Interpreter in `~/.openinterpreter/config.toml` with the desired MCP servers (`playwright`, `searxng`, etc.). Open Interpreter considers all tools active, while JimRouteLLM transparently prunes tools unneeded for the current prompt:

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

[projects."/home/jac-jim/src/jac-jim/mgmt"]
trust_level = "trusted"
```

### Running Open Interpreter

Interactive session:
```bash
interpreter
```

Non-interactive autonomous task:
```bash
interpreter exec --dangerously-bypass-approvals-and-sandbox \
  "Use searxng_search to find the latest Linux kernel version and print it."
```

---

## 8. Troubleshooting
 
 - **502 Bad Gateway / Connection Refused**:
   - Verify LM Studio is running on `http://127.0.0.1:1234` with the models loaded.
 - **Classifier Hardware Status**:
   - JimRouteLLM probes `/dev/accel/accel0` on startup and executes the INT8 ModernBERT ONNX graph using AMD Zen 4 AVX-512 VNNI instructions via `CPUExecutionProvider`, achieving sub-35ms prompt scoring.
 - **High TTFT Delay**:
   - Check `X-RouteLLM-Tools-Pruned` header in server responses to ensure dynamic MCP pruning is active (`ENABLE_MCP_ROUTING=true`).
