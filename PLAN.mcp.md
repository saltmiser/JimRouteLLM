# 🛠️ RouteLLM Dynamic MCP Server & Tool Routing Plan

**Target Repository:** `~/src/jac-frc/RouteLLM` (`routellm_proxy/`)  
**Objective:** Transform RouteLLM from a model-only hybrid router into an **Intelligent Model & Dynamic MCP Tool Gateway**.  
**Core Benefit:** Eliminate client-side MCP configuration on developer workstations, prevent context-window explosion (saving 30k–50k tokens per turn), and allow local Apple Silicon M4 cluster models (`nvidia/nemotron-3-nano`) to execute tool calls reliably without getting overwhelmed by 100+ schemas.

---

## 1. Problem Statement & Architecture Goals

### 1.1. The Context Bloat & Local Model Bottleneck
Your LAN Docker container farm hosts **8 MCP servers with 100+ tools**:
* `shipstation-v1-mcp.frc` & `shipstation-v2-mcp.frc` (Shipping & Tracking)
* `shopify` & `recharge-v1w-mcp.frc` (Orders & Subscriptions)
* `cropster-v1-mcp.frc` & `listingmirror-v3-mcp` (Inventory & Roasting)
* `canto-v1-mcp.frc` (Digital Asset Management)
* `searxng-mcp.frc` (Private Web Search)

**The Failure Mode Today:**
1. **Context Window Exhaustion:** Eagerly injecting all 100+ tool schemas consumes **35,000–50,000 tokens** per request. Even for simple code queries, cloud API costs spike and context window limits are hit prematurely.
2. **Local Model Failure:** When your M4 cluster nodes (`nvidia/nemotron-3-nano`) receive 100 tools simultaneously in `tools: [...]`, attention degrades, latency spikes, and tool-call hallucinations skyrocket. Small models can reliably handle **2–4 relevant tools**, but not 100.
3. **Workstation Overhead:** Every developer laptop running Cline, Continue, Cursor, or Aider must maintain brittle local configurations, tokens, and network endpoints for all 8 servers.

---

## 2. Target Architecture: Two-Stage Hybrid Neural Router

```
                                      ┌───────────────────────────────────────────────────────────────┐
                                      │                     RouteLLM PROXY SERVER                     │
                                      │                                                               │
Client (Cline / Continue / Aider) ───►│ 1. POST /v1/chat/completions (Clean prompt or with tools)     │
(Zero local MCP configuration)        │                                                               │
                                      │ 2. STAGE 1: Domain & Tool Classifier                          │
                                      │    • Fast Regex / Keyword Gate (<0.2ms)                       │
                                      │    • Semantic Embedding / Domain Matcher                      │
                                      │    • Resolves: ["shipping", "ecommerce"]                      │
                                      │    • Filters 100 tools ──► 4 relevant tool schemas            │
                                      │                                                               │
                                      │ 3. STAGE 2: Model Complexity Classifier                       │
                                      │    • RouteLLM BERT Win-Rate Controller                        │
                                      │    • Text & Score < 0.45 ──► Local M4 Cluster (Nemotron)      │
                                      │    • Multimodal / Images ──► Vision Omni Node (Nemotron-Omni) │
                                      │    • Complex / Score >= 0.45 ──► Cloud Claude / Gemini        │
                                      │                                                               │
                                      │ 4. LiteLLM Execution:                                         │
                                      │    Prompt + ONLY the 4 targeted tools forwarded to model      │
                                      └───────────────────────────────────────────────────────────────┘
                                                │                               │
                        ┌───────────────────────┴───────────────┐               ▼
                        ▼                                       ▼       Cloud (Claude/Gemini)
                 Local M4 Cluster                       Vision Omni Node
           (Executes 4 tools cleanly)             (Executes 4 tools cleanly)
```

---

## 3. Detailed Component Design in `routellm_proxy`

### 3.1. Directory Structure Additions
```
routellm_proxy/
├── __init__.py
├── config.py           # Add MCP server URLs, tokens, and domain definitions
├── server.py           # Integrate tool filtering & MCP management endpoints
├── router.py           # Augment RoutingDecision with active tools metadata
├── main.py
│
├── mcp_registry.py     # NEW: Background sync & cache of Docker MCP tool schemas
├── mcp_classifier.py   # NEW: Fast keyword + semantic domain/tool classifier
└── mcp_executor.py     # NEW: (Optional) Server-side tool execution gateway
```

---

### 3.2. Domain & Tool Classification Specification (`mcp_classifier.py`)

Tools are partitioned into high-cohesion domain clusters:

| Domain Key | Target MCP Servers | Trigger Keywords & Semantic Entities |
| :--- | :--- | :--- |
| **`shipping`** | `shipstation-v1`, `shipstation-v2` | tracking, track, shipment, label, fedex, usps, ups, package, carrier, fulfill |
| **`ecommerce`** | `shopify`, `recharge-v1w-mcp` | order, customer, refund, subscription, charge, discount, cancel order, payment |
| **`roasting_inventory`**| `cropster-v1-mcp`, `listingmirror-v3` | lot, roast, green coffee, cupping, profile, inventory, listing, sku, vendor |
| **`digital_assets`** | `canto-v1-mcp` | canto, image asset, album, tag asset, download preset, photo metadata |
| **`web_search`** | `searxng-mcp` | search web, google, latest docs, look up online, search query |
| **`general_coding`** | *(None)* | python, react, bug, fix, refactor, typescript, function, sql, git, docker |

#### Classifier Logic:
1. **Stage 1A (Fast Heuristic - <0.2ms):** Regex pattern matching on user input for exact identifiers (e.g. order numbers `#836318`, tracking numbers `383016408966`, email addresses, domain keywords).
2. **Stage 1B (Semantic Intent Matcher):** If no direct regex match, use embedding cosine similarity (or lightweight Zero-Shot classification) against domain description centroids.
3. **Zero-Tool Fast Path:** If classified as `general_coding`, `tools` is set to `[]`. This immediately strips **~40,000 tokens** of JSON schema overhead, maximizing speed and context window for coding tasks.

---

### 3.3. Background MCP Registry (`mcp_registry.py`)

RouteLLM maintains active cached manifests of all tools across LAN containers:

```python
class MCPRegistry:
    """Manages connections to LAN Docker MCP servers and caches tool schemas."""

    def __init__(self):
        self.servers = settings.mcp_servers
        self.cached_tools: dict[str, list[dict]] = {}  # server_name -> tool schemas
        self.domain_map: dict[str, list[str]] = settings.mcp_domain_map

    async def sync_all_servers(self):
        """Polls /mcp or /sse or /openapi.json on each container and caches schemas."""
        for server_name, cfg in self.servers.items():
            try:
                tools = await self._fetch_server_tools(cfg["url"], cfg.get("token"))
                self.cached_tools[server_name] = tools
                logger.info(f"Synced {len(tools)} tools from MCP server '{server_name}'")
            except Exception as e:
                logger.warning(f"Failed to sync MCP server '{server_name}': {e}")

    def get_tools_for_domains(self, domains: list[str]) -> list[dict]:
        """Return combined, deduplicated tool schemas for active domains."""
        active_tools = []
        for domain in domains:
            for server_name in self.domain_map.get(domain, []):
                active_tools.extend(self.cached_tools.get(server_name, []))
        return active_tools
```

---

### 3.4. Integration into `routellm_proxy/server.py`

Update `POST /v1/chat/completions`:

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    verify_auth(authorization)
    body = await request.json()
    messages = body.get("messages", [])
    
    # 1. MCP Tool Classification & Dynamic Pruning
    user_prompt = router.extract_prompt_text(messages)
    incoming_tools = body.get("tools", [])
    
    # Classify intent domains
    active_domains = mcp_classifier.classify_domains(user_prompt)
    
    if settings.mcp_mode == "filter" and incoming_tools:
        # Filter workstation-provided tools down to relevant subset
        pruned_tools = mcp_classifier.filter_tools_by_domain(incoming_tools, active_domains)
        call_tools = pruned_tools
    elif settings.mcp_mode == "gateway":
        # Inject tools centrally from RouteLLM registry
        call_tools = mcp_registry.get_tools_for_domains(active_domains)
    else:
        call_tools = incoming_tools

    # 2. Model Routing Decision (Local vs Cloud)
    decision = router.decide(messages, requested_model=body.get("model", "routellm"))

    # 3. Build LiteLLM kwargs with pruned toolset
    call_kwargs = {
        "messages": messages,
        "stream": body.get("stream", False),
        "tools": call_tools if call_tools else None,
    }
    if call_tools and "tool_choice" in body:
        call_kwargs["tool_choice"] = body["tool_choice"]
```

---

## 4. Configuration Reference (`.env` and `config.py`)

Add to `RouteLLM/.env`:

```bash
# ── MCP Dynamic Tool Routing Settings ────────────────────────────────────────
ENABLE_MCP_ROUTING=true
MCP_ROUTING_MODE=filter                  # "filter" (prunes client tools) or "gateway" (injects from registry)
MCP_CACHE_TTL_SECONDS=300                # Re-sync container tool schemas every 5 minutes

# Docker Farm MCP Container Endpoints (on LAN)
MCP_SHIPSTATION_V1_URL=http://shipstation-v1-mcp.frc:8000/mcp
MCP_SHIPSTATION_V1_TOKEN=3782729026ec8ddbed684dc040148246

MCP_SHIPSTATION_V2_URL=http://shipstation-v2-mcp.frc:8000/mcp
MCP_SHIPSTATION_V2_TOKEN=3782729026ec8ddbed684dc040148246

MCP_SHOPIFY_URL=http://shopify-mcp.frc:8000/mcp
MCP_SHOPIFY_TOKEN=6a9d7c08b273b4e195da188f6a91176c

MCP_RECHARGE_URL=http://recharge-v1w-mcp.frc:8000/mcp
MCP_RECHARGE_TOKEN=f25061413e7f56878b73f58d28911add

MCP_CROPSTER_URL=http://cropster-v1-mcp.frc:8000/mcp
MCP_CROPSTER_TOKEN=0b0ec41ae08b5330602aaf1be51ff3f4

MCP_LISTINGMIRROR_URL=http://listingmirror-v3-mcp.frc:8000/mcp
MCP_LISTINGMIRROR_TOKEN=bbe6accfb4eaf5a5930cd38a357ed0cc

MCP_CANTO_URL=http://canto-v1-mcp.frc:8000/mcp
MCP_CANTO_TOKEN=ce2ed411b913968bac48b61b1b365ecf

MCP_SEARXNG_URL=http://searxng-mcp.frc:8000/mcp
MCP_SEARXNG_TOKEN=8c4e0b512e023ca270769cf3c7fbe8e9
```

---

## 5. Workstation Configuration (Zero-Friction Setup)

Once RouteLLM is deployed with MCP Dynamic Routing, developer workstations configure **only one endpoint**:

### Cline / Continue / Roo Code / Cursor / Aider
```json
{
  "api_provider": "OpenAI Compatible",
  "base_url": "http://jacfrcmac.frc:8000/v1",
  "model_id": "routellm",
  "api_key": "routellm-key"
}
```
* **No MCP servers configured in Cline or Continue.**
* When the user types `"Where is order #868278?"`, RouteLLM automatically detects the shipping intent, equips the model with `shipstation_list_orders` and `shopify_get_order`, and routes the prompt to the local M4 cluster or Claude.
* When the user types `"Write a Python unit test for this FastAPI endpoint"`, RouteLLM strips all tools, saving 40,000 tokens and executing at maximum speed.

---

## 6. Implementation Milestones & Roadmap

- [ ] **Milestone 1: Dynamic Tool Pruning (`filter` mode)**
  - Implement `mcp_classifier.py` with fast regex & domain keyword dictionary.
  - Intercept `request.tools` in `routellm_proxy/server.py` and prune irrelevant tools before calling `litellm`.
  - Add response headers: `X-RouteLLM-Active-Domains`, `X-RouteLLM-Tools-Pruned`, `X-RouteLLM-Tokens-Saved`.
- [ ] **Milestone 2: Container Registry Sync (`mcp_registry.py`)**
  - Implement periodic async polling of all 8 Docker MCP containers over LAN.
  - Store sanitized JSON schemas in memory.
- [ ] **Milestone 3: Server-Side Gateway (`gateway` mode)**
  - Allow clients to send zero tools; RouteLLM injects the appropriate tool catalog dynamically based on query intent.
- [ ] **Milestone 4: Observability & Admin UI**
  - Expose `/v1/mcp/stats` to report total tokens saved, tool usage frequency, and classification accuracy.

---

## 7. Performance & Token Savings Projection

| Metric | Workstation Eager MCP (Before) | RouteLLM Dynamic MCP (After) | Improvement |
| :--- | :--- | :--- | :--- |
| **Tool Token Overhead (Coding Prompt)** | ~42,000 tokens / call | **0 tokens / call** | **100% reduction** |
| **Tool Token Overhead (Shipping Prompt)**| ~42,000 tokens / call | **~1,800 tokens / call** | **95.7% reduction** |
| **Local Model Reliability (Nemotron)** | Low (fails with >20 tools) | **High (passes with 2-4 tools)**| Reliable local execution |
| **Workstation Config Management** | 8 servers on every laptop | **0 servers on laptops** | Centralized in Docker |
| **Average Cost per Coding Turn (Cloud)** | ~$0.15–$0.30 | **~$0.005–$0.01** | **~90% cost savings** |
