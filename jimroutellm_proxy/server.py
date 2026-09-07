import json
import time
import uuid
import hashlib
import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Header, Request, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import litellm
import requests

from jimroutellm_proxy.config import settings
from jimroutellm_proxy.router import router, RoutingDecision
from jimroutellm_proxy.nodes import node_manager
from jimroutellm_proxy.mcp_registry import mcp_registry, clean_openai_tool
from jimroutellm_proxy.mcp_classifier import mcp_classifier

logger = logging.getLogger("jimroutellm.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Suppress noisy LiteLLM logs
litellm.set_verbose = False
litellm.drop_params = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup background tasks."""
    logger.info("Starting JimRouteLLM Proxy Server...")
    
    # 1. Start node health check background loop
    health_task = asyncio.create_task(node_manager.background_health_check_loop())
    
    # 2. MCP scaffolding initialization
    mcp_sync_task = None
    if settings.enable_mcp_routing:
        mcp_registry.load_config()
        mcp_sync_task = asyncio.create_task(mcp_registry.sync_all_servers())

    yield

    # Shutdown
    health_task.cancel()
    if mcp_sync_task:
        mcp_sync_task.cancel()


app = FastAPI(
    title="JimRouteLLM Hybrid Proxy",
    description="Intelligent Heterogeneous LAN Cluster + Gemini Cloud Router powered by ModernBERT-Large (395M)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_auth(authorization: Optional[str] = Header(None)):
    if settings.proxy_api_key:
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing Authorization header",
            )
        token = authorization.replace("Bearer ", "").strip()
        if token != settings.proxy_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )


def extract_session_key(request: Request, body: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
    """
    Extract a deterministic session key to preserve KV prompt cache across turns.
    Priority:
    1. Header: X-Session-ID / X-Conversation-ID
    2. Payload 'user' attribute
    3. First message content fingerprint
    4. Client IP
    """
    for header in ["x-session-id", "x-conversation-id", "session-id", "conversation-id"]:
        val = request.headers.get(header)
        if val:
            return val.strip()

    user_val = body.get("user")
    if user_val and isinstance(user_val, str):
        return user_val.strip()

    if messages and len(messages) > 0:
        first_content = str(messages[0].get("content", ""))
        if first_content:
            return hashlib.md5(first_content.encode("utf-8")).hexdigest()

    client_host = request.client.host if request.client else "default-client"
    return client_host


@app.get("/")
async def root():
    return {
        "service": "JimRouteLLM Hybrid Proxy",
        "version": "1.0.0",
        "classifier": "ModernBERT-Large (395M, 8k context, INT8)",
        "cloud_provider": "Google AI Studio (Gemini)" if settings.is_gemini_configured() else "None",
        "lan_nodes_count": len(settings.lan_nodes),
        "lan_nodes": [n.to_dict() for n in settings.lan_nodes],
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/v1/models")
async def list_models():
    """Return all routable virtual models and active LAN / Cloud models."""
    models = [
        {"id": "routellm", "object": "model", "owned_by": "jimroutellm"},
        {"id": "jimroutellm", "object": "model", "owned_by": "jimroutellm"},
        {"id": "router-modernbert", "object": "model", "owned_by": "jimroutellm"},
        {"id": "router-modernbert-0.35", "object": "model", "owned_by": "jimroutellm"},
        {"id": "router-modernbert-0.45", "object": "model", "owned_by": "jimroutellm"},
        {"id": "router-modernbert-0.60", "object": "model", "owned_by": "jimroutellm"},
    ]

    # Cloud Models
    if settings.is_gemini_configured():
        models.extend([
            {"id": settings.gemini_pro_model, "object": "model", "owned_by": "google"},
            {"id": settings.gemini_flash_model, "object": "model", "owned_by": "google"},
        ])
    if settings.is_anthropic_configured():
        models.extend([
            {"id": settings.claude_sonnet_model, "object": "model", "owned_by": "anthropic"},
            {"id": settings.claude_opus_model, "object": "model", "owned_by": "anthropic"},
        ])

    # Distinct LAN Nodes
    for node in node_manager.get_all_nodes():
        for m in node.active_models:
            models.append({
                "id": m,
                "object": "model",
                "owned_by": f"lan-node-{node.id}",
                "base_url": node.base_url
            })

    return {"object": "list", "data": models}


@app.get("/v1/nodes/status")
async def get_nodes_status():
    """Status endpoint showing connectivity of the 4 heterogeneous LAN nodes."""
    return {
        "nodes": [
            {
                "id": n.id,
                "name": n.name,
                "base_url": n.base_url,
                "primary_model": n.primary_model,
                "active_models": n.active_models,
                "capabilities": list(n.capabilities),
                "is_online": n.is_online,
            }
            for n in node_manager.get_all_nodes()
        ]
    }


@app.get("/v1/dashboard/stats")
async def get_dashboard_stats():
    """Routing statistics and recent decisions."""
    return {
        "stats": router.stats,
        "recent_decisions": [
            {
                "target": d.target,
                "model": d.model_name,
                "score": f"{d.score:.3f}" if d.score is not None else "N/A",
                "threshold": d.threshold,
                "snippet": d.prompt_snippet,
                "node_id": d.node_id,
                "api_base": d.api_base,
                "reason": d.reason,
                "timestamp": d.timestamp,
            }
            for d in list(router.history)[-20:]
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible Chat Completions endpoint.
    Performs ModernBERT-Large complexity analysis, sticky session resolution, and
    dispatches to the assigned LAN node or Gemini/Claude cloud backend.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' in request body")

    requested_model = body.get("model", "routellm")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens")
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    # Extract sticky session key
    session_key = extract_session_key(request, body, messages)

    # Route decision
    decision: RoutingDecision = router.decide(
        messages=messages,
        requested_model=requested_model,
        session_key=session_key
    )

    headers = {
        "X-RouteLLM-Target": decision.target,
        "X-RouteLLM-Model": decision.model_name,
        "X-RouteLLM-Score": f"{decision.score:.3f}" if decision.score is not None else "0.000",
        "X-RouteLLM-Node": decision.node_id or "cloud",
        "X-RouteLLM-Reason": decision.reason,
    }

    # Prepare LiteLLM arguments
    litellm_kwargs: Dict[str, Any] = {
        "model": decision.litellm_model,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }

    if max_tokens:
        litellm_kwargs["max_tokens"] = max_tokens

    # Dynamic MCP Tool Pruning (Prunes unused tool schemas to save prompt tokens & reduce TTFT)
    if settings.enable_mcp_routing and tools:
        active_domains = mcp_classifier.classify_conversation(messages)
        max_tools = settings.mcp_max_local_tools if decision.target == "local" else settings.mcp_max_cloud_tools
        prompt_text = router.extract_prompt_text(messages)
        pruned_tools, orig_cnt, pruned_cnt, tokens_saved = mcp_classifier.filter_tools(
            incoming_tools=tools,
            active_domains=active_domains,
            prompt=prompt_text,
            max_tools_per_domain=max_tools
        )
        if pruned_tools:
            litellm_kwargs["tools"] = pruned_tools
            if tool_choice:
                litellm_kwargs["tool_choice"] = tool_choice
        headers["X-RouteLLM-MCP-Domains"] = ",".join(active_domains) or "none"
        headers["X-RouteLLM-Tools-Original"] = str(orig_cnt)
        headers["X-RouteLLM-Tools-Pruned"] = str(pruned_cnt)
        headers["X-RouteLLM-Tokens-Saved"] = str(tokens_saved)
        if pruned_cnt > 0:
            logger.info(
                f"[MCP PRUNER] Pruned {pruned_cnt}/{orig_cnt} tools (saved ~{tokens_saved} tokens | active domains: {active_domains or ['none']})"
            )
    elif tools:
        litellm_kwargs["tools"] = tools
        if tool_choice:
            litellm_kwargs["tool_choice"] = tool_choice

    # Local LAN node routing parameters
    if decision.target == "local":
        litellm_kwargs["api_base"] = decision.api_base
        litellm_kwargs["api_key"] = "lm-studio"
        litellm_kwargs["custom_llm_provider"] = "openai"

    elif decision.target == "gemini":
        litellm_kwargs["api_key"] = settings.gemini_api_key

    elif decision.target == "claude":
        litellm_kwargs["api_key"] = settings.anthropic_api_key

    # -------------------------------------------------------------
    # Streaming Response (SSE)
    # -------------------------------------------------------------
    if stream:
        async def stream_generator():
            try:
                response = await litellm.acompletion(**litellm_kwargs)
                async for chunk in response:
                    chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming inference error ({decision.litellm_model}): {e}")
                err_payload = {
                    "error": {
                        "message": f"Inference failed on {decision.model_name} @ {decision.api_base or 'cloud'}: {str(e)}",
                        "type": "routellm_error",
                    }
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers=headers
        )

    # -------------------------------------------------------------
    # Non-Streaming JSON Response
    # -------------------------------------------------------------
    try:
        response = await litellm.acompletion(**litellm_kwargs)
        resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return JSONResponse(content=resp_dict, headers=headers)
    except Exception as e:
        logger.error(f"Inference error on {decision.model_name}: {e}")
        
        # Automatic fallback to secondary model if available
        if decision.fallback_models:
            for fallback in decision.fallback_models:
                try:
                    logger.info(f"Attempting fallback to {fallback}...")
                    litellm_kwargs["model"] = fallback
                    response = await litellm.acompletion(**litellm_kwargs)
                    resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
                    headers["X-RouteLLM-Fallback"] = fallback
                    return JSONResponse(content=resp_dict, headers=headers)
                except Exception as fb_err:
                    logger.warning(f"Fallback {fallback} failed: {fb_err}")

        raise HTTPException(
            status_code=502,
            detail=f"Inference error ({decision.model_name} @ {decision.api_base or 'cloud'}): {str(e)}",
            headers=headers
        )
