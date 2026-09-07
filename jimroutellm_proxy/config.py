import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger("jimroutellm.config")

# Load .env file from project root
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    load_dotenv(override=True)


class NodeConfig:
    def __init__(
        self,
        id: str,
        name: str,
        base_url: str,
        primary_model: str,
        capabilities: Optional[List[str]] = None,
        priority: int = 1,
        enabled: bool = True
    ):
        self.id = id.strip()
        self.name = name.strip()
        self.base_url = base_url.rstrip("/")
        self.primary_model = primary_model.strip()
        self.capabilities = [c.lower().strip() for c in (capabilities or ["text"])]
        self.priority = priority
        self.enabled = enabled

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "base_url": self.base_url,
            "primary_model": self.primary_model,
            "capabilities": self.capabilities,
            "priority": self.priority,
            "enabled": self.enabled,
        }


class Settings:
    def __init__(self):
        # --- Server Settings ---
        self.proxy_host: str = os.getenv("PROXY_HOST", "0.0.0.0").strip()
        self.proxy_port: int = int(os.getenv("PROXY_PORT", "8000"))
        self.proxy_api_key: Optional[str] = os.getenv("PROXY_API_KEY", "").strip() or None

        # --- Primary Cloud Provider: Google AI Studio (Gemini) ---
        self.default_cloud_provider: str = os.getenv("DEFAULT_CLOUD_PROVIDER", "gemini").strip().lower()
        self.gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY", "").strip() or None
        self.gemini_pro_model: str = os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro").strip()
        self.gemini_flash_model: str = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash").strip()
        self.gemini_tier_threshold: float = float(os.getenv("GEMINI_TIER_THRESHOLD", "0.60"))

        # --- Secondary Cloud Provider: Anthropic Claude (Preserved) ---
        self.anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", "").strip() or None
        self.claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5").strip()
        self.claude_strong_model: str = os.getenv("CLAUDE_STRONG_MODEL", "claude-fable-5").strip()
        self.claude_opus_model: str = os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-5").strip()
        self.claude_sonnet_model: str = os.getenv("CLAUDE_SONNET_MODEL", "claude-sonnet-5").strip()
        self.claude_opus_threshold: float = float(os.getenv("CLAUDE_OPUS_THRESHOLD", "0.65"))

        # --- Router & ModernBERT-Large Classifier ---
        self.router_type: str = os.getenv("ROUTER_TYPE", "modernbert").strip().lower()
        self.routing_threshold: float = float(os.getenv("ROUTING_THRESHOLD", "0.28"))
        self.modernbert_model_id: str = os.getenv("MODERNBERT_MODEL_ID", "answerdotai/ModernBERT-large").strip()
        self.modernbert_use_onnx: bool = os.getenv("MODERNBERT_USE_ONNX", "true").lower() in ("true", "1", "yes")
        self.classifier_device: str = os.getenv("CLASSIFIER_DEVICE", "cuda").strip().lower()
        self.classifier_max_tokens: int = int(os.getenv("CLASSIFIER_MAX_TOKENS", "2048"))

        # --- Heterogeneous LAN Cluster Configuration ---
        self.lan_nodes: List[NodeConfig] = self._load_lan_nodes()

        # --- Local Target Models (Gemma 4 E2B & Gemma 4 26B-A4B) ---
        self.local_easy_model: str = os.getenv("LOCAL_EASY_MODEL", "google/gemma-4-e2b").strip()
        self.local_hard_model: str = os.getenv("LOCAL_HARD_MODEL", "google/gemma-4-26b-a4b-qat").strip()
        self.local_vision_model: str = os.getenv("LOCAL_VISION_MODEL", "google/gemma-4-26b-a4b-qat").strip()
        self.local_lm_studio_url: str = os.getenv("LOCAL_LM_STUDIO_URL", "http://127.0.0.1:1234/v1").rstrip("/")

        # Dedicated vision shortcut fallback
        self.lm_studio_vision_url: str = os.getenv("LM_STUDIO_VISION_URL", "http://127.0.0.1:1234/v1").rstrip("/")
        self.lm_studio_vision_model: str = os.getenv("LM_STUDIO_VISION_MODEL", "google/gemma-4-26b-a4b-qat").strip()
        self.all_models_support_vision: bool = os.getenv("ALL_MODELS_SUPPORT_VISION", "true").lower() in ("true", "1", "yes")

        # --- MCP Scaffolding Settings ---
        self.enable_mcp_routing: bool = os.getenv("ENABLE_MCP_ROUTING", "false").lower() in ("true", "1", "yes")
        self.mcp_routing_mode: str = os.getenv("MCP_ROUTING_MODE", "filter").strip().lower()
        self.mcp_servers_file: str = os.getenv("MCP_SERVERS_FILE", "mcpServers.json").strip()
        self.mcp_cache_ttl: int = int(os.getenv("MCP_CACHE_TTL_SECONDS", "300"))
        self.mcp_max_local_tools: int = int(os.getenv("MCP_MAX_LOCAL_TOOLS", "16"))
        self.mcp_max_cloud_tools: int = int(os.getenv("MCP_MAX_CLOUD_TOOLS", "30"))

        # Export keys to process environment for LiteLLM
        if self.gemini_api_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key

    def _load_lan_nodes(self) -> List[NodeConfig]:
        """Parse structured LAN node definitions from JSON or fallback to defaults."""
        raw_json = os.getenv("LAN_NODES", "").strip()
        nodes: List[NodeConfig] = []
        if raw_json:
            try:
                data = json.loads(raw_json)
                for item in data:
                    nodes.append(
                        NodeConfig(
                            id=item.get("id", f"node-{len(nodes)+1}"),
                            name=item.get("name", f"Node {len(nodes)+1}"),
                            base_url=item.get("base_url", "http://127.0.0.1:1234/v1"),
                            primary_model=item.get("primary_model", "local-model"),
                            capabilities=item.get("capabilities", ["text"]),
                            priority=item.get("priority", 1),
                            enabled=item.get("enabled", True),
                        )
                    )
                logger.info(f"Loaded {len(nodes)} distinct LAN nodes from configuration.")
                return nodes
            except Exception as e:
                logger.error(f"Failed to parse LAN_NODES JSON: {e}. Falling back to single local node.")

        # Default single node fallback
        return [
            NodeConfig(
                id="local-zbook",
                name="HP ZBook (Local)",
                base_url="http://127.0.0.1:1234/v1",
                primary_model="qwen2.5-coder-7b-instruct",
                capabilities=["code", "text"],
                priority=1,
            )
        ]

    def is_gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and not self.gemini_api_key.startswith("your_google_ai"))

    def is_anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key and not self.anthropic_api_key.startswith("your_anthropic"))


settings = Settings()
