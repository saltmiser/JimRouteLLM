"""
Heterogeneous Multi-Node LAN Manager & Sticky KV Cache Router
============================================================
Manages distinct LAN nodes with varying hardware capacities and model assignments.
Implements consistent hashing per session/conversation to preserve LM Studio KV
prompt caches across multi-turn chats (fixing the naive round-robin bug).
"""

import hashlib
import logging
import asyncio
from typing import List, Dict, Any, Optional
import requests

from jimroutellm_proxy.config import settings, NodeConfig

logger = logging.getLogger("jimroutellm.nodes")


class LANNode:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.id = config.id
        self.name = config.name
        self.base_url = config.base_url
        self.primary_model = config.primary_model
        self.capabilities = set(config.capabilities)
        self.priority = config.priority
        self.is_online = True
        self.active_models: List[str] = [config.primary_model]
        self.last_health_check = 0.0

    def check_health(self) -> bool:
        """Poll the node's /v1/models endpoint to verify connectivity and active models."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                models_data = data.get("data", [])
                if models_data:
                    self.active_models = [m.get("id") for m in models_data if m.get("id")]
                self.is_online = True
                return True
        except Exception:
            pass
        
        self.is_online = False
        return False

    def get_current_model(self) -> str:
        """Return the currently loaded model on this node or primary configured model."""
        if self.active_models:
            return self.active_models[0]
        return self.primary_model


class NodeManager:
    def __init__(self):
        self.nodes: Dict[str, LANNode] = {}
        for cfg in settings.lan_nodes:
            self.nodes[cfg.id] = LANNode(cfg)
        self._round_robin_idx = 0

    def get_all_nodes(self) -> List[LANNode]:
        return list(self.nodes.values())

    def get_online_nodes(self) -> List[LANNode]:
        online = [n for n in self.nodes.values() if n.is_online]
        return online if online else list(self.nodes.values())

    def find_node_by_model(self, model_name: str) -> Optional[LANNode]:
        """Find the specific LAN node configured for or currently running the requested model."""
        model_lower = model_name.lower().strip()
        
        # 1. Exact or substring match on primary model
        for node in self.get_online_nodes():
            if model_lower in node.primary_model.lower() or node.primary_model.lower() in model_lower:
                return node

        # 2. Match in active loaded models
        for node in self.get_online_nodes():
            for m in node.active_models:
                if model_lower in m.lower() or m.lower() in model_lower:
                    return node

        return None

    def select_node(
        self,
        requested_model: Optional[str] = None,
        capability: Optional[str] = None,
        session_key: Optional[str] = None
    ) -> LANNode:
        """
        Selects the optimal LAN node based on:
        1. Model-to-node mapping (if a specific model is targeted).
        2. Hardware capability (e.g., 'vision', 'heavy', 'code').
        3. Consistent hashing on `session_key` to keep multi-turn chats sticky to one node's KV cache.
        4. Healthy node fallback.
        """
        online_nodes = self.get_online_nodes()
        if not online_nodes:
            # Fallback to the first configured node
            return list(self.nodes.values())[0]

        # 1. Direct Model Assignment
        if requested_model:
            matched_node = self.find_node_by_model(requested_model)
            if matched_node and matched_node.is_online:
                return matched_node

        # 2. Capability Filtering
        candidate_nodes = online_nodes
        if capability:
            cap_filtered = [n for n in online_nodes if capability.lower() in n.capabilities]
            if cap_filtered:
                candidate_nodes = cap_filtered

        # 3. Sticky Session Hashing (Preserve LM Studio KV Cache)
        if session_key and candidate_nodes:
            hash_val = int(hashlib.md5(session_key.encode("utf-8")).hexdigest(), 16)
            selected = candidate_nodes[hash_val % len(candidate_nodes)]
            return selected

        # 4. Priority / Round Robin Fallback
        candidate_nodes.sort(key=lambda n: n.priority)
        selected = candidate_nodes[self._round_robin_idx % len(candidate_nodes)]
        self._round_robin_idx += 1
        return selected

    async def background_health_check_loop(self):
        """Periodically ping nodes to update online status and active models."""
        while True:
            try:
                for node in self.nodes.values():
                    await asyncio.to_thread(node.check_health)
            except Exception as e:
                logger.debug(f"Error in background node health check: {e}")
            await asyncio.sleep(15)


node_manager = NodeManager()
