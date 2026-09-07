import os
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import deque

from jimroutellm_proxy.config import settings
from jimroutellm_proxy.classifier import classifier
from jimroutellm_proxy.nodes import node_manager, LANNode

logger = logging.getLogger("jimroutellm.router")


@dataclass
class RoutingDecision:
    target: str  # "local", "gemini", "claude"
    model_name: str
    litellm_model: str
    score: Optional[float]
    threshold: float
    router_name: str
    prompt_snippet: str
    node_id: Optional[str] = None
    api_base: Optional[str] = None
    is_vision: bool = False
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    fallback_models: List[str] = field(default_factory=list)


def has_image_content(messages: List[Dict[str, Any]]) -> bool:
    """Check if the messages contain any images (multimodal content)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and (part.get("type") in ["image_url", "image"] or "image_url" in part):
                    return True
    return False


class HybridRouter:
    def __init__(self):
        self.router_type = settings.router_type
        self.default_threshold = settings.routing_threshold
        self.history = deque(maxlen=100)
        self.session_nodes: Dict[str, str] = {}
        self.stats = {
            "total_requests": 0,
            "routed_to_local": 0,
            "routed_to_gemini": 0,
            "routed_to_claude": 0,
            "direct_local": 0,
            "direct_cloud": 0,
        }

    def extract_prompt_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract user prompt text for complexity evaluation."""
        if not messages:
            return ""

        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    return " ".join(parts)

        last_msg = messages[-1].get("content", "")
        return str(last_msg)

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total token count across all messages in the conversation."""
        chars = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                chars += len(content)
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        chars += len(p.get("text", ""))
        return max(1, chars // 4)

    def parse_requested_model(self, model_name: Optional[str]) -> tuple[Optional[str], Optional[float]]:
        """Parse model name for explicit router and threshold overrides (e.g. router-modernbert-0.45)."""
        if not model_name:
            return None, None

        match = re.match(r"^router-([a-zA-Z0-9_\-]+)-([0-9.]+)$", model_name)
        if match:
            r_name = match.group(1)
            try:
                thresh = float(match.group(2))
                return r_name, thresh
            except ValueError:
                pass

        return None, None

    def _format_gemini_model(self, model_name: str) -> str:
        if model_name.startswith(("gemini/", "vertex_ai/")):
            return model_name
        return f"gemini/{model_name}"

    def decide(
        self,
        messages: List[Dict[str, Any]],
        requested_model: Optional[str] = None,
        session_key: Optional[str] = None
    ) -> RoutingDecision:
        """
        Evaluate messages and determine whether to route to a specific LAN node,
        Gemini (Google AI Studio), or Claude (Anthropic).
        """
        prompt_text = self.extract_prompt_text(messages)
        approx_tokens = self.estimate_tokens(messages)
        snippet = prompt_text[:80].replace("\n", " ")
        is_vision = has_image_content(messages)

        req_model_lower = (requested_model or "").lower().strip()
        is_router_call = any(req_model_lower.startswith(p) for p in ["routellm", "router-", "route-", "jimroute"]) or req_model_lower in ["", "default"]

        self.stats["total_requests"] += 1

        # -------------------------------------------------------------
        # 1. Explicit Direct Overrides
        # -------------------------------------------------------------
        if not is_router_call:
            # 1.1 Direct Gemini Overrides
            if any(g in req_model_lower for g in ["gemini", "google"]):
                self.stats["direct_cloud"] += 1
                self.stats["routed_to_gemini"] += 1
                target_model = requested_model or settings.gemini_pro_model
                decision = RoutingDecision(
                    target="gemini",
                    model_name=target_model,
                    litellm_model=self._format_gemini_model(target_model),
                    score=1.0,
                    threshold=self.default_threshold,
                    router_name="direct_override",
                    prompt_snippet=snippet,
                    is_vision=is_vision,
                    reason=f"Explicit model request for {target_model}",
                    fallback_models=[self._format_gemini_model(settings.gemini_flash_model)],
                )
                self.history.append(decision)
                return decision

            # 1.2 Direct Claude Overrides (Piping Preserved)
            if any(c in req_model_lower for c in ["claude", "anthropic", "sonnet", "opus", "haiku", "fable"]):
                self.stats["direct_cloud"] += 1
                self.stats["routed_to_claude"] += 1
                chosen_claude = requested_model if "claude" in req_model_lower else settings.claude_sonnet_model
                decision = RoutingDecision(
                    target="claude",
                    model_name=chosen_claude,
                    litellm_model=f"anthropic/{chosen_claude}" if not chosen_claude.startswith("anthropic/") else chosen_claude,
                    score=1.0,
                    threshold=self.default_threshold,
                    router_name="direct_override",
                    prompt_snippet=snippet,
                    is_vision=is_vision,
                    reason=f"Explicit model request for Claude ({chosen_claude})",
                    fallback_models=[f"anthropic/{settings.claude_opus_model}"],
                )
                self.history.append(decision)
                return decision

            # 1.3 Direct Local Node / Model Overrides
            matched_node = node_manager.find_node_by_model(requested_model)
            if matched_node or any(l in req_model_lower for l in ["local", "lmstudio", "lm-studio", "qwen", "deepseek", "llama"]):
                self.stats["direct_local"] += 1
                self.stats["routed_to_local"] += 1
                
                selected_node = matched_node or node_manager.select_node(
                    requested_model=requested_model,
                    capability="vision" if is_vision else None,
                    session_key=session_key
                )
                target_model = requested_model if (requested_model and requested_model.lower() not in ["local", "lmstudio", "lm-studio"]) else selected_node.get_current_model()

                decision = RoutingDecision(
                    target="local",
                    model_name=target_model,
                    litellm_model=f"openai/{target_model}",
                    score=0.0,
                    threshold=self.default_threshold,
                    router_name="direct_override",
                    prompt_snippet=snippet,
                    node_id=selected_node.id,
                    api_base=selected_node.base_url,
                    is_vision=is_vision,
                    reason=f"Explicit local request routed to {selected_node.name} ({selected_node.base_url})",
                )
                self.history.append(decision)
                return decision

        # -------------------------------------------------------------
        # 2. ModernRoBERTa / ModernBERT NPU Complexity Evaluation
        # -------------------------------------------------------------
        router_override, threshold_override = self.parse_requested_model(requested_model)
        active_router = router_override or self.router_type
        threshold = threshold_override if threshold_override is not None else self.default_threshold

        score = classifier.calculate_complexity_score(prompt_text)
        dev_tag = "NPU" if classifier.npu_active else "CPU"

        # -------------------------------------------------------------
        # 3. Multimodal & Tiered Complexity Routing
        #    • If ALL_MODELS_SUPPORT_VISION is False:
        #      Images must route to dedicated vision model (preventing 400 errors on text-only models).
        #    • If ALL_MODELS_SUPPORT_VISION is True:
        #      Images follow the normal complexity score (easy image -> E2B, hard image -> 26B)
        #      without hogging the strong model.
        # -------------------------------------------------------------
        prior_node_id = self.session_nodes.get(session_key) if session_key else None
        prior_node = node_manager.nodes.get(prior_node_id) if prior_node_id else None
        prior_is_heavy = prior_node and ("heavy" in prior_node.capabilities or "hard" in prior_node.capabilities)

        if is_vision and not settings.all_models_support_vision:
            target_model = settings.local_vision_model
            selected_node = node_manager.find_node_by_model(target_model) or node_manager.select_node(
                capability="vision", session_key=session_key
            )
            reason = f"Multimodal image query (ALL_MODELS_SUPPORT_VISION=false) -> routed to dedicated vision model {target_model} ({selected_node.name})"
            fallback = [f"openai/{settings.local_hard_model}"] if target_model != settings.local_hard_model else []
        elif prior_is_heavy:
            target_model = prior_node.primary_model
            selected_node = prior_node
            task_type = "multimodal vision/thinking" if is_vision else "text/thinking"
            reason = f"Sticky session affinity ({session_key}) preserves {target_model} KV cache ({selected_node.name})"
            fallback = [f"openai/{settings.local_easy_model}"]
        elif score >= threshold:
            target_model = settings.local_hard_model
            selected_node = node_manager.find_node_by_model(target_model) or node_manager.select_node(
                capability="heavy", session_key=session_key
            )
            task_type = "multimodal vision/thinking" if is_vision else "text/thinking"
            reason = f"Hard {task_type} task (Score {score:.3f} >= {threshold:.2f} via {dev_tag} classifier) -> routed to {target_model} ({selected_node.name})"
            fallback = [f"openai/{settings.local_easy_model}"]
        else:
            target_model = settings.local_easy_model
            selected_node = node_manager.find_node_by_model(target_model) or node_manager.select_node(
                capability="easy", session_key=session_key
            )
            task_type = "multimodal vision/thinking" if is_vision else "text/thinking"
            reason = f"Easy {task_type} task (Score {score:.3f} < {threshold:.2f} via {dev_tag} classifier) -> routed to {target_model} ({selected_node.name})"
            fallback = [f"openai/{settings.local_hard_model}"]

        if session_key and selected_node:
            self.session_nodes[session_key] = selected_node.id
            if len(self.session_nodes) > 2000:
                self.session_nodes.pop(next(iter(self.session_nodes)))

        self.stats["routed_to_local"] += 1
        decision = RoutingDecision(
            target="local",
            model_name=target_model,
            litellm_model=f"openai/{target_model}",
            score=score,
            threshold=threshold,
            router_name=f"modernroberta-{dev_tag.lower()}",
            prompt_snippet=snippet,
            node_id=selected_node.id,
            api_base=selected_node.base_url,
            is_vision=is_vision,
            reason=reason,
            fallback_models=fallback,
        )
        self.history.append(decision)
        logger.info(f"[{dev_tag}] Score={score:.3f} (thresh={threshold:.2f}) -> {target_model} @ {selected_node.name} | '{snippet}'")
        return decision


router = HybridRouter()
