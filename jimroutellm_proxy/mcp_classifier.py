"""
RouteLLM MCP Domain & Tool Classifier
=====================================
Fast (<0.2ms) regex, entity, and keyword classifier that maps prompts to
specialized MCP domain clusters and prunes large tool catalogs to only the
relevant tool schemas while preserving core IDE/agent orchestration tools.
"""

import re
import json
import logging
from typing import Dict, List, Set, Any, Tuple, Optional
from jimroutellm_proxy.mcp_registry import mcp_registry, DOMAIN_TO_SERVERS, clean_openai_tool

logger = logging.getLogger("routellm.mcp_classifier")


def is_mcp_tool(tool_name: str) -> bool:
    """Determine if a tool originates from an MCP server catalog."""
    if not tool_name:
        return False
    # 1. Direct registry lookup across registered MCP tools
    if mcp_registry.get_server_for_tool(tool_name) is not None:
        return True
    
    # 2. Known MCP server prefixes
    tool_lower = tool_name.lower()
    mcp_prefixes = [
        "shipstation", "shopify", "recharge", "cropster", "listingmirror",
        "canto", "searxng", "mcp__", "mcp_", "browser_", "playwright",
    ]
    if any(tool_lower.startswith(p) for p in mcp_prefixes):
        return True

    return False


def is_core_agent_tool(tool_name: str) -> bool:
    """
    Determine if a tool is a built-in agent orchestration tool.
    Any tool NOT from an external MCP server is treated as a client agent tool.
    """
    if not tool_name:
        return False
    if is_mcp_tool(tool_name):
        return False
    return True


# Domain Keyword & Entity Definitions
DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "shipping": {
        "shipping", "shipment", "shipments", "ship", "shipped", "tracking", "track",
        "carrier", "carriers", "label", "labels", "usps", "ups", "fedex", "dhl",
        "fulfill", "fulfillment", "fulfillments", "warehouse", "warehouses", "package",
        "packages", "parcel", "delivery", "deliveries", "postage", "manifest", "manifests",
        "pickup", "pickups", "void_label", "create_label", "get_rates", "estimate_rates",
        "shipstation", "return_label", "packing_slip", "tote", "totes", "rate_shopper",
    },
    "ecommerce": {
        "shopify", "recharge", "order", "orders", "draft_order", "draft_orders",
        "customer", "customers", "refund", "refunds", "subscription", "subscriptions",
        "charge", "charges", "discount", "discounts", "coupon", "cancel_order",
        "payment", "payments", "checkout", "line_item", "metafield", "metafields",
        "product_variant", "product_variants", "store_credit", "portal_session",
        "tax_line", "billing", "invoice", "subscriber", "swap_subscription",
        "skip_charge", "pause_subscription",
    },
    "roasting_inventory": {
        "cropster", "listingmirror", "roast", "roasts", "roasting", "roasted",
        "green coffee", "green_lot", "green lot", "lot", "lots", "cupping",
        "cuppings", "cupping score", "blend", "blended", "roast profile", "roast curve",
        "physical analysis", "inventory", "inventory source", "inventory_source",
        "listing", "listings", "recipe", "recipes", "vendor", "vendors", "mrp",
        "bean", "beans", "coffee", "roaster", "roast_profile", "green_coffee",
    },
    "digital_assets": {
        "canto", "digital asset", "digital assets", "dam", "album", "albums",
        "asset", "assets", "image library", "tag asset", "smart tag", "download preset",
        "watermark", "preview url", "direct url", "photo metadata", "batch edit",
        "share link", "content detail", "media library", "brand asset", "assetview",
    },
    "web_search": {
        "search", "searxng_search", "search web", "search the web", "search online", "look up online", "google",
        "current events", "latest news", "latest documentation", "search query",
        "searxng", "browse the web", "search internet", "browse internet", "web_search",
        "look online", "find online", "search for", "live search", "lookup",
    },
    "browser": {
        "browser", "browse", "webpage", "website", "url", "urls", "navigate", "navigation",
        "click", "clicks", "button", "buttons", "screenshot", "screenshots", "dom", "html",
        "form", "forms", "fill", "input", "scroll", "hover", "chrome", "link", "links",
        "playwright", "headless", "tab", "tabs", "press_key", "snapshot", "css",
        "page", "pages", "select_option",
    },
}

# Regex Patterns for Identifiers
DOMAIN_PATTERNS: Dict[str, List[re.Pattern]] = {
    "shipping": [
        re.compile(r"\b1Z[0-9A-Z]{16}\b", re.IGNORECASE),  # UPS Tracking
        re.compile(r"\b9[2345]\d{20,24}\b"),               # USPS Tracking
        re.compile(r"\b\d{12,15}\b"),                       # FedEx / DHL Tracking
        re.compile(r"\b(track(ing)?|ship(ment)?)\s+#?[0-9A-Z]+\b", re.IGNORECASE),
        re.compile(r"\b(shipstation|carrier|postage|packing\s*slip)\b", re.IGNORECASE),
    ],
    "ecommerce": [
        re.compile(r"#\d{4,8}\b"),                          # Order numbers like #836318
        re.compile(r"\border\s*#?\d{4,8}\b", re.IGNORECASE),
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), # Email address
        re.compile(r"\b(shopify|recharge|refund|subscription|customer\s*id)\b", re.IGNORECASE),
    ],
    "roasting_inventory": [
        re.compile(r"\b(GL|RL|BL|PR|LOT)[-_#]?\d+\b", re.IGNORECASE), # Cropster Lots
        re.compile(r"\b(cropster|listingmirror|cupping\s*score|roast\s*(profile|curve))\b", re.IGNORECASE),
    ],
    "digital_assets": [
        re.compile(r"\b(canto|dam|album\s*id|asset\s*id|download\s*preset)\b", re.IGNORECASE),
    ],
    "web_search": [
        re.compile(r"\b(search\s*(the\s*)?(web|internet|online)|browse\s*(the\s*)?(web|internet|online)|look\s*up\s*online|searxng)\b", re.IGNORECASE),
        re.compile(r"\b(current\s*news|latest\s*news|news\s*regarding|search\s*for)\b", re.IGNORECASE),
    ],
    "browser": [
        re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE),
        re.compile(r"\b(browser|playwright|chrome|headless)\b", re.IGNORECASE),
        re.compile(r"\b(navigate(\s+to)?|visit|open\s+(url|website|page)|click(\s+on)?|take\s+(a\s+)?screenshot)\b", re.IGNORECASE),
        re.compile(r"\b(fill\s+(the\s+)?form|submit\s+(the\s+)?form|press\s+key)\b", re.IGNORECASE),
    ],
}


class MCPClassifier:
    """Classifies user prompts into active tool domains and filters tool catalogs."""

    def __init__(self):
        self.stats = {
            "total_classified": 0,
            "zero_tool_queries": 0,
            "domain_activations": {d: 0 for d in DOMAIN_KEYWORDS},
            "total_tools_pruned": 0,
            "total_tokens_saved": 0,
        }

    def is_core_agent_tool(self, tool_name: str) -> bool:
        return is_core_agent_tool(tool_name)

    def is_mcp_tool(self, tool_name: str) -> bool:
        return is_mcp_tool(tool_name)

    def classify_domains(self, prompt: str) -> List[str]:
        """Classify a prompt into one or more active domain keys."""
        self.stats["total_classified"] += 1
        prompt_lower = prompt.lower()
        active_domains: Set[str] = set()

        # 1. Regex Pattern Matching
        for domain, patterns in DOMAIN_PATTERNS.items():
            for pat in patterns:
                if pat.search(prompt):
                    active_domains.add(domain)
                    break

        # 2. Keyword Matching
        words = set(w for w in re.split(r'[^a-z0-9]+', prompt_lower) if w) | set(re.findall(r"\b[a-z0-9_-]+\b", prompt_lower))
        for domain, kws in DOMAIN_KEYWORDS.items():
            if domain in active_domains:
                continue
            for kw in kws:
                if " " in kw:
                    if kw in prompt_lower:
                        active_domains.add(domain)
                        break
                elif kw in words:
                    active_domains.add(domain)
                    break

        # 3. Check for pure coding signal
        if not active_domains:
            self.stats["zero_tool_queries"] += 1
            return []

        # Record metrics
        for d in active_domains:
            if d in self.stats["domain_activations"]:
                self.stats["domain_activations"][d] += 1

        return sorted(list(active_domains))

    def _get_tool_domain(self, tool: Dict[str, Any]) -> str:
        """Resolve domain of an individual tool schema."""
        tool_fn = tool.get("function", {}) if "function" in tool else tool
        tool_name = tool_fn.get("name", "").lower()
        tool_desc = tool_fn.get("description", "").lower()

        # 0. Core Agent Tools (NEVER PRUNED)
        if is_core_agent_tool(tool_name):
            return "core_agent"

        # 1. Check internal metadata if present
        if "_domain" in tool:
            return tool["_domain"]

        # 2. Check registry index
        reg_domain = mcp_registry.get_domain_for_tool(tool_name)
        if reg_domain:
            return reg_domain

        # 3. Substring & Prefix matching
        if any(p in tool_name for p in ["playwright", "browser_"]):
            return "browser"
        if any(p in tool_name for p in ["searxng", "web_search"]):
            return "web_search"
        if any(tool_name.startswith(p) for p in ["shipstation", "tracking", "carrier", "label", "warehouse"]):
            return "shipping"
        if any(tool_name.startswith(p) for p in ["shopify", "recharge", "order", "customer", "refund", "subscription"]):
            return "ecommerce"
        if any(tool_name.startswith(p) for p in ["cropster", "listingmirror", "roast", "cupping", "inventory"]):
            return "roasting_inventory"
        if any(tool_name.startswith(p) for p in ["canto", "album", "asset"]):
            return "digital_assets"

        # 4. Description heuristic matching
        for domain, kws in DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw in tool_name or (len(kw) > 4 and kw in tool_desc):
                    return domain

        return "custom"

    def classify_conversation(self, messages: List[Dict[str, Any]]) -> List[str]:
        """
        Classify active domains across the conversation history:
        1. Inspect recent user prompts (last 2 user messages).
        2. Inspect recent assistant tool calls (maintains continuity across turns).
        """
        active_domains: Set[str] = set()

        # 1. Inspect recent user prompts
        user_msgs = [m for m in messages if m.get("role") == "user"][-2:]
        for u in user_msgs:
            content = u.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        parts.append(p.get("text", ""))
                content = " ".join(parts)
            if isinstance(content, str) and content:
                for d in self.classify_domains(content):
                    active_domains.add(d)

        # 2. Inspect recent assistant tool calls
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"][-2:]
        for a in assistant_msgs:
            tool_calls = a.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    if fn_name:
                        domain = self._get_tool_domain({"function": {"name": fn_name}})
                        if domain and domain not in ("core_agent", "custom"):
                            active_domains.add(domain)

        return sorted(list(active_domains))

    def rank_and_limit_tools(
        self, tools: List[Dict[str, Any]], prompt: str, max_tools: int = 12
    ) -> List[Dict[str, Any]]:
        """Rank tools by lexical & semantic relevance to the prompt with stemming and return the top-K."""
        if not tools or len(tools) <= max_tools:
            return [clean_openai_tool(t) for t in tools]

        raw_words = set(re.findall(r"\b[a-z0-9_-]+\b", prompt.lower()))
        prompt_stems = set(w.rstrip("s") for w in raw_words if len(w) > 2) | raw_words
        
        scored_tools: List[Tuple[float, Dict[str, Any]]] = []
        for tool in tools:
            tool_fn = tool.get("function", {}) if "function" in tool else tool
            name = tool_fn.get("name", "").lower()
            desc = tool_fn.get("description", "").lower()
            params = tool_fn.get("parameters", {}).get("properties", {})
            param_names = set(p.lower() for p in params.keys())

            score = 0.0
            # Name match (high weight with stemming)
            name_raw_tokens = set(re.split(r"[-_]", name))
            name_stems = set(t.rstrip("s") for t in name_raw_tokens if len(t) > 2) | name_raw_tokens
            
            score += len(name_stems & prompt_stems) * 6.0
            
            # Substring match in name
            for pw in prompt_stems:
                if len(pw) > 3 and pw in name:
                    score += 4.0

            # Boost primary retrieval tools (get_orders, list_orders, search, navigate)
            if any(ret in name for ret in ["get-order", "get_order", "list_order", "list-order", "search", "navigate", "find"]):
                score += 5.0

            # Description match
            desc_tokens = set(re.findall(r"\b[a-z0-9_-]+\b", desc))
            desc_stems = set(d.rstrip("s") for d in desc_tokens if len(d) > 2) | desc_tokens
            score += len(desc_stems & prompt_stems) * 1.5

            # Parameter names match
            param_stems = set(p.rstrip("s") for p in param_names if len(p) > 2) | param_names
            score += len(param_stems & prompt_stems) * 2.0

            scored_tools.append((score, tool))

        # Sort descending by score
        scored_tools.sort(key=lambda x: x[0], reverse=True)
        top_tools = [clean_openai_tool(t[1]) for t in scored_tools[:max_tools]]
        return top_tools

    def filter_tools(
        self,
        incoming_tools: List[Dict[str, Any]],
        active_domains: List[str],
        prompt: Optional[str] = None,
        max_tools_per_domain: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int, int, int]:
        """
        Filter incoming tools down to active domains while preserving core agent tools.
        Returns: (pruned_tools, original_count, pruned_count, estimated_tokens_saved)
        """
        if not incoming_tools:
            return [], 0, 0, 0

        original_count = len(incoming_tools)
        core_tools: List[Dict[str, Any]] = []
        mcp_tools: List[Dict[str, Any]] = []
        stripped_chars = 0
        active_domain_set = set(active_domains)

        for tool in incoming_tools:
            domain = self._get_tool_domain(tool)
            if domain == "core_agent":
                core_tools.append(clean_openai_tool(tool))
            elif domain in active_domain_set:
                mcp_tools.append(tool)
            else:
                stripped_chars += len(json.dumps(tool))

        # Top-K ranking for MCP tools if above budget
        if max_tools_per_domain and prompt and len(mcp_tools) > max_tools_per_domain:
            mcp_tools = self.rank_and_limit_tools(mcp_tools, prompt, max_tools=max_tools_per_domain)
        else:
            mcp_tools = [clean_openai_tool(t) for t in mcp_tools]

        kept_tools = core_tools + mcp_tools
        pruned_count = original_count - len(kept_tools)
        tokens_saved = max(0, stripped_chars // 4)

        self.stats["total_tools_pruned"] += pruned_count
        self.stats["total_tokens_saved"] += tokens_saved

        return kept_tools, original_count, pruned_count, tokens_saved

    def get_stats(self) -> Dict[str, Any]:
        """Return classification and pruning statistics."""
        return dict(self.stats)


# Global singleton instance
mcp_classifier = MCPClassifier()
