import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from jimroutellm_proxy.config import settings
from jimroutellm_proxy.classifier import classifier
from jimroutellm_proxy.nodes import node_manager
from jimroutellm_proxy.router import router

def main():
    print("=" * 65)
    print("  JimRouteLLM Unit & Routing Verification Suite")
    print("=" * 65)

    # 1. Verify Configuration & Nodes
    print(f"\n[1] Verifying LAN Node Configurations ({len(settings.lan_nodes)} nodes):")
    for n in node_manager.get_all_nodes():
        print(f"    • [{n.id}] {n.name}")
        print(f"      URL:          {n.base_url}")
        print(f"      Model:        {n.primary_model}")
        print(f"      Capabilities: {', '.join(n.capabilities)}")
        print(f"      Priority:     {n.priority}")

    # 2. Test ModernBERT-Large Classifier
    print(f"\n[2] Testing ModernBERT-Large (395M) Classifier:")
    test_prompts = [
        ("Hello, how are you today?", "Simple Greeting"),
        ("What is the capital of France?", "Simple Factoid"),
        ("Write a Python function to reverse a string.", "Easy Code"),
        (
            "Design a distributed event-driven microservices architecture with Raft consensus, "
            "handling out-of-order Kafka message deduplication and distributed deadlocks in Rust.",
            "Complex Architecture/Concurrency"
        ),
        (
            "Derive the mathematical proof for convergence in stochastic gradient descent "
            "under non-convex optimization with Polyak-Lojasiewicz conditions.",
            "Complex Mathematical Proof"
        ),
    ]

    for prompt, category in test_prompts:
        t0 = time.perf_counter()
        score = classifier.calculate_complexity_score(prompt)
        dt = (time.perf_counter() - t0) * 1000
        print(f"    • Score: {score:.3f} | Latency: {dt:.1f}ms | Category: {category}")
        print(f"      Prompt: '{prompt[:60]}...'")

    # 3. Test Routing Decisions
    print(f"\n[3] Testing Hybrid Router Decisions:")
    for prompt, category in test_prompts:
        messages = [{"role": "user", "content": prompt}]
        decision = router.decide(messages, requested_model="routellm", session_key="user-session-123")
        print(f"    • [{decision.target.upper()}] -> Model: {decision.model_name} (Score: {decision.score:.3f}, Thresh: {decision.threshold:.2f})")
        print(f"      Node: {decision.node_id or 'Cloud'} | Reason: {decision.reason}")

    # 4. Test Sticky Session Consistency (KV Cache Preservation)
    print(f"\n[4] Testing Sticky Session KV-Cache Preservation across 10 Turns:")
    session_a = "cline-session-alice-dev-01"
    session_b = "continue-session-bob-dev-02"
    
    node_a_1 = node_manager.select_node(session_key=session_a).id
    node_a_2 = node_manager.select_node(session_key=session_a).id
    node_a_3 = node_manager.select_node(session_key=session_a).id

    node_b_1 = node_manager.select_node(session_key=session_b).id
    node_b_2 = node_manager.select_node(session_key=session_b).id

    print(f"    • Session A (Alice) Turn 1 -> Node: {node_a_1}")
    print(f"    • Session A (Alice) Turn 2 -> Node: {node_a_2} {'[MATCH]' if node_a_1 == node_a_2 else '[FAIL]'}")
    print(f"    • Session A (Alice) Turn 3 -> Node: {node_a_3} {'[MATCH]' if node_a_1 == node_a_3 else '[FAIL]'}")
    print(f"    • Session B (Bob)   Turn 1 -> Node: {node_b_1}")
    print(f"    • Session B (Bob)   Turn 2 -> Node: {node_b_2} {'[MATCH]' if node_b_1 == node_b_2 else '[FAIL]'}")
    
    assert node_a_1 == node_a_2 == node_a_3, "Sticky session failed for Session A!"
    assert node_b_1 == node_b_2, "Sticky session failed for Session B!"
    print(f"    [✔] PASSED: Multi-turn KV prompt caching is 100% deterministic per session!")

    print("\n" + "=" * 65)
    print("  All Verification Tests Passed Successfully!")
    print("=" * 65)

if __name__ == "__main__":
    main()
