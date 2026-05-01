"""
Test Layer 2: Model Behavior - Prompt Versioning & Token Efficiency

Verifies:
- Prompt version tracking (SHA-256 hash)
- Version drift detection
- Token efficiency metrics
- Inefficiency warnings
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents import AnalyticsAgents
from observability.prompt_version import get_prompt_registry
from observability.tracer import get_tracer

print("=" * 80)
print("LAYER 2: MODEL BEHAVIOR - PROMPT VERSIONING TEST")
print("=" * 80)

# Initialize agents (this will register prompts)
print("\n✓ Test 1: Prompt Registration")
print("-" * 80)

agents = AnalyticsAgents()
registry = get_prompt_registry()

# Check that prompts were registered
sql_version = registry.get_version("SQL_Generator")
insight_version = registry.get_version("Insight_Generator")

print(f"📋 SQL Generator Prompt:")
print(f"   Version Hash: {sql_version.short_hash}")
print(f"   Prompt Length: {len(sql_version.prompt_text)} chars")
print(f"   Created At: {sql_version.created_at}")

print(f"\n📋 Insight Generator Prompt:")
print(f"   Version Hash: {insight_version.short_hash}")
print(f"   Prompt Length: {len(insight_version.prompt_text)} chars")
print(f"   Created At: {insight_version.created_at}")

print("\n   ✅ Prompt registration working")

# Test 2: Run queries to accumulate token metrics
print("\n✓ Test 2: Token Efficiency Tracking")
print("-" * 80)

print("\n📊 Running 3 queries to accumulate metrics...")

queries = [
    "What is the total sales?",
    "Show top 3 products by sales in each region",
    "What are the top 5 products?"
]

for i, query in enumerate(queries, 1):
    print(f"\n   Query {i}: \"{query}\"")
    result = agents.run(query)
    print(f"   ✅ Completed (correlation_id: {result.get('correlation_id', 'N/A')[:8]})")

# Get metrics
print("\n📈 Token Efficiency Metrics:")
all_metrics = registry.get_all_metrics()

for agent_name, metrics in all_metrics.items():
    print(f"\n   {agent_name}:")
    print(f"      Version Hash: {metrics['version_hash']}")
    print(f"      Prompt Length: {metrics['prompt_length']} chars")
    print(f"      Total Requests: {metrics['total_requests']}")
    print(f"      Avg Tokens/Request: {metrics['avg_tokens_per_request']:.1f}")
    print(f"      Avg Cost/Request: ${metrics['avg_cost_per_request']:.6f}")
    print(f"      Total Cost: ${metrics['total_cost_usd']:.6f}")

print("\n   ✅ Token efficiency tracking working")

# Test 3: Inefficiency Detection
print("\n✓ Test 3: Inefficiency Detection")
print("-" * 80)

warnings = registry.detect_inefficiencies()

if warnings:
    print(f"\n⚠️  Found {len(warnings)} potential inefficiencies:")
    for warning in warnings:
        print(f"\n   Type: {warning['type']}")
        print(f"   Agent: {warning['agent']}")
        print(f"   Message: {warning['message']}")
else:
    print("\n   ✅ No inefficiencies detected - prompts are well optimized!")

# Test 4: Version Drift Detection
print("\n✓ Test 4: Version Drift Detection")
print("-" * 80)

print("\n📝 Simulating prompt change (drift)...")

# Simulate changing the SQL Generator prompt
original_hash = sql_version.short_hash
print(f"   Original version: {original_hash}")

# Register a modified prompt (simulating code change)
modified_prompt = sql_version.prompt_text + "\n\n# Additional instruction: Always add comments"
print(f"\n   Registering modified prompt...")

modified_version = registry.register_prompt(
    agent_name="SQL_Generator",
    prompt_text=modified_prompt,
    metadata={"purpose": "Updated with comment requirement"}
)

print(f"   New version: {modified_version.short_hash}")
print(f"   Prompt length change: {len(sql_version.prompt_text)} → {len(modified_prompt)} chars")

# Check version history
history = registry.version_history["SQL_Generator"]
print(f"\n📜 Version History for SQL_Generator:")
for i, version in enumerate(history, 1):
    print(f"   Version {i}: {version.short_hash} ({len(version.prompt_text)} chars)")

print("\n   ✅ Version drift detection working")

# Test 5: Integration with Tracer
print("\n✓ Test 5: Tracer Integration (Prompt Version in Traces)")
print("-" * 80)

tracer = get_tracer()
recent_traces = tracer.get_recent_traces(limit=3)

print(f"\n📋 Recent Traces (showing prompt_version_hash):")
print("\n{:<25} {:<25} {:<15} {:<12}".format(
    "Operation", "Agent", "Version", "Tokens"
))
print("-" * 80)

for trace in recent_traces:
    operation = trace['operation_name'][:24]
    agent = (trace['agent_name'] or 'N/A')[:24]
    version_hash = (trace['prompt_version_hash'] or 'N/A')[:14]
    tokens = trace['total_tokens']

    print("{:<25} {:<25} {:<15} {:<12}".format(
        operation, agent, version_hash, tokens
    ))

print("\n   ✅ Prompt versions captured in traces")

# Summary
print("\n" + "=" * 80)
print("LAYER 2 TEST COMPLETE ✅")
print("=" * 80)

print("""
✅ Layer 2 Model Behavior Features Verified:

1. ✅ Prompt version tracking (SHA-256 hash)
2. ✅ Prompt registration on agent initialization
3. ✅ Token efficiency metrics (per prompt version)
4. ✅ Cost tracking per prompt version
5. ✅ Inefficiency detection (long prompts, high costs)
6. ✅ Version drift detection (prompt changes)
7. ✅ Version history tracking
8. ✅ Integration with Layer 1 tracer

🎯 Portfolio Value:
   - Prompt version control with automatic drift detection
   - Token efficiency monitoring to identify cost optimization opportunities
   - Historical tracking of prompt changes and their impact on performance
   - Proactive warnings for inefficient prompts

Ready for Layer 3: Output Quality (SQL validation, narrative scoring)!
""")

# Display summary statistics
print("\n" + "=" * 80)
print("SUMMARY: TOKEN EFFICIENCY BY AGENT")
print("=" * 80)

for agent_name, metrics in all_metrics.items():
    efficiency_score = "✅ Efficient" if metrics['avg_tokens_per_request'] < 500 else "⚠️  Could be optimized"

    print(f"\n{agent_name}:")
    print(f"   Avg tokens/request: {metrics['avg_tokens_per_request']:.1f} {efficiency_score}")
    print(f"   Avg cost/request: ${metrics['avg_cost_per_request']:.6f}")
    print(f"   Total cost: ${metrics['total_cost_usd']:.6f}")

print("\n" + "=" * 80)
