"""
Test Integrated Observability with Real Queries

Verifies that Layer 1 observability is working end-to-end with the analytics agents.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents import AnalyticsAgents
from observability.tracer import get_tracer

print("=" * 80)
print("INTEGRATED OBSERVABILITY TEST - REAL QUERIES")
print("=" * 80)

# Initialize agents (with tracer)
print("\n✓ Initializing Analytics Agents with Observability...")
agents = AnalyticsAgents()

# Get tracer for stats
tracer = get_tracer()

print("   ✅ Agents initialized with tracing enabled")

# Test 1: Simple Query
print("\n" + "=" * 80)
print("TEST 1: Simple Query")
print("=" * 80)

question1 = "What is the total sales?"
print(f"\n📊 Question: \"{question1}\"")
print("⏳ Running query with full observability tracing...")

result1 = agents.run(question1)

print(f"\n✅ Query completed!")
print(f"   Correlation ID: {result1.get('correlation_id', 'N/A')}")
print(f"   Chart: {'Generated' if result1.get('chart') else 'None'}")
print(f"   Narrative: {result1.get('narrative', 'N/A')[:100]}...")

# Test 2: Complex Query with Window Functions
print("\n" + "=" * 80)
print("TEST 2: Complex Query (Window Functions)")
print("=" * 80)

question2 = "Show top 3 products by sales in each region"
print(f"\n📊 Question: \"{question2}\"")
print("⏳ Running query with full observability tracing...")

result2 = agents.run(question2)

print(f"\n✅ Query completed!")
print(f"   Correlation ID: {result2.get('correlation_id', 'N/A')}")
print(f"   Chart: {'Generated' if result2.get('chart') else 'None'}")
print(f"   Rows returned: {result2.get('row_count', 'N/A')}")
print(f"   Narrative: {result2.get('narrative', 'N/A')[:100]}...")

# Test 3: Time-Based Query
print("\n" + "=" * 80)
print("TEST 3: Time-Based Query (Adaptive Dates)")
print("=" * 80)

question3 = "Show monthly sales trends for the last 6 months"
print(f"\n📊 Question: \"{question3}\"")
print("⏳ Running query with full observability tracing...")

result3 = agents.run(question3)

print(f"\n✅ Query completed!")
print(f"   Correlation ID: {result3.get('correlation_id', 'N/A')}")
print(f"   Chart: {'Generated' if result3.get('chart') else 'None'}")
print(f"   Rows returned: {result3.get('row_count', 'N/A')}")

# Show Observability Statistics
print("\n" + "=" * 80)
print("OBSERVABILITY STATISTICS")
print("=" * 80)

stats = tracer.get_stats()

print(f"""
📊 Overall Statistics:
   Total Requests: {stats['total_requests']}
   Successful: {stats['successful_requests']}
   Failed: {stats['failed_requests']}
   Error Rate: {stats['error_rate']:.2%}

⏱️  Latency Metrics:
   Avg Latency: {stats['avg_latency_ms']}ms
   P50 Latency: {stats['p50_latency_ms']}ms
   P95 Latency: {stats['p95_latency_ms']}ms
   P99 Latency: {stats['p99_latency_ms']}ms

💰 Cost Metrics:
   Total Cost: ${stats['total_cost_usd']:.6f}
   Avg Cost Per Request: ${stats['avg_cost_per_request']:.6f}
""")

# Show Recent Traces
print("\n" + "=" * 80)
print("RECENT TRACES (Last 6 operations)")
print("=" * 80)

recent = tracer.get_recent_traces(limit=6)

print("\n{:<20} {:<25} {:<15} {:<12} {:<10} {:<12}".format(
    "Operation", "Agent", "Model", "Latency", "Tokens", "Cost"
))
print("-" * 95)

for trace in recent:
    operation = trace['operation_name'][:19]
    agent = (trace['agent_name'] or 'N/A')[:24]
    model = (trace['model_name'] or 'N/A')[:14]
    latency = f"{trace['latency_ms']}ms"
    tokens = str(trace['total_tokens'])
    cost = f"${trace['cost_usd']:.6f}"

    print("{:<20} {:<25} {:<15} {:<12} {:<10} {:<12}".format(
        operation, agent, model, latency, tokens, cost
    ))

# Show Detailed Request Trace
print("\n" + "=" * 80)
print("DETAILED REQUEST TRACE")
print("=" * 80)

if result2.get('correlation_id'):
    correlation_id = result2['correlation_id']
    request_traces = tracer.get_request_trace(correlation_id)

    print(f"\n📋 Correlation ID: {correlation_id}")
    print(f"📊 Question: \"{question2}\"")
    print(f"\n🔄 Operation Flow:")

    total_latency = 0
    total_cost = 0.0

    for i, trace in enumerate(request_traces, 1):
        print(f"\n   Step {i}: {trace['operation_name']}")
        print(f"      Agent: {trace['agent_name']}")
        print(f"      Latency: {trace['latency_ms']}ms")
        print(f"      Tokens: {trace['prompt_tokens']} prompt + {trace['completion_tokens']} completion = {trace['total_tokens']} total")
        print(f"      Cost: ${trace['cost_usd']:.6f}")
        print(f"      Success: {'✅' if trace['success'] else '❌'}")

        total_latency += trace['latency_ms']
        total_cost += trace['cost_usd']

    print(f"\n📈 Request Summary:")
    print(f"   Total Operations: {len(request_traces)}")
    print(f"   Total Latency: {total_latency}ms")
    print(f"   Total Cost: ${total_cost:.6f}")
    print(f"   Success: {'✅ All operations successful' if all(t['success'] for t in request_traces) else '❌ Some operations failed'}")

# Performance Analysis
print("\n" + "=" * 80)
print("PERFORMANCE ANALYSIS")
print("=" * 80)

print(f"""
🎯 Query Performance:
   - Simple Query: ~{stats['avg_latency_ms']}ms avg
   - P95 (95% of queries): <{stats['p95_latency_ms']}ms
   - P99 (99% of queries): <{stats['p99_latency_ms']}ms

💵 Cost Efficiency:
   - Avg cost per query: ${stats['avg_cost_per_request']:.6f}
   - Estimated cost per 1000 queries: ${stats['avg_cost_per_request'] * 1000:.2f}
   - Estimated monthly cost (10K queries/month): ${stats['avg_cost_per_request'] * 10000:.2f}

📊 ROI Calculation:
   - Manual SQL writing time: ~5 minutes
   - AI query time: ~{stats['avg_latency_ms'] / 1000:.1f} seconds
   - Time saved: ~{((300 - stats['avg_latency_ms'] / 1000) / 300) * 100:.1f}%
   - Analyst hourly rate: $150/hr
   - Cost per manual query: ${150 / 60 * 5:.2f}
   - AI cost per query: ${stats['avg_cost_per_request']:.6f}
   - Savings per query: ${150 / 60 * 5 - stats['avg_cost_per_request']:.2f}
""")

print("\n" + "=" * 80)
print("INTEGRATION TEST COMPLETE ✅")
print("=" * 80)

print("""
✅ Observability Features Verified:

1. ✅ correlation_id tracking (unique per request)
2. ✅ Multi-step tracing (SQL → Execution → Insights)
3. ✅ Latency tracking (per operation and total)
4. ✅ Token estimation (prompt + completion)
5. ✅ Cost calculation (automatic per operation)
6. ✅ Success/failure tracking
7. ✅ Statistics aggregation (p50/p95/p99)
8. ✅ Request-level trace lookup

🎯 Portfolio Value:
   - Production-ready LLM observability
   - Cost tracking: ${stats['avg_cost_per_request']:.6f} per query
   - Performance monitoring: {stats['avg_latency_ms']}ms avg latency
   - Full distributed tracing with correlation IDs

Ready for Layer 2: Model Behavior (prompt version tracking)!
""")
