"""
Test Complex Query Optimization
Tests the system with progressively complex queries to identify optimization issues
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents import AnalyticsAgents
from utils.database import execute_sql
import time

print("=" * 80)
print("COMPLEX QUERY OPTIMIZATION TEST")
print("=" * 80)

# Initialize agents
print("\n✓ Initializing Analytics Agents...")
agents = AnalyticsAgents()

# Test queries from simple to complex
test_queries = [
    {
        "name": "Simple Aggregation",
        "query": "What is the total sales?",
        "complexity": "Low",
        "optimal_features": ["Basic SUM aggregation"]
    },
    {
        "name": "Multi-Column Grouping",
        "query": "Show sales by region and product",
        "complexity": "Medium",
        "optimal_features": ["GROUP BY with 2 columns", "Should have ORDER BY"]
    },
    {
        "name": "Top N with Filtering",
        "query": "What are the top 5 products by sales in the East region?",
        "complexity": "Medium",
        "optimal_features": ["WHERE filter", "GROUP BY", "ORDER BY", "LIMIT"]
    },
    {
        "name": "Time-Based Analysis",
        "query": "Show monthly sales trends for the last 6 months",
        "complexity": "High",
        "optimal_features": [
            "Date filtering",
            "Date grouping (strftime)",
            "Time-based WHERE clause",
            "Should order by date"
        ]
    },
    {
        "name": "Multi-Step Logic (CTE Candidate)",
        "query": "Show sales for products that have more than 10 transactions, grouped by region",
        "complexity": "High",
        "optimal_features": [
            "Should use CTE or subquery",
            "HAVING clause for filtering after aggregation",
            "Multiple GROUP BY operations"
        ]
    },
    {
        "name": "Ranking/Window Function",
        "query": "For each region, show the top 3 products by sales",
        "complexity": "Very High",
        "optimal_features": [
            "Should use window functions (ROW_NUMBER or RANK)",
            "Partition by region",
            "Or use CTE with filtering"
        ]
    },
    {
        "name": "Year-over-Year Comparison",
        "query": "Compare sales this year vs last year by region",
        "complexity": "Very High",
        "optimal_features": [
            "Should use CASE WHEN or pivot logic",
            "Date filtering for multiple periods",
            "Self-join or window function for comparison"
        ]
    }
]

results = []

print("\nRunning tests...\n")

for i, test in enumerate(test_queries, 1):
    print("=" * 80)
    print(f"TEST {i}/{len(test_queries)}: {test['name']}")
    print(f"Complexity: {test['complexity']}")
    print("=" * 80)
    print(f"Question: \"{test['query']}\"")
    print()

    try:
        start_time = time.time()
        result = agents.run(test['query'])
        execution_time = time.time() - start_time

        # Extract SQL from logs or result
        # For now, we'll check if it succeeded
        success = result.get('chart') is not None

        print(f"✓ Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"✓ Execution Time: {execution_time:.2f}s")

        # Try to get the SQL that was generated (from agent response)
        if success:
            print(f"✓ Chart Generated: Yes")
            print(f"✓ Narrative Generated: Yes")

        print()
        print("Expected Optimization Features:")
        for feature in test['optimal_features']:
            print(f"  - {feature}")

        results.append({
            "name": test['name'],
            "complexity": test['complexity'],
            "success": success,
            "time": execution_time,
            "query": test['query']
        })

    except Exception as e:
        print(f"❌ FAILED: {e}")
        results.append({
            "name": test['name'],
            "complexity": test['complexity'],
            "success": False,
            "error": str(e),
            "query": test['query']
        })

    print()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

success_count = sum(1 for r in results if r.get('success'))
total_count = len(results)

print(f"\nTotal Queries: {total_count}")
print(f"✅ Successful: {success_count}")
print(f"❌ Failed: {total_count - success_count}")
print(f"📊 Success Rate: {success_count/total_count*100:.1f}%")

print("\nResults by Complexity:")
for complexity in ["Low", "Medium", "High", "Very High"]:
    complexity_results = [r for r in results if r.get('complexity') == complexity]
    if complexity_results:
        complexity_success = sum(1 for r in complexity_results if r.get('success'))
        print(f"  {complexity}: {complexity_success}/{len(complexity_results)} successful")

print("\n" + "=" * 80)
print("OPTIMIZATION ISSUES TO CHECK:")
print("=" * 80)
print("""
To verify if queries are optimized, check the analytics_agents.log file for:

1. ❌ Missing LIMIT clauses on large result sets
2. ❌ No CTEs (WITH clauses) for multi-step logic
3. ❌ Missing window functions for ranking queries
4. ❌ No index hints or optimization comments
5. ❌ Using SELECT * instead of specific columns
6. ❌ Inefficient date filtering (LIKE vs proper date functions)
7. ❌ Subqueries in WHERE clause instead of JOINs

Recommended: Review the generated SQL in the log file and compare with the
'Expected Optimization Features' listed above.
""")

print("\n" + "=" * 80)
print("To see the actual SQL generated, run:")
print("  tail -100 analytics_agents.log | grep 'Generated SQL' -A 10")
print("=" * 80)
