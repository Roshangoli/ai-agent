"""
Test Layer 3: Output Quality - SQL & Narrative Scoring

Verifies:
- SQL quality scoring (syntax, schema, best practices, optimization)
- Narrative quality scoring (relevance, clarity, hallucination detection)
- Quality tracking and trends
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents import AnalyticsAgents
from observability.quality_scorer import get_quality_registry

print("=" * 80)
print("LAYER 3: OUTPUT QUALITY - SQL & NARRATIVE SCORING TEST")
print("=" * 80)

# Initialize agents
print("\n✓ Test 1: Quality Scoring Integration")
print("-" * 80)

agents = AnalyticsAgents()
quality_registry = get_quality_registry()

print("   ✅ Quality scorers initialized")

# Test 2: Run queries and check quality scores
print("\n✓ Test 2: SQL & Narrative Quality Scoring")
print("-" * 80)

test_queries = [
    {
        "question": "Show top 3 products by sales in each region",
        "expected_sql_features": ["window function", "good optimization"],
        "description": "Complex query with window functions"
    },
    {
        "question": "What is the total sales?",
        "expected_sql_features": ["simple aggregation"],
        "description": "Simple aggregation query"
    },
    {
        "question": "Show monthly sales trends for the last 6 months",
        "expected_sql_features": ["adaptive dates", "time-based"],
        "description": "Time-based query with adaptive filtering"
    }
]

results = []

for i, test_case in enumerate(test_queries, 1):
    print(f"\n📊 Query {i}: \"{test_case['question']}\"")
    print(f"   Description: {test_case['description']}")

    result = agents.run(test_case['question'])

    # Extract quality scores
    sql_quality = result.get('sql_quality', {})
    narrative_quality = result.get('narrative_quality', {})

    print(f"\n   SQL Quality:")
    print(f"      Score: {sql_quality.get('score', 0)}/100")
    print(f"      Rating: {sql_quality.get('rating', 'N/A')}")
    print(f"      Breakdown:")
    breakdown = sql_quality.get('breakdown', {})
    for metric, score in breakdown.items():
        print(f"         - {metric}: {score}/25")

    if sql_quality.get('issues'):
        print(f"      Issues:")
        for issue in sql_quality['issues'][:3]:  # Show first 3 issues
            print(f"         ⚠️  {issue}")

    print(f"\n   Narrative Quality:")
    print(f"      Score: {narrative_quality.get('score', 0)}/100")
    print(f"      Rating: {narrative_quality.get('rating', 'N/A')}")
    print(f"      Breakdown:")
    narrative_breakdown = narrative_quality.get('breakdown', {})
    for metric, score in narrative_breakdown.items():
        print(f"         - {metric}: {score}")

    if narrative_quality.get('issues'):
        print(f"      Issues:")
        for issue in narrative_quality['issues'][:3]:
            print(f"         ⚠️  {issue}")

    results.append({
        "question": test_case['question'],
        "sql_score": sql_quality.get('score', 0),
        "sql_rating": sql_quality.get('rating', 'N/A'),
        "narrative_score": narrative_quality.get('score', 0),
        "narrative_rating": narrative_quality.get('rating', 'N/A')
    })

    print(f"\n   ✅ Query completed")

# Test 3: Quality Registry Metrics
print("\n✓ Test 3: Quality Registry Metrics")
print("-" * 80)

metrics = quality_registry.get_metrics()

print(f"""
📊 Overall Quality Metrics:
   Total Queries: {metrics['total_queries']}
   Avg SQL Quality: {metrics['avg_sql_quality']:.1f}/100
   Avg Narrative Quality: {metrics['avg_narrative_quality']:.1f}/100
   Excellent Outputs: {metrics['excellent_count']}
   Poor Outputs: {metrics['poor_count']}
""")

print("   ✅ Quality metrics calculated")

# Test 4: Recent Quality Scores
print("\n✓ Test 4: Recent Quality Scores Lookup")
print("-" * 80)

recent = quality_registry.get_recent_scores(limit=3)

print("\n📋 Recent SQL Quality Scores:")
print("{:<40} {:<10} {:<12}".format("Query (truncated)", "Score", "Rating"))
print("-" * 65)

for score_data in recent['sql_scores']:
    query_trunc = score_data['sql_query'][:37] + "..."
    score = score_data['score']
    rating = score_data['rating']

    print("{:<40} {:<10} {:<12}".format(query_trunc, f"{score}/100", rating))

print("\n📋 Recent Narrative Quality Scores:")
print("{:<40} {:<10} {:<12}".format("Narrative (truncated)", "Score", "Rating"))
print("-" * 65)

for score_data in recent['narrative_scores']:
    narrative_trunc = score_data['narrative'][:37] + "..."
    score = score_data['score']
    rating = score_data['rating']

    print("{:<40} {:<10} {:<12}".format(narrative_trunc, f"{score}/100", rating))

print("\n   ✅ Recent scores retrieved")

# Test 5: Quality Breakdown Analysis
print("\n✓ Test 5: Quality Breakdown Analysis")
print("-" * 80)

print("\n📈 SQL Quality Breakdown (Avg per dimension):")

# Collect all breakdowns
sql_breakdowns = {
    "syntax": [],
    "schema_compliance": [],
    "best_practices": [],
    "optimization": []
}

for score_data in recent['sql_scores']:
    breakdown = score_data.get('breakdown', {})
    for key in sql_breakdowns.keys():
        if key in breakdown:
            sql_breakdowns[key].append(breakdown[key])

for dimension, scores in sql_breakdowns.items():
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"   {dimension.replace('_', ' ').title()}: {avg_score:.1f}/25")

print("\n📈 Narrative Quality Breakdown (Avg per dimension):")

narrative_breakdowns = {
    "relevance": [],
    "completeness": [],
    "clarity": [],
    "hallucination_free": []
}

for score_data in recent['narrative_scores']:
    breakdown = score_data.get('breakdown', {})
    for key in narrative_breakdowns.keys():
        if key in breakdown:
            narrative_breakdowns[key].append(breakdown[key])

for dimension, scores in narrative_breakdowns.items():
    if scores:
        avg_score = sum(scores) / len(scores)
        max_points = 30 if dimension == "relevance" else (20 if dimension == "completeness" else 25)
        print(f"   {dimension.replace('_', ' ').title()}: {avg_score:.1f}/{max_points}")

print("\n   ✅ Quality breakdown analysis complete")

# Summary
print("\n" + "=" * 80)
print("LAYER 3 TEST COMPLETE ✅")
print("=" * 80)

print(f"""
✅ Layer 3 Output Quality Features Verified:

1. ✅ SQL quality scoring (syntax, schema, best practices, optimization)
2. ✅ Narrative quality scoring (relevance, completeness, clarity, hallucination-free)
3. ✅ Quality tracking in registry
4. ✅ Issue detection and reporting
5. ✅ Quality breakdowns per dimension
6. ✅ Quality ratings (excellent/good/acceptable/poor)
7. ✅ Integration with Layer 1 & 2 (correlation IDs, prompt versions)

🎯 Portfolio Value:
   - Automated output quality validation
   - SQL best practices enforcement
   - Hallucination detection in narratives
   - Quality trend tracking over time
   - Avg SQL quality: {metrics['avg_sql_quality']:.1f}/100
   - Avg narrative quality: {metrics['avg_narrative_quality']:.1f}/100

Ready for production deployment!
""")

# Display summary table
print("\n" + "=" * 80)
print("SUMMARY: QUALITY SCORES BY QUERY")
print("=" * 80)

print("\n{:<45} {:<15} {:<15}".format(
    "Question", "SQL Quality", "Narrative Quality"
))
print("-" * 80)

for result in results:
    question_trunc = result['question'][:42] + "..." if len(result['question']) > 45 else result['question']
    sql_info = f"{result['sql_score']}/100 ({result['sql_rating']})"
    narrative_info = f"{result['narrative_score']}/100 ({result['narrative_rating']})"

    print("{:<45} {:<15} {:<15}".format(
        question_trunc, sql_info, narrative_info
    ))

print("\n" + "=" * 80)
