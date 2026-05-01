"""
Test script to demonstrate Validator feedback loop with complex queries
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents_with_validator import AnalyticsAgentsWithValidator

load_dotenv()


def test_validator_feedback():
    """Test the validator feedback loop with various query types."""

    print("\n" + "="*80)
    print("🧪 VALIDATOR FEEDBACK LOOP TESTING")
    print("="*80 + "\n")

    # Initialize agent with validator
    agents = AnalyticsAgentsWithValidator(max_validation_attempts=3)

    test_cases = [
        {
            "name": "Simple Query (should pass first try)",
            "query": "Show total sales by region",
            "expected_attempts": 1
        },
        {
            "name": "Ambiguous Query (might need refinement)",
            "query": "Show recent sales",
            "expected_attempts": "1-2"
        },
        {
            "name": "Complex Query (may need iteration)",
            "query": "Show year over year sales growth by product",
            "expected_attempts": "2-3"
        },
        {
            "name": "Time-based Query",
            "query": "What were sales in Q1 of this year?",
            "expected_attempts": "1-2"
        }
    ]

    results_summary = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(test_cases)}: {test_case['name']}")
        print(f"Query: \"{test_case['query']}\"")
        print(f"Expected attempts: {test_case['expected_attempts']}")
        print('='*80 + "\n")

        try:
            result = agents.run_with_validation(test_case['query'])

            # Extract key information
            attempts = result.get('attempts', 'N/A')
            validated = result.get('validated', False)
            sql_query = result.get('sql_query', 'N/A')
            narrative = result.get('narrative', 'N/A')

            print(f"\n📊 RESULTS:")
            print(f"  ✅ Attempts: {attempts}")
            print(f"  ✅ Validated: {validated}")
            print(f"  ✅ Success: {result.get('chart') is not None or narrative != 'N/A'}")
            print(f"\n  SQL Generated:")
            print(f"  {sql_query[:150]}...")
            print(f"\n  Narrative:")
            print(f"  {narrative[:150]}...")

            # Store results
            results_summary.append({
                "test": test_case['name'],
                "query": test_case['query'],
                "attempts": attempts,
                "validated": validated,
                "success": result.get('chart') is not None or narrative != 'N/A'
            })

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            results_summary.append({
                "test": test_case['name'],
                "query": test_case['query'],
                "attempts": "ERROR",
                "validated": False,
                "success": False
            })

    # Print summary
    print(f"\n\n{'='*80}")
    print("📈 TEST SUMMARY")
    print('='*80 + "\n")

    print(f"{'Test':<45} {'Attempts':<12} {'Validated':<12} {'Success'}")
    print("-" * 80)

    for result in results_summary:
        test_name = result['test'][:43] + ".." if len(result['test']) > 43 else result['test']
        attempts = str(result['attempts'])
        validated = "✅" if result['validated'] else "❌"
        success = "✅" if result['success'] else "❌"

        print(f"{test_name:<45} {attempts:<12} {validated:<12} {success}")

    # Get validation statistics
    print(f"\n{'='*80}")
    print("📊 VALIDATION STATISTICS")
    print('='*80 + "\n")

    stats = agents.get_validation_stats()

    print(f"Total Queries: {stats['total_queries']}")
    print(f"First-Try Success: {stats['first_try_success']} ({stats['first_try_success_rate']:.1f}%)")
    print(f"Average Iterations: {stats['avg_iterations']:.2f}")
    print(f"Min/Max Iterations: {stats['min_iterations']}-{stats['max_iterations']}")

    print(f"\nIterations Distribution:")
    dist = stats['iterations_distribution']
    print(f"  1 attempt: {dist['1']} queries")
    print(f"  2 attempts: {dist['2']} queries")
    print(f"  3 attempts: {dist['3']} queries")

    print(f"\n{'='*80}")
    print("✅ TESTING COMPLETE")
    print('='*80 + "\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        print("Please set it in your .env file")
        sys.exit(1)

    test_validator_feedback()
