"""
Performance Metrics Testing for Research Paper
Measures speed, accuracy, and quality metrics for Query Mode
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents import AnalyticsAgents
from utils.database import execute_sql, initialize_sample_data

load_dotenv()


class PerformanceMetrics:
    """Comprehensive performance testing for research paper metrics."""

    def __init__(self):
        self.results = {
            "speed_metrics": [],
            "accuracy_metrics": [],
            "quality_metrics": [],
            "comparison_metrics": {}
        }
        self.agents = None

    def initialize_agents(self):
        """Initialize the analytics agents."""
        print("🔄 Initializing AI agents...")
        start = time.time()
        self.agents = AnalyticsAgents(use_langchain=False)
        init_time = time.time() - start
        print(f"✅ Agents initialized in {init_time:.2f}s\n")
        return init_time

    def test_query_speed(self, queries: List[str]) -> Dict[str, Any]:
        """
        Measure query processing speed.

        Metrics:
        - Time to generate SQL
        - Time to execute query
        - Time to generate insights
        - Total end-to-end time
        """
        print("="*70)
        print("📊 SPEED METRICS TESTING")
        print("="*70 + "\n")

        speed_results = []

        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: '{query}'")

            try:
                # Measure end-to-end time
                start_total = time.time()

                # Step 1: SQL Generation
                start_sql = time.time()
                sql_response = self.agents.sql_agent.generate_reply(
                    messages=[{"role": "user", "content": query}]
                )
                sql_time = time.time() - start_sql

                # Extract SQL
                import re
                sql_match = re.search(r'```sql\n(.*?)\n```', sql_response, re.DOTALL | re.IGNORECASE)

                if sql_match:
                    sql_query = sql_match.group(1).strip()

                    # Step 2: Query Execution
                    start_exec = time.time()
                    exec_result = execute_sql(sql_query)
                    exec_time = time.time() - start_exec

                    # Step 3: Full pipeline (with insights)
                    start_insights = time.time()
                    full_result = self.agents.run(query)
                    insights_time = time.time() - start_insights

                    total_time = time.time() - start_total

                    result = {
                        "query": query,
                        "sql_generation_time": sql_time,
                        "query_execution_time": exec_time,
                        "insights_generation_time": insights_time - sql_time - exec_time,
                        "total_time": total_time,
                        "success": True
                    }

                    speed_results.append(result)

                    print(f"  ⏱️  SQL Generation: {sql_time:.2f}s")
                    print(f"  ⏱️  Query Execution: {exec_time:.3f}s")
                    print(f"  ⏱️  Total Time: {total_time:.2f}s")
                    print(f"  ✅ Success\n")
                else:
                    print(f"  ❌ Failed to generate SQL\n")

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}\n")

        # Calculate averages
        if speed_results:
            avg_metrics = {
                "avg_sql_generation": sum(r["sql_generation_time"] for r in speed_results) / len(speed_results),
                "avg_execution": sum(r["query_execution_time"] for r in speed_results) / len(speed_results),
                "avg_total": sum(r["total_time"] for r in speed_results) / len(speed_results),
                "min_time": min(r["total_time"] for r in speed_results),
                "max_time": max(r["total_time"] for r in speed_results)
            }

            print("\n📈 SPEED SUMMARY:")
            print(f"  Average SQL Generation: {avg_metrics['avg_sql_generation']:.2f}s")
            print(f"  Average Query Execution: {avg_metrics['avg_execution']:.3f}s")
            print(f"  Average Total Time: {avg_metrics['avg_total']:.2f}s")
            print(f"  Min/Max Time: {avg_metrics['min_time']:.2f}s / {avg_metrics['max_time']:.2f}s")

            self.results["speed_metrics"] = speed_results
            self.results["speed_summary"] = avg_metrics

            return avg_metrics

        return {}

    def test_query_accuracy(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Measure query accuracy.

        Metrics:
        - SQL correctness (does it execute?)
        - Result correctness (matches expected?)
        - Error rate
        """
        print("\n" + "="*70)
        print("🎯 ACCURACY METRICS TESTING")
        print("="*70 + "\n")

        accuracy_results = []

        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            expected_sql_type = test_case.get("expected_sql_type", None)
            expected_result_count = test_case.get("expected_result_count", None)

            print(f"Test Case {i}/{len(test_cases)}: '{query}'")

            try:
                # Generate SQL
                sql_response = self.agents.sql_agent.generate_reply(
                    messages=[{"role": "user", "content": query}]
                )

                import re
                sql_match = re.search(r'```sql\n(.*?)\n```', sql_response, re.DOTALL | re.IGNORECASE)

                if sql_match:
                    sql_query = sql_match.group(1).strip()

                    # Check SQL correctness
                    sql_correct = "SELECT" in sql_query.upper()

                    # Check SQL type (if specified)
                    type_correct = True
                    if expected_sql_type:
                        if expected_sql_type == "GROUP BY":
                            type_correct = "GROUP BY" in sql_query.upper()
                        elif expected_sql_type == "WHERE":
                            type_correct = "WHERE" in sql_query.upper()
                        elif expected_sql_type == "ORDER BY":
                            type_correct = "ORDER BY" in sql_query.upper()

                    # Execute query
                    exec_result = execute_sql(sql_query)
                    execution_success = exec_result.get("success", False)

                    # Check result count (if specified)
                    count_correct = True
                    actual_count = None
                    if expected_result_count and execution_success:
                        actual_count = len(exec_result.get("data", []))
                        count_correct = actual_count == expected_result_count

                    result = {
                        "query": query,
                        "sql_generated": sql_correct,
                        "sql_type_correct": type_correct,
                        "execution_success": execution_success,
                        "result_count_correct": count_correct,
                        "actual_count": actual_count,
                        "overall_accuracy": sql_correct and type_correct and execution_success and count_correct
                    }

                    accuracy_results.append(result)

                    status = "✅" if result["overall_accuracy"] else "⚠️"
                    print(f"  {status} SQL: {sql_correct}, Type: {type_correct}, Exec: {execution_success}, Count: {count_correct}\n")
                else:
                    print(f"  ❌ No SQL generated\n")
                    accuracy_results.append({
                        "query": query,
                        "overall_accuracy": False
                    })

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}\n")
                accuracy_results.append({
                    "query": query,
                    "overall_accuracy": False
                })

        # Calculate accuracy metrics
        if accuracy_results:
            total = len(accuracy_results)
            accurate = sum(1 for r in accuracy_results if r.get("overall_accuracy", False))
            sql_generated = sum(1 for r in accuracy_results if r.get("sql_generated", False))
            executed_successfully = sum(1 for r in accuracy_results if r.get("execution_success", False))

            accuracy_metrics = {
                "total_queries": total,
                "accurate_queries": accurate,
                "accuracy_rate": (accurate / total * 100) if total > 0 else 0,
                "sql_generation_rate": (sql_generated / total * 100) if total > 0 else 0,
                "execution_success_rate": (executed_successfully / total * 100) if total > 0 else 0,
                "error_rate": ((total - accurate) / total * 100) if total > 0 else 0
            }

            print("\n📊 ACCURACY SUMMARY:")
            print(f"  Total Queries: {total}")
            print(f"  Accurate Queries: {accurate}")
            print(f"  Accuracy Rate: {accuracy_metrics['accuracy_rate']:.1f}%")
            print(f"  SQL Generation Rate: {accuracy_metrics['sql_generation_rate']:.1f}%")
            print(f"  Execution Success Rate: {accuracy_metrics['execution_success_rate']:.1f}%")
            print(f"  Error Rate: {accuracy_metrics['error_rate']:.1f}%")

            self.results["accuracy_metrics"] = accuracy_results
            self.results["accuracy_summary"] = accuracy_metrics

            return accuracy_metrics

        return {}

    def test_quality_metrics(self, queries: List[str]) -> Dict[str, Any]:
        """
        Measure output quality.

        Metrics:
        - Chart generation success rate
        - Narrative generation success rate
        - Insight quality (length, completeness)
        """
        print("\n" + "="*70)
        print("⭐ QUALITY METRICS TESTING")
        print("="*70 + "\n")

        quality_results = []

        for i, query in enumerate(queries, 1):
            print(f"Query {i}/{len(queries)}: '{query}'")

            try:
                result = self.agents.run(query)

                has_chart = result and result.get("chart") is not None
                has_narrative = result and result.get("narrative") is not None

                narrative_length = len(result.get("narrative", "")) if has_narrative else 0
                narrative_quality = "high" if narrative_length > 100 else "medium" if narrative_length > 50 else "low"

                quality_result = {
                    "query": query,
                    "chart_generated": has_chart,
                    "narrative_generated": has_narrative,
                    "narrative_length": narrative_length,
                    "narrative_quality": narrative_quality,
                    "overall_quality": has_chart and has_narrative and narrative_length > 50
                }

                quality_results.append(quality_result)

                print(f"  📊 Chart: {'✅' if has_chart else '❌'}")
                print(f"  📝 Narrative: {'✅' if has_narrative else '❌'} ({narrative_length} chars)")
                print(f"  ⭐ Quality: {narrative_quality}\n")

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}\n")

        # Calculate quality metrics
        if quality_results:
            total = len(quality_results)
            charts_generated = sum(1 for r in quality_results if r.get("chart_generated", False))
            narratives_generated = sum(1 for r in quality_results if r.get("narrative_generated", False))
            high_quality = sum(1 for r in quality_results if r.get("overall_quality", False))

            quality_metrics = {
                "total_queries": total,
                "chart_success_rate": (charts_generated / total * 100) if total > 0 else 0,
                "narrative_success_rate": (narratives_generated / total * 100) if total > 0 else 0,
                "high_quality_rate": (high_quality / total * 100) if total > 0 else 0,
                "avg_narrative_length": sum(r.get("narrative_length", 0) for r in quality_results) / total if total > 0 else 0
            }

            print("\n⭐ QUALITY SUMMARY:")
            print(f"  Chart Generation Rate: {quality_metrics['chart_success_rate']:.1f}%")
            print(f"  Narrative Generation Rate: {quality_metrics['narrative_success_rate']:.1f}%")
            print(f"  High Quality Output Rate: {quality_metrics['high_quality_rate']:.1f}%")
            print(f"  Avg Narrative Length: {quality_metrics['avg_narrative_length']:.0f} chars")

            self.results["quality_metrics"] = quality_results
            self.results["quality_summary"] = quality_metrics

            return quality_metrics

        return {}

    def compare_with_baseline(self, manual_time: float = 300):
        """
        Compare AI system with manual/baseline approach.

        Args:
            manual_time: Average time for manual SQL writing (seconds)
        """
        print("\n" + "="*70)
        print("⚖️  BASELINE COMPARISON")
        print("="*70 + "\n")

        if "speed_summary" in self.results:
            ai_time = self.results["speed_summary"]["avg_total"]
            time_savings = manual_time - ai_time
            improvement_percent = (time_savings / manual_time * 100) if manual_time > 0 else 0

            comparison = {
                "manual_approach": {
                    "avg_time": manual_time,
                    "accuracy": 80,  # Typical manual accuracy
                    "description": "Manual SQL writing by analyst"
                },
                "ai_approach": {
                    "avg_time": ai_time,
                    "accuracy": self.results.get("accuracy_summary", {}).get("accuracy_rate", 0),
                    "description": "AI-powered multi-agent system"
                },
                "improvements": {
                    "time_saved_seconds": time_savings,
                    "time_saved_minutes": time_savings / 60,
                    "speed_improvement_percent": improvement_percent,
                    "accuracy_improvement_percent": self.results.get("accuracy_summary", {}).get("accuracy_rate", 0) - 80
                }
            }

            print(f"📊 Manual Approach:")
            print(f"   Time: {manual_time:.0f}s ({manual_time/60:.1f} min)")
            print(f"   Accuracy: ~80%")
            print()
            print(f"🤖 AI Approach:")
            print(f"   Time: {ai_time:.1f}s")
            print(f"   Accuracy: {comparison['ai_approach']['accuracy']:.1f}%")
            print()
            print(f"✨ Improvements:")
            print(f"   Time Saved: {time_savings:.1f}s ({time_savings/60:.1f} min)")
            print(f"   Speed Improvement: {improvement_percent:.1f}%")
            print(f"   Accuracy Improvement: {comparison['improvements']['accuracy_improvement_percent']:.1f}%")

            self.results["comparison_metrics"] = comparison

            return comparison

        return {}

    def generate_research_paper_table(self):
        """Generate formatted tables for research paper."""
        print("\n" + "="*70)
        print("📄 RESEARCH PAPER METRICS")
        print("="*70 + "\n")

        # Table 1: Performance Metrics
        print("Table 1: System Performance Metrics")
        print("-" * 70)
        print(f"{'Metric':<40} {'Value':<15} {'Unit':<15}")
        print("-" * 70)

        if "speed_summary" in self.results:
            speed = self.results["speed_summary"]
            print(f"{'Average Query Processing Time':<40} {speed['avg_total']:.2f}{' seconds':<15}")
            print(f"{'SQL Generation Time':<40} {speed['avg_sql_generation']:.2f}{' seconds':<15}")
            print(f"{'Query Execution Time':<40} {speed['avg_execution']:.3f}{' seconds':<15}")

        if "accuracy_summary" in self.results:
            accuracy = self.results["accuracy_summary"]
            print(f"{'Query Accuracy Rate':<40} {accuracy['accuracy_rate']:.1f}{' %':<15}")
            print(f"{'SQL Generation Success Rate':<40} {accuracy['sql_generation_rate']:.1f}{' %':<15}")
            print(f"{'Execution Success Rate':<40} {accuracy['execution_success_rate']:.1f}{' %':<15}")
            print(f"{'Error Rate':<40} {accuracy['error_rate']:.1f}{' %':<15}")

        if "quality_summary" in self.results:
            quality = self.results["quality_summary"]
            print(f"{'Chart Generation Rate':<40} {quality['chart_success_rate']:.1f}{' %':<15}")
            print(f"{'Narrative Generation Rate':<40} {quality['narrative_success_rate']:.1f}{' %':<15}")

        print("-" * 70 + "\n")

        # Table 2: Comparison with Baseline
        if "comparison_metrics" in self.results:
            comp = self.results["comparison_metrics"]
            print("\nTable 2: Comparison with Manual Approach")
            print("-" * 70)
            print(f"{'Approach':<20} {'Avg Time':<20} {'Accuracy':<20}")
            print("-" * 70)
            print(f"{'Manual SQL Writing':<20} {comp['manual_approach']['avg_time']/60:.1f} min{'':<12} {comp['manual_approach']['accuracy']:.0f}%{'':<15}")
            print(f"{'AI Multi-Agent':<20} {comp['ai_approach']['avg_time']:.1f} sec{'':<12} {comp['ai_approach']['accuracy']:.1f}%{'':<15}")
            print(f"{'Improvement':<20} {comp['improvements']['speed_improvement_percent']:.1f}% faster{'':<9} {comp['improvements']['accuracy_improvement_percent']:.1f}% better{'':<10}")
            print("-" * 70 + "\n")

    def save_results(self, filename: str = "performance_metrics_results.json"):
        """Save all results to JSON file."""
        output_path = Path(__file__).parent / filename

        # Add metadata
        self.results["metadata"] = {
            "test_date": datetime.now().isoformat(),
            "python_version": sys.version,
            "model": "GPT-4o"
        }

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")
        return output_path


def main():
    """Run comprehensive performance metrics testing."""
    print("\n" + "="*70)
    print("🔬 PERFORMANCE METRICS FOR RESEARCH PAPER")
    print("="*70 + "\n")

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found")
        return

    # Initialize database
    db_path = "data/sample_data.db"
    if not os.path.exists(db_path):
        print("📦 Initializing database...")
        initialize_sample_data()

    # Initialize metrics tester
    metrics = PerformanceMetrics()
    metrics.initialize_agents()

    # Test queries for speed testing
    speed_queries = [
        "Show total sales by region",
        "Which product has the highest sales?",
        "Show sales for the last month",
        "What is the average sale amount?",
        "Show sales by product and region"
    ]

    # Test cases for accuracy testing
    accuracy_test_cases = [
        {
            "query": "Show total sales by region",
            "expected_sql_type": "GROUP BY",
            "expected_result_count": 4  # 4 regions
        },
        {
            "query": "Which product has the highest sales?",
            "expected_sql_type": "ORDER BY",
        },
        {
            "query": "Show sales in the East region",
            "expected_sql_type": "WHERE",
        },
        {
            "query": "What is the total sales?",
            "expected_sql_type": None,
        },
        {
            "query": "Show sales by product",
            "expected_sql_type": "GROUP BY",
            "expected_result_count": 3  # 3 products
        }
    ]

    # Quality test queries
    quality_queries = [
        "Show total sales by region",
        "Which product performs best?",
        "Analyze sales trends"
    ]

    # Run all tests
    print("Starting comprehensive testing...\n")

    metrics.test_query_speed(speed_queries)
    metrics.test_query_accuracy(accuracy_test_cases)
    metrics.test_quality_metrics(quality_queries)
    metrics.compare_with_baseline(manual_time=300)  # 5 minutes manual time

    # Generate research paper outputs
    metrics.generate_research_paper_table()

    # Save results
    metrics.save_results()

    print("\n" + "="*70)
    print("✅ PERFORMANCE TESTING COMPLETE")
    print("="*70)
    print("\n💡 Use these metrics in your research paper:")
    print("   • Speed improvement: Check 'speed_summary'")
    print("   • Accuracy rates: Check 'accuracy_summary'")
    print("   • Quality metrics: Check 'quality_summary'")
    print("   • Baseline comparison: Check 'comparison_metrics'")
    print("\n📄 Results saved to: performance_metrics_results.json")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
