"""
Comprehensive Test Suite for Query Mode
Tests the complete natural language → SQL → Chart → Narrative pipeline
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from agents.analytics_agents import AnalyticsAgents
from agents.orchestrator import AgentOrchestrator
from utils.database import execute_sql, initialize_sample_data

load_dotenv()


class QueryModeTestSuite:
    """Comprehensive test suite for Query Mode functionality."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.test_results = []

    def log_result(self, test_name, passed, message=""):
        """Log test result."""
        status = "✅ PASSED" if passed else "❌ FAILED"
        result = {
            "test": test_name,
            "passed": passed,
            "message": message
        }
        self.test_results.append(result)

        if passed:
            self.passed += 1
        else:
            self.failed += 1

        print(f"{status} - {test_name}")
        if message:
            print(f"   {message}")

    def test_database_connectivity(self):
        """Test 1: Database connectivity and data existence."""
        print("\n" + "="*70)
        print("TEST 1: Database Connectivity")
        print("="*70)

        try:
            # Check database file exists
            db_path = "data/sample_data.db"
            if not os.path.exists(db_path):
                initialize_sample_data()

            # Test simple query
            result = execute_sql("SELECT COUNT(*) as count FROM sales;")

            if result.get("success") and result.get("data"):
                count = result["data"][0]["count"]
                self.log_result(
                    "Database Connectivity",
                    True,
                    f"Database has {count} records"
                )
                return True
            else:
                self.log_result(
                    "Database Connectivity",
                    False,
                    f"Query failed: {result.get('error')}"
                )
                return False

        except Exception as e:
            self.log_result("Database Connectivity", False, str(e))
            return False

    def test_agent_initialization(self):
        """Test 2: Agent system initialization."""
        print("\n" + "="*70)
        print("TEST 2: Agent Initialization")
        print("="*70)

        try:
            # Test Analytics Agents
            agents = AnalyticsAgents(use_langchain=False)

            # Check if agents are initialized
            has_sql_agent = hasattr(agents, 'sql_agent') and agents.sql_agent is not None
            has_insight_agent = hasattr(agents, 'insight_agent') and agents.insight_agent is not None

            if has_sql_agent and has_insight_agent:
                self.log_result(
                    "Agent Initialization",
                    True,
                    "SQL Generator and Insight Generator initialized"
                )
                return agents
            else:
                self.log_result(
                    "Agent Initialization",
                    False,
                    "Some agents failed to initialize"
                )
                return None

        except Exception as e:
            self.log_result("Agent Initialization", False, str(e))
            return None

    def test_sql_generation(self, agents):
        """Test 3: SQL query generation from natural language."""
        print("\n" + "="*70)
        print("TEST 3: SQL Query Generation")
        print("="*70)

        if not agents:
            self.log_result("SQL Generation", False, "Agents not initialized")
            return False

        test_questions = [
            "Show total sales by region",
            "Which product has the highest sales?",
            "Show sales for the last month"
        ]

        passed_count = 0

        for i, question in enumerate(test_questions, 1):
            print(f"\n  Question {i}: {question}")

            try:
                # Generate SQL using the agent
                sql_response = agents.sql_agent.generate_reply(
                    messages=[{"role": "user", "content": question}]
                )

                # Check if SQL was generated
                if "SELECT" in sql_response.upper():
                    print(f"  ✅ SQL generated")
                    passed_count += 1
                else:
                    print(f"  ❌ No SQL found in response")

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")

        success = passed_count == len(test_questions)
        self.log_result(
            "SQL Generation",
            success,
            f"{passed_count}/{len(test_questions)} queries generated successfully"
        )
        return success

    def test_end_to_end_query(self, agents):
        """Test 4: End-to-end query execution (NL → SQL → Results → Chart)."""
        print("\n" + "="*70)
        print("TEST 4: End-to-End Query Execution")
        print("="*70)

        if not agents:
            self.log_result("End-to-End Query", False, "Agents not initialized")
            return False

        test_query = "Show total sales by region"
        print(f"\n  Query: '{test_query}'")

        try:
            # Run the full pipeline
            result = agents.run(test_query)

            # Check results
            has_chart = result and result.get("chart") is not None
            has_narrative = result and result.get("narrative") is not None

            print(f"\n  Results:")
            print(f"    - Chart generated: {has_chart}")
            print(f"    - Narrative generated: {has_narrative}")

            if has_narrative:
                narrative = result["narrative"][:150]
                print(f"    - Narrative preview: {narrative}...")

            success = has_chart or has_narrative

            self.log_result(
                "End-to-End Query",
                success,
                "Pipeline completed successfully" if success else "Pipeline failed"
            )

            return success

        except Exception as e:
            self.log_result("End-to-End Query", False, str(e))
            return False

    def test_orchestrator_routing(self):
        """Test 5: Orchestrator correctly routes to Query Mode."""
        print("\n" + "="*70)
        print("TEST 5: Orchestrator Routing")
        print("="*70)

        try:
            orchestrator = AgentOrchestrator()

            # Test query mode detection
            test_inputs = [
                "Show sales by region",
                {"query": "What are the top products?"},
                "Analyze the sales data"
            ]

            passed_count = 0

            for i, input_data in enumerate(test_inputs, 1):
                detected_mode = orchestrator.detect_mode(input_data)
                print(f"  Input {i}: {str(input_data)[:50]}... → Mode: {detected_mode}")

                if detected_mode == "query":
                    passed_count += 1

            success = passed_count == len(test_inputs)

            self.log_result(
                "Orchestrator Routing",
                success,
                f"{passed_count}/{len(test_inputs)} inputs routed correctly"
            )

            return success

        except Exception as e:
            self.log_result("Orchestrator Routing", False, str(e))
            return False

    def test_query_variations(self, agents):
        """Test 6: Handle various query types."""
        print("\n" + "="*70)
        print("TEST 6: Query Variation Handling")
        print("="*70)

        if not agents:
            self.log_result("Query Variations", False, "Agents not initialized")
            return False

        query_types = {
            "Aggregation": "What is the total sales?",
            "Filtering": "Show sales in the East region",
            "Sorting": "Which region has the highest sales?",
            "Grouping": "Show sales by product",
            "Time-based": "Show sales for the last quarter"
        }

        passed_count = 0

        for query_type, question in query_types.items():
            print(f"\n  Testing {query_type}: '{question}'")

            try:
                sql_response = agents.sql_agent.generate_reply(
                    messages=[{"role": "user", "content": question}]
                )

                if "SELECT" in sql_response.upper():
                    print(f"  ✅ {query_type} query handled")
                    passed_count += 1
                else:
                    print(f"  ❌ {query_type} query failed")

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:50]}")

        success = passed_count >= len(query_types) * 0.8  # 80% pass rate

        self.log_result(
            "Query Variations",
            success,
            f"{passed_count}/{len(query_types)} query types handled"
        )

        return success

    def test_error_handling(self, agents):
        """Test 7: Error handling for invalid queries."""
        print("\n" + "="*70)
        print("TEST 7: Error Handling")
        print("="*70)

        if not agents:
            self.log_result("Error Handling", False, "Agents not initialized")
            return False

        # Test with queries that might cause issues
        problematic_queries = [
            "Show me data from a table that doesn't exist",
            "",  # Empty query
            "Random nonsense that isn't a question"
        ]

        handled_gracefully = 0

        for i, query in enumerate(problematic_queries, 1):
            print(f"\n  Test {i}: '{query[:50]}...'")

            try:
                result = agents.run(query)

                # Check if system handled it without crashing
                if result is not None:
                    print(f"  ✅ Handled gracefully")
                    handled_gracefully += 1
                else:
                    print(f"  ⚠️  Returned None")

            except Exception as e:
                # Even exceptions should be handled gracefully
                print(f"  ⚠️  Exception raised: {str(e)[:50]}")

        success = handled_gracefully >= 2  # At least 2/3 should be handled

        self.log_result(
            "Error Handling",
            success,
            f"{handled_gracefully}/{len(problematic_queries)} errors handled gracefully"
        )

        return success

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print(f"\nTotal Tests: {total}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📊 Pass Rate: {pass_rate:.1f}%")

        if self.failed > 0:
            print("\n⚠️  Failed Tests:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"   • {result['test']}: {result['message']}")

        print("\n" + "="*70)

        if pass_rate >= 80:
            print("🎉 Query Mode is working well!")
        elif pass_rate >= 60:
            print("⚠️  Query Mode has some issues but is functional")
        else:
            print("❌ Query Mode needs attention")

        print("="*70)

        return pass_rate


def main():
    """Run the complete test suite."""
    print("\n" + "="*70)
    print("🧪 QUERY MODE - COMPREHENSIVE TEST SUITE")
    print("="*70)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file to run Query Mode tests")
        return

    print("\nInitializing test suite...\n")

    suite = QueryModeTestSuite()

    # Run tests
    suite.test_database_connectivity()
    agents = suite.test_agent_initialization()

    if agents:
        suite.test_sql_generation(agents)
        suite.test_end_to_end_query(agents)
        suite.test_query_variations(agents)
        suite.test_error_handling(agents)

    suite.test_orchestrator_routing()

    # Print summary
    pass_rate = suite.print_summary()

    # Exit code based on results
    sys.exit(0 if pass_rate >= 60 else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
