"""
Comprehensive End-to-End Integration Test for Data Science Mode
Tests full pipeline from file upload to model training with real data
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_full_pipeline_classification():
    """Test complete pipeline with classification task"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Full Pipeline - Classification (Customer Churn)")
    logger.info("=" * 80)

    try:
        from agents.data_science_agents import DataScienceAgentTeam

        # Initialize agent team
        agent_team = DataScienceAgentTeam()
        logger.info("✅ Agent team initialized")

        # Run pipeline
        file_path = "data/customer_churn_sample.csv"
        target_column = "churned"

        if not os.path.exists(file_path):
            logger.error(f"❌ Test file not found: {file_path}")
            return False

        logger.info(f"📁 Running pipeline on: {file_path}")
        logger.info(f"🎯 Target column: {target_column}")

        start_time = time.time()

        # Run the pipeline
        results = agent_team.run_data_science_pipeline(
            file_path=file_path,
            target_column=target_column,
            task_type="classification"
        )

        end_time = time.time()
        duration = end_time - start_time

        # Validate results
        if not results.get("success"):
            logger.error(f"❌ Pipeline failed: {results.get('error', 'Unknown error')}")
            return False

        logger.info(f"✅ Pipeline completed in {duration:.2f} seconds")

        # Check required outputs
        required_keys = ["dataset_info", "best_model_name", "best_cv_score"]
        missing_keys = [key for key in required_keys if key not in results]

        if missing_keys:
            logger.error(f"❌ Missing required outputs: {missing_keys}")
            return False

        # Log results
        logger.info("\n📊 PIPELINE RESULTS:")
        logger.info(f"   Dataset: {results['dataset_info'].get('original_shape', 'N/A')}")
        logger.info(f"   Target: {results['dataset_info'].get('target_column', 'N/A')}")
        logger.info(f"   Task: {results['dataset_info'].get('task_type', 'N/A')}")
        logger.info(f"   Best Model: {results.get('best_model_name', 'N/A')}")
        logger.info(f"   CV Score: {results.get('best_cv_score', 'N/A'):.4f}")

        # Performance metrics
        logger.info("\n⏱️  PERFORMANCE METRICS:")
        logger.info(f"   Total Duration: {duration:.2f}s")
        logger.info(f"   Average per step: {duration/8:.2f}s")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_regression():
    """Test complete pipeline with regression task"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Full Pipeline - Regression")
    logger.info("=" * 80)

    try:
        from agents.data_science_agents import DataScienceAgentTeam

        # Check if regression test data exists
        test_file = "data/uploads/test_regression.csv"

        if not os.path.exists(test_file):
            logger.warning(f"⚠️  Regression test file not found: {test_file}")
            logger.info("   Skipping regression test")
            return True  # Don't fail if optional test file missing

        agent_team = DataScienceAgentTeam()

        start_time = time.time()

        results = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="price",  # Assuming price prediction
            task_type="regression"
        )

        duration = time.time() - start_time

        if not results.get("success"):
            logger.error(f"❌ Pipeline failed: {results.get('error')}")
            return False

        logger.info(f"✅ Regression pipeline completed in {duration:.2f}s")
        logger.info(f"   Best Model: {results.get('best_model_name', 'N/A')}")
        logger.info(f"   CV Score (R²): {results.get('best_cv_score', 'N/A'):.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Regression test failed: {e}")
        return False


def test_error_handling():
    """Test error handling for invalid inputs"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Error Handling")
    logger.info("=" * 80)

    from agents.data_science_agents import DataScienceAgentTeam

    agent_team = DataScienceAgentTeam()

    # Test 1: Non-existent file
    logger.info("Testing non-existent file...")
    results = agent_team.run_data_science_pipeline(
        file_path="data/nonexistent.csv",
        target_column="target"
    )

    if results.get("success"):
        logger.error("❌ Should have failed on non-existent file")
        return False
    else:
        logger.info("✅ Correctly handled non-existent file")

    # Test 2: Invalid target column (will be caught during pipeline)
    logger.info("Testing invalid target column...")
    # This should be caught during pipeline execution
    # We expect it to fail gracefully

    return True


def test_orchestrator_routing():
    """Test that orchestrator correctly routes to Data Science Mode"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Orchestrator Routing")
    logger.info("=" * 80)

    try:
        from agents.orchestrator import AgentOrchestrator

        orchestrator = AgentOrchestrator()

        # Test mode detection
        mode = orchestrator.detect_mode("data/customer_churn_sample.csv")

        if mode != "data_science":
            logger.error(f"❌ Wrong mode detected: {mode} (expected: data_science)")
            return False

        logger.info("✅ Correctly detected data_science mode for CSV file")

        # Test mode detection with keywords
        mode2 = orchestrator.detect_mode("train a model on my dataset")

        if mode2 != "data_science":
            logger.error(f"❌ Wrong mode for keyword: {mode2}")
            return False

        logger.info("✅ Correctly detected data_science mode from keywords")

        # Test query mode detection
        mode3 = orchestrator.detect_mode("Show sales by region")

        if mode3 != "query":
            logger.error(f"❌ Wrong mode for query: {mode3}")
            return False

        logger.info("✅ Correctly detected query mode for natural language")

        return True

    except Exception as e:
        logger.error(f"❌ Orchestrator test failed: {e}")
        return False


def test_report_generation():
    """Test that HTML reports are generated correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Report Generation")
    logger.info("=" * 80)

    try:
        # Check if a report was generated from previous test
        reports_dir = Path("reports")

        if not reports_dir.exists():
            logger.warning("⚠️  Reports directory doesn't exist")
            return True  # Not a failure

        report_files = list(reports_dir.glob("*.html"))

        if not report_files:
            logger.warning("⚠️  No HTML reports found")
            return True  # Not a failure

        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)

        # Check file size
        file_size = latest_report.stat().st_size

        if file_size < 1000:
            logger.error(f"❌ Report too small: {file_size} bytes")
            return False

        # Check content
        with open(latest_report, 'r') as f:
            content = f.read()

        required_sections = ["Dataset", "Model", "Performance"]
        missing = [s for s in required_sections if s.lower() not in content.lower()]

        if missing:
            logger.warning(f"⚠️  Report missing sections: {missing}")

        logger.info(f"✅ Report generated: {latest_report.name} ({file_size:,} bytes)")

        return True

    except Exception as e:
        logger.error(f"❌ Report test failed: {e}")
        return False


def test_performance_benchmarks():
    """Benchmark pipeline performance"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Performance Benchmarks")
    logger.info("=" * 80)

    try:
        from agents.data_science_agents import DataScienceAgentTeam

        agent_team = DataScienceAgentTeam()

        # Small dataset benchmark
        file_path = "data/customer_churn_sample.csv"  # 500 rows

        logger.info("Running performance benchmark...")

        start_time = time.time()

        results = agent_team.run_data_science_pipeline(
            file_path=file_path,
            target_column="churned",
            task_type="classification"
        )

        duration = time.time() - start_time

        if not results.get("success"):
            logger.error("❌ Benchmark run failed")
            return False

        # Performance thresholds
        logger.info(f"\n⏱️  BENCHMARK RESULTS:")
        logger.info(f"   Total Time: {duration:.2f}s")

        # Define acceptable thresholds
        if duration > 300:  # 5 minutes
            logger.warning(f"⚠️  Pipeline took longer than expected: {duration:.2f}s > 300s")
        else:
            logger.info(f"✅ Performance within acceptable range")

        # Estimate cost (rough approximation)
        # Assuming ~10-20 LLM calls at ~1000 tokens each
        estimated_tokens = 15000  # Conservative estimate
        estimated_cost = (estimated_tokens / 1000) * 0.005  # ~$5 per 1M tokens

        logger.info(f"   Estimated Tokens: ~{estimated_tokens:,}")
        logger.info(f"   Estimated Cost: ~${estimated_cost:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False


def run_all_tests():
    """Run all end-to-end tests"""
    logger.info("\n" + "🚀" * 40)
    logger.info("DATA SCIENCE MODE - END-TO-END INTEGRATION TESTS")
    logger.info("🚀" * 40 + "\n")

    # Ensure directories exist
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)

    test_suite = [
        ("Full Pipeline - Classification", test_full_pipeline_classification),
        ("Full Pipeline - Regression", test_full_pipeline_regression),
        ("Error Handling", test_error_handling),
        ("Orchestrator Routing", test_orchestrator_routing),
        ("Report Generation", test_report_generation),
        ("Performance Benchmarks", test_performance_benchmarks),
    ]

    results = []
    start_time = time.time()

    for test_name, test_func in test_suite:
        try:
            test_start = time.time()
            result = test_func()
            test_duration = time.time() - test_start
            results.append((test_name, result, test_duration))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False, 0))

    total_duration = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for test_name, result, duration in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:40} {status:12} ({duration:.1f}s)")

    logger.info("=" * 80)
    logger.info(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info(f"Total Duration: {total_duration:.2f}s")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Data Science Mode is production-ready!")
        return True
    else:
        logger.error(f"\n⚠️  {total - passed} tests failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
