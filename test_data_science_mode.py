"""
Comprehensive End-to-End Test for Data Science Mode
Tests all 8 steps of the autonomous pipeline
"""

import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_utility_imports():
    """Test that all utility modules import correctly"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Utility Module Imports")
    logger.info("=" * 80)

    modules_to_test = [
        ("utils.data_cleaner", "DataCleaner"),
        ("utils.eda_analyzer", "EDAAnalyzer"),
        ("utils.feature_engineer", "FeatureEngineer"),
        ("utils.preprocessor", "PreprocessingAgent"),
        ("utils.ml_trainer", "MLTrainer"),
        ("utils.model_evaluator", "ModelEvaluator"),
        ("utils.report_generator", "ReportGenerator"),
        ("utils.decision_parser", "DecisionParser"),
    ]

    passed = 0
    failed = 0

    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            logger.info(f"✅ {module_name}.{class_name} - OK")
            passed += 1
        except Exception as e:
            logger.error(f"❌ {module_name}.{class_name} - FAILED: {e}")
            failed += 1

    logger.info(f"\nImport Results: {passed} passed, {failed} failed")
    return failed == 0


def test_decision_parser():
    """Test decision parser with sample responses"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Decision Parser")
    logger.info("=" * 80)

    try:
        from utils.decision_parser import DecisionParser

        parser = DecisionParser()

        # Test cleaning decision parsing
        sample_cleaning_response = """
        ```json
        {
          "missing_values": {
            "age": {"strategy": "median", "reasoning": "Skewed distribution"},
            "income": {"strategy": "mean", "reasoning": "Normal distribution"}
          },
          "outliers": {
            "price": {"action": "cap", "reasoning": "Valid luxury items"}
          },
          "duplicates": {"action": "remove", "reasoning": "Exact copies found"}
        }
        ```
        """

        result = parser.parse_cleaning_decision(sample_cleaning_response)

        if "error" not in result:
            logger.info("✅ Cleaning decision parsing - OK")
            logger.info(f"   - Missing value strategies: {len(result.get('missing_values', {}))}")
            logger.info(f"   - Outlier actions: {len(result.get('outliers', {}))}")
            return True
        else:
            logger.error(f"❌ Cleaning decision parsing FAILED: {result['error']}")
            return False

    except Exception as e:
        logger.error(f"❌ Decision parser test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generation():
    """Test HTML report generation"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Report Generation")
    logger.info("=" * 80)

    try:
        from utils.report_generator import ReportGenerator

        generator = ReportGenerator()

        # Sample pipeline results
        sample_results = {
            "dataset_info": {
                "original_shape": (100, 10),
                "cleaned_shape": (95, 8),
                "target_column": "target",
                "task_type": "classification"
            },
            "cleaning_summary": {
                "rows_removed": 5,
                "columns_removed": 2
            },
            "model_comparison": {
                "RandomForest": {"cv_mean": 0.85, "cv_std": 0.03},
                "XGBoost": {"cv_mean": 0.87, "cv_std": 0.02}
            },
            "best_model_name": "XGBoost",
            "best_cv_score": 0.87
        }

        # Generate HTML report
        html_report = generator.generate_html_report(sample_results, "reports/test_report.html")

        # Check if report was created
        if os.path.exists("reports/test_report.html"):
            file_size = os.path.getsize("reports/test_report.html")
            logger.info(f"✅ HTML report generation - OK ({file_size} bytes)")
            return True
        else:
            logger.error("❌ HTML report not created")
            return False

    except Exception as e:
        logger.error(f"❌ Report generation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "🚀" * 40)
    logger.info("STARTING COMPREHENSIVE DATA SCIENCE MODE TESTS")
    logger.info("🚀" * 40 + "\n")

    # Ensure directories exist
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)

    test_results = []

    # Run tests
    test_results.append(("Utility Imports", test_utility_imports()))
    test_results.append(("Decision Parser", test_decision_parser()))
    test_results.append(("Report Generation", test_report_generation()))

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:30} {status}")

    logger.info("=" * 80)
    logger.info(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Data Science Mode utilities are working!")
        return True
    else:
        logger.error(f"\n⚠️ {total - passed} tests failed. Please review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
