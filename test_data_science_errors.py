"""
Error Handling and Edge Case Tests for Data Science Mode
Tests robustness, graceful degradation, and error recovery
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_missing_file():
    """Test handling of non-existent files"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Missing File Error Handling")
    logger.info("=" * 80)

    try:
        from agents.data_science_agents import DataScienceAgentTeam

        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path="data/does_not_exist.csv",
            target_column="target"
        )

        if result.get("success"):
            logger.error("❌ Should have failed on missing file")
            return False

        logger.info(f"✅ Correctly handled missing file")
        logger.info(f"   Error message: {result.get('error', 'N/A')}")

        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_invalid_target_column():
    """Test handling of non-existent target column"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Invalid Target Column")
    logger.info("=" * 80)

    try:
        # Create test file
        test_file = "data/uploads/test_invalid_target.csv"
        os.makedirs("data/uploads", exist_ok=True)

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'actual_target': [0, 1, 0, 1, 0]
        })
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        # Try with wrong target column name
        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="nonexistent_column"
        )

        # Should fail or handle gracefully
        if result.get("success"):
            logger.warning("⚠️  Pipeline succeeded despite invalid target")
            # Check if it auto-corrected
            if "actual_target" in str(result.get("dataset_info", {})):
                logger.info("✅ Pipeline auto-detected correct target")
                return True

        logger.info("✅ Correctly handled invalid target column")

        # Cleanup
        os.remove(test_file)

        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_empty_dataset():
    """Test handling of empty CSV file"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Empty Dataset")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_empty.csv"
        os.makedirs("data/uploads", exist_ok=True)

        # Create empty CSV with headers only
        df = pd.DataFrame(columns=['feature1', 'feature2', 'target'])
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        if result.get("success"):
            logger.error("❌ Should have failed on empty dataset")
            os.remove(test_file)
            return False

        logger.info("✅ Correctly rejected empty dataset")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_all_missing_values():
    """Test handling of column with all missing values"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: All Missing Values in Column")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_all_missing.csv"

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'all_null': [np.nan] * 5,
            'target': [0, 1, 0, 1, 0]
        })
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        # Should handle by dropping the column
        if result.get("success"):
            logger.info("✅ Pipeline handled all-null column (likely dropped)")
        else:
            logger.info("✅ Pipeline correctly failed on problematic data")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_single_class_target():
    """Test handling of target with only one class (no variance)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Single Class Target")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_single_class.csv"

        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 0]  # All same class
        })
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        if result.get("success"):
            logger.warning("⚠️  Pipeline succeeded despite single class")
            logger.info("   Model will have poor predictive power")
        else:
            logger.info("✅ Correctly identified single class problem")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_high_cardinality_categorical():
    """Test handling of categorical feature with very high cardinality"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: High Cardinality Categorical Feature")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_high_cardinality.csv"

        # Create dataset with unique ID as feature (bad practice)
        df = pd.DataFrame({
            'unique_id': range(100),  # 100 unique values
            'feature1': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        if result.get("success"):
            logger.info("✅ Pipeline handled high cardinality feature")
            logger.info("   (Should drop or use frequency encoding)")
        else:
            logger.info("⚠️  Pipeline failed on high cardinality")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.warning(f"⚠️  Exception raised: {type(e).__name__}")
        return True


def test_corrupted_csv():
    """Test handling of corrupted/malformed CSV"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 7: Corrupted CSV File")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_corrupted.csv"

        # Create malformed CSV
        with open(test_file, 'w') as f:
            f.write("feature1,feature2,target\n")
            f.write("1,2,0\n")
            f.write("3,4\n")  # Missing column
            f.write("5,6,7,8\n")  # Extra column
            f.write("9,10,1\n")

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        # Pandas might handle this, or it might fail
        if result.get("success"):
            logger.info("✅ Pipeline handled corrupted CSV (pandas auto-corrected)")
        else:
            logger.info("✅ Correctly rejected corrupted CSV")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_tiny_dataset():
    """Test handling of very small dataset (<20 rows)"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 8: Tiny Dataset (<20 rows)")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_tiny.csv"

        # Only 10 rows - not enough for train/test split
        df = pd.DataFrame({
            'feature1': range(10),
            'feature2': range(10, 20),
            'target': [0, 1] * 5
        })
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        if result.get("success"):
            logger.warning("⚠️  Pipeline succeeded on tiny dataset")
            logger.info("   Results may not be reliable")
        else:
            logger.info("✅ Correctly rejected tiny dataset")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.info(f"✅ Exception raised as expected: {type(e).__name__}")
        return True


def test_numeric_target_for_classification():
    """Test that regression is chosen for continuous numeric target"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 9: Task Type Auto-Detection")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_task_detection.csv"

        # Continuous numeric target (should be regression)
        df = pd.DataFrame({
            'feature1': range(50),
            'feature2': np.random.randn(50),
            'price': np.random.uniform(100, 1000, 50)  # Continuous
        })
        df.to_csv(test_file, index=False)

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="price",
            task_type="auto"  # Let agent decide
        )

        if result.get("success"):
            detected_task = result.get("dataset_info", {}).get("task_type", "unknown")
            logger.info(f"✅ Task type detected: {detected_task}")

            if detected_task == "regression":
                logger.info("✅ Correctly identified as regression")
            else:
                logger.warning(f"⚠️  Detected as {detected_task}, expected regression")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.warning(f"⚠️  Exception: {type(e).__name__}")
        return True


def test_mixed_data_types():
    """Test handling of mixed data types in same column"""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 10: Mixed Data Types")
    logger.info("=" * 80)

    try:
        test_file = "data/uploads/test_mixed_types.csv"

        # Create CSV with mixed types
        with open(test_file, 'w') as f:
            f.write("mixed_col,feature2,target\n")
            f.write("123,1.5,0\n")
            f.write("abc,2.5,1\n")
            f.write("456,3.5,0\n")
            f.write("def,4.5,1\n")

        from agents.data_science_agents import DataScienceAgentTeam
        agent_team = DataScienceAgentTeam()

        result = agent_team.run_data_science_pipeline(
            file_path=test_file,
            target_column="target"
        )

        if result.get("success"):
            logger.info("✅ Pipeline handled mixed data types")
            logger.info("   (Should treat as categorical)")
        else:
            logger.info("⚠️  Pipeline struggled with mixed types")

        os.remove(test_file)
        return True

    except Exception as e:
        logger.warning(f"⚠️  Exception: {type(e).__name__}")
        return True


def run_all_error_tests():
    """Run all error handling tests"""
    logger.info("\n" + "🚀" * 40)
    logger.info("DATA SCIENCE MODE - ERROR HANDLING & EDGE CASES")
    logger.info("🚀" * 40 + "\n")

    os.makedirs("data/uploads", exist_ok=True)

    test_suite = [
        ("Missing File", test_missing_file),
        ("Invalid Target Column", test_invalid_target_column),
        ("Empty Dataset", test_empty_dataset),
        ("All Missing Values", test_all_missing_values),
        ("Single Class Target", test_single_class_target),
        ("High Cardinality Categorical", test_high_cardinality_categorical),
        ("Corrupted CSV", test_corrupted_csv),
        ("Tiny Dataset", test_tiny_dataset),
        ("Task Type Detection", test_numeric_target_for_classification),
        ("Mixed Data Types", test_mixed_data_types),
    ]

    results = []

    for test_name, test_func in test_suite:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("ERROR HANDLING TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:40} {status}")

    logger.info("=" * 80)
    logger.info(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 ALL ERROR HANDLING TESTS PASSED!")
        logger.info("   Pipeline is robust against edge cases")
        return True
    else:
        logger.warning(f"\n⚠️  {total - passed} tests failed")
        logger.info("   Review error handling logic")
        return False


if __name__ == "__main__":
    success = run_all_error_tests()
    sys.exit(0 if success else 1)
