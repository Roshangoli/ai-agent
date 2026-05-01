"""
Quick integration test for Data Science utilities
Tests data flow through cleaning → EDA → feature engineering → preprocessing
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_cleaning_pipeline():
    """Test the data cleaning pipeline"""
    logger.info("=" * 80)
    logger.info("INTEGRATION TEST: Data Cleaning Pipeline")
    logger.info("=" * 80)

    try:
        from utils.data_cleaner import DataCleaner

        # Load test data
        df = pd.read_csv("data/uploads/test_classification.csv")
        logger.info(f"Loaded test data: {df.shape}")

        # Add some messiness to test cleaning
        df.loc[0, 'age'] = None  # Missing value
        df.loc[1, 'income'] = -5000  # Invalid negative
        df = pd.concat([df, df.iloc[0:1]], ignore_index=True)  # Duplicate

        logger.info(f"After adding issues: {df.shape}")
        logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
        logger.info(f"  - Duplicates: {df.duplicated().sum()}")

        # Initialize cleaner
        cleaner = DataCleaner()

        # Test cleaning methods
        df_cleaned, _ = cleaner.remove_duplicates(df)
        logger.info(f"✅ After duplicate removal: {df_cleaned.shape}")

        df_cleaned, _ = cleaner.detect_invalid_values(df_cleaned)
        logger.info(f"✅ After invalid value detection: {df_cleaned.shape}")

        df_cleaned, _ = cleaner.handle_high_null_columns(df_cleaned, threshold=0.8)
        logger.info(f"✅ After high null column handling: {df_cleaned.shape}")

        df_cleaned, _ = cleaner.detect_low_variance_columns(df_cleaned, threshold=0.95)
        logger.info(f"✅ After low variance detection: {df_cleaned.shape}")

        logger.info(f"\nFinal cleaned data: {df_cleaned.shape}")
        logger.info(f"  - Missing values: {df_cleaned.isnull().sum().sum()}")
        logger.info(f"  - Duplicates: {df_cleaned.duplicated().sum()}")

        return True

    except Exception as e:
        logger.error(f"❌ Data cleaning test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eda_pipeline():
    """Test EDA pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST: EDA Pipeline")
    logger.info("=" * 80)

    try:
        from utils.eda_analyzer import EDAAnalyzer

        # Load test data
        df = pd.read_csv("data/uploads/test_classification.csv")
        target_column = "purchased"

        logger.info(f"Loaded test data: {df.shape}")

        # Initialize analyzer
        analyzer = EDAAnalyzer()

        # Test EDA methods
        imbalance_report = analyzer.detect_class_imbalance(df, target_column)
        logger.info(f"✅ Class imbalance detected: {imbalance_report.get('is_imbalanced', False)}")

        leakage_report = analyzer.detect_data_leakage(df, target_column)
        logger.info(f"✅ Data leakage check: {len(leakage_report.get('high_correlations', []))} high correlations found")

        cardinality_report = analyzer.analyze_high_cardinality(df)
        logger.info(f"✅ High cardinality features: {len(cardinality_report.get('high_cardinality_features', []))}")

        distribution_report = analyzer.analyze_distribution_per_feature(df)
        logger.info(f"✅ Distribution analysis: {len(distribution_report.get('features', []))} features analyzed")

        target_report = analyzer.analyze_target_variable(df, target_column, "classification")
        logger.info(f"✅ Target analysis: {target_report.get('class_balance', 'N/A')}")

        return True

    except Exception as e:
        logger.error(f"❌ EDA test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering_pipeline():
    """Test feature engineering pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST: Feature Engineering Pipeline")
    logger.info("=" * 80)

    try:
        from utils.feature_engineer import FeatureEngineer

        # Load test data
        df = pd.read_csv("data/uploads/test_regression.csv")
        target_column = "price"

        logger.info(f"Loaded test data: {df.shape}")

        # Initialize engineer
        engineer = FeatureEngineer()

        # Test feature engineering methods
        df_engineered, encoding_report = engineer.advanced_encoding(
            df.copy(),
            target_column=target_column,
            categorical_columns=['has_garage']
        )
        logger.info(f"✅ After encoding: {df_engineered.shape}")

        df_engineered, ratio_report = engineer.create_ratio_features(
            df_engineered,
            target_column=target_column
        )
        logger.info(f"✅ After ratio features: {df_engineered.shape}")

        df_engineered, bins_report = engineer.create_bins(
            df_engineered,
            column='age_years',
            n_bins=3
        )
        logger.info(f"✅ After binning: {df_engineered.shape}")

        df_engineered, missing_report = engineer.create_missing_indicators(
            df_engineered,
            df
        )
        logger.info(f"✅ After missing indicators: {df_engineered.shape}")

        logger.info(f"\nFinal engineered data: {df_engineered.shape}")
        logger.info(f"  - New features created: {df_engineered.shape[1] - df.shape[1]}")

        return True

    except Exception as e:
        logger.error(f"❌ Feature engineering test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_training_pipeline():
    """Test ML training pipeline"""
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST: ML Training Pipeline")
    logger.info("=" * 80)

    try:
        from utils.ml_trainer import MLTrainer
        from sklearn.model_selection import train_test_split

        # Load test data
        df = pd.read_csv("data/uploads/test_classification.csv")

        # Prepare data
        X = df.drop('purchased', axis=1)
        X = pd.get_dummies(X, drop_first=True)  # Quick encoding
        y = df['purchased']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        logger.info(f"Training data: {X_train.shape}")
        logger.info(f"Test data: {X_test.shape}")

        # Initialize trainer
        trainer = MLTrainer(task_type="classification")

        # Train models (limit to 2 for speed)
        training_report = trainer.train_models(
            X_train, y_train, X_test, y_test,
            models_to_try=["RandomForest", "LogisticRegression"],
            cv_folds=3
        )

        logger.info(f"✅ Models trained: {training_report['total_models']}")
        logger.info(f"✅ Successful models: {training_report['successful_models']}")
        logger.info(f"✅ Best model: {training_report['best_model_name']}")
        logger.info(f"✅ Best CV score: {training_report['best_cv_score']:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ ML training test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    logger.info("\n🔬 STARTING INTEGRATION TESTS\n")

    tests = [
        ("Data Cleaning", test_data_cleaning_pipeline),
        ("EDA Analysis", test_eda_pipeline),
        ("Feature Engineering", test_feature_engineering_pipeline),
        ("ML Training", test_ml_training_pipeline),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:25} {status}")

    logger.info("=" * 80)
    logger.info(f"OVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 ALL INTEGRATION TESTS PASSED!")
    else:
        logger.error(f"\n⚠️ {total - passed} tests failed")
