"""
Preprocessing Module
Advanced data preparation with intelligent strategy selection.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, KFold, GroupKFold,
    TimeSeriesSplit, RepeatedStratifiedKFold, LeaveOneOut
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, MaxAbsScaler, Normalizer
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    chi2, f_classif, RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import ks_2samp
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PreprocessingAgent:
    """
    Intelligent preprocessing with automatic strategy selection.
    """

    def __init__(self):
        """Initialize PreprocessingAgent."""
        self.preprocessing_report = {
            "actions": [],
            "pipeline": None
        }

    def intelligent_split(
        self,
        df: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        group_column: Optional[str] = None,
        datetime_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Intelligently split data based on characteristics.

        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            test_size: Proportion for test set
            group_column: Optional column for grouped split
            datetime_column: Optional column for temporal split

        Returns:
            Tuple of (train_df, test_df, split report)
        """
        try:
            split_report = {
                "strategy": None,
                "train_size": 0,
                "test_size": 0,
                "reasoning": ""
            }

            X = df.drop(columns=[target_column])
            y = df[target_column]

            # 1. Time-based split (if datetime column exists)
            if datetime_column and datetime_column in df.columns:
                df_sorted = df.sort_values(datetime_column)
                split_idx = int(len(df_sorted) * (1 - test_size))

                train_df = df_sorted.iloc[:split_idx]
                test_df = df_sorted.iloc[split_idx:]

                split_report["strategy"] = "time_based"
                split_report["reasoning"] = f"Temporal data detected. Using chronological split: train (first {(1-test_size)*100:.0f}%), test (last {test_size*100:.0f}%)."

            # 2. Group-based split (if group column exists with repeated values)
            elif group_column and group_column in df.columns:
                unique_groups = df[group_column].unique()

                if len(unique_groups) < len(df) * 0.8:  # At least some repetition
                    # Randomly assign groups to train/test
                    test_groups = np.random.choice(
                        unique_groups,
                        size=int(len(unique_groups) * test_size),
                        replace=False
                    )

                    train_df = df[~df[group_column].isin(test_groups)]
                    test_df = df[df[group_column].isin(test_groups)]

                    split_report["strategy"] = "group_based"
                    split_report["reasoning"] = f"Grouped data detected ({group_column}). Using GroupSplit to prevent leakage."
                else:
                    # Fall through to stratified/random
                    group_column = None

            # 3. Stratified split (for classification)
            if split_report["strategy"] is None:
                # Check if classification task
                if y.dtype == 'object' or y.nunique() <= 20:
                    train_df, test_df = train_test_split(
                        df,
                        test_size=test_size,
                        stratify=y,
                        random_state=42
                    )

                    split_report["strategy"] = "stratified"
                    class_dist = y.value_counts(normalize=True)
                    split_report["reasoning"] = f"Classification detected. Using stratified split to maintain class distribution ({dict(class_dist)})."

                # 4. Random split (for regression)
                else:
                    train_df, test_df = train_test_split(
                        df,
                        test_size=test_size,
                        random_state=42
                    )

                    split_report["strategy"] = "random"
                    split_report["reasoning"] = f"Regression task. Using random {int((1-test_size)*100)}/{int(test_size*100)} split."

            split_report["train_size"] = len(train_df)
            split_report["test_size"] = len(test_df)

            logger.info(f"✅ {split_report['reasoning']}")

            self.preprocessing_report["actions"].append({
                "action": "intelligent_split",
                "details": split_report
            })

            return train_df, test_df, split_report

        except Exception as e:
            logger.error(f"❌ Data split failed: {e}")
            return df, pd.DataFrame(), {"error": str(e)}

    def select_cv_strategy(
        self,
        df: pd.DataFrame,
        target_column: str,
        group_column: Optional[str] = None,
        datetime_column: Optional[str] = None,
        n_splits: int = 5
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Select appropriate cross-validation strategy.

        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            group_column: Optional column for grouped CV
            datetime_column: Optional column for temporal CV
            n_splits: Number of CV folds

        Returns:
            Tuple of (CV object, CV report)
        """
        try:
            cv_report = {
                "strategy": None,
                "n_splits": n_splits,
                "reasoning": ""
            }

            y = df[target_column]

            # 1. TimeSeriesSplit for temporal data
            if datetime_column and datetime_column in df.columns:
                cv = TimeSeriesSplit(n_splits=n_splits)
                cv_report["strategy"] = "TimeSeriesSplit"
                cv_report["reasoning"] = f"Time-series data with {len(df)} samples. Using TimeSeriesSplit with {n_splits} folds."

            # 2. GroupKFold for grouped data
            elif group_column and group_column in df.columns:
                unique_groups = df[group_column].nunique()
                if unique_groups >= n_splits:
                    cv = GroupKFold(n_splits=n_splits)
                    cv_report["strategy"] = "GroupKFold"
                    cv_report["reasoning"] = f"Grouped data ({unique_groups} groups). Using GroupKFold to prevent leakage."
                else:
                    # Fall through to stratified/regular
                    group_column = None

            # 3. StratifiedKFold for classification
            if cv_report["strategy"] is None:
                if y.dtype == 'object' or y.nunique() <= 20:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    cv_report["strategy"] = "StratifiedKFold"
                    cv_report["reasoning"] = f"Classification task. Using StratifiedKFold with {n_splits} folds."

                # 4. LeaveOneOut for very small datasets
                elif len(df) < 100:
                    cv = LeaveOneOut()
                    cv_report["strategy"] = "LeaveOneOut"
                    cv_report["reasoning"] = f"Very small dataset ({len(df)} samples). Using Leave-One-Out CV."

                # 5. Regular KFold for regression
                else:
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    cv_report["strategy"] = "KFold"
                    cv_report["reasoning"] = f"Regression task. Using KFold with {n_splits} folds."

            logger.info(f"✅ {cv_report['reasoning']}")

            self.preprocessing_report["actions"].append({
                "action": "select_cv_strategy",
                "details": cv_report
            })

            return cv, cv_report

        except Exception as e:
            logger.error(f"❌ CV strategy selection failed: {e}")
            return KFold(n_splits=5), {"error": str(e)}

    def select_scaler(
        self,
        df: pd.DataFrame,
        numerical_columns: Optional[List[str]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Intelligently select scaling method based on data distribution.

        Args:
            df: Pandas DataFrame
            numerical_columns: Columns to analyze (auto-detects if None)

        Returns:
            Tuple of (Scaler object, scaler report)
        """
        try:
            if numerical_columns is None:
                numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            scaler_report = {
                "scaler_type": None,
                "reasoning": ""
            }

            # Analyze distribution characteristics
            has_outliers = False
            is_gaussian = True
            is_skewed = False
            all_positive = True

            for col in numerical_columns:
                col_data = df[col].dropna()

                if len(col_data) == 0:
                    continue

                # Check for outliers (IQR method)
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (col_data < Q1 - 1.5 * IQR) | (col_data > Q3 + 1.5 * IQR)
                if outlier_mask.sum() / len(col_data) > 0.05:
                    has_outliers = True

                # Check skewness
                skewness = col_data.skew()
                if abs(skewness) > 1:
                    is_skewed = True
                    is_gaussian = False

                # Check if all positive
                if (col_data < 0).any():
                    all_positive = False

            # Decision logic
            if has_outliers and is_skewed:
                scaler = RobustScaler()
                scaler_report["scaler_type"] = "RobustScaler"
                scaler_report["reasoning"] = "Features have outliers and skewed distributions. Using RobustScaler (outlier-resistant)."

            elif is_skewed and all_positive:
                scaler = PowerTransformer(method='yeo-johnson')
                scaler_report["scaler_type"] = "PowerTransformer"
                scaler_report["reasoning"] = "Heavily skewed distributions. Using PowerTransformer for normalization."

            elif has_outliers:
                scaler = RobustScaler()
                scaler_report["scaler_type"] = "RobustScaler"
                scaler_report["reasoning"] = "Outliers detected. Using RobustScaler."

            elif is_gaussian:
                scaler = StandardScaler()
                scaler_report["scaler_type"] = "StandardScaler"
                scaler_report["reasoning"] = "Gaussian distributions. Using StandardScaler (z-score normalization)."

            else:
                scaler = MinMaxScaler()
                scaler_report["scaler_type"] = "MinMaxScaler"
                scaler_report["reasoning"] = "General case. Using MinMaxScaler to bound features to [0, 1]."

            logger.info(f"✅ {scaler_report['reasoning']}")

            self.preprocessing_report["actions"].append({
                "action": "select_scaler",
                "details": scaler_report
            })

            return scaler, scaler_report

        except Exception as e:
            logger.error(f"❌ Scaler selection failed: {e}")
            return StandardScaler(), {"error": str(e)}

    def handle_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: str = "auto"
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Handle class imbalance intelligently (TRAINING DATA ONLY).

        Args:
            X_train: Training features
            y_train: Training target
            strategy: "auto", "smote", "class_weights", or "none"

        Returns:
            Tuple of (balanced X_train, balanced y_train, imbalance report)
        """
        try:
            # Check if classification task
            if y_train.dtype == 'object' or y_train.nunique() > 20:
                return X_train, y_train, {"error": "Not a classification task"}

            # Calculate imbalance
            value_counts = y_train.value_counts()
            minority_pct = value_counts.min() / len(y_train) * 100

            imbalance_report = {
                "minority_percentage": float(minority_pct),
                "strategy_applied": None,
                "reasoning": ""
            }

            # Auto-select strategy based on imbalance severity
            if strategy == "auto":
                if minority_pct >= 40:
                    strategy = "none"
                elif minority_pct >= 30:
                    strategy = "class_weights"
                elif minority_pct >= 15:
                    strategy = "smote"
                else:
                    strategy = "smote_tomek"

            # Apply strategy
            if strategy == "none":
                imbalance_report["strategy_applied"] = "none"
                imbalance_report["reasoning"] = f"Mild imbalance ({minority_pct:.1f}%). No resampling needed."
                return X_train, y_train, imbalance_report

            elif strategy == "class_weights":
                # Class weights (returned in report, not applied to data)
                weights = {cls: len(y_train) / (len(value_counts) * count)
                          for cls, count in value_counts.items()}

                imbalance_report["strategy_applied"] = "class_weights"
                imbalance_report["class_weights"] = {str(k): float(v) for k, v in weights.items()}
                imbalance_report["reasoning"] = f"Moderate imbalance ({minority_pct:.1f}%). Using class weights (no resampling)."

                logger.info(f"✅ {imbalance_report['reasoning']}")
                return X_train, y_train, imbalance_report

            elif strategy in ["smote", "smote_tomek"]:
                try:
                    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek

                    if strategy == "smote":
                        sampler = SMOTE(random_state=42)
                        method_name = "SMOTE"
                    else:
                        sampler = SMOTETomek(random_state=42)
                        method_name = "SMOTETomek"

                    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

                    imbalance_report["strategy_applied"] = method_name
                    imbalance_report["samples_before"] = len(X_train)
                    imbalance_report["samples_after"] = len(X_resampled)
                    imbalance_report["reasoning"] = f"Severe imbalance ({minority_pct:.1f}%). Applied {method_name} to training set only."

                    logger.info(f"✅ {imbalance_report['reasoning']}")

                    self.preprocessing_report["actions"].append({
                        "action": "handle_imbalance",
                        "details": imbalance_report
                    })

                    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled), imbalance_report

                except ImportError:
                    logger.warning("⚠️ imbalanced-learn not installed. Skipping SMOTE.")
                    imbalance_report["strategy_applied"] = "none"
                    imbalance_report["reasoning"] = "SMOTE requested but imbalanced-learn not installed."
                    return X_train, y_train, imbalance_report

            return X_train, y_train, imbalance_report

        except Exception as e:
            logger.error(f"❌ Imbalance handling failed: {e}")
            return X_train, y_train, {"error": str(e)}

    def feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task_type: str = "auto",
        max_features: int = 30
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Intelligent feature selection based on task and feature count.

        Args:
            X_train: Training features
            y_train: Training target
            task_type: "classification", "regression", or "auto"
            max_features: Maximum features to keep

        Returns:
            Tuple of (selected X_train, selected feature names, selection report)
        """
        try:
            n_features = X_train.shape[1]

            selection_report = {
                "original_features": n_features,
                "selected_features": n_features,
                "method": None,
                "reasoning": ""
            }

            # Skip if already below threshold
            if n_features <= max_features:
                selection_report["method"] = "none"
                selection_report["reasoning"] = f"{n_features} features (below {max_features} threshold). No selection needed."
                return X_train, X_train.columns.tolist(), selection_report

            # Auto-detect task type
            if task_type == "auto":
                if y_train.dtype == 'object' or y_train.nunique() <= 20:
                    task_type = "classification"
                else:
                    task_type = "regression"

            # Select method based on feature count
            if n_features > 50:
                # Tree-based importance
                if task_type == "classification":
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

                model.fit(X_train, y_train)
                importances = pd.Series(model.feature_importances_, index=X_train.columns)
                top_features = importances.nlargest(max_features).index.tolist()

                selection_report["method"] = "tree_based_importance"
                selection_report["reasoning"] = f"{n_features} features. Using RandomForest importance → keeping top {len(top_features)}."

            elif n_features > 30:
                # Mutual information
                if task_type == "classification":
                    mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

                mi_series = pd.Series(mi_scores, index=X_train.columns)
                top_features = mi_series.nlargest(max_features).index.tolist()

                selection_report["method"] = "mutual_information"
                selection_report["reasoning"] = f"{n_features} features. Using mutual information → keeping top {len(top_features)}."

            else:
                # Keep all
                top_features = X_train.columns.tolist()
                selection_report["method"] = "none"
                selection_report["reasoning"] = f"{n_features} features (reasonable count). Keeping all."

            X_selected = X_train[top_features]
            selection_report["selected_features"] = len(top_features)
            selection_report["features_dropped"] = n_features - len(top_features)

            logger.info(f"✅ {selection_report['reasoning']}")

            self.preprocessing_report["actions"].append({
                "action": "feature_selection",
                "details": selection_report
            })

            return X_selected, top_features, selection_report

        except Exception as e:
            logger.error(f"❌ Feature selection failed: {e}")
            return X_train, X_train.columns.tolist(), {"error": str(e)}

    def validate_preprocessing(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Comprehensive validation after preprocessing.

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target

        Returns:
            Validation report
        """
        try:
            validation_report = {
                "checks": {},
                "all_passed": True
            }

            # 1. Check for NaN/inf
            has_nan_train = X_train.isna().any().any()
            has_nan_test = X_test.isna().any().any()
            has_inf_train = np.isinf(X_train.select_dtypes(include=[np.number])).any().any()
            has_inf_test = np.isinf(X_test.select_dtypes(include=[np.number])).any().any()

            validation_report["checks"]["no_nans"] = not (has_nan_train or has_nan_test)
            validation_report["checks"]["no_infs"] = not (has_inf_train or has_inf_test)

            if has_nan_train or has_nan_test or has_inf_train or has_inf_test:
                validation_report["all_passed"] = False
                logger.error("❌ Validation failed: NaN or Inf values found")

            # 2. Check shapes match
            shapes_match = X_train.shape[1] == X_test.shape[1]
            validation_report["checks"]["shapes_match"] = shapes_match

            if not shapes_match:
                validation_report["all_passed"] = False
                logger.error(f"❌ Validation failed: Shape mismatch (train: {X_train.shape[1]}, test: {X_test.shape[1]})")

            # 3. Check for data leakage (no identical rows)
            train_test_overlap = pd.merge(X_train, X_test, how='inner', indicator=False)
            has_leakage = len(train_test_overlap) > 0

            validation_report["checks"]["no_leakage"] = not has_leakage

            if has_leakage:
                validation_report["all_passed"] = False
                logger.warning(f"⚠️ Potential leakage: {len(train_test_overlap)} identical rows in train/test")

            # 4. Compare distributions (KS test for numerical features)
            distribution_similar = True
            for col in X_train.select_dtypes(include=[np.number]).columns:
                if col in X_test.columns:
                    ks_stat, p_value = ks_2samp(X_train[col].dropna(), X_test[col].dropna())
                    if p_value < 0.05:  # Significantly different distributions
                        distribution_similar = False
                        logger.warning(f"⚠️ Distribution shift in '{col}' (p={p_value:.4f})")

            validation_report["checks"]["similar_distributions"] = distribution_similar

            # 5. Check for empty dataframes
            not_empty = len(X_train) > 0 and len(X_test) > 0
            validation_report["checks"]["not_empty"] = not_empty

            if not not_empty:
                validation_report["all_passed"] = False
                logger.error("❌ Validation failed: Empty dataframes")

            # Summary
            if validation_report["all_passed"]:
                logger.info("✅ Validation complete: ✓ No NaNs ✓ No leakage ✓ Similar distributions ✓ Shapes match")
            else:
                logger.error("❌ Validation failed - check logs for details")

            self.preprocessing_report["actions"].append({
                "action": "validate_preprocessing",
                "details": validation_report
            })

            return validation_report

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {"error": str(e)}

    def build_pipeline(
        self,
        scaler: Any,
        selector: Optional[Any] = None,
        save_path: Optional[str] = None
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Build sklearn Pipeline for production use.

        Args:
            scaler: Scaler object (StandardScaler, RobustScaler, etc.)
            selector: Optional feature selector
            save_path: Optional path to save pipeline

        Returns:
            Tuple of (Pipeline object, pipeline report)
        """
        try:
            steps = []

            # Add scaler
            steps.append(('scaler', scaler))

            # Add selector if provided
            if selector is not None:
                steps.append(('selector', selector))

            # Build pipeline
            pipeline = Pipeline(steps)

            pipeline_report = {
                "steps": [step[0] for step in steps],
                "saved_path": save_path,
                "reasoning": f"Built sklearn Pipeline: {' → '.join([step[0] for step in steps])}"
            }

            # Save pipeline if path provided
            if save_path:
                joblib.dump(pipeline, save_path)
                pipeline_report["reasoning"] += f". Saved to {save_path}"

            logger.info(f"✅ {pipeline_report['reasoning']}")

            self.preprocessing_report["actions"].append({
                "action": "build_pipeline",
                "details": pipeline_report
            })
            self.preprocessing_report["pipeline"] = pipeline

            return pipeline, pipeline_report

        except Exception as e:
            logger.error(f"❌ Pipeline building failed: {e}")
            return Pipeline([]), {"error": str(e)}

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all preprocessing performed.

        Returns:
            Dictionary with preprocessing summary
        """
        return self.preprocessing_report
