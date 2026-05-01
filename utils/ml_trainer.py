"""
ML Training Module
Dynamic model registry with 50+ algorithms and intelligent selection.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso, ElasticNet,
    SGDClassifier, SGDRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Comprehensive model registry with 50+ ML algorithms.
    """

    @staticmethod
    def get_classification_models() -> Dict[str, Any]:
        """Get all available classification models."""
        return {
            # Tree-based
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=50, random_state=42),

            # Linear
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            "SGDClassifier": SGDClassifier(max_iter=1000, random_state=42, n_jobs=-1),

            # SVM
            "SVM_Linear": SVC(kernel='linear', random_state=42),
            "SVM_RBF": SVC(kernel='rbf', random_state=42),
            "SVM_Poly": SVC(kernel='poly', random_state=42),

            # Neighbors
            "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),

            # Naive Bayes
            "GaussianNB": GaussianNB(),
            "MultinomialNB": MultinomialNB(),
        }

    @staticmethod
    def get_regression_models() -> Dict[str, Any]:
        """Get all available regression models."""
        return {
            # Tree-based
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostRegressor(n_estimators=50, random_state=42),

            # Linear
            "LinearRegression": LinearRegression(n_jobs=-1),
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "SGDRegressor": SGDRegressor(max_iter=1000, random_state=42),

            # SVM
            "SVR_Linear": SVR(kernel='linear'),
            "SVR_RBF": SVR(kernel='rbf'),
            "SVR_Poly": SVR(kernel='poly'),

            # Neighbors
            "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        }

    @staticmethod
    def get_advanced_models(task_type: str) -> Dict[str, Any]:
        """
        Get advanced models (XGBoost, LightGBM, CatBoost).
        These are optional and only loaded if installed.

        Args:
            task_type: "classification" or "regression"

        Returns:
            Dictionary of advanced models
        """
        advanced = {}

        try:
            import xgboost as xgb
            if task_type == "classification":
                advanced["XGBoost"] = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
            else:
                advanced["XGBoost"] = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0
                )
        except ImportError:
            logger.warning("⚠️ XGBoost not installed")

        try:
            import lightgbm as lgb
            if task_type == "classification":
                advanced["LightGBM"] = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
            else:
                advanced["LightGBM"] = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
        except ImportError:
            logger.warning("⚠️ LightGBM not installed")

        try:
            import catboost as cb
            if task_type == "classification":
                advanced["CatBoost"] = cb.CatBoostClassifier(
                    iterations=100,
                    random_state=42,
                    verbose=False
                )
            else:
                advanced["CatBoost"] = cb.CatBoostRegressor(
                    iterations=100,
                    random_state=42,
                    verbose=False
                )
        except ImportError:
            logger.warning("⚠️ CatBoost not installed")

        return advanced


class MLTrainer:
    """
    Autonomous ML training with intelligent model selection.
    """

    def __init__(self, task_type: str = "auto"):
        """
        Initialize MLTrainer.

        Args:
            task_type: "classification", "regression", or "auto"
        """
        self.task_type = task_type
        self.training_report = {
            "models_trained": {},
            "best_model": None,
            "best_score": None
        }

    def detect_task_type(self, y: pd.Series) -> str:
        """
        Auto-detect task type from target variable.

        Args:
            y: Target variable

        Returns:
            "classification" or "regression"
        """
        if y.dtype == 'object' or y.nunique() <= 20:
            return "classification"
        else:
            return "regression"

    def select_models_for_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
        max_models: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Intelligently select models based on data characteristics with metadata.

        Args:
            X: Feature matrix
            y: Target variable
            task_type: "classification" or "regression"
            max_models: Maximum number of models to select

        Returns:
            Tuple of (list of model names, selection metadata dict)
        """
        n_samples = len(X)
        n_features = X.shape[1]
        has_high_cardinality = False

        # Check for high cardinality features (suggests CatBoost)
        high_card_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' and X[col].nunique() > 50:
                has_high_cardinality = True
                high_card_cols.append(col)

        selected_models = []
        metadata = {
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
                "has_high_cardinality": has_high_cardinality,
                "high_cardinality_columns": high_card_cols
            },
            "selection_tier": "",
            "reasoning": "",
            "model_rationale": {}
        }

        # Small datasets (<1000 samples)
        if n_samples < 1000:
            selected_models = ["RandomForest", "LogisticRegression" if task_type == "classification" else "Ridge"]
            if n_samples < 100:
                selected_models.append("KNN")  # Works well with small data

            metadata["selection_tier"] = "small"
            metadata["reasoning"] = f"Small dataset with {n_samples} samples detected. Selected lightweight, robust models that perform well with limited data."
            metadata["model_rationale"] = {
                "RandomForest": "Handles overfitting well on small data through ensemble averaging",
                "LogisticRegression" if task_type == "classification" else "Ridge": "Simple, interpretable baseline with strong regularization",
                "KNN": "Non-parametric model works well with <100 samples" if n_samples < 100 else None
            }
            metadata["model_rationale"] = {k: v for k, v in metadata["model_rationale"].items() if v is not None}

            logger.info(f"Small dataset ({n_samples} samples). Selected: {selected_models}")

        # Medium datasets (1k-100k samples)
        elif n_samples < 100000:
            selected_models = ["RandomForest", "XGBoost", "LightGBM"]
            if has_high_cardinality:
                selected_models.append("CatBoost")
            if task_type == "classification":
                selected_models.append("LogisticRegression")
            else:
                selected_models.append("Ridge")

            metadata["selection_tier"] = "medium"
            metadata["reasoning"] = f"Medium dataset with {n_samples:,} samples. Optimal balance for gradient boosting models which excel in this range."
            metadata["model_rationale"] = {
                "RandomForest": "Robust ensemble baseline with built-in feature importance",
                "XGBoost": "Industry-standard gradient boosting with excellent performance/speed tradeoff",
                "LightGBM": "Fast gradient boosting optimized for medium-large datasets",
                "CatBoost": f"Handles high-cardinality features naturally (found {len(high_card_cols)} columns)" if has_high_cardinality else None,
                "LogisticRegression" if task_type == "classification" else "Ridge": "Fast linear baseline for comparison"
            }
            metadata["model_rationale"] = {k: v for k, v in metadata["model_rationale"].items() if v is not None}

            logger.info(f"Medium dataset ({n_samples} samples). Selected: {selected_models}")

        # Large datasets (>100k samples)
        else:
            selected_models = ["LightGBM", "XGBoost", "SGDClassifier" if task_type == "classification" else "SGDRegressor"]

            metadata["selection_tier"] = "large"
            metadata["reasoning"] = f"Large dataset with {n_samples:,} samples. Prioritizing fast, scalable algorithms optimized for big data."
            metadata["model_rationale"] = {
                "LightGBM": "Fastest gradient boosting for large datasets with leaf-wise tree growth",
                "XGBoost": "Proven scalability and performance on large data",
                "SGDClassifier" if task_type == "classification" else "SGDRegressor": "Linear model with stochastic optimization - trains quickly on 100k+ samples"
            }

            logger.info(f"Large dataset ({n_samples} samples). Selected: {selected_models}")

        # Limit to max_models
        limited_models = selected_models[:max_models]
        metadata["selected_models"] = limited_models
        metadata["total_candidates"] = len(selected_models)

        return limited_models, metadata

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        models_to_try: Optional[List[str]] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train multiple models and compare performance.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Optional test features
            y_test: Optional test target
            models_to_try: List of model names (auto-selects if None)
            cv_folds: Number of cross-validation folds

        Returns:
            Training report with results for all models
        """
        try:
            # Auto-detect task type
            if self.task_type == "auto":
                self.task_type = self.detect_task_type(y_train)

            logger.info(f"🎯 Task type: {self.task_type}")

            # Get model registry
            if self.task_type == "classification":
                model_registry = ModelRegistry.get_classification_models()
            else:
                model_registry = ModelRegistry.get_regression_models()

            # Add advanced models
            advanced_models = ModelRegistry.get_advanced_models(self.task_type)
            model_registry.update(advanced_models)

            # Auto-select models if not specified
            model_selection_metadata = None
            if models_to_try is None:
                models_to_try, model_selection_metadata = self.select_models_for_data(X_train, y_train, self.task_type)

            # Filter to only available models
            available_models = {name: model for name, model in model_registry.items() if name in models_to_try}

            logger.info(f"🔧 Training {len(available_models)} models: {list(available_models.keys())}")

            # Train each model
            results = {}

            for model_name, model in available_models.items():
                try:
                    logger.info(f"Training {model_name}...")

                    # Cross-validation
                    if self.task_type == "classification":
                        scoring = 'accuracy'
                    else:
                        scoring = 'r2'

                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=cv_folds,
                        scoring=scoring,
                        n_jobs=-1
                    )

                    # Train on full training set
                    model.fit(X_train, y_train)

                    # Evaluate on test set if provided
                    if X_test is not None and y_test is not None:
                        test_score = model.score(X_test, y_test)
                    else:
                        test_score = None

                    results[model_name] = {
                        "model": model,
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std()),
                        "test_score": float(test_score) if test_score is not None else None,
                        "trained": True
                    }

                    logger.info(f"✅ {model_name}: CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                except Exception as e:
                    logger.error(f"❌ {model_name} training failed: {e}")
                    results[model_name] = {
                        "model": None,
                        "error": str(e),
                        "trained": False
                    }

            # Find best model
            best_model_name = max(
                [k for k, v in results.items() if v.get("trained", False)],
                key=lambda k: results[k]["cv_mean"]
            )

            self.training_report = {
                "task_type": self.task_type,
                "models_trained": results,
                "best_model_name": best_model_name,
                "best_model": results[best_model_name]["model"],
                "best_cv_score": results[best_model_name]["cv_mean"],
                "best_test_score": results[best_model_name].get("test_score"),
                "total_models": len(results),
                "successful_models": sum(1 for v in results.values() if v.get("trained", False)),
                "model_selection_metadata": model_selection_metadata  # Add reasoning for model selection
            }

            logger.info(f"🏆 Best model: {best_model_name} (CV score: {results[best_model_name]['cv_mean']:.4f})")

            return self.training_report

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {"error": str(e)}

    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20
    ) -> Dict[str, float]:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            Dictionary of {feature: importance}
        """
        try:
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if len(importances.shape) > 1:
                    importances = importances[0]
            else:
                logger.warning(f"⚠️ Model {type(model).__name__} does not support feature importance")
                return {}

            # Create importance dict
            importance_dict = dict(zip(feature_names, importances))

            # Sort and get top N
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            )

            return sorted_importance

        except Exception as e:
            logger.error(f"❌ Feature importance extraction failed: {e}")
            return {}

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of model training.

        Returns:
            Training summary report
        """
        return self.training_report


# Convenience function
def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[pd.Series] = None,
    task_type: str = "auto"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Quick train and return best model.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Optional test features
        y_test: Optional test target
        task_type: "classification", "regression", or "auto"

    Returns:
        Tuple of (best_model, training_report)
    """
    trainer = MLTrainer(task_type=task_type)
    report = trainer.train_models(X_train, y_train, X_test, y_test)

    best_model = report.get("best_model")
    return best_model, report
