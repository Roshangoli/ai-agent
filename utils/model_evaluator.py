"""
Model Evaluation Module
Comprehensive metrics and model comparison for ML models.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with 30+ metrics.
    """

    def __init__(self):
        """Initialize ModelEvaluator."""
        self.evaluation_report = {
            "metrics": {},
            "comparison": {}
        }

    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Calculate all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
            model_name: Name of model being evaluated

        Returns:
            Dictionary with all metrics
        """
        try:
            metrics = {
                "model_name": model_name,
                "task_type": "classification"
            }

            # Basic metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

            # Multi-class handling
            n_classes = len(np.unique(y_true))
            average_method = 'binary' if n_classes == 2 else 'weighted'

            metrics["precision"] = float(precision_score(y_true, y_pred, average=average_method, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, average=average_method, zero_division=0))
            metrics["f1_score"] = float(f1_score(y_true, y_pred, average=average_method, zero_division=0))

            # ROC-AUC (if probabilities provided)
            if y_pred_proba is not None:
                try:
                    if n_classes == 2:
                        # Binary classification
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                    else:
                        # Multi-class
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted'))
                except:
                    metrics["roc_auc"] = None

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            # Per-class metrics
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            metrics["per_class_metrics"] = class_report

            # Matthews Correlation Coefficient (MCC)
            try:
                from sklearn.metrics import matthews_corrcoef
                metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
            except:
                metrics["mcc"] = None

            # Log-loss (if probabilities provided)
            if y_pred_proba is not None:
                try:
                    from sklearn.metrics import log_loss
                    metrics["log_loss"] = float(log_loss(y_true, y_pred_proba))
                except:
                    metrics["log_loss"] = None

            logger.info(f"✅ {model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Classification evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Calculate all regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of model being evaluated

        Returns:
            Dictionary with all metrics
        """
        try:
            metrics = {
                "model_name": model_name,
                "task_type": "regression"
            }

            # Core metrics
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2_score"] = float(r2_score(y_true, y_pred))

            # MAPE (Mean Absolute Percentage Error)
            try:
                metrics["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
            except:
                # Manual calculation if sklearn version doesn't have it
                mask = y_true != 0
                metrics["mape"] = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

            # Additional metrics
            metrics["max_error"] = float(np.max(np.abs(y_true - y_pred)))
            metrics["median_absolute_error"] = float(np.median(np.abs(y_true - y_pred)))

            # Explained variance
            from sklearn.metrics import explained_variance_score
            metrics["explained_variance"] = float(explained_variance_score(y_true, y_pred))

            # Residuals analysis
            residuals = y_true - y_pred
            metrics["residuals_mean"] = float(np.mean(residuals))
            metrics["residuals_std"] = float(np.std(residuals))

            logger.info(f"✅ {model_name} - R²: {metrics['r2_score']:.4f}, RMSE: {metrics['rmse']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Regression evaluation failed: {e}")
            return {"error": str(e)}

    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = "auto",
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            task_type: "classification", "regression", or "auto"
            model_name: Name of model

        Returns:
            Evaluation metrics dictionary
        """
        try:
            # Auto-detect task type
            if task_type == "auto":
                if y_test.dtype == 'object' or y_test.nunique() <= 20:
                    task_type = "classification"
                else:
                    task_type = "regression"

            # Get predictions
            y_pred = model.predict(X_test)

            # Get probabilities for classification
            y_pred_proba = None
            if task_type == "classification" and hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)

            # Evaluate based on task type
            if task_type == "classification":
                metrics = self.evaluate_classification(y_test, y_pred, y_pred_proba, model_name)
            else:
                metrics = self.evaluate_regression(y_test, y_pred, model_name)

            return metrics

        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {"error": str(e)}

    def compare_models(
        self,
        model_results: Dict[str, Dict[str, Any]],
        metric: str = "auto"
    ) -> Dict[str, Any]:
        """
        Compare multiple models and rank them.

        Args:
            model_results: Dictionary of {model_name: metrics_dict}
            metric: Metric to use for ranking ("auto", "accuracy", "r2_score", etc.)

        Returns:
            Comparison report with rankings
        """
        try:
            comparison = {
                "models_compared": len(model_results),
                "ranking": [],
                "best_model": None,
                "comparison_table": []
            }

            # Auto-select metric based on task type
            if metric == "auto":
                task_type = list(model_results.values())[0].get("task_type", "classification")
                if task_type == "classification":
                    metric = "f1_score"
                else:
                    metric = "r2_score"

            # Extract metric values
            model_scores = {}
            for model_name, metrics in model_results.items():
                if metric in metrics and metrics[metric] is not None:
                    model_scores[model_name] = metrics[metric]

            # Rank models (higher is better for most metrics, except loss/error)
            reverse = metric not in ["mse", "rmse", "mae", "log_loss", "mape"]
            ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=reverse)

            # Build ranking
            for rank, (model_name, score) in enumerate(ranked_models, 1):
                comparison["ranking"].append({
                    "rank": rank,
                    "model": model_name,
                    "score": float(score),
                    "metric": metric
                })

            comparison["best_model"] = ranked_models[0][0] if ranked_models else None
            comparison["best_score"] = float(ranked_models[0][1]) if ranked_models else None

            # Build comparison table
            for model_name, metrics in model_results.items():
                row = {"model": model_name}

                # Add key metrics
                if metrics.get("task_type") == "classification":
                    row["accuracy"] = metrics.get("accuracy")
                    row["precision"] = metrics.get("precision")
                    row["recall"] = metrics.get("recall")
                    row["f1_score"] = metrics.get("f1_score")
                    row["roc_auc"] = metrics.get("roc_auc")
                else:
                    row["r2_score"] = metrics.get("r2_score")
                    row["rmse"] = metrics.get("rmse")
                    row["mae"] = metrics.get("mae")
                    row["mape"] = metrics.get("mape")

                comparison["comparison_table"].append(row)

            logger.info(f"✅ Model comparison complete. Best: {comparison['best_model']} ({metric}: {comparison['best_score']:.4f})")

            self.evaluation_report["comparison"] = comparison

            return comparison

        except Exception as e:
            logger.error(f"❌ Model comparison failed: {e}")
            return {"error": str(e)}

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate confusion matrix as ASCII art (for logging/reports).

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class names

        Returns:
            String representation of confusion matrix
        """
        try:
            cm = confusion_matrix(y_true, y_pred)

            # Create ASCII representation
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(cm))]

            output = "\nConfusion Matrix:\n"
            output += "=" * 50 + "\n"

            # Header
            output += f"{'':15}"
            for name in class_names:
                output += f"{name:>15}"
            output += "\n"

            # Rows
            for i, name in enumerate(class_names):
                output += f"{name:15}"
                for j in range(len(cm)):
                    output += f"{cm[i, j]:>15}"
                output += "\n"

            return output

        except Exception as e:
            logger.error(f"❌ Confusion matrix plotting failed: {e}")
            return str(e)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed.

        Returns:
            Evaluation summary report
        """
        return self.evaluation_report


# Convenience functions
def quick_evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Quick model evaluation.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Model name

    Returns:
        Metrics dictionary
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)


def compare_multiple_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate and compare multiple models.

    Args:
        models: Dictionary of {model_name: trained_model}
        X_test: Test features
        y_test: Test target

    Returns:
        Comparison report
    """
    evaluator = ModelEvaluator()

    # Evaluate each model
    model_results = {}
    for model_name, model in models.items():
        metrics = evaluator.evaluate_model(model, X_test, y_test, model_name=model_name)
        model_results[model_name] = metrics

    # Compare
    comparison = evaluator.compare_models(model_results)

    return comparison
