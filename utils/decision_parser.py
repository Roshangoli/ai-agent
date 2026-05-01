"""
Decision Parser Module
Parses agent JSON responses and executes decisions.
"""

import logging
import json
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class DecisionParser:
    """
    Parses and validates agent JSON decisions.
    """

    def __init__(self):
        """Initialize DecisionParser."""
        self.parsed_decisions = []

    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from text that may contain markdown or other content.

        Args:
            text: Raw text from agent response

        Returns:
            Parsed JSON dict or None
        """
        try:
            # Try direct JSON parse first
            if text.strip().startswith('{'):
                return json.loads(text.strip())

            # Try extracting from markdown code blocks
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # ```json ... ```
                r'```\s*(.*?)\s*```',       # ``` ... ```
                r'\{.*\}',                   # Any JSON object
            ]

            for pattern in json_patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    for match in matches:
                        try:
                            return json.loads(match)
                        except:
                            continue

            logger.warning("⚠️ Could not extract JSON from agent response")
            return None

        except Exception as e:
            logger.error(f"❌ JSON extraction failed: {e}")
            return None

    def parse_cleaning_decision(self, response: str) -> Dict[str, Any]:
        """
        Parse Data Cleaning Agent decisions.

        Expected format:
        {
          "missing_values": {
            "age": {"strategy": "median", "reasoning": "Skewed distribution"},
            "income": {"strategy": "mean", "reasoning": "Normal distribution"}
          },
          "outliers": {
            "price": {"action": "cap", "reasoning": "Valid luxury items"},
            "quantity": {"action": "remove", "reasoning": "Data errors"}
          },
          "duplicates": {"action": "remove", "reasoning": "Exact copies found"}
        }

        Args:
            response: Agent response text

        Returns:
            Parsed and validated cleaning decisions
        """
        try:
            decisions = self.extract_json_from_text(response)

            if not decisions:
                return {"error": "No valid JSON found"}

            # Validate structure
            valid_decisions = {
                "missing_values": {},
                "outliers": {},
                "duplicates": {},
                "text_cleaning": {},
                "date_standardization": {}
            }

            # Parse missing value decisions
            if "missing_values" in decisions:
                for col, decision in decisions["missing_values"].items():
                    strategy = decision.get("strategy", "auto")
                    if strategy in ["mean", "median", "mode", "knn", "mice", "drop", "forward_fill", "backward_fill"]:
                        valid_decisions["missing_values"][col] = {
                            "strategy": strategy,
                            "reasoning": decision.get("reasoning", "")
                        }

            # Parse outlier decisions
            if "outliers" in decisions:
                for col, decision in decisions["outliers"].items():
                    action = decision.get("action", "keep")
                    if action in ["keep", "cap", "remove", "clip"]:
                        valid_decisions["outliers"][col] = {
                            "action": action,
                            "reasoning": decision.get("reasoning", "")
                        }

            # Parse duplicate decision
            if "duplicates" in decisions:
                action = decisions["duplicates"].get("action", "keep")
                if action in ["keep", "remove"]:
                    valid_decisions["duplicates"] = {
                        "action": action,
                        "reasoning": decisions["duplicates"].get("reasoning", "")
                    }

            logger.info(f"✅ Parsed cleaning decisions: {len(valid_decisions['missing_values'])} imputation strategies, {len(valid_decisions['outliers'])} outlier actions")

            return valid_decisions

        except Exception as e:
            logger.error(f"❌ Cleaning decision parsing failed: {e}")
            return {"error": str(e)}

    def parse_eda_decision(self, response: str) -> Dict[str, Any]:
        """
        Parse EDA Agent decisions.

        Expected format:
        {
          "features_to_drop": ["feature1", "feature2"],
          "correlations_found": {"feature1": {"feature2": 0.95}},
          "recommendations": [
            "Create interaction feature: sqft × bathrooms",
            "Apply log transformation to income"
          ],
          "class_imbalance": {"detected": true, "action": "SMOTE"}
        }

        Args:
            response: Agent response text

        Returns:
            Parsed EDA insights and recommendations
        """
        try:
            decisions = self.extract_json_from_text(response)

            if not decisions:
                return {"error": "No valid JSON found"}

            valid_decisions = {
                "features_to_drop": [],
                "transformations_recommended": {},
                "interactions_to_create": [],
                "imbalance_action": None
            }

            # Features to drop
            if "features_to_drop" in decisions:
                valid_decisions["features_to_drop"] = decisions["features_to_drop"]

            # Transformation recommendations
            if "transformations" in decisions:
                valid_decisions["transformations_recommended"] = decisions["transformations"]

            # Feature interactions
            if "interactions" in decisions or "recommendations" in decisions:
                interactions = decisions.get("interactions", decisions.get("recommendations", []))
                valid_decisions["interactions_to_create"] = interactions

            # Imbalance handling
            if "class_imbalance" in decisions:
                valid_decisions["imbalance_action"] = decisions["class_imbalance"].get("action", "none")

            logger.info(f"✅ Parsed EDA decisions: {len(valid_decisions['features_to_drop'])} features to drop, {len(valid_decisions['transformations_recommended'])} transformations")

            return valid_decisions

        except Exception as e:
            logger.error(f"❌ EDA decision parsing failed: {e}")
            return {"error": str(e)}

    def parse_feature_engineering_decision(self, response: str) -> Dict[str, Any]:
        """
        Parse Feature Engineering Agent decisions.

        Expected format:
        {
          "categorical_encoding": {
            "city": {"method": "target_encoding", "reasoning": "High cardinality (347 values)"},
            "gender": {"method": "one_hot", "reasoning": "Low cardinality (2 values)"}
          },
          "numerical_transformations": {
            "income": {"method": "log", "reasoning": "Right-skewed distribution"},
            "age": {"method": "none", "reasoning": "Normal distribution"}
          },
          "features_to_create": [
            {"type": "ratio", "name": "price_per_sqft", "numerator": "price", "denominator": "sqft"},
            {"type": "interaction", "name": "beds_x_baths", "col1": "bedrooms", "col2": "bathrooms"}
          ],
          "datetime_extractions": ["month", "dayofweek", "is_weekend", "quarter"]
        }

        Args:
            response: Agent response text

        Returns:
            Parsed feature engineering plan
        """
        try:
            decisions = self.extract_json_from_text(response)

            if not decisions:
                return {"error": "No valid JSON found"}

            valid_decisions = {
                "categorical_encoding": {},
                "numerical_transformations": {},
                "features_to_create": [],
                "datetime_extractions": []
            }

            # Categorical encoding
            if "categorical_encoding" in decisions:
                for col, decision in decisions["categorical_encoding"].items():
                    method = decision.get("method", "auto")
                    if method in ["one_hot", "label", "target", "frequency", "hash", "binary", "ordinal"]:
                        valid_decisions["categorical_encoding"][col] = {
                            "method": method,
                            "reasoning": decision.get("reasoning", "")
                        }

            # Numerical transformations
            if "numerical_transformations" in decisions:
                for col, decision in decisions["numerical_transformations"].items():
                    method = decision.get("method", "none")
                    if method in ["log", "log1p", "sqrt", "boxcox", "yeo-johnson", "quantile", "none"]:
                        valid_decisions["numerical_transformations"][col] = {
                            "method": method,
                            "reasoning": decision.get("reasoning", "")
                        }

            # Features to create
            if "features_to_create" in decisions:
                valid_decisions["features_to_create"] = decisions["features_to_create"]

            # Datetime extractions
            if "datetime_extractions" in decisions:
                valid_decisions["datetime_extractions"] = decisions["datetime_extractions"]

            logger.info(f"✅ Parsed feature engineering: {len(valid_decisions['categorical_encoding'])} encodings, {len(valid_decisions['numerical_transformations'])} transformations")

            return valid_decisions

        except Exception as e:
            logger.error(f"❌ Feature engineering decision parsing failed: {e}")
            return {"error": str(e)}

    def parse_ml_training_decision(self, response: str) -> Dict[str, Any]:
        """
        Parse ML Training Agent decisions.

        Expected format:
        {
          "task_type": "classification",
          "models_to_try": ["RandomForest", "XGBoost", "LogisticRegression"],
          "reasoning": "Small dataset (500 rows), avoid deep learning. High cardinality handled by tree models.",
          "hyperparameters": {
            "RandomForest": {"n_estimators": 100, "max_depth": 10},
            "XGBoost": {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
          },
          "cv_strategy": "StratifiedKFold",
          "evaluation_metrics": ["accuracy", "f1_score", "roc_auc"]
        }

        Args:
            response: Agent response text

        Returns:
            Parsed ML training plan
        """
        try:
            decisions = self.extract_json_from_text(response)

            if not decisions:
                return {"error": "No valid JSON found"}

            valid_decisions = {
                "task_type": "auto",
                "models_to_try": [],
                "hyperparameters": {},
                "cv_strategy": "auto",
                "evaluation_metrics": []
            }

            # Task type
            if "task_type" in decisions:
                task = decisions["task_type"].lower()
                if task in ["classification", "regression", "binary_classification", "multi_class"]:
                    valid_decisions["task_type"] = task

            # Models to try
            if "models_to_try" in decisions:
                # Validate model names exist
                valid_models = [
                    "RandomForest", "XGBoost", "LightGBM", "CatBoost",
                    "LogisticRegression", "LinearRegression", "Ridge", "Lasso",
                    "SVM", "KNN", "DecisionTree", "GradientBoosting",
                    "AdaBoost", "ExtraTrees", "NeuralNetwork"
                ]
                for model in decisions["models_to_try"]:
                    if model in valid_models:
                        valid_decisions["models_to_try"].append(model)

            # Hyperparameters
            if "hyperparameters" in decisions:
                valid_decisions["hyperparameters"] = decisions["hyperparameters"]

            # CV strategy
            if "cv_strategy" in decisions:
                strategy = decisions["cv_strategy"]
                if strategy in ["KFold", "StratifiedKFold", "TimeSeriesSplit", "GroupKFold"]:
                    valid_decisions["cv_strategy"] = strategy

            # Evaluation metrics
            if "evaluation_metrics" in decisions:
                valid_decisions["evaluation_metrics"] = decisions["evaluation_metrics"]

            logger.info(f"✅ Parsed ML training plan: {len(valid_decisions['models_to_try'])} models selected")

            return valid_decisions

        except Exception as e:
            logger.error(f"❌ ML training decision parsing failed: {e}")
            return {"error": str(e)}

    def get_parsing_summary(self) -> Dict[str, Any]:
        """
        Get summary of all parsed decisions.

        Returns:
            Dictionary with parsing summary
        """
        return {
            "total_decisions_parsed": len(self.parsed_decisions),
            "decisions": self.parsed_decisions
        }


# Convenience functions for quick parsing
def parse_agent_response(response: str, agent_type: str) -> Dict[str, Any]:
    """
    Quick parse for any agent type.

    Args:
        response: Agent response text
        agent_type: "cleaning", "eda", "feature_engineering", "ml_training"

    Returns:
        Parsed decisions dict
    """
    parser = DecisionParser()

    if agent_type == "cleaning":
        return parser.parse_cleaning_decision(response)
    elif agent_type == "eda":
        return parser.parse_eda_decision(response)
    elif agent_type == "feature_engineering":
        return parser.parse_feature_engineering_decision(response)
    elif agent_type == "ml_training":
        return parser.parse_ml_training_decision(response)
    else:
        return {"error": f"Unknown agent type: {agent_type}"}
