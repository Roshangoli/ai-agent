"""
Decision Validator Module
Validates LLM decisions before execution to prevent pipeline failures.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DecisionValidator:
    """
    Validates LLM agent decisions before execution.
    Prevents invalid strategies from breaking the pipeline.
    """

    # Valid strategies for each decision type
    VALID_IMPUTATION_STRATEGIES = {
        "mean", "median", "mode", "forward_fill", "backward_fill",
        "zero", "constant", "drop", "auto"
    }

    VALID_OUTLIER_STRATEGIES = {
        "clip", "cap", "remove", "winsorize", "log_transform",
        "ignore", "cap_at_iqr", "cap_at_percentile"
    }

    VALID_ENCODING_METHODS = {
        "one_hot", "label", "target_encoding", "frequency_encoding",
        "hash_encoding", "ordinal", "binary"
    }

    VALID_SCALING_METHODS = {
        "standard", "minmax", "robust", "maxabs", "normalizer", "none"
    }

    VALID_IMBALANCE_STRATEGIES = {
        "smote", "adasyn", "random_oversample", "random_undersample",
        "class_weight", "none", "auto"
    }

    VALID_MODEL_NAMES = {
        # Classification
        "RandomForest", "ExtraTrees", "DecisionTree", "GradientBoosting",
        "AdaBoost", "LogisticRegression", "SGDClassifier", "SVM_Linear",
        "SVM_RBF", "SVM_Poly", "KNN", "GaussianNB", "MultinomialNB",
        "XGBoost", "LightGBM", "CatBoost",
        # Regression
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "SGDRegressor",
        "SVR_Linear", "SVR_RBF", "SVR_Poly"
    }

    def __init__(self):
        """Initialize validator."""
        self.validation_report = {
            "total_validations": 0,
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "errors": []
        }

    def validate_cleaning_decision(
        self,
        decision: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate data cleaning decisions.

        Args:
            decision: Cleaning decision from LLM

        Returns:
            Tuple of (is_valid, validation_report)
        """
        self.validation_report["total_validations"] += 1
        errors = []
        warnings = []
        valid = True

        # Validate structure
        if not isinstance(decision, dict):
            errors.append("Decision must be a dictionary")
            valid = False
            self.validation_report["failed"] += 1
            return False, {"errors": errors, "warnings": warnings}

        # Validate missing value strategies
        if "missing_values" in decision or "missing_value_decisions" in decision:
            mv_decisions = decision.get("missing_values") or decision.get("missing_value_decisions", [])

            # Handle both dict and list formats
            if isinstance(mv_decisions, list):
                for item in mv_decisions:
                    col = item.get("column")
                    strategy_dict = item.get("decision", {})
                    strategy = strategy_dict.get("strategy", "").lower()

                    if strategy and strategy not in self.VALID_IMPUTATION_STRATEGIES:
                        errors.append(
                            f"Invalid imputation strategy '{strategy}' for column '{col}'. "
                            f"Valid: {self.VALID_IMPUTATION_STRATEGIES}"
                        )
                        valid = False
            elif isinstance(mv_decisions, dict):
                for col, strategy_info in mv_decisions.items():
                    strategy = strategy_info.get("strategy", "").lower()

                    if strategy and strategy not in self.VALID_IMPUTATION_STRATEGIES:
                        errors.append(
                            f"Invalid imputation strategy '{strategy}' for column '{col}'. "
                            f"Valid: {self.VALID_IMPUTATION_STRATEGIES}"
                        )
                        valid = False

        # Validate outlier strategies
        if "outlier_decisions" in decision or "outliers" in decision:
            outlier_decisions = decision.get("outlier_decisions") or decision.get("outliers", [])

            if isinstance(outlier_decisions, list):
                for item in outlier_decisions:
                    col = item.get("column")
                    decision_dict = item.get("decision", {})
                    action = decision_dict.get("action", "").lower()

                    if action and action not in self.VALID_OUTLIER_STRATEGIES:
                        errors.append(
                            f"Invalid outlier strategy '{action}' for column '{col}'. "
                            f"Valid: {self.VALID_OUTLIER_STRATEGIES}"
                        )
                        valid = False
            elif isinstance(outlier_decisions, dict):
                for col, strategy_info in outlier_decisions.items():
                    action = strategy_info.get("action", "").lower()

                    if action and action not in self.VALID_OUTLIER_STRATEGIES:
                        errors.append(
                            f"Invalid outlier strategy '{action}' for column '{col}'. "
                            f"Valid: {self.VALID_OUTLIER_STRATEGIES}"
                        )
                        valid = False

        # Update report
        if valid:
            self.validation_report["passed"] += 1
            logger.info("✅ Cleaning decision validation PASSED")
        else:
            self.validation_report["failed"] += 1
            self.validation_report["errors"].extend(errors)
            logger.error(f"❌ Cleaning decision validation FAILED: {errors}")

        if warnings:
            self.validation_report["warnings"] += len(warnings)
            logger.warning(f"⚠️ Cleaning decision warnings: {warnings}")

        return valid, {
            "valid": valid,
            "errors": errors,
            "warnings": warnings
        }

    def validate_encoding_decision(
        self,
        decision: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate feature encoding decisions.

        Args:
            decision: Encoding decision from LLM

        Returns:
            Tuple of (is_valid, validation_report)
        """
        self.validation_report["total_validations"] += 1
        errors = []
        warnings = []
        valid = True

        if "encoding_decisions" in decision:
            encoding_decisions = decision["encoding_decisions"]

            if isinstance(encoding_decisions, list):
                for item in encoding_decisions:
                    col = item.get("column")
                    decision_dict = item.get("decision", {})
                    method = decision_dict.get("method", "").lower()

                    if method and method not in self.VALID_ENCODING_METHODS:
                        errors.append(
                            f"Invalid encoding method '{method}' for column '{col}'. "
                            f"Valid: {self.VALID_ENCODING_METHODS}"
                        )
                        valid = False

                    # Warn if target encoding without target column
                    if method == "target_encoding" and not decision_dict.get("target_column"):
                        warnings.append(
                            f"Target encoding requested for '{col}' but no target column specified"
                        )

        # Update report
        if valid:
            self.validation_report["passed"] += 1
            logger.info("✅ Encoding decision validation PASSED")
        else:
            self.validation_report["failed"] += 1
            self.validation_report["errors"].extend(errors)
            logger.error(f"❌ Encoding decision validation FAILED: {errors}")

        if warnings:
            self.validation_report["warnings"] += len(warnings)
            logger.warning(f"⚠️ Encoding decision warnings: {warnings}")

        return valid, {
            "valid": valid,
            "errors": errors,
            "warnings": warnings
        }

    def validate_preprocessing_decision(
        self,
        decision: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate preprocessing decisions (scaling, imbalance handling).

        Args:
            decision: Preprocessing decision from LLM

        Returns:
            Tuple of (is_valid, validation_report)
        """
        self.validation_report["total_validations"] += 1
        errors = []
        warnings = []
        valid = True

        # Validate scaler choice
        if "scaler" in decision or "scaling_method" in decision:
            scaler = (decision.get("scaler") or decision.get("scaling_method", "")).lower()

            if scaler and scaler not in self.VALID_SCALING_METHODS:
                errors.append(
                    f"Invalid scaling method '{scaler}'. "
                    f"Valid: {self.VALID_SCALING_METHODS}"
                )
                valid = False

        # Validate imbalance strategy
        if "imbalance_strategy" in decision:
            strategy = decision["imbalance_strategy"].lower()

            if strategy and strategy not in self.VALID_IMBALANCE_STRATEGIES:
                errors.append(
                    f"Invalid imbalance strategy '{strategy}'. "
                    f"Valid: {self.VALID_IMBALANCE_STRATEGIES}"
                )
                valid = False

        # Update report
        if valid:
            self.validation_report["passed"] += 1
            logger.info("✅ Preprocessing decision validation PASSED")
        else:
            self.validation_report["failed"] += 1
            self.validation_report["errors"].extend(errors)
            logger.error(f"❌ Preprocessing decision validation FAILED: {errors}")

        if warnings:
            self.validation_report["warnings"] += len(warnings)

        return valid, {
            "valid": valid,
            "errors": errors,
            "warnings": warnings
        }

    def validate_model_selection(
        self,
        models: List[str]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate model selection.

        Args:
            models: List of model names

        Returns:
            Tuple of (is_valid, validation_report)
        """
        self.validation_report["total_validations"] += 1
        errors = []
        warnings = []
        valid = True

        if not isinstance(models, list):
            errors.append("Models must be a list")
            valid = False
        else:
            for model in models:
                if model not in self.VALID_MODEL_NAMES:
                    errors.append(
                        f"Invalid model name '{model}'. "
                        f"Valid: {self.VALID_MODEL_NAMES}"
                    )
                    valid = False

            if len(models) == 0:
                errors.append("At least one model must be selected")
                valid = False
            elif len(models) > 10:
                warnings.append(
                    f"Training {len(models)} models may be slow. Consider reducing to top 5."
                )

        # Update report
        if valid:
            self.validation_report["passed"] += 1
            logger.info(f"✅ Model selection validation PASSED: {models}")
        else:
            self.validation_report["failed"] += 1
            self.validation_report["errors"].extend(errors)
            logger.error(f"❌ Model selection validation FAILED: {errors}")

        if warnings:
            self.validation_report["warnings"] += len(warnings)

        return valid, {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "validated_models": [m for m in models if m in self.VALID_MODEL_NAMES]
        }

    def get_fallback_strategy(
        self,
        decision_type: str,
        column_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get safe fallback strategy when validation fails.

        Args:
            decision_type: Type of decision (imputation, outlier, encoding, etc.)
            column_info: Optional column metadata for smart fallback

        Returns:
            Safe fallback strategy
        """
        fallbacks = {
            "imputation": "median",  # Safe for both numerical (median) and auto-detects categorical (mode)
            "outlier": "clip",  # Conservative - keeps outliers but limits them
            "encoding": "one_hot",  # Safe default for low cardinality
            "scaling": "standard",  # Most common choice
            "imbalance": "class_weight"  # Doesn't modify data
        }

        fallback = fallbacks.get(decision_type, "auto")

        logger.warning(
            f"⚠️ Using fallback strategy '{fallback}' for {decision_type} "
            f"due to validation failure"
        )

        return fallback

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validations performed.

        Returns:
            Validation summary dictionary
        """
        success_rate = (
            (self.validation_report["passed"] / self.validation_report["total_validations"] * 100)
            if self.validation_report["total_validations"] > 0
            else 0
        )

        return {
            **self.validation_report,
            "success_rate": success_rate
        }


# Convenience functions
def validate_and_fallback(
    validator: DecisionValidator,
    decision: Dict[str, Any],
    decision_type: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate decision and provide fallback if invalid.

    Args:
        validator: DecisionValidator instance
        decision: Decision to validate
        decision_type: Type of decision

    Returns:
        Tuple of (validated_decision, validation_report)
    """
    if decision_type == "cleaning":
        valid, report = validator.validate_cleaning_decision(decision)
    elif decision_type == "encoding":
        valid, report = validator.validate_encoding_decision(decision)
    elif decision_type == "preprocessing":
        valid, report = validator.validate_preprocessing_decision(decision)
    else:
        valid, report = True, {"valid": True, "errors": [], "warnings": []}

    if not valid:
        # Use fallback
        logger.warning(f"⚠️ Decision validation failed, using fallback strategy")
        fallback_strategy = validator.get_fallback_strategy(decision_type)

        # Modify decision to use fallback
        if decision_type == "cleaning" and "missing_values" in decision:
            for col in decision["missing_values"]:
                decision["missing_values"][col]["strategy"] = fallback_strategy

    return decision, report


if __name__ == "__main__":
    # Test the validator
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*60)
    print("DECISION VALIDATOR TEST")
    print("="*60)

    validator = DecisionValidator()

    # Test 1: Valid cleaning decision
    print("\n1. Testing VALID cleaning decision...")
    valid_decision = {
        "missing_value_decisions": [
            {
                "column": "age",
                "decision": {
                    "strategy": "median",
                    "reasoning": "Right-skewed distribution"
                }
            }
        ]
    }
    valid, report = validator.validate_cleaning_decision(valid_decision)
    print(f"   Result: {'PASS' if valid else 'FAIL'}")
    print(f"   Report: {report}")

    # Test 2: Invalid cleaning decision
    print("\n2. Testing INVALID cleaning decision...")
    invalid_decision = {
        "missing_value_decisions": [
            {
                "column": "age",
                "decision": {
                    "strategy": "interpolate",  # NOT in valid strategies!
                    "reasoning": "Linear interpolation"
                }
            }
        ]
    }
    valid, report = validator.validate_cleaning_decision(invalid_decision)
    print(f"   Result: {'PASS' if valid else 'FAIL'}")
    print(f"   Report: {report}")

    # Test 3: Valid model selection
    print("\n3. Testing VALID model selection...")
    valid_models = ["RandomForest", "XGBoost", "LogisticRegression"]
    valid, report = validator.validate_model_selection(valid_models)
    print(f"   Result: {'PASS' if valid else 'FAIL'}")
    print(f"   Report: {report}")

    # Test 4: Invalid model selection
    print("\n4. Testing INVALID model selection...")
    invalid_models = ["RandomForest", "MagicModel", "SuperAI"]  # Invalid models!
    valid, report = validator.validate_model_selection(invalid_models)
    print(f"   Result: {'PASS' if valid else 'FAIL'}")
    print(f"   Report: {report}")

    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    summary = validator.get_validation_summary()
    print(f"Total validations: {summary['total_validations']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print("="*60 + "\n")
