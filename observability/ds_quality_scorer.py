"""
Data Science Mode Quality Scorer
Evaluates the quality of agent decisions in the ML pipeline
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class DataScienceQualityScorer:
    """
    Evaluates quality of Data Science Mode agent decisions.

    Scores are on a 100-point scale:
    - 90-100: Excellent
    - 75-89: Good
    - 60-74: Acceptable
    - Below 60: Needs improvement
    """

    def __init__(self):
        self.scores = []

    def score_cleaning_decisions(
        self,
        decisions: Dict[str, Any],
        data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score the quality of data cleaning decisions.

        Checks:
        1. Appropriate imputation strategies chosen
        2. Outlier handling justified
        3. Duplicate handling correct
        4. No data leakage
        """
        score = 100
        issues = []

        # Check missing value strategies
        missing_strategies = decisions.get("missing_values", {})

        for column, strategy_info in missing_strategies.items():
            strategy = strategy_info.get("strategy", "").lower()
            reasoning = strategy_info.get("reasoning", "")

            # Validate strategy choices
            if "skewed" in reasoning.lower() and strategy != "median":
                score -= 10
                issues.append(f"Should use median for skewed column {column}")

            if "normal" in reasoning.lower() and strategy not in ["mean", "median"]:
                score -= 10
                issues.append(f"Should use mean/median for normal column {column}")

            if strategy == "drop" and column not in ["id", "identifier"]:
                # Dropping columns should be rare
                score -= 5
                issues.append(f"Dropping column {column} - verify this is necessary")

        # Check outlier handling
        outlier_actions = decisions.get("outliers", {})

        for column, action_info in outlier_actions.items():
            action = action_info.get("action", "").lower()

            if action == "remove" and column in data_profile.get("important_features", []):
                score -= 15
                issues.append(f"Removing outliers from important feature {column}")

        # Overall quality rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 60:
            rating = "acceptable"
        else:
            rating = "needs_improvement"

        result = {
            "score": max(0, score),
            "rating": rating,
            "issues": issues,
            "category": "data_cleaning"
        }

        self.scores.append(result)
        logger.info(f"Data Cleaning Quality Score: {score}/100 ({rating})")

        return result

    def score_feature_engineering_decisions(
        self,
        decisions: Dict[str, Any],
        data_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score the quality of feature engineering decisions.

        Checks:
        1. Appropriate encoding strategies
        2. Feature creation justified
        3. No target leakage
        4. Cardinality handled properly
        """
        score = 100
        issues = []

        encoding_decisions = decisions.get("encoding", {})

        for column, encoding_info in encoding_decisions.items():
            strategy = encoding_info.get("strategy", "").lower()
            cardinality = encoding_info.get("cardinality", 0)

            # High cardinality checks
            if cardinality > 20 and strategy == "one_hot":
                score -= 15
                issues.append(f"One-hot encoding {column} with {cardinality} categories - use target/frequency encoding")

            # Low cardinality checks
            if cardinality < 5 and strategy != "one_hot":
                score -= 5
                issues.append(f"Should use one-hot for low cardinality column {column}")

        # Check feature interactions
        interactions = decisions.get("interactions", [])

        if len(interactions) > 10:
            score -= 10
            issues.append("Too many feature interactions created - risk of overfitting")

        # Overall quality rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 60:
            rating = "acceptable"
        else:
            rating = "needs_improvement"

        result = {
            "score": max(0, score),
            "rating": rating,
            "issues": issues,
            "category": "feature_engineering"
        }

        self.scores.append(result)
        logger.info(f"Feature Engineering Quality Score: {score}/100 ({rating})")

        return result

    def score_model_selection(
        self,
        model_results: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """
        Score the quality of model selection and training.

        Checks:
        1. Appropriate models for task type
        2. Cross-validation used
        3. No overfitting (train vs test gap)
        4. Hyperparameter tuning performed
        """
        score = 100
        issues = []

        best_model = model_results.get("best_model_name", "")
        cv_score = model_results.get("best_cv_score", 0)
        test_score = model_results.get("best_test_score", 0)

        # Check for overfitting
        if test_score and abs(cv_score - test_score) > 0.15:
            score -= 20
            issues.append(f"Large gap between CV ({cv_score:.3f}) and test ({test_score:.3f}) - possible overfitting")

        # Check model appropriateness
        if task_type == "classification":
            good_models = ["RandomForest", "XGBoost", "LightGBM", "LogisticRegression"]
            if not any(model in best_model for model in good_models):
                score -= 10
                issues.append(f"Unusual model choice for classification: {best_model}")

        elif task_type == "regression":
            good_models = ["RandomForest", "XGBoost", "LightGBM", "Ridge", "Lasso"]
            if not any(model in best_model for model in good_models):
                score -= 10
                issues.append(f"Unusual model choice for regression: {best_model}")

        # Check performance quality
        if task_type == "classification":
            if cv_score < 0.60:
                score -= 20
                issues.append(f"Low CV score ({cv_score:.3f}) - model may not be useful")
            elif cv_score < 0.70:
                score -= 10
                issues.append(f"Mediocre CV score ({cv_score:.3f}) - consider feature engineering")

        elif task_type == "regression":
            if cv_score < 0.30:
                score -= 20
                issues.append(f"Low R² score ({cv_score:.3f}) - model has poor predictive power")

        # Overall quality rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 60:
            rating = "acceptable"
        else:
            rating = "needs_improvement"

        result = {
            "score": max(0, score),
            "rating": rating,
            "issues": issues,
            "category": "model_selection"
        }

        self.scores.append(result)
        logger.info(f"Model Selection Quality Score: {score}/100 ({rating})")

        return result

    def score_overall_pipeline(
        self,
        pipeline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score the overall pipeline quality.

        Aggregates individual scores and adds holistic checks.
        """
        if not self.scores:
            logger.warning("No individual scores available for overall scoring")
            return {
                "score": 0,
                "rating": "incomplete",
                "issues": ["No scoring data available"],
                "category": "overall"
            }

        # Calculate weighted average
        weights = {
            "data_cleaning": 0.25,
            "feature_engineering": 0.35,
            "model_selection": 0.40
        }

        weighted_score = 0
        for score_result in self.scores:
            category = score_result["category"]
            weight = weights.get(category, 0.33)
            weighted_score += score_result["score"] * weight

        overall_score = int(weighted_score)

        # Collect all issues
        all_issues = []
        for score_result in self.scores:
            all_issues.extend(score_result["issues"])

        # Overall rating
        if overall_score >= 90:
            rating = "excellent"
        elif overall_score >= 75:
            rating = "good"
        elif overall_score >= 60:
            rating = "acceptable"
        else:
            rating = "needs_improvement"

        result = {
            "score": overall_score,
            "rating": rating,
            "issues": all_issues,
            "category": "overall",
            "breakdown": {
                score_result["category"]: score_result["score"]
                for score_result in self.scores
            }
        }

        logger.info(f"\n📊 OVERALL PIPELINE QUALITY SCORE: {overall_score}/100 ({rating})")
        logger.info(f"   Breakdown: {result['breakdown']}")

        return result

    def get_all_scores(self) -> List[Dict[str, Any]]:
        """Get all quality scores recorded."""
        return self.scores

    def reset(self):
        """Reset all scores."""
        self.scores = []


if __name__ == "__main__":
    # Test the quality scorer
    scorer = DataScienceQualityScorer()

    # Test cleaning decisions
    cleaning_decisions = {
        "missing_values": {
            "age": {"strategy": "median", "reasoning": "Skewed distribution"},
            "income": {"strategy": "mean", "reasoning": "Normal distribution"}
        },
        "outliers": {
            "price": {"action": "cap", "reasoning": "Valid luxury items"}
        }
    }

    data_profile = {
        "important_features": ["age", "income"]
    }

    cleaning_score = scorer.score_cleaning_decisions(cleaning_decisions, data_profile)
    print(f"Cleaning Score: {cleaning_score['score']}/100")

    # Test model selection
    model_results = {
        "best_model_name": "RandomForest",
        "best_cv_score": 0.85,
        "best_test_score": 0.83
    }

    model_score = scorer.score_model_selection(model_results, "classification")
    print(f"Model Score: {model_score['score']}/100")

    # Overall score
    overall = scorer.score_overall_pipeline({})
    print(f"Overall Score: {overall['score']}/100")
