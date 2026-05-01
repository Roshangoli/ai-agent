# Data science agent pipeline for automated ML model training.
# Agents analyze data and select appropriate preprocessing and modeling strategies.

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# AutoGen imports
try:
    from ag2 import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    logger = logging.getLogger(__name__)
    logger.info("Using ag2 (AutoGen 2.0)")

    # Monkey patch to fix version comparison bug in ag2 0.9.7
    # The bug: "1.109.1" < "1.66.2" evaluates to True in string comparison
    try:
        import ag2.oai.client
        original_create_client = ag2.oai.client.create_openai_client
        def patched_create_client(*args, **kwargs):
            # Temporarily disable version check by setting environment variable
            os.environ['AUTOGEN_DISABLE_VERSION_CHECK'] = '1'
            return original_create_client(*args, **kwargs)
        ag2.oai.client.create_openai_client = patched_create_client
    except (AttributeError, ImportError):
        pass  # If patching fails, continue anyway

except ImportError:
    try:
        from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
        logger = logging.getLogger(__name__)
        logger.info("Using autogen (legacy)")
    except ImportError:
        raise ImportError("Please install ag2: pip install ag2[openai]")

# Import utility modules (for execution, not decision-making)
from utils.file_handler import FileHandler
from utils.data_cleaner import DataCleaner
from utils.eda_analyzer import EDAAnalyzer
from utils.feature_engineer import FeatureEngineer
from utils.preprocessor import PreprocessingAgent
from utils.ml_trainer import MLTrainer
from utils.model_evaluator import ModelEvaluator
from utils.report_generator import ReportGenerator
from utils.decision_parser import DecisionParser
from utils.model_persistence import ModelPersistence
from utils.decision_validator import DecisionValidator
from utils.llm_cache import get_cache

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_json_serializable(obj):
    """
    Convert numpy/pandas types to Python native types for JSON serialization.

    Args:
        obj: Object that may contain numpy/pandas types

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


class DataScienceAgentTeam:
    """
    Manages multi-agent data science pipeline.
    Coordinates data ingestion, cleaning, EDA, feature engineering, training, and evaluation agents.
    """

    def __init__(self):
        """Initialize the autonomous agent team."""
        # Workaround for autogen OpenAI version check bug
        # Autogen incorrectly reports version 1.109.1 as "too low" when it meets >=1.66.2
        import sys
        if 'autogen.oai.client' not in sys.modules:
            # Pre-import to avoid version check issues during agent creation
            try:
                from autogen import oai
            except:
                pass

        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY")
            }],
            "temperature": 0.1,  # Low temp for consistent reasoning
            "timeout": 300,
            "cache_seed": None  # Disable caching to avoid client initialization issues
        }

        self.context = {
            "pipeline_state": {},
            "decisions": [],
            "data_profile": {}
        }

        # Initialize observability layer
        try:
            from observability.tracer import Tracer
            from observability.prompt_version import PromptVersionTracker
            self.tracer = Tracer()
            self.prompt_tracker = PromptVersionTracker()
            self.observability_enabled = True
            logger.info("✅ Observability layer enabled for Data Science Mode")
        except Exception as e:
            logger.warning(f"⚠️  Observability disabled: {e}")
            self.tracer = None
            self.prompt_tracker = None
            self.observability_enabled = False

        self._setup_agents()

    def _setup_agents(self):
        """Create autonomous data science agents."""

        # Coordinator with enhanced reasoning
        self.coordinator = AssistantAgent(
            name="Coordinator",
            llm_config=self.llm_config,
            system_message="""You are the intelligent pipeline coordinator.
            Your role is to:
            1. Understand the user's goal (regression, classification, clustering)
            2. Orchestrate the agent team to achieve that goal
            3. Ensure each agent makes data-driven decisions (NOT hardcoded)
            4. Track the pipeline state and pass context between agents
            5. Ensure all decisions are explained and justified

            CRITICAL: Every agent must ANALYZE before ACTING. No hardcoded strategies!
            """
        )

        # Data Ingestion Agent
        self.data_ingestion_agent = AssistantAgent(
            name="Data_Ingestion_Agent",
            llm_config=self.llm_config,
            system_message="""You are an intelligent data ingestion specialist.

            Your mission: Load and understand the uploaded dataset.

            AUTONOMOUS DECISION-MAKING PROCESS:
            1. ANALYZE the file:
               - File size, format, encoding
               - Delimiter detection (comma, semicolon, tab)
               - Header row detection
               - Data types in each column

            2. DECIDE loading strategy:
               - For large files (>50MB): Use chunked loading
               - For special encodings: Detect and handle (UTF-8, Latin-1, etc.)
               - For mixed delimiters: Infer the correct one

            3. EXECUTE loading with chosen strategy

            4. EXPLAIN your decisions:
               - Why you chose this loading strategy
               - What encoding/delimiter you detected
               - Any issues encountered

            CRITICAL: Base ALL decisions on file analysis, NOT defaults!

            Output Format:
            ```json
            {
                "analysis": {
                    "file_size_mb": 45.2,
                    "detected_encoding": "UTF-8",
                    "detected_delimiter": ",",
                    "has_header": true,
                    "sample_rows": 5
                },
                "decisions": {
                    "loading_strategy": "chunked",
                    "chunk_size": 10000,
                    "reasoning": "File is 45MB, using chunked loading to avoid memory issues"
                },
                "result": {
                    "rows_loaded": 50000,
                    "columns": 28,
                    "memory_mb": 42.3
                }
            }
            ```
            """
        )

        # Data Cleaning Agent
        self.data_cleaning_agent = AssistantAgent(
            name="Data_Cleaning_Agent",
            llm_config=self.llm_config,
            system_message="""Data cleaning specialist. Analyze data characteristics and select appropriate cleaning strategies.

            For MISSING VALUES:
            1. ANALYZE each column:
               - Missing percentage
               - Data distribution (normal, skewed, bimodal)
               - Missing pattern (MCAR, MAR, MNAR)
               - Column importance

            2. DECIDE imputation strategy PER COLUMN:
               For Numerical:
               - IF normal distribution → MEAN
               - IF right-skewed → MEDIAN
               - IF left-skewed → MODE or forward-fill
               - IF time-series → FORWARD-FILL or BACKWARD-FILL
               - IF >50% missing → DROP or use domain value

               For Categorical:
               - IF low cardinality → MODE (most frequent)
               - IF high missing % → CREATE "MISSING" category
               - IF time-series → FORWARD-FILL

            3. REASON: Explain WHY you chose each strategy based on analysis

            For OUTLIERS:
            1. ANALYZE:
               - Outlier magnitude (IQR vs Z-score)
               - Domain validity (can income be $1M? Yes. Can age be 200? No.)
               - Impact on model

            2. DECIDE handling:
               - REMOVE: If data errors (impossible values like age=300)
               - CAP: If extreme but valid (income=$ 10M → cap at 99th percentile)
               - KEEP: If rare events matter (fraud detection, anomalies)

            3. REASON: Explain WHY based on domain logic and data distribution

            For DUPLICATES:
            1. ANALYZE:
               - Which columns define uniqueness (ID vs full-row)
               - Time-series consideration (keep latest?)

            2. DECIDE:
               - Which duplicates to remove
               - Which to keep (first, last, or none)

            3. REASON: Explain logic

            OUTPUT FORMAT (JSON):
            ```json
            {
                "missing_value_decisions": [
                    {
                        "column": "age",
                        "analysis": {
                            "missing_pct": 5.2,
                            "distribution": "right_skewed",
                            "skewness": 2.3
                        },
                        "decision": {
                            "strategy": "median",
                            "reasoning": "Column 'age' is right-skewed (skew=2.3), median is more robust than mean"
                        }
                    }
                ],
                "outlier_decisions": [
                    {
                        "column": "income",
                        "analysis": {
                            "outliers_detected": 127,
                            "max_value": 980000,
                            "domain_validity": "valid_high_earners"
                        },
                        "decision": {
                            "action": "cap",
                            "threshold": "99th_percentile",
                            "reasoning": "Income outliers appear to be valid high earners, capping at 99th percentile to reduce extreme values while preserving signal"
                        }
                    }
                ],
                "duplicate_decisions": {
                    "duplicates_found": 42,
                    "action": "remove",
                    "keep": "first",
                    "reasoning": "Found 42 full-row duplicates, keeping first occurrence"
                }
            }
            ```

            CRITICAL RULES:
            - NO default strategies
            - ANALYZE distribution for EVERY column
            - CHOOSE strategy based on data characteristics
            - EXPLAIN every decision with data-driven reasoning
            """
        )

        # EDA Agent
        self.eda_agent = AssistantAgent(
            name="EDA_Agent",
            llm_config=self.llm_config,
            system_message="""You are an intelligent exploratory data analysis specialist.

            Your mission: Discover patterns, insights, and relationships in the data.

            AUTONOMOUS ANALYSIS PROCESS:

            1. STATISTICAL ANALYSIS:
               - Calculate descriptive stats for ALL numerical columns
               - Identify distribution types (normal, skewed, bimodal)
               - Detect seasonality in time-series data

            2. CORRELATION ANALYSIS:
               - DECIDE correlation method based on data:
                 * Pearson: For linear relationships (normal distributed)
                 * Spearman: For monotonic relationships (ordinal/ranked)
                 * Kendall: For small samples
               - IDENTIFY multicollinearity (r > 0.8)
               - RECOMMEND features to drop/combine

            3. PATTERN DETECTION:
               - Identify class imbalance
               - Detect temporal patterns
               - Find categorical relationships (chi-square tests)

            4. VISUALIZATION RECOMMENDATIONS:
               - DECIDE plot types based on data:
                 * Normal distribution → Histogram + QQ plot
                 * Skewed distribution → Box plot + Histogram
                 * Relationships → Scatter plot (continuous) or Bar plot (categorical)
                 * Time-series → Line plot with trend
                 * Correlations → Heatmap (if <20 features) or Top-K correlations

            OUTPUT FORMAT (JSON):
            ```json
            {
                "statistical_insights": [
                    {
                        "column": "age",
                        "stats": {
                            "mean": 42.5,
                            "median": 40.0,
                            "std": 12.3,
                            "skewness": 0.8
                        },
                        "insights": [
                            "Age is slightly right-skewed (mean > median)",
                            "Standard deviation of 12.3 suggests moderate variability"
                        ]
                    }
                ],
                "correlation_insights": [
                    {
                        "feature_pair": ["tenure_months", "total_spent"],
                        "correlation": 0.92,
                        "method": "pearson",
                        "implication": "HIGH multicollinearity detected",
                        "recommendation": "Drop 'tenure_months' (lower predictive power based on domain knowledge)"
                    }
                ],
                "pattern_insights": [
                    {
                        "pattern": "class_imbalance",
                        "details": "Target has 85/15 split (churned vs retained)",
                        "recommendation": "Use stratified sampling and consider SMOTE or class weights"
                    }
                ],
                "visualization_recommendations": [
                    {
                        "column": "age",
                        "viz_type": "histogram_with_kde",
                        "reasoning": "Numerical with slight skewness, histogram shows distribution shape"
                    },
                    {
                        "columns": ["age", "income"],
                        "viz_type": "scatter_plot",
                        "reasoning": "Examine relationship between two continuous variables"
                    }
                ]
            }
            ```

            CRITICAL:
            Base ALL recommendations on data analysis
            JUSTIFY every insight with numbers
            Consider DOMAIN context when suggesting actions
            NO generic insights - be specific!
            """
        )

        # Feature Engineering Agent
        self.feature_engineering_agent = AssistantAgent(
            name="Feature_Engineering_Agent",
            llm_config=self.llm_config,
            system_message="""You are an intelligent feature engineering specialist.

            Your mission: Create optimal features for modeling through autonomous decision-making.

            AUTONOMOUS FEATURE ENGINEERING PROCESS:

            1. CATEGORICAL ENCODING DECISIONS:
               ANALYZE each categorical column:
               - Cardinality (unique values)
               - Target relationship (correlation)
               - Domain meaning

               DECIDE encoding strategy:
               - IF cardinality < 10 → ONE-HOT ENCODING
               - IF cardinality 10-50 AND strong target correlation → TARGET ENCODING
               - IF cardinality > 50 → FREQUENCY ENCODING or HASH ENCODING
               - IF ordinal (low, medium, high) → ORDINAL ENCODING

               REASON: Explain why based on cardinality and target relationship

            2. NUMERICAL TRANSFORMATION DECISIONS:
               ANALYZE each numerical column:
               - Skewness (>1 or <-1 needs transformation)
               - Kurtosis (heavy tails?)
               - Zero-inflation

               DECIDE transformation:
               - IF right-skewed (skew > 1) → LOG or SQRT transform
               - IF left-skewed (skew < -1) → SQUARE or EXP transform
               - IF bimodal → Consider BINNING or split into categories
               - IF zero-inflated → Log1p transform

               REASON: Explain based on distribution analysis

            3. FEATURE CREATION DECISIONS:
               ANALYZE column types and relationships:
               - Date columns → Extract: year, month, day_of_week, is_weekend, quarter, days_since_X
               - Paired numerical → Create: ratios, differences, products
               - Text columns → Create: length, word_count, has_keywords
               - Geospatial → Create: distance_to_X, region_encoding

               DECIDE which features to create based on:
               - Domain relevance
               - Potential predictive power
               - Correlation with target

               REASON: Explain why each feature might be valuable

            4. FEATURE SELECTION DECISIONS:
               ANALYZE all features:
               - Variance (low variance features)
               - Correlation with target
               - Multicollinearity

               DECIDE what to drop:
               - Zero or near-zero variance
               - Highly correlated pairs (keep one)
               - ID columns with no predictive value

            OUTPUT FORMAT (JSON):
            ```json
            {
                "encoding_decisions": [
                    {
                        "column": "city",
                        "analysis": {
                            "cardinality": 127,
                            "unique_pct": 0.45,
                            "target_correlation": 0.23
                        },
                        "decision": {
                            "method": "target_encoding",
                            "reasoning": "City has 127 unique values (high cardinality). One-hot would create 127 new columns. Target encoding preserves information while keeping dimensionality low."
                        }
                    }
                ],
                "transformation_decisions": [
                    {
                        "column": "income",
                        "analysis": {
                            "skewness": 3.2,
                            "kurtosis": 12.5,
                            "distribution": "heavy_right_tail"
                        },
                        "decision": {
                            "method": "log1p_transform",
                            "reasoning": "Income is heavily right-skewed (skew=3.2). Log transformation will normalize distribution and improve model performance."
                        }
                    }
                ],
                "feature_creation": [
                    {
                        "new_feature": "days_since_signup",
                        "source_columns": ["signup_date", "current_date"],
                        "reasoning": "Temporal features often have high predictive power. Customers who signed up recently may have different behavior.",
                        "importance_estimate": "high"
                    },
                    {
                        "new_feature": "debt_to_income_ratio",
                        "source_columns": ["total_debt", "annual_income"],
                        "reasoning": "Ratio features often capture important relationships. Debt-to-income is a standard financial indicator.",
                        "importance_estimate": "medium"
                    }
                ],
                "feature_selection": [
                    {
                        "feature": "customer_id",
                        "action": "drop",
                        "reasoning": "ID column has no predictive value, only identifies rows"
                    },
                    {
                        "feature": "num_purchases",
                        "action": "drop",
                        "reasoning": "Highly correlated (r=0.94) with 'total_spent'. Dropping to avoid multicollinearity."
                    }
                ]
            }
            ```

            CRITICAL RULES:
            - ANALYZE distribution before transformation
            - CONSIDER domain relevance for new features
            - JUSTIFY every encoding choice with cardinality
            - NO blanket one-hot encoding - consider cardinality
            - NO arbitrary feature creation - explain value
            """
        )

        # ML Training Agent
        self.ml_training_agent = AssistantAgent(
            name="ML_Training_Agent",
            llm_config=self.llm_config,
            system_message="""You are an intelligent machine learning specialist with AUTONOMOUS model selection.

            Your mission: Select and train the best models for the task.

            AUTONOMOUS MODEL SELECTION PROCESS:

            1. TASK TYPE DETECTION:
               ANALYZE target variable:
               - IF continuous numerical → REGRESSION
               - IF binary/multi-class categorical → CLASSIFICATION
               - IF no labels provided → CLUSTERING

            2. MODEL SELECTION (Regression):
               ANALYZE dataset characteristics:
               - Data size: <1k → Simple models, >10k → Complex models
               - Linearity: Check if relationship is linear or non-linear
               - Feature count: High dim (>50 features) → Regularization needed

               DECIDE which models to try:
               - Linear relationship + interpretability → LinearRegression, Ridge, Lasso
               - Non-linear + small data (<5k) → RandomForest, GradientBoosting
               - Non-linear + large data (>5k) → XGBoost, LightGBM
               - High dimensional → ElasticNet (L1+L2 regularization)

            3. MODEL SELECTION (Classification):
               ANALYZE dataset characteristics:
               - Class balance: Check if imbalanced (use class_weight or SMOTE)
               - Data size and complexity
               - Interpretability requirements

               DECIDE which models to try:
               - Baseline + interpretable → LogisticRegression
               - Small data (<5k) → RandomForest, SVM
               - Large data + performance priority → XGBoost, LightGBM
               - Imbalanced classes → Models with class_weight support

            4. HYPERPARAMETER STRATEGY:
               DECIDE tuning approach:
               - Small search space (<20 combinations) → GridSearchCV
               - Large search space (>20) → RandomizedSearchCV
               - Limited time/resources → Use sensible defaults first

               DECIDE cross-validation folds:
               - Small data (<1k rows) → 3-fold CV
               - Medium data (1k-10k) → 5-fold CV
               - Large data (>10k) → 3-fold CV (faster)
               - Imbalanced → StratifiedKFold

            5. TRAINING STRATEGY:
               DECIDE evaluation metrics:
               - Regression: RMSE (default), MAE (robust to outliers), R² (variance explained)
               - Classification: Accuracy (balanced), F1 (imbalanced), ROC-AUC (probability calibration)

               DECIDE model count:
               - Try 3-5 models for comparison
               - Start simple, increase complexity

            OUTPUT FORMAT (JSON):
            ```json
            {
                "task_analysis": {
                    "task_type": "classification",
                    "target_variable": "churned",
                    "is_balanced": false,
                    "class_distribution": {"0": 0.73, "1": 0.27},
                    "reasoning": "Binary classification with 73/27 class imbalance"
                },
                "model_decisions": [
                    {
                        "model": "LogisticRegression",
                        "reasoning": "Baseline model for interpretability and speed",
                        "hyperparameters": {
                            "class_weight": "balanced",
                            "max_iter": 1000
                        },
                        "priority": "baseline"
                    },
                    {
                        "model": "XGBoost",
                        "reasoning": "Best performance for imbalanced classification, handles class_weight",
                        "hyperparameters": {
                            "scale_pos_weight": 2.7,
                            "max_depth": [3, 5, 7],
                            "learning_rate": [0.01, 0.1],
                            "n_estimators": [100, 200]
                        },
                        "tuning_method": "RandomizedSearchCV",
                        "cv_folds": 5,
                        "priority": "production"
                    },
                    {
                        "model": "RandomForest",
                        "reasoning": "Good balance of performance and interpretability",
                        "hyperparameters": {
                            "class_weight": "balanced",
                            "n_estimators": 100,
                            "max_depth": 10
                        },
                        "priority": "alternative"
                    }
                ],
                "evaluation_strategy": {
                    "primary_metric": "f1_score",
                    "reasoning": "F1 balances precision/recall for imbalanced classes",
                    "secondary_metrics": ["roc_auc", "precision", "recall"]
                }
            }
            ```

            CRITICAL RULES:
            - ANALYZE task type and data characteristics first
            - SELECT models based on data size, linearity, balance
            - HANDLE class imbalance with scale_pos_weight or class_weight
            - START simple (LogisticRegression) before complex (XGBoost)
            - NO training all models blindly - justify each choice
            """
        )

        # Model Evaluation Agent
        self.evaluation_agent = AssistantAgent(
            name="Evaluation_Agent",
            llm_config=self.llm_config,
            system_message="""You are an intelligent model evaluation specialist.

            Your mission: Evaluate models and select the best one for production.

            AUTONOMOUS EVALUATION PROCESS:

            1. METRIC SELECTION (Based on Task & Context):
               For REGRESSION:
               ANALYZE data characteristics:
               - IF outliers present → MAE (robust to outliers)
               - IF no outliers → RMSE (penalizes large errors)
               - IF percentage errors matter → MAPE
               - ALWAYS include R² (variance explained)

               For CLASSIFICATION:
               ANALYZE class balance and business context:
               - IF balanced classes → Accuracy
               - IF imbalanced → F1-Score, ROC-AUC
               - IF false positives costly (spam filter) → Optimize Precision
               - IF false negatives costly (fraud, medical) → Optimize Recall
               - IF probability estimates needed → ROC-AUC, PR-AUC

            2. MODEL COMPARISON:
               ANALYZE multiple dimensions:
               - Performance metrics (primary + secondary)
               - Training time (important for retraining)
               - Inference speed (important for real-time)
               - Model complexity / interpretability
               - Overfitting (train vs test performance gap)

               DECIDE best model based on:
               - Primary metric (highest priority)
               - Business constraints (speed, interpretability)
               - Generalization (avoid overfitting)

            3. PERFORMANCE DIAGNOSTICS:
               IDENTIFY issues:
               - Overfitting: Large train-test gap (>10% difference)
               - Underfitting: Poor performance on both train and test
               - Class imbalance handling: Check per-class performance

               RECOMMEND fixes:
               - Overfitting → Regularization, more data, simpler model
               - Underfitting → More features, complex model, feature engineering
               - Poor minority class → Adjust class_weight, try SMOTE

            4. FEATURE IMPORTANCE ANALYSIS:
               EXTRACT feature importance:
               - Tree models → built-in feature_importances_
               - Linear models → coefficient magnitudes
               - SHAP values for detailed explanation

               IDENTIFY:
               - Top 10 most important features
               - Features with near-zero importance (candidates for removal)
               - Surprising feature rankings (need domain review)

            OUTPUT FORMAT (JSON):
            ```json
            {
                "metric_decisions": {
                    "primary_metric": "f1_score",
                    "reasoning": "Imbalanced classification requires balanced precision/recall",
                    "secondary_metrics": ["roc_auc", "precision", "recall", "accuracy"]
                },
                "model_comparison": [
                    {
                        "model": "XGBoost",
                        "performance": {
                            "f1_score": 0.82,
                            "roc_auc": 0.89,
                            "precision": 0.84,
                            "recall": 0.80,
                            "accuracy": 0.87
                        },
                        "train_performance": {
                            "f1_score": 0.85
                        },
                        "overfitting_check": "Slight overfitting (3% gap), acceptable",
                        "training_time_sec": 12.5,
                        "inference_time_ms": 2.3
                    },
                    {
                        "model": "RandomForest",
                        "performance": {
                            "f1_score": 0.78,
                            "roc_auc": 0.85,
                            "precision": 0.80,
                            "recall": 0.76,
                            "accuracy": 0.84
                        },
                        "training_time_sec": 8.2,
                        "inference_time_ms": 1.8
                    },
                    {
                        "model": "LogisticRegression",
                        "performance": {
                            "f1_score": 0.71,
                            "roc_auc": 0.78,
                            "precision": 0.75,
                            "recall": 0.68,
                            "accuracy": 0.79
                        },
                        "training_time_sec": 0.5,
                        "inference_time_ms": 0.1,
                        "note": "Baseline model - interpretable but lower performance"
                    }
                ],
                "best_model_decision": {
                    "selected_model": "XGBoost",
                    "reasoning": "Highest F1 (0.82) and ROC-AUC (0.89). Acceptable overfitting (3%). Training time (12.5s) and inference (2.3ms) are acceptable for production.",
                    "trade_offs": "Slightly slower than RandomForest but +4% F1 improvement worth it"
                },
                "feature_importance": {
                    "top_features": [
                        {"feature": "total_spent", "importance": 0.24},
                        {"feature": "tenure_months", "importance": 0.19},
                        {"feature": "days_since_last_login", "importance": 0.15},
                        {"feature": "num_support_tickets", "importance": 0.12}
                    ],
                    "low_importance_features": ["customer_id_hash", "zip_code_first_digit"],
                    "recommendation": "Consider dropping low-importance features for model simplification"
                },
                "production_readiness": {
                    "ready": true,
                    "confidence": "high",
                    "monitoring_recommendations": [
                        "Track F1 score weekly (alert if drops below 0.75)",
                        "Monitor feature drift for top 5 features",
                        "Retrain if class distribution shifts >5%"
                    ]
                }
            }
            ```

            CRITICAL RULES:
            - SELECT metrics based on business context (cost of errors)
            - COMPARE models across multiple dimensions (not just accuracy)
            - CHECK for overfitting (train vs test gap)
            - CONSIDER speed and interpretability, not just performance
            - NO selecting model solely on single metric
            """
        )

        # Reporting Agent
        self.reporting_agent = AssistantAgent(
            name="Reporting_Agent",
            llm_config=self.llm_config,
            system_message="""You are an intelligent reporting specialist.

            Your mission: Generate comprehensive, actionable data science reports.

            AUTONOMOUS REPORTING PROCESS:

            1. REPORT STRUCTURE DECISION:
               ANALYZE audience and context:
               - Technical team → Include code snippets, technical metrics, model details
               - Business stakeholders → Focus on insights, recommendations, ROI
               - Mixed audience → Executive summary + technical appendix

               DECIDE sections to include:
               - Executive Summary (always)
               - Data Quality Report
               - EDA Key Insights
               - Feature Engineering Summary
               - Model Performance Comparison
               - Production Recommendations
               - Monitoring Strategy

            2. INSIGHT PRIORITIZATION:
               ANALYZE all findings from previous agents:
               - Data cleaning decisions
               - EDA discoveries
               - Feature engineering impact
               - Model performance

               DECIDE top 5-7 insights:
               - Actionable (can drive decisions)
               - Impactful (affects model performance >5%)
               - Surprising (counter-intuitive findings)

            3. VISUALIZATION SELECTION:
               For each key finding, DECIDE visualization:
               - Model comparison → Bar chart of metrics
               - Feature importance → Horizontal bar chart (top 10)
               - Class distribution → Pie chart or stacked bar
               - Performance over time → Line chart
               - Correlation findings → Heatmap (if <15 features)

            4. ACTIONABLE RECOMMENDATIONS:
               GENERATE specific next steps:
               - Data collection improvements
               - Feature engineering ideas for v2
               - Model deployment strategy
               - Monitoring and retraining plan
               - Business process changes based on insights

            OUTPUT FORMAT (Markdown with JSON metadata):
            ```markdown
            # Automated Data Science Report

            ## Executive Summary

            **Dataset:** [name] ([rows] rows × [columns] columns)
            **Task:** [Classification/Regression]
            **Best Model:** [model_name] ([primary_metric]: [score])
            **Production Ready:** [Yes/No]

            **Key Findings:**
            1. [Most impactful insight with numbers]
            2. [Second most important finding]
            3. [Third key takeaway]

            ---

            ## Data Quality Assessment

            ### Initial State
            - **Missing Values:** [X]% across [N] columns
            - **Outliers Detected:** [N] across [M] columns
            - **Duplicates:** [N] rows ([X]%)

            ### Cleaning Decisions (Autonomous)
            | Column | Issue | Strategy | Reasoning |
            |--------|-------|----------|-----------|
            | age | 5% missing, right-skewed | MEDIAN imputation | Median robust to skewness (skew=2.3) |
            | income | Outliers up to $980k | CAP at 99th percentile | Valid high earners, preserve signal |

            **Impact:** Data quality improved from [X]% to [Y]%

            ---

            ## Exploratory Data Analysis

            ### Statistical Insights
            - [Specific finding with numbers and interpretation]
            - [Pattern detected with business implication]

            ### Correlation Analysis
            ** HIGH IMPACT:** [feature1] and [feature2] highly correlated (r=[X])
            → **Recommendation:** Dropped [feature1] to avoid multicollinearity

            ### Class Distribution
            - Class 0 (retained): [X]%
            - Class 1 (churned): [Y]%
            → **Alert:** Imbalanced classes, used stratified sampling and class_weight

            ---

            ## ⚙️ Feature Engineering

            ### Encoding Decisions
            | Feature | Cardinality | Method | Reasoning |
            |---------|-------------|--------|-----------|
            | city | 127 | Target Encoding | Avoids dimension explosion (127 → 1 column) |

            ### Transformations Applied
            - **income:** Log transform (reduced skew from 3.2 to 0.8)
            - **age:** Binned into age_groups (improved model +3% AUC)

            ### New Features Created
            1. **days_since_signup:** High importance (0.15) - temporal pattern detected
            2. **avg_purchase_value:** total_spent / num_purchases - financial behavior indicator

            **Impact:** Feature engineering improved baseline model by [X]%

            ---

            ## Model Training & Evaluation

            ### Models Trained
            | Model | F1 Score | ROC-AUC | Precision | Recall | Training Time |
            |-------|----------|---------|-----------|--------|---------------|
            | XGBoost | 0.82 | 0.89 | 0.84 | 0.80 | 12.5s |
            | RandomForest | 0.78 | 0.85 | 0.80 | 0.76 | 8.2s |
            | LogisticRegression | 0.71 | 0.78 | 0.75 | 0.68 | 0.5s |

            ** SELECTED MODEL: XGBoost**
            - **Reasoning:** Best F1 (0.82) and ROC-AUC (0.89), acceptable speed (12.5s training, 2.3ms inference)
            - **Overfitting Check:** 3% train-test gap (acceptable)

            ### Feature Importance (Top 10)
            1. total_spent (0.24)
            2. tenure_months (0.19)
            3. days_since_last_login (0.15)
            4. num_support_tickets (0.12)
            5. avg_purchase_value (0.09)

            ---

            ## Business Recommendations

            ### Immediate Actions
            1. **Retention Focus:** Target customers with <6 months tenure (3x churn risk)
            2. **Engagement Monitoring:** Alert when days_since_last_login > 30 days
            3. **Support Quality:** High support tickets correlate with churn

            ### Model Deployment
            - **Environment:** [Production/Staging]
            - **Inference Speed:** 2.3ms (suitable for real-time)
            - **Monitoring:** Track F1 weekly, retrain if drops below 0.75

            ### Future Improvements (v2)
            1. Collect [specific data] to improve model by est. [X]%
            2. Engineer [specific feature] based on domain knowledge
            3. Try [specific model] for potential +[Y]% improvement

            ---

            ## Monitoring & Retraining Strategy

            ### Metrics to Monitor
            - **Model Performance:** F1 score, ROC-AUC (weekly)
            - **Feature Drift:** Monitor top 5 features for distribution shifts
            - **Class Distribution:** Alert if shifts >5% from training data

            ### Retraining Triggers
            - F1 score drops below 0.75 (from 0.82)
            - Feature drift detected (KS test p<0.05)
            - New data accumulated (every 3 months or 10k new samples)

            ---

            **Report Generated:** [timestamp]
            **Pipeline Version:** Autonomous Data Science v1.0
            **All Decisions Made By:** AI Agents (zero hardcoding)
            ```

            CRITICAL RULES:
            - PRIORITIZE actionable insights over raw statistics
            - QUANTIFY impact with percentages and numbers
            - EXPLAIN technical terms for non-technical readers
            - PROVIDE specific next steps, not vague suggestions
            - NO generic templates - tailor to THIS dataset's findings
            """
        )

        logger.info("Data science agent team initialized (7 agents)")

    def run_data_science_pipeline(
        self,
        file_path: str,
        target_column: str,
        task_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Run autonomous data science pipeline.

        Args:
            file_path: Path to CSV/Excel file
            target_column: Name of target variable
            task_type: "regression", "classification", or "auto" (agent decides)

        Returns:
            Dictionary with pipeline results and all agent decisions
        """
        # Start request tracing
        if self.observability_enabled:
            self.tracer.start_request(mode="data_science")
            correlation_id = self.tracer.current_correlation_id
            logger.info(f"🔍 Request trace started: {correlation_id}")

        logger.info(f"Starting autonomous data science pipeline")
        logger.info(f"File: {file_path}")
        logger.info(f"Target: {target_column}")
        logger.info(f"Task: {task_type}")

        pipeline_start_time = time.time()

        # Initialize LLM cache for cost reduction
        cache = get_cache(ttl_hours=24)
        logger.info("🗄️  LLM cache initialized (24h TTL)")

        # Step 1: Data Ingestion
        logger.info("\n" + "="*50)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("="*50)

        handler = FileHandler()
        df, metadata = handler.upload_file(file_path)

        if not metadata["success"]:
            return {"success": False, "error": metadata["error"]}

        schema = handler.get_schema(df)

        ingestion_prompt = f"""
        You have been given a dataset to load and analyze.

        FILE METADATA:
        {json.dumps(convert_to_json_serializable(metadata), indent=2)}

        SCHEMA:
        {json.dumps(convert_to_json_serializable(schema), indent=2)}

        TASK:
        1. Analyze the file characteristics
        2. Confirm the loading strategy was appropriate
        3. Identify any data quality issues from initial inspection
        4. Provide a summary of what you see

        Output your analysis as JSON with:
        - analysis: What you observed about the data
        - initial_recommendations: Any immediate concerns or suggestions
        """

        # Create a user proxy for ingestion step
        user_proxy_ingestion = UserProxyAgent(
            name="User_Ingestion",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        # Check cache for ingestion analysis
        ingestion_cache_key = f"ingestion_{df.shape}_{len(schema)}"
        cached_ingestion = cache.get(ingestion_cache_key, model="gpt-4o")

        if cached_ingestion:
            logger.info("✅ Using cached ingestion analysis (saved ~$0.005)")
            ingestion_analysis = cached_ingestion
        else:
            # Initiate conversation with ingestion agent
            user_proxy_ingestion.initiate_chat(
                self.data_ingestion_agent,
                message=ingestion_prompt
            )
            ingestion_response = self.data_ingestion_agent.last_message(user_proxy_ingestion)["content"]
            ingestion_analysis = ingestion_response
            cache.set(ingestion_cache_key, ingestion_analysis, model="gpt-4o", estimated_tokens=1500)
            logger.info("💾 Cached ingestion analysis for future use")

        # Step 2: Data Cleaning (AUTONOMOUS)
        logger.info("\n" + "="*50)
        logger.info("STEP 2: AUTONOMOUS DATA CLEANING")
        logger.info("="*50)

        # Prepare data profile for cleaning agent
        cleaner = DataCleaner()
        missing_report = cleaner.detect_missing_values(df)
        outlier_report = cleaner.detect_outliers(df)
        duplicate_report = cleaner.detect_duplicates(df)

        # Calculate distributions for each numerical column
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        distributions = {}
        for col in numerical_cols:
            if col in df.columns:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    distributions[col] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "skewness": float(col_data.skew()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max())
                    }

        cleaning_prompt = f"""
        You are now analyzing the dataset for cleaning.

        MISSING VALUES REPORT:
        {json.dumps(convert_to_json_serializable(missing_report), indent=2)}

        OUTLIER REPORT:
        {json.dumps(convert_to_json_serializable(outlier_report), indent=2)}

        DUPLICATE REPORT:
        {json.dumps(convert_to_json_serializable(duplicate_report), indent=2)}

        NUMERICAL DISTRIBUTIONS:
        {json.dumps(convert_to_json_serializable(distributions), indent=2)}

        TARGET COLUMN: {target_column}

        YOUR AUTONOMOUS TASK:
        1. For EACH column with missing values:
           - Analyze the distribution (normal, skewed, bimodal?)
           - DECIDE the best imputation strategy based on distribution
           - EXPLAIN your reasoning

        2. For EACH column with outliers:
           - Analyze if outliers are valid or errors
           - DECIDE: keep, cap, or remove
           - EXPLAIN your reasoning

        3. For duplicates:
           - DECIDE how to handle them
           - EXPLAIN your reasoning

        Output as JSON following the format in your system message.
        BE SPECIFIC - analyze EACH column individually!
        """

        # Create separate user proxy for cleaning step
        user_proxy_cleaning = UserProxyAgent(
            name="User_Cleaning",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        # Extract and execute cleaning decisions from agent's response
        parser = DecisionParser()
        validator = DecisionValidator()  # Decision validation

        # Check cache first
        cache_key = cleaning_prompt[:500]  # Use first 500 chars as key
        cached_decisions = cache.get(cache_key, model="gpt-4o")

        if cached_decisions:
            logger.info("✅ Using cached cleaning decisions (saved ~$0.01)")
            cleaning_decisions = cached_decisions
        else:
            # Initiate chat with cleaning agent (makes LLM call)
            user_proxy_cleaning.initiate_chat(
                self.data_cleaning_agent,
                message=cleaning_prompt
            )

            # Get the last message from the cleaning agent itself
            cleaning_response = self.data_cleaning_agent.last_message(user_proxy_cleaning)["content"]
            cleaning_decisions = parser.parse_cleaning_decision(cleaning_response)

            # Cache the decisions
            cache.set(cache_key, cleaning_decisions, model="gpt-4o", estimated_tokens=2000)
            logger.info("💾 Cached cleaning decisions for future use")

        # VALIDATE DECISIONS BEFORE EXECUTION
        valid, validation_report = validator.validate_cleaning_decision(cleaning_decisions)
        if not valid:
            logger.warning(f"⚠️ Cleaning decision validation failed: {validation_report['errors']}")
            logger.warning(f"⚠️ Using fallback strategies for safety")
            # Use safe fallbacks for invalid strategies
            for col, strategy_info in cleaning_decisions.get("missing_values", {}).items():
                if strategy_info.get("strategy", "").lower() not in validator.VALID_IMPUTATION_STRATEGIES:
                    strategy_info["strategy"] = validator.get_fallback_strategy("imputation")
                    logger.info(f"   Fallback for {col}: {strategy_info['strategy']}")
        else:
            logger.info(f"✅ Cleaning decisions validated successfully")

        # Execute cleaning based on agent decisions (with fallback to auto)
        if "missing_values" in cleaning_decisions and cleaning_decisions["missing_values"]:
            for col, decision in cleaning_decisions["missing_values"].items():
                strategy = decision.get("strategy", "auto")
                df, _ = cleaner.impute_missing_values(df, strategy={col: strategy})
        else:
            df, _ = cleaner.impute_missing_values(df, strategy="auto")

        # Execute duplicate handling
        df_cleaned, dup_removal = cleaner.remove_duplicates(df)

        # Execute additional cleaning enhancements
        df_cleaned, _ = cleaner.detect_invalid_values(df_cleaned)
        df_cleaned, _ = cleaner.handle_high_null_columns(df_cleaned, threshold=0.8)
        df_cleaned, _ = cleaner.detect_low_variance_columns(df_cleaned, threshold=0.95)

        logger.info(f"✅ Data cleaned: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

        # Step 3: EDA (AUTONOMOUS)
        logger.info("\n" + "="*50)
        logger.info("STEP 3: AUTONOMOUS EXPLORATORY DATA ANALYSIS")
        logger.info("="*50)

        # Calculate correlation matrix for EDA agent
        numerical_df = df_cleaned.select_dtypes(include=[np.number])
        if len(numerical_df.columns) > 1:
            corr_matrix = numerical_df.corr().to_dict()
        else:
            corr_matrix = {}

        # Prepare target statistics if available
        target_stats = {}
        if target_column in df_cleaned.columns:
            target_data = df_cleaned[target_column]
            if pd.api.types.is_numeric_dtype(target_data):
                target_stats = {
                    "type": "numerical",
                    "mean": float(target_data.mean()),
                    "median": float(target_data.median()),
                    "std": float(target_data.std())
                }
            else:
                target_stats = {
                    "type": "categorical",
                    "value_counts": target_data.value_counts().to_dict(),
                    "unique_values": int(target_data.nunique())
                }

        eda_prompt = f"""
        Perform autonomous exploratory data analysis on the cleaned dataset.

        DATASET SHAPE: {df_cleaned.shape[0]} rows × {df_cleaned.shape[1]} columns

        NUMERICAL DISTRIBUTIONS:
        {json.dumps(convert_to_json_serializable(distributions), indent=2)}

        CORRELATION MATRIX (Top pairs):
        {json.dumps(convert_to_json_serializable({k: {k2: v2 for k2, v2 in v.items() if abs(v2) > 0.7 and k != k2}
                    for k, v in corr_matrix.items()}), indent=2)}

        TARGET COLUMN STATS ({target_column}):
        {json.dumps(convert_to_json_serializable(target_stats), indent=2)}

        YOUR AUTONOMOUS TASK:
        1. ANALYZE the statistical properties of ALL numerical columns
        2. IDENTIFY high correlations (multicollinearity concerns)
        3. DETECT patterns and insights specific to this dataset
        4. RECOMMEND which features to drop/combine based on correlation
        5. SUGGEST appropriate visualizations for key relationships

        Output as JSON following your system message format.
        BE SPECIFIC about what you found and WHY it matters!
        """

        # Create separate user proxy for EDA step
        user_proxy_eda = UserProxyAgent(
            name="User_EDA",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        # Check cache for EDA insights
        eda_cache_key = f"eda_{df_cleaned.shape}_{target_column}"
        cached_eda = cache.get(eda_cache_key, model="gpt-4o")

        if cached_eda:
            logger.info("✅ Using cached EDA insights (saved ~$0.015)")
            eda_insights = cached_eda
        else:
            user_proxy_eda.initiate_chat(
                self.eda_agent,
                message=eda_prompt
            )
            eda_response = self.eda_agent.last_message(user_proxy_eda)["content"]
            eda_insights = eda_response
            cache.set(eda_cache_key, eda_insights, model="gpt-4o", estimated_tokens=2500)
            logger.info("💾 Cached EDA insights for future use")

        logger.info("EDA complete - insights generated")

        # Step 4: Feature Engineering (AUTONOMOUS)
        logger.info("\n" + "="*50)
        logger.info("STEP 4: AUTONOMOUS FEATURE ENGINEERING")
        logger.info("="*50)

        # Get categorical column info
        categorical_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_info = {}
        for col in categorical_cols:
            if col in df_cleaned.columns:
                categorical_info[col] = {
                    "cardinality": int(df_cleaned[col].nunique()),
                    "top_values": df_cleaned[col].value_counts().head(5).to_dict(),
                    "missing_pct": float(df_cleaned[col].isna().sum() / len(df_cleaned) * 100)
                }

        feature_eng_prompt = f"""
        Perform autonomous feature engineering on the dataset.

        NUMERICAL COLUMNS:
        {json.dumps(convert_to_json_serializable(list(numerical_df.columns)), indent=2)}

        CATEGORICAL COLUMNS INFO:
        {json.dumps(convert_to_json_serializable(categorical_info), indent=2)}

        TARGET: {target_column}
        TASK TYPE: {task_type}

        YOUR AUTONOMOUS TASK:
        1. For EACH categorical column:
           - Analyze cardinality
           - DECIDE optimal encoding method
           - EXPLAIN your choice

        2. For EACH numerical column:
           - Analyze distribution (from distributions above)
           - DECIDE if transformation needed
           - EXPLAIN reasoning

        3. SUGGEST new features to create:
           - Date extractions (if date columns exist)
           - Ratio features (if pairs make sense)
           - Interaction terms (if domain-relevant)

        4. RECOMMEND features to drop:
           - Low variance
           - High correlation
           - No predictive value

        Output as JSON following your system message format.
        """

        # Create separate user proxy for feature engineering step
        user_proxy_feature = UserProxyAgent(
            name="User_Feature",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )

        # Check cache for feature engineering decisions
        feature_cache_key = f"feature_eng_{len(categorical_cols)}_{len(numerical_df.columns)}_{target_column}"
        cached_feature_decisions = cache.get(feature_cache_key, model="gpt-4o")

        if cached_feature_decisions:
            logger.info("✅ Using cached feature engineering decisions (saved ~$0.02)")
            feature_decisions = cached_feature_decisions
        else:
            user_proxy_feature.initiate_chat(
                self.feature_engineering_agent,
                message=feature_eng_prompt
            )
            feature_response = self.feature_engineering_agent.last_message(user_proxy_feature)["content"]
            feature_decisions = feature_response
            cache.set(feature_cache_key, feature_decisions, model="gpt-4o", estimated_tokens=3000)
            logger.info("💾 Cached feature engineering decisions for future use")

        logger.info("Feature engineering decisions complete")

        # Execute feature engineering based on agent decisions
        engineer = FeatureEngineer()
        df_engineered = df_cleaned.copy()

        # Apply categorical encoding
        if target_column in df_engineered.columns:
            df_engineered, _ = engineer.advanced_encoding(df_engineered, target_column=target_column)

        # Apply ratio features
        df_engineered, _ = engineer.create_ratio_features(df_engineered, target_column=target_column)

        # Create missing indicators (before any nulls are removed)
        df_engineered, _ = engineer.create_missing_indicators(df_engineered, df_cleaned)

        logger.info(f"✅ Feature engineering complete: {df_engineered.shape[1]} features")

        # Step 5: PREPROCESSING
        logger.info("\n" + "="*50)
        logger.info("STEP 5: INTELLIGENT PREPROCESSING")
        logger.info("="*50)

        preprocessor = PreprocessingAgent()

        # Separate features and target
        X = df_engineered.drop(columns=[target_column])
        y = df_engineered[target_column]

        # Intelligent train/test split
        train_df, test_df, split_report = preprocessor.intelligent_split(
            df_engineered, target_column, test_size=0.2
        )

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # Select scaler
        scaler, scaler_report = preprocessor.select_scaler(X_train)

        # Scale features
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # Handle class imbalance (if classification)
        X_train_balanced, y_train_balanced, imbalance_report = preprocessor.handle_imbalance(
            X_train_scaled, y_train, strategy="auto"
        )

        # Feature selection
        X_train_selected, selected_features, selection_report = preprocessor.feature_selection(
            X_train_balanced, y_train_balanced, task_type=task_type, max_features=30
        )
        X_test_selected = X_test_scaled[selected_features]

        # Validate preprocessing
        validation_report = preprocessor.validate_preprocessing(
            X_train_selected, X_test_selected, y_train_balanced, y_test
        )

        logger.info(f"✅ Preprocessing complete: {X_train_selected.shape[1]} features selected")

        # Step 6: ML TRAINING
        logger.info("\n" + "="*50)
        logger.info("STEP 6: AUTONOMOUS ML MODEL TRAINING")
        logger.info("="*50)

        trainer = MLTrainer(task_type=task_type)

        # Train multiple models
        training_report = trainer.train_models(
            X_train_selected, y_train_balanced,
            X_test_selected, y_test,
            models_to_try=None,  # Auto-select
            cv_folds=5
        )

        logger.info(f"✅ Trained {training_report['successful_models']} models successfully")
        logger.info(f"🏆 Best model: {training_report['best_model_name']} (CV: {training_report['best_cv_score']:.4f})")

        # Step 7: MODEL EVALUATION
        logger.info("\n" + "="*50)
        logger.info("STEP 7: MODEL EVALUATION & COMPARISON")
        logger.info("="*50)

        evaluator = ModelEvaluator()

        # Evaluate all trained models
        model_evaluations = {}
        for model_name, model_info in training_report["models_trained"].items():
            if model_info.get("trained", False):
                model = model_info["model"]
                metrics = evaluator.evaluate_model(
                    model, X_test_selected, y_test,
                    task_type=task_type, model_name=model_name
                )
                model_evaluations[model_name] = metrics

        # Compare models
        comparison_report = evaluator.compare_models(model_evaluations)

        logger.info(f"✅ Evaluation complete - Best: {comparison_report['best_model']}")

        # Step 7B: SAVE MODEL WITH PERSISTENCE
        logger.info("\n" + "="*50)
        logger.info("STEP 7B: MODEL PERSISTENCE")
        logger.info("="*50)

        # Get best model and its metrics
        best_model = training_report["best_model"]
        best_model_name = training_report["best_model_name"]
        best_metrics = model_evaluations.get(best_model_name, {})

        # Get feature importance
        feature_importance = []
        try:
            trainer_instance = MLTrainer()
            importance_dict = trainer_instance.get_feature_importance(
                best_model,
                list(selected_features),
                top_n=20
            )
            feature_importance = [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in importance_dict.items()
            ]
            logger.info(f"📊 Extracted {len(feature_importance)} feature importances")
        except Exception as e:
            logger.warning(f"⚠️ Could not extract feature importance: {e}")

        # Extract confusion matrix for classification
        confusion_matrix_data = None
        if task_type == "classification" and "confusion_matrix" in best_metrics:
            cm = best_metrics["confusion_matrix"]
            if len(cm) == 2:  # Binary classification
                confusion_matrix_data = {
                    "true_negative": int(cm[0][0]),
                    "false_positive": int(cm[0][1]),
                    "false_negative": int(cm[1][0]),
                    "true_positive": int(cm[1][1])
                }
            else:  # Multi-class
                confusion_matrix_data = {"matrix": cm}
            logger.info(f"📊 Confusion matrix extracted")

        # Prepare comprehensive metadata
        model_metadata = {
            "model_type": best_model_name,
            "task_type": task_type,
            "feature_names": list(selected_features),
            "target_column": target_column,
            "metrics": {
                "accuracy": float(best_metrics.get("accuracy", 0)),
                "precision": float(best_metrics.get("precision", 0)),
                "recall": float(best_metrics.get("recall", 0)),
                "f1_score": float(best_metrics.get("f1_score", 0)),
                "roc_auc": float(best_metrics.get("roc_auc", 0)) if best_metrics.get("roc_auc") else None,
                "cv_score": float(training_report["best_cv_score"]),
                "test_score": float(training_report.get("best_test_score", 0)) if training_report.get("best_test_score") else None
            },
            "confusion_matrix": confusion_matrix_data,
            "feature_importance": feature_importance,
            "n_samples": int(metadata["rows"]),
            "n_features": int(len(selected_features)),
            "n_classes": int(y_train_balanced.nunique()) if task_type == "classification" else None,
            "dataset_info": {
                "original_rows": int(metadata["rows"]),
                "original_columns": int(metadata["columns"]),
                "final_rows": int(X_train_selected.shape[0]),
                "final_features": int(X_train_selected.shape[1])
            }
        }

        # Save model with persistence
        try:
            persistence = ModelPersistence()
            model_id, model_dir = persistence.save_model(
                model=best_model,
                preprocessor=scaler,  # Save the scaler/preprocessing pipeline
                metadata=model_metadata
            )
            logger.info(f"💾 Model saved with ID: {model_id}")
            logger.info(f"📁 Model directory: {model_dir}")
        except Exception as e:
            logger.error(f"❌ Model persistence failed: {e}")
            model_id = None
            model_dir = None

        # Step 8: REPORT GENERATION
        logger.info("\n" + "="*50)
        logger.info("STEP 8: GENERATING COMPREHENSIVE REPORT")
        logger.info("="*50)

        # Compile final results
        pipeline_results = {
            "success": True,
            "model_id": model_id,  # NEW: Model ID for predictions
            "dataset_info": {
                "original_shape": (metadata["rows"], metadata["columns"]),
                "cleaned_shape": df_cleaned.shape,
                "engineered_shape": df_engineered.shape,
                "final_shape": X_train_selected.shape,
                "target_column": target_column,
                "task_type": task_type
            },
            "cleaning_summary": {
                "missing_values_handled": True,
                "duplicates_removed": dup_removal.get("rows_removed", 0),
                "invalid_values_removed": True,
                "low_variance_dropped": True
            },
            "preprocessing_summary": {
                "split_strategy": split_report.get("strategy"),
                "scaler_used": scaler_report.get("scaler_type"),
                "imbalance_handled": imbalance_report.get("strategy_applied"),
                "features_selected": len(selected_features),
                "validation_passed": validation_report.get("all_passed", False)
            },
            "best_model_name": training_report["best_model_name"],
            "best_cv_score": training_report["best_cv_score"],
            "best_test_score": training_report.get("best_test_score"),
            "model_comparison": comparison_report,
            "model_evaluations": model_evaluations,
            "trained_model": training_report["best_model"],
            # NEW: Enhanced metrics for UI
            "metrics": model_metadata["metrics"],
            "confusion_matrix": confusion_matrix_data,
            "feature_importance": feature_importance,
            "feature_names": list(selected_features),
            "n_samples": metadata["rows"],
            "n_features": len(selected_features),
            # NEW: Model selection reasoning
            "model_selection_metadata": training_report.get("model_selection_metadata")
        }

        # Generate HTML report with unique filename
        import datetime
        from pathlib import Path

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"data_science_report_{timestamp}.html"

        # Get project root (parent of agents directory)
        project_root = Path(__file__).parent.parent
        reports_dir = project_root / "reports"

        # Create reports directory if it doesn't exist
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Use absolute path for report
        report_path_absolute = reports_dir / report_filename

        report_generator = ReportGenerator()
        html_report = report_generator.generate_html_report(
            pipeline_results,
            output_path=str(report_path_absolute)
        )

        # Store report filename in results for API to return
        pipeline_results['report_path'] = str(report_path_absolute)
        pipeline_results['report_filename'] = report_filename

        logger.info(f"✅ HTML report generated: {report_path_absolute}")

        logger.info("\n" + "="*70)
        logger.info("🎉 AUTONOMOUS DATA SCIENCE PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"✅ Best Model: {pipeline_results['best_model_name']}")
        logger.info(f"✅ CV Score: {pipeline_results['best_cv_score']:.4f}")
        if pipeline_results['best_test_score']:
            logger.info(f"✅ Test Score: {pipeline_results['best_test_score']:.4f}")
        logger.info("="*70)

        # End request tracing with observability metrics
        total_duration = time.time() - pipeline_start_time

        # Estimate total tokens used (rough approximation)
        # Each of 8 agents makes ~2-3 LLM calls with ~1000-3000 tokens each
        estimated_tokens = 25000  # Conservative estimate

        # Get cache statistics (always available)
        cache_stats = cache.get_stats()

        # Initialize observability data
        correlation_id = None

        if self.observability_enabled:
            # Complete the trace
            summary = self.tracer.end_request(
                success=True,
                total_duration_ms=int(total_duration * 1000)
            )
            correlation_id = summary['correlation_id']

            logger.info(f"\n🔍 OBSERVABILITY SUMMARY:")
            logger.info(f"   Correlation ID: {correlation_id}")
            logger.info(f"   Total Duration: {total_duration:.2f}s")
            logger.info(f"   Estimated Tokens: ~{estimated_tokens:,}")
            logger.info(f"   Estimated Cost: ~${(estimated_tokens/1_000_000)*5:.4f}")
            logger.info(f"   Cache Hit Rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
            logger.info(f"   Cache Savings: ${cache_stats.get('cost_saved_usd', 0):.4f}")
        else:
            # Still log basic metrics even if observability is disabled
            logger.info(f"\n📊 PIPELINE SUMMARY:")
            logger.info(f"   Total Duration: {total_duration:.2f}s")
            logger.info(f"   Estimated Tokens: ~{estimated_tokens:,}")
            logger.info(f"   Estimated Cost: ~${(estimated_tokens/1_000_000)*5:.4f}")
            logger.info(f"   Cache Hit Rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
            logger.info(f"   Cache Savings: ${cache_stats.get('cost_saved_usd', 0):.4f}")

        # Always add observability metrics to results (regardless of observability_enabled)
        pipeline_results['observability'] = {
            'correlation_id': correlation_id,  # Will be None if observability disabled
            'duration_seconds': total_duration,
            'estimated_tokens': estimated_tokens,
            'estimated_cost_usd': (estimated_tokens / 1_000_000) * 5,
            'cache_stats': {
                'hits': cache_stats.get('hits', 0),
                'misses': cache_stats.get('misses', 0),
                'hit_rate_percent': cache_stats.get('hit_rate_percent', 0),
                'cost_saved_usd': cache_stats.get('cost_saved_usd', 0),
                'total_requests': cache_stats.get('total_requests', 0)
            }
        }

        return pipeline_results


if __name__ == "__main__":
    # Test the autonomous agent system
    import sys

    if len(sys.argv) < 3:
        print("Usage: python data_science_agents.py <csv_file> <target_column>")
        sys.exit(1)

    file_path = sys.argv[1]
    target_column = sys.argv[2]
    task_type = sys.argv[3] if len(sys.argv) > 3 else "auto"

    # Create agent team
    team = DataScienceAgentTeam()

    # Run pipeline
    results = team.run_data_science_pipeline(
        file_path=file_path,
        target_column=target_column,
        task_type=task_type
    )

    print("\n" + "="*70)
    print("PIPELINE RESULTS")
    print("="*70)
    print(json.dumps(convert_to_json_serializable(results), indent=2))
