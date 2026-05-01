"""
EDA (Exploratory Data Analysis) Module
Advanced analysis capabilities for data science pipeline.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

logger = logging.getLogger(__name__)


class EDAAnalyzer:
    """
    Comprehensive EDA utility with advanced analysis capabilities.
    """

    def __init__(self):
        """Initialize EDAAnalyzer."""
        self.analysis_report = {
            "actions": [],
            "insights": []
        }

    def detect_class_imbalance(
        self,
        df: pd.DataFrame,
        target_column: str,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detect class imbalance in target variable for classification tasks.

        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            threshold: Minimum percentage for minority class (default 0.2 = 20%)

        Returns:
            Dictionary with imbalance analysis and recommendations
        """
        try:
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found"}

            # Calculate class distribution
            value_counts = df[target_column].value_counts()
            total_count = len(df[target_column].dropna())

            class_distribution = {}
            for cls, count in value_counts.items():
                class_distribution[str(cls)] = {
                    "count": int(count),
                    "percentage": float(count / total_count * 100)
                }

            # Find minority and majority classes
            minority_class = value_counts.idxmin()
            majority_class = value_counts.idxmax()
            minority_pct = value_counts.min() / total_count
            imbalance_ratio = value_counts.max() / value_counts.min()

            # Determine severity and recommendation
            is_imbalanced = minority_pct < threshold
            severity = "severe" if minority_pct < 0.05 else "moderate" if minority_pct < 0.2 else "mild"

            recommendations = []
            if is_imbalanced:
                if minority_pct < 0.05:
                    recommendations.append("SMOTE (Synthetic Minority Over-sampling)")
                    recommendations.append("Class weights in model training")
                elif minority_pct < 0.2:
                    recommendations.append("Class weights in model training")
                    recommendations.append("Stratified sampling")
                else:
                    recommendations.append("Stratified k-fold cross-validation")

            imbalance_report = {
                "is_imbalanced": bool(is_imbalanced),
                "severity": severity,
                "class_distribution": class_distribution,
                "minority_class": str(minority_class),
                "majority_class": str(majority_class),
                "minority_percentage": float(minority_pct * 100),
                "imbalance_ratio": float(imbalance_ratio),
                "recommendations": recommendations,
                "reasoning": f"Severe imbalance detected: {minority_pct*100:.1f}% minority class (threshold: {threshold*100}%). Recommend {recommendations[0] if recommendations else 'no action'}."
            }

            if is_imbalanced:
                logger.info(f"⚠️ Class imbalance detected: {minority_pct*100:.1f}% minority class")
                self.analysis_report["insights"].append({
                    "type": "class_imbalance",
                    "severity": severity,
                    "recommendation": recommendations[0] if recommendations else None
                })

            return imbalance_report

        except Exception as e:
            logger.error(f"❌ Class imbalance detection failed: {e}")
            return {"error": str(e)}

    def detect_data_leakage(
        self,
        df: pd.DataFrame,
        target_column: str,
        correlation_threshold: float = 0.95,
        future_keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect potential data leakage (features correlated too highly with target).

        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            correlation_threshold: Threshold for suspicious correlation (default 0.95)
            future_keywords: Keywords indicating future leakage (e.g., 'approved', 'result')

        Returns:
            Dictionary with leakage analysis and columns to drop
        """
        try:
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found"}

            if future_keywords is None:
                future_keywords = ['approved', 'result', 'outcome', 'status', 'decision',
                                 'final', 'completed', 'confirmed', 'verified']

            leakage_report = {
                "suspicious_columns": {},
                "columns_to_drop": [],
                "total_leaks_found": 0
            }

            # Check correlation-based leakage (for numerical target)
            if pd.api.types.is_numeric_dtype(df[target_column]):
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                numerical_cols = [col for col in numerical_cols if col != target_column]

                for col in numerical_cols:
                    correlation = df[col].corr(df[target_column])
                    if abs(correlation) >= correlation_threshold:
                        leakage_report["suspicious_columns"][col] = {
                            "type": "high_correlation",
                            "correlation": float(correlation),
                            "reasoning": f"Correlation {correlation:.3f} with target (threshold: {correlation_threshold}) - likely future leakage"
                        }
                        leakage_report["columns_to_drop"].append(col)

            # Check keyword-based leakage
            for col in df.columns:
                if col == target_column:
                    continue

                col_lower = col.lower()
                for keyword in future_keywords:
                    if keyword in col_lower:
                        if col not in leakage_report["columns_to_drop"]:
                            leakage_report["suspicious_columns"][col] = {
                                "type": "future_keyword",
                                "keyword": keyword,
                                "reasoning": f"Column name contains '{keyword}' - likely unavailable at prediction time"
                            }
                            leakage_report["columns_to_drop"].append(col)
                        break

            leakage_report["total_leaks_found"] = len(leakage_report["columns_to_drop"])

            if leakage_report["total_leaks_found"] > 0:
                logger.warning(f"⚠️ Found {leakage_report['total_leaks_found']} potential data leakage columns")
                self.analysis_report["insights"].append({
                    "type": "data_leakage",
                    "columns": leakage_report["columns_to_drop"]
                })

            return leakage_report

        except Exception as e:
            logger.error(f"❌ Data leakage detection failed: {e}")
            return {"error": str(e)}

    def analyze_high_cardinality(
        self,
        df: pd.DataFrame,
        threshold: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze categorical columns with high cardinality and suggest encoding.

        Args:
            df: Pandas DataFrame
            threshold: Cardinality threshold (default 50 unique values)

        Returns:
            Dictionary with cardinality analysis and encoding recommendations
        """
        try:
            cardinality_report = {
                "high_cardinality_columns": {},
                "total_high_card_columns": 0
            }

            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            for col in categorical_cols:
                unique_count = df[col].nunique()

                if unique_count >= threshold:
                    # Determine encoding recommendation
                    if unique_count > 100:
                        encoding_recommendation = "target_encoding"
                        reasoning = f"Very high cardinality ({unique_count} unique values). One-hot would create {unique_count} columns. Use target encoding."
                    elif unique_count > 50:
                        encoding_recommendation = "target_encoding"
                        reasoning = f"High cardinality ({unique_count} unique values). Recommend target encoding over one-hot."
                    else:
                        encoding_recommendation = "one_hot"
                        reasoning = f"Moderate cardinality ({unique_count} unique values). One-hot encoding acceptable."

                    cardinality_report["high_cardinality_columns"][col] = {
                        "unique_count": int(unique_count),
                        "percentage_unique": float(unique_count / len(df) * 100),
                        "encoding_recommendation": encoding_recommendation,
                        "reasoning": reasoning
                    }

            cardinality_report["total_high_card_columns"] = len(cardinality_report["high_cardinality_columns"])

            if cardinality_report["total_high_card_columns"] > 0:
                logger.info(f"📊 Found {cardinality_report['total_high_card_columns']} high-cardinality columns")

            return cardinality_report

        except Exception as e:
            logger.error(f"❌ Cardinality analysis failed: {e}")
            return {"error": str(e)}

    def detect_temporal_patterns(
        self,
        df: pd.DataFrame,
        date_column: Optional[str] = None,
        value_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal patterns: trends, seasonality, data drift.

        Args:
            df: Pandas DataFrame
            date_column: Name of datetime column (auto-detects if None)
            value_column: Column to analyze over time (uses numerical columns if None)

        Returns:
            Dictionary with temporal pattern analysis
        """
        try:
            temporal_report = {
                "patterns_found": {},
                "recommendations": []
            }

            # Auto-detect date column
            if date_column is None:
                datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                if not datetime_cols:
                    # Try to convert object columns
                    for col in df.select_dtypes(include=['object']).columns:
                        try:
                            pd.to_datetime(df[col].dropna().head(10))
                            datetime_cols.append(col)
                            break
                        except:
                            pass

                if datetime_cols:
                    date_column = datetime_cols[0]
                else:
                    return {"error": "No datetime column found"}

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

            # Auto-select value column
            if value_column is None:
                numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numerical_cols:
                    value_column = numerical_cols[0]
                else:
                    return {"error": "No numerical column found for temporal analysis"}

            # Sort by date
            df_sorted = df.sort_values(date_column).copy()

            # 1. Trend Detection (linear regression on time)
            df_sorted['time_index'] = range(len(df_sorted))
            valid_data = df_sorted[[value_column, 'time_index']].dropna()

            if len(valid_data) > 2:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    valid_data['time_index'], valid_data[value_column]
                )

                trend_strength = "strong" if abs(r_value) > 0.7 else "moderate" if abs(r_value) > 0.4 else "weak"
                trend_direction = "increasing" if slope > 0 else "decreasing"

                temporal_report["patterns_found"]["trend"] = {
                    "detected": bool(abs(r_value) > 0.4),
                    "direction": trend_direction,
                    "strength": trend_strength,
                    "r_squared": float(r_value ** 2),
                    "p_value": float(p_value),
                    "reasoning": f"{trend_strength.capitalize()} {trend_direction} trend detected (R²={r_value**2:.3f})"
                }

            # 2. Seasonality Detection (month-based patterns)
            df_sorted['month'] = pd.to_datetime(df_sorted[date_column]).dt.month
            monthly_means = df_sorted.groupby('month')[value_column].mean()

            if len(monthly_means) >= 3:
                monthly_variation = monthly_means.std() / monthly_means.mean() if monthly_means.mean() != 0 else 0
                has_seasonality = monthly_variation > 0.2  # >20% variation

                if has_seasonality:
                    temporal_report["patterns_found"]["seasonality"] = {
                        "detected": True,
                        "variation_coefficient": float(monthly_variation),
                        "reasoning": f"Strong monthly seasonality detected (CV={monthly_variation:.2f}). Recommend creating 'month' feature."
                    }
                    temporal_report["recommendations"].append("Create 'month' feature for seasonality")

            # 3. Data Drift Detection (distribution change over time)
            mid_point = len(df_sorted) // 2
            first_half = df_sorted[value_column].iloc[:mid_point].dropna()
            second_half = df_sorted[value_column].iloc[mid_point:].dropna()

            if len(first_half) > 30 and len(second_half) > 30:
                # KS test for distribution difference
                ks_statistic, ks_pvalue = stats.ks_2samp(first_half, second_half)
                has_drift = ks_pvalue < 0.05

                temporal_report["patterns_found"]["data_drift"] = {
                    "detected": bool(has_drift),
                    "ks_statistic": float(ks_statistic),
                    "p_value": float(ks_pvalue),
                    "mean_change": float((second_half.mean() - first_half.mean()) / first_half.mean() * 100),
                    "reasoning": f"Distribution shift detected (p={ks_pvalue:.4f}). Data may be non-stationary." if has_drift else "No significant drift"
                }

            logger.info(f"✅ Temporal analysis complete: {len(temporal_report['patterns_found'])} patterns analyzed")

            return temporal_report

        except Exception as e:
            logger.error(f"❌ Temporal pattern detection failed: {e}")
            return {"error": str(e)}

    def discover_feature_interactions(
        self,
        df: pd.DataFrame,
        target_column: str,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Discover promising feature interactions based on correlation with target.

        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            top_n: Number of top interactions to return

        Returns:
            Dictionary with interaction analysis
        """
        try:
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found"}

            interaction_report = {
                "interactions_tested": 0,
                "promising_interactions": [],
                "top_interactions": []
            }

            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numerical_cols = [col for col in numerical_cols if col != target_column]

            if len(numerical_cols) < 2:
                return {"error": "Need at least 2 numerical columns for interaction discovery"}

            # Test interactions between top correlated features
            correlations = {}
            for col in numerical_cols:
                correlations[col] = abs(df[col].corr(df[target_column]))

            top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:min(10, len(numerical_cols))]
            top_feature_names = [f[0] for f in top_features]

            interactions = []

            # Test multiplication and division interactions
            for i, col1 in enumerate(top_feature_names):
                for col2 in top_feature_names[i+1:]:
                    # Multiplication
                    interaction_mult = df[col1] * df[col2]
                    corr_mult = abs(interaction_mult.corr(df[target_column]))

                    interactions.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "operation": "multiply",
                        "correlation": float(corr_mult),
                        "feature_name": f"{col1}_x_{col2}"
                    })

                    # Division (if no zeros)
                    if (df[col2] != 0).all():
                        interaction_div = df[col1] / df[col2]
                        corr_div = abs(interaction_div.corr(df[target_column]))

                        interactions.append({
                            "feature_1": col1,
                            "feature_2": col2,
                            "operation": "divide",
                            "correlation": float(corr_div),
                            "feature_name": f"{col1}_div_{col2}"
                        })

            interaction_report["interactions_tested"] = len(interactions)

            # Sort by correlation and get top N
            interactions_sorted = sorted(interactions, key=lambda x: x['correlation'], reverse=True)

            # Filter promising interactions (higher correlation than individual features)
            for interaction in interactions_sorted:
                orig_corr_1 = correlations.get(interaction['feature_1'], 0)
                orig_corr_2 = correlations.get(interaction['feature_2'], 0)
                max_orig_corr = max(orig_corr_1, orig_corr_2)

                if interaction['correlation'] > max_orig_corr * 1.1:  # At least 10% improvement
                    interaction['improvement'] = float((interaction['correlation'] - max_orig_corr) / max_orig_corr * 100)
                    interaction['reasoning'] = f"Interaction '{interaction['feature_name']}' has {interaction['correlation']:.3f} correlation with target ({interaction['improvement']:.1f}% better than best individual feature)"
                    interaction_report["promising_interactions"].append(interaction)

            interaction_report["top_interactions"] = interactions_sorted[:top_n]

            if interaction_report["promising_interactions"]:
                logger.info(f"✅ Found {len(interaction_report['promising_interactions'])} promising feature interactions")

            return interaction_report

        except Exception as e:
            logger.error(f"❌ Feature interaction discovery failed: {e}")
            return {"error": str(e)}

    def analyze_distribution_per_feature(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze distribution of each feature and recommend transformations.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary with distribution analysis per feature
        """
        try:
            distribution_report = {
                "numerical_features": {},
                "categorical_features": {},
                "datetime_features": {}
            }

            # Numerical features
            for col in df.select_dtypes(include=[np.number]).columns:
                col_data = df[col].dropna()

                if len(col_data) == 0:
                    continue

                # Calculate distribution metrics
                skewness = float(col_data.skew())
                kurtosis = float(col_data.kurtosis())

                # Shapiro-Wilk normality test (if sample size reasonable)
                if 3 <= len(col_data) <= 5000:
                    _, p_value = stats.shapiro(col_data[:5000])
                    is_normal = p_value > 0.05
                else:
                    is_normal = abs(skewness) < 0.5

                # Count zeros
                zero_pct = (col_data == 0).sum() / len(col_data)
                is_zero_inflated = zero_pct > 0.3

                # Determine distribution type
                if is_zero_inflated:
                    dist_type = "zero_inflated"
                elif abs(skewness) > 1:
                    dist_type = "highly_skewed"
                elif abs(skewness) > 0.5:
                    dist_type = "moderately_skewed"
                elif abs(kurtosis) > 3:
                    dist_type = "heavy_tailed"
                elif is_normal:
                    dist_type = "normal"
                else:
                    dist_type = "other"

                # Recommend transformation
                if dist_type == "highly_skewed" and skewness > 0:
                    transformation = "log"
                    reasoning = f"Right-skewed (skewness={skewness:.2f}). Recommend log transformation."
                elif dist_type == "highly_skewed" and skewness < 0:
                    transformation = "sqrt"
                    reasoning = f"Left-skewed (skewness={skewness:.2f}). Recommend sqrt transformation."
                elif dist_type == "zero_inflated":
                    transformation = "log1p"
                    reasoning = f"{zero_pct*100:.1f}% zeros. Recommend log1p transformation."
                else:
                    transformation = "none"
                    reasoning = f"Distribution is {dist_type}. No transformation needed."

                distribution_report["numerical_features"][col] = {
                    "distribution_type": dist_type,
                    "skewness": float(skewness),
                    "kurtosis": float(kurtosis),
                    "is_normal": bool(is_normal),
                    "zero_percentage": float(zero_pct * 100),
                    "recommended_transformation": transformation,
                    "reasoning": reasoning
                }

            # Categorical features
            for col in df.select_dtypes(include=['object', 'category']).columns:
                value_counts = df[col].value_counts()

                if len(value_counts) == 0:
                    continue

                # Check if balanced
                max_pct = value_counts.iloc[0] / len(df[col].dropna())
                is_balanced = max_pct < 0.7

                distribution_report["categorical_features"][col] = {
                    "unique_count": int(len(value_counts)),
                    "most_common": str(value_counts.index[0]),
                    "most_common_percentage": float(max_pct * 100),
                    "is_balanced": bool(is_balanced),
                    "reasoning": f"{'Balanced' if is_balanced else 'Imbalanced'} distribution. Most common value: {value_counts.index[0]} ({max_pct*100:.1f}%)"
                }

            # Datetime features
            for col in df.select_dtypes(include=['datetime64']).columns:
                col_data = df[col].dropna()

                if len(col_data) == 0:
                    continue

                # Check for continuity (gaps)
                sorted_dates = col_data.sort_values()
                if len(sorted_dates) > 1:
                    date_diffs = sorted_dates.diff().dropna()
                    median_diff = date_diffs.median()
                    max_diff = date_diffs.max()

                    has_gaps = max_diff > median_diff * 5

                    distribution_report["datetime_features"][col] = {
                        "min_date": str(sorted_dates.min()),
                        "max_date": str(sorted_dates.max()),
                        "has_gaps": bool(has_gaps),
                        "median_interval": str(median_diff),
                        "reasoning": f"{'Gaps detected' if has_gaps else 'Continuous'} date sequence from {sorted_dates.min()} to {sorted_dates.max()}"
                    }

            logger.info(f"✅ Distribution analysis complete: {len(distribution_report['numerical_features'])} numerical, {len(distribution_report['categorical_features'])} categorical features")

            return distribution_report

        except Exception as e:
            logger.error(f"❌ Distribution analysis failed: {e}")
            return {"error": str(e)}

    def analyze_target_variable(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep analysis of target variable with task-specific insights.

        Args:
            df: Pandas DataFrame
            target_column: Name of target column
            task_type: "regression", "classification", or "time-series" (auto-detects if None)

        Returns:
            Dictionary with target variable analysis
        """
        try:
            if target_column not in df.columns:
                return {"error": f"Target column '{target_column}' not found"}

            target_report = {
                "task_type": task_type,
                "analysis": {},
                "issues_found": [],
                "recommendations": []
            }

            target_data = df[target_column].dropna()

            # Auto-detect task type
            if task_type is None:
                if pd.api.types.is_numeric_dtype(target_data):
                    unique_count = target_data.nunique()
                    if unique_count <= 20:
                        task_type = "classification"
                    else:
                        task_type = "regression"
                else:
                    task_type = "classification"
                target_report["task_type"] = task_type

            # Regression analysis
            if task_type == "regression":
                # Check range and outliers
                q1 = target_data.quantile(0.25)
                q3 = target_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
                outlier_pct = len(outliers) / len(target_data) * 100

                target_report["analysis"]["regression"] = {
                    "min": float(target_data.min()),
                    "max": float(target_data.max()),
                    "mean": float(target_data.mean()),
                    "median": float(target_data.median()),
                    "std": float(target_data.std()),
                    "skewness": float(target_data.skew()),
                    "outlier_count": int(len(outliers)),
                    "outlier_percentage": float(outlier_pct),
                    "outlier_threshold_upper": float(upper_bound)
                }

                if outlier_pct > 5:
                    target_report["issues_found"].append(f"{len(outliers)} outliers (>{upper_bound:.2f})")
                    target_report["recommendations"].append(f"Consider capping target at {upper_bound:.2f} or separate treatment for outliers")

            # Classification analysis
            elif task_type == "classification":
                value_counts = target_data.value_counts()
                class_distribution = {str(k): int(v) for k, v in value_counts.items()}

                minority_pct = value_counts.min() / len(target_data) * 100

                target_report["analysis"]["classification"] = {
                    "class_count": int(len(value_counts)),
                    "class_distribution": class_distribution,
                    "minority_class_percentage": float(minority_pct),
                    "is_balanced": bool(minority_pct > 20)
                }

                if minority_pct < 5:
                    target_report["issues_found"].append(f"Severe class imbalance ({minority_pct:.1f}% minority class)")
                    target_report["recommendations"].append("Use SMOTE or class weights")
                elif minority_pct < 20:
                    target_report["issues_found"].append(f"Moderate class imbalance ({minority_pct:.1f}% minority class)")
                    target_report["recommendations"].append("Use stratified sampling and class weights")

            logger.info(f"✅ Target variable analysis complete: {task_type} task")

            return target_report

        except Exception as e:
            logger.error(f"❌ Target variable analysis failed: {e}")
            return {"error": str(e)}

    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of all EDA actions performed.

        Returns:
            Dictionary with analysis summary
        """
        return self.analysis_report
