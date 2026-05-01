"""
Feature Engineering Module
Advanced feature creation and transformation capabilities.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import hashlib

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Comprehensive feature engineering with 50+ techniques.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        self.engineering_report = {
            "actions": [],
            "features_created": 0,
            "features_dropped": 0,
            "original_count": 0,
            "final_count": 0
        }
        self.max_features = 100  # Prevent feature explosion

    def advanced_encoding(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply intelligent encoding based on cardinality.

        Args:
            df: Pandas DataFrame
            target_column: Optional target for target encoding
            categorical_columns: Columns to encode (auto-detects if None)

        Returns:
            Tuple of (encoded DataFrame, encoding report)
        """
        try:
            df_encoded = df.copy()

            if categorical_columns is None:
                categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
                if target_column and target_column in categorical_columns:
                    categorical_columns.remove(target_column)

            encoding_report = {
                "columns_encoded": {},
                "total_columns": len(categorical_columns)
            }

            for col in categorical_columns:
                if col not in df_encoded.columns:
                    continue

                cardinality = df_encoded[col].nunique()

                # Decision logic based on cardinality
                if cardinality < 10:
                    # One-hot encoding
                    encoded = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, encoded], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])
                    method = "one_hot"
                    reasoning = f"Low cardinality ({cardinality}). One-hot creates {len(encoded.columns)} features."

                elif cardinality <= 50:
                    # Target encoding
                    if target_column and target_column in df.columns:
                        target_means = df.groupby(col)[target_column].mean()
                        df_encoded[f"{col}_target_enc"] = df_encoded[col].map(target_means).fillna(target_means.mean())
                        df_encoded = df_encoded.drop(columns=[col])
                        method = "target_encoding"
                        reasoning = f"Medium cardinality ({cardinality}). Target encoding prevents {cardinality} one-hot columns."
                    else:
                        # Frequency encoding fallback
                        freq = df_encoded[col].value_counts(normalize=True)
                        df_encoded[f"{col}_freq"] = df_encoded[col].map(freq)
                        df_encoded = df_encoded.drop(columns=[col])
                        method = "frequency_encoding"
                        reasoning = f"Medium cardinality ({cardinality}), no target. Using frequency encoding."

                elif cardinality <= 1000:
                    # Hash encoding
                    num_buckets = min(100, cardinality // 10)
                    df_encoded[f"{col}_hash"] = df_encoded[col].apply(
                        lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % num_buckets
                    )
                    df_encoded = df_encoded.drop(columns=[col])
                    method = "hash_encoding"
                    reasoning = f"High cardinality ({cardinality}). Hash encoding with {num_buckets} buckets."

                else:
                    # Frequency encoding for very high cardinality
                    freq = df_encoded[col].value_counts(normalize=True)
                    df_encoded[f"{col}_freq"] = df_encoded[col].map(freq).fillna(0)
                    df_encoded = df_encoded.drop(columns=[col])
                    method = "frequency_encoding"
                    reasoning = f"Very high cardinality ({cardinality}). Frequency encoding to avoid explosion."

                encoding_report["columns_encoded"][col] = {
                    "cardinality": int(cardinality),
                    "method": method,
                    "reasoning": reasoning
                }

                logger.info(f"✅ {reasoning}")

            self.engineering_report["actions"].append({
                "action": "advanced_encoding",
                "details": encoding_report
            })

            return df_encoded, encoding_report

        except Exception as e:
            logger.error(f"❌ Advanced encoding failed: {e}")
            return df, {"error": str(e)}

    def create_polynomial_features(
        self,
        df: pd.DataFrame,
        numerical_columns: Optional[List[str]] = None,
        degree: int = 2,
        max_input_features: int = 10
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create polynomial and interaction features.

        Args:
            df: Pandas DataFrame
            numerical_columns: Columns to use (auto-detects if None)
            degree: Polynomial degree (default 2)
            max_input_features: Max features to prevent explosion

        Returns:
            Tuple of (transformed DataFrame, polynomial report)
        """
        try:
            df_poly = df.copy()

            if numerical_columns is None:
                numerical_columns = df_poly.select_dtypes(include=[np.number]).columns.tolist()

            poly_report = {
                "features_created": 0,
                "input_features": len(numerical_columns),
                "applied": False
            }

            # Prevent feature explosion
            if len(numerical_columns) > max_input_features:
                poly_report["reasoning"] = f"Too many features ({len(numerical_columns)}). Skipping polynomial to prevent explosion."
                logger.warning(f"⚠️ {poly_report['reasoning']}")
                return df_poly, poly_report

            if len(numerical_columns) < 2:
                poly_report["reasoning"] = "Need at least 2 numerical features for polynomial expansion."
                return df_poly, poly_report

            # Create polynomial features
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            poly_array = poly.fit_transform(df_poly[numerical_columns])
            poly_feature_names = poly.get_feature_names_out(numerical_columns)

            # Add new features (exclude originals)
            new_feature_count = len(poly_feature_names) - len(numerical_columns)

            for i, name in enumerate(poly_feature_names[len(numerical_columns):]):
                df_poly[f"poly_{name}"] = poly_array[:, len(numerical_columns) + i]

            poly_report["features_created"] = new_feature_count
            poly_report["applied"] = True
            poly_report["reasoning"] = f"Created degree-{degree} polynomial features from {numerical_columns} → {new_feature_count} new features"

            logger.info(f"✅ {poly_report['reasoning']}")

            self.engineering_report["actions"].append({
                "action": "create_polynomial_features",
                "details": poly_report
            })
            self.engineering_report["features_created"] += new_feature_count

            return df_poly, poly_report

        except Exception as e:
            logger.error(f"❌ Polynomial feature creation failed: {e}")
            return df, {"error": str(e)}

    def create_bins(
        self,
        df: pd.DataFrame,
        column: str,
        n_bins: int = 5,
        strategy: str = "quantile"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Discretize numerical column into bins.

        Args:
            df: Pandas DataFrame
            column: Column to bin
            n_bins: Number of bins
            strategy: "quantile" (equal-frequency) or "uniform" (equal-width)

        Returns:
            Tuple of (binned DataFrame, binning report)
        """
        try:
            df_binned = df.copy()

            if column not in df_binned.columns:
                return df, {"error": f"Column '{column}' not found"}

            if strategy == "quantile":
                df_binned[f"{column}_binned"] = pd.qcut(
                    df_binned[column],
                    q=n_bins,
                    labels=False,
                    duplicates='drop'
                )
                reasoning = f"Binned '{column}' into {n_bins} equal-frequency groups (better captures non-linear patterns)"
            else:  # uniform
                df_binned[f"{column}_binned"] = pd.cut(
                    df_binned[column],
                    bins=n_bins,
                    labels=False
                )
                reasoning = f"Binned '{column}' into {n_bins} equal-width groups"

            binning_report = {
                "column": column,
                "n_bins": n_bins,
                "strategy": strategy,
                "new_column": f"{column}_binned",
                "reasoning": reasoning
            }

            logger.info(f"✅ {reasoning}")

            self.engineering_report["actions"].append({
                "action": "create_bins",
                "details": binning_report
            })
            self.engineering_report["features_created"] += 1

            return df_binned, binning_report

        except Exception as e:
            logger.error(f"❌ Binning failed: {e}")
            return df, {"error": str(e)}

    def vectorize_text(
        self,
        df: pd.DataFrame,
        text_column: str,
        method: str = "auto",
        max_features: int = 100
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert text to numerical features using TF-IDF or Count vectorization.

        Args:
            df: Pandas DataFrame
            text_column: Column containing text
            method: "tfidf", "count", or "auto"
            max_features: Maximum number of features to create

        Returns:
            Tuple of (vectorized DataFrame, vectorization report)
        """
        try:
            df_vec = df.copy()

            if text_column not in df_vec.columns:
                return df, {"error": f"Column '{text_column}' not found"}

            # Calculate average text length
            avg_length = df_vec[text_column].dropna().astype(str).str.split().str.len().mean()

            # Auto-select method
            if method == "auto":
                if avg_length > 20:
                    method = "tfidf"
                else:
                    method = "count"

            # Apply vectorization
            if method == "tfidf":
                vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
                vectors = vectorizer.fit_transform(df_vec[text_column].fillna(""))
                feature_names = [f"{text_column}_tfidf_{i}" for i in range(vectors.shape[1])]
                reasoning = f"Applied TF-IDF to '{text_column}' (avg {avg_length:.0f} words) → top {vectors.shape[1]} features"
            else:  # count
                vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
                vectors = vectorizer.fit_transform(df_vec[text_column].fillna(""))
                feature_names = [f"{text_column}_count_{i}" for i in range(vectors.shape[1])]
                reasoning = f"Applied CountVectorizer to '{text_column}' (avg {avg_length:.0f} words) → {vectors.shape[1]} features"

            # Add vector features to dataframe
            vec_df = pd.DataFrame(vectors.toarray(), columns=feature_names, index=df_vec.index)
            df_vec = pd.concat([df_vec, vec_df], axis=1)
            df_vec = df_vec.drop(columns=[text_column])

            vectorization_report = {
                "column": text_column,
                "method": method,
                "features_created": len(feature_names),
                "avg_text_length": float(avg_length),
                "reasoning": reasoning
            }

            logger.info(f"✅ {reasoning}")

            self.engineering_report["actions"].append({
                "action": "vectorize_text",
                "details": vectorization_report
            })
            self.engineering_report["features_created"] += len(feature_names)

            return df_vec, vectorization_report

        except Exception as e:
            logger.error(f"❌ Text vectorization failed: {e}")
            return df, {"error": str(e)}

    def create_ratio_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        correlation_threshold: float = 0.3
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create ratio features from pairs of numerical columns.

        Args:
            df: Pandas DataFrame
            target_column: Optional target to check correlation
            correlation_threshold: Minimum correlation to create ratio

        Returns:
            Tuple of (enhanced DataFrame, ratio report)
        """
        try:
            df_ratio = df.copy()

            numerical_cols = df_ratio.select_dtypes(include=[np.number]).columns.tolist()
            if target_column and target_column in numerical_cols:
                numerical_cols.remove(target_column)

            ratio_report = {
                "ratios_created": [],
                "total_ratios": 0
            }

            # Find pairs where both correlate with target
            for i, col1 in enumerate(numerical_cols):
                for col2 in numerical_cols[i+1:]:
                    # Check if both correlate with target
                    if target_column and target_column in df.columns:
                        corr1 = abs(df_ratio[col1].corr(df_ratio[target_column]))
                        corr2 = abs(df_ratio[col2].corr(df_ratio[target_column]))

                        if corr1 > correlation_threshold and corr2 > correlation_threshold:
                            # Create ratio (avoid division by zero)
                            if (df_ratio[col2] != 0).all():
                                ratio_name = f"{col1}_per_{col2}"
                                df_ratio[ratio_name] = df_ratio[col1] / df_ratio[col2].replace(0, 1)

                                ratio_report["ratios_created"].append({
                                    "feature": ratio_name,
                                    "numerator": col1,
                                    "denominator": col2,
                                    "reasoning": f"Both {col1} ({corr1:.2f}) and {col2} ({corr2:.2f}) correlate with target"
                                })

                                logger.info(f"✅ Created '{ratio_name}' - both features correlate with target")

            ratio_report["total_ratios"] = len(ratio_report["ratios_created"])

            if ratio_report["total_ratios"] > 0:
                self.engineering_report["actions"].append({
                    "action": "create_ratio_features",
                    "details": ratio_report
                })
                self.engineering_report["features_created"] += ratio_report["total_ratios"]

            return df_ratio, ratio_report

        except Exception as e:
            logger.error(f"❌ Ratio feature creation failed: {e}")
            return df, {"error": str(e)}

    def advanced_datetime_features(
        self,
        df: pd.DataFrame,
        datetime_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extract advanced datetime features including cyclical encoding.

        Args:
            df: Pandas DataFrame
            datetime_columns: Columns to process (auto-detects if None)

        Returns:
            Tuple of (enhanced DataFrame, datetime report)
        """
        try:
            df_dt = df.copy()

            if datetime_columns is None:
                datetime_columns = df_dt.select_dtypes(include=['datetime64']).columns.tolist()

            datetime_report = {
                "columns_processed": {},
                "total_features_created": 0
            }

            for col in datetime_columns:
                if col not in df_dt.columns:
                    continue

                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(df_dt[col]):
                    df_dt[col] = pd.to_datetime(df_dt[col], errors='coerce')

                features_created = []

                # Basic extractions
                df_dt[f"{col}_year"] = df_dt[col].dt.year
                df_dt[f"{col}_month"] = df_dt[col].dt.month
                df_dt[f"{col}_day"] = df_dt[col].dt.day
                df_dt[f"{col}_dayofweek"] = df_dt[col].dt.dayofweek
                df_dt[f"{col}_hour"] = df_dt[col].dt.hour
                features_created.extend(["year", "month", "day", "dayofweek", "hour"])

                # Advanced extractions
                df_dt[f"{col}_quarter"] = df_dt[col].dt.quarter
                df_dt[f"{col}_week_of_year"] = df_dt[col].dt.isocalendar().week
                df_dt[f"{col}_is_month_start"] = df_dt[col].dt.is_month_start.astype(int)
                df_dt[f"{col}_is_month_end"] = df_dt[col].dt.is_month_end.astype(int)
                df_dt[f"{col}_is_quarter_end"] = df_dt[col].dt.is_quarter_end.astype(int)
                df_dt[f"{col}_is_weekend"] = (df_dt[col].dt.dayofweek >= 5).astype(int)
                features_created.extend(["quarter", "week_of_year", "is_month_start", "is_month_end",
                                       "is_quarter_end", "is_weekend"])

                # Days since reference (first date)
                min_date = df_dt[col].min()
                df_dt[f"{col}_days_since_start"] = (df_dt[col] - min_date).dt.days
                features_created.append("days_since_start")

                # Cyclical encoding for month (sin/cos for continuity)
                df_dt[f"{col}_month_sin"] = np.sin(2 * np.pi * df_dt[col].dt.month / 12)
                df_dt[f"{col}_month_cos"] = np.cos(2 * np.pi * df_dt[col].dt.month / 12)
                features_created.extend(["month_sin", "month_cos"])

                # Cyclical encoding for day of week
                df_dt[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * df_dt[col].dt.dayofweek / 7)
                df_dt[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * df_dt[col].dt.dayofweek / 7)
                features_created.extend(["dayofweek_sin", "dayofweek_cos"])

                datetime_report["columns_processed"][col] = {
                    "features_created": features_created,
                    "count": len(features_created),
                    "reasoning": f"Extracted {len(features_created)} features from '{col}': {', '.join(features_created[:5])}..."
                }

                datetime_report["total_features_created"] += len(features_created)

                logger.info(f"✅ Extracted {len(features_created)} datetime features from '{col}'")

            if datetime_report["columns_processed"]:
                self.engineering_report["actions"].append({
                    "action": "advanced_datetime_features",
                    "details": datetime_report
                })
                self.engineering_report["features_created"] += datetime_report["total_features_created"]

            return df_dt, datetime_report

        except Exception as e:
            logger.error(f"❌ Advanced datetime extraction failed: {e}")
            return df, {"error": str(e)}

    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_column: str,
        value_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create group-level aggregation features.

        Args:
            df: Pandas DataFrame
            group_column: Column to group by (e.g., customer_id)
            value_columns: Columns to aggregate (uses numerical if None)

        Returns:
            Tuple of (enhanced DataFrame, aggregation report)
        """
        try:
            df_agg = df.copy()

            if group_column not in df_agg.columns:
                return df, {"error": f"Group column '{group_column}' not found"}

            # Check if grouping makes sense (multiple rows per group)
            group_counts = df_agg[group_column].value_counts()
            if (group_counts == 1).all():
                return df, {"error": f"No repeated values in '{group_column}' - grouping not meaningful"}

            if value_columns is None:
                value_columns = df_agg.select_dtypes(include=[np.number]).columns.tolist()

            agg_report = {
                "group_column": group_column,
                "features_created": [],
                "total_features": 0
            }

            # Create aggregations
            for col in value_columns:
                if col == group_column:
                    continue

                # Count
                group_counts = df_agg.groupby(group_column).size()
                df_agg[f"{group_column}_count"] = df_agg[group_column].map(group_counts)
                if f"{group_column}_count" not in agg_report["features_created"]:
                    agg_report["features_created"].append(f"{group_column}_count")

                # Mean
                group_mean = df_agg.groupby(group_column)[col].mean()
                df_agg[f"{group_column}_{col}_mean"] = df_agg[group_column].map(group_mean)
                agg_report["features_created"].append(f"{group_column}_{col}_mean")

                # Sum
                group_sum = df_agg.groupby(group_column)[col].sum()
                df_agg[f"{group_column}_{col}_sum"] = df_agg[group_column].map(group_sum)
                agg_report["features_created"].append(f"{group_column}_{col}_sum")

            agg_report["total_features"] = len(agg_report["features_created"])
            agg_report["reasoning"] = f"Detected {group_column} with multiple occurrences. Created {agg_report['total_features']} aggregation features."

            logger.info(f"✅ {agg_report['reasoning']}")

            self.engineering_report["actions"].append({
                "action": "create_aggregation_features",
                "details": agg_report
            })
            self.engineering_report["features_created"] += agg_report["total_features"]

            return df_agg, agg_report

        except Exception as e:
            logger.error(f"❌ Aggregation feature creation failed: {e}")
            return df, {"error": str(e)}

    def create_missing_indicators(
        self,
        df: pd.DataFrame,
        original_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create binary indicators for columns that had missing values.

        Args:
            df: Current DataFrame (after imputation)
            original_df: Original DataFrame (before imputation)

        Returns:
            Tuple of (enhanced DataFrame, missing indicator report)
        """
        try:
            df_indicators = df.copy()

            indicator_report = {
                "indicators_created": [],
                "total_indicators": 0
            }

            for col in original_df.columns:
                if col in df_indicators.columns:
                    # Check if column had nulls in original
                    null_pct = original_df[col].isna().sum() / len(original_df) * 100

                    if null_pct > 0:
                        # Create binary indicator
                        df_indicators[f"{col}_was_missing"] = original_df[col].isna().astype(int)

                        indicator_report["indicators_created"].append({
                            "column": col,
                            "indicator_name": f"{col}_was_missing",
                            "null_percentage": float(null_pct),
                            "reasoning": f"Created '{col}_was_missing' ({null_pct:.1f}% had nulls before imputation)"
                        })

            indicator_report["total_indicators"] = len(indicator_report["indicators_created"])

            if indicator_report["total_indicators"] > 0:
                logger.info(f"✅ Created {indicator_report['total_indicators']} missing value indicators")

                self.engineering_report["actions"].append({
                    "action": "create_missing_indicators",
                    "details": indicator_report
                })
                self.engineering_report["features_created"] += indicator_report["total_indicators"]

            return df_indicators, indicator_report

        except Exception as e:
            logger.error(f"❌ Missing indicator creation failed: {e}")
            return df, {"error": str(e)}

    def get_engineering_summary(self) -> Dict[str, Any]:
        """
        Get summary of all feature engineering performed.

        Returns:
            Dictionary with engineering summary
        """
        return self.engineering_report
