"""
Data Cleaning Module
Handles missing values, duplicates, outliers, and data type conversions.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from difflib import SequenceMatcher
import re
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Comprehensive data cleaning utility for data science pipeline.
    """

    def __init__(self):
        """Initialize DataCleaner."""
        self.cleaning_report = {
            "actions": [],
            "summary": {}
        }

    def detect_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect and analyze missing values in DataFrame.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary with missing value analysis
        """
        try:
            total_cells = df.shape[0] * df.shape[1]
            total_missing = df.isna().sum().sum()
            missing_percent = (total_missing / total_cells * 100) if total_cells > 0 else 0

            missing_by_column = {}
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    missing_by_column[col] = {
                        "count": int(missing_count),
                        "percent": float(missing_count / len(df) * 100)
                    }

            missing_report = {
                "total_missing": int(total_missing),
                "total_cells": int(total_cells),
                "missing_percent": float(missing_percent),
                "columns_with_missing": missing_by_column,
                "columns_affected": len(missing_by_column),
                "rows_with_any_missing": int(df.isna().any(axis=1).sum())
            }

            logger.info(f"📊 Missing values: {total_missing} ({missing_percent:.2f}%) across {len(missing_by_column)} columns")

            return missing_report

        except Exception as e:
            logger.error(f"❌ Failed to detect missing values: {e}")
            return {"error": str(e)}

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        strategy: Union[str, Dict[str, str]] = "auto",
        numerical_strategy: str = "median",
        categorical_strategy: str = "mode"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Impute missing values using specified strategies.

        Args:
            df: Pandas DataFrame
            strategy: Imputation strategy ('auto', 'drop', or dict mapping columns to strategies)
            numerical_strategy: Strategy for numerical columns ('mean', 'median', 'forward_fill', 'backward_fill', 'zero')
            categorical_strategy: Strategy for categorical columns ('mode', 'constant', 'forward_fill', 'backward_fill')

        Returns:
            Tuple of (cleaned DataFrame, imputation report)
        """
        try:
            df_clean = df.copy()
            imputation_report = {
                "columns_imputed": {},
                "rows_dropped": 0,
                "total_values_imputed": 0
            }

            missing_before = df_clean.isna().sum().sum()

            # Drop strategy
            if strategy == "drop":
                rows_before = len(df_clean)
                df_clean = df_clean.dropna()
                rows_dropped = rows_before - len(df_clean)
                imputation_report["rows_dropped"] = rows_dropped
                logger.info(f"🗑️ Dropped {rows_dropped} rows with missing values")
                return df_clean, imputation_report

            # Auto strategy: impute based on column type
            for col in df_clean.columns:
                if df_clean[col].isna().sum() == 0:
                    continue

                missing_count = df_clean[col].isna().sum()
                col_strategy = None

                # Determine strategy for this column
                if isinstance(strategy, dict) and col in strategy:
                    col_strategy = strategy[col]
                elif strategy == "auto":
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        col_strategy = numerical_strategy
                    else:
                        col_strategy = categorical_strategy

                # Apply imputation
                if col_strategy == "mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].mean()
                    df_clean[col].fillna(fill_value, inplace=True)

                elif col_strategy == "median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                    fill_value = df_clean[col].median()
                    df_clean[col].fillna(fill_value, inplace=True)

                elif col_strategy == "mode":
                    mode_values = df_clean[col].mode()
                    if len(mode_values) > 0:
                        fill_value = mode_values[0]
                        df_clean[col].fillna(fill_value, inplace=True)

                elif col_strategy == "forward_fill":
                    df_clean[col].fillna(method='ffill', inplace=True)

                elif col_strategy == "backward_fill":
                    df_clean[col].fillna(method='bfill', inplace=True)

                elif col_strategy == "zero":
                    df_clean[col].fillna(0, inplace=True)

                elif col_strategy == "constant":
                    df_clean[col].fillna("MISSING", inplace=True)

                imputation_report["columns_imputed"][col] = {
                    "strategy": col_strategy,
                    "values_imputed": int(missing_count),
                    "fill_value": str(fill_value) if 'fill_value' in locals() else "N/A"
                }

            missing_after = df_clean.isna().sum().sum()
            imputation_report["total_values_imputed"] = int(missing_before - missing_after)

            logger.info(f"✅ Imputed {imputation_report['total_values_imputed']} missing values across {len(imputation_report['columns_imputed'])} columns")

            self.cleaning_report["actions"].append({
                "action": "impute_missing_values",
                "details": imputation_report
            })

            return df_clean, imputation_report

        except Exception as e:
            logger.error(f"❌ Imputation failed: {e}")
            return df, {"error": str(e)}

    def detect_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect duplicate rows in DataFrame.

        Args:
            df: Pandas DataFrame
            subset: Optional list of columns to check for duplicates

        Returns:
            Dictionary with duplicate analysis
        """
        try:
            duplicate_mask = df.duplicated(subset=subset, keep='first')
            duplicate_count = duplicate_mask.sum()
            duplicate_percent = (duplicate_count / len(df) * 100) if len(df) > 0 else 0

            duplicate_report = {
                "total_duplicates": int(duplicate_count),
                "duplicate_percent": float(duplicate_percent),
                "unique_rows": int(len(df) - duplicate_count),
                "subset_columns": subset if subset else "all"
            }

            logger.info(f"🔍 Found {duplicate_count} duplicate rows ({duplicate_percent:.2f}%)")

            return duplicate_report

        except Exception as e:
            logger.error(f"❌ Failed to detect duplicates: {e}")
            return {"error": str(e)}

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Remove duplicate rows from DataFrame.

        Args:
            df: Pandas DataFrame
            subset: Optional list of columns to check for duplicates
            keep: Which duplicates to keep ('first', 'last', False)

        Returns:
            Tuple of (cleaned DataFrame, removal report)
        """
        try:
            rows_before = len(df)
            df_clean = df.drop_duplicates(subset=subset, keep=keep)
            rows_removed = rows_before - len(df_clean)

            removal_report = {
                "rows_removed": int(rows_removed),
                "rows_remaining": int(len(df_clean)),
                "subset_columns": subset if subset else "all",
                "keep_strategy": keep
            }

            logger.info(f"✅ Removed {rows_removed} duplicate rows")

            self.cleaning_report["actions"].append({
                "action": "remove_duplicates",
                "details": removal_report
            })

            return df_clean, removal_report

        except Exception as e:
            logger.error(f"❌ Failed to remove duplicates: {e}")
            return df, {"error": str(e)}

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect outliers in numerical columns.

        Args:
            df: Pandas DataFrame
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold value (1.5 for IQR, 3 for Z-score)
            columns: Optional list of columns to check

        Returns:
            Dictionary with outlier analysis
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            outlier_report = {
                "method": method,
                "threshold": threshold,
                "columns": {}
            }

            for col in columns:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue

                col_data = df[col].dropna()

                if len(col_data) == 0:
                    continue

                if method == "iqr":
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)

                elif method == "zscore":
                    z_scores = np.abs(stats.zscore(col_data))
                    outlier_mask = pd.Series([False] * len(df), index=df.index)
                    outlier_mask.loc[col_data.index] = z_scores > threshold

                else:
                    logger.warning(f"⚠️ Unknown outlier detection method: {method}")
                    continue

                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    outlier_report["columns"][col] = {
                        "count": int(outlier_count),
                        "percent": float(outlier_count / len(df) * 100),
                        "lower_bound": float(lower_bound) if method == "iqr" else None,
                        "upper_bound": float(upper_bound) if method == "iqr" else None,
                        "outlier_indices": outlier_mask[outlier_mask].index.tolist()[:10]  # First 10
                    }

            total_outliers = sum(info["count"] for info in outlier_report["columns"].values())
            outlier_report["total_outliers"] = int(total_outliers)
            outlier_report["columns_affected"] = len(outlier_report["columns"])

            logger.info(f"📊 Detected {total_outliers} outliers across {len(outlier_report['columns'])} columns using {method} method")

            return outlier_report

        except Exception as e:
            logger.error(f"❌ Failed to detect outliers: {e}")
            return {"error": str(e)}

    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = "clip",
        detection_method: str = "iqr",
        threshold: float = 1.5,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle outliers using specified method.

        Args:
            df: Pandas DataFrame
            method: Handling method ('clip', 'remove', 'cap')
            detection_method: Detection method ('iqr' or 'zscore')
            threshold: Detection threshold
            columns: Optional list of columns to process

        Returns:
            Tuple of (cleaned DataFrame, handling report)
        """
        try:
            df_clean = df.copy()

            if columns is None:
                columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

            handling_report = {
                "method": method,
                "detection_method": detection_method,
                "columns_processed": {},
                "total_outliers_handled": 0,
                "rows_removed": 0
            }

            rows_before = len(df_clean)

            for col in columns:
                if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
                    continue

                col_data = df_clean[col].dropna()

                if len(col_data) == 0:
                    continue

                # Detect outliers
                if detection_method == "iqr":
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                elif detection_method == "zscore":
                    mean_val = col_data.mean()
                    std_val = col_data.std()
                    lower_bound = mean_val - threshold * std_val
                    upper_bound = mean_val + threshold * std_val

                else:
                    continue

                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count == 0:
                    continue

                # Handle outliers
                if method == "clip":
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

                elif method == "cap":
                    df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                    df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

                elif method == "remove":
                    df_clean = df_clean[~outlier_mask]

                handling_report["columns_processed"][col] = {
                    "outliers_handled": int(outlier_count),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
                handling_report["total_outliers_handled"] += int(outlier_count)

            handling_report["rows_removed"] = int(rows_before - len(df_clean))

            logger.info(f"✅ Handled {handling_report['total_outliers_handled']} outliers using {method} method")

            self.cleaning_report["actions"].append({
                "action": "handle_outliers",
                "details": handling_report
            })

            return df_clean, handling_report

        except Exception as e:
            logger.error(f"❌ Failed to handle outliers: {e}")
            return df, {"error": str(e)}

    def convert_data_types(
        self,
        df: pd.DataFrame,
        type_mapping: Optional[Dict[str, str]] = None,
        auto_convert: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert column data types.

        Args:
            df: Pandas DataFrame
            type_mapping: Optional dictionary mapping column names to target types
            auto_convert: Whether to automatically detect and convert types

        Returns:
            Tuple of (converted DataFrame, conversion report)
        """
        try:
            df_clean = df.copy()
            conversion_report = {
                "columns_converted": {},
                "errors": []
            }

            # Auto-convert datetime strings
            if auto_convert:
                for col in df_clean.select_dtypes(include=['object']).columns:
                    try:
                        # Try to convert to datetime
                        converted = pd.to_datetime(df_clean[col], errors='coerce')
                        if converted.notna().sum() / len(df_clean) > 0.8:  # 80% successfully converted
                            df_clean[col] = converted
                            conversion_report["columns_converted"][col] = {
                                "from": "object",
                                "to": "datetime64",
                                "auto": True
                            }
                            logger.info(f"🔄 Auto-converted {col} to datetime")
                    except:
                        pass

            # Apply manual type mapping
            if type_mapping:
                for col, target_type in type_mapping.items():
                    if col not in df_clean.columns:
                        conversion_report["errors"].append(f"Column '{col}' not found")
                        continue

                    try:
                        original_type = str(df_clean[col].dtype)

                        if target_type == "int":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').astype('Int64')
                        elif target_type == "float":
                            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                        elif target_type == "str":
                            df_clean[col] = df_clean[col].astype(str)
                        elif target_type == "category":
                            df_clean[col] = df_clean[col].astype('category')
                        elif target_type == "datetime":
                            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                        elif target_type == "bool":
                            df_clean[col] = df_clean[col].astype(bool)

                        conversion_report["columns_converted"][col] = {
                            "from": original_type,
                            "to": target_type,
                            "auto": False
                        }

                        logger.info(f"🔄 Converted {col} from {original_type} to {target_type}")

                    except Exception as e:
                        error_msg = f"Failed to convert {col} to {target_type}: {str(e)}"
                        conversion_report["errors"].append(error_msg)
                        logger.error(f"❌ {error_msg}")

            logger.info(f"✅ Converted {len(conversion_report['columns_converted'])} columns")

            self.cleaning_report["actions"].append({
                "action": "convert_data_types",
                "details": conversion_report
            })

            return df_clean, conversion_report

        except Exception as e:
            logger.error(f"❌ Type conversion failed: {e}")
            return df, {"error": str(e)}

    def standardize_categorical(
        self,
        df: pd.DataFrame,
        similarity_threshold: float = 0.8,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Standardize inconsistent categorical values using fuzzy matching.

        Args:
            df: Pandas DataFrame
            similarity_threshold: Minimum similarity ratio (0-1) to group values
            columns: Optional list of columns to standardize

        Returns:
            Tuple of (standardized DataFrame, standardization report)
        """
        try:
            df_clean = df.copy()

            if columns is None:
                columns = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

            standardization_report = {
                "columns_standardized": {},
                "total_replacements": 0
            }

            for col in columns:
                if col not in df_clean.columns:
                    continue

                # Get unique values (case-sensitive)
                unique_vals = df_clean[col].dropna().unique()

                if len(unique_vals) <= 1:
                    continue

                # Find groups of similar values
                value_groups = {}
                processed = set()

                for val in unique_vals:
                    if val in processed:
                        continue

                    val_str = str(val).strip().lower()
                    similar_group = [val]

                    for other_val in unique_vals:
                        if val == other_val or other_val in processed:
                            continue

                        other_str = str(other_val).strip().lower()
                        similarity = SequenceMatcher(None, val_str, other_str).ratio()

                        if similarity >= similarity_threshold:
                            similar_group.append(other_val)
                            processed.add(other_val)

                    if len(similar_group) > 1:
                        # Choose most frequent as canonical
                        value_counts = df_clean[col].value_counts()
                        canonical = max(similar_group, key=lambda x: value_counts.get(x, 0))
                        value_groups[canonical] = similar_group
                        processed.add(val)

                # Apply standardization
                if value_groups:
                    replacements = 0
                    for canonical, group in value_groups.items():
                        for variant in group:
                            if variant != canonical:
                                mask = df_clean[col] == variant
                                replacements += mask.sum()
                                df_clean.loc[mask, col] = canonical

                    standardization_report["columns_standardized"][col] = {
                        "groups_found": len(value_groups),
                        "replacements": int(replacements),
                        "examples": {str(canonical): [str(v) for v in group]
                                   for canonical, group in list(value_groups.items())[:3]},
                        "reasoning": f"Standardized {replacements} values using fuzzy matching (threshold={similarity_threshold})"
                    }
                    standardization_report["total_replacements"] += replacements

                    logger.info(f"✅ Standardized {col}: {replacements} values grouped into {len(value_groups)} canonical forms")

            self.cleaning_report["actions"].append({
                "action": "standardize_categorical",
                "details": standardization_report
            })

            return df_clean, standardization_report

        except Exception as e:
            logger.error(f"❌ Categorical standardization failed: {e}")
            return df, {"error": str(e)}

    def detect_invalid_values(
        self,
        df: pd.DataFrame,
        rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and remove invalid values based on domain rules.

        Args:
            df: Pandas DataFrame
            rules: Optional dict mapping column names to validation rules
                   Example: {"age": {"min": 0, "max": 120}, "price": {"min": 0}}

        Returns:
            Tuple of (cleaned DataFrame, invalid values report)
        """
        try:
            df_clean = df.copy()
            invalid_report = {
                "columns_checked": {},
                "total_invalid_rows": 0,
                "rows_removed": 0
            }

            rows_before = len(df_clean)
            invalid_mask = pd.Series([False] * len(df_clean), index=df_clean.index)

            # Auto-detect common invalid patterns
            for col in df_clean.columns:
                col_invalids = 0
                reasoning = []

                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    # Check for common numerical invalid values
                    col_name_lower = col.lower()

                    # Age validation
                    if 'age' in col_name_lower:
                        mask = (df_clean[col] < 0) | (df_clean[col] > 120)
                        col_invalids += mask.sum()
                        invalid_mask |= mask
                        if mask.sum() > 0:
                            reasoning.append(f"{mask.sum()} values outside valid age range (0-120)")

                    # Price/amount validation
                    elif any(word in col_name_lower for word in ['price', 'amount', 'cost', 'salary']):
                        mask = df_clean[col] < 0
                        col_invalids += mask.sum()
                        invalid_mask |= mask
                        if mask.sum() > 0:
                            reasoning.append(f"{mask.sum()} negative values (prices cannot be negative)")

                    # Percentage validation
                    elif any(word in col_name_lower for word in ['percent', 'pct', 'rate']):
                        mask = (df_clean[col] < 0) | (df_clean[col] > 100)
                        col_invalids += mask.sum()
                        invalid_mask |= mask
                        if mask.sum() > 0:
                            reasoning.append(f"{mask.sum()} values outside 0-100 range")

                # Apply custom rules if provided
                if rules and col in rules:
                    rule = rules[col]
                    if 'min' in rule:
                        mask = df_clean[col] < rule['min']
                        col_invalids += mask.sum()
                        invalid_mask |= mask
                        if mask.sum() > 0:
                            reasoning.append(f"{mask.sum()} values below minimum ({rule['min']})")

                    if 'max' in rule:
                        mask = df_clean[col] > rule['max']
                        col_invalids += mask.sum()
                        invalid_mask |= mask
                        if mask.sum() > 0:
                            reasoning.append(f"{mask.sum()} values above maximum ({rule['max']})")

                    if 'allowed_values' in rule:
                        mask = ~df_clean[col].isin(rule['allowed_values'])
                        col_invalids += mask.sum()
                        invalid_mask |= mask
                        if mask.sum() > 0:
                            reasoning.append(f"{mask.sum()} values not in allowed set")

                if col_invalids > 0:
                    invalid_report["columns_checked"][col] = {
                        "invalid_count": int(col_invalids),
                        "reasoning": "; ".join(reasoning)
                    }

            # Remove invalid rows
            df_clean = df_clean[~invalid_mask]
            invalid_report["total_invalid_rows"] = int(invalid_mask.sum())
            invalid_report["rows_removed"] = int(rows_before - len(df_clean))

            if invalid_report["rows_removed"] > 0:
                logger.info(f"✅ Removed {invalid_report['rows_removed']} rows with invalid values")

            self.cleaning_report["actions"].append({
                "action": "detect_invalid_values",
                "details": invalid_report
            })

            return df_clean, invalid_report

        except Exception as e:
            logger.error(f"❌ Invalid value detection failed: {e}")
            return df, {"error": str(e)}

    def clean_text_data(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean text columns: fix encoding, remove HTML, strip whitespace.

        Args:
            df: Pandas DataFrame
            columns: Optional list of text columns to clean

        Returns:
            Tuple of (cleaned DataFrame, text cleaning report)
        """
        try:
            df_clean = df.copy()

            if columns is None:
                columns = df_clean.select_dtypes(include=['object']).columns.tolist()

            text_cleaning_report = {
                "columns_cleaned": {},
                "total_transformations": 0
            }

            for col in columns:
                if col not in df_clean.columns:
                    continue

                transformations = []

                # Check if column contains text (not just short categorical values)
                avg_length = df_clean[col].dropna().astype(str).str.len().mean()

                if avg_length < 10:  # Skip short categorical columns
                    continue

                # Strip leading/trailing whitespace
                before_strip = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].astype(str).str.strip()
                if (before_strip != df_clean[col]).sum() > 0:
                    transformations.append("Removed leading/trailing whitespace")

                # Remove HTML tags
                html_pattern = re.compile('<.*?>')
                if df_clean[col].astype(str).str.contains(html_pattern, regex=True).any():
                    df_clean[col] = df_clean[col].astype(str).str.replace(html_pattern, '', regex=True)
                    transformations.append("Removed HTML tags")

                # Fix common encoding errors
                encoding_fixes = {
                    'Ã©': 'é', 'Ã¨': 'è', 'Ã ': 'à', 'Ã¢': 'â',
                    'Ã´': 'ô', 'Ã»': 'û', 'Ã§': 'ç', 'Ã«': 'ë'
                }
                for wrong, correct in encoding_fixes.items():
                    if df_clean[col].astype(str).str.contains(wrong, regex=False).any():
                        df_clean[col] = df_clean[col].astype(str).str.replace(wrong, correct, regex=False)
                        transformations.append(f"Fixed encoding error: {wrong}→{correct}")

                # Remove excessive whitespace (multiple spaces → single space)
                df_clean[col] = df_clean[col].astype(str).str.replace(r'\s+', ' ', regex=True)

                if transformations:
                    text_cleaning_report["columns_cleaned"][col] = {
                        "transformations": transformations,
                        "reasoning": f"Cleaned '{col}' column: {', '.join(transformations)}"
                    }
                    text_cleaning_report["total_transformations"] += len(transformations)
                    logger.info(f"✅ Cleaned text column '{col}': {len(transformations)} transformations")

            self.cleaning_report["actions"].append({
                "action": "clean_text_data",
                "details": text_cleaning_report
            })

            return df_clean, text_cleaning_report

        except Exception as e:
            logger.error(f"❌ Text cleaning failed: {e}")
            return df, {"error": str(e)}

    def handle_high_null_columns(
        self,
        df: pd.DataFrame,
        threshold: float = 0.8
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Drop columns with high percentage of null values.

        Args:
            df: Pandas DataFrame
            threshold: Null percentage threshold (default 0.8 = 80%)

        Returns:
            Tuple of (cleaned DataFrame, high null columns report)
        """
        try:
            df_clean = df.copy()
            high_null_report = {
                "columns_dropped": {},
                "total_dropped": 0
            }

            columns_to_drop = []

            for col in df_clean.columns:
                null_pct = df_clean[col].isna().sum() / len(df_clean)

                if null_pct >= threshold:
                    columns_to_drop.append(col)
                    high_null_report["columns_dropped"][col] = {
                        "null_percentage": float(null_pct * 100),
                        "reasoning": f"Dropped '{col}' ({null_pct*100:.1f}% null - threshold: {threshold*100}%)"
                    }

            if columns_to_drop:
                df_clean = df_clean.drop(columns=columns_to_drop)
                high_null_report["total_dropped"] = len(columns_to_drop)
                logger.info(f"✅ Dropped {len(columns_to_drop)} columns with >{threshold*100}% nulls: {columns_to_drop}")

            self.cleaning_report["actions"].append({
                "action": "handle_high_null_columns",
                "details": high_null_report
            })

            return df_clean, high_null_report

        except Exception as e:
            logger.error(f"❌ High null column handling failed: {e}")
            return df, {"error": str(e)}

    def validate_and_standardize_dates(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        output_format: str = 'ISO'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect, validate, and standardize date columns.

        Args:
            df: Pandas DataFrame
            columns: Optional list of date columns (auto-detects if None)
            output_format: Output format ('ISO' for YYYY-MM-DD)

        Returns:
            Tuple of (cleaned DataFrame, date validation report)
        """
        try:
            df_clean = df.copy()
            date_report = {
                "columns_standardized": {},
                "invalid_dates_removed": 0
            }

            # Auto-detect date columns if not specified
            if columns is None:
                columns = []
                for col in df_clean.select_dtypes(include=['object']).columns:
                    sample = df_clean[col].dropna().head(10)
                    try:
                        pd.to_datetime(sample, errors='coerce')
                        if pd.to_datetime(sample, errors='coerce').notna().sum() / len(sample) > 0.5:
                            columns.append(col)
                    except:
                        pass

            for col in columns:
                if col not in df_clean.columns:
                    continue

                original_dtype = str(df_clean[col].dtype)

                # Convert to datetime
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

                # Count invalid dates (resulted in NaT)
                invalid_count = df_clean[col].isna().sum() - df[col].isna().sum()

                # Standardize to ISO format (YYYY-MM-DD)
                if output_format == 'ISO':
                    df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d')

                date_report["columns_standardized"][col] = {
                    "original_type": original_dtype,
                    "invalid_dates": int(invalid_count),
                    "output_format": output_format,
                    "reasoning": f"Standardized mixed date formats to {output_format}; {invalid_count} invalid dates found"
                }

                date_report["invalid_dates_removed"] += int(invalid_count)
                logger.info(f"✅ Standardized date column '{col}' to {output_format} format")

            self.cleaning_report["actions"].append({
                "action": "validate_and_standardize_dates",
                "details": date_report
            })

            return df_clean, date_report

        except Exception as e:
            logger.error(f"❌ Date validation failed: {e}")
            return df, {"error": str(e)}

    def detect_low_variance_columns(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect and drop columns where most values are identical.

        Args:
            df: Pandas DataFrame
            threshold: Percentage threshold (default 0.95 = 95% same value)

        Returns:
            Tuple of (cleaned DataFrame, low variance report)
        """
        try:
            df_clean = df.copy()
            low_var_report = {
                "columns_dropped": {},
                "total_dropped": 0
            }

            columns_to_drop = []

            for col in df_clean.columns:
                if len(df_clean[col]) == 0:
                    continue

                # Get most common value
                value_counts = df_clean[col].value_counts()
                if len(value_counts) == 0:
                    continue

                most_common_count = value_counts.iloc[0]
                most_common_value = value_counts.index[0]
                most_common_pct = most_common_count / len(df_clean[col].dropna())

                if most_common_pct >= threshold:
                    columns_to_drop.append(col)
                    low_var_report["columns_dropped"][col] = {
                        "dominant_value": str(most_common_value),
                        "dominant_percentage": float(most_common_pct * 100),
                        "unique_values": int(len(value_counts)),
                        "reasoning": f"Dropped '{col}' ({most_common_pct*100:.1f}% same value - no predictive power)"
                    }

            if columns_to_drop:
                df_clean = df_clean.drop(columns=columns_to_drop)
                low_var_report["total_dropped"] = len(columns_to_drop)
                logger.info(f"✅ Dropped {len(columns_to_drop)} low-variance columns: {columns_to_drop}")

            self.cleaning_report["actions"].append({
                "action": "detect_low_variance_columns",
                "details": low_var_report
            })

            return df_clean, low_var_report

        except Exception as e:
            logger.error(f"❌ Low variance detection failed: {e}")
            return df, {"error": str(e)}

    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of all cleaning actions performed.

        Returns:
            Dictionary with cleaning summary
        """
        return self.cleaning_report


# Convenience function
def clean_dataframe(
    df: pd.DataFrame,
    handle_missing: bool = True,
    handle_duplicates: bool = True,
    handle_outliers: bool = False,
    missing_strategy: str = "auto",
    outlier_method: str = "clip"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Perform comprehensive data cleaning.

    Args:
        df: Pandas DataFrame
        handle_missing: Whether to handle missing values
        handle_duplicates: Whether to remove duplicates
        handle_outliers: Whether to handle outliers
        missing_strategy: Strategy for missing values
        outlier_method: Method for handling outliers

    Returns:
        Tuple of (cleaned DataFrame, full cleaning report)
    """
    cleaner = DataCleaner()
    df_clean = df.copy()
    full_report = {"steps": []}

    # Handle missing values
    if handle_missing:
        df_clean, missing_report = cleaner.impute_missing_values(df_clean, strategy=missing_strategy)
        full_report["steps"].append({"step": "missing_values", "report": missing_report})

    # Handle duplicates
    if handle_duplicates:
        df_clean, dup_report = cleaner.remove_duplicates(df_clean)
        full_report["steps"].append({"step": "duplicates", "report": dup_report})

    # Handle outliers
    if handle_outliers:
        df_clean, outlier_report = cleaner.handle_outliers(df_clean, method=outlier_method)
        full_report["steps"].append({"step": "outliers", "report": outlier_report})

    full_report["summary"] = cleaner.get_cleaning_summary()

    return df_clean, full_report


if __name__ == "__main__":
    # Test data cleaning
    logging.basicConfig(level=logging.INFO)

    # Create test DataFrame
    test_data = {
        'A': [1, 2, np.nan, 4, 5, 100, 7, 8],
        'B': ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b'],
        'C': [10, 20, 30, 10, 20, 30, 10, 20]
    }
    df = pd.DataFrame(test_data)

    print("Original DataFrame:")
    print(df)

    cleaner = DataCleaner()

    # Detect missing values
    missing = cleaner.detect_missing_values(df)
    print(f"\n📊 Missing values: {missing}")

    # Impute missing values
    df_imputed, impute_report = cleaner.impute_missing_values(df)
    print(f"\n✅ After imputation:")
    print(df_imputed)

    # Detect outliers
    outliers = cleaner.detect_outliers(df_imputed)
    print(f"\n📊 Outliers: {outliers}")

    # Handle outliers
    df_clean, outlier_report = cleaner.handle_outliers(df_imputed, method="clip")
    print(f"\n✅ After handling outliers:")
    print(df_clean)

    # Get summary
    summary = cleaner.get_cleaning_summary()
    print(f"\n📋 Cleaning summary: {summary}")
