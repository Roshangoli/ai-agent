"""
File Upload and Validation Handler
Handles CSV and Excel file uploads with validation and schema extraction.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration
MAX_FILE_SIZE_MB = 100
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}


class FileHandler:
    """
    Handles file upload, validation, and schema extraction for data science pipeline.
    """

    def __init__(self, max_size_mb: int = MAX_FILE_SIZE_MB):
        """
        Initialize FileHandler.

        Args:
            max_size_mb: Maximum allowed file size in megabytes
        """
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def upload_file(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Upload and load file into DataFrame.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (DataFrame, metadata_dict)
        """
        try:
            logger.info(f"📁 Uploading file: {file_path}")

            # Validate file exists
            if not os.path.exists(file_path):
                return None, {
                    "success": False,
                    "error": f"File not found: {file_path}"
                }

            # Validate file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_size_bytes:
                return None, {
                    "success": False,
                    "error": f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds maximum allowed size ({self.max_size_mb} MB)"
                }

            # Validate file extension
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                return None, {
                    "success": False,
                    "error": f"File type '{file_ext}' not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
                }

            # Load file based on extension
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return None, {
                    "success": False,
                    "error": f"Unsupported file format: {file_ext}"
                }

            logger.info(f"✅ File loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

            # Generate metadata
            metadata = {
                "success": True,
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_size_mb": file_size / 1024 / 1024,
                "file_type": file_ext,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
            }

            return df, metadata

        except pd.errors.EmptyDataError:
            logger.error("❌ File is empty")
            return None, {
                "success": False,
                "error": "File is empty or has no data"
            }
        except pd.errors.ParserError as e:
            logger.error(f"❌ Failed to parse file: {e}")
            return None, {
                "success": False,
                "error": f"Failed to parse file: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ File upload failed: {e}")
            return None, {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate DataFrame and return validation report.

        Args:
            df: Pandas DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        try:
            validation_report = {
                "success": True,
                "issues": [],
                "warnings": [],
                "info": {}
            }

            # Check if DataFrame is empty
            if df.empty:
                validation_report["success"] = False
                validation_report["issues"].append("DataFrame is empty")
                return validation_report

            # Check for duplicate column names
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            if duplicate_cols:
                validation_report["warnings"].append(
                    f"Duplicate column names found: {duplicate_cols}"
                )

            # Check for completely empty columns
            empty_cols = df.columns[df.isna().all()].tolist()
            if empty_cols:
                validation_report["warnings"].append(
                    f"Completely empty columns: {empty_cols}"
                )

            # Check for columns with single unique value
            single_value_cols = [col for col in df.columns if df[col].nunique() == 1]
            if single_value_cols:
                validation_report["warnings"].append(
                    f"Columns with single unique value (consider dropping): {single_value_cols}"
                )

            # Check data types
            validation_report["info"]["data_types"] = df.dtypes.value_counts().to_dict()

            # Check missing values percentage
            missing_pct = (df.isna().sum() / len(df) * 100).to_dict()
            high_missing = {k: v for k, v in missing_pct.items() if v > 50}
            if high_missing:
                validation_report["warnings"].append(
                    f"Columns with >50% missing values: {high_missing}"
                )

            validation_report["info"]["missing_values_percent"] = missing_pct

            # Memory usage
            validation_report["info"]["memory_usage_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024

            logger.info(f"✅ DataFrame validation complete: {len(validation_report['issues'])} issues, {len(validation_report['warnings'])} warnings")

            return validation_report

        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return {
                "success": False,
                "error": f"Validation error: {str(e)}"
            }

    def detect_data_types(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detect and classify data types for each column.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary mapping column names to type information
        """
        type_mapping = {}

        for col in df.columns:
            col_info = {
                "pandas_dtype": str(df[col].dtype),
                "inferred_type": None,
                "unique_values": df[col].nunique(),
                "missing_count": df[col].isna().sum(),
                "missing_percent": (df[col].isna().sum() / len(df) * 100),
                "sample_values": df[col].dropna().head(5).tolist()
            }

            # Infer semantic type
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 10 and df[col].nunique() / len(df) < 0.05:
                    col_info["inferred_type"] = "categorical_numeric"
                else:
                    col_info["inferred_type"] = "numerical"
                    col_info["stats"] = {
                        "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                        "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                        "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                        "median": float(df[col].median()) if not pd.isna(df[col].median()) else None
                    }

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                col_info["inferred_type"] = "datetime"
                col_info["date_range"] = {
                    "min": str(df[col].min()),
                    "max": str(df[col].max())
                }

            elif pd.api.types.is_object_dtype(df[col]):
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(100))
                    col_info["inferred_type"] = "datetime_string"
                except:
                    # Check if categorical
                    if df[col].nunique() / len(df) < 0.05:
                        col_info["inferred_type"] = "categorical"
                        col_info["categories"] = df[col].value_counts().head(10).to_dict()
                    else:
                        col_info["inferred_type"] = "text"

            elif pd.api.types.is_bool_dtype(df[col]):
                col_info["inferred_type"] = "boolean"

            else:
                col_info["inferred_type"] = "unknown"

            type_mapping[col] = col_info

        return type_mapping

    def get_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract comprehensive schema information from DataFrame.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary with schema information
        """
        type_info = self.detect_data_types(df)

        schema = {
            "shape": {
                "rows": df.shape[0],
                "columns": df.shape[1]
            },
            "columns": list(df.columns),
            "type_info": type_info,
            "summary": {
                "numerical_columns": [col for col, info in type_info.items() if info["inferred_type"] == "numerical"],
                "categorical_columns": [col for col, info in type_info.items() if info["inferred_type"] in ["categorical", "categorical_numeric"]],
                "datetime_columns": [col for col, info in type_info.items() if info["inferred_type"] in ["datetime", "datetime_string"]],
                "text_columns": [col for col, info in type_info.items() if info["inferred_type"] == "text"],
                "total_missing_values": int(df.isna().sum().sum())
            }
        }

        return schema

    def preview_data(self, df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
        """
        Generate data preview with head and tail samples.

        Args:
            df: Pandas DataFrame
            n_rows: Number of rows to show from head and tail

        Returns:
            Dictionary with preview data
        """
        preview = {
            "head": df.head(n_rows).to_dict(orient='records'),
            "tail": df.tail(n_rows).to_dict(orient='records'),
            "random_sample": df.sample(min(n_rows, len(df))).to_dict(orient='records') if len(df) > 0 else []
        }

        return preview


# Convenience functions
def upload_and_validate_file(file_path: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Upload file and run validation.

    Args:
        file_path: Path to file

    Returns:
        Tuple of (DataFrame, combined_report)
    """
    handler = FileHandler()
    df, upload_report = handler.upload_file(file_path)

    if not upload_report["success"]:
        return None, upload_report

    validation_report = handler.validate_dataframe(df)
    schema = handler.get_schema(df)

    combined_report = {
        **upload_report,
        "validation": validation_report,
        "schema": schema
    }

    return df, combined_report


if __name__ == "__main__":
    # Test the file handler
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        print("Usage: python file_handler.py <path_to_csv_or_excel>")
        sys.exit(1)

    handler = FileHandler()

    # Upload file
    df, metadata = handler.upload_file(test_file)

    if metadata["success"]:
        print(f"\n✅ File uploaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"\nMetadata: {metadata}")

        # Validate
        validation = handler.validate_dataframe(df)
        print(f"\n📋 Validation Report:")
        print(f"Success: {validation['success']}")
        print(f"Issues: {validation.get('issues', [])}")
        print(f"Warnings: {validation.get('warnings', [])}")

        # Get schema
        schema = handler.get_schema(df)
        print(f"\n📊 Schema:")
        print(f"Numerical columns: {schema['summary']['numerical_columns']}")
        print(f"Categorical columns: {schema['summary']['categorical_columns']}")

        # Preview
        preview = handler.preview_data(df, n_rows=3)
        print(f"\n👀 Data Preview (first 3 rows):")
        print(pd.DataFrame(preview['head']))

    else:
        print(f"\n❌ Upload failed: {metadata['error']}")
