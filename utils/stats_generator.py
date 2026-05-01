"""
Intelligent Stats Generator
Analyzes CSV data and generates relevant KPI cards dynamically.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StatsGenerator:
    """
    Generates intelligent statistics/KPIs based on dataset characteristics.
    Adapts to any CSV structure instead of assuming specific columns.
    """

    def generate_stats_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Analyze CSV and generate relevant statistics.

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with dynamic stats
        """
        try:
            df = pd.read_csv(csv_path)
            stats = self._analyze_and_generate_stats(df)
            logger.info(f"✅ Generated {len(stats)} stat cards")
            return stats

        except Exception as e:
            logger.error(f"❌ Stats generation failed: {e}")
            return self._get_fallback_stats()

    def generate_stats_from_metadata(
        self,
        column_names: List[str],
        row_count: int,
        column_types: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        Generate basic stats from metadata (faster, no need to read full CSV).

        Args:
            column_names: List of column names
            row_count: Number of rows
            column_types: Optional dict of column types

        Returns:
            Dictionary with basic stats
        """
        try:
            # Count different column types
            numeric_count = 0
            categorical_count = 0

            if column_types:
                for col_type in column_types.values():
                    if 'int' in col_type.lower() or 'float' in col_type.lower():
                        numeric_count += 1
                    else:
                        categorical_count += 1

            stats = {
                "total_rows": {
                    "label": "Total Records",
                    "value": f"{row_count:,}",
                    "icon": "database"
                },
                "total_columns": {
                    "label": "Columns",
                    "value": str(len(column_names)),
                    "icon": "columns"
                },
                "numeric_fields": {
                    "label": "Numeric Fields",
                    "value": str(numeric_count) if numeric_count > 0 else "-",
                    "icon": "hash"
                },
                "text_fields": {
                    "label": "Text Fields",
                    "value": str(categorical_count) if categorical_count > 0 else "-",
                    "icon": "text"
                }
            }

            logger.info(f"✅ Generated metadata-based stats")
            return stats

        except Exception as e:
            logger.error(f"❌ Metadata stats generation failed: {e}")
            return self._get_fallback_stats()

    def _analyze_and_generate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Deep analysis of DataFrame to generate contextual statistics.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary with dynamic statistics
        """
        stats = {}

        # Basic stats
        row_count = len(df)
        col_count = len(df.columns)

        # Analyze column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Generate stat cards based on data characteristics

        # Card 1: Total Records
        stats["total_rows"] = {
            "label": "Total Records",
            "value": f"{row_count:,}",
            "icon": "database"
        }

        # Card 2: Smart numeric summary
        if numeric_cols:
            # Find most likely "value" column (sales, amount, revenue, price, etc.)
            value_col = self._find_value_column(df, numeric_cols)

            if value_col:
                total_value = df[value_col].sum()
                avg_value = df[value_col].mean()

                # Format based on magnitude
                if total_value > 1000000:
                    formatted_value = f"${total_value/1000000:.1f}M"
                elif total_value > 1000:
                    formatted_value = f"${total_value/1000:.1f}K"
                else:
                    formatted_value = f"${total_value:,.0f}"

                label = self._generate_label_from_column(value_col)
                stats["numeric_summary"] = {
                    "label": f"Total {label}",
                    "value": formatted_value,
                    "icon": "dollar-sign"
                }
            else:
                # No clear value column, show count of numeric fields
                stats["numeric_fields"] = {
                    "label": "Numeric Fields",
                    "value": str(len(numeric_cols)),
                    "icon": "hash"
                }
        else:
            stats["columns"] = {
                "label": "Total Columns",
                "value": str(col_count),
                "icon": "columns"
            }

        # Card 3: Categorical diversity or unique entities
        if categorical_cols:
            # Find most diverse categorical column (likely products, customers, etc.)
            diversity_col = max(categorical_cols, key=lambda x: df[x].nunique())
            unique_count = df[diversity_col].nunique()

            label = self._generate_label_from_column(diversity_col)
            stats["diversity"] = {
                "label": f"Unique {label}",
                "value": f"{unique_count:,}",
                "icon": "layers"
            }
        else:
            stats["text_fields"] = {
                "label": "Text Fields",
                "value": str(len(categorical_cols)),
                "icon": "text"
            }

        # Card 4: Date range or completeness
        if date_cols:
            date_col = date_cols[0]
            try:
                df[date_col] = pd.to_datetime(df[date_col])
                date_range_days = (df[date_col].max() - df[date_col].min()).days

                if date_range_days > 365:
                    value = f"{date_range_days/365:.1f} years"
                elif date_range_days > 30:
                    value = f"{date_range_days/30:.0f} months"
                else:
                    value = f"{date_range_days} days"

                stats["date_range"] = {
                    "label": "Date Range",
                    "value": value,
                    "icon": "calendar"
                }
            except:
                stats["completeness"] = self._calculate_completeness(df)
        else:
            # No date columns, show data completeness
            stats["completeness"] = self._calculate_completeness(df)

        return stats

    def _find_value_column(self, df: pd.DataFrame, numeric_cols: List[str]) -> str:
        """
        Find the most likely "value" column (sales, revenue, amount, price, etc.).

        Args:
            df: DataFrame
            numeric_cols: List of numeric column names

        Returns:
            Column name or None
        """
        # Keywords that indicate value columns
        value_keywords = [
            'sales', 'revenue', 'amount', 'price', 'total', 'cost',
            'value', 'payment', 'income', 'profit', 'spend', 'purchase'
        ]

        # Check for keyword matches
        for col in numeric_cols:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in value_keywords):
                return col

        # If no keyword match, return column with highest sum (likely the value)
        if numeric_cols:
            sums = {col: df[col].sum() for col in numeric_cols}
            return max(sums, key=sums.get)

        return None

    def _generate_label_from_column(self, column_name: str) -> str:
        """
        Generate a friendly label from column name.

        Args:
            column_name: Original column name

        Returns:
            Friendly label
        """
        # Remove underscores, capitalize words
        label = column_name.replace('_', ' ').replace('-', ' ')
        label = ' '.join(word.capitalize() for word in label.split())

        # Remove common suffixes
        label = label.replace(' Id', '').replace(' Name', '')

        # Pluralize if needed
        if not label.endswith('s') and not any(x in label.lower() for x in ['count', 'total', 'amount', 'sum']):
            label = label + 's'

        return label

    def _calculate_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate data completeness percentage.

        Args:
            df: DataFrame

        Returns:
            Completeness stat card
        """
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        completeness_pct = (non_null_cells / total_cells * 100) if total_cells > 0 else 0

        return {
            "label": "Data Complete",
            "value": f"{completeness_pct:.1f}%",
            "icon": "check-circle"
        }

    def _get_fallback_stats(self) -> Dict[str, Any]:
        """
        Return basic fallback stats when analysis fails.

        Returns:
            Dictionary with fallback stats
        """
        return {
            "total_rows": {
                "label": "Records",
                "value": "-",
                "icon": "database"
            },
            "total_columns": {
                "label": "Columns",
                "value": "-",
                "icon": "columns"
            },
            "status": {
                "label": "Status",
                "value": "Ready",
                "icon": "activity"
            },
            "info": {
                "label": "Info",
                "value": "Upload CSV",
                "icon": "info"
            }
        }


if __name__ == "__main__":
    # Test the stats generator
    logging.basicConfig(level=logging.INFO)

    generator = StatsGenerator()

    # Test with metadata
    stats = generator.generate_stats_from_metadata(
        column_names=["product_name", "sales", "region", "date", "quantity"],
        row_count=1000
    )

    print("\n📊 Generated Stats:")
    for key, stat in stats.items():
        print(f"  {stat['label']}: {stat['value']}")
