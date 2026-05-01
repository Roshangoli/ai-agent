"""
CSV to SQLite Converter for Analytics Mode
Allows users to upload CSV and query it with natural language
"""

import pandas as pd
import sqlite3
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CSVToSQLite:
    """Convert uploaded CSV files to temporary SQLite databases for querying"""

    def __init__(self, temp_db_dir: str = "data/temp_databases"):
        """
        Initialize CSV to SQLite converter.

        Args:
            temp_db_dir: Directory to store temporary databases
        """
        self.temp_db_dir = Path(temp_db_dir)
        self.temp_db_dir.mkdir(parents=True, exist_ok=True)

    def convert_csv_to_sqlite(
        self,
        csv_path: str,
        table_name: str = "uploaded_data"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert CSV file to SQLite database.

        Args:
            csv_path: Path to CSV file
            table_name: Name for the table in SQLite

        Returns:
            Tuple of (database_path, metadata)
        """
        try:
            # Read CSV
            logger.info(f"📁 Loading CSV: {csv_path}")
            df = pd.read_csv(csv_path)

            # Clean column names (remove special chars, spaces)
            df.columns = [
                col.lower()
                .replace(' ', '_')
                .replace('-', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('/', '_')
                for col in df.columns
            ]

            # Create temporary database
            csv_filename = Path(csv_path).stem
            db_path = self.temp_db_dir / f"{csv_filename}.db"

            # Remove existing database if present
            if db_path.exists():
                db_path.unlink()

            # Write to SQLite
            logger.info(f"💾 Creating SQLite database: {db_path}")
            conn = sqlite3.connect(str(db_path))
            df.to_sql(table_name, conn, index=False, if_exists='replace')

            # Get schema information
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema = cursor.fetchall()

            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_rows = cursor.fetchall()

            conn.close()

            # Build metadata
            metadata = {
                "success": True,
                "db_path": str(db_path),
                "table_name": table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "schema": schema,
                "sample_rows": sample_rows[:3],
                "csv_filename": csv_filename
            }

            logger.info(f"✅ Database created: {len(df)} rows, {len(df.columns)} columns")

            return str(db_path), metadata

        except Exception as e:
            logger.error(f"❌ Failed to convert CSV to SQLite: {e}")
            return "", {
                "success": False,
                "error": str(e)
            }

    def get_schema_info(self, db_path: str, table_name: str = "uploaded_data") -> Dict[str, Any]:
        """
        Get schema information for a database.

        Args:
            db_path: Path to SQLite database
            table_name: Name of table to inspect

        Returns:
            Schema information dictionary
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()

            conn.close()

            return {
                "table_name": table_name,
                "columns": columns,
                "row_count": row_count,
                "sample_data": sample_data,
                "column_names": [col[1] for col in columns],
                "column_types": [col[2] for col in columns]
            }

        except Exception as e:
            logger.error(f"❌ Failed to get schema: {e}")
            return {"error": str(e)}

    def cleanup_old_databases(self, max_age_hours: int = 24):
        """
        Clean up temporary databases older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours before deletion
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        deleted_count = 0

        for db_file in self.temp_db_dir.glob("*.db"):
            # Skip sample_data.db (permanent database)
            if db_file.name == "sample_data.db":
                continue

            file_age = current_time - db_file.stat().st_mtime

            if file_age > max_age_seconds:
                db_file.unlink()
                deleted_count += 1
                logger.info(f"🗑️  Deleted old database: {db_file.name}")

        if deleted_count > 0:
            logger.info(f"✅ Cleaned up {deleted_count} old database(s)")


def convert_uploaded_file_to_db(
    uploaded_file,
    save_dir: str = "data/uploads"
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to convert Streamlit uploaded file to SQLite.

    Args:
        uploaded_file: Streamlit UploadedFile object
        save_dir: Directory to save uploaded files

    Returns:
        Tuple of (database_path, metadata)
    """
    # Save uploaded file to temp location
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert to SQLite
    converter = CSVToSQLite()
    db_path, metadata = converter.convert_csv_to_sqlite(str(file_path))

    return db_path, metadata


if __name__ == "__main__":
    # Test the converter
    converter = CSVToSQLite()

    # Test with sample data
    test_csv = "data/customer_churn_sample.csv"

    if os.path.exists(test_csv):
        db_path, metadata = converter.convert_csv_to_sqlite(test_csv)

        print(f"\n{'='*70}")
        print("CSV TO SQLITE CONVERTER TEST")
        print(f"{'='*70}")

        if metadata.get("success"):
            print(f"✅ Success!")
            print(f"   Database: {db_path}")
            print(f"   Table: {metadata['table_name']}")
            print(f"   Rows: {metadata['rows']:,}")
            print(f"   Columns: {metadata['columns']}")
            print(f"   Column Names: {', '.join(metadata['column_names'][:5])}...")

            # Test schema retrieval
            schema = converter.get_schema_info(db_path)
            print(f"\n📊 Schema Info:")
            print(f"   Total Rows: {schema['row_count']:,}")
            print(f"   Columns: {len(schema['columns'])}")

        else:
            print(f"❌ Failed: {metadata.get('error')}")

    else:
        print(f"Test file not found: {test_csv}")
