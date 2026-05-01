import os
import sqlite3
import random
from datetime import datetime as dt, timedelta
from typing import Dict, Any
import pandas as pd
from pathlib import Path

# Get project root directory (2 levels up from utils/)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DB_PATH = PROJECT_ROOT / "data" / "sample_data.db"

def create_connection(db_path: str = None) -> sqlite3.Connection:
    try:
        # Use custom path if provided, otherwise use default
        if db_path is None:
            db_path = str(DB_PATH)
        print(f"🔄 Connecting to database at: {db_path}")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        return conn
    except sqlite3.Error as e:
        print(f"❌ Database connection error: {e}")
        return None

def execute_sql(query: str, db_path: str = None) -> Dict[str, Any]:
    """Execute SQL query with enhanced safety and validation

    Args:
        query: SQL query to execute
        db_path: Optional custom database path (uses default if not provided)
    """
    conn = None
    try:
        query = query.strip()
        print(f"🔍 Executing query: {query}")  # Debug logging

        # Security checks - allow SELECT and WITH (for CTEs)
        query_upper = query.upper()
        is_read_only = query_upper.startswith("SELECT") or query_upper.startswith("WITH")

        if not is_read_only:
            return {
                "error": f"Security Error: Only SELECT/WITH queries allowed. Query was: {query[:100]}...",
                "success": False,
                "query": query
            }

        # Add LIMIT if missing
        if "LIMIT" not in query.upper() and ";" in query:
            query = query.rstrip(";") + " LIMIT 100;"

        conn = create_connection(db_path)
        if not conn:
            return {
                "error": "Database connection failed. Please check if the database file exists and has proper permissions.",
                "success": False,
                "query": query
            }

        cur = conn.cursor()
        print(f"🔄 Executing SQL: {query}")  # Debug logging
        cur.execute(query)
        
        results = cur.fetchall()
        if not results:
            return {
                "error": "No results returned from query",
                "success": False,
                "query": query
            }
            
        return {
            "columns": [desc[0] for desc in cur.description],
            "data": [dict(row) for row in results],
            "query": query,
            "success": True,
            "timestamp": dt.now().isoformat(),
            "row_count": len(results)
        }
            
    except sqlite3.Error as e:
        error_msg = str(e)
        if "no such table" in error_msg.lower():
            return {
                "error": f"Table not found: {error_msg}",
                "query": query,
                "success": False
            }
        elif "syntax error" in error_msg.lower():
            return {
                "error": f"SQL Syntax Error: {error_msg}",
                "query": query,
                "success": False
            }
        else:
            return {
                "error": f"SQL Error: {error_msg}",
                "query": query,
                "success": False
            }
    except Exception as e:
        return {
            "error": f"Unexpected Error: {str(e)}",  # More specific error message
            "query": query,
            "success": False
        }
    finally:
        if conn:
            conn.close()

def get_schema() -> Dict[str, Any]:
    """Return schema optimized for LLM understanding"""
    return {
        "tables": {
            "sales": {
                "description": "Contains daily sales transactions with time-based data",
                "columns": {
                    "date": {
                        "type": "TEXT",
                        "format": "YYYY-MM-DD",
                        "time_functions": [
                            "DATE('now', '-3 months')",
                            "DATE('now', 'start of month')",
                            "strftime('%Y-%m', 'now')",
                            "strftime('%Y', 'now')"
                        ],
                        "description": "Transaction date (UTC)"
                    },
                    "region": {
                        "type": "TEXT",
                        "values": ["North", "South", "East", "West"],
                        "indexed": True
                    },
                    "amount": {
                        "type": "REAL",
                        "aggregations": ["SUM", "AVG", "COUNT"],
                        "description": "USD value"
                    }
                },
                "indexes": [
                    "CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)",
                    "CREATE INDEX IF NOT EXISTS idx_sales_region ON sales(region)"
                ]
            }
        },
        "query_examples": [
            {
                "description": "Last 3 months sales by region",
                "sql": """SELECT 
    region, 
    SUM(amount) AS total_sales,
    strftime('%Y-%m', date) AS month
FROM sales
WHERE date >= DATE('now', '-3 months')
GROUP BY region, month
ORDER BY total_sales DESC;"""
            },
            {
                "description": "Current month sales by product",
                "sql": """SELECT
    product,
    SUM(amount) AS total_sales
FROM sales
WHERE date BETWEEN DATE('now', 'start of month') AND DATE('now')
GROUP BY product;"""
            }
        ]
    }

def initialize_sample_data() -> None:
    """Generate realistic sales data for the last 6 months"""
    conn = create_connection()
    if not conn:
        return

    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS sales")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL,
                region TEXT NOT NULL,
                product TEXT NOT NULL,
                amount REAL NOT NULL
            )
        """)

        # Generate data for the last 180 days
        regions = ["North", "South", "East", "West"]
        products = ["Phone", "Tablet", "Laptop"]
        base_date = dt.now() - timedelta(days=180)
        
        sample_data = []
        for i in range(300):
            # 70% of data concentrated in the last 90 days
            days_offset = random.randint(0, 90 if random.random() < 0.7 else 180)
            transaction_date = (base_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
            
            region = random.choice(regions)
            product_choice = random.choices(
                products, weights=[0.5, 0.3, 0.2], k=1
            )[0]
            
            base_price = {"Phone": 800, "Tablet": 500, "Laptop": 1200}[product_choice]
            amount = base_price * random.uniform(0.8, 1.2)
            
            sample_data.append((
                i + 1,
                transaction_date,
                region,
                product_choice,
                round(amount, 2)
            ))

        cur.executemany(
            "INSERT INTO sales (id, date, region, product, amount) VALUES (?, ?, ?, ?, ?)",
            sample_data
        )
        conn.commit()
        print(f"✅ Initialized {len(sample_data)} sales records from {base_date.date()} to {dt.now().date()}")
        
    except sqlite3.Error as e:
        print(f"❌ Data initialization failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    initialize_sample_data()
    
    test_query = """SELECT 
        region AS sales_region,
        SUM(amount) AS total_sales
    FROM sales
    WHERE date >= DATE('now', '-3 months')
    GROUP BY sales_region
    ORDER BY total_sales DESC;"""
    
    results = execute_sql(test_query)
    
    print("\nTEST RESULTS:")
    print(f"Query: {test_query}")
    
    if results.get('success'):
        print(f"Returned {len(results['data'])} rows")
        print(pd.DataFrame(results["data"]))
    else:
        print(f"❌ Query failed: {results.get('error', 'Unknown error')}")
        print(f"Full response: {results}")