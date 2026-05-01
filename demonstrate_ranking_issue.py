"""
Demonstrate the Ranking Query Issue
Shows the WRONG vs RIGHT SQL for "Top N per Group" queries
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.database import execute_sql
import pandas as pd

print("=" * 80)
print("DEMONSTRATING: Top N Per Group Query Issue")
print("=" * 80)

print("\n📊 Question: 'Show top 3 products by sales in EACH region'")
print("\n" + "-" * 80)

# Show what data we have
print("\n1️⃣  First, let's see ALL product sales by region:")
print("-" * 80)

all_data = execute_sql("""
SELECT
    region,
    product,
    SUM(amount) AS total_sales,
    COUNT(*) as transactions
FROM sales
GROUP BY region, product
ORDER BY region, total_sales DESC
""")

if all_data.get("success"):
    df = pd.DataFrame(all_data['data'])
    print(df.to_string(index=False))
    print(f"\n   Total combinations: {len(df)} (4 regions × 3 products)")

# Show the WRONG query (what the current system generates)
print("\n\n2️⃣  CURRENT SYSTEM - What it generates:")
print("-" * 80)
print("""
❌ WRONG SQL:
SELECT
    region,
    product,
    SUM(amount) AS total_sales
FROM sales
GROUP BY region, product
ORDER BY region, total_sales DESC
LIMIT 3;
""")

wrong_result = execute_sql("""
SELECT
    region,
    product,
    SUM(amount) AS total_sales
FROM sales
GROUP BY region, product
ORDER BY region, total_sales DESC
LIMIT 3
""")

print("\n📉 Result from WRONG query:")
if wrong_result.get("success"):
    df_wrong = pd.DataFrame(wrong_result['data'])
    print(df_wrong.to_string(index=False))
    print(f"\n   ❌ Only {len(df_wrong)} rows returned!")
    print("   ❌ Missing products from South, West, and North regions!")
    print("   ❌ This is INCORRECT - user wanted top 3 in EACH region!")

# Show the RIGHT query (optimized with window function)
print("\n\n3️⃣  OPTIMIZED VERSION - What it SHOULD generate:")
print("-" * 80)
print("""
✅ CORRECT SQL (Using Window Function):
WITH RankedProducts AS (
    SELECT
        region,
        product,
        SUM(amount) AS total_sales,
        ROW_NUMBER() OVER (
            PARTITION BY region
            ORDER BY SUM(amount) DESC
        ) AS rank
    FROM sales
    GROUP BY region, product
)
SELECT region, product, total_sales, rank
FROM RankedProducts
WHERE rank <= 3
ORDER BY region, rank;
""")

# Note: SQLite supports window functions since version 3.25.0 (2018)
try:
    right_result = execute_sql("""
    WITH RankedProducts AS (
        SELECT
            region,
            product,
            SUM(amount) AS total_sales,
            ROW_NUMBER() OVER (
                PARTITION BY region
                ORDER BY SUM(amount) DESC
            ) AS rank
        FROM sales
        GROUP BY region, product
    )
    SELECT region, product, total_sales, rank
    FROM RankedProducts
    WHERE rank <= 3
    ORDER BY region, rank
    """)

    print("\n📈 Result from CORRECT query:")
    if right_result.get("success"):
        df_right = pd.DataFrame(right_result['data'])
        print(df_right.to_string(index=False))
        print(f"\n   ✅ {len(df_right)} rows returned (top 3 for each of 4 regions)")
        print("   ✅ Each region gets exactly 3 products")
        print("   ✅ This is CORRECT!")
    else:
        print(f"   ❌ Query failed: {right_result.get('error')}")
        print("   (Your SQLite version might not support window functions)")

        # Alternative approach without window functions
        print("\n   📝 Alternative without window functions:")
        print("""
        WITH RegionalSales AS (
            SELECT region, product, SUM(amount) AS total_sales
            FROM sales
            GROUP BY region, product
        )
        SELECT r1.region, r1.product, r1.total_sales
        FROM RegionalSales r1
        WHERE (
            SELECT COUNT(*)
            FROM RegionalSales r2
            WHERE r2.region = r1.region
              AND r2.total_sales >= r1.total_sales
        ) <= 3
        ORDER BY r1.region, r1.total_sales DESC;
        """)

except Exception as e:
    print(f"\n   ❌ Error: {e}")
    print("   Your SQLite version might be too old for window functions")

# Summary
print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
🔴 CRITICAL ISSUE FOUND:

   Query Type: "Top N per Group" (e.g., "top 3 products in each region")

   What Happens:
   - User asks: "Show top 3 products by sales in each region"
   - System generates: Simple LIMIT 3 query
   - Result: Only 3 products total (from 1 region)
   - Expected: 12 products (3 from each of 4 regions)

   Impact:
   ❌ Users get WRONG results
   ❌ Business decisions based on incomplete data
   ❌ User thinks other regions have no data

   Solution:
   ✅ Use window functions (ROW_NUMBER with PARTITION BY)
   ✅ Or use CTEs with correlated subqueries
   ✅ Teach the SQL agent these patterns

   Frequency: HIGH
   - Very common in analytics
   - "Top N customers per segment"
   - "Best performing products per category"
   - "Highest sales per region"
   - "Top employees per department"

💡 This is why "Complex queries need optimization" - not just speed,
   but CORRECTNESS!
""")

print("\n" + "=" * 80)
print("Next step: Update the SQL agent to handle window functions")
print("=" * 80)
