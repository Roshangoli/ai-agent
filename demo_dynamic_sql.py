"""
Demo: Dynamic SQL Generation from Natural Language

This demonstrates how the system generates SQL dynamically based on ANY user question,
with NO hardcoded queries.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from utils.database import execute_sql, get_schema
from dotenv import load_dotenv

load_dotenv()


def simulate_dynamic_query_generation(user_question: str):
    """
    Simulates how the system dynamically generates SQL based on user questions.

    In production, this is done by:
    1. LangChain analyzing the question
    2. Understanding the schema
    3. Generating appropriate SQL
    4. AutoGen agents validating and executing
    """

    print("\n" + "=" * 80)
    print(f"USER QUESTION: {user_question}")
    print("=" * 80)

    # Get schema context (this is what the AI sees)
    schema = get_schema()

    print("\n🧠 AI Understanding Process:")
    print("   1. Analyzing user intent...")

    # Simulate intent detection (in production, LangChain does this)
    intent = detect_intent(user_question)
    print(f"   2. Detected intent: {intent['type']}")
    print(f"   3. Required data: {intent['fields']}")
    print(f"   4. Filters: {intent['filters']}")

    # Generate SQL dynamically (in production, LangChain generates this)
    sql = generate_sql_from_intent(intent, schema)

    print(f"\n✨ DYNAMICALLY GENERATED SQL:")
    print(f"   {sql}")

    # Execute the dynamically generated query
    print(f"\n⚡ Executing query...")
    result = execute_sql(sql)

    if result.get("success"):
        print(f"✅ SUCCESS - Returned {result.get('row_count', 0)} rows\n")

        # Show results
        data = result.get("data", [])
        if data:
            print("📊 Results:")
            for idx, row in enumerate(data[:5], 1):
                print(f"   {idx}. {row}")
            if len(data) > 5:
                print(f"   ... and {len(data) - 5} more rows")
    else:
        print(f"❌ FAILED: {result.get('error')}")

    print("\n" + "=" * 80)


def detect_intent(question: str) -> dict:
    """
    Detect user intent from natural language question.
    In production, LangChain does this with LLM reasoning.
    """
    question_lower = question.lower()

    intent = {
        "type": "unknown",
        "fields": [],
        "filters": [],
        "aggregation": None,
        "grouping": [],
        "sorting": None,
        "limit": None
    }

    # Detect aggregation type
    if any(word in question_lower for word in ["total", "sum"]):
        intent["aggregation"] = "SUM"
        intent["type"] = "aggregation"
    elif any(word in question_lower for word in ["average", "avg", "mean"]):
        intent["aggregation"] = "AVG"
        intent["type"] = "aggregation"
    elif any(word in question_lower for word in ["count", "how many", "number of"]):
        intent["aggregation"] = "COUNT"
        intent["type"] = "aggregation"
    elif any(word in question_lower for word in ["highest", "top", "maximum", "best"]):
        intent["type"] = "ranking"
        intent["sorting"] = "DESC"
    elif any(word in question_lower for word in ["lowest", "bottom", "minimum", "worst"]):
        intent["type"] = "ranking"
        intent["sorting"] = "ASC"
    elif any(word in question_lower for word in ["trend", "over time", "monthly", "daily"]):
        intent["type"] = "time_series"
    else:
        intent["type"] = "listing"

    # Detect fields
    if "region" in question_lower:
        intent["fields"].append("region")
        intent["grouping"].append("region")
    if "product" in question_lower:
        intent["fields"].append("product")
        intent["grouping"].append("product")
    if "sales" in question_lower or "amount" in question_lower:
        intent["fields"].append("amount")
    if "date" in question_lower or "time" in question_lower or "month" in question_lower:
        intent["fields"].append("date")

    # Detect time filters
    if "last month" in question_lower:
        intent["filters"].append(("date", ">=", "DATE('now', '-1 month')"))
    elif "last quarter" in question_lower or "last 3 months" in question_lower:
        intent["filters"].append(("date", ">=", "DATE('now', '-3 months')"))
    elif "last year" in question_lower:
        intent["filters"].append(("date", ">=", "DATE('now', '-1 year')"))
    elif "this year" in question_lower:
        intent["filters"].append(("date", ">=", "DATE('now', 'start of year')"))

    # Detect region filters
    for region in ["north", "south", "east", "west"]:
        if region in question_lower:
            intent["filters"].append(("region", "=", f"'{region.capitalize()}'"))

    # Detect product filters
    for product in ["phone", "tablet", "laptop"]:
        if product in question_lower:
            intent["filters"].append(("product", "=", f"'{product.capitalize()}'"))

    # Detect limits
    if "top 5" in question_lower or "first 5" in question_lower:
        intent["limit"] = 5
    elif "top 10" in question_lower or "first 10" in question_lower:
        intent["limit"] = 10

    return intent


def generate_sql_from_intent(intent: dict, schema: dict) -> str:
    """
    Generate SQL query based on detected intent.
    In production, LangChain does this with more sophistication.
    """

    # Build SELECT clause
    if intent["type"] == "aggregation":
        if intent["grouping"]:
            select_parts = intent["grouping"].copy()
            if "amount" in intent["fields"]:
                agg = intent["aggregation"]
                select_parts.append(f"{agg}(amount) as total_sales")
        else:
            select_parts = [f"{intent['aggregation']}(amount) as total"]
    elif intent["type"] == "time_series":
        select_parts = ["strftime('%Y-%m', date) as month"]
        if "amount" in intent["fields"]:
            select_parts.append("SUM(amount) as total_sales")
    elif intent["type"] == "ranking":
        select_parts = intent["fields"].copy()
        if "amount" in select_parts:
            select_parts.remove("amount")
            select_parts.append("SUM(amount) as total_sales")
        if not intent["grouping"]:
            intent["grouping"] = [f for f in select_parts if f in ["region", "product"]]
    else:
        # Listing
        select_parts = ["*"] if not intent["fields"] else intent["fields"]

    select_clause = "SELECT " + ", ".join(select_parts)

    # Build FROM clause
    from_clause = "FROM sales"

    # Build WHERE clause
    where_clause = ""
    if intent["filters"]:
        conditions = []
        for field, op, value in intent["filters"]:
            conditions.append(f"{field} {op} {value}")
        where_clause = "WHERE " + " AND ".join(conditions)

    # Build GROUP BY clause
    group_by_clause = ""
    if intent["grouping"]:
        group_by_clause = "GROUP BY " + ", ".join(intent["grouping"])

    # Build ORDER BY clause
    order_by_clause = ""
    if intent["type"] == "time_series":
        order_by_clause = "ORDER BY month ASC"
    elif intent["sorting"]:
        if intent["type"] == "ranking":
            order_by_clause = f"ORDER BY total_sales {intent['sorting']}"
        elif intent["fields"]:
            order_by_clause = f"ORDER BY {intent['fields'][0]} {intent['sorting']}"

    # Build LIMIT clause
    limit_clause = ""
    if intent["limit"]:
        limit_clause = f"LIMIT {intent['limit']}"
    elif intent["type"] == "listing":
        limit_clause = "LIMIT 10"  # Default limit for listings

    # Combine all clauses
    query_parts = [
        select_clause,
        from_clause,
        where_clause,
        group_by_clause,
        order_by_clause,
        limit_clause
    ]

    sql = " ".join(part for part in query_parts if part) + ";"

    return sql


def main():
    """
    Demo: Show dynamic SQL generation for various user questions.
    NO hardcoded queries - everything generated on the fly!
    """

    print("\n" + "🚀" * 40)
    print("DYNAMIC SQL GENERATION DEMO")
    print("NO Hardcoded Queries - All Generated from User Intent!")
    print("🚀" * 40)

    # Test with various user questions
    test_questions = [
        # Aggregation queries
        "What are the total sales by region?",
        "Show me the sum of sales for each product",
        "What's the average sale amount per region?",

        # Ranking queries
        "Which region has the highest sales?",
        "Show me the top 5 products by sales",
        "What product has the lowest sales?",

        # Time-based queries
        "Show me sales trends over time",
        "What are the sales for last quarter?",
        "Show monthly sales for this year",

        # Filtered queries
        "What are the total sales in the East region?",
        "Show me all Phone sales from last month",
        "What are the sales for Laptops?",

        # Complex queries
        "Show me the top 10 sales by region and product",
        "What's the average sale amount for Phones in the North region?",
    ]

    print(f"\n📝 Testing {len(test_questions)} different user questions...")
    print("Each question will DYNAMICALLY generate unique SQL\n")

    for i, question in enumerate(test_questions, 1):
        print(f"\n\n{'#' * 80}")
        print(f"EXAMPLE {i}/{len(test_questions)}")
        print(f"{'#' * 80}")

        simulate_dynamic_query_generation(question)

        if i < len(test_questions):
            input("\nPress Enter to see next example...")

    print("\n\n" + "=" * 80)
    print("✅ DEMO COMPLETE!")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   • NO hardcoded SQL queries in the system")
    print("   • Each user question generates unique SQL dynamically")
    print("   • System understands intent, entities, filters, and aggregations")
    print("   • In production: LangChain + AutoGen handle this with LLMs")
    print("   • This demo shows the logic - actual system uses AI reasoning")
    print("\n🎯 This is exactly what happens when you use:")
    print("   python test_langchain_integration.py")
    print("   streamlit run ui/streamlit_app.py")


if __name__ == "__main__":
    main()