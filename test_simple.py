"""
Simple test to verify the system works without LangChain dependencies.
Tests the existing AutoGen + database integration.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from utils.database import execute_sql, initialize_sample_data, get_schema
from dotenv import load_dotenv

load_dotenv()


def test_database():
    """Test database connection and queries."""
    print("=" * 70)
    print("🧪 TESTING DATABASE CONNECTION")
    print("=" * 70)

    # Ensure database exists
    db_path = "data/sample_data.db"
    if not os.path.exists(db_path):
        print("\n📦 Initializing sample database...")
        initialize_sample_data()
    else:
        print(f"✅ Database found at: {db_path}")

    # Test queries
    test_queries = [
        ("Total sales by region", """
            SELECT region, SUM(amount) as total_sales, COUNT(*) as num_transactions
            FROM sales
            GROUP BY region
            ORDER BY total_sales DESC;
        """),
        ("Top 5 products by sales", """
            SELECT product, SUM(amount) as total_sales
            FROM sales
            GROUP BY product
            ORDER BY total_sales DESC
            LIMIT 5;
        """),
        ("Recent sales (last 10)", """
            SELECT date, region, product, amount
            FROM sales
            ORDER BY date DESC
            LIMIT 10;
        """),
    ]

    print("\n📊 Running test queries...\n")

    for title, query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Test: {title}")
        print('─' * 70)
        print(f"Query: {query.strip()[:100]}...")

        result = execute_sql(query)

        if result.get("success"):
            print(f"✅ SUCCESS - Returned {result.get('row_count', 0)} rows")

            # Show sample data
            data = result.get("data", [])
            if data:
                print("\n📋 Sample Results:")
                for idx, row in enumerate(data[:3], 1):
                    print(f"   {idx}. {row}")
        else:
            print(f"❌ FAILED: {result.get('error')}")


def test_schema():
    """Test schema retrieval."""
    print("\n\n" + "=" * 70)
    print("🧪 TESTING SCHEMA RETRIEVAL")
    print("=" * 70)

    schema = get_schema()

    print("\n📊 Database Schema:")
    for table_name, table_info in schema.get("tables", {}).items():
        print(f"\n  Table: {table_name}")
        print(f"  Description: {table_info.get('description', 'N/A')}")
        print("  Columns:")
        for col_name, col_info in table_info.get("columns", {}).items():
            print(f"    - {col_name}: {col_info.get('type', 'UNKNOWN')}")


def test_autogen_agents():
    """Test AutoGen agents initialization."""
    print("\n\n" + "=" * 70)
    print("🧪 TESTING AUTOGEN AGENTS")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  OPENAI_API_KEY not found - Skipping AutoGen test")
        print("   Set OPENAI_API_KEY in .env to test full agent workflow")
        return

    try:
        from agents.analytics_agents import AnalyticsAgents

        print("\n🤖 Initializing AutoGen agent team (without LangChain)...")
        agents = AnalyticsAgents(use_langchain=False)
        print("✅ AutoGen agents initialized successfully!")

        print("\n📝 Agent Team:")
        print("   • Coordinator - Project manager")
        print("   • Data_Analyst - SQL query writer")
        print("   • Executor - Query executor")
        print("   • Validator - Quality assurance")
        print("   • Insight_Agent - Chart & narrative generator")

    except Exception as e:
        print(f"❌ AutoGen initialization failed: {e}")
        import traceback
        traceback.print_exc()


def test_langchain_optional():
    """Test LangChain integration if available."""
    print("\n\n" + "=" * 70)
    print("🧪 TESTING LANGCHAIN INTEGRATION (OPTIONAL)")
    print("=" * 70)

    try:
        from utils.langchain_sql import LangChainSQLGenerator

        print("\n🔗 Initializing LangChain SQL Generator...")
        generator = LangChainSQLGenerator()
        print("✅ LangChain available and working!")

        # Try a simple query
        question = "What are the total sales by region?"
        print(f"\n📝 Test Question: {question}")

        result = generator.generate_query(question)
        if result.get("success"):
            print(f"✅ Generated SQL:\n   {result['query']}")
        else:
            print(f"⚠️  Generation failed: {result.get('error')}")

    except ImportError as e:
        print(f"\n⚠️  LangChain not fully installed: {e}")
        print("   This is optional - system works without it")
        print("   To install: pip install langchain langchain-openai langchain-community")
    except Exception as e:
        print(f"\n⚠️  LangChain test failed: {e}")
        print("   System can still work with AutoGen only")


def main():
    """Run all tests."""
    print("\n🚀 ANALYTICS SYSTEM TEST SUITE")
    print("Testing core functionality\n")

    try:
        # Core tests
        test_database()
        test_schema()
        test_autogen_agents()

        # Optional test
        test_langchain_optional()

        print("\n\n" + "=" * 70)
        print("✅ CORE TESTS COMPLETED")
        print("=" * 70)

        print("\n💡 Next Steps:")
        print("   1. Run Streamlit app: streamlit run ui/streamlit_app.py")
        print("   2. Ask questions like: 'Show sales by region'")
        print("   3. View agent collaboration in real-time")

    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()