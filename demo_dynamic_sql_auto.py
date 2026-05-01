"""
Auto-run demo of Dynamic SQL Generation (no user input required)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from demo_dynamic_sql import simulate_dynamic_query_generation

# Test questions
test_questions = [
    "What are the total sales by region?",
    "Show me the top 5 products by sales",
    "Show me sales trends over time",
    "What are the total sales in the East region?",
    "What's the average sale amount for Phones?",
]

print("\n🚀 DYNAMIC SQL GENERATION - AUTO DEMO")
print("=" * 80)
print("Demonstrating: NO hardcoded queries - all generated dynamically!")
print("=" * 80)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'─' * 80}")
    print(f"EXAMPLE {i}/{len(test_questions)}")
    simulate_dynamic_query_generation(question)

print("\n\n✅ DEMO COMPLETE!")
print("\n💡 Key Point: Each SQL query was generated DYNAMICALLY from the user's question")
print("   • No hardcoded SQL templates")
print("   • Intent detection → Field extraction → Query generation")
print("   • In production: LangChain + GPT-4 do this reasoning")