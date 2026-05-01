# Analytics agents with validator feedback loop.
# Adds a validator agent that checks SQL and provides feedback for iterative improvement.

import os
import sys
import json
import logging
import re
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import AutoGen/ag2 for AI agents
try:
    from ag2 import AssistantAgent
    logger = logging.getLogger(__name__)
    logger.info("Using ag2 (AutoGen 2.0)")
except ImportError:
    try:
        from autogen import AssistantAgent
        logger = logging.getLogger(__name__)
        logger.info("Using autogen (legacy)")
    except ImportError:
        raise ImportError("Please install ag2: pip install ag2[openai]")

from utils.database import execute_sql, get_schema
from utils.chart_generator import generate_chart

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analytics_agents_validator.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class AnalyticsAgentsWithValidator:
    """
    Enhanced multi-agent system with Validator feedback loop.

    This system uses 3 agents:
    1. SQL_Generator - Converts natural language to SQL
    2. SQL_Validator - Validates SQL and provides feedback
    3. Insight_Generator - Creates visualizations and narratives

    The Validator can send feedback to the SQL_Generator for iterative improvement.
    """

    def __init__(self, max_validation_attempts: int = 3):
        """
        Initialize the enhanced analytics agent system.

        Args:
            max_validation_attempts: Maximum iterations for SQL generation/validation
        """
        # Configuration for OpenAI GPT-4
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY")
            }],
            "temperature": 0,
            "timeout": 300
        }

        self.max_validation_attempts = max_validation_attempts
        self.validation_stats = {
            "total_queries": 0,
            "first_try_success": 0,
            "iterations_used": []
        }

        self._create_agents()

    def _create_agents(self):
        """Create the AI agents including the Validator."""
        # Get the current database schema for context
        db_schema = get_schema()
        schema_str = json.dumps(db_schema, indent=2)

        # Agent 1: SQL Generator
        self.sql_agent = AssistantAgent(
            name="SQL_Generator",
            llm_config=self.llm_config,
            system_message=f"""You're an expert SQL database engineer specializing in query optimization and advanced analytics.

Database Schema:
{schema_str}

When the user asks a question, write an optimized SQL query that answers it correctly.
Return ONLY the SQL in a ```sql code block. Nothing else.

If you receive feedback from the validator, use it to improve your next attempt.

CRITICAL RULES:
1. **Top N per Group Queries** - ALWAYS use window functions with PARTITION BY
   Example: "Top 3 products in each region"
   ```sql
   WITH RankedProducts AS (
       SELECT
           region,
           product,
           SUM(amount) AS total_sales,
           ROW_NUMBER() OVER (PARTITION BY region ORDER BY SUM(amount) DESC) AS rank
       FROM sales
       GROUP BY region, product
   )
   SELECT region, product, total_sales
   FROM RankedProducts
   WHERE rank <= 3
   ORDER BY region, rank;
   ```

2. **Multi-Step Logic** - Use CTEs (WITH clauses) for clarity
3. **Date Ranges** - Use adaptive date filtering (data-aware)
   Example: "WHERE date >= DATE((SELECT MAX(date) FROM sales), '-6 months')"

4. **Optimization**:
   - Select specific columns (avoid SELECT *)
   - Add LIMIT clauses for large result sets
   - Use WHERE before GROUP BY for filtering
   - Use HAVING only for post-aggregation filtering

5. **Window Functions** - Use for rankings, running totals, percentiles
   Always PARTITION BY the grouping column

Simple Example:
User: "What are the top 5 products by sales?"
```sql
SELECT product, SUM(amount) as total_sales
FROM sales
GROUP BY product
ORDER BY total_sales DESC
LIMIT 5;
```"""
        )

        # Agent 2: SQL Validator (NEW!)
        self.validator_agent = AssistantAgent(
            name="SQL_Validator",
            llm_config=self.llm_config,
            system_message=f"""You're a SQL validation and optimization expert. Your job is to validate SQL queries for correctness AND optimization.

Database Schema:
{schema_str}

For each query, check:
1. **Syntax Correctness** - Is the SQL syntactically valid?
2. **Schema Compliance** - Do all tables and columns exist?
3. **Logic Correctness** - Does the query match the user's intent?
4. **OPTIMIZATION CHECKS** (CRITICAL):
   a) **Top N per Group** - If user asks for "top N in each X", query MUST use window functions with PARTITION BY
      ❌ INVALID: "SELECT ... GROUP BY region, product ORDER BY sales DESC LIMIT 3"
      ✅ VALID: "WITH Ranked AS (SELECT ..., ROW_NUMBER() OVER (PARTITION BY region ORDER BY sales DESC) AS rank ...) WHERE rank <= 3"

   b) **Date Filtering** - MUST use adaptive date ranges (not 'now' for historical data)
      ❌ INVALID: "WHERE date >= DATE('now', '-6 months')" (fails on historical data)
      ✅ VALID: "WHERE date >= DATE((SELECT MAX(date) FROM sales), '-6 months')"

   c) **Performance Anti-Patterns**:
      - ❌ SELECT * without LIMIT on potentially large results
      - ❌ Missing LIMIT when query could return 1000+ rows
      - ❌ Subqueries in WHERE when JOIN would be faster

   d) **Multi-Step Logic** - Complex queries should use CTEs for clarity
5. **Security** - Is it a safe read-only query?

Response Format:
If valid AND optimized, respond EXACTLY: "VALID: Query is correct and ready to execute."

If invalid OR needs optimization, respond with:
"INVALID: [Specific feedback]

Issues found:
1. [First issue with suggestion]
2. [Second issue with suggestion]

Suggested improvements:
- [Concrete suggestion with example]"

CRITICAL Examples:

Example 1 - Wrong Logic (Top N per Group):
Query: SELECT region, product, SUM(amount) AS sales FROM sales GROUP BY region, product ORDER BY sales DESC LIMIT 3;
User Question: "Top 3 products in each region"

Response:
"INVALID: Incorrect logic for 'Top N per Group' query

Issues found:
1. LIMIT 3 returns only 3 products TOTAL, not 3 per region (query will miss most regions)
2. User wants top 3 in EACH region, but query returns global top 3

Suggested improvements:
- Use window function with PARTITION BY:
  WITH Ranked AS (
      SELECT region, product, SUM(amount) AS sales,
             ROW_NUMBER() OVER (PARTITION BY region ORDER BY SUM(amount) DESC) AS rank
      FROM sales GROUP BY region, product
  )
  SELECT region, product, sales FROM Ranked WHERE rank <= 3;"

Example 2 - Date Handling:
Query: SELECT * FROM sales WHERE date >= DATE('now', '-6 months');
User Question: "Last 6 months of sales"

Response:
"INVALID: Date filtering will fail on historical data

Issues found:
1. Using 'now' assumes data is current (will return empty if data is from 2024-2025 and today is 2026)
2. SELECT * retrieves all columns (inefficient)

Suggested improvements:
- Use adaptive date range: WHERE date >= DATE((SELECT MAX(date) FROM sales), '-6 months')
- Select specific columns needed"

Be strict about optimization. Incorrect logic = INVALID."""
        )

        # Agent 3: Insight Generator
        self.insight_agent = AssistantAgent(
            name="Insight_Generator",
            llm_config=self.llm_config,
            system_message="""You're a data storyteller. Given query results, you create:
1. A chart specification (Altair format)
2. A narrative explaining the insights

Return your response as JSON with this exact structure:
{
  "chart": {
    "mark": "bar",  // or "line", "point", "area"
    "encoding": {
      "x": {"field": "column_name", "type": "nominal/quantitative/temporal"},
      "y": {"field": "column_name", "type": "quantitative"}
    }
  },
  "narrative": "Your insight summary here. Be concise and highlight key findings."
}

Important:
- Use actual column names from the data provided
- Choose appropriate mark types (bar for comparisons, line for trends, etc.)
- Keep narrative under 3 sentences
- Return ONLY valid JSON, nothing else"""
        )

    def _extract_sql(self, response: str) -> str:
        """Extract SQL query from agent response."""
        sql_match = re.search(r'```sql\n(.*?)\n```', response, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()
        return None

    def _validate_sql(self, sql_query: str, user_question: str, attempt: int) -> dict:
        """
        Validate SQL query using the Validator agent.

        Returns:
            {
                "valid": bool,
                "feedback": str,
                "issues": list
            }
        """
        logger.info(f" Validating SQL (Attempt {attempt})...")

        validation_prompt = f"""
User Question: "{user_question}"

SQL Query to Validate:
```sql
{sql_query}
```

Please validate this query against the database schema and user's intent.
"""

        try:
            validation_response = self.validator_agent.generate_reply(
                messages=[{"role": "user", "content": validation_prompt}]
            )

            logger.info(f" Validator response: {validation_response[:200]}...")

            # Check if valid
            is_valid = "VALID:" in validation_response.upper()

            # Extract issues if invalid
            issues = []
            if not is_valid:
                # Try to extract numbered issues
                issue_matches = re.findall(r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|$)', validation_response, re.DOTALL)
                issues = [issue.strip() for issue in issue_matches]

            return {
                "valid": is_valid,
                "feedback": validation_response,
                "issues": issues
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "valid": True,  # Default to valid if validator fails
                "feedback": "Validator error, proceeding with query",
                "issues": []
            }

    def run_with_validation(self, user_question: str) -> dict:
        """
        Main workflow with validator feedback loop.

        Steps:
        1. Generate SQL
        2. Validate SQL
        3. If invalid, send feedback and regenerate (up to max_attempts)
        4. Execute valid SQL
        5. Generate insights
        """
        logger.info(f"User question: {user_question}")
        logger.info(f"Using validation mode (max {self.max_validation_attempts} attempts)")

        self.validation_stats["total_queries"] += 1

        sql_query = None
        validation_result = None
        conversation_history = [{"role": "user", "content": user_question}]

        # Validation loop
        for attempt in range(1, self.max_validation_attempts + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ATTEMPT {attempt}/{self.max_validation_attempts}")
            logger.info(f"{'='*60}")

            # Step 1: Generate SQL
            logger.info(" Generating SQL...")
            sql_response = self.sql_agent.generate_reply(messages=conversation_history)

            sql_query = self._extract_sql(sql_response)

            if not sql_query:
                logger.error(f"Could not extract SQL from response")
                if attempt == self.max_validation_attempts:
                    return {
                        "chart": None,
                        "narrative": "Could not generate SQL query. Please rephrase your question.",
                        "attempts": attempt,
                        "validation_failed": True
                    }
                continue

            logger.info(f"Generated SQL:\n{sql_query}")

            # Step 2: Validate SQL
            validation_result = self._validate_sql(sql_query, user_question, attempt)

            if validation_result["valid"]:
                # Success!
                logger.info("Validation PASSED!")
                if attempt == 1:
                    self.validation_stats["first_try_success"] += 1
                self.validation_stats["iterations_used"].append(attempt)
                break
            else:
                # Invalid - log feedback
                logger.warning(f"Validation FAILED!")
                logger.warning(f"Feedback: {validation_result['feedback'][:200]}...")

                if validation_result["issues"]:
                    logger.warning(f"Issues found:")
                    for i, issue in enumerate(validation_result["issues"], 1):
                        logger.warning(f"  {i}. {issue}")

                # If last attempt, give up
                if attempt == self.max_validation_attempts:
                    logger.error(f"Max attempts reached. Using last generated SQL anyway.")
                    self.validation_stats["iterations_used"].append(attempt)
                    break

                # Send feedback to SQL generator for next attempt
                feedback_message = f"""
Your previous SQL query had issues according to the validator:

{validation_result['feedback']}

User's original question: "{user_question}"

Please generate a corrected SQL query that addresses these issues.
"""
                conversation_history.append({
                    "role": "assistant",
                    "content": sql_response
                })
                conversation_history.append({
                    "role": "user",
                    "content": feedback_message
                })

                logger.info(" Sending feedback to SQL_Generator for retry...")

        # Step 3: Execute the query (valid or last attempt)
        logger.info(f"\nExecuting final SQL query...")

        try:
            results = execute_sql(sql_query)

            if not results.get("success"):
                error_msg = results.get("error", "Unknown error")
                logger.error(f"Query execution failed: {error_msg}")
                return {
                    "chart": None,
                    "narrative": f"Query execution failed: {error_msg}",
                    "sql_query": sql_query,
                    "attempts": attempt,
                    "validation_result": validation_result
                }

            row_count = len(results.get("data", []))
            logger.info(f"Query returned {row_count} rows")

            if row_count == 0:
                return {
                    "chart": None,
                    "narrative": "The query executed successfully but returned no data.",
                    "sql_query": sql_query,
                    "attempts": attempt
                }

            # Step 4: Generate insights
            logger.info(f"Generating insights...")

            data_preview = results["data"][:5]
            columns = results["columns"]

            insight_prompt = f"""Analyze this query result and create a visualization:

Question: {user_question}
SQL: {sql_query}

Columns: {', '.join(columns)}
Sample Data (first {len(data_preview)} rows):
{json.dumps(data_preview, indent=2)}

Total rows: {row_count}

Create a chart spec and narrative. Return JSON only."""

            insight_response = self.insight_agent.generate_reply(
                messages=[{"role": "user", "content": insight_prompt}]
            )

            # Extract JSON from response
            insights = None
            if '```json' in insight_response.lower():
                json_match = re.search(r'```json\n(.*?)\n```', insight_response, re.DOTALL | re.IGNORECASE)
                if json_match:
                    insights = json.loads(json_match.group(1))
            elif insight_response.strip().startswith('{'):
                insights = json.loads(insight_response.strip())

            if not insights:
                logger.warning("Could not extract insights JSON, using fallback")
                first_col = columns[0]
                second_col = columns[1] if len(columns) > 1 else columns[0]

                insights = {
                    "chart": {
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": first_col, "type": "nominal"},
                            "y": {"field": second_col, "type": "quantitative"}
                        }
                    },
                    "narrative": f"Analysis complete. Showing {row_count} results from your query."
                }

            # Step 5: Generate the actual chart
            logger.info(f" Creating visualization...")
            chart_spec = insights.get("chart", {})

            try:
                chart_type = chart_spec.get("mark", "bar")
                final_chart = generate_chart(
                    data=results,
                    chart_type=chart_type,
                    output_format='file'
                )
            except Exception as chart_error:
                logger.error(f"Chart generation failed: {chart_error}")
                final_chart = None

            logger.info(f"Analysis complete with validation!")

            return {
                "chart": final_chart,
                "narrative": insights.get("narrative", "Analysis complete."),
                "sql_query": sql_query,
                "row_count": row_count,
                "attempts": attempt,
                "validation_result": validation_result,
                "validated": validation_result["valid"] if validation_result else False
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {
                "chart": None,
                "narrative": "I had trouble formatting the insights. Please try again.",
                "attempts": attempt
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "chart": None,
                "narrative": f"An error occurred: {str(e)}",
                "attempts": attempt
            }

    def run(self, user_question: str, use_validation: bool = True):
        """
        Run the analytics system.

        Args:
            user_question: Natural language question
            use_validation: If True, use validator feedback loop

        Returns:
            Dictionary with results
        """
        if use_validation:
            return self.run_with_validation(user_question)
        else:
            # Direct mode without validation (faster)
            return self._run_direct(user_question)

    def _run_direct(self, user_question: str):
        """Direct mode without validation (for comparison)."""
        logger.info(f" User question: {user_question} (Direct mode)")

        # Just use SQL_Generator without validation
        sql_response = self.sql_agent.generate_reply(
            messages=[{"role": "user", "content": user_question}]
        )

        sql_query = self._extract_sql(sql_response)

        if not sql_query:
            return {
                "chart": None,
                "narrative": "Could not generate SQL query."
            }

        # Execute and generate insights (same as before)
        results = execute_sql(sql_query)

        if not results.get("success"):
            return {
                "chart": None,
                "narrative": f"Query failed: {results.get('error')}"
            }

        # Continue with insight generation...
        # (Rest of the code same as validation mode)
        return {"chart": None, "narrative": "Direct mode - implementation pending"}

    def get_validation_stats(self) -> dict:
        """Get statistics about validation performance."""
        if self.validation_stats["total_queries"] == 0:
            return {
                "total_queries": 0,
                "first_try_success_rate": 0,
                "avg_iterations": 0
            }

        total = self.validation_stats["total_queries"]
        first_try = self.validation_stats["first_try_success"]
        iterations = self.validation_stats["iterations_used"]

        return {
            "total_queries": total,
            "first_try_success": first_try,
            "first_try_success_rate": (first_try / total * 100) if total > 0 else 0,
            "avg_iterations": sum(iterations) / len(iterations) if iterations else 0,
            "max_iterations": max(iterations) if iterations else 0,
            "min_iterations": min(iterations) if iterations else 0,
            "iterations_distribution": {
                "1": iterations.count(1),
                "2": iterations.count(2),
                "3": iterations.count(3)
            }
        }


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*70)
    print(" TESTING VALIDATOR FEEDBACK LOOP")
    print("="*70 + "\n")

    agents = AnalyticsAgentsWithValidator(max_validation_attempts=3)

    # Test queries (including intentionally ambiguous/complex ones)
    test_queries = [
        "Show total sales by region",  # Simple - should work first try
        "Show recent sales data",  # Ambiguous - might need iteration
        "Compare sales growth between regions",  # Complex - might need iteration
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {query}")
        print('='*70)

        result = agents.run_with_validation(query)

        print(f"\n RESULT:")
        print(f"  Attempts: {result.get('attempts', 'N/A')}")
        print(f"  Validated: {result.get('validated', 'N/A')}")
        print(f"  SQL: {result.get('sql_query', 'N/A')[:100]}...")
        print(f"  Narrative: {result.get('narrative', 'N/A')[:100]}...")

    # Print overall stats
    print(f"\n{'='*70}")
    print(" VALIDATION STATISTICS")
    print('='*70)
    stats = agents.get_validation_stats()
    print(json.dumps(stats, indent=2))
