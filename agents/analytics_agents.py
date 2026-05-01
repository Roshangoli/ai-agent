import os
import json
import logging
import re
from dotenv import load_dotenv

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

# Import observability layers
from observability.tracer import get_tracer
from observability.prompt_version import get_prompt_registry
from observability.quality_scorer import (
    SQLQualityScorer,
    NarrativeQualityScorer,
    get_quality_registry
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analytics_agents.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class AnalyticsAgents:
    """
    Multi-agent system for natural language database analytics.
    Converts questions to SQL, executes queries, and generates insights.
    """

    def __init__(self, use_langchain: bool = False, custom_db_path: str = None, custom_table_name: str = None):
        """
        Initialize the analytics agent system.

        Args:
            use_langchain: Legacy parameter, kept for compatibility
            custom_db_path: Optional path to custom SQLite database (for CSV uploads)
            custom_table_name: Optional table name in custom database
        """
        # Initialize observability layers
        self.tracer = get_tracer()
        self.prompt_registry = get_prompt_registry()  # Layer 2: Prompt versioning
        self.quality_registry = get_quality_registry()  # Layer 3: Quality scoring

        # Configuration for OpenAI GPT-4
        self.llm_config = {
            "config_list": [{
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY")
            }],
            "temperature": 0,
            "timeout": 300
        }

        self.use_langchain = False  # Simplified: no LangChain needed

        # Database configuration (support custom uploads)
        if custom_db_path:
            self.db_path = custom_db_path
            self.table_name = custom_table_name or "uploaded_data"
            self.is_custom_db = True
            logger.info(f"📊 Using custom database: {custom_db_path}")
        else:
            self.db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_data.db')
            self.table_name = None  # Use all tables
            self.is_custom_db = False
            logger.info(f"📊 Using default database: {self.db_path}")

        # Initialize quality scorers (Layer 3)
        self.sql_scorer = SQLQualityScorer(self.db_path)
        self.narrative_scorer = NarrativeQualityScorer()

        self._create_agents()

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from text.
        Rule of thumb: ~0.75 tokens per word for English text.

        Args:
            text: Input text

        Returns:
            Estimated token count
        """
        # Simple estimation: count words and multiply by 0.75
        # Then add characters/4 for non-word tokens
        words = len(text.split())
        chars = len(text)
        return int(words * 0.75 + chars / 4)

    def _get_custom_schema(self):
        """Get schema for custom uploaded database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get table schema
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns = cursor.fetchall()

        # Get sample data
        cursor.execute(f"SELECT * FROM {self.table_name} LIMIT 5")
        sample_data = cursor.fetchall()

        conn.close()

        # Format schema for LLM
        schema_text = f"Table: {self.table_name}\nColumns:\n"
        for col in columns:
            schema_text += f"  - {col[1]} ({col[2]})\n"

        return {"tables": {self.table_name: {"columns": schema_text, "description": "Uploaded CSV data"}}}

    def _create_agents(self):
        """Create the AI agents that power the analytics system."""
        # Get the current database schema for context
        if self.is_custom_db:
            # For custom databases, get schema dynamically
            db_schema = self._get_custom_schema()
        else:
            db_schema = get_schema()

        # Define prompts (for version tracking)
        self.sql_prompt = f"""You're an expert SQL database engineer specializing in query optimization and advanced analytics.

Database Schema:
{db_schema}

When the user asks a question, write an optimized SQL query that answers it correctly.
Return ONLY the SQL in a ```sql code block. Nothing else.

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
   Example: "Products with >10 transactions, grouped by region"
   ```sql
   WITH ActiveProducts AS (
       SELECT product
       FROM sales
       GROUP BY product
       HAVING COUNT(*) > 10
   )
   SELECT
       region,
       product,
       SUM(amount) AS total_sales
   FROM sales
   WHERE product IN (SELECT product FROM ActiveProducts)
   GROUP BY region, product
   ORDER BY region, total_sales DESC;
   ```

3. **Date Ranges** - Use adaptive date filtering (data-aware, not 'now')
   Example: "Last 6 months of data"
   ```sql
   SELECT
       strftime('%Y-%m', date) AS month,
       SUM(amount) AS total_sales
   FROM sales
   WHERE date >= DATE((SELECT MAX(date) FROM sales), '-6 months')
   GROUP BY month
   ORDER BY month;
   ```

4. **Optimization Best Practices**:
   - Select specific columns (avoid SELECT *)
   - Add LIMIT clauses for large result sets (default: LIMIT 100)
   - Use WHERE before GROUP BY for filtering
   - Use HAVING only for post-aggregation filtering
   - Order results logically (dates ascending, amounts descending)

5. **Window Functions** - Use for:
   - Rankings: ROW_NUMBER(), RANK(), DENSE_RANK()
   - Running totals: SUM() OVER (ORDER BY ...)
   - Percentiles: NTILE(n) OVER (...)
   - Always PARTITION BY the grouping column

6. **Year-over-Year Comparisons** - Use CASE WHEN:
   ```sql
   SELECT
       region,
       SUM(CASE WHEN strftime('%Y', date) = '2025' THEN amount ELSE 0 END) AS sales_2025,
       SUM(CASE WHEN strftime('%Y', date) = '2024' THEN amount ELSE 0 END) AS sales_2024
   FROM sales
   GROUP BY region;
   ```

Simple Example:
User: "What are the top 5 products by sales?"
```sql
SELECT product, SUM(amount) as total_sales
FROM sales
GROUP BY product
ORDER BY total_sales DESC
LIMIT 5;
```"""

        # Register SQL Generator prompt (Layer 2: Prompt versioning)
        sql_version = self.prompt_registry.register_prompt(
            agent_name="SQL_Generator",
            prompt_text=self.sql_prompt,
            metadata={"purpose": "Convert natural language to optimized SQL"}
        )
        self.sql_prompt_hash = sql_version.short_hash

        # Create SQL Generator Agent
        self.sql_agent = AssistantAgent(
            name="SQL_Generator",
            llm_config=self.llm_config,
            system_message=self.sql_prompt
        )

        # Define Insight Generator prompt
        self.insight_prompt = """You're a data storyteller. Given query results, you create:
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

        # Register Insight Generator prompt (Layer 2: Prompt versioning)
        insight_version = self.prompt_registry.register_prompt(
            agent_name="Insight_Generator",
            prompt_text=self.insight_prompt,
            metadata={"purpose": "Generate visualizations and narratives from query results"}
        )
        self.insight_prompt_hash = insight_version.short_hash

        # Create Insight Generator Agent
        self.insight_agent = AssistantAgent(
            name="Insight_Generator",
            llm_config=self.llm_config,
            system_message=self.insight_prompt
        )

    def run(self, user_question: str):
        """
        Main workflow with observability: Question → SQL → Execution → Insights → Chart

        This is where the magic happens. The system:
        1. Asks the SQL agent to write a query
        2. Executes it against the database
        3. Generates insights and visualizations

        All dynamically, nothing hardcoded.
        Now with full observability tracing!
        """
        # START REQUEST TRACE
        correlation_id = self.tracer.start_request(mode="query")

        try:
            # Step 1: Generate SQL from natural language (TRACED)
            logger.info(f"User question: {user_question}")

            with self.tracer.trace(
                operation_name="sql_generation",
                agent_name="SQL_Generator",
                model_name="gpt-4o",
                mode="query",
                prompt_version_hash=self.sql_prompt_hash  # Layer 2
            ):
                sql_response = self.sql_agent.generate_reply(
                    messages=[{
                        "role": "user",
                        "content": user_question
                    }]
                )

                # Estimate tokens
                prompt_tokens = self._estimate_tokens(user_question)
                completion_tokens = self._estimate_tokens(sql_response)
                self.tracer.set_tokens(prompt_tokens, completion_tokens)

                # Record usage metrics (Layer 2: Token efficiency)
                total_tokens = prompt_tokens + completion_tokens
                cost = prompt_tokens * 0.000005 + completion_tokens * 0.000015
                self.prompt_registry.record_usage("SQL_Generator", total_tokens, cost)

            # Extract the SQL query from the response
            sql_match = re.search(r'```sql\n(.*?)\n```', sql_response, re.DOTALL | re.IGNORECASE)
            if not sql_match:
                logger.error(f"Could not find SQL in response: {sql_response[:200]}")
                return {
                    "chart": None,
                    "narrative": "I couldn't generate a SQL query for that question. Could you rephrase it?"
                }

            sql_query = sql_match.group(1).strip()
            logger.info(f"Generated SQL: {sql_query}")

            # Layer 3: Score SQL quality (BEFORE execution to catch syntax errors)
            sql_quality = self.sql_scorer.score_sql(
                sql_query=sql_query,
                user_question=user_question,
                execution_success=True,  # Will update if execution fails
                execution_error=None
            )
            logger.info(f"SQL Quality Score: {sql_quality['score']}/100 ({sql_quality['rating']})")

            # Step 2: Execute the query (TRACED)
            logger.info(f"Executing query...")

            with self.tracer.trace(
                operation_name="sql_execution",
                agent_name="Database",
                model_name=None,  # Not an LLM call
                mode="query"
            ):
                # Pass custom db_path if using uploaded CSV
                results = execute_sql(sql_query, self.db_path if self.is_custom_db else None)
                # No tokens for database execution
                self.tracer.set_tokens(0, 0)

            if not results.get("success"):
                error_msg = results.get("error", "Unknown error")
                logger.error(f"Query failed: {error_msg}")
                return {
                    "chart": None,
                    "narrative": f"Query execution failed: {error_msg}",
                    "correlation_id": correlation_id
                }

            row_count = len(results.get("data", []))
            logger.info(f"Query returned {row_count} rows")

            # Layer 3: Record SQL quality score
            self.quality_registry.record_sql_quality(
                correlation_id=correlation_id,
                sql_query=sql_query,
                quality_score=sql_quality
            )

            if row_count == 0:
                return {
                    "chart": None,
                    "narrative": "The query executed successfully but returned no data.",
                    "correlation_id": correlation_id
                }

            # Step 3: Generate insights (TRACED)
            logger.info(f"Generating insights...")

            # Prepare data summary for the insight agent
            data_preview = results["data"][:5]  # Show first 5 rows
            columns = results["columns"]

            insight_prompt = f"""Analyze this query result and create a visualization:

Question: {user_question}
SQL: {sql_query}

Columns: {', '.join(columns)}
Sample Data (first {len(data_preview)} rows):
{json.dumps(data_preview, indent=2)}

Total rows: {row_count}

Create a chart spec and narrative. Return JSON only."""

            with self.tracer.trace(
                operation_name="insight_generation",
                agent_name="Insight_Generator",
                model_name="gpt-4o",
                mode="query",
                prompt_version_hash=self.insight_prompt_hash  # Layer 2
            ):
                insight_response = self.insight_agent.generate_reply(
                    messages=[{"role": "user", "content": insight_prompt}]
                )

                # Estimate tokens
                prompt_tokens = self._estimate_tokens(insight_prompt)
                completion_tokens = self._estimate_tokens(insight_response)
                self.tracer.set_tokens(prompt_tokens, completion_tokens)

                # Record usage metrics (Layer 2: Token efficiency)
                total_tokens = prompt_tokens + completion_tokens
                cost = prompt_tokens * 0.000005 + completion_tokens * 0.000015
                self.prompt_registry.record_usage("Insight_Generator", total_tokens, cost)

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
                # Fallback: create a simple chart
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

            # Layer 3: Score narrative quality
            narrative_text = insights.get("narrative", "")
            narrative_quality = self.narrative_scorer.score_narrative(
                narrative=narrative_text,
                user_question=user_question,
                query_results=results,
                sql_query=sql_query
            )
            logger.info(f"Narrative Quality Score: {narrative_quality['score']}/100 ({narrative_quality['rating']})")

            # Record narrative quality score
            self.quality_registry.record_narrative_quality(
                correlation_id=correlation_id,
                narrative=narrative_text,
                quality_score=narrative_quality
            )

            # Step 4: Generate the actual chart
            logger.info(f"Creating visualization...")
            chart_spec = insights.get("chart", {})

            try:
                # Extract the chart type from Altair spec (mark field)
                chart_type = chart_spec.get("mark", "bar")

                # Call generate_chart with proper parameters
                final_chart = generate_chart(
                    data=results,  # Pass the database results
                    chart_type=chart_type,  # Pass the chart type
                    output_format='base64'  # Return base64 for frontend display
                )
            except Exception as chart_error:
                logger.error(f"Chart generation failed: {chart_error}")
                # Return data without chart
                final_chart = None

            logger.info(f"Analysis complete!")
            return {
                "chart": final_chart,
                "narrative": insights.get("narrative", "Analysis complete."),
                "sql_query": sql_query,
                "row_count": row_count,
                "correlation_id": correlation_id,
                # Layer 3: Quality scores
                "sql_quality": sql_quality,
                "narrative_quality": narrative_quality
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response was: {insight_response[:500]}")
            return {
                "chart": None,
                "narrative": "I had trouble formatting the insights. Please try again.",
                "correlation_id": correlation_id
            }
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "chart": None,
                "narrative": f"An error occurred: {str(e)}",
                "correlation_id": correlation_id
            }
        finally:
            # END REQUEST TRACE
            self.tracer.end_request()

    # Legacy method for compatibility
    def generate_sql_with_langchain(self, question: str) -> dict:
        """Legacy method - not used in new simplified workflow."""
        return {"success": False, "error": "LangChain integration deprecated"}
