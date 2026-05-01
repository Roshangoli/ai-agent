"""
LangChain-powered SQL query generation module.
Integrates with AutoGen agents to convert natural language to SQL queries.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

try:
    # LangChain v1.0+ imports (updated Nov 2024)
    from langchain_openai import ChatOpenAI
    from langchain_community.utilities import SQLDatabase
    from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
    from langchain_community.agent_toolkits.sql.base import create_sql_agent
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    raise ImportError(
        f"LangChain v1.0+ dependencies not installed: {e}\n"
        "Run: pip install langchain>=1.0 langchain-openai langchain-community langchain-core sqlalchemy"
    )

load_dotenv()
logger = logging.getLogger(__name__)


class LangChainSQLGenerator:
    """
    LangChain-powered SQL query generator with enhanced natural language understanding.
    """

    def __init__(
        self,
        db_path: str = "data/sample_data.db",
        model: str = "gpt-4o",
        temperature: float = 0
    ):
        """
        Initialize LangChain SQL generator.

        Args:
            db_path: Path to SQLite database
            model: OpenAI model to use
            temperature: Model temperature (0 for deterministic)
        """
        self.db_path = db_path
        self.model = model
        self.temperature = temperature

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize database connection
        try:
            self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
            logger.info(f"✅ LangChain connected to database: {db_path}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise

        # Create SQL agent with toolkit (LangChain v1.0 syntax)
        self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        self.agent = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            agent_type="openai-tools",  # v1.0: Use string literal instead of AgentType enum
            verbose=True,
            max_iterations=5
        )

        # Custom prompt template for query generation
        self.custom_prompt = PromptTemplate(
            input_variables=["question", "schema", "examples"],
            template="""You are a SQL expert. Given a natural language question, generate a valid SQLite query.

Database Schema:
{schema}

Example Queries:
{examples}

Question: {question}

Generate ONLY the SQL query without any explanation. The query must:
1. Be valid SQLite syntax
2. Use proper JOIN syntax if needed
3. Include appropriate GROUP BY, ORDER BY, and LIMIT clauses
4. Handle date filtering correctly using SQLite date functions
5. Return meaningful column aliases

SQL Query:"""
        )

    def generate_query(
        self,
        question: str,
        use_agent: bool = False
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question.

        Args:
            question: Natural language question
            use_agent: If True, use SQL agent (more powerful but slower)

        Returns:
            Dictionary with query and metadata
        """
        try:
            logger.info(f"🔍 Generating SQL for: {question}")

            if use_agent:
                # Use SQL agent for complex queries
                result = self.agent.invoke({"input": question})
                query = self._extract_query_from_agent(result)
            else:
                # Use LLM directly with prompt for faster generation (v1.0 LCEL)
                schema_info = self.db.get_table_info()
                prompt = f"""You are a SQL expert. Generate a valid SQLite query for this question.

Database Schema:
{schema_info}

Question: {question}

Generate ONLY the SQL query without explanation. The query must be valid SQLite syntax.

SQL Query:"""

                response = self.llm.invoke(prompt)
                query = response.content if hasattr(response, 'content') else str(response)

            # Clean up query
            query = self._clean_query(query)

            logger.info(f"✅ Generated query: {query}")

            return {
                "success": True,
                "query": query,
                "question": question,
                "method": "agent" if use_agent else "direct_llm"
            }

        except Exception as e:
            logger.error(f"❌ Query generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    def generate_query_with_schema(
        self,
        question: str,
        schema_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate SQL query with explicit schema information.

        Args:
            question: Natural language question
            schema_info: Database schema information

        Returns:
            Dictionary with query and metadata
        """
        try:
            # Format schema for prompt
            schema_text = self._format_schema(schema_info)
            examples_text = self._format_examples(schema_info)

            # Use LCEL (LangChain Expression Language) - v1.0 way
            # Build chain: prompt | llm | output_parser
            chain = self.custom_prompt | self.llm | StrOutputParser()

            # Generate query
            result = chain.invoke({
                "question": question,
                "schema": schema_text,
                "examples": examples_text
            })

            query = self._clean_query(result)

            logger.info(f"✅ Generated query with schema: {query}")

            return {
                "success": True,
                "query": query,
                "question": question,
                "method": "schema_enhanced_lcel"
            }

        except Exception as e:
            logger.error(f"❌ Schema-enhanced query generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }

    def validate_and_explain_query(self, query: str) -> Dict[str, Any]:
        """
        Validate SQL query and provide explanation.

        Args:
            query: SQL query to validate

        Returns:
            Dictionary with validation results and explanation
        """
        try:
            # Check basic syntax
            if not query.strip().upper().startswith("SELECT"):
                return {
                    "valid": False,
                    "error": "Query must start with SELECT",
                    "query": query
                }

            # Test query execution (with LIMIT 1 for safety)
            test_query = f"{query.rstrip(';')} LIMIT 1;"
            self.db.run(test_query)

            # Generate explanation using LLM
            explanation_prompt = f"""Explain what this SQL query does in simple business terms:

{query}

Provide a brief 1-2 sentence explanation."""

            explanation = self.llm.invoke(explanation_prompt).content

            return {
                "valid": True,
                "query": query,
                "explanation": explanation
            }

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "query": query
            }

    def get_table_info(self, table_name: Optional[str] = None) -> str:
        """
        Get database table information.

        Args:
            table_name: Specific table name (optional)

        Returns:
            Formatted table information
        """
        try:
            if table_name:
                return self.db.get_table_info_no_throw([table_name])
            else:
                return self.db.get_table_info()
        except Exception as e:
            logger.error(f"Failed to get table info: {e}")
            return ""

    def _extract_query_from_agent(self, agent_result: Dict[str, Any]) -> str:
        """Extract SQL query from agent result."""
        output = agent_result.get("output", "")

        # Try to extract SQL from markdown code blocks
        if "```sql" in output:
            start = output.find("```sql") + 6
            end = output.find("```", start)
            return output[start:end].strip()
        elif "```" in output:
            start = output.find("```") + 3
            end = output.find("```", start)
            return output[start:end].strip()

        return output.strip()

    def _clean_query(self, query: str) -> str:
        """Clean and format SQL query."""
        # Remove markdown code blocks
        query = query.replace("```sql", "").replace("```", "")

        # Remove extra whitespace
        query = " ".join(query.split())

        # Ensure proper ending
        query = query.rstrip(";") + ";"

        return query.strip()

    def _format_schema(self, schema_info: Dict[str, Any]) -> str:
        """Format schema information for prompt."""
        schema_text = []

        for table_name, table_info in schema_info.get("tables", {}).items():
            schema_text.append(f"\nTable: {table_name}")
            schema_text.append(f"Description: {table_info.get('description', 'N/A')}")
            schema_text.append("Columns:")

            for col_name, col_info in table_info.get("columns", {}).items():
                col_type = col_info.get("type", "UNKNOWN")
                col_desc = col_info.get("description", "")
                schema_text.append(f"  - {col_name} ({col_type}): {col_desc}")

        return "\n".join(schema_text)

    def _format_examples(self, schema_info: Dict[str, Any]) -> str:
        """Format example queries for prompt."""
        examples = schema_info.get("query_examples", [])

        if not examples:
            return "No examples available"

        examples_text = []
        for i, example in enumerate(examples, 1):
            examples_text.append(f"\nExample {i}: {example.get('description', 'N/A')}")
            examples_text.append(f"SQL: {example.get('sql', 'N/A')}")

        return "\n".join(examples_text)


class LangChainSQLAnalyzer:
    """
    Analyzes SQL queries and provides insights using LangChain.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0):
        """
        Initialize SQL analyzer.

        Args:
            model: OpenAI model to use
            temperature: Model temperature
        """
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def analyze_query_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        question: str
    ) -> Dict[str, Any]:
        """
        Analyze query results and generate insights.

        Args:
            query: Executed SQL query
            results: Query results
            question: Original question

        Returns:
            Dictionary with insights and recommendations
        """
        try:
            # Format results for analysis
            results_summary = self._summarize_results(results)

            prompt = f"""You are a business analyst. Analyze the following SQL query results and provide insights.

Original Question: {question}
SQL Query: {query}
Results Summary: {results_summary}

Provide:
1. Key findings (2-3 bullet points)
2. Business recommendations (1-2 sentences)
3. Suggested follow-up questions (2 questions)

Format as JSON with keys: findings, recommendations, follow_up_questions"""

            response = self.llm.invoke(prompt)
            insights = response.content

            return {
                "success": True,
                "insights": insights,
                "query": query,
                "question": question
            }

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize query results for analysis."""
        if not results:
            return "No results returned"

        num_rows = len(results)
        columns = list(results[0].keys()) if results else []

        summary = f"Returned {num_rows} rows with columns: {', '.join(columns)}\n"

        # Add first few rows as sample
        if num_rows > 0:
            summary += "\nSample data:\n"
            for i, row in enumerate(results[:3], 1):
                summary += f"Row {i}: {row}\n"

        return summary


# Convenience functions for integration with AutoGen
def generate_sql_with_langchain(
    question: str,
    db_path: str = "data/sample_data.db",
    schema_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate SQL query using LangChain (for AutoGen integration).

    Args:
        question: Natural language question
        db_path: Path to database
        schema_info: Optional schema information

    Returns:
        SQL query string
    """
    generator = LangChainSQLGenerator(db_path=db_path)

    if schema_info:
        result = generator.generate_query_with_schema(question, schema_info)
    else:
        result = generator.generate_query(question)

    if result.get("success"):
        return result["query"]
    else:
        raise ValueError(f"Query generation failed: {result.get('error')}")


def validate_sql_with_langchain(query: str, db_path: str = "data/sample_data.db") -> Dict[str, Any]:
    """
    Validate SQL query using LangChain (for AutoGen integration).

    Args:
        query: SQL query to validate
        db_path: Path to database

    Returns:
        Validation result dictionary
    """
    generator = LangChainSQLGenerator(db_path=db_path)
    return generator.validate_and_explain_query(query)


if __name__ == "__main__":
    # Test the LangChain SQL generator
    print("🧪 Testing LangChain SQL Generator\n")

    generator = LangChainSQLGenerator()

    # Test queries
    test_questions = [
        "Show total sales by region for the last 3 months",
        "Which product has the highest sales?",
        "Show monthly sales trends"
    ]

    for question in test_questions:
        print(f"\n📝 Question: {question}")
        result = generator.generate_query(question)

        if result["success"]:
            print(f"✅ Generated Query:\n{result['query']}\n")

            # Validate query
            validation = generator.validate_and_explain_query(result["query"])
            if validation["valid"]:
                print(f"✅ Valid query")
                print(f"📊 Explanation: {validation['explanation']}")
            else:
                print(f"❌ Invalid: {validation['error']}")
        else:
            print(f"❌ Failed: {result['error']}")