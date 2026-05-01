"""
Intelligent Question Generator
Analyzes CSV structure and generates relevant example questions using LLM.
"""

import logging
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """
    Generates intelligent example questions based on dataset characteristics.
    Uses LLM to create contextually relevant queries.
    """

    def __init__(self, llm_config: Dict[str, Any] = None):
        """
        Initialize Question Generator.

        Args:
            llm_config: Optional LLM configuration for AutoGen agents
        """
        if llm_config is None:
            import os
            from dotenv import load_dotenv
            load_dotenv()

            self.llm_config = {
                "model": "gpt-4o",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "temperature": 0.7
            }
        else:
            self.llm_config = llm_config

    def generate_questions_from_csv(
        self,
        csv_path: str,
        num_questions: int = 5
    ) -> List[str]:
        """
        Generate example questions by analyzing CSV structure.

        Args:
            csv_path: Path to CSV file
            num_questions: Number of questions to generate

        Returns:
            List of example questions
        """
        try:
            # Read CSV and analyze structure
            df = pd.read_csv(csv_path)

            # Get dataset characteristics
            characteristics = self._analyze_dataset(df)

            # Generate questions using LLM
            questions = self._generate_with_llm(characteristics, num_questions)

            logger.info(f"✅ Generated {len(questions)} example questions")
            return questions

        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}")
            return self._get_fallback_questions()

    def generate_questions_from_metadata(
        self,
        column_names: List[str],
        table_name: str,
        row_count: int,
        num_questions: int = 5
    ) -> List[str]:
        """
        Generate questions from metadata without reading full CSV.

        Args:
            column_names: List of column names
            table_name: Name of the table
            row_count: Number of rows
            num_questions: Number of questions to generate

        Returns:
            List of example questions
        """
        try:
            characteristics = {
                "column_names": column_names,
                "table_name": table_name,
                "row_count": row_count,
                "num_columns": len(column_names)
            }

            questions = self._generate_with_llm(characteristics, num_questions)

            logger.info(f"✅ Generated {len(questions)} example questions from metadata")
            return questions

        except Exception as e:
            logger.error(f"❌ Question generation failed: {e}")
            return self._get_fallback_questions()

    def _analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze dataset to extract characteristics.

        Args:
            df: Pandas DataFrame

        Returns:
            Dictionary with dataset characteristics
        """
        characteristics = {
            "column_names": df.columns.tolist(),
            "num_columns": len(df.columns),
            "row_count": len(df),
            "column_types": {},
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": [],
            "sample_values": {}
        }

        for col in df.columns:
            dtype = str(df[col].dtype)
            characteristics["column_types"][col] = dtype

            # Categorize columns
            if pd.api.types.is_numeric_dtype(df[col]):
                characteristics["numeric_columns"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                characteristics["date_columns"].append(col)
            else:
                characteristics["categorical_columns"].append(col)

            # Get sample values (first non-null value)
            sample = df[col].dropna().head(1)
            if len(sample) > 0:
                characteristics["sample_values"][col] = str(sample.iloc[0])

        return characteristics

    def _generate_with_llm(
        self,
        characteristics: Dict[str, Any],
        num_questions: int
    ) -> List[str]:
        """
        Use LLM to generate contextually relevant questions.

        Args:
            characteristics: Dataset characteristics
            num_questions: Number of questions to generate

        Returns:
            List of generated questions
        """
        try:
            from autogen import AssistantAgent, UserProxyAgent

            # Create question generator agent
            question_agent = AssistantAgent(
                name="Question_Generator",
                llm_config=self.llm_config,
                system_message=f"""You are an expert at generating insightful data analysis questions.

Given a dataset's structure, generate {num_questions} natural language questions that would be interesting to ask.

Rules:
1. Questions should be SPECIFIC to the actual columns in the dataset
2. Use actual column names from the dataset
3. Generate diverse question types: aggregations, comparisons, trends, filters, top N
4. Questions should be answerable with SQL queries
5. Make questions actionable and business-relevant
6. Return ONLY the questions, one per line, no numbering or extra text

Example output format:
Show total sales by region
Which product has the highest revenue?
What are the top 5 customers by order count?
"""
            )

            user_proxy = UserProxyAgent(
                name="User",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=0,
                code_execution_config=False
            )

            # Create prompt with dataset info
            prompt = f"""
Generate {num_questions} example questions for this dataset:

DATASET CHARACTERISTICS:
- Columns: {', '.join(characteristics['column_names'])}
- Number of rows: {characteristics.get('row_count', 'Unknown')}
- Numeric columns: {', '.join(characteristics.get('numeric_columns', []))}
- Categorical columns: {', '.join(characteristics.get('categorical_columns', []))}
- Date columns: {', '.join(characteristics.get('date_columns', []))}

Generate {num_questions} specific, actionable questions that use these actual column names.
Return one question per line, no numbering.
"""

            # Initiate conversation
            user_proxy.initiate_chat(question_agent, message=prompt)

            # Extract response
            response = question_agent.last_message(user_proxy)["content"]

            # Parse questions from response
            questions = [
                q.strip()
                for q in response.strip().split('\n')
                if q.strip() and not q.strip().startswith('#')
            ]

            # Return exactly num_questions
            return questions[:num_questions]

        except Exception as e:
            logger.error(f"❌ LLM question generation failed: {e}")
            return self._get_fallback_questions()

    def _get_fallback_questions(self) -> List[str]:
        """
        Get generic fallback questions when generation fails.

        Returns:
            List of generic questions
        """
        return [
            "Show me a summary of all data",
            "What are the top 10 rows?",
            "Show distinct values in each column",
            "What is the total count of records?",
            "Show me records sorted by the first column"
        ]


if __name__ == "__main__":
    # Test the question generator
    logging.basicConfig(level=logging.INFO)

    generator = QuestionGenerator()

    # Test with sample metadata
    questions = generator.generate_questions_from_metadata(
        column_names=["product_name", "sales", "region", "date", "quantity"],
        table_name="sales_data",
        row_count=1000,
        num_questions=5
    )

    print("\n📋 Generated Questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
