# Agent orchestrator for routing between query and data science modes.

import os
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

# Import existing Query Mode agents
from agents.analytics_agents import AnalyticsAgents

# Import new Data Science Mode agents
from agents.data_science_agents import DataScienceAgentTeam

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Routes user requests to the appropriate agent system.
    - Query Mode: Natural language → SQL → Charts
    - Data Science Mode: CSV upload → ML Pipeline → Model
    """

    def __init__(self):
        """Initialize the orchestrator with both agent systems."""
        logger.info("Initializing Agent Orchestrator")

        # Initialize Query Mode agents
        try:
            self.query_agents = AnalyticsAgents(use_langchain=True)
            logger.info("Query Mode agents initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Query Mode: {e}")
            self.query_agents = None

        # Initialize Data Science Mode agents
        try:
            self.data_science_agents = DataScienceAgentTeam()
            logger.info("Data Science Mode agents initialized (7 agents)")
        except Exception as e:
            logger.error(f"Failed to initialize Data Science Mode: {e}")
            self.data_science_agents = None

        self.mode_history = []

    def detect_mode(self, user_input: Union[str, Dict[str, Any]]) -> str:
        """
        Intelligently detect which mode to use based on user input.

        Args:
            user_input: Either a natural language query (str) or a dict with file_path

        Returns:
            "query" or "data_science"
        """
        # Check if user provided a file path (Data Science Mode)
        if isinstance(user_input, dict):
            if "file_path" in user_input or "csv_path" in user_input:
                logger.info("Mode detected: DATA SCIENCE (file upload)")
                return "data_science"
            elif "query" in user_input or "question" in user_input:
                logger.info("Mode detected: QUERY (natural language)")
                return "query"

        # Check if user_input is a file path
        if isinstance(user_input, str):
            # Check if it's a file path
            if os.path.exists(user_input) or user_input.endswith(('.csv', '.xlsx', '.xls')):
                logger.info("Mode detected: DATA SCIENCE (file path)")
                return "data_science"

            # Check for keywords indicating data science tasks
            ds_keywords = [
                'upload', 'csv', 'dataset', 'train', 'model', 'predict',
                'classify', 'regression', 'machine learning', 'ml',
                'feature engineering', 'eda', 'analyze dataset'
            ]

            if any(keyword in user_input.lower() for keyword in ds_keywords):
                logger.info("Mode detected: DATA SCIENCE (keywords)")
                return "data_science"

            # Default to Query Mode for natural language questions
            logger.info("Mode detected: QUERY (natural language question)")
            return "query"

        logger.warning("Could not detect mode, defaulting to QUERY")
        return "query"

    def route_to_query_mode(self, query: str) -> Dict[str, Any]:
        """
        Route to Query Mode (existing analytics system).

        Args:
            query: Natural language question about database

        Returns:
            Dictionary with chart and narrative
        """
        if not self.query_agents:
            return {
                "success": False,
                "error": "Query Mode agents not initialized",
                "mode": "query"
            }

        logger.info(f" Routing to Query Mode: '{query}'")

        try:
            # Use existing analytics agents
            result = self.query_agents.run(query)

            self.mode_history.append({
                "mode": "query",
                "input": query,
                "success": True
            })

            return {
                "success": True,
                "mode": "query",
                "chart": result.get("chart"),
                "narrative": result.get("narrative"),
                "query_used": "See agent conversation"
            }

        except Exception as e:
            logger.error(f"Query Mode failed: {e}")
            return {
                "success": False,
                "mode": "query",
                "error": str(e)
            }

    def route_to_data_science_mode(
        self,
        file_path: str,
        target_column: str,
        task_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Route to Data Science Mode (new autonomous ML pipeline).

        Args:
            file_path: Path to CSV/Excel file
            target_column: Name of target variable
            task_type: "regression", "classification", or "auto"

        Returns:
            Dictionary with pipeline results and agent decisions
        """
        if not self.data_science_agents:
            return {
                "success": False,
                "error": "Data Science Mode agents not initialized",
                "mode": "data_science"
            }

        logger.info(f" Routing to Data Science Mode")
        logger.info(f"   File: {file_path}")
        logger.info(f"   Target: {target_column}")
        logger.info(f"   Task: {task_type}")

        try:
            # Use autonomous data science agents
            result = self.data_science_agents.run_data_science_pipeline(
                file_path=file_path,
                target_column=target_column,
                task_type=task_type
            )

            self.mode_history.append({
                "mode": "data_science",
                "input": {"file": file_path, "target": target_column},
                "success": result.get("success", False)
            })

            result["mode"] = "data_science"
            return result

        except Exception as e:
            logger.error(f"Data Science Mode failed: {e}")
            return {
                "success": False,
                "mode": "data_science",
                "error": str(e)
            }

    def run(
        self,
        user_input: Union[str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main orchestration method - intelligently routes to correct mode.

        Args:
            user_input: Natural language query OR dict with file_path
            **kwargs: Additional parameters (target_column, task_type, etc.)

        Returns:
            Results from the appropriate agent system
        """
        # Detect which mode to use
        mode = self.detect_mode(user_input)

        # Route to appropriate system
        if mode == "query":
            # Extract query string
            if isinstance(user_input, dict):
                query = user_input.get("query") or user_input.get("question")
            else:
                query = user_input

            return self.route_to_query_mode(query)

        elif mode == "data_science":
            # Extract file path and parameters
            if isinstance(user_input, dict):
                file_path = user_input.get("file_path") or user_input.get("csv_path")
                target_column = user_input.get("target_column") or kwargs.get("target_column")
                task_type = user_input.get("task_type", "auto")
            else:
                file_path = user_input
                target_column = kwargs.get("target_column")
                task_type = kwargs.get("task_type", "auto")

            # Validate inputs
            if not file_path:
                return {
                    "success": False,
                    "error": "File path required for Data Science Mode"
                }

            if not target_column:
                return {
                    "success": False,
                    "error": "Target column required for Data Science Mode. Please specify which column to predict."
                }

            return self.route_to_data_science_mode(
                file_path=file_path,
                target_column=target_column,
                task_type=task_type
            )

        else:
            return {
                "success": False,
                "error": f"Unknown mode: {mode}"
            }

    def get_mode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mode usage.

        Returns:
            Dictionary with usage stats
        """
        total_requests = len(self.mode_history)

        if total_requests == 0:
            return {
                "total_requests": 0,
                "query_mode_count": 0,
                "data_science_mode_count": 0
            }

        query_count = sum(1 for req in self.mode_history if req["mode"] == "query")
        ds_count = sum(1 for req in self.mode_history if req["mode"] == "data_science")

        query_success = sum(1 for req in self.mode_history
                           if req["mode"] == "query" and req["success"])
        ds_success = sum(1 for req in self.mode_history
                        if req["mode"] == "data_science" and req["success"])

        return {
            "total_requests": total_requests,
            "query_mode": {
                "count": query_count,
                "success_count": query_success,
                "success_rate": f"{(query_success/query_count*100):.1f}%" if query_count > 0 else "0%"
            },
            "data_science_mode": {
                "count": ds_count,
                "success_count": ds_success,
                "success_rate": f"{(ds_success/ds_count*100):.1f}%" if ds_count > 0 else "0%"
            }
        }


# Convenience function for simple usage
def process_request(
    user_input: Union[str, Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process a request with automatic mode detection.

    Args:
        user_input: Query string or dict with file_path
        **kwargs: Additional parameters

    Returns:
        Results from appropriate mode

    Examples:
        # Query Mode
        result = process_request("Show sales by region")

        # Data Science Mode
        result = process_request(
            "customer_data.csv",
            target_column="churned",
            task_type="classification"
        )

        # Or with dict
        result = process_request({
            "file_path": "customer_data.csv",
            "target_column": "churned"
        })
    """
    orchestrator = AgentOrchestrator()
    return orchestrator.run(user_input, **kwargs)


if __name__ == "__main__":
    # Test the orchestrator
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("ENHANCED AGENT ORCHESTRATOR - TEST")
    print("="*70)

    orchestrator = AgentOrchestrator()

    # Test 1: Query Mode
    print("\n TEST 1: Query Mode (Natural Language → SQL)")
    print("-" * 70)

    query_result = orchestrator.run("Show total sales by region")
    print(f"Mode: {query_result.get('mode')}")
    print(f"Success: {query_result.get('success')}")
    if query_result.get('narrative'):
        print(f"Narrative: {query_result['narrative'][:200]}...")

    # Test 2: Data Science Mode (if file provided)
    if len(sys.argv) > 2:
        print("\n TEST 2: Data Science Mode (CSV → ML Pipeline)")
        print("-" * 70)

        file_path = sys.argv[1]
        target_column = sys.argv[2]

        ds_result = orchestrator.run(
            file_path,
            target_column=target_column,
            task_type="auto"
        )

        print(f"Mode: {ds_result.get('mode')}")
        print(f"Success: {ds_result.get('success')}")
        if ds_result.get('dataset_info'):
            print(f"Dataset: {ds_result['dataset_info']}")

    # Show statistics
    print("\nORCHESTRATOR STATISTICS")
    print("-" * 70)
    stats = orchestrator.get_mode_statistics()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Query Mode: {stats.get('query_mode', {}).get('count', 0)} requests")
    print(f"Data Science Mode: {stats.get('data_science_mode', {}).get('count', 0)} requests")

    print("\nOrchestrator test complete!")
