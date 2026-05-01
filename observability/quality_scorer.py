"""
Quality Scorer - Layer 3: Output Quality

Validates and scores:
- SQL quality (syntax, schema compliance, best practices)
- Narrative quality (relevance, hallucination detection)
- Tracks quality trends over time
"""

import re
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class SQLQualityScorer:
    """
    Scores SQL query quality on multiple dimensions.

    Scoring criteria:
    - Syntax validity (25 points)
    - Schema compliance (25 points)
    - Best practices (25 points)
    - Query optimization (25 points)

    Total: 0-100 score
    """

    def __init__(self, db_path: str):
        """
        Initialize SQL quality scorer.

        Args:
            db_path: Path to SQLite database for schema validation
        """
        self.db_path = db_path

    def score_sql(
        self,
        sql_query: str,
        user_question: str,
        execution_success: bool = True,
        execution_error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Score SQL query quality.

        Args:
            sql_query: The SQL query to score
            user_question: Original user question
            execution_success: Whether query executed successfully
            execution_error: Error message if execution failed

        Returns:
            Dictionary with score and breakdown
        """
        score = 0
        breakdown = {}
        issues = []

        # 1. Syntax Validity (25 points)
        syntax_score, syntax_issues = self._score_syntax(sql_query, execution_success, execution_error)
        score += syntax_score
        breakdown["syntax"] = syntax_score
        issues.extend(syntax_issues)

        # 2. Schema Compliance (25 points)
        schema_score, schema_issues = self._score_schema_compliance(sql_query)
        score += schema_score
        breakdown["schema_compliance"] = schema_score
        issues.extend(schema_issues)

        # 3. Best Practices (25 points)
        best_practices_score, bp_issues = self._score_best_practices(sql_query)
        score += best_practices_score
        breakdown["best_practices"] = best_practices_score
        issues.extend(bp_issues)

        # 4. Query Optimization (25 points)
        optimization_score, opt_issues = self._score_optimization(sql_query, user_question)
        score += optimization_score
        breakdown["optimization"] = optimization_score
        issues.extend(opt_issues)

        # Quality rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 50:
            rating = "acceptable"
        else:
            rating = "poor"

        return {
            "score": score,
            "rating": rating,
            "breakdown": breakdown,
            "issues": issues,
            "recommendations": self._generate_recommendations(issues)
        }

    def _score_syntax(
        self,
        sql_query: str,
        execution_success: bool,
        execution_error: Optional[str]
    ) -> Tuple[int, List[str]]:
        """Score SQL syntax validity (0-25 points)"""
        score = 25
        issues = []

        if not execution_success:
            # Major syntax error (execution failed)
            score = 0
            issues.append(f"Syntax error: {execution_error}")
        else:
            # Check for common syntax issues
            if not sql_query.strip().upper().startswith('SELECT') and \
               not sql_query.strip().upper().startswith('WITH'):
                score -= 5
                issues.append("Query should start with SELECT or WITH")

            # Check for unbalanced parentheses
            if sql_query.count('(') != sql_query.count(')'):
                score -= 10
                issues.append("Unbalanced parentheses")

            # Check for missing semicolon at end (minor)
            if not sql_query.strip().endswith(';'):
                score -= 2
                issues.append("Missing semicolon at end (minor)")

        return max(0, score), issues

    def _score_schema_compliance(self, sql_query: str) -> Tuple[int, List[str]]:
        """Score schema compliance (0-25 points)"""
        score = 25
        issues = []

        try:
            # Connect to database to check schema
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all table and column names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]

            valid_columns = set()
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = [row[1] for row in cursor.fetchall()]
                valid_columns.update(columns)

            conn.close()

            # Extract mentioned columns from query (simplified)
            # Look for common patterns like: column_name, table.column_name
            query_upper = sql_query.upper()

            # Check if query references valid tables
            mentioned_tables = []
            for table in tables:
                if table.upper() in query_upper or f"{table.upper()}." in query_upper:
                    mentioned_tables.append(table)

            if not mentioned_tables:
                score -= 10
                issues.append("No valid table references found")

            # Simple column validation (checks for common column names)
            common_columns = ['PRODUCT', 'REGION', 'AMOUNT', 'DATE', 'CUSTOMER']
            found_columns = sum(1 for col in common_columns if col in query_upper)

            if found_columns == 0:
                score -= 5
                issues.append("No recognizable columns found")

        except Exception as e:
            logger.warning(f"Schema compliance check failed: {e}")
            score -= 5
            issues.append(f"Could not verify schema compliance: {str(e)}")

        return max(0, score), issues

    def _score_best_practices(self, sql_query: str) -> Tuple[int, List[str]]:
        """Score SQL best practices (0-25 points)"""
        score = 25
        issues = []

        query_upper = sql_query.upper()

        # Check for SELECT *
        if 'SELECT *' in query_upper:
            score -= 5
            issues.append("Avoid SELECT * - specify columns explicitly")

        # Check for LIMIT clause (safety)
        if 'LIMIT' not in query_upper:
            score -= 3
            issues.append("Missing LIMIT clause (could return many rows)")

        # Check for proper aggregation
        has_group_by = 'GROUP BY' in query_upper
        has_aggregation = any(agg in query_upper for agg in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN('])

        if has_aggregation and not has_group_by:
            # This is OK if aggregating entire table
            pass
        elif has_group_by and not has_aggregation:
            score -= 5
            issues.append("GROUP BY without aggregation function")

        # Check for WHERE before GROUP BY (optimization)
        where_pos = query_upper.find('WHERE')
        group_pos = query_upper.find('GROUP BY')

        if where_pos > -1 and group_pos > -1:
            if where_pos > group_pos:
                score -= 5
                issues.append("WHERE clause should come before GROUP BY")

        # Check for proper ORDER BY
        if 'ORDER BY' in query_upper:
            # Check if ordering by aggregated column (good practice)
            if has_aggregation:
                score += 2  # Bonus for ordering aggregated results

        return max(0, score), issues

    def _score_optimization(self, sql_query: str, user_question: str) -> Tuple[int, List[str]]:
        """Score query optimization (0-25 points)"""
        score = 25
        issues = []

        query_upper = sql_query.upper()
        question_lower = user_question.lower()

        # Check for "top N per group" patterns
        top_n_keywords = ['top', 'best', 'worst', 'each', 'per', 'by']
        has_top_n_intent = sum(1 for kw in top_n_keywords if kw in question_lower) >= 2

        if has_top_n_intent:
            # Should use window functions
            has_window_function = any(wf in query_upper for wf in [
                'ROW_NUMBER()', 'RANK()', 'DENSE_RANK()', 'PARTITION BY'
            ])

            if not has_window_function:
                score -= 15
                issues.append("Query should use window functions (ROW_NUMBER, PARTITION BY) for 'top N per group'")

        # Check for CTE usage (good for readability)
        if 'WITH' in query_upper and 'AS (' in query_upper:
            score += 5  # Bonus for using CTEs

        # Check for adaptive date filtering
        if 'date' in question_lower or 'month' in question_lower or 'year' in question_lower:
            # Should use adaptive dates, not hardcoded
            if "DATE('NOW'" in query_upper or "DATETIME('NOW'" in query_upper:
                score -= 5
                issues.append("Use adaptive date filtering: DATE((SELECT MAX(date) FROM ...), '-N months')")
            elif "MAX(DATE)" in query_upper or "MAX(date)" in query_upper.replace(' ', ''):
                score += 3  # Bonus for adaptive dates

        return max(0, score), issues

    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate actionable recommendations from issues"""
        recommendations = []

        issue_text = ' '.join(issues).lower()

        if 'syntax error' in issue_text:
            recommendations.append("Fix syntax errors before deployment")

        if 'select *' in issue_text:
            recommendations.append("Replace SELECT * with specific column names")

        if 'limit' in issue_text:
            recommendations.append("Add LIMIT clause to prevent large result sets")

        if 'window function' in issue_text:
            recommendations.append("Use ROW_NUMBER() OVER (PARTITION BY ...) for top N per group queries")

        if 'adaptive date' in issue_text:
            recommendations.append("Use DATE((SELECT MAX(date) FROM table), '-N months') instead of NOW()")

        return recommendations


class NarrativeQualityScorer:
    """
    Scores narrative quality on multiple dimensions.

    Scoring criteria:
    - Relevance to data (30 points)
    - Completeness (20 points)
    - Clarity (25 points)
    - No hallucinations (25 points)

    Total: 0-100 score
    """

    def score_narrative(
        self,
        narrative: str,
        user_question: str,
        query_results: Dict[str, Any],
        sql_query: str
    ) -> Dict[str, Any]:
        """
        Score narrative quality.

        Args:
            narrative: The generated narrative
            user_question: Original user question
            query_results: Database query results
            sql_query: The SQL query that was executed

        Returns:
            Dictionary with score and breakdown
        """
        score = 0
        breakdown = {}
        issues = []

        # 1. Relevance (30 points)
        relevance_score, relevance_issues = self._score_relevance(narrative, user_question)
        score += relevance_score
        breakdown["relevance"] = relevance_score
        issues.extend(relevance_issues)

        # 2. Completeness (20 points)
        completeness_score, completeness_issues = self._score_completeness(
            narrative, query_results
        )
        score += completeness_score
        breakdown["completeness"] = completeness_score
        issues.extend(completeness_issues)

        # 3. Clarity (25 points)
        clarity_score, clarity_issues = self._score_clarity(narrative)
        score += clarity_score
        breakdown["clarity"] = clarity_score
        issues.extend(clarity_issues)

        # 4. Hallucination Detection (25 points)
        hallucination_score, hallucination_issues = self._score_hallucination(
            narrative, query_results, sql_query
        )
        score += hallucination_score
        breakdown["hallucination_free"] = hallucination_score
        issues.extend(hallucination_issues)

        # Quality rating
        if score >= 90:
            rating = "excellent"
        elif score >= 75:
            rating = "good"
        elif score >= 50:
            rating = "acceptable"
        else:
            rating = "poor"

        return {
            "score": score,
            "rating": rating,
            "breakdown": breakdown,
            "issues": issues
        }

    def _score_relevance(self, narrative: str, user_question: str) -> Tuple[int, List[str]]:
        """Score narrative relevance to question (0-30 points)"""
        score = 30
        issues = []

        # Extract key terms from question
        question_words = set(user_question.lower().split())
        narrative_words = set(narrative.lower().split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'what', 'show', 'me', 'get'}
        question_keywords = question_words - stop_words

        # Check overlap
        if question_keywords:
            overlap = len(question_keywords & narrative_words) / len(question_keywords)

            if overlap < 0.3:
                score -= 15
                issues.append("Narrative doesn't address key terms from question")
            elif overlap < 0.5:
                score -= 5
                issues.append("Narrative could be more relevant to question")

        return max(0, score), issues

    def _score_completeness(
        self,
        narrative: str,
        query_results: Dict[str, Any]
    ) -> Tuple[int, List[str]]:
        """Score narrative completeness (0-20 points)"""
        score = 20
        issues = []

        # Check length
        word_count = len(narrative.split())

        if word_count < 10:
            score -= 10
            issues.append("Narrative too short (< 10 words)")
        elif word_count > 100:
            score -= 5
            issues.append("Narrative too long (> 100 words) - be concise")

        # Check if narrative mentions key metrics
        has_numbers = bool(re.search(r'\d+', narrative))
        if not has_numbers and query_results.get('data'):
            score -= 5
            issues.append("Narrative should mention specific numbers from results")

        return max(0, score), issues

    def _score_clarity(self, narrative: str) -> Tuple[int, List[str]]:
        """Score narrative clarity (0-25 points)"""
        score = 25
        issues = []

        # Check for complete sentences
        sentence_count = narrative.count('.') + narrative.count('!') + narrative.count('?')

        if sentence_count == 0:
            score -= 10
            issues.append("Narrative should have complete sentences")

        # Check for jargon or unclear terms
        unclear_terms = ['etc', 'stuff', 'things', 'basically']
        for term in unclear_terms:
            if term in narrative.lower():
                score -= 3
                issues.append(f"Avoid vague terms like '{term}'")

        # Check capitalization
        if narrative and not narrative[0].isupper():
            score -= 2
            issues.append("Narrative should start with capital letter")

        return max(0, score), issues

    def _score_hallucination(
        self,
        narrative: str,
        query_results: Dict[str, Any],
        sql_query: str
    ) -> Tuple[int, List[str]]:
        """Score hallucination detection (0-25 points)"""
        score = 25
        issues = []

        # Get actual columns from results
        actual_columns = query_results.get('columns', [])

        # Check if narrative mentions columns that don't exist
        # Extract potential column references (simplified)
        narrative_lower = narrative.lower()

        # Common column name patterns
        potential_columns = re.findall(r'\b(product|region|amount|date|customer|sales|revenue|total)\b', narrative_lower)

        for col in set(potential_columns):
            # Check if this column exists in actual results
            if actual_columns and col not in [c.lower() for c in actual_columns]:
                # Might be a hallucination
                score -= 5
                issues.append(f"Narrative mentions '{col}' which may not be in results")

        # Check for specific numbers mentioned
        narrative_numbers = re.findall(r'\$?[\d,]+\.?\d*', narrative)

        if narrative_numbers and query_results.get('data'):
            # Simplified check: ensure narrative numbers exist somewhere in data
            data_str = str(query_results['data'])

            for num in narrative_numbers[:3]:  # Check first 3 numbers
                num_clean = num.replace('$', '').replace(',', '')
                if num_clean not in data_str:
                    score -= 3
                    issues.append(f"Number '{num}' in narrative not found in results (possible hallucination)")

        return max(0, score), issues


class QualityRegistry:
    """
    Central registry for tracking quality scores over time.
    """

    def __init__(self):
        self.sql_scores: List[Dict[str, Any]] = []
        self.narrative_scores: List[Dict[str, Any]] = []

        # Aggregate metrics
        self.metrics = {
            "total_queries": 0,
            "avg_sql_quality": 0.0,
            "avg_narrative_quality": 0.0,
            "excellent_count": 0,
            "poor_count": 0
        }

    def record_sql_quality(
        self,
        correlation_id: str,
        sql_query: str,
        quality_score: Dict[str, Any]
    ):
        """Record SQL quality score"""
        self.sql_scores.append({
            "correlation_id": correlation_id,
            "sql_query": sql_query[:100],  # Store first 100 chars
            "score": quality_score["score"],
            "rating": quality_score["rating"],
            "breakdown": quality_score["breakdown"],
            "issues": quality_score["issues"]
        })

        self._update_metrics()

    def record_narrative_quality(
        self,
        correlation_id: str,
        narrative: str,
        quality_score: Dict[str, Any]
    ):
        """Record narrative quality score"""
        self.narrative_scores.append({
            "correlation_id": correlation_id,
            "narrative": narrative[:100],  # Store first 100 chars
            "score": quality_score["score"],
            "rating": quality_score["rating"],
            "breakdown": quality_score["breakdown"],
            "issues": quality_score["issues"]
        })

        self._update_metrics()

    def _update_metrics(self):
        """Update aggregate metrics"""
        self.metrics["total_queries"] = len(self.sql_scores)

        if self.sql_scores:
            self.metrics["avg_sql_quality"] = sum(s["score"] for s in self.sql_scores) / len(self.sql_scores)

        if self.narrative_scores:
            self.metrics["avg_narrative_quality"] = sum(s["score"] for s in self.narrative_scores) / len(self.narrative_scores)

        # Count excellent/poor
        all_scores = self.sql_scores + self.narrative_scores
        self.metrics["excellent_count"] = sum(1 for s in all_scores if s["rating"] == "excellent")
        self.metrics["poor_count"] = sum(1 for s in all_scores if s["rating"] == "poor")

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate quality metrics"""
        return self.metrics.copy()

    def get_recent_scores(self, limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent quality scores"""
        return {
            "sql_scores": self.sql_scores[-limit:],
            "narrative_scores": self.narrative_scores[-limit:]
        }


# Global registry instance
_global_quality_registry: Optional[QualityRegistry] = None


def get_quality_registry() -> QualityRegistry:
    """Get the global quality registry (singleton)"""
    global _global_quality_registry

    if _global_quality_registry is None:
        _global_quality_registry = QualityRegistry()

    return _global_quality_registry
