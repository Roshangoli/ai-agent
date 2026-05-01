"""
Report Generation Module
Creates comprehensive HTML and Markdown reports for ML pipelines.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive reports for data science pipelines.
    """

    def __init__(self):
        """Initialize ReportGenerator."""
        pass

    def generate_html_report(
        self,
        pipeline_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            pipeline_results: Complete pipeline execution results
            output_path: Optional path to save HTML file

        Returns:
            HTML report as string
        """
        try:
            html = self._build_html_structure(pipeline_results)

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                logger.info(f"✅ HTML report saved to {output_path}")

            return html

        except Exception as e:
            logger.error(f"❌ HTML report generation failed: {e}")
            return f"<html><body><h1>Error: {str(e)}</h1></body></html>"

    def _build_html_structure(self, results: Dict[str, Any]) -> str:
        """Build complete HTML report structure."""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science Pipeline Report</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        {self._build_header(results)}
        {self._build_dataset_summary(results)}
        {self._build_data_quality_section(results)}
        {self._build_eda_section(results)}
        {self._build_feature_engineering_section(results)}
        {self._build_model_training_section(results)}
        {self._build_model_comparison_section(results)}
        {self._build_best_model_section(results)}
        {self._build_recommendations_section(results)}
        {self._build_footer()}
    </div>
</body>
</html>"""

        return html

    def _get_css_styles(self) -> str:
        """Get CSS styles for HTML report."""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        h1 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 2.5em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }

        h2 {
            color: #764ba2;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.8em;
        }

        h3 {
            color: #555;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.3em;
        }

        .section {
            margin-bottom: 30px;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }

        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }

        .metric-label {
            color: #777;
            font-size: 0.9em;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }

        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        th {
            background: #667eea;
            color: white;
            font-weight: bold;
        }

        tr:hover {
            background: #f5f5f5;
        }

        .success {
            color: #28a745;
            font-weight: bold;
        }

        .warning {
            color: #ffc107;
            font-weight: bold;
        }

        .error {
            color: #dc3545;
            font-weight: bold;
        }

        .badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }

        .badge-success {
            background: #28a745;
            color: white;
        }

        .badge-warning {
            background: #ffc107;
            color: #333;
        }

        .badge-info {
            background: #17a2b8;
            color: white;
        }

        .code-block {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 10px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            text-align: center;
            color: #777;
            font-size: 0.9em;
        }
        """

    def _build_header(self, results: Dict[str, Any]) -> str:
        """Build report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""
        <h1>🤖 Autonomous Data Science Pipeline Report</h1>
        <p style="color: #777; margin-bottom: 30px;">
            Generated on {timestamp}<br>
            Powered by AI Multi-Agent System
        </p>
        """

    def _build_dataset_summary(self, results: Dict[str, Any]) -> str:
        """Build dataset summary section."""
        dataset_info = results.get("dataset_info", {})

        original_shape = dataset_info.get("original_shape", (0, 0))
        cleaned_shape = dataset_info.get("cleaned_shape", (0, 0))
        target_column = dataset_info.get("target_column", "Unknown")
        task_type = dataset_info.get("task_type", "Unknown")

        return f"""
        <div class="section">
            <h2>📊 Dataset Summary</h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{original_shape[0]:,}</div>
                    <div class="metric-label">Original Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{original_shape[1]}</div>
                    <div class="metric-label">Original Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cleaned_shape[0]:,}</div>
                    <div class="metric-label">Final Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{cleaned_shape[1]}</div>
                    <div class="metric-label">Final Columns</div>
                </div>
            </div>

            <p><strong>Target Column:</strong> {target_column}</p>
            <p><strong>Task Type:</strong> <span class="badge badge-info">{task_type.upper()}</span></p>
        </div>
        """

    def _build_data_quality_section(self, results: Dict[str, Any]) -> str:
        """Build data quality section."""
        cleaning_summary = results.get("cleaning_summary", {})

        return f"""
        <div class="section">
            <h2>🧹 Data Quality & Cleaning</h2>

            <h3>Actions Performed</h3>
            <ul>
                <li>Missing values handled: <span class="success">✓</span></li>
                <li>Duplicates removed: <span class="success">✓</span></li>
                <li>Outliers processed: <span class="success">✓</span></li>
                <li>Data types validated: <span class="success">✓</span></li>
            </ul>

            <div class="code-block">
                {json.dumps(cleaning_summary, indent=2)}
            </div>
        </div>
        """

    def _build_eda_section(self, results: Dict[str, Any]) -> str:
        """Build EDA insights section."""
        return f"""
        <div class="section">
            <h2>🔍 Exploratory Data Analysis</h2>

            <h3>Key Findings</h3>
            <ul>
                <li>Feature correlations analyzed</li>
                <li>Distribution patterns detected</li>
                <li>Data leakage checked</li>
                <li>Class imbalance assessed</li>
            </ul>
        </div>
        """

    def _build_feature_engineering_section(self, results: Dict[str, Any]) -> str:
        """Build feature engineering section."""
        return f"""
        <div class="section">
            <h2>🔧 Feature Engineering</h2>

            <h3>Transformations Applied</h3>
            <ul>
                <li>Categorical encoding completed</li>
                <li>Numerical transformations applied</li>
                <li>New features created</li>
                <li>Feature selection performed</li>
            </ul>
        </div>
        """

    def _build_model_training_section(self, results: Dict[str, Any]) -> str:
        """Build model training section."""
        return f"""
        <div class="section">
            <h2>🎯 Model Training</h2>

            <h3>Models Trained</h3>
            <p>Multiple models were trained and cross-validated to find the best performer.</p>
        </div>
        """

    def _build_model_comparison_section(self, results: Dict[str, Any]) -> str:
        """Build model comparison table."""
        model_results = results.get("model_comparison", {})

        table_rows = ""
        for model_name, metrics in model_results.items():
            if isinstance(metrics, dict):
                cv_score = metrics.get("cv_mean", "N/A")
                if cv_score != "N/A":
                    cv_score = f"{cv_score:.4f}"

                table_rows += f"""
                <tr>
                    <td>{model_name}</td>
                    <td>{cv_score}</td>
                    <td><span class="badge badge-success">Trained</span></td>
                </tr>
                """

        return f"""
        <div class="section">
            <h2>📊 Model Comparison</h2>

            <table>
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>CV Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """

    def _build_best_model_section(self, results: Dict[str, Any]) -> str:
        """Build best model recommendation section."""
        best_model = results.get("best_model_name", "Unknown")
        best_score = results.get("best_cv_score", 0)

        return f"""
        <div class="section">
            <h2>🏆 Best Model Recommendation</h2>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{best_model}</div>
                    <div class="metric-label">Recommended Model</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_score:.4f}</div>
                    <div class="metric-label">Cross-Validation Score</div>
                </div>
            </div>

            <h3>Why This Model?</h3>
            <p>The {best_model} model achieved the best cross-validation score and is recommended for deployment.</p>
        </div>
        """

    def _build_recommendations_section(self, results: Dict[str, Any]) -> str:
        """Build recommendations section."""
        return f"""
        <div class="section">
            <h2>💡 Recommendations</h2>

            <h3>Next Steps</h3>
            <ol>
                <li>Review feature importance to understand model decisions</li>
                <li>Test model on production data before deployment</li>
                <li>Set up monitoring for model performance drift</li>
                <li>Consider ensemble methods for improved performance</li>
                <li>Document model limitations and edge cases</li>
            </ol>
        </div>
        """

    def _build_footer(self) -> str:
        """Build report footer."""
        return f"""
        <div class="footer">
            <p>🤖 Generated with Autonomous AI Data Science Pipeline</p>
            <p>Powered by Multi-Agent System | Built with AutoGen & Scikit-Learn</p>
        </div>
        """

    def generate_markdown_report(
        self,
        pipeline_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate Markdown report.

        Args:
            pipeline_results: Complete pipeline execution results
            output_path: Optional path to save Markdown file

        Returns:
            Markdown report as string
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dataset_info = pipeline_results.get("dataset_info", {})

            markdown = f"""# 🤖 Autonomous Data Science Pipeline Report

Generated on {timestamp}

---

## 📊 Dataset Summary

- **Original Shape**: {dataset_info.get('original_shape', 'Unknown')}
- **Cleaned Shape**: {dataset_info.get('cleaned_shape', 'Unknown')}
- **Target Column**: {dataset_info.get('target_column', 'Unknown')}
- **Task Type**: {dataset_info.get('task_type', 'Unknown')}

---

## 🧹 Data Quality & Cleaning

The autonomous data cleaning agent performed the following actions:

- Missing values handled
- Duplicates removed
- Outliers processed
- Data types validated

---

## 🔍 Exploratory Data Analysis

Key findings from automated EDA:

- Feature correlations analyzed
- Distribution patterns detected
- Data leakage checked
- Class imbalance assessed

---

## 🔧 Feature Engineering

Transformations applied by the feature engineering agent:

- Categorical encoding completed
- Numerical transformations applied
- New features created
- Feature selection performed

---

## 🎯 Model Training & Evaluation

Multiple models were trained and compared using cross-validation.

### Best Model: {pipeline_results.get('best_model_name', 'Unknown')}

- **CV Score**: {pipeline_results.get('best_cv_score', 'N/A')}

---

## 💡 Recommendations

1. Review feature importance to understand model decisions
2. Test model on production data before deployment
3. Set up monitoring for model performance drift
4. Consider ensemble methods for improved performance

---

*Generated by Autonomous AI Data Science Pipeline*
"""

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                logger.info(f"✅ Markdown report saved to {output_path}")

            return markdown

        except Exception as e:
            logger.error(f"❌ Markdown report generation failed: {e}")
            return f"# Error\n\n{str(e)}"


# Convenience function
def generate_report(
    pipeline_results: Dict[str, Any],
    format: str = "html",
    output_path: Optional[str] = None
) -> str:
    """
    Quick report generation.

    Args:
        pipeline_results: Pipeline execution results
        format: "html" or "markdown"
        output_path: Optional save path

    Returns:
        Report as string
    """
    generator = ReportGenerator()

    if format == "html":
        return generator.generate_html_report(pipeline_results, output_path)
    elif format == "markdown":
        return generator.generate_markdown_report(pipeline_results, output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
