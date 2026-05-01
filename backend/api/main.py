"""
FastAPI Backend for AI Analytics System
Provides REST API for React frontend
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path
import logging
import os
import tempfile
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.analytics_agents import AnalyticsAgents
from agents.data_science_agents import DataScienceAgentTeam
from utils.database import execute_sql, get_schema, initialize_sample_data
from utils.chart_generator import generate_chart
from utils.csv_to_sqlite import CSVToSQLite
from utils.model_persistence import ModelPersistence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Analytics API",
    description="REST API for Multi-Agent AI Data Analytics System",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents (lazy loading)
analytics_agents = None
datascience_agents = None


def get_analytics_agents():
    """Lazy load analytics agents"""
    global analytics_agents
    if analytics_agents is None:
        try:
            analytics_agents = AnalyticsAgents(use_langchain=True)
            logger.info("✅ Analytics agents initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize agents: {e}")
            analytics_agents = AnalyticsAgents(use_langchain=False)
    return analytics_agents


def get_datascience_agents():
    """Lazy load data science agents"""
    global datascience_agents
    if datascience_agents is None:
        try:
            datascience_agents = DataScienceAgentTeam()
            logger.info("✅ Data science agents initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize data science agents: {e}")
            raise e
    return datascience_agents


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization
    """
    import pandas as pd

    if isinstance(obj, dict):
        return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return list(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif hasattr(obj, '__dict__') and not callable(obj):
        # Skip sklearn models and other non-serializable objects
        return str(type(obj).__name__)
    elif callable(obj):
        # Skip functions
        return None
    else:
        return obj


# Request/Response Models
class QueryRequest(BaseModel):
    question: str
    use_langchain: bool = True
    db_path: Optional[str] = None


class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    chart: Optional[Dict[str, Any]] = None
    narrative: Optional[str] = None
    error: Optional[str] = None


class SQLExecuteRequest(BaseModel):
    sql: str


class SchemaResponse(BaseModel):
    tables: Dict[str, Any]
    query_examples: List[Dict[str, Any]]


# API Endpoints

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "online",
        "service": "AI Analytics API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "sql": "/api/sql/execute",
            "schema": "/api/schema",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        result = execute_sql("SELECT COUNT(*) as count FROM sales LIMIT 1;")
        db_healthy = result.get("success", False)

        return {
            "status": "healthy" if db_healthy else "degraded",
            "database": "connected" if db_healthy else "error",
            "agents": "initialized" if analytics_agents else "not_loaded"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/api/schema", response_model=SchemaResponse)
async def get_database_schema():
    """Get database schema information"""
    try:
        schema = get_schema()
        return schema
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_analytics(request: QueryRequest):
    """
    Main analytics endpoint - processes natural language questions

    Example:
        POST /api/query
        {
            "question": "Show total sales by region",
            "use_langchain": true,
            "db_path": "/path/to/custom.db"  # optional
        }
    """
    try:
        logger.info(f"📝 Received query: {request.question}")
        if request.db_path:
            logger.info(f"📊 Using custom database: {request.db_path}")

        # Get analytics agents (with optional custom DB)
        if request.db_path:
            agents = AnalyticsAgents(
                use_langchain=request.use_langchain,
                custom_db_path=request.db_path,
                custom_table_name="uploaded_data"  # Table name from CSV conversion
            )
        else:
            agents = get_analytics_agents()

        # Process query
        result = agents.run(request.question)

        if result and result.get("chart"):
            # Convert numpy types to Python native types
            result = convert_numpy_types(result)
            return QueryResponse(
                success=True,
                chart=result.get("chart"),
                narrative=result.get("narrative"),
                data=result.get("data")
            )
        else:
            return QueryResponse(
                success=False,
                error=result.get("narrative", "Query processing failed"),
                narrative=result.get("narrative")
            )

    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@app.post("/api/sql/execute")
async def execute_sql_query(request: SQLExecuteRequest):
    """
    Execute a SQL query directly (for advanced users)

    Example:
        POST /api/sql/execute
        {
            "sql": "SELECT * FROM sales LIMIT 10;"
        }
    """
    try:
        logger.info(f"🔍 Executing SQL: {request.sql}")

        # Execute query
        result = execute_sql(request.sql)

        if result.get("success"):
            return {
                "success": True,
                "data": result.get("data", []),
                "columns": result.get("columns", []),
                "row_count": result.get("row_count", 0)
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=result.get("error", "Query execution failed")
            )

    except Exception as e:
        logger.error(f"❌ SQL execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics():
    """Get database statistics"""
    try:
        stats_queries = {
            "total_sales": "SELECT SUM(amount) as total FROM sales;",
            "total_transactions": "SELECT COUNT(*) as count FROM sales;",
            "regions": "SELECT COUNT(DISTINCT region) as count FROM sales;",
            "products": "SELECT COUNT(DISTINCT product) as count FROM sales;",
            "avg_sale": "SELECT AVG(amount) as avg FROM sales;",
            "date_range": "SELECT MIN(date) as min_date, MAX(date) as max_date FROM sales;"
        }

        stats = {}
        for key, query in stats_queries.items():
            result = execute_sql(query)
            if result.get("success") and result.get("data"):
                stats[key] = result["data"][0]

        # Convert numpy types to Python native types
        stats = convert_numpy_types(stats)

        return {
            "success": True,
            "stats": stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chart/generate")
async def generate_chart_api(data: Dict[str, Any]):
    """
    Generate chart from data

    Example:
        POST /api/chart/generate
        {
            "data": {...},
            "chart_type": "bar",
            "title": "Sales by Region"
        }
    """
    try:
        chart_result = generate_chart(
            data=data.get("data"),
            chart_type=data.get("chart_type", "auto"),
            title=data.get("title"),
            output_format="base64"
        )

        if chart_result.get("success"):
            return {
                "success": True,
                "chart": chart_result.get("data"),
                "chart_type": chart_result.get("chart_type")
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=chart_result.get("error", "Chart generation failed")
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/init/database")
async def initialize_database():
    """Initialize database with sample data (development only)"""
    try:
        initialize_sample_data()
        return {
            "success": True,
            "message": "Database initialized with sample data"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/csv")
async def upload_csv_for_query(file: UploadFile = File(...)):
    """
    Upload CSV file and convert to SQLite for Query Mode

    Returns the database path to use with /api/query endpoint

    Example:
        POST /api/upload/csv
        Form Data:
            file: sales_data.csv
    """
    temp_csv_path = None

    try:
        logger.info(f"📤 CSV Upload for Query Mode: {file.filename}")

        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_csv_path = temp_file.name

        logger.info(f"💾 CSV saved to: {temp_csv_path}")

        # Convert CSV to SQLite
        converter = CSVToSQLite()
        db_path, metadata = converter.convert_csv_to_sqlite(temp_csv_path)

        # Clean up temporary CSV file
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        if not metadata.get("success"):
            raise HTTPException(status_code=500, detail=metadata.get("error", "CSV conversion failed"))

        logger.info(f"✅ SQLite database created: {db_path}")
        logger.info(f"   Table: {metadata['table_name']}")
        logger.info(f"   Rows: {metadata['rows']}")
        logger.info(f"   Columns: {metadata['columns']}")

        # Generate intelligent example questions
        example_questions = []
        try:
            from utils.question_generator import QuestionGenerator
            question_gen = QuestionGenerator()
            example_questions = question_gen.generate_questions_from_metadata(
                column_names=metadata['column_names'],
                table_name=metadata['table_name'],
                row_count=metadata['rows'],
                num_questions=5
            )
            logger.info(f"✅ Generated {len(example_questions)} example questions")
        except Exception as e:
            logger.warning(f"⚠️ Question generation failed, using fallback: {e}")
            example_questions = [
                "Show me a summary of all data",
                "What are the top 10 rows?",
                "Show me the total count of records"
            ]

        # Generate intelligent stats/KPI cards
        dynamic_stats = {}
        try:
            from utils.stats_generator import StatsGenerator
            stats_gen = StatsGenerator()
            dynamic_stats = stats_gen.generate_stats_from_metadata(
                column_names=metadata['column_names'],
                row_count=metadata['rows']
            )
            logger.info(f"✅ Generated {len(dynamic_stats)} stat cards")
        except Exception as e:
            logger.warning(f"⚠️ Stats generation failed: {e}")
            dynamic_stats = {}

        return {
            "success": True,
            "db_path": db_path,
            "table_name": metadata['table_name'],
            "row_count": metadata['rows'],
            "column_count": metadata['columns'],
            "columns": metadata['column_names'],
            "example_questions": example_questions,  # NEW: Auto-generated questions
            "dynamic_stats": dynamic_stats,  # NEW: Auto-generated stats
            "message": f"CSV converted to SQLite successfully. Use db_path in /api/query"
        }

    except Exception as e:
        logger.error(f"❌ CSV upload failed: {e}")

        # Clean up on error
        if temp_csv_path and os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)

        raise HTTPException(
            status_code=500,
            detail=f"CSV upload failed: {str(e)}"
        )


@app.post("/api/datascience/train")
async def train_ml_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    task_type: str = Form("auto")
):
    """
    Data Science Mode - Train ML model autonomously

    Upload CSV file and train ML model with 8 autonomous agents

    Example:
        POST /api/datascience/train
        Form Data:
            file: customer_churn.csv
            target_column: churned
            task_type: classification (or auto)
    """
    temp_file_path = None

    try:
        logger.info(f"📊 Data Science Mode initiated")
        logger.info(f"   File: {file.filename}")
        logger.info(f"   Target: {target_column}")
        logger.info(f"   Task: {task_type}")

        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"💾 File saved to: {temp_file_path}")

        # Get data science agents
        ds_agents = get_datascience_agents()

        # Run ML pipeline
        logger.info("🤖 Starting autonomous ML pipeline...")
        results = ds_agents.run_data_science_pipeline(
            file_path=temp_file_path,
            target_column=target_column,
            task_type=task_type
        )

        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("🗑️  Temporary file cleaned up")

        # Return results
        if results.get("success"):
            logger.info(f"✅ Pipeline complete: {results.get('best_model_name')} - {results.get('best_cv_score'):.2%}")

            # Remove non-serializable objects (models, pipelines, etc.)
            safe_results = {
                "success": True,
                "model_id": results.get("model_id"),  # NEW: For predictions
                "best_model_name": results.get("best_model_name"),
                "best_cv_score": float(results.get("best_cv_score", 0)),
                "best_test_score": float(results.get("best_test_score", 0)) if results.get("best_test_score") else None,
                "task_type": results.get("dataset_info", {}).get("task_type"),
                "n_samples": int(results.get("n_samples", 0)) if results.get("n_samples") else None,
                "n_features": int(results.get("n_features", 0)) if results.get("n_features") else None,
                "feature_names": results.get("feature_names", []),
                "target_column": results.get("dataset_info", {}).get("target_column"),
                # NEW: Enhanced metrics
                "metrics": results.get("metrics", {}),
                "confusion_matrix": results.get("confusion_matrix"),
                "feature_importance": results.get("feature_importance", []),
                # Existing fields
                "dataset_info": results.get("dataset_info", {}),
                "cleaning_summary": results.get("cleaning_summary", {}),
                "preprocessing_summary": results.get("preprocessing_summary", {}),
                "report_path": results.get("report_path"),
                "report_filename": results.get("report_filename"),
                "observability": results.get("observability", {}),
                "model_selection_metadata": results.get("model_selection_metadata")
            }

            # Convert numpy types to Python native types for JSON serialization
            serializable_results = convert_numpy_types(safe_results)
            return serializable_results
        else:
            raise HTTPException(
                status_code=500,
                detail=results.get("error", "Pipeline failed")
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Data Science pipeline failed: {e}")

        # Clean up temporary file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}"
        )


@app.get("/api/datascience/report/{filename}")
async def download_report(filename: str):
    """
    Download generated HTML report from Data Science Mode

    Example:
        GET /api/datascience/report/data_science_report.html
    """
    try:
        # Get project root (2 levels up from backend/api)
        project_root = Path(__file__).parent.parent.parent
        reports_dir = project_root / "reports"

        # Security: validate filename to prevent path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")

        file_path = reports_dir / filename

        # Check if file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report not found: {filename}"
            )

        logger.info(f"📥 Downloading report: {filename}")

        return FileResponse(
            path=str(file_path),
            media_type="text/html",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Report download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Report download failed: {str(e)}"
        )


class PredictionRequest(BaseModel):
    """Request model for single predictions"""
    model_id: str
    data: Dict[str, Any]


@app.post("/api/datascience/predict")
async def predict_single(request: PredictionRequest):
    """
    Make single prediction using trained model

    Example:
        POST /api/datascience/predict
        {
            "model_id": "abc-123-def",
            "data": {
                "age": 25,
                "credit_limit": 30000,
                "payment_delay": 2,
                ...
            }
        }
    """
    try:
        logger.info(f"🔮 Single prediction request for model: {request.model_id}")

        # Load model and metadata
        persistence = ModelPersistence()
        model, preprocessor, metadata = persistence.load_model(request.model_id)

        logger.info(f"📂 Model loaded: {metadata['model_type']}")

        # Get required features
        required_features = metadata["feature_names"]
        task_type = metadata["task_type"]

        # Validate input has all required features
        missing_features = set(required_features) - set(request.data.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features: {list(missing_features)}"
            )

        # Prepare input data (ensure correct order)
        import pandas as pd
        input_df = pd.DataFrame([request.data])
        input_df = input_df[required_features]  # Ensure correct feature order

        logger.info(f"📊 Input shape: {input_df.shape}")

        # Transform input using preprocessor
        X_transformed = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(X_transformed)[0]
        prediction_proba = model.predict_proba(X_transformed)[0] if hasattr(model, 'predict_proba') else None

        logger.info(f"✅ Prediction: {prediction}")

        # Get prediction label (for classification)
        if task_type == "classification":
            # Get probability of positive class (or highest probability class)
            if prediction_proba is not None:
                if len(prediction_proba) == 2:  # Binary classification
                    confidence = float(prediction_proba[1])
                    prediction_label = str(prediction)
                else:  # Multi-class
                    confidence = float(max(prediction_proba))
                    prediction_label = str(prediction)
            else:
                confidence = None
                prediction_label = str(prediction)

            # Determine risk level based on probability
            if confidence is not None:
                if confidence > 0.7:
                    risk_level = "high"
                elif confidence > 0.4:
                    risk_level = "medium"
                else:
                    risk_level = "low"
            else:
                risk_level = "unknown"

        else:  # Regression
            prediction_label = float(prediction)
            confidence = None
            risk_level = None

        # Get top contributing features
        top_factors = []
        if "feature_importance" in metadata and metadata["feature_importance"]:
            # Sort by importance and get top 3
            sorted_importance = sorted(
                metadata["feature_importance"],
                key=lambda x: x["importance"],
                reverse=True
            )[:3]

            for feat_info in sorted_importance:
                feat_name = feat_info["feature"]
                feat_value = request.data.get(feat_name, "N/A")
                top_factors.append({
                    "feature": feat_name,
                    "contribution": float(feat_info["importance"]),
                    "value": feat_value
                })

        # Prepare response
        response = {
            "prediction": int(prediction) if task_type == "classification" else float(prediction),
            "prediction_label": prediction_label,
            "probability": confidence,
            "confidence": confidence,
            "risk_level": risk_level,
            "top_factors": top_factors,
            "model_type": metadata["model_type"],
            "task_type": task_type
        }

        # Convert numpy types
        response = convert_numpy_types(response)

        logger.info(f"🎉 Prediction complete")

        return response

    except FileNotFoundError as e:
        logger.error(f"❌ Model not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {request.model_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/datascience/predict-batch")
async def predict_batch(
    model_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Make batch predictions on uploaded CSV file

    Example:
        POST /api/datascience/predict-batch
        Form Data:
            model_id: abc-123-def
            file: new_customers.csv
    """
    temp_file_path = None

    try:
        logger.info(f"📊 Batch prediction request for model: {model_id}")

        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv', mode='wb') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        logger.info(f"💾 CSV saved to: {temp_file_path}")

        # Load model and metadata
        persistence = ModelPersistence()
        model, preprocessor, metadata = persistence.load_model(model_id)

        logger.info(f"📂 Model loaded: {metadata['model_type']}")

        # Load CSV
        import pandas as pd
        df = pd.read_csv(temp_file_path)

        logger.info(f"📊 CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Get required features
        required_features = metadata["feature_names"]
        task_type = metadata["task_type"]

        # Validate features
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required features in CSV: {list(missing_features)}"
            )

        # Ensure correct feature order
        df_features = df[required_features]

        # Transform features
        X_transformed = preprocessor.transform(df_features)

        # Make predictions
        predictions = model.predict(X_transformed)
        predictions_proba = model.predict_proba(X_transformed) if hasattr(model, 'predict_proba') else None

        # Add predictions to DataFrame
        df_results = df.copy()
        df_results['prediction'] = predictions

        if predictions_proba is not None:
            if task_type == "classification":
                # Add probability of positive class (binary) or max probability (multi-class)
                if predictions_proba.shape[1] == 2:
                    df_results['probability'] = predictions_proba[:, 1]
                else:
                    df_results['probability'] = predictions_proba.max(axis=1)
            else:
                df_results['probability'] = None

            # Calculate risk level for classification
            if task_type == "classification":
                df_results['risk_level'] = df_results['probability'].apply(
                    lambda p: 'high' if p > 0.7 else 'medium' if p > 0.4 else 'low'
                )

        # Generate prediction ID
        import uuid
        prediction_id = str(uuid.uuid4())

        # Save predictions to file
        project_root = Path(__file__).parent.parent.parent
        predictions_dir = project_root / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        predictions_file = predictions_dir / f"{prediction_id}.csv"
        df_results.to_csv(predictions_file, index=False)

        logger.info(f"💾 Predictions saved: {predictions_file}")

        # Calculate summary
        summary = {}
        if task_type == "classification" and 'risk_level' in df_results.columns:
            risk_counts = df_results['risk_level'].value_counts().to_dict()
            summary = {
                "high_risk": int(risk_counts.get('high', 0)),
                "medium_risk": int(risk_counts.get('medium', 0)),
                "low_risk": int(risk_counts.get('low', 0))
            }
        else:
            # Regression summary
            summary = {
                "mean_prediction": float(df_results['prediction'].mean()),
                "min_prediction": float(df_results['prediction'].min()),
                "max_prediction": float(df_results['prediction'].max())
            }

        # Sample predictions (first 20)
        sample_predictions = []
        for idx, row in df_results.head(20).iterrows():
            pred_dict = {
                "row": int(idx),
                "prediction": int(row['prediction']) if task_type == "classification" else float(row['prediction'])
            }
            if 'probability' in row and pd.notna(row['probability']):
                pred_dict["probability"] = float(row['probability'])
            if 'risk_level' in row:
                pred_dict["risk_level"] = row['risk_level']
            sample_predictions.append(pred_dict)

        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        logger.info(f"✅ Batch prediction complete: {len(df_results)} rows")

        return {
            "success": True,
            "prediction_id": prediction_id,
            "total_rows": len(df_results),
            "summary": summary,
            "sample_predictions": sample_predictions,
            "download_url": f"/api/datascience/download-predictions/{prediction_id}",
            "task_type": task_type
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Batch prediction failed: {e}")

        # Clean up temp file on error
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/api/datascience/download-predictions/{prediction_id}")
async def download_predictions(prediction_id: str):
    """
    Download prediction results as CSV

    Example:
        GET /api/datascience/download-predictions/abc-123-def
    """
    try:
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        predictions_dir = project_root / "predictions"

        # Security: validate prediction_id to prevent path traversal
        if ".." in prediction_id or "/" in prediction_id or "\\" in prediction_id:
            raise HTTPException(status_code=400, detail="Invalid prediction ID")

        file_path = predictions_dir / f"{prediction_id}.csv"

        # Check if file exists
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Predictions not found: {prediction_id}"
            )

        logger.info(f"📥 Downloading predictions: {prediction_id}")

        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"predictions_{timestamp}.csv"

        return FileResponse(
            path=str(file_path),
            media_type="text/csv",
            filename=download_filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )


@app.get("/api/datascience/download-model/{model_id}")
async def download_model_package(model_id: str):
    """
    Download complete model package as ZIP

    Includes:
    - model.pkl (trained model)
    - preprocessor.pkl (preprocessing pipeline)
    - metadata.json (model info)
    - README.txt (usage instructions)
    - usage_example.py (Python example code)

    Example:
        GET /api/datascience/download-model/abc-123-def
    """
    try:
        logger.info(f"📦 Model package download request: {model_id}")

        # Load model metadata to get info
        persistence = ModelPersistence()
        metadata = persistence.get_model_metadata(model_id)

        # Get model directory
        project_root = Path(__file__).parent.parent.parent
        model_dir = project_root / "models" / model_id

        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        # Create ZIP file in memory
        import zipfile
        import io
        from datetime import datetime

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add model file
            model_file = model_dir / "model.pkl"
            if model_file.exists():
                zip_file.write(model_file, "model.pkl")

            # Add preprocessor file
            preprocessor_file = model_dir / "preprocessor.pkl"
            if preprocessor_file.exists():
                zip_file.write(preprocessor_file, "preprocessor.pkl")

            # Add metadata file
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                zip_file.write(metadata_file, "feature_info.json")

            # Create README.txt
            readme_content = f"""Model Package - {metadata['model_type']} for {metadata['task_type'].title()}

TRAINED: {metadata.get('saved_at', 'Unknown')}
ACCURACY: {metadata['metrics'].get('accuracy', 0) * 100:.1f}%

FILES:
- model.pkl: Trained {metadata['model_type']} model
- preprocessor.pkl: Feature preprocessing pipeline
- feature_info.json: Model metadata and configuration
- usage_example.py: Python code example for making predictions

MODEL DETAILS:
- Model Type: {metadata['model_type']}
- Task Type: {metadata['task_type'].title()}
- Number of Features: {metadata['n_features']}
- Target Column: {metadata['target_column']}

PERFORMANCE METRICS:
- Accuracy: {metadata['metrics'].get('accuracy', 0) * 100:.1f}%
- Precision: {metadata['metrics'].get('precision', 0) * 100:.1f}%
- Recall: {metadata['metrics'].get('recall', 0) * 100:.1f}%
- F1 Score: {metadata['metrics'].get('f1_score', 0) * 100:.1f}%

REQUIREMENTS:
- Python 3.8+
- pandas
- scikit-learn
- xgboost (if using XGBoost model)
- lightgbm (if using LightGBM model)
- joblib

INSTALLATION:
    pip install pandas scikit-learn joblib

USAGE:
See usage_example.py for complete code example.

Quick start:
    import joblib
    import pandas as pd

    # Load model and preprocessor
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')

    # Load your data
    new_data = pd.read_csv('your_data.csv')

    # Make predictions
    X = preprocessor.transform(new_data)
    predictions = model.predict(X)
"""
            zip_file.writestr("README.txt", readme_content)

            # Create usage_example.py
            feature_names_str = "', '".join(metadata['feature_names'])
            usage_example = f'''"""
Example usage of the trained model
"""

import joblib
import pandas as pd
import json

# Load model artifacts
print("Loading model...")
model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load metadata
with open('feature_info.json', 'r') as f:
    info = json.load(f)

print(f"Model: {{info['model_type']}}")
print(f"Task: {{info['task_type']}}")
print(f"Accuracy: {{info['metrics']['accuracy'] * 100:.1f}}%")
print(f"Required features: {{info['feature_names']}}")

# Load new data (CSV without target column)
print("\\nLoading new data...")
new_data = pd.read_csv('your_new_data.csv')

# Ensure correct feature order and presence
required_features = info['feature_names']
missing_features = set(required_features) - set(new_data.columns)

if missing_features:
    print(f"ERROR: Missing required features: {{missing_features}}")
    exit(1)

# Select and order features correctly
X = new_data[required_features]

# Transform features using preprocessor
print("Transforming features...")
X_transformed = preprocessor.transform(X)

# Make predictions
print("Making predictions...")
predictions = model.predict(X_transformed)

# Add predictions to dataframe
new_data['prediction'] = predictions

# Get probabilities (if classification)
if info['task_type'] == 'classification' and hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X_transformed)

    # Binary classification
    if probabilities.shape[1] == 2:
        new_data['probability'] = probabilities[:, 1]
        new_data['risk_level'] = new_data['probability'].apply(
            lambda p: 'high' if p > 0.7 else 'medium' if p > 0.4 else 'low'
        )
    else:
        # Multi-class
        new_data['probability'] = probabilities.max(axis=1)

# Save results
output_file = 'predictions.csv'
new_data.to_csv(output_file, index=False)
print(f"\\n✅ Predictions saved to {{output_file}}")

# Display summary
print(f"\\nSummary:")
print(f"Total rows: {{len(new_data)}}")

if 'risk_level' in new_data.columns:
    print(f"High risk: {{(new_data['risk_level'] == 'high').sum()}}")
    print(f"Medium risk: {{(new_data['risk_level'] == 'medium').sum()}}")
    print(f"Low risk: {{(new_data['risk_level'] == 'low').sum()}}")

print("\\nFirst 5 predictions:")
print(new_data.head())
'''
            zip_file.writestr("usage_example.py", usage_example)

        # Prepare ZIP for download
        zip_buffer.seek(0)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_filename = f"{metadata['model_type']}_{timestamp}.zip"

        logger.info(f"✅ Model package created: {download_filename}")

        # Return as streaming response
        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            iter([zip_buffer.getvalue()]),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={download_filename}"}
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Model package creation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model package creation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    print("\n🚀 Starting AI Analytics API Server")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API docs: http://localhost:8000/docs")
    print("🔧 Interactive API: http://localhost:8000/redoc\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )