"""
Enhanced Streamlit UI - Supports Both Query Mode and Data Science Mode
Dual-interface for: Natural language SQL queries + Autonomous ML pipeline
"""

import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.orchestrator import AgentOrchestrator
from utils.database import initialize_sample_data

def display_query_results(results):
    """Display results from Query Mode (SQL analytics)."""
    with st.container():
        if not results.get("success"):
            st.error(f"❌ Error: {results.get('error', 'Unknown error')}")
            return

        # Narrative Section
        st.subheader("📄 Executive Summary")
        if results.get("narrative"):
            st.markdown(results["narrative"])
        else:
            st.info("No summary generated.")

        # Visualization Section
        st.subheader("📈 Visualization")
        if results.get("chart"):
            chart_data = results["chart"]
            # Check if chart is a dict with file path (matplotlib PNG)
            if isinstance(chart_data, dict) and chart_data.get("success") and chart_data.get("path"):
                # Display the saved PNG file
                if os.path.exists(chart_data["path"]):
                    st.image(chart_data["path"], use_column_width=True)
                else:
                    st.error(f"Chart file not found: {chart_data['path']}")
            elif isinstance(chart_data, dict) and not chart_data.get("success"):
                st.error(f"Chart generation failed: {chart_data.get('error', 'Unknown error')}")
            else:
                # Fallback for other chart types (Altair, etc.)
                st.altair_chart(chart_data, use_container_width=True)
        else:
            st.warning("No visualization generated.")


def display_data_science_results(results):
    """Display results from Data Science Mode (ML pipeline)."""
    with st.container():
        if not results.get("success"):
            st.error(f"❌ Pipeline Error: {results.get('error', 'Unknown error')}")
            return

        # Dataset Info
        st.subheader("📊 Dataset Information")
        dataset_info = results.get("dataset_info", {})
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Original Size",
                f"{dataset_info.get('original_shape', ('?', '?'))[0]:,} rows"
            )
        with col2:
            st.metric(
                "Columns",
                f"{dataset_info.get('original_shape', ('?', '?'))[1]}"
            )
        with col3:
            st.metric(
                "After Cleaning",
                f"{dataset_info.get('cleaned_shape', ('?', '?'))[0]:,} rows"
            )

        # Cleaning Summary
        st.subheader("🧹 Data Cleaning Summary")
        cleaning = results.get("cleaning_summary", {})

        if cleaning.get("missing_values_handled"):
            with st.expander("Missing Values Handled"):
                st.json(cleaning["missing_values_handled"])

        if cleaning.get("duplicates_removed"):
            with st.expander("Duplicates Removed"):
                st.json(cleaning["duplicates_removed"])

        # Agent Decisions
        st.subheader("🤖 Agent Decisions")
        st.info("""
        All cleaning, feature engineering, and model selection decisions were made
        **autonomously** by AI agents analyzing your specific dataset.
        No hardcoded strategies used!
        """)

        agent_decisions = results.get("agent_decisions", {})
        tabs = st.tabs(["Ingestion", "Cleaning", "EDA", "Feature Engineering"])

        with tabs[0]:
            st.write(agent_decisions.get("ingestion", "See logs"))
        with tabs[1]:
            st.write(agent_decisions.get("cleaning", "See logs"))
        with tabs[2]:
            st.write(agent_decisions.get("eda", "See logs"))
        with tabs[3]:
            st.write(agent_decisions.get("feature_engineering", "See logs"))

        # Next Steps
        st.subheader("🎯 Next Steps")
        next_steps = results.get("next_steps", [])
        for i, step in enumerate(next_steps, 1):
            st.write(f"{i}. {step}")


def main():
    # Configure page
    st.set_page_config(
        page_title="AI Multi-Agent Analytics & ML Platform",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize orchestrator
    if "orchestrator" not in st.session_state:
        # Initialize sample database
        db_path = os.path.abspath("data/sample_data.db")
        if not os.path.exists(db_path):
            initialize_sample_data()

        # Initialize orchestrator
        st.session_state.orchestrator = AgentOrchestrator()
        st.session_state.mode_history = []

    # Header
    st.title("🤖 AI Multi-Agent Analytics & ML Platform")
    st.markdown("""
    **Two Powerful Modes:**
    - 💬 **Query Mode**: Ask questions → Get SQL insights + charts
    - 🔬 **Data Science Mode**: Upload CSV → Get autonomous ML pipeline
    """)

    # Tech Stack Badge
    st.markdown("""
    <div style='background-color: #1e3a5f; color: white; padding: 15px; border-radius: 10px; margin-bottom: 25px;'>
        <strong>🚀 Powered by:</strong> GPT-4 + AutoGen + LangChain |
        <strong>🤖 Agents:</strong> 7 Autonomous AI Agents |
        <strong>🎯 Innovation:</strong> ZERO Hardcoded Strategies
    </div>
    """, unsafe_allow_html=True)

    # Sidebar - Mode Selection
    st.sidebar.title("🎛️ Mode Selection")

    mode = st.sidebar.radio(
        "Choose your mode:",
        ["💬 Query Mode", "🔬 Data Science Mode"],
        help="""
        Query Mode: Ask questions about existing database
        Data Science Mode: Upload CSV for full ML pipeline
        """
    )

    # Display mode statistics
    stats = st.session_state.orchestrator.get_mode_statistics()
    if stats["total_requests"] > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Session Statistics")
        st.sidebar.write(f"**Total Requests:** {stats['total_requests']}")

        if stats.get('query_mode', {}).get('count', 0) > 0:
            qm = stats['query_mode']
            st.sidebar.write(f"**Query Mode:** {qm['count']} ({qm['success_rate']})")

        if stats.get('data_science_mode', {}).get('count', 0) > 0:
            ds = stats['data_science_mode']
            st.sidebar.write(f"**Data Science:** {ds['count']} ({ds['success_rate']})")

    # Main Content Area
    st.markdown("---")

    # MODE 1: Query Mode
    if mode == "💬 Query Mode":
        st.header("💬 Query Mode - Natural Language Analytics")
        st.markdown("""
        Ask questions in plain English about the sales database.
        AI agents will generate SQL, execute it, and create visualizations.
        """)

        # Sample queries
        with st.expander("💡 Example Queries"):
            st.code("""
            • Show total sales by region
            • Which product has the highest sales?
            • Show monthly sales trends for the last 6 months
            • Compare sales across all regions
            • What are the top 5 best-selling products?
            """)

        # Query input
        query = st.text_area(
            "Ask your business question:",
            placeholder="e.g., Show total sales by region for the last quarter",
            height=100,
            key="query_input"
        )

        # Submit button
        if st.button("🚀 Run Query", type="primary", key="query_submit"):
            if not query.strip():
                st.warning("⚠️ Please enter a question")
            else:
                with st.spinner("🤖 AI agents are analyzing your query..."):
                    try:
                        # Route through orchestrator
                        results = st.session_state.orchestrator.run(query)

                        # Store in history
                        st.session_state.mode_history.append({
                            "mode": "query",
                            "input": query,
                            "results": results
                        })

                        # Display results
                        st.markdown("---")
                        st.success("✅ Analysis Complete!")
                        display_query_results(results)

                    except Exception as e:
                        st.error(f"🚨 System Error: {e}")

    # MODE 2: Data Science Mode
    else:
        st.header("🔬 Data Science Mode - Autonomous ML Pipeline")
        st.markdown("""
        Upload a CSV file and let **7 autonomous AI agents** analyze your data,
        clean it, engineer features, and train ML models - all with ZERO hardcoded strategies!
        """)

        # File Upload
        uploaded_file = st.file_uploader(
            "📁 Upload your dataset (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a dataset with your target variable"
        )

        if uploaded_file:
            # Save uploaded file temporarily
            temp_path = f"data/uploads/{uploaded_file.name}"
            os.makedirs("data/uploads", exist_ok=True)

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # Load and preview
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(temp_path)
                else:
                    df = pd.read_excel(temp_path)

                st.subheader("📊 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

                # Show basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

                # Configuration
                st.subheader("⚙️ Pipeline Configuration")

                col1, col2 = st.columns(2)

                with col1:
                    target_column = st.selectbox(
                        "🎯 Select Target Column (what to predict):",
                        options=df.columns.tolist(),
                        help="Choose the column you want to predict"
                    )

                with col2:
                    task_type = st.selectbox(
                        "📋 Task Type:",
                        options=["auto", "classification", "regression"],
                        help="Leave as 'auto' for agents to decide"
                    )

                # Run Pipeline Button
                if st.button("🚀 Run Autonomous ML Pipeline", type="primary", key="ds_submit"):
                    with st.spinner("🤖 7 AI agents are analyzing your dataset... This may take a minute."):
                        try:
                            # Route through orchestrator
                            results = st.session_state.orchestrator.run(
                                temp_path,
                                target_column=target_column,
                                task_type=task_type
                            )

                            # Store in history
                            st.session_state.mode_history.append({
                                "mode": "data_science",
                                "input": {"file": uploaded_file.name, "target": target_column},
                                "results": results
                            })

                            # Display results
                            st.markdown("---")
                            st.success("✅ Pipeline Complete!")
                            display_data_science_results(results)

                        except Exception as e:
                            st.error(f"🚨 Pipeline Error: {e}")
                            st.exception(e)

            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")

        else:
            # Instructions when no file uploaded
            st.info("👆 Upload a CSV/Excel file to get started")

            with st.expander("💡 What the AI Agents Will Do"):
                st.markdown("""
                **1. Data_Ingestion_Agent** - Load and validate your file
                **2. Data_Cleaning_Agent** - Autonomously clean data:
                   - Analyze distributions → Choose imputation strategies
                   - Detect outliers → Decide: keep, cap, or remove
                   - Handle duplicates intelligently

                **3. EDA_Agent** - Discover patterns:
                   - Identify correlations → Recommend feature drops
                   - Detect class imbalance → Suggest strategies
                   - Choose appropriate visualizations

                **4. Feature_Engineering_Agent** - Create optimal features:
                   - Smart encoding (target/one-hot/frequency)
                   - Transform skewed data (log, sqrt)
                   - Create domain-relevant features

                **5. ML_Training_Agent** - Select and train models:
                   - Choose 3-5 models based on data characteristics
                   - Handle class imbalance automatically
                   - Tune hyperparameters

                **6. Evaluation_Agent** - Compare models:
                   - Select metrics based on business context
                   - Check for overfitting
                   - Identify feature importance

                **7. Reporting_Agent** - Generate insights:
                   - Comprehensive markdown report
                   - Actionable business recommendations
                   - Model deployment guidelines

                **🎯 Key Feature: ZERO hardcoded strategies!**
                Every decision is made by analyzing YOUR specific dataset.
                """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>🤖 Autonomous AI Multi-Agent System</strong></p>
        <p>Powered by GPT-4 + AutoGen + LangChain | Built with Streamlit</p>
        <p><em>All decisions made dynamically - no hardcoded strategies</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
