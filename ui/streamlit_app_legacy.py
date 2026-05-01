import streamlit as st
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.analytics_agents import AnalyticsAgents
from utils.database import initialize_sample_data

def display_results(results):
    """Render analytics outputs with error handling."""
    with st.container():
        if not results or not results.get("chart"):
            st.error(f"❌ Error: {results.get('narrative', 'An unknown error occurred.')}")
            return

        # Narrative Section
        st.subheader("📄 Executive Summary")
        if results.get("narrative"):
            st.markdown(results["narrative"])
        else:
            st.info("No business summary was generated.")

        # Visualization Section
        st.subheader("📈 Business Visualization")
        if results.get("chart"):
            st.altair_chart(results["chart"], use_container_width=True)
        else:
            st.warning("No visualization was generated.")

def main():
    # Configure page
    st.set_page_config(
        page_title="AI Analytics Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize the database and agents
    if "initialized" not in st.session_state:
        db_path = os.path.abspath("data/sample_data.db")
        if not os.path.exists(db_path):
            initialize_sample_data()
        st.session_state.analytics_agents = AnalyticsAgents()
        st.session_state.initialized = True

    # Main interface
    st.title("📊 AI-Powered Business Analytics Simulator")
    st.markdown("Ask a question, and a team of AI agents will work together to provide an answer.")

    # Show tech stack badge
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <strong>🚀 Powered by:</strong> GPT-4 + AutoGen + LangChain |
        <strong>📊 Data:</strong> AWS Redshift-ready |
        <strong>📈 Visualization:</strong> Matplotlib + Altair
    </div>
    """, unsafe_allow_html=True)

    # Query input
    query = st.text_input(
        "Ask your business question:",
        placeholder="e.g., Show monthly sales by region for the last quarter",
        key="query_input"
    )

    # Process query
    if query:
        with st.spinner("🚀 The AI agent team is analyzing your query..."):
            try:
                results = st.session_state.analytics_agents.run(query)
                display_results(results)
            except Exception as e:
                st.error(f"🚨 System Error: {e}")

if __name__ == "__main__":
    main()