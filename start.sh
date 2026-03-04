#!/bin/bash
# Entrypoint script: starts both the pipeline orchestrator and the Streamlit UI

echo "🚀 Starting UHSBot Pipeline Orchestrator..."
python -u main.py &

echo "🌐 Starting Streamlit Admin Dashboard on port 8501..."
streamlit run src/ui.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
