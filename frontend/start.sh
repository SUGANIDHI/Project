#!/bin/bash
# Render start script for frontend

echo "Starting StripUnetMCSA Frontend on port $PORT..."

# Start Streamlit with production settings
streamlit run app.py \
  --server.port=$PORT \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false
