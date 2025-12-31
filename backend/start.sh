#!/bin/bash
# Render start script for backend

echo "Starting StripUnetMCSA Backend on port $PORT..."

# Start the FastAPI application
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
