#!/bin/bash
# Render build script for backend

echo "Building StripUnetMCSA Backend..."

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Note: Model weights need to be uploaded separately or fetched from cloud storage
# For now, this script assumes best_f1_0.778.pt is in the repo or will be added manually

echo "Backend build complete!"
