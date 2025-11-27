#!/bin/bash

# This script ensures the application starts correctly.
# It creates temporary folders required by Streamlit (if they do not already exist).

# Create the temporary folders from streamlit_app.py,
# in case they are not covered by a volume mount.
mkdir -p data/streamlit_temp/uploads
mkdir -p data/streamlit_temp/output

echo "Starting Streamlit App..."

# Start the Streamlit app.
# --server.port 8501: Keep port 8501
# --server.address 0.0.0.0: Mandatory for Docker containers,
# so the app is accessible from outside.
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# optional: If you want the app to stay in the container after an error:
# exit $?
