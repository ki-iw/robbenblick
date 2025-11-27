FROM python:3.10-slim

# System-Libs, die OpenCV & Co. gern haben
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV YOLO_CONFIG_DIR=/app/.ultralytics
RUN mkdir -p $YOLO_CONFIG_DIR


# Nur RUNTIME-Dependencies installieren, NICHT dein komplettes requirements.txt
# WICHTIG: hier KEIN numpy und KEIN opencv-<...> pinnen.
RUN pip install --no-cache-dir \
    ultralytics \
    sahi==0.11.36 \
    streamlit==1.51.0 \
    opencv-python-headless \
    supervision \
    pillow \
    pandas \
    plotly \
    python-dotenv==1.0.0 \
    loguru==0.7.3 \
    shapely==2.1.2 \
    tqdm==4.67.1 \
    dotmap==1.3.30

# We must assume that robbenblick.utils and robbenblick.inference exist.
# Therefore, we create a module structure.
RUN mkdir -p robbenblick

# Copy the relevant Python files into the module
COPY robbenblick/inference.py robbenblick/inference.py
COPY robbenblick/utils.py robbenblick/utils.py
COPY robbenblick/__init__.py robbenblick/__init__.py

# We need the configuration file
RUN mkdir -p configs
COPY configs/base_config.yaml configs/base_config.yaml

# The Streamlit App
COPY robbenblick/streamlit_app.py .
COPY .streamlit .streamlit

# The model path is currently hardcoded in the Streamlit code:
ENV RUN_ID="iter_run_model_yolov8m.pt_freeze_10_epochs_150"
RUN mkdir -p runs/detect/$RUN_ID/weights

# If you want to copy the model directly into the image (only for smaller models):
COPY runs/detect/$RUN_ID/weights/best.pt runs/detect/$RUN_ID/weights/best.pt

EXPOSE 8501

# Add a startup script to ensure Streamlit is started
COPY start.sh .
RUN chmod +x start.sh

# Start the Streamlit application
CMD ["./start.sh"]
