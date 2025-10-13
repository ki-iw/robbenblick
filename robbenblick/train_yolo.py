import torch
from ultralytics import YOLO

from robbenblick import DATA_PATH, logger


def main():
    """
    Detects the best available device and trains a YOLOv8 model.
    """
    # --- 1. Device Detection ---
    # Check for NVIDIA CUDA GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("✅ Found NVIDIA CUDA GPU. Training will use the GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("✅ Found Apple Silicon GPU. Training will use MPS.")
    else:
        device = "cpu"
        logger.warning("⚠️ No GPU found. Training will use the CPU (this will be slow).")

    # --- 2. Configuration ---
    # Define your dataset and model configuration here
    DATA_YAML_PATH = "data/processed/dataset_yolo/data.yaml"
    MODEL_TO_USE = "yolov8n.pt"  # yolov8n.pt, yolov8s.pt, etc.
    EPOCHS = 50
    IMAGE_SIZE = 640
    BATCH_SIZE = 16

    # --- 3. Training ---
    try:
        # Load a pretrained YOLOv8 model
        model = YOLO(MODEL_TO_USE)

        # Train the model using the detected device
        logger.info(f"Starting training on device: '{device}'...")
        model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,  # This is the crucial part!
        )
        logger.info("Training complete! 🎉")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


if __name__ == "__main__":
    # This ensures the script runs when called from the command line
    main()
