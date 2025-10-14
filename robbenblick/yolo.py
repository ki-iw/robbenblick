import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results

from robbenblick import DATA_PATH, logger, model_config


def train(run_id=model_config.run_id):
    """
    Detects the best available device and trains a YOLOv8 model.
    """
    # Check for NVIDIA CUDA GPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Found NVIDIA CUDA GPU. Training will use the GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Found Apple Silicon GPU. Training will use MPS.")
    else:
        device = "cpu"
        logger.warning(" No GPU found. Training will use the CPU.")

    try:
        # Load a pretrained YOLOv8 model
        model = YOLO(model_config.model)
        # Set the directory to save training results
        logger.info(f"Starting training model {model_config.model}, run_id: {run_id}.")
        model.train(
            data=model_config.data_yaml_path,
            epochs=model_config.epochs,
            imgsz=model_config.image_size,
            batch=model_config.batch_size,
            device=device,
            name=run_id,
        )
        logger.info("Training complete! 🎉")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


def validate(yaml_path=model_config.data_yaml_path, model=model_config.model):
    model = YOLO(model)
    result = model.val(data=yaml_path, plots=False)
    return result


def predict(model: YOLO, images_dir: str, run_id=model_config.run_id) -> None:
    """
    Runs YOLOv8 segmentation inference and saves predictions to text files.

    This function takes a trained model and a directory of images, runs the
    prediction process, and saves the output segmentation masks in the standard
    YOLOv8 .txt format (class-index x1 y1 x2 y2 ...).

    Args:
        model (YOLO): A trained YOLOv8 model instance from the ultralytics library.
        images_dir (str): The path to the directory containing images for prediction.
        run_id (str): Identifier for the prediction run, used to name the output directory.
    """
    # Run inference on the specified directory
    # 'save_txt=True' is crucial as it saves the predictions in the required YOLO format

    results: list[Results] = model.predict(
        source=images_dir,
        save_txt=True,
        save=False,
        conf=0.4,  # Optional: Set a confidence threshold for predictions
        name=run_id + "_predict",  # Name of the folder to save predictions
        exist_ok=True,  # This overwrites any existing predict directory with the same name
    )

    # The .txt files are located in a 'labels' subdirectory within the save_dir

    logger.info(f"✅ Predictions successfully saved to: {results[0].save_dir}")


def load_model(run_id=model_config.run_id):
    """
    Loads a YOLOv8 model from the specified run directory.
    The runs are under the 'runs/detect' folder by default.
    """
    models_path = "runs/detect"

    best_model_path = Path(models_path) / run_id / "weights" / "best.pt"

    try:
        yolo_model = YOLO(str(best_model_path))
        logger.info(f"Loaded model from {best_model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {best_model_path}: {e}")
        raise e

    return yolo_model


if __name__ == "__main__":
    # This ensures the script runs when called from the command line
    parser = argparse.ArgumentParser(description="YOLOv8 Trainer and Predictor")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "validate_data"],
        required=True,
        help="Choose whether to train a model or run prediction",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        default=model_config.run_id,
        help="Specify the run ID to create or to load, defaults to the one in model_config.py",
    )

    args = parser.parse_args()
    if args.mode == "train":
        # Validate dataset

        train(run_id=args.run_id)
    elif args.mode == "predict":
        model = load_model(run_id=args.run_id)
        test_images = str(DATA_PATH / "processed" / "dataset_yolo" / "images" / "test")
        predict(model, test_images, run_id=args.run_id)
    elif args.mode == "validate_data":
        results = validate()
        logger.info(f"Validation results: {results}")
