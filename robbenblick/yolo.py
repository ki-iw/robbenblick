import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

from robbenblick import logger, CONFIG_PATH
from robbenblick.utils import load_config, get_device

# Define the correct path to the YAML file
BASE_CONFIG_PATH = CONFIG_PATH / "base_config.yaml"


def train(run_id, model_name, hyp_dict: dict, dataset_yaml_path, project_dir: Path = None, freeze_layers=None):
    """
    Detects the best available device and trains a YOLOv8 model
    using hyperparameters from a config file.
    """
    # Detects the best available device.
    device = get_device()

    try:
        # Load a pretrained YOLOv8 model
        model = YOLO(model_name)

        if freeze_layers is not None:
            logger.info(f"Freezing the first {freeze_layers} layers for training.")

        logger.info(f"Starting training model {model_name}, run_id: {run_id}.")

        # Train the model using the data.yaml and the hyperparameter.yaml
        model.train(
            data=str(dataset_yaml_path),
            device=device,
            name=run_id,
            project=project_dir,
            task="detect",
            freeze=freeze_layers,
            exist_ok=True,
            **hyp_dict
        )

        logger.info("Training complete! 🎉")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}")


def validate_on_test_set(run_id, dataset_yaml_path):
    """
    Validates the trained model on the 'test' split defined in dataset.yaml.
    """
    model_path = Path("runs/detect") / run_id / "weights" / "best.pt"
    if not model_path.exists():
        logger.error(f"Cannot validate: Model file not found at {model_path}")
        return None

    model = YOLO(str(model_path))

    # Run validation specifically on the 'test' split
    # This requires 'test: images/test' to be defined in dataset.yaml
    try:
        logger.info(f"Validating model {run_id} on TEST split...")
        result = model.val(data=str(dataset_yaml_path), split="test", plots=False)
        return result
    except Exception as e:
        logger.error(f"Error during validation on test split: {e}")
        logger.warning("Ensure your dataset.yaml has a 'test:' entry pointing to 'images/test'.")
        return None


def predict(model: YOLO, images_dir: str, run_id) -> None:
    """
    Runs YOLOv8 detection inference and saves predictions to text files.

    Args:
        model (YOLO): A trained YOLOv8 model instance from the ultralytics library.
        images_dir (str): The path to the directory containing images for prediction.
        run_id (str): Identifier for the prediction run, used to name the output directory.
    """
    # Run inference on the specified directory
    # 'save_txt=True' saves predictions in YOLO format
    results: list[Results] = model.predict(
        source=images_dir,
        save_txt=True,
        save=False,
        conf=0.4,  # Optional: Set a confidence threshold
        name=run_id + "_predict",  # Name of the folder to save predictions
        exist_ok=True,  # Overwrite existing predict directory
    )

    if results:
        logger.info(f"✅ Predictions successfully saved to: {results[0].save_dir}")
    else:
        logger.warning(f"Prediction ran, but no results were returned. Is {images_dir} empty?")


def load_model(run_id: str):
    """
    Loads a YOLOv8 model from the specified run directory.
    Runs are under the 'runs/detect' folder by default.
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
    parser = argparse.ArgumentParser(description="YOLOv8 Trainer and Predictor")

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH / "base_config.yaml",
        help="Path to the central YAML config file."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "validate"],
        default="train",
        help="Choose 'train' to train, 'predict' to run inference, or 'validate' to test on the test-split.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Override the run ID from the config file.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="For 'predict' mode: choose which split to run inference on (default: test).",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Override the 'freeze' setting from the config file."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default='runs/detect',
        help="Explicit project directory to save runs (e.g., 'runs/detect')."
    )

    args = parser.parse_args()

    config_data = load_config(args.config)
    if config_data is None:
        exit(1)

    run_id = args.run_id if args.run_id is not None else config_data.get('run_id')
    if run_id is None:
        logger.error("No 'run_id' provided in CLI or config file.")
        exit(1)

    freeze_layers = args.freeze if args.freeze is not None else config_data.get('freeze')

    model_name = config_data.get('model')
    if model_name is None:
        logger.error("No 'model' defined in config file.")
        exit(1)

    dataset_output_dir_str = config_data.get('dataset_output_dir')
    if dataset_output_dir_str is None:
        logger.error("Config Error: 'dataset_output_dir' not defined in config file.")
        exit(1)

    dataset_yaml_path = Path(dataset_output_dir_str) / 'dataset.yaml'

    if not dataset_yaml_path.exists():
        logger.error(f"'dataset.yaml' not found at expected path: {dataset_yaml_path}")
        logger.error("Did you run 'create_dataset.py' (Source 1) first?")
        exit(1)
    else:
        logger.info(f"Using dataset configuration: {dataset_yaml_path}")

    yolo_hyp_data = config_data.get('yolo_hyperparams')
    if yolo_hyp_data is None:
        logger.error("Config Error: 'yolo_hyperparams' Sektion nicht in config file gefunden.")
        exit(1)

    if args.mode == "train":
        train(
            run_id=run_id,
            model_name=model_name,
            hyp_dict=yolo_hyp_data,
            dataset_yaml_path=dataset_yaml_path,
            project_dir=args.project_dir,
            freeze_layers=freeze_layers
        )

    elif args.mode == "predict":
        model = load_model(run_id=run_id)
        images_path = Path(dataset_output_dir_str) / "images" / args.split
        if not images_path.exists():
            logger.error(f"Prediction images directory not found: {images_path}")
        else:
            logger.info(f"Running prediction on split: {args.split}")
            predict(model, str(images_path), run_id=run_id)

    elif args.mode == "validate":

        results = validate_on_test_set(run_id=run_id, dataset_yaml_path=dataset_yaml_path)
        if results:
            logger.info("Validation results on TEST split:")
            logger.info(f"  mAP50-95: {results.box.map:.4f}")
            logger.info(f"  mAP50:    {results.box.map50:.4f}")
            logger.info(f"  mAP75:    {results.box.map75:.4f}")
