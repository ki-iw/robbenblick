import argparse
from pathlib import Path
import fiftyone as fo
from ultralytics import YOLO

from robbenblick import logger, CONFIG_PATH
from robbenblick.utils import load_config


def fo_yolo_groundtruth_dataset(dataset_dir: Path, split: str, dataset_name: str):
    """Loads a processed YOLO dataset (images + ground truth labels) into FiftyOne."""
    yaml_path_dataset = dataset_dir / "dataset.yaml"
    if not yaml_path_dataset.exists():
        logger.error(
            f"No dataset.yaml found in {dataset_dir}. Run create_dataset.py first."
        )
        return None

    try:
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            dataset_dir=dataset_dir,
            split=split,
            name=dataset_name,
            label_field="ground_truth",  # Field for the loaded labels
        )
        return dataset
    except Exception as e:
        logger.error(f"Failed to load YOLO dataset from {dataset_dir}: {e}")
        return None


def fo_load_images_only(images_dir: str, dataset_name: str):
    """Loads only the images from a directory into FiftyOne."""
    if not Path(images_dir).exists():
        logger.error(f"Image directory not found: {images_dir}")
        logger.error("Did you run 'create_dataset.py'?")
        return None

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=images_dir,
        name=dataset_name,
    )
    return dataset


def add_predictions_to_dataset(
    dataset: fo.Dataset, run_id: str, confidence_thresh: float
):
    """Applies a YOLO model to an existing FiftyOne dataset."""
    model_path = f"runs/detect/{run_id}/weights/best.pt"
    if not Path(model_path).exists():
        logger.error(f"Model file not found at {model_path}. Cannot run predictions.")
        return  # Return without adding predictions

    logger.info(f"Loading model {model_path} with conf_thresh={confidence_thresh}")
    model = YOLO(model_path)
    dataset.apply_model(
        model,
        label_field="predictions",
        confidence_thresh=confidence_thresh,
    )
    logger.info("Predictions added successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch FiftyOne visualization for YOLO ground truth or predictions."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH / "base_config.yaml",
        help="Path to the central YAML config file.",
    )
    parser.add_argument(
        "--dataset",
        choices=["groundtruth", "predictions", "all"],
        default="all",
        required=False,
        help="Visualize 'groundtruth', model 'predictions', or 'all' (both).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Which data split to visualize (default: val).",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run ID to use for predictions or as dataset name (overrides config).",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the dataset if it exists.",
    )
    args = parser.parse_args()

    # Load Config and Paths
    config_data = load_config(args.config)
    if config_data is None:
        exit(1)

    run_id = args.run_id if args.run_id is not None else config_data.get("run_id")
    if args.dataset != "groundtruth" and run_id is None:
        logger.error(f"No 'run_id' provided. Required for mode '{args.dataset}'.")
        exit(1)

    dataset_output_dir_str = config_data.get("dataset_output_dir")
    if dataset_output_dir_str is None:
        logger.error("Config Error: 'dataset_output_dir' not defined in config file.")
        exit(1)

    yolo_dataset_dir = Path(dataset_output_dir_str)
    images_dir = str(yolo_dataset_dir / "images" / args.split)
    confidence_thresh = config_data.get("confidence_thresh", 0.25)

    # Determine Dataset Name and Handle Deletion
    if args.dataset == "groundtruth":
        dataset_name = f"yolo_groundtruth_{args.split}"
    elif args.dataset == "predictions":
        dataset_name = f"{run_id}_{args.split}_predictions"
    else:  # args.dataset == "all"
        dataset_name = f"{run_id}_{args.split}_evaluation"

    if args.recreate and dataset_name in fo.list_datasets():
        logger.info(f"Deleting dataset '{dataset_name}' to recreate it.")
        fo.delete_dataset(dataset_name)

    # Load or Create Base Dataset
    dataset = None
    if dataset_name in fo.list_datasets():
        logger.info(f"Loading existing dataset '{dataset_name}'.")
        dataset = fo.load_dataset(dataset_name)
    else:
        logger.info(f"Creating new dataset '{dataset_name}'.")
        if args.dataset == "predictions":
            # Load images only
            dataset = fo_load_images_only(images_dir, dataset_name)
        else:
            # Load images + ground truth (for "groundtruth" and "all")
            dataset = fo_yolo_groundtruth_dataset(
                dataset_dir=yolo_dataset_dir,
                split=args.split,
                dataset_name=dataset_name,
            )

    if dataset is None:
        logger.error("Failed to load or create base dataset. Exiting.")
        exit(1)

    # Add Predictions (if required)
    needs_predictions = args.dataset in ["predictions", "all"]
    has_predictions = "predictions" in dataset.get_field_schema()

    if needs_predictions and not has_predictions:
        logger.info("Adding predictions to dataset...")
        add_predictions_to_dataset(dataset, run_id, confidence_thresh)
    elif needs_predictions and has_predictions:
        logger.info("Dataset already contains predictions, skipping model run.")

    # Launch App
    dataset.persistent = True
    session = fo.launch_app(dataset, port=5157, auto=False)
    logger.info(f"FiftyOne app launched. View at: {session.url}")
    session.wait()


if __name__ == "__main__":
    main()
