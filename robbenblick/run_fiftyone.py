import argparse
import fiftyone as fo
from pathlib import Path
from ultralytics import YOLO

from robbenblick import DATA_PATH, logger, model_config


def fo_yolo_groundtruth_dataset(dataset_dir: Path, split: str, dataset_name: str):
    """
    Loads a processed YOLO dataset (images + ground truth labels) into FiftyOne.

    Args:
        dataset_dir: The root directory of the processed YOLO dataset
                     (e.g., .../processed/dataset_yolo)
        split (str): The split to load ("train", "val", or "test").
        dataset_name (str): The name for the FiftyOne dataset.
    """
    # data.yaml or dataset.yaml must be in the dataset_dir for this to work
    yaml_path_data = dataset_dir / "data.yaml"
    yaml_path_dataset = dataset_dir / "dataset.yaml"

    if not yaml_path_data.exists() and not yaml_path_dataset.exists():
        logger.error(f"No data.yaml or dataset.yaml found in {dataset_dir}. Run create_dataset.py first.")
        return None

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        dataset_dir=dataset_dir,
        # dataset_yaml parameter removed - relying on default detection
        split=split,
        name=dataset_name,
        label_field="ground_truth",  # Field for the loaded labels
    )
    return dataset


def fo_yolo_predictions_dataset(
        images_dir: str, run_id: str, dataset_name: str, classes=None
):
    """
    Loads YOLO test images, runs a trained model to get predictions,
    and visualizes them in FiftyOne.

    Args:
        images_dir (str): Path to directory with test/val images.
        run_id (str): The run_id of the trained model to use.
        dataset_name (str): The name for the FiftyOne dataset.
        classes (list, optional): List of class names. Defaults to None.
    """
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.ImageDirectory,
        dataset_dir=images_dir,
        name=dataset_name,
    )

    model_path = f"runs/detect/{run_id}/weights/best.pt"
    if not Path(model_path).exists():
        logger.error(f"Model file not found at {model_path}. Cannot run predictions.")
        return dataset  # Return dataset with images only

    model = YOLO(model_path)
    dataset.apply_model(
        model,
        label_field="predictions",
        confidence_thresh=model_config.confidence_thresh,
    )

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch FiftyOne visualization for YOLO ground truth or predictions."
    )
    parser.add_argument(
        "--dataset",
        choices=["groundtruth", "predictions"],  # Renamed from "cvat" and "yolo"
        default="groundtruth",
        required=False,
        help="Visualize 'groundtruth' data (output of create_dataset.py) or model 'predictions'.",
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
        default=model_config.run_id,
        help="Run ID to use for predictions or as dataset name.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the dataset if it exists.",
    )
    args = parser.parse_args()

    # Define the processed dataset directory
    yolo_dataset_dir = DATA_PATH / "processed" / "dataset_yolo"
    dataset = None
    if args.dataset == "predictions":
        # Mode 1: Visualize Model PREDICTIONS on a split
        dataset_name = f"{args.run_id}_{args.split}_predictions"
        if args.recreate and dataset_name in fo.list_datasets():
            logger.info(f"Deleting dataset '{dataset_name}' to recreate it.")
            fo.delete_dataset(dataset_name)

        image_dir = str(yolo_dataset_dir / "images" / args.split)

        if not Path(image_dir).exists():
            logger.error(f"Image directory not found: {image_dir}")
            exit()

        if dataset_name in fo.list_datasets():
            logger.info(f"Loading existing dataset '{dataset_name}'.")
            dataset = fo.load_dataset(dataset_name)
        else:
            logger.info(f"Creating new dataset '{dataset_name}' for predictions.")
            dataset = fo_yolo_predictions_dataset(
                images_dir=image_dir,
                run_id=args.run_id,
                dataset_name=dataset_name,
                classes=0
            )

    elif args.dataset == "groundtruth":
        # Mode 2: Visualize GROUND TRUTH of the processed dataset
        dataset_name = f"yolo_groundtruth_{args.split}"
        if args.recreate and dataset_name in fo.list_datasets():
            logger.info(f"Deleting dataset '{dataset_name}' to recreate it.")
            fo.delete_dataset(dataset_name)

        if dataset_name in fo.list_datasets():
            logger.info(f"Loading existing dataset '{dataset_name}'.")
            dataset = fo.load_dataset(dataset_name)
        else:
            logger.info(
                f"Dataset '{dataset_name}' not found. Creating new dataset from {yolo_dataset_dir}."
            )
            dataset = fo_yolo_groundtruth_dataset(
                dataset_dir=yolo_dataset_dir,
                split=args.split,
                dataset_name=dataset_name
            )

    if dataset:
        dataset.persistent = True
        session = fo.launch_app(dataset, port=5157, auto=False)
        logger.info(f"FiftyOne app launched. View at: {session.url}")
        session.wait()
    else:
        logger.error("Failed to load or create dataset.")
