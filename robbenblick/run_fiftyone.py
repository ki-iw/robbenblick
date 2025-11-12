import argparse
from pathlib import Path
import fiftyone as fo
from ultralytics import YOLO

from robbenblick import logger, CONFIG_PATH
from robbenblick.utils import load_config


def fo_yolo_groundtruth_dataset(dataset_dir: Path, split: str, dataset_name: str):
    """Loads a processed YOLO dataset (images + ground truth labels) into FiftyOne."""
    # data.yaml must be in the dataset_dir for this to work
    yaml_path_dataset = dataset_dir / "dataset.yaml"

    if not yaml_path_dataset.exists():
        logger.error(
            f"No dataset.yaml found in {dataset_dir}. Run create_dataset.py first."
        )
        return None

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,
        dataset_dir=dataset_dir,
        split=split,
        name=dataset_name,
        label_field="ground_truth",  # Field for the loaded labels
    )
    return dataset


def fo_yolo_predictions_dataset(
    images_dir: str,
    run_id: str,
    dataset_name: str,
    confidence_thresh: float,
):
    """
    Loads YOLO test images, runs a trained model to get predictions,
    and visualizes them in FiftyOne.
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

    logger.info(f"Loading model {model_path} with conf_thresh={confidence_thresh}")
    model = YOLO(model_path)
    dataset.apply_model(
        model,
        label_field="predictions",
        confidence_thresh=confidence_thresh,
    )

    return dataset


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
        choices=["groundtruth", "predictions"],
        default="groundtruth",
        required=False,
        help="Visualize 'groundtruth' data or model 'predictions'.",
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

    config_data = load_config(args.config)
    if config_data is None:
        exit(1)

    # run_id (CLI overwrites config)
    run_id = args.run_id if args.run_id is not None else config_data.get("run_id")
    if run_id is None:
        logger.error("No 'run_id' provided in CLI or config file.")
        exit(1)

    # dataset path
    dataset_output_dir_str = config_data.get("dataset_output_dir")
    if dataset_output_dir_str is None:
        logger.error("Config Error: 'dataset_output_dir' not defined in config file.")
        exit(1)

    yolo_dataset_dir = Path(dataset_output_dir_str)

    # confidence threshold (from config, with fallback)
    confidence_thresh = config_data.get("confidence_thresh", 0.25)

    dataset = None
    if args.dataset == "predictions":
        # Visualize Model PREDICTIONS on a split
        dataset_name = f"{run_id}_{args.split}_predictions"
        if args.recreate and dataset_name in fo.list_datasets():
            logger.info(f"Deleting dataset '{dataset_name}' to recreate it.")
            fo.delete_dataset(dataset_name)

        image_dir = str(yolo_dataset_dir / "images" / args.split)

        if not Path(image_dir).exists():
            logger.error(f"Image directory not found: {image_dir}")
            logger.error("Did you run 'create_dataset.py'?")
            exit()

        if dataset_name in fo.list_datasets():
            logger.info(f"Loading existing dataset '{dataset_name}'.")
            dataset = fo.load_dataset(dataset_name)
        else:
            logger.info(f"Creating new dataset '{dataset_name}' for predictions.")
            dataset = fo_yolo_predictions_dataset(
                images_dir=image_dir,
                run_id=run_id,
                dataset_name=dataset_name,
                confidence_thresh=confidence_thresh,
            )

    elif args.dataset == "groundtruth":
        # Visualize GROUND TRUTH of the processed dataset
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
                dataset_name=dataset_name,
            )

    if dataset:
        dataset.persistent = True
        session = fo.launch_app(dataset, port=5157, auto=False)
        logger.info(f"FiftyOne app launched. View at: {session.url}")
        session.wait()
    else:
        logger.error("Failed to load or create dataset.")


if __name__ == "__main__":
    main()
