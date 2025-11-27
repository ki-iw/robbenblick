import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from robbenblick import logger, CONFIG_PATH
from robbenblick.utils import load_config
from robbenblick.inference import load_detection_model, run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Run tiled (SAHI) inference with a trained YOLOv8 model."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH / "base_config.yaml",
        help="Path to the base YAML config file.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Override the 'run_id' from the config file to select a specific model.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the results (visuals, labels, and counts.csv).",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="Overlap ratio (overrides 'tile_overlap' from config).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (overrides 'confidence_thresh' from config).",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save the original images with predictions drawn on them.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    source_path = Path(args.source)
    output_dir_path = Path(args.output_dir)

    config_data = load_config(config_path)
    if config_data is None:
        exit(1)

    run_id = args.run_id if args.run_id is not None else config_data.get("run_id")
    if not run_id:
        logger.error("No 'run_id' provided in CLI or config file.")
        return
    model_path = Path(f"runs/detect/{run_id}/weights/best.pt")
    logger.info(f"Using model from run: {run_id}")

    try:
        slice_size = config_data["yolo_hyperparams"]["imgsz"]
        logger.info(f"Using slice_size (from 'imgsz'): {slice_size}")
    except KeyError:
        logger.error(
            "'yolo_hyperparams.imgsz' not found in config. Cannot set --slice-size."
        )
        return

    overlap_ratio = (
        args.overlap_ratio
        if args.overlap_ratio is not None
        else config_data.get("tile_overlap", 0.2)
    )
    logger.info(f"Using overlap_ratio: {overlap_ratio}")

    conf_thresh = (
        args.conf
        if args.conf is not None
        else config_data.get("confidence_thresh", 0.25)
    )
    logger.info(f"Using confidence_threshold: {conf_thresh}")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    if not source_path.exists():
        logger.error(f"Source file or directory not found: {source_path}")
        return

    logger.info("Loading model...")
    detection_model = load_detection_model(
        model_path=model_path, conf_thresh=conf_thresh
    )
    if detection_model is None:
        logger.error("Failed to load model. Exiting.")
        return
    logger.info("Model loaded.")

    # find images
    image_paths = []
    if source_path.is_dir():
        supported_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}
        logger.info(f"Scanning for images in {source_path}...")

        for file_path in source_path.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                image_paths.append(file_path)

        image_paths.sort()
        logger.info(f"Found {len(image_paths)} images in {source_path}")
    elif source_path.is_file():
        image_paths = [source_path]
        logger.info(f"Processing single image: {source_path}")
    else:
        logger.error(f"Source is not a valid file or directory: {source_path}")
        return

    if not image_paths:
        logger.warning(
            f"No images with extensions .jpg/.jpeg/.png found at {source_path}. Aborting."
        )
        return

    # inference
    total_detections = 0
    detection_counts = []

    for img_path in tqdm(image_paths, desc="Running inference"):
        result_data = run_inference(
            detection_model=detection_model,
            image_path=img_path,
            output_dir=output_dir_path,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            save_visuals=args.save_visuals,
            hide_labels=False,
            hide_conf=False,
        )

        num_dets = result_data["count"]
        total_detections += num_dets

        logger.info(f"    -> {img_path.name}: {num_dets} objects found.")
        detection_counts.append({"image_name": img_path.name, "count": num_dets})

    logger.info("Inference complete! 🎉")
    logger.info(f"Total detections found: {total_detections}")

    if detection_counts:
        csv_path = output_dir_path / "detection_counts.csv"
        try:
            df = pd.DataFrame(detection_counts)
            df.to_csv(csv_path, index=False)
            logger.info(f"Detection counts saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save detection counts CSV: {e}")
    else:
        logger.warning("No images were processed or no detections were found.")

    logger.info(f"All results saved to: {output_dir_path}")


if __name__ == "__main__":
    main()
