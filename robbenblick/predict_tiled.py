import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import yaml
import pandas as pd

from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.predict import get_sliced_prediction

from robbenblick import logger, CONFIG_PATH
from robbenblick.utils import get_device, load_config


def run_inference_on_image(
        detection_model: Yolov8DetectionModel,
        image_path: Path,
        output_dir: Path,
        slice_size: int,
        overlap_ratio: float,
        save_visuals: bool,
        save_yolo: bool,
):
    """
    Runs sliced inference on a single image and exports the results.
    """
    try:
        # Run sliced inference
        result = get_sliced_prediction(
            image=str(image_path),
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
        )

        output_stem = image_path.stem

        # export results in YOLO format
        if save_yolo:
            yolo_out_path = output_dir / "labels" / f"{output_stem}.txt"
            yolo_out_path.parent.mkdir(parents=True, exist_ok=True)
            result.export_predictions(
                export_format="yolo",
                output_dir=str(yolo_out_path.parent),
                file_name=output_stem,
            )

        # export results as visualized image
        if save_visuals:
            visual_out_path = output_dir / "visuals"
            visual_out_path.mkdir(parents=True, exist_ok=True)
            result.export_visuals(
                export_dir=str(visual_out_path),
                file_name=output_stem,
            )

        # returns count of detected objects
        return len(result.object_prediction_list)

    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Run tiled (SAHI) inference with a trained YOLOv8 model."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH / "base_config.yaml",
        help="Path to the base YAML config file."
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Override the 'run_id' from the config file to select a specific model."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a single image or a directory of images."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the results (visuals, labels, and counts.csv)."
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="Overlap ratio (overrides 'tile_overlap' from config)."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold (overrides 'confidence_thresh' from config)."
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Save the original images with predictions drawn on them."
    )
    parser.add_argument(
        "--save-yolo",
        action="store_true",
        help="Save the predictions in YOLO .txt format."
    )

    args = parser.parse_args()

    config_data = load_config(args.config)
    if config_data is None:
        exit(1)

    run_id = args.run_id if args.run_id is not None else config_data.get('run_id')
    if not run_id:
        logger.error("No 'run_id' provided in CLI or config file.")
        return
    model_path = Path(f"runs/detect/{run_id}/weights/best.pt")
    logger.info(f"Using model from run: {run_id}")

    try:
        slice_size = config_data['yolo_hyperparams']['imgsz']
        logger.info(f"Using slice_size (from 'imgsz'): {slice_size}")
    except KeyError:
        logger.error("'yolo_hyperparams.imgsz' not found in config. Cannot set --slice-size.")
        return

    overlap_ratio = args.overlap_ratio if args.overlap_ratio is not None else config_data.get('tile_overlap', 0.2)
    logger.info(f"Using overlap_ratio: {overlap_ratio}")

    conf_thresh = args.conf if args.conf is not None else config_data.get('confidence_thresh', 0.25)
    logger.info(f"Using confidence_threshold: {conf_thresh}")

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return

    if not args.source.exists():
        logger.error(f"Source file or directory not found: {args.source}")
        return

    if not args.save_visuals and not args.save_yolo:
        logger.warning("No output format specified (use --save-visuals or --save-yolo). Only counts will be saved.")

    # load trained model
    device = get_device()
    detection_model = Yolov8DetectionModel(
        model_path=str(model_path),
        confidence_threshold=conf_thresh,
        device=device,
    )

    # find images
    if args.source.is_dir():
        image_paths = sorted(list(args.source.glob("*.jpg")) + \
                             list(args.source.glob("*.png")) + \
                             list(args.source.glob("*.jpeg")))
        logger.info(f"Found {len(image_paths)} images in {args.source}")
    elif args.source.is_file():
        image_paths = [args.source]
        logger.info(f"Processing single image: {args.source}")
    else:
        logger.error(f"Source is not a valid file or directory: {args.source}")
        return

    # inference
    total_detections = 0
    detection_counts = []

    for img_path in tqdm(image_paths, desc="Running inference"):
        num_dets = run_inference_on_image(
            detection_model=detection_model,
            image_path=img_path,
            output_dir=args.output_dir,
            slice_size=slice_size,
            overlap_ratio=overlap_ratio,
            save_visuals=args.save_visuals,
            save_yolo=args.save_yolo,
        )
        total_detections += num_dets

        logger.info(f"    -> {img_path.name}: {num_dets} objects found.")
        detection_counts.append({"image_name": img_path.name, "count": num_dets})

    logger.info("Inference complete! 🎉")
    logger.info(f"Total detections found: {total_detections}")

    if detection_counts:
        csv_path = args.output_dir / "detection_counts.csv"
        try:
            df = pd.DataFrame(detection_counts)
            df.to_csv(csv_path, index=False)
            logger.info(f"Detection counts saved to: {csv_path}")
        except Exception as e:
            logger.error(f"Failed to save detection counts CSV: {e}")
    else:
        logger.warning("No images were processed or no detections were found.")

    logger.info(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()