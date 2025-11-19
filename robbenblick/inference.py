from pathlib import Path
from sahi.models.ultralytics import UltralyticsDetectionModel
from sahi.predict import get_sliced_prediction

from robbenblick import logger
from robbenblick.utils import get_device


def load_detection_model(
    model_path: Path, conf_thresh: float
) -> UltralyticsDetectionModel | None:
    """
    Loads and caches the YOLOv8 model with SAHI.
    """
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    try:
        device = get_device()
        detection_model = UltralyticsDetectionModel(
            model_path=str(model_path),
            confidence_threshold=conf_thresh,
            device=device,
        )
        logger.info(f"Model {model_path} successfully loaded on {device}.")
        return detection_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def run_inference(
    detection_model: UltralyticsDetectionModel,
    image_path: Path,
    output_dir: Path,
    slice_size: int,
    overlap_ratio: float,
    save_visuals: bool = True,
    hide_labels: bool = False,
    hide_conf: bool = False,
) -> dict:
    """
    Runs sliced inference on a single image and returns structured results.

    Args:
        detection_model: The loaded SAHI detection model.
        image_path: Path to the source image.
        output_dir: Root directory to save results.
        slice_size: Size of the slices.
        overlap_ratio: Overlap ratio for slices.
        save_visuals: Whether to save the visualization.
        hide_labels: Hide class labels on visualization.
        hide_conf: Hide confidence scores on visualization.

    Returns:
        A dictionary containing:
            - "count" (int): Number of detections.
            - "visual_path" (Path | None): Path to the saved visualization.
            - "sahi_result" (PredictionResult): The raw SAHI result object.
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

        num_detections = len(result.object_prediction_list)
        visual_result_path = None
        output_stem = image_path.stem

        # Save visualized results if requested
        if save_visuals:
            visual_out_path = output_dir / "visuals"
            visual_out_path.mkdir(parents=True, exist_ok=True)
            visual_result_path = visual_out_path / f"{output_stem}.png"

            result.export_visuals(
                export_dir=str(visual_out_path),
                file_name=output_stem,
                hide_labels=hide_labels,
                hide_conf=hide_conf,
                rect_th=1,
            )

            # Fallback check in case SAHI saved a different extension
            if not visual_result_path.exists():
                visual_result_path = next(
                    visual_out_path.glob(f"{output_stem}.*"), None
                )

            if not visual_result_path:
                logger.error(f"Could not find visualized file for {output_stem}.")

    except Exception as e:
        logger.error(f"Error processing {image_path.name}: {e}")
        return {
            "count": 0,
            "visual_path": None,
            "sahi_result": None,
        }

    return {
        "count": num_detections,
        "visual_path": visual_result_path,
        "sahi_result": result,
    }
