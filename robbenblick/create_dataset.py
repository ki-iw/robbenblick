import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import collections
import math

import numpy as np
import cv2
import yaml
from tqdm import tqdm
import pandas as pd

from robbenblick import DATA_PATH, CONFIG_PATH, logger
from robbenblick.utils import load_config

# Default paths for the command-line arguments
DEFAULT_RAW_DIR = DATA_PATH / "raw"
DEFAULT_CONFIG_PATH = CONFIG_PATH / "base_config.yaml"


def parse_cvat_xml(xml_file):
    """Parses a CVAT XML 1.1 file and extracts annotations."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = {}
    class_names = set()

    # Read task name
    task_name_tag = root.find("meta/task/name")
    task_name = task_name_tag.text if task_name_tag is not None else "Unknown Task"

    for image_tag in root.findall("image"):
        image_name = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))

        polygons = []

        # Find both <polyline> and <polygon> tags.
        # CVAT exports can contain both polyline and polygon tags depending on annotation tool used.
        annotation_tags = image_tag.findall("polyline")
        annotation_tags.extend(image_tag.findall("polygon"))

        for poly_tag in annotation_tags:
            label = poly_tag.get("label")
            class_names.add(label)
            points_str = poly_tag.get("points")
            # Convert points string to a list of (x, y) tuples
            points = [tuple(map(float, p.split(","))) for p in points_str.split(";")]
            polygons.append({"label": label, "points": points})

        annotations[image_name] = {
            "width": width,
            "height": height,
            "polygons": polygons,
        }

    return annotations, sorted(list(class_names)), task_name


def polygon_to_bbox(points):
    """Converts a list of polygon points to a bounding box [xmin, ymin, xmax, ymax]."""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    xmin = min(x_coords)
    ymin = min(y_coords)
    xmax = max(x_coords)
    ymax = max(y_coords)
    return [xmin, ymin, xmax, ymax]


def convert_to_yolo_format(bbox, img_width, img_height):
    """Converts a bounding box [xmin, ymin, xmax, ymax] to YOLO format."""

    # Failsafe Check: Prevent division by zero if img_width or img_height is 0
    if img_width <= 0 or img_height <= 0:
        logger.error(
            f"Invalid image dimensions encountered in conversion: w={img_width}, h={img_height}"
        )
        return None

    dw = 1.0 / img_width
    dh = 1.0 / img_height
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x_center_norm = x_center * dw
    width_norm = width * dw
    y_center_norm = y_center * dh
    height_norm = height * dh

    result = (x_center_norm, y_center_norm, width_norm, height_norm)

    # Failsafe Check:
    # Ensure no NaN or inf values are returned
    # Also ensures values are within the valid [0.0, 1.0] range
    if any(math.isnan(v) or math.isinf(v) or v < 0.0 or v > 1.0 for v in result):
        logger.warning(
            f"Generated invalid YOLO coords (NaN/inf or out of bounds): {result}"
        )
        return None

    return result


def find_labels_for_tile(img_annotations, tile_coords, tile_dims, class_to_id):
    """Finds all annotations within a tile and converts them to YOLO format."""
    tile_x_min, tile_y_min, tile_x_max, tile_y_max = tile_coords
    tile_w, tile_h = tile_dims
    tile_labels = []

    for poly_info in img_annotations["polygons"]:
        bbox = polygon_to_bbox(poly_info["points"])

        # Check for intersection between bbox and tile
        intersect_x_min = max(bbox[0], tile_x_min)
        intersect_y_min = max(bbox[1], tile_y_min)
        intersect_x_max = min(bbox[2], tile_x_max)
        intersect_y_max = min(bbox[3], tile_y_max)

        if intersect_x_min < intersect_x_max and intersect_y_min < intersect_y_max:
            # Convert intersection coords to tile-local coords
            local_bbox = [
                intersect_x_min - tile_x_min,
                intersect_y_min - tile_y_min,
                intersect_x_max - tile_x_min,
                intersect_y_max - tile_y_min,
            ]

            # Skip invalid or tiny bboxes
            if (local_bbox[2] <= local_bbox[0]) or (local_bbox[3] <= local_bbox[1]):
                continue
            if (local_bbox[2] - local_bbox[0]) * (local_bbox[3] - local_bbox[1]) < (
                tile_h * tile_w * 0.0001
            ):
                continue

            # Convert to YOLO format
            yolo_bbox = convert_to_yolo_format(local_bbox, tile_w, tile_h)

            # Failsafe Check: Only add the label if the conversion was successful and returned valid coordinates
            if yolo_bbox:
                class_id = class_to_id[poly_info["label"]]
                tile_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")

    return tile_labels


def save_tile(tile_img, tile_labels, base_filename, img_output_dir, label_output_dir):
    """Saves a single tile image and its corresponding label file."""
    img_save_path = img_output_dir / f"{base_filename}.png"
    label_save_path = label_output_dir / f"{base_filename}.txt"

    cv2.imwrite(str(img_save_path), tile_img)

    with open(label_save_path, "w") as f:
        f.write("\n".join(tile_labels))


def tile_one_image(
    unique_key,
    annotations,
    class_to_id,
    img_output_dir,
    label_output_dir,
    tile_size,
    tile_overlap,
    save_only_with_labels,
):
    """Loads a single image, tiles it, and processes annotations for each tile."""
    img_annotations = annotations[unique_key]
    image_path = img_annotations["full_image_path"]

    if not image_path.exists():
        logger.warning(f"Image file not found for {image_path.name}, skipping.")
        return

    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Could not read image {image_path.name}, skipping.")
        return

    img_h, img_w, _ = img.shape
    step_size = int(tile_size * (1 - tile_overlap))

    for y in range(0, img_h, step_size):
        for x in range(0, img_w, step_size):
            tile_x_min, tile_y_min = x, y
            tile_x_max, tile_y_max = (
                x + tile_size,
                y + tile_size,
            )

            tile_img = img[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
            tile_h, tile_w, _ = tile_img.shape

            # Calculate 25% of tile size, but enforce a minimum of 2 pixels
            # to prevent division-by-zero errors (NaN bug).
            min_allowed_dim = max(2, int(tile_size * 0.25))

            if tile_h < min_allowed_dim or tile_w < min_allowed_dim:
                continue

            tile_coords = (tile_x_min, tile_y_min, tile_x_max, tile_y_max)
            tile_dims = (tile_w, tile_h)

            tile_labels = find_labels_for_tile(
                img_annotations, tile_coords, tile_dims, class_to_id
            )

            # Only save tiles that have labels (or if config allows empty)
            should_save = tile_labels or not save_only_with_labels
            if should_save:
                base_filename = f"{Path(unique_key).stem}_tile_{y}_{x}"
                save_tile(
                    tile_img,
                    tile_labels,
                    base_filename,
                    img_output_dir,
                    label_output_dir,
                )


def process_images(
    image_files,
    annotations,
    class_to_id,
    split,
    output_dir,
    tile_size,
    tile_overlap,
    save_only_with_labels,
):
    """Tiles images and converts annotations for a given split (train/val/test)."""
    logger.info(f"Processing {split} set ({len(image_files)} images)...")

    img_output_dir = output_dir / "images" / split
    label_output_dir = output_dir / "labels" / split

    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for unique_key in tqdm(image_files, desc=f"Tiling {split} images"):
        tile_one_image(
            unique_key,
            annotations,
            class_to_id,
            img_output_dir,
            label_output_dir,
            tile_size,
            tile_overlap,
            save_only_with_labels,
        )


def create_yaml_file(class_names, has_test_set, output_dir):
    """Creates the dataset.yaml file for YOLO training."""
    yaml_content = {
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if has_test_set:
        yaml_content["test"] = "images/test"

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    logger.info(f"Created dataset.yaml at {yaml_path}")
    logger.info("Dataset creation complete! 🎉")


def load_and_prepare_data(raw_dir):
    logger.info(f"Scanning for datasets in {raw_dir}...")

    all_annotations = {}
    dataset_batches = []
    source_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    if not source_dirs:
        logger.error(f"No dataset subdirectories found in {raw_dir}. Exiting.")
        return None, None

    for src_dir in source_dirs:
        logger.info(f"Scanning dataset: {src_dir.name}")
        xml_file = src_dir / "annotations.xml"
        image_dir = src_dir / "images"

        if not xml_file.exists():
            logger.warning(f"Skipping {src_dir.name}: 'annotations.xml' not found.")
            continue
        if not image_dir.is_dir():
            logger.warning(f"Skipping {src_dir.name}: 'images' directory not found.")
            continue

        annotations_batch, class_names_batch, task_name = parse_cvat_xml(xml_file)

        # Generate ground truth CSV-file
        ground_truth_counts = []
        for image_name, data in annotations_batch.items():
            # Count number of polygons (objects) in this image
            count = len(data.get("polygons", []))
            ground_truth_counts.append({"image_name": image_name, "count": count})

        if ground_truth_counts:
            # Sort by image name for consistency
            ground_truth_counts.sort(key=lambda x: x["image_name"])
            try:
                df = pd.DataFrame(ground_truth_counts)
                # Store CSV in the source directory
                csv_path = src_dir / "ground_truth_counts.csv"
                df.to_csv(csv_path, index=False)
                logger.info(
                    f"Saved ground truth counts for {src_dir.name} to {csv_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to save ground truth counts CSV for {src_dir.name}: {e}"
                )
        else:
            logger.warning(f"No ground truth counts to save for {src_dir.name}.")

        processed_annotations = {}
        for image_name, data in annotations_batch.items():
            unique_key = f"{src_dir.name}_{image_name}"
            data["full_image_path"] = image_dir / image_name
            if unique_key in all_annotations:
                logger.warning(
                    f"Duplicate unique key '{unique_key}' found. Overwriting previous entry."
                )

            processed_annotations[unique_key] = data
            all_annotations[unique_key] = data  # Global dict for process_images

        dataset_batches.append(
            {
                "name": src_dir.name,
                "task_name": task_name,
                "annotations": processed_annotations,
                "class_names": class_names_batch,
            }
        )

    return dataset_batches, all_annotations


def print_dataset_statistics(dataset_batches):
    logger.info("=" * 30)
    logger.info("--- 📊 DATASET STATISTICS ---")

    all_class_names = set()
    total_annotations_count = 0
    total_image_count = 0
    overall_class_distribution = collections.defaultdict(int)

    for i, batch in enumerate(dataset_batches):
        batch_class_distribution = collections.defaultdict(int)
        num_annotations_in_batch = 0

        for data in batch["annotations"].values():
            num_polygons = len(data["polygons"])
            num_annotations_in_batch += num_polygons
            all_class_names.update(batch["class_names"])
            for poly in data["polygons"]:
                label = poly["label"]
                batch_class_distribution[label] += 1
                overall_class_distribution[label] += 1

        num_images_in_batch = len(batch["annotations"])
        total_image_count += num_images_in_batch
        total_annotations_count += num_annotations_in_batch

        logger.info(f"--- Dataset #{i + 1}: {batch['name']} ---")
        logger.info(f"  Task Name: {batch['task_name']}")
        logger.info(f"  Images found: {num_images_in_batch}")
        logger.info(f"  Total annotations: {num_annotations_in_batch}")
        logger.info("  Class distribution:")
        for label, count in sorted(batch_class_distribution.items()):
            logger.info(f"    - {label}: {count}")

        # statistics for bbox dimensions and areas
        box_heights = []
        box_widths = []
        for data in batch["annotations"].values():
            for poly in data["polygons"]:
                bbox = polygon_to_bbox(poly["points"])
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                box_widths.append(width)
                box_heights.append(height)

        if box_heights:  # Check if the list is not empty
            heights_np = np.array(box_heights)
            widths_np = np.array(box_widths)
            areas_np = heights_np * widths_np

            logger.info("  BBox Dimension Statistics (in pixels):")
            logger.info(f"    Average Height: {np.mean(heights_np):.2f} (px)")
            logger.info(f"    Average Width:  {np.mean(widths_np):.2f} (px)")

            logger.info("  BBox Area Statistics (in pixels^2):")
            logger.info(f"    Median (50%): {np.median(areas_np):.2f}")
            logger.info(f"    25% Quantile: {np.quantile(areas_np, 0.25):.2f}")
            logger.info(f"    75% Quantile: {np.quantile(areas_np, 0.75):.2f}")
            logger.info(
                f"    Min / Max: {np.min(areas_np):.2f} / {np.max(areas_np):.2f}"
            )

        logger.info("-" * 20)

    logger.info("--- OVERALL DATASET STATISTICS ---")
    logger.info(f"Total directories processed: {len(dataset_batches)}")
    logger.info(f"Total unique images: {total_image_count}")
    logger.info(f"Total annotations (all images): {total_annotations_count}")
    class_names = sorted(list(all_class_names))
    logger.info(f"Total unique classes: {len(class_names)} ({', '.join(class_names)})")
    logger.info("Overall class distribution:")
    for label, count in sorted(overall_class_distribution.items()):
        logger.info(f"  - {label}: {count}")
    logger.info("=" * 30)

    return class_names


def split_data(dataset_batches, train_indices, val_indices, test_indices):
    """
    Splits data strictly by dataset indices provided by the user.
    Ensures zero leakage between datasets (e.g. for counting tasks).

    Args:
        dataset_batches: List of dataset dictionaries.
        train_indices: List of 1-based indices for the training set.
        val_indices: List of 1-based indices for the validation set.
        test_indices: List of 1-based indices for the test set.
    """
    train_files = []
    val_files = []
    test_files = []

    # Validate indices boundaries
    total_batches = len(dataset_batches)
    # Filter out None values if indices weren't provided
    train_indices = train_indices or []
    val_indices = val_indices or []
    test_indices = test_indices or []

    all_input_indices = train_indices + val_indices + test_indices

    if not all_input_indices:
        logger.error("No indices provided for splitting.")
        return None, None, None

    if max(all_input_indices) > total_batches or min(all_input_indices) < 1:
        logger.error(f"Indices must be between 1 and {total_batches}.")
        return None, None, None

    # Check for overlaps (crucial for strict separation)
    train_set, val_set, test_set = (
        set(train_indices),
        set(val_indices),
        set(test_indices),
    )
    if (
        not train_set.isdisjoint(val_set)
        or not train_set.isdisjoint(test_set)
        or not val_set.isdisjoint(test_set)
    ):
        logger.error(
            "Overlap detected between Train/Val/Test indices! A dataset cannot belong to multiple splits."
        )
        return None, None, None

    logger.info(f"Splitting {total_batches} datasets strictly by folder...")

    # Iterate through batches and assign based on index
    for i, batch in enumerate(dataset_batches):
        current_idx = i + 1  # Use 1-based index for user friendliness
        keys = list(batch["annotations"].keys())
        batch_name = batch["name"]

        if current_idx in train_indices:
            train_files.extend(keys)
            logger.info(f"  [TRAIN] Dataset #{current_idx} ({batch_name})")
        elif current_idx in val_indices:
            val_files.extend(keys)
            logger.info(f"  [VAL]   Dataset #{current_idx} ({batch_name})")
        elif current_idx in test_indices:
            test_files.extend(keys)
            logger.info(f"  [TEST]  Dataset #{current_idx} ({batch_name})")
        else:
            logger.warning(
                f"  [UNUSED] Dataset #{current_idx} ({batch_name}) is not assigned to any split."
            )

    logger.info(
        f"Split complete: {len(train_files)} TRAIN, {len(val_files)} VAL, {len(test_files)} TEST images."
    )
    return train_files, val_files, test_files


def _get_split_counts(file_keys, all_annotations):
    """
    Helper to count images and total polygons (seals) in a list of keys.
    """
    num_imgs = len(file_keys)
    num_anns = 0
    for key in file_keys:
        data = all_annotations.get(key)
        if data:
            num_anns += len(data.get("polygons", []))
    return num_imgs, num_anns


def print_split_statistics(train_files, val_files, test_files, all_annotations):
    logger.info("=" * 50)
    logger.info("--- FINAL SPLIT STATISTICS ---")

    # 1. Calculate raw counts
    n_train_img, n_train_ann = _get_split_counts(train_files, all_annotations)
    n_val_img, n_val_ann = _get_split_counts(val_files, all_annotations)
    n_test_img, n_test_ann = _get_split_counts(test_files, all_annotations)

    total_ann = n_train_ann + n_val_ann + n_test_ann
    total_img = n_train_img + n_val_img + n_test_img

    # 2. Print formatted lines (aligned with Auto-Recommendation style)
    def print_line(label, n_img, n_ann):
        pct = (n_ann / total_ann * 100) if total_ann > 0 else 0
        # Format: Label | X seals (Y%) | Z images
        logger.info(
            f"  {label:<6} | {n_ann:>6} seals ({pct:>5.1f}%) | {n_img:>5} images"
        )

    print_line("TRAIN", n_train_img, n_train_ann)
    print_line("VAL", n_val_img, n_val_ann)
    print_line("TEST", n_test_img, n_test_ann)

    logger.info("-" * 50)
    logger.info(f"  TOTAL  | {total_ann:>6} seals (100.0%) | {total_img:>5} images")
    logger.info("=" * 50)

    # 3. Critical Warnings
    if n_val_img > 0 and n_val_ann == 0:
        logger.warning("CRITICAL WARNING")
        logger.warning(
            "Your VALIDATION set contains 0 seals. Training metrics will be NaN/useless."
        )
        logger.warning("Please choose a different validation split.")


def suggest_split_indices(
    dataset_batches, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
):
    """
    Calculates a recommended split based on ANNOTATION COUNTS (polygons)
    to ensure the model sees roughly 80% of the seals in training.

    Uses a greedy algorithm sorting by annotation density.
    """
    batch_info = []
    total_annotations = 0
    total_images_global = 0

    # 1. Analyze contents
    for i, batch in enumerate(dataset_batches):
        n_imgs = len(batch["annotations"])
        n_anns = 0
        for data in batch["annotations"].values():
            n_anns += len(data["polygons"])

        batch_info.append(
            {"id": i + 1, "n_imgs": n_imgs, "n_anns": n_anns, "name": batch["name"]}
        )
        total_annotations += n_anns
        total_images_global += n_imgs

    # 2. Sort descending by Annotation count (priority) and then Image count
    batch_info.sort(key=lambda x: (x["n_anns"], x["n_imgs"]), reverse=True)

    # 3. Calculate targets (based on Annotations)
    target_train = total_annotations * train_ratio
    target_val = total_annotations * val_ratio
    target_test = total_annotations * test_ratio

    # Current accumulators
    c_train = {"anns": 0, "imgs": 0, "ids": []}
    c_val = {"anns": 0, "imgs": 0, "ids": []}
    c_test = {"anns": 0, "imgs": 0, "ids": []}

    for batch in batch_info:
        # Calculate deficits based on ANNOTATIONS
        def_train = target_train - c_train["anns"]
        def_val = target_val - c_val["anns"]
        def_test = target_test - c_test["anns"]

        # Assign to the bucket with the largest need for annotations
        # Priority order to prevent starvation of smaller sets: Test > Val > Train
        if def_test >= def_val and def_test >= def_train:
            target = c_test
        elif def_val >= def_train and def_val >= def_test:
            target = c_val
        else:
            target = c_train

        # Update selected bucket
        target["ids"].append(batch["id"])
        target["anns"] += batch["n_anns"]
        target["imgs"] += batch["n_imgs"]

    # Sort IDs strictly for display
    c_train["ids"].sort()
    c_val["ids"].sort()
    c_test["ids"].sort()

    return {
        "train": c_train,
        "val": c_val,
        "test": c_test,
        "total_anns": total_annotations,
        "total_imgs": total_images_global,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create YOLO dataset from CVAT XML with strict folder splitting."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the central YAML config file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in statistics-only mode. No files will be written.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Base directory containing raw dataset subfolders (default: {DEFAULT_RAW_DIR})",
    )

    # Optional arguments (checked manually later depending on dry-run state)
    parser.add_argument(
        "--train-indices",
        nargs="+",
        type=int,
        help="List of 1-based dataset indices to use for TRAINING (e.g. --train-indices 1 2 3)",
    )
    parser.add_argument(
        "--val-indices",
        nargs="+",
        type=int,
        help="List of 1-based dataset indices to use for VALIDATION (e.g. --val-indices 4)",
    )
    parser.add_argument(
        "--test-indices",
        nargs="+",
        type=int,
        default=[],
        help="List of 1-based dataset indices to use for TESTING (e.g. --test-indices 5)",
    )

    args = parser.parse_args()

    # --- MANUAL VALIDATION ---
    # Enforce indices only if NOT in dry-run mode (or if user wants to see split stats in dry-run)
    # If we are NOT in dry-run, we absolutely need train and val indices.
    if not args.dry_run:
        if not args.train_indices or not args.val_indices:
            parser.error(
                "The following arguments are required (unless --dry-run is used): --train-indices, --val-indices"
            )

    config = load_config(args.config)
    if config is None:
        exit(1)

    tile_size = config.get("imgsz")
    tile_overlap = config.get("tile_overlap")
    save_only_with_labels = config.get("save_only_with_labels")

    if tile_size is None or tile_overlap is None:
        logger.error("Missing tiling parameters in config (imgsz, tile_overlap).")
        return

    logger.info(f"Tile size: {tile_size}, Tile overlap: {tile_overlap}")

    # Load Data
    dataset_batches, all_annotations = load_and_prepare_data(args.raw_dir)
    if dataset_batches is None:
        return

    # Print Statistics
    # This shows the user which ID belongs to which folder
    class_names = print_dataset_statistics(dataset_batches)

    # --- LOGIC FOR SPLIT ---
    # If in dry-run and NO indices provided, suggest a split
    if args.dry_run and not (args.train_indices or args.val_indices):
        logger.info("=" * 40)
        logger.info("--- 🤖 AUTO-RECOMMENDATION (Balanced by Annotations) ---")

        s = suggest_split_indices(dataset_batches)

        t_anns = s["total_anns"]

        # Helper to format string
        def fmt(part, label):
            p_anns = (part["anns"] / t_anns * 100) if t_anns else 0
            return (
                f"  {label}: Indices {part['ids']} | "
                f"{part['anns']} seals ({p_anns:.1f}%) | "
                f"{part['imgs']} imgs"
            )

        logger.info(fmt(s["train"], "TRAIN"))
        logger.info(fmt(s["val"], "VAL  "))
        logger.info(fmt(s["test"], "TEST "))

        # Construct command string
        cmd_str = f"python create_dataset.py --train-indices {' '.join(map(str, s['train']['ids']))} --val-indices {' '.join(map(str, s['val']['ids']))}"
        if s["test"]["ids"]:
            cmd_str += f" --test-indices {' '.join(map(str, s['test']['ids']))}"

        logger.info("To use this split, run:")
        logger.info(f"\033[92m{cmd_str}\033[0m")
        logger.info("=" * 40)

        return

    # Split Data strictly by indices
    # We pass empty lists if None to avoid errors in split_data
    train_files, val_files, test_files = split_data(
        dataset_batches,
        args.train_indices or [],
        args.val_indices or [],
        args.test_indices or [],
    )

    if train_files is None:
        return

    # Print Post-Split Statistics
    print_split_statistics(train_files, val_files, test_files, all_annotations)

    # Handle stats-only mode
    if args.dry_run:
        logger.info("Dry-run mode finished. No files were written.")
        return

    config_output_dir_str = config.get("dataset_output_dir")
    if config_output_dir_str:
        output_dir = Path(config_output_dir_str)
    else:
        logger.error(f"'dataset_output_dir' not found in {args.config}.")
        return
    logger.info(f"Proceeding with dataset creation (Output Dir: {output_dir})...")

    # Process Images
    class_to_id = {name: i for i, name in enumerate(class_names)}

    process_images(
        train_files,
        all_annotations,
        class_to_id,
        "train",
        output_dir,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        save_only_with_labels=save_only_with_labels,
    )
    process_images(
        val_files,
        all_annotations,
        class_to_id,
        "val",
        output_dir,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        save_only_with_labels=save_only_with_labels,
    )
    if test_files:
        process_images(
            test_files,
            all_annotations,
            class_to_id,
            "test",
            output_dir,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            save_only_with_labels=save_only_with_labels,
        )

    # Create YAML file
    create_yaml_file(class_names, has_test_set=bool(test_files), output_dir=output_dir)


if __name__ == "__main__":
    main()
