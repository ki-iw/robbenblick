import random
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import collections

import cv2
import yaml
from tqdm import tqdm

from robbenblick import DATA_PATH, dataset_config, logger

# Default paths for the command-line arguments
DEFAULT_RAW_DIR = DATA_PATH / "raw"
DEFAULT_OUTPUT_DIR = DATA_PATH / "processed" / "dataset_yolo"

# Tiling parameters from config
logger.info(
    f"Tile size: {dataset_config.tile_size}, Tile overlap: {dataset_config.tile_overlap}"
)
logger.info(f"Save only tiles with labels: {dataset_config.save_only_with_labels}")


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

        # Find both <polyline> and <polygon> tags, as CVAT can export both.
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

    return (x_center_norm, y_center_norm, width_norm, height_norm)


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
            if (local_bbox[2] <= local_bbox[0] or local_bbox[3] <= local_bbox[1]):
                continue
            if (local_bbox[2] - local_bbox[0]) * (
                    local_bbox[3] - local_bbox[1]
            ) < (tile_h * tile_w * 0.0001):
                continue

            # Convert to YOLO format
            yolo_bbox = convert_to_yolo_format(local_bbox, tile_w, tile_h)
            class_id = class_to_id[poly_info["label"]]

            tile_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")

    return tile_labels


def save_tile(tile_img, tile_labels, base_filename, img_output_dir, label_output_dir):
    """Saves a single tile image and its corresponding label file."""
    img_save_path = img_output_dir / f"{base_filename}.jpg"
    label_save_path = label_output_dir / f"{base_filename}.txt"

    cv2.imwrite(str(img_save_path), tile_img)

    with open(label_save_path, "w") as f:
        f.write("\n".join(tile_labels))


def tile_one_image(unique_key, annotations, class_to_id, img_output_dir, label_output_dir):
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
    step_size = int(dataset_config.tile_size * (1 - dataset_config.tile_overlap))

    for y in range(0, img_h, step_size):
        for x in range(0, img_w, step_size):
            tile_x_min, tile_y_min = x, y
            tile_x_max, tile_y_max = (
                x + dataset_config.tile_size,
                y + dataset_config.tile_size,
            )

            tile_img = img[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
            tile_h, tile_w, _ = tile_img.shape

            # Skip tiny tiles at the edges
            if (
                    tile_h < dataset_config.tile_size * 0.25
                    or tile_w < dataset_config.tile_size * 0.25
            ):
                continue

            tile_coords = (tile_x_min, tile_y_min, tile_x_max, tile_y_max)
            tile_dims = (tile_w, tile_h)

            tile_labels = find_labels_for_tile(
                img_annotations, tile_coords, tile_dims, class_to_id
            )

            # Only save tiles that have labels (or if config allows empty)
            should_save = tile_labels or not dataset_config.save_only_with_labels
            if should_save:
                base_filename = f"{Path(unique_key).stem}_tile_{y}_{x}"
                save_tile(
                    tile_img, tile_labels, base_filename, img_output_dir, label_output_dir
                )


def process_images(image_files, annotations, class_to_id, split, output_dir):
    """Tiles images and converts annotations for a given split (train/val/test)."""
    logger.info(f"Processing {split} set ({len(image_files)} images)...")

    img_output_dir = output_dir / "images" / split
    label_output_dir = output_dir / "labels" / split

    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for unique_key in tqdm(image_files, desc=f"Tiling {split} images"):
        tile_one_image(
            unique_key, annotations, class_to_id, img_output_dir, label_output_dir
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
    """--- Step 1: Load and Prepare Data ---"""
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

        processed_annotations = {}
        for image_name, data in annotations_batch.items():
            unique_key = f"{src_dir.name}_{image_name}"
            data["full_image_path"] = image_dir / image_name
            if unique_key in all_annotations:
                logger.warning(f"Duplicate unique key '{unique_key}' found. Overwriting previous entry.")

            processed_annotations[unique_key] = data
            all_annotations[unique_key] = data  # Global dict for process_images

        dataset_batches.append({
            "name": src_dir.name,
            "task_name": task_name,
            "annotations": processed_annotations,
            "class_names": class_names_batch
        })

    return dataset_batches, all_annotations


def print_dataset_statistics(dataset_batches):
    """--- Step 2: Print Dataset Statistics ---"""
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
        logger.info("-" * 20)

    logger.info("--- 📊 OVERALL DATASET STATISTICS ---")
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


def split_data(dataset_batches, all_annotations, test_dir_index, val_ratio):
    """--- Step 3: Split Data ---"""
    train_files = []
    val_files = []
    test_files = []

    if test_dir_index is not None:
        # Hold-Out-Dataset as Test-Set
        if not (0 < test_dir_index <= len(dataset_batches)):
            logger.error(f"Invalid --test-dir-index {test_dir_index}. Must be between 1 and {len(dataset_batches)}.")
            return None, None, None

        logger.info(f"Using Dataset #{test_dir_index} ({dataset_batches[test_dir_index - 1]['name']}) as TEST set.")
        logger.info(
            f"Splitting remaining {len(dataset_batches) - 1} datasets into TRAIN/VAL with {val_ratio * 100}% validation ratio.")

        test_batch = dataset_batches.pop(test_dir_index - 1)
        test_files = list(test_batch["annotations"].keys())

        # Collect remaining batches for Train/Val
        train_val_image_keys = []
        for batch in dataset_batches:
            train_val_image_keys.extend(list(batch["annotations"].keys()))

        random.shuffle(train_val_image_keys)
        split_idx = int(len(train_val_image_keys) * val_ratio)
        val_files = train_val_image_keys[:split_idx]
        train_files = train_val_image_keys[split_idx:]

    else:
        # Mix all datasets for Train/Val
        logger.info(f"No --test-dir-index provided. Mixing all {len(dataset_batches)} datasets.")
        logger.info(f"Splitting all images into TRAIN/VAL with {val_ratio * 100}% validation ratio.")

        all_image_keys = list(all_annotations.keys())
        random.shuffle(all_image_keys)

        split_idx = int(len(all_image_keys) * val_ratio)
        val_files = all_image_keys[:split_idx]
        train_files = all_image_keys[split_idx:]
        # test_files remains empty

    logger.info(
        f"Splitting data: {len(train_files)} TRAIN images, {len(val_files)} VAL images, {len(test_files)} TEST images.")
    return train_files, val_files, test_files


def main(stats_only, test_dir_index, val_ratio, raw_dir, output_dir):
    """Main function to run the script."""

    # --- Step 1: Load Data ---
    dataset_batches, all_annotations = load_and_prepare_data(raw_dir)
    if dataset_batches is None:
        return

    # --- Step 2: Print Statistics ---
    class_names = print_dataset_statistics(dataset_batches)

    # --- Step 3: Handle stats-only mode ---
    if stats_only:
        logger.info("Stats-only mode finished. No files were written.")
        return

    logger.info(f"Proceeding with dataset creation (Output Dir: {output_dir})...")

    # --- Step 4: Split Data ---
    train_files, val_files, test_files = split_data(
        dataset_batches, all_annotations, test_dir_index, val_ratio
    )
    if train_files is None:
        return  # Error already logged in split_data

    # --- Step 5: Process Images ---
    class_to_id = {name: i for i, name in enumerate(class_names)}

    process_images(train_files, all_annotations, class_to_id, "train", output_dir)
    process_images(val_files, all_annotations, class_to_id, "val", output_dir)
    if test_files:
        process_images(test_files, all_annotations, class_to_id, "test", output_dir)

    # --- Step 6: Create YAML file ---
    create_yaml_file(class_names, has_test_set=bool(test_files), output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create YOLO dataset from CVAT XML.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Run in statistics-only mode. No files will be written."
    )
    parser.add_argument(
        "--test-dir-index",
        type=int,
        default=None,
        help="Specify the 1-based index of the dataset (from --stats-only) to use as the TEST set. All other datasets will be split into train/val."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="The ratio of images to use for the VALIDATION set (default: 0.2)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Base directory containing raw dataset subfolders (default: {DEFAULT_RAW_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the processed YOLO dataset (default: {DEFAULT_OUTPUT_DIR})"
    )
    args = parser.parse_args()

    # Logic-Check: val_ratio must be between 0 and 1
    if not (0.0 <= args.val_ratio <= 1.0):
        logger.error("--val-ratio must be between 0.0 and 1.0.")
    else:
        main(
            stats_only=args.stats_only,
            test_dir_index=args.test_dir_index,
            val_ratio=args.val_ratio,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir
        )