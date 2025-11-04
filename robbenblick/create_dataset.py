import random
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import collections

import cv2
import yaml
from tqdm import tqdm

from robbenblick import DATA_PATH, dataset_config, logger

# Default paths
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

    task_name_tag = root.find("meta/task/name")
    task_name = task_name_tag.text if task_name_tag is not None else "Unknown Task"

    for image_tag in root.findall("image"):
        image_name = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))

        polygons = []

        # Find <polyline> as well as <polygon> tags
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


def process_images(image_files, annotations, class_to_id, split, output_dir):
    """Tiles images and converts annotations for a given split (train/val/test)."""
    logger.info(f"Processing {split} set...")

    img_output_dir = output_dir / "images" / split
    label_output_dir = output_dir / "labels" / split

    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for unique_key in tqdm(image_files, desc=f"Tiling {split} images"):
        img_annotations = annotations[unique_key]

        image_path = img_annotations["full_image_path"]

        # De-comment for very detailed logging
        # logger.info(
        #     f"Processing image: {image_path.name}, number of polygons: {len(img_annotations['polygons'])}"
        # )

        if not image_path.exists():
            logger.warning(f"Image file not found for {image_path.name}, skipping.")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image {image_path.name}, skipping.")
            continue

        img_h, img_w, _ = img.shape

        step_size = int(dataset_config.tile_size * (1 - dataset_config.tile_overlap))

        for y in range(0, img_h, step_size):
            for x in range(0, img_w, step_size):
                # Define tile boundaries
                tile_x_min, tile_y_min = x, y
                tile_x_max, tile_y_max = (
                    x + dataset_config.tile_size,
                    y + dataset_config.tile_size,
                )

                # Extract the tile
                tile_img = img[tile_y_min:tile_y_max, tile_x_min:tile_x_max]
                tile_h, tile_w, _ = tile_img.shape

                # Skip tiny tiles at the edges
                if (
                        tile_h < dataset_config.tile_size * 0.25
                        or tile_w < dataset_config.tile_size * 0.25
                ):
                    continue

                tile_labels = []

                for poly_info in img_annotations["polygons"]:
                    bbox = polygon_to_bbox(poly_info["points"])

                    # Check for intersection between bbox and tile
                    intersect_x_min = max(bbox[0], tile_x_min)
                    intersect_y_min = max(bbox[1], tile_y_min)
                    intersect_x_max = min(bbox[2], tile_x_max)
                    intersect_y_max = min(bbox[3], tile_y_max)

                    if (
                            intersect_x_min < intersect_x_max
                            and intersect_y_min < intersect_y_max
                    ):
                        # Convert intersection coords to tile-local coords
                        local_bbox = [
                            intersect_x_min - tile_x_min,
                            intersect_y_min - tile_y_min,
                            intersect_x_max - tile_x_min,
                            intersect_y_max - tile_y_min,
                        ]

                        if (
                                local_bbox[2] <= local_bbox[0]
                                or local_bbox[3] <= local_bbox[1]
                        ):
                            # logger.warning(
                            #     f"Invalid bbox in image {image_path.name}, skipping bbox."
                            # )
                            continue

                        if (local_bbox[2] - local_bbox[0]) * (
                                local_bbox[3] - local_bbox[1]
                        ) < (tile_h * tile_w * 0.0001):
                            # logger.warning(
                            #     f"Small bbox in image {image_path.name}, skipping bbox."
                            # )
                            continue

                        # Convert to YOLO format
                        yolo_bbox = convert_to_yolo_format(local_bbox, tile_w, tile_h)
                        class_id = class_to_id[poly_info["label"]]

                        tile_labels.append(
                            f"{class_id} {' '.join(map(str, yolo_bbox))}"
                        )

                # Only save tiles that have labels
                should_save = tile_labels or not dataset_config.save_only_with_labels

                if should_save:
                    base_filename = f"{Path(unique_key).stem}_tile_{y}_{x}"
                    img_save_path = img_output_dir / f"{base_filename}.jpg"
                    label_save_path = label_output_dir / f"{base_filename}.txt"

                    cv2.imwrite(str(img_save_path), tile_img)

                    with open(label_save_path, "w") as f:
                        f.write("\n".join(tile_labels))


def create_yaml_file(class_names, has_test_set, output_dir):
    """Creates the data.yaml file for YOLO training."""
    yaml_content = {
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if has_test_set:
        yaml_content["test"] = "images/test"

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    logger.info(f"Created data.yaml at {yaml_path}")
    logger.info("Dataset creation complete! 🎉")


def main(stats_only=False, test_dir_index=None, val_ratio=0.2,
         raw_dir=DEFAULT_RAW_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    """Main function to run the script."""

    logger.info(f"Scanning for datasets in {raw_dir}...")

    all_annotations = {}
    all_class_names = set()
    dataset_batches = []

    total_annotations_count = 0
    total_image_count = 0
    overall_class_distribution = collections.defaultdict(int)

    source_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])

    if not source_dirs:
        logger.error(f"No dataset subdirectories found in {raw_dir}. Exiting.")
        return

    # Step 1: read all datasets
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
        all_class_names.update(class_names_batch)

        # Enrich annotations dict with full path and store for batch
        processed_annotations = {}
        for image_name, data in annotations_batch.items():
            unique_key = f"{src_dir.name}_{image_name}"
            data["full_image_path"] = image_dir / image_name
            if unique_key in all_annotations:
                logger.warning(f"Duplicate unique key '{unique_key}' found. Overwriting previous entry.")

            processed_annotations[unique_key] = data
            all_annotations[unique_key] = data  # Globales Dict für process_images

        dataset_batches.append({
            "name": src_dir.name,
            "task_name": task_name,
            "annotations": processed_annotations,
            "class_names": class_names_batch
        })

    # --- Step 2: show enumerated statistics
    logger.info("=" * 30)
    logger.info("--- 📊 DATASET STATISTICS ---")

    for i, batch in enumerate(dataset_batches):
        batch_class_distribution = collections.defaultdict(int)
        num_annotations_in_batch = 0

        for data in batch["annotations"].values():
            num_polygons = len(data["polygons"])
            num_annotations_in_batch += num_polygons
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

    if stats_only:
        logger.info("Stats-only mode finished. No files were written.")
        return

    # --- Step 3: split data
    logger.info(f"Proceeding with dataset creation (Output Dir: {output_dir})...")

    train_files = []
    val_files = []
    test_files = []

    if test_dir_index is not None:
        # --- NEUE LOGIK: Hold-Out-Datensatz als Test-Set ---
        if not (0 < test_dir_index <= len(dataset_batches)):
            logger.error(f"Invalid --test-dir-index {test_dir_index}. Must be between 1 and {len(dataset_batches)}.")
            return

        logger.info(f"Using Dataset #{test_dir_index} ({dataset_batches[test_dir_index - 1]['name']}) as TEST set.")
        logger.info(
            f"Splitting remaining {len(dataset_batches) - 1} datasets into TRAIN/VAL with {val_ratio * 100}% validation ratio.")

        test_batch = dataset_batches.pop(test_dir_index - 1)
        test_files = list(test_batch["annotations"].keys())

        # Restliche Batches für Train/Val sammeln
        train_val_image_keys = []
        for batch in dataset_batches:
            train_val_image_keys.extend(list(batch["annotations"].keys()))

        random.shuffle(train_val_image_keys)
        split_idx = int(len(train_val_image_keys) * val_ratio)
        val_files = train_val_image_keys[:split_idx]
        train_files = train_val_image_keys[split_idx:]

    else:
        # --- ALTE LOGIK: Alles mischen für Train/Val ---
        logger.info(f"No --test-dir-index provided. Mixing all {len(dataset_batches)} datasets.")
        logger.info(f"Splitting all images into TRAIN/VAL with {val_ratio * 100}% validation ratio.")

        all_image_keys = list(all_annotations.keys())
        random.shuffle(all_image_keys)

        split_idx = int(len(all_image_keys) * val_ratio)
        val_files = all_image_keys[:split_idx]
        train_files = all_image_keys[split_idx:]
        # test_files bleibt leer

    logger.info(
        f"Splitting data: {len(train_files)} TRAIN images, {len(val_files)} VAL images, {len(test_files)} TEST images.")

    # --- Schritt 4: Bilder verarbeiten ---
    class_to_id = {name: i for i, name in enumerate(class_names)}

    process_images(train_files, all_annotations, class_to_id, "train", output_dir)
    process_images(val_files, all_annotations, class_to_id, "val", output_dir)
    if test_files:
        process_images(test_files, all_annotations, class_to_id, "test", output_dir)

    # --- Schritt 5: YAML-Datei erstellen ---
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

    # Logik-Check: val_ratio muss zwischen 0 und 1 sein
    if not (0.0 <= args.val_ratio <= 1.0):
        logger.error("--val-ratio must be between 0.0 and 1.0.")
    else:
        main(
            stats_only=True,
            test_dir_index=args.test_dir_index,
            val_ratio=args.val_ratio,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir
        )