import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from robbenblick import DATA_PATH, dataset_config, logger

# --- CONFIGURATION ---
# Input paths
IMAGE_DIR = DATA_PATH / "raw" / "images"
XML_PATH = DATA_PATH / "raw" / "annotations.xml"

# Output paths
OUTPUT_DIR = DATA_PATH / "processed" / "dataset_yolo"


# Tiling parameters and data split ratio from config
logger.info(
    f"Tile size: {dataset_config.tile_size}, Tile overlap: {dataset_config.tile_overlap}, Train ratio: {dataset_config.train_ratio}"
)
logger.info(f"Save only tiles with labels: {dataset_config.save_only_with_labels}")


def parse_cvat_xml(xml_file):
    """Parses a CVAT XML 1.1 file and extracts annotations."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = {}
    class_names = set()

    for image_tag in root.findall("image"):
        image_name = image_tag.get("name")
        width = int(image_tag.get("width"))
        height = int(image_tag.get("height"))

        polygons = []
        for poly_tag in image_tag.findall("polyline"):
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

    return annotations, sorted(list(class_names))


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


def process_images(image_files, annotations, class_to_id, split):
    """Tiles images and converts annotations for a given split (train/test)."""
    logger.info(f"Processing {split} set...")

    img_output_dir = OUTPUT_DIR / "images" / split
    label_output_dir = OUTPUT_DIR / "labels" / split

    img_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    for image_name in tqdm(image_files, desc=f"Tiling {split} images"):
        img_annotations = annotations[image_name]
        logger.info(
            f"Processing image: {image_name}, number of polygons: {len(img_annotations['polygons'])}"
        )
        image_path = IMAGE_DIR / image_name
        if not image_path.exists():
            logger.warning(f"Image file not found for {image_name}, skipping.")
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image {image_name}, skipping.")
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
                            logger.warning(
                                f"Invalid bbox in image {image_name}, skipping bbox."
                            )
                            continue

                        if (local_bbox[2] - local_bbox[0]) * (
                            local_bbox[3] - local_bbox[1]
                        ) < (tile_h * tile_w * 0.0001):
                            logger.warning(
                                f"Small bbox in image {image_name}, skipping bbox."
                            )
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
                    base_filename = f"{Path(image_name).stem}_tile_{y}_{x}"
                    img_save_path = img_output_dir / f"{base_filename}.jpg"
                    label_save_path = label_output_dir / f"{base_filename}.txt"

                    cv2.imwrite(str(img_save_path), tile_img)

                    with open(label_save_path, "w") as f:
                        f.write("\n".join(tile_labels))


def create_yaml_file(class_names):
    """Creates the data.yaml file for YOLO training."""
    yaml_content = {
        # "path": .. not necessary, since the yaml file is in the right folder
        "train": "images/train",
        "val": "images/test",  # Using test set as validation
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    logger.info(f"Created data.yaml at {yaml_path}")
    logger.info("Dataset creation complete! 🎉")


def main():
    """Main function to run the script."""
    # 1. Parse annotations
    logger.info("Parsing CVAT XML file...")
    annotations, class_names = parse_cvat_xml(XML_PATH)
    class_to_id = {name: i for i, name in enumerate(class_names)}
    logger.info(
        f"Found {len(annotations)} images and {len(class_names)} classes: {class_names}"
    )

    # 2. Split data
    all_image_files = list(annotations.keys())
    random.shuffle(all_image_files)

    split_index = int(len(all_image_files) * dataset_config.train_ratio)
    train_files = all_image_files[:split_index]
    test_files = all_image_files[split_index:]

    logger.info(
        f"Splitting data: {len(train_files)} training images, {len(test_files)} test images."
    )

    # 3. Process images for each split
    process_images(train_files, annotations, class_to_id, "train")
    process_images(test_files, annotations, class_to_id, "test")

    # 4. Create YAML file
    create_yaml_file(class_names)


if __name__ == "__main__":
    main()
