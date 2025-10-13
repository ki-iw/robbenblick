import os

from tqdm import tqdm

from robbenblick import DATA_PATH, logger
from robbenblick.tile import crop_and_save_yolo_tiles, parse_cvat_polygons


def main() -> str:
    logger.info("Starting up")
    # Parse polygons from CVAT XML
    xml_path = os.path.join(DATA_PATH, "raw", "annotations.xml")
    image_polygons = parse_cvat_polygons(xml_path)

    # Directory containing images
    image_dir = os.path.join(DATA_PATH, "raw", "images")

    # Output directory for tiles and labels
    output_dir = os.path.join(DATA_PATH, "yolo_tiles")
    # Tile size (width, height)
    tile_size = (1024, 1024)
    # Class id for YOLO
    class_id = 0
    # Process each imag
    for img_name, polygons in tqdm(image_polygons.items(), desc="Processing images"):
        image_path = os.path.join(image_dir, img_name)
        logger.info(
            "Processing image: {}, number polygons: {}", image_path, len(polygons)
        )
        crop_and_save_yolo_tiles(
            image_path, polygons, tile_size, output_dir, class_id, img_name
        )
    logger.info("All images processed.")
    return "Processing complete."


def process_raw_data() -> None:
    """Process raw data into processed data."""
    logger.info("Processing raw data")
    # Add your data processing code here


if __name__ == "__main__":
    main()
