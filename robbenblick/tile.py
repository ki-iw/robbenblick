import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from shapely.errors import GEOSException
from shapely.geometry import Polygon, box

from robbenblick import logger


def parse_cvat_polygons(xml_path):
    """
    Parse a CVAT XML annotation file and extract polygons for each image.

    Args:
        xml_path (str): Path to the CVAT XML annotation file.

    Returns:
        dict: Mapping from image filename to list of polygons, where each polygon is a list of (x, y) tuples.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_polygons = {}
    for image in root.findall("image"):
        img_name = image.get("name")
        polygons = []
        for poly in image.findall("polyline"):
            points_str = poly.get("points")
            # Split points and convert to float tuples
            points = [
                tuple(map(float, pt.split(","))) for pt in points_str.split(";") if pt
            ]
            polygons.append(points)
        image_polygons[img_name] = polygons
    return image_polygons


def tile_images(source_dir, dest_dir, tile_size, overlap_percent):
    """
    Tiles all .tif images from a source directory into smaller, overlapping tiles.

    Args:
        source_dir (str): The directory containing the original .tif images.
        dest_dir (str): The directory where the new tiles will be saved.
        tile_size (tuple): The (width, height) of the tiles in pixels.
        overlap_percent (int): The percentage of overlap between tiles (0-100).
    """

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    logger.info("Created destination directory: %s", dest_dir)

    # Calculate overlap in pixels
    overlap_pixels_x = int(tile_size[0] * (overlap_percent / 100))
    overlap_pixels_y = int(tile_size[1] * (overlap_percent / 100))

    # Get a list of all .tif files in the source directory
    image_files = glob.glob(os.path.join(source_dir, "*.TIF"))
    if not image_files:
        logger.warning("No .tif images found in %s. Exiting.", source_dir)
        return

    logger.info("Found %d images to tile.", len(image_files))

    # Iterate through each image and tile it
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img_width, img_height = img.size

                logger.info(
                    "Processing '%s.tif' (%dx%d pixels)...",
                    img_name,
                    img_width,
                    img_height,
                )

                tile_count = 0
                for y in range(0, img_height, tile_size[1] - overlap_pixels_y):
                    for x in range(0, img_width, tile_size[0] - overlap_pixels_x):
                        # Define the crop box
                        left = x
                        upper = y
                        right = x + tile_size[0]
                        lower = y + tile_size[1]

                        # Adjust the box if it extends beyond the image boundaries
                        if right > img_width:
                            right = img_width
                            left = right - tile_size[0]
                        if lower > img_height:
                            lower = img_height
                            upper = lower - tile_size[1]

                        # Crop the image
                        tile = img.crop((left, upper, right, lower))

                        # Save the new tile
                        tile_filename = f"{img_name}_tile_{tile_count}.tif"
                        tile_path = os.path.join(dest_dir, tile_filename)
                        tile.save(tile_path)
                        tile_count += 1

                logger.info(
                    "Successfully created %d tiles for '%s.tif'.", tile_count, img_name
                )

        except Exception as e:
            logger.error("Error processing %s: %s", img_path, e)


def save_yolo_segmentation(txt_path, polygons, tile_size, class_id=0):
    """
    Save polygons in YOLOv8 segmentation format for a single image tile.

    Args:
        txt_path (str): Path to save the YOLO annotation .txt file.
        polygons (list): List of polygons, each as a list of (x, y) tuples.
        tile_size (tuple): (width, height) of the tile image.
        class_id (int): Class index for YOLO annotation.
    """
    with open(txt_path, "w") as f:
        for poly in polygons:
            # Flatten and normalize coordinates
            coords = np.array(poly)
            coords[:, 0] = coords[:, 0] / tile_size[0]  # x normalization
            coords[:, 1] = coords[:, 1] / tile_size[1]  # y normalization
            coords_flat = coords.flatten()
            coords_str = " ".join([f"{c:.6f}" for c in coords_flat])
            f.write(f"{class_id} {coords_str}\n")


def crop_and_save_yolo_tiles(
    image_path, polygons, tile_size, output_dir, class_id=0, img_name=None
):
    """
    Crop an image into tiles and save YOLOv8 segmentation annotations for each tile.

    Args:
        image_path (str): Path to the input image.
        polygons (list): List of polygons for the image, each as a list of (x, y) tuples.
        tile_size (tuple): (width, height) of each tile.
        output_dir (str): Directory to save cropped tiles and annotation files.
        class_id (int): Class index for YOLO annotation.
    """
    img = Image.open(image_path)
    img_width, img_height = img.size

    images_dir = os.path.join(output_dir, "images")
    annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    tile_id = 0
    base_name = os.path.splitext(img_name)[0] if img_name else "image"
    for y in range(0, img_height, tile_size[1]):
        for x in range(0, img_width, tile_size[0]):
            tile_box = box(
                x,
                y,
                min(x + tile_size[0], img_width),
                min(y + tile_size[1], img_height),
            )
            tile_polygons = []
            for poly in polygons:
                poly_obj = Polygon(poly)
                try:
                    intersection = poly_obj.intersection(tile_box)
                except Exception as e:
                    if isinstance(e, GEOSException):
                        logger.error(
                            f"GEOSException for polygon in tile ({x}, {y}): {e}"
                        )
                        continue
                    else:
                        logger.error(
                            f"Unexpected exception for polygon in tile ({x}, {y}): {e}"
                        )
                        continue
                if not intersection.is_empty and intersection.geom_type == "Polygon":
                    rel_coords = [
                        (pt[0] - x, pt[1] - y)
                        for pt in intersection.exterior.coords[:-1]
                    ]
                    tile_polygons.append(rel_coords)
            # Save tile image
            tile = img.crop(
                (
                    x,
                    y,
                    min(x + tile_size[0], img_width),
                    min(y + tile_size[1], img_height),
                )
            )
            tile_filename = f"{base_name}_tile_{tile_id}.png"
            # Save tile image
            tile.save(os.path.join(images_dir, tile_filename))
            # Save YOLO segmentation annotation
            if tile_polygons:
                txt_path = os.path.join(
                    annotations_dir, f"{base_name}_tile_{tile_id}.txt"
                )
                save_yolo_segmentation(txt_path, tile_polygons, tile.size, class_id)
            tile_id += 1
