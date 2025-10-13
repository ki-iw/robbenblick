import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.yolo as fouy

from robbenblick import DATA_PATH, logger
from robbenblick.utils import convert_tif_to_png, convert_xml_tif_to_png


def fo_cvat_dataset(xml_path, images_dir):
    """
    Launch FiftyOne app to visualize CVAT XML annotation data.

    Args:
        xml_path (str): Path to CVAT XML annotation file.
        images_dir (str): Path to directory with images.
    """
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.CVATImageDataset,
        data_path=images_dir,
        labels_path=xml_path,
        name="raw_cvat",
    )
    # session = fo.launch_app(dataset)
    return dataset


def fo_yolo_dataset(images_dir, labels_dir, classes=None):
    """
    Launch FiftyOne app to visualize YOLOv8 segmentation data.

    Args:
        images_dir (str): Path to directory with images.
        labels_dir (str): Path to directory with YOLOv8 .txt annotation files.
        classes (list, optional): List of class names. Defaults to None.
    """
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.YOLOv5Dataset,  # Supports YOLOv5 and YOLOv8
        data_path=images_dir,
        labels_path=labels_dir,
        name="yolo_segmentation",
        classers=classes,
    )
    # session = fo.launch_app(dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch FiftyOne visualization for YOLOv8 or CVAT annotations."
    )
    parser.add_argument(
        "--dataset",
        choices=["yolo", "cvat"],
        default="cvat",
        required=False,
        help="Choose which annotation format to visualize.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the dataset if it exists.",
    )
    args = parser.parse_args()

    if args.dataset == "yolo":
        image_dir = DATA_PATH / "yolo_tiles" / "images"
        labels_dir = DATA_PATH / "yolo_tiles" / "annotations"
        dataset = fo_yolo_dataset(image_dir, labels_dir, 0)
    elif args.dataset == "cvat":
        xml_path = DATA_PATH / "raw" / "annotations.xml"
        image_dir = DATA_PATH / "raw" / "images"
        tif_images = [f.name for f in image_dir.iterdir() if f.suffix.lower() == ".tif"]
        if tif_images:
            png_dir = DATA_PATH / "raw" / "images_png"
            png_dir.mkdir(parents=True, exist_ok=True)
            png_images = (
                [f.name for f in png_dir.iterdir() if f.suffix.lower() == ".png"]
                if png_dir.exists()
                else []
            )
            if png_images:
                logger.info("PNG images already exist, using converted folder.")
                xml_path_png = xml_path.with_name(xml_path.stem + "_png.xml")
                convert_xml_tif_to_png(str(xml_path), str(xml_path_png))
                xml_path = xml_path_png
            else:
                logger.info(
                    "The images are .tif format, and need to be converted to .png for FiftyOne visualization."
                )
                convert_tif_to_png(str(image_dir), str(png_dir))
                logger.info("Converting XML annotations from .tif to .png references.")
                xml_path_png = xml_path.with_name(xml_path.stem + "_png.xml")
                convert_xml_tif_to_png(str(xml_path), str(xml_path_png))
                xml_path = xml_path_png
            image_dir = png_dir
        ds_name = "raw_cvat"
        if args.recreate:
            if ds_name in fo.list_datasets():
                logger.info(f"Deleting dataset '{ds_name}' to recreate it.")
                fo.delete_dataset(ds_name)

            logger.info(f"Creating dataset '{ds_name}' from directory {image_dir}.")
            dataset = fo_cvat_dataset(xml_path, image_dir)
        else:
            if ds_name in fo.list_datasets():
                logger.info(f"Loading existing dataset '{ds_name}'.")
                dataset = fo.load_dataset(ds_name)
            else:
                logger.info(
                    f"Dataset '{ds_name}' not found. Creating new dataset from directory {image_dir}."
                )
                dataset = fo_cvat_dataset(xml_path, image_dir)
    dataset.persistent = True
    session = fo.launch_app(dataset, port=5157)
    session.wait()
