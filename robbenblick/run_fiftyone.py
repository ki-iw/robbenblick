import os
import argparse
import fiftyone as fo
import fiftyone.utils.yolo as fouy
from robbenblick.utils import convert_tif_to_png, convert_xml_tif_to_png

from robbenblick import DATA_PATH, logger

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
        name="raw_cvat"
    )
    #session = fo.launch_app(dataset)
    return dataset

def fo_yolo_dataset(images_dir, labels_dir, classes=None):
    """
    Launch FiftyOne app to visualize YOLOv8 segmentation data.

    Args:
        images_dir (str): Path to directory with images.
        labels_dir (str): Path to directory with YOLOv8 .txt annotation files.
        classes (list, optional): List of class names. Defaults to None.
    """
    dataset = fouy.load_yolo_dataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        data_type="segmentation",
        classes=classes,
        name="yolo_segmentation"
    )
    #session = fo.launch_app(dataset)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch FiftyOne visualization for YOLOv8 or CVAT annotations.")
    parser.add_argument("--dataset", choices=["yolo", "cvat"], default="cvat", required=False, help="Choose which annotation format to visualize.")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the dataset if it exists.")
    args = parser.parse_args()

    if args.dataset == "yolo":
        image_dir = os.path.join(DATA_PATH, "yolo_tiles", "images")
        labels_dir = os.path.join(DATA_PATH, "yolo_tiles", "annotations")
        dataset = fo_yolo_dataset(image_dir, labels_dir, 0)
    elif args.dataset == "cvat":
        xml_path = os.path.join(DATA_PATH, "raw", "annotations.xml")
        image_dir = os.path.join(DATA_PATH, "raw", "images")
        tif_images = [f for f in os.listdir(image_dir) if f.lower().endswith('.tif')]
        if tif_images:
            png_dir = os.path.join(DATA_PATH, "raw", "images_png")
            os.makedirs(png_dir, exist_ok=True)
            png_images = [f for f in os.listdir(png_dir) if f.lower().endswith('.png')] if os.path.exists(png_dir) else []
            if png_images:
                logger.info("PNG images already exist, using converted folder.")
                xml_path_png = os.path.splitext(xml_path)[0] + '_png.xml'
                convert_xml_tif_to_png(xml_path, xml_path_png)
                xml_path = xml_path_png
            else:
                logger.info("The images are .tif format, and need to be converted to .png for FiftyOne visualization.")
                convert_tif_to_png(image_dir, png_dir)
                logger.info("Converting XML annotations from .tif to .png references.")
                xml_path_png = os.path.splitext(xml_path)[0] + '_png.xml'
                convert_xml_tif_to_png(xml_path, xml_path_png)
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
                logger.info(f"Dataset '{ds_name}' not found. Creating new dataset from directory {image_dir}.")
                dataset = fo_cvat_dataset(xml_path, image_dir)
    dataset.persistent = True
    session = fo.launch_app(dataset, port=5157)
    session.wait()
