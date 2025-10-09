import os
import argparse
import fiftyone as fo
import fiftyone.utils.cvat as fouc
import fiftyone.utils.yolo as fouy

from robben_blick import DATA_PATH

def launch_fiftyone_cvat(xml_path, images_dir):
    """
    Launch FiftyOne app to visualize CVAT XML annotation data.

    Args:
        xml_path (str): Path to CVAT XML annotation file.
        images_dir (str): Path to directory with images.
    """
    dataset = fouc.load_cvat_dataset(
        xml_path=xml_path,
        images_dir=images_dir,
        name="raw_cvat"
    )
    session = fo.launch_app(dataset)
    return session

def launch_fiftyone_yolo(images_dir, labels_dir, classes=None):
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
    session = fo.launch_app(dataset)
    return session

# import fiftyone as fo
# import os 
# import fiftyone.utils.yolo as fouy
#import fiftyone.types as fot

# name = "robben"
# image_path = "data/robben/images"
# labels_path = "data/robben/annotations.xml"

# datapath_dir = os.path.abspath(image_path)
# labels_path = os.path.abspath(labels_path)

# #recreate = True 

# print("Available FiftyOne datasets:", fo.list_datasets())
# # dataset = fo.Dataset(name)
# recreate = True

# if recreate == True:
#     for ds in fo.list_datasets():
#         fo.delete_dataset(ds)
            
# if name in fo.list_datasets():
#     dataset = fo.load_dataset(name)
#     print(f"Dataset '{name}' loaded with {len(dataset)} samples")
# else: 
#     dataset = fo.Dataset.from_dir(
#         name=name,
#         #dataset_type=fo.types.CVATImageDataset,
#         dataset_type=fo.types.CVATImageDataset, 
#         #dataset_type=fo.types.ImageDirectory,
#         #dataset_dir=datapath_dir,
#         data_path=datapath_dir,
#         labels_path=labels_path
#         )
#     print(f"Dataset '{name}' created with {len(dataset)} samples")

# if recreate==True: 
#     dataset = fo.Dataset.from_dir(
#         dataset_type=fo.types.CVATImageDataset,
#         data_path="data/robben/images",
#         labels_path="data/robben/annotations.xml"
#         )
# else:
#     dataset = fo.load_dataset(name, create_if_necessary=True)    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch FiftyOne visualization for YOLOv8 or CVAT annotations.")
    parser.add_argument("--dataset", choices=["yolo", "cvat"], required=True, help="Choose which annotation format to visualize.")
    parser.add_argument("--classes", nargs="*", help="List of class names (for yolo mode).")
    args = parser.parse_args()

    if args.mode == "yolo":
        image_dir = os.path.join(DATA_PATH, "yolo_tiles", "images")
        labels_dir = os.path.join(DATA_PATH, "yolo_tiles", "annotations")
        launch_fiftyone_yolo(args.images_dir, args.labels_dir, args.classes)
    elif args.mode == "cvat":
        xml_path = os.path.join(DATA_PATH, "raw", "annotations.xml")
        image_dir = os.path.join(DATA_PATH, "raw", "images")
        launch_fiftyone_cvat(args.xml_path, args.images_dir)