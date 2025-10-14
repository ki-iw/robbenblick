# robbenblick
A Computer Vision project for object detection and annotation management using YOLOv8 and FiftyOne.

## Overview
This repository provides a pipeline for:
- Preparing and tiling annotated image datasets (from CVAT)
- Training and evaluating YOLOv8 models (Ultralytics)
- Visualizing datasets and predictions with FiftyOne

## Main Scripts & Functionality

### `create_dataset.py`
- **Purpose:** Converts raw CVAT-annotated images and XML files into a YOLO-compatible dataset, including tiling and label conversion.
- **How it works:**
  - Loads configuration from `configs/create_dataset.yaml` using DotMap for easy access.
  - Parses CVAT XML annotations, extracts polygons, and tiles images into smaller crops.
  - Converts polygon annotations to YOLO bounding box format for each tile.
  - Splits data into train/test sets and writes images/labels to `data/processed/dataset_yolo`.
  - Generates a `data.yaml` file for YOLO training.
- **Run:**
  ```
  python -m robbenblick.create_dataset
  ```

### `yolo.py`
- **Purpose:** Trains and evaluates YOLOv8 models using the processed dataset.
- **How it works:**
  - Loads model and training parameters from `configs/model.yaml` via DotMap.
  - Detects available hardware (CUDA, MPS, or CPU) and logs device info.
  - Trains YOLOv8 using Ultralytics with the specified config and dataset.
  - Supports multiple modes (e.g., `train`, `predict`) via argparse arguments.
  - Saves predictions to `run/detect/<run_id>_predict` when in predict mode.
- **Run:**
  ```
  python -m robbenblick.yolo --mode train --run_id <name-of-run>
  python -m robbenblick.yolo --mode predict --run_id <name-of-run>
  ```

### `run_fiftyone.py`
- **Purpose:** Visualizes datasets and predictions using FiftyOne.
- **How it works:**
  - Loads either YOLO or CVAT datasets for visualization.
  - Handles conversion of `.tif` images to `.png` for browser compatibility.
  - Creates a fiftyone dataset with either:
        - CVAT images and the CVAT annotation
        - A new set of images, runs inference with the selected model and displays the result
  - Launches the FiftyOne app .
- **Run:**
  ```
  python -m robbenblick.run_fiftyone --run_id <name-of-run> --dataset cvat --recreate
  python -m robbenblick.run_fiftyone --run_id <name-of-run> --dataset yolo --recreate
  ```

## Configuration
- All major parameters (tiling, training, model, etc.) are set in YAML files under `configs/`.
- These are loaded as DotMap objects for dot-access in code.

## Environment Setup
```sh
conda env create --file environment.yml
conda activate RobbenBlick
```

## Pre-commit Hooks
To run code style and quality checks:
```sh
pre-commit run
```

## CVAT Annotation Workflow
1. Log into CVAT and go to "jobs", export ```CVAT for images``` and toggle on "save images".
2. Place images in `data/raw/images` and XML in `data/raw/annotations.xml`.
3. Run `create_dataset.py` to prepare the dataset for YOLO training.

## FiftyOne Visualization
- Place images and annotations in `data/raw/images` and `data/raw/annotations.xml`.
- Run:
  ```
  python -m robbenblick.run_fiftyone --dataset cvat --recreate
  ```
- If `.tif` images, they will be converted and saved as "png".


## Known Issues
### FiftyOne: failed to bind port
If you get:
```
fiftyone.core.service.ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port
```
Try:
```
pkill -f fiftyone
pkill -f mongod
```
Then rerun your script.

---
For more details, see the code comments and configuration files in `configs/`.
