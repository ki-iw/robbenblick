# robbenblick
A Computer Vision project for object detection and annotation management using YOLOv8, SAHI, and FiftyOne,
with the primary aim of counting objects (Robben) in large aerial images.

## Overview
This repository provides a complete MLOps pipeline for:
* **Data Preparation:** Converting raw CVAT annotations (XML) and large images into a tiled, YOLO-compatible dataset.
* **Automated Experiments:** Systematically training and tuning YOLOv8 models.
* **Tiled Inference:** Running optimized inference (SAHI) on large, high-resolution images for object counting.
* **Evaluation:** Assessing model performance for both detection (mAP) and counting (MAE, RMSE, R²).
* **Visualization:** Analyzing datasets and model predictions interactively with FiftyOne.

## Pretrained Model Weights

Pretrained model weights are available on Hugging Face:
 https://huggingface.co/ki-ideenwerkstatt-23/robbenblick/

## Project Workflow

The project is designed to follow a clear, sequential workflow:

1.  **Prepare Data (`create_dataset.py`):**
    Organize your raw images and CVAT `annotations.xml` in `data/raw/` as shown below.
    ```text
    data/raw/
    ├── dataset_01/
    │   ├── annotations.xml
    │   └── images/
    └── dataset_02/ ...
    ```
    Run the script to generate a tiled, YOLO-formatted dataset in `data/processed/` and ground truth count CSVs.
2.  **Tune Model (`run_experiments.py`):**
    Define a set of hyperparameters (e.g., models, freeze layers, augmentation) in `configs/base_iter_config.yaml`. Run the script to train a model for every combination and find the best performer.
3.  **Validate Model (`yolo.py`):**
    Take the `run_id` of your best experiment and run validation on the hold-out `test` set to get **detection metrics (mAP)**.
4.  **Infer & Count (`predict_tiled.py`):**
    Use the best `run_id` to run sliced inference on new, large images. This script generates final counts and visual outputs.
5.  **Evaluate Counts (`evaluate_counts.py`):**
    Compare the `detection_counts.csv` from inference against the `ground_truth_counts.csv` to get **counting metrics (MAE, RMSE)**.
6.  **Visualize (`run_fiftyone.py`):**
    Visually inspect your ground truth dataset or your model's predictions at any stage.

## Configuration
This project uses two separate configuration files, managed by `robbenblick.utils.load_config`.

* **`configs/base_config.yaml`**
    * **Purpose:** The single source of truth for **single runs**.
    * **Used By:** `create_dataset.py`, `predict_tiled.py`, `run_fiftyone.py`, and `yolo.py` (for validation/single-predict).
    * **Content:** Defines static parameters like data paths (`dataset_output_dir`), model (`model`), and inference settings (`confidence_thresh`).

* **`configs/base_iter_config.yaml`**
    * **Purpose:** The configuration file for **experiments and tuning**.
    * **Used By:** `run_experiments.py`.
    * **Content:** Any parameter defined as a **YAML list** (e.g., `model: [yolov8n.pt, yolov8s.pt]`) will be iterated over. `run_experiments.py` will test every possible combination of all lists.

## Environment Setup

1.  Clone the repository:
    ```sh
    git clone git@github.com:ki-iw/robbenblick.git
    cd robbenblick
    ```

2.  Create the Conda environment:
    ```sh
    conda env create --file environment.yml
    conda activate RobbenBlick
    ```

3.  (Optional) Install pre-commit hooks:
    ```sh
    pre-commit install
    ```

## Core Scripts & Usage

### `create_dataset.py`
* **Purpose:** Converts raw CVAT-annotated images and XML files into a YOLO-compatible dataset, including tiling and label conversion.
* **How it works:**
    * Loads configuration from a config file.
    * Scans `data/raw/` for dataset subfolders.
    * Parses CVAT XML annotations and extracts polygons.
    * Tiles large images into smaller crops based on `imgsz` and `tile_overlap` from the config.
    * Converts polygon annotations to YOLO bounding box format for each tile.
    * Splits data into `train`, `val`, and `test` sets and writes them to `data/processed/dataset_yolo`.
    * Saves a `ground_truth_counts.csv` file in each raw dataset subfolder, providing a baseline for counting evaluation.
* **Run:**
    ```sh
    # Do a 'dry run' to see statistics without writing files
    python -m robbenblick.create_dataset --dry-run --config configs/base_config.yaml

    # Create the dataset, holding out dataset #4 as the test set
    python -m robbenblick.create_dataset --config configs/base_config.yaml --test-dir-index 4
    ```
* **Key Arguments:**
    * `--config`: Path to the `base_config.yaml` file.
    * `--dry-run`: Run in statistics-only mode.
    * `--test-dir-index`: 1-based index of the dataset subfolder to use as a hold-out test set.
    * `--val-ratio`: Ratio of the remaining data to use for validation.

### `run_experiments.py`
* **Purpose:** **This is the main training script.** It automates hyperparameter tuning by iterating over parameters defined in `base_iter_config.yaml`.
* **How it works:**
    * Finds all parameters in the config file that are lists (e.g., `freeze: [None, 10]`).
    * Generates a "variant" for every possible combination of these parameters.
    * For each variant, it calls `yolo.py --mode train` as a subprocess with a unique `run_id`.
    * After all runs are complete, it reads the `results.csv` from each run directory, sorts them by `mAP50`, and prints a final ranking table.
* **Run:**
    ```sh
    # Start the experiment run defined in the iteration config
    python -m robbenblick.run_experiments --config configs/base_iter_config.yaml

    # Run experiments and only show the top 5 results
    python -m robbenblick.run_experiments --config configs/base_iter_config.yaml --top-n 5
    ```

### `predict_tiled.py`
* **Purpose:** **This is the main inference script.** It runs a trained YOLOv8 model on new, full-sized images using Sliced Aided Hyper Inference (SAHI).
* **How it works:**
    * Loads a trained `best.pt` model specified by the `--run_id` argument.
    * Loads inference parameters (like `confidence_thresh`, `tile_overlap`) from the `base_config.yaml`.
    * Uses `get_sliced_prediction` from SAHI to perform tiled inference on each image.
    * Saves outputs, including visualized images (if `--save-visuals`), YOLO `.txt` labels (if `--save-yolo`), and a `detection_counts.csv` file.
* **Run:**
    ```sh
    # Run inference on a folder of new images and save the visual results
    python -m robbenblick.predict_tiled \
        --config configs/base_config.yaml \
        --run_id "best_run_from_experiments" \
        --source "data/new_images_to_count/" \
        --output-dir "data/inference_results/" \
        --save-visuals
    ```

### `evaluate_counts.py`
* **Purpose:** Evaluates the *counting* performance of a model by comparing its predicted counts against the ground truth counts.
* **How it works:**
    * Loads the `ground_truth_counts.csv` generated by `create_dataset.py`.
    * Loads the `detection_counts.csv` generated by `predict_tiled.py`.
    * Merges them by `image_name`.
    * Calculates and prints key regression metrics (MAE, RMSE, R²) to assess the accuracy of the object counting.
* **Run:**
    ```sh
    # Evaluate the counts from a specific run
    python -m robbenblick.evaluate_counts \
        --gt-csv "data/raw/dataset_02/ground_truth_counts.csv" \
        --pred-csv "data/inference_results/detection_counts.csv"
    ```

### `yolo.py`
* **Purpose:** The core engine for training, validation, and standard prediction. This script is called by `run_experiments.py` for training. You can use it directly for validation.
* **How it works:**
    * `--mode train`: Loads a base model (`yolov8s.pt`) and trains it on the dataset specified in the config.
    * `--mode validate`: Loads a *trained* model (`best.pt` from a run directory) and validates it against the `test` split defined in `dataset.yaml`. This provides **detection metrics (mAP)**.
    * `--mode predict`: Runs standard (non-tiled) YOLO prediction on a folder.
* **Run:**
    ```sh
    # Validate the 'test' set performance of a completed run
    python -m robbenblick.yolo \
        --config configs/base_config.yaml \
        --mode validate \
        --run_id "best_run_from_experiments"
    ```

### `run_fiftyone.py`
* **Purpose:** Visualizes datasets and predictions using FiftyOne.
* **How it works:**
    * `--dataset groundtruth`: Loads the processed YOLO dataset (images and ground truth labels) from `data/processed/`.
    * `--dataset predictions`: Loads images, runs a specified model (`--run_id`) on them, and displays the model's predictions.
* **Run:**
    ```sh
    # View the ground truth annotations for the 'val' split
    python -m robbenblick.run_fiftyone \
        --config configs/base_config.yaml \
        --dataset groundtruth \
        --split val \
        --recreate

    # View the predictions from 'my_best_run' on the 'test' split
    python -m robbenblick.run_fiftyone \
        --config configs/base_config.yaml \
        --dataset predictions \
        --split test \
        --run_id "my_best_run" \
        --recreate
    ```

### `streamlit_app.py`
* **Purpose:** Quick test runs with the trained model of your choice for counting the seals in the image(s) and visualization.
* **How it works:**
    * Loads the selected YOLO model from `runs/detect/`.
    * Upload images, run model, then displays the counts and model's predictions as image visualization.
* **Run:**
    ```sh
    # View the ground truth annotations for the 'val' split
    export PYTHONPATH=$PWD && streamlit run robbenblick/streamlit_app.py
    ```

##  Recommended Full Workflow

1.  **Add Raw Data:**
    * Place your first set of images and annotations in `data/raw/dataset_01/images/` and `data/raw/dataset_01/annotations.xml`.
    * Place your second set (e.g., from a different location) in `data/raw/dataset_02/images/` and `data/raw/dataset_02/annotations.xml`.

2.  **Create Dataset:**
    * Run `python -m robbenblick.create_dataset --dry-run` to see your dataset statistics. Note the indices of your datasets.
    * Let's say `dataset_02` is a good hold-out set. Run:
        `python -m robbenblick.create_dataset --config configs/base_config.yaml --test-dir-index 2`
    * This creates `data/raw/dataset_02/ground_truth_counts.csv` for later.

3.  **Find Best Model:**
    * Edit `configs/base_iter_config.yaml`. Define your experiments.
        ```yaml
        # Example: Test two models and two freeze strategies
        model: ['yolov8s.pt', 'yolov8m.pt']
        freeze: [None, 10]
        yolo_hyperparams:
          scale: [0.3, 0.5]
        ```
    * Run the experiments: `python -m robbenblick.run_experiments`.
    * Note the `run_id` of the top-ranked model, e.g., `iter_run_model_yolov8m.pt_freeze_10_scale_0.3`.

4.  **Validate on Test Set (Detection mAP):**
    * Check your best model's performance on the unseen test data:
        `python -m robbenblick.yolo --mode validate --run_id "iter_run_model_yolov8m.pt_freeze_10_scale_0.3" --config configs/base_config.yaml`
    * This tells you how well it *detects* objects (mAP).

5.  **Apply Model for Counting:**
    * Get a new folder of large, un-annotated images (e.g., `data/to_be_counted/`).
    * Run `predict_tiled.py`:
        `python -m robbenblick.predict_tiled --run_id "iter_run_model_yolov8m.pt_freeze_10_scale_0.3" --source "data/to_be_counted/" --output-dir "data/final_counts/" --save-visuals`
    * This creates `data/final_counts/detection_counts.csv`.

6.  **Evaluate Counting Performance (MAE, RMSE):**
    * Now, compare the predicted counts (Step 5) with the ground truth counts (Step 2). Let's assume your "to_be_counted" folder *was* your `dataset_02`.
        `python -m robbenblick.evaluate_counts --gt-csv "data/raw/dataset_02/ground_truth_counts.csv" --pred-csv "data/final_counts/detection_counts.csv"`
    * This gives you the final MAE, RMSE, and R² metrics for your **counting task**.

## Additional Notes
This repository contains only the source code of the project. The training data and the fine-tuned model weights are not included or published.

The repository is currently not being actively maintained. Future updates are not planned at this time.

For transparency, please note that the underlying model used throughout this project is based on **YOLOv8 by Ultralytics**.

## License
Copyright (c) 2025 **Birds on Mars**.

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This aligns with the license of the underlying **YOLOv8** model architecture used in this project.

Please note:
**Training data and fine-tuned model weights are not part of the licensed materials** and are not included in this repository.

For full details, see the LICENSE file.

## Troubleshooting

### FiftyOne: images (partially) not visible
Try using `--recreate` flag to force FiftyOne to reload the dataset:
```sh
python robbenblick/run_fiftyone.py --dataset groundtruth --split val --recreate
```

### FiftyOne: failed to bind port
If you get:
```
fiftyone.core.service.ServiceListenTimeout: fiftyone.core.service.DatabaseService failed to bind to port
```

Try killing any lingering `fiftyone` or `mongod` processes:
```sh
pkill -f fiftyone
pkill -f mongod
Then rerun your script.
```

# Collaborators 

The code for this project has been developed through a collaborative effort between [WWF Büro Ostsee](https://www.wwf.de/themen-projekte/projektregionen/ostsee) and [KI-Ideenwerkstatt](https://www.ki-ideenwerkstatt.de), technical implementation by [Birds on Mars](https://birdsonmars.com).

<p></p>
<a href="https://ki-ideenwerkstatt.de" target="_blank" rel="noopener noreferrer">
  <img src="assets/kiiw.jpg" alt="KI Ideenwerkstatt" height="100">
</a>
<p></p>
Technical realization
<br>
<a href="https://birdsonmars.com" target="_blank" rel="noopener noreferrer">
  <img src="assets/bom.jpg" alt="Birds On Mars" height="100">
</a>
<p></p>
An AI initiative by
<br>
<a href="https://www.bundesumweltministerium.de/" target="_blank" rel="noopener noreferrer">
  <img src="assets/bmukn.svg" alt="Bundesministerium für Umwelt, Klimaschutz, Naturschutz und nukleare Sicherheit" height="100">
</a>
<p></p>
In the context of
<br>
<a href="https://civic-coding.de" target="_blank" rel="noopener noreferrer">
  <img src="assets/civic.svg" alt="Civic Coding" height="100">
</a>
