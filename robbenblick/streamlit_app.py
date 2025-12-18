import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import time
import shutil
import plotly.express as px

from sahi.models.ultralytics import UltralyticsDetectionModel
from robbenblick.utils import load_config
from robbenblick.inference import load_detection_model, run_inference

st.set_page_config(
    layout="wide", page_title="Seal Detection", initial_sidebar_state="collapsed"
)
st.title("🦭 Seal Detection")


@st.cache_resource
def cached_load_model(
    model_path: Path, conf_thresh: float
) -> UltralyticsDetectionModel | None:
    """
    Wrapper to cache the external loading function in Streamlit.
    """
    # Calls the external function
    return load_detection_model(model_path=model_path, conf_thresh=conf_thresh)


@st.cache_data(show_spinner=False, max_entries=3)
def get_plotly_figure(image_path: Path, do_downsample: bool):
    """
    Loads the image and creates the Plotly figure.
    Uses LRU caching: Only the last 3 viewed images are kept in RAM.
    """
    img = Image.open(image_path)

    if do_downsample:
        # Max 2000px edge length, maintains aspect ratio
        img.thumbnail((2000, 2000))

    fig = px.imshow(img)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        dragmode=False,  # "pan"
    )
    return fig


# --- App Initialization ---

TEMP_DIR = Path("data/streamlit_temp")
UPLOAD_DIR = TEMP_DIR / "uploads"
OUTPUT_DIR = TEMP_DIR / "output"

# Clean up on start
if "app_loaded" not in st.session_state:
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    st.session_state.app_loaded = True

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load configuration (using imported function)
CONFIG_PATH = Path("configs/base_config.yaml")
config_data = load_config(CONFIG_PATH)

if config_data is None:
    st.error("Could not load 'base_config.yaml'. Make sure the file exists.")
    st.stop()

# --- Sidebar: Configuration & Model Selection ---

st.sidebar.header("Model & Configuration")

RUNS_DIR = Path("runs/detect")

# Pull config-driven values (same as before)
try:
    SLICE_SIZE = config_data["yolo_hyperparams"]["imgsz"]
except KeyError as e:
    st.error(f"Error in 'base_config.yaml': Missing key {e}")
    st.stop()

OVERLAP_RATIO = config_data.get("tile_overlap", 0.2)
CONF_THRESH = config_data.get("confidence_thresh", 0.25)

# Collect only valid runs (those with weights/best.pt)
run_folders = [
    p.name for p in RUNS_DIR.iterdir()
    if p.is_dir() and (p / "weights" / "best.pt").exists()
] if RUNS_DIR.exists() else []

if not run_folders:
    st.sidebar.error(f"No valid YOLO runs found in {RUNS_DIR}")
    st.stop()

run_folders = sorted(run_folders)

# Default to last selected run or most recent one
default_run = st.session_state.get("selected_run", run_folders[-1])

RUN_ID = st.sidebar.selectbox(
    "Select YOLO Model",
    options=run_folders,
    index=run_folders.index(default_run) if default_run in run_folders else len(run_folders) - 1,
)

st.session_state["selected_run"] = RUN_ID

MODEL_PATH = RUNS_DIR / RUN_ID / "weights" / "best.pt"

# --- Sidebar: Model Status ---

st.sidebar.info(f"**Run ID:** `{RUN_ID}`")
st.sidebar.info(f"**Model Path:** `{MODEL_PATH}`")

if MODEL_PATH.exists():
    st.sidebar.success("Model file (best.pt) found.")
else:
    st.sidebar.error("Model file (best.pt) NOT found!")
    st.stop()


st.sidebar.subheader("Inference Parameters")
st.sidebar.markdown(f"**Confidence Threshold:** `{CONF_THRESH}`")
st.sidebar.markdown(f"**Tile Size (imgsz):** `{SLICE_SIZE}x{SLICE_SIZE}`")
st.sidebar.markdown(f"**Tile Overlap:** `{OVERLAP_RATIO}`")

st.sidebar.subheader("Visualization Settings")
USE_DOWNSAMPLING = st.sidebar.checkbox(
    "Downsample Images for Display",
    value=False,
    help="Reduces image resolution for display to prevent browser crashes.",
)

# --- Main Interface ---

# 1. Image Upload
st.header("1. Upload Images")
uploaded_files = st.file_uploader(
    "Choose JPG, PNG, or TIF images",
    type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
    accept_multiple_files=True,
)

saved_images = []
for f in uploaded_files:
    save_path = UPLOAD_DIR / f.name
    with open(save_path, "wb") as out_f:
        out_f.write(f.read())
    saved_images.append(save_path)
st.success(f"{len(saved_images)} images ready for inference.")

# 2. Start Inference
st.header("2. Start Inference")

if "inference_done" not in st.session_state:
    st.session_state.inference_done = False
    st.session_state.results = []

if st.button("Start Seal Count", disabled=(len(saved_images) == 0)):
    st.session_state.inference_done = False
    st.session_state.results = []

    # Load model (using cached wrapper)
    with st.spinner(f"Loading model '{RUN_ID}'..."):
        detection_model = cached_load_model(MODEL_PATH, CONF_THRESH)

    if detection_model:
        st.success("Model loaded successfully.")

        if (OUTPUT_DIR / "visuals").exists():
            shutil.rmtree(OUTPUT_DIR / "visuals")

        progress_bar = st.progress(0, text="Starting inference...")
        start_time = time.time()
        results_list = []

        for i, img_path in enumerate(saved_images):
            progress_text = (
                f"Processing image {i + 1}/{len(saved_images)}: {img_path.name}"
            )
            progress_bar.progress((i) / len(saved_images), text=progress_text)

            with st.spinner(progress_text):
                result_data = run_inference(
                    detection_model=detection_model,
                    image_path=img_path,
                    output_dir=OUTPUT_DIR,
                    slice_size=SLICE_SIZE,
                    overlap_ratio=OVERLAP_RATIO,
                    save_visuals=True,
                    hide_labels=True,
                    hide_conf=True,
                )

            if result_data["visual_path"]:
                results_list.append(
                    {
                        "image_name": img_path.name,
                        "count": result_data["count"],
                        "visual_path": result_data["visual_path"],
                    }
                )

        progress_bar.progress(1.0, text="Inference complete!")
        end_time = time.time()

        st.session_state.results = results_list
        st.session_state.inference_done = True

        st.success(
            f"Inference for {len(saved_images)} images completed in {end_time - start_time:.2f} seconds."
        )

# 3. Display Results
st.header("3. Results")

if st.session_state.inference_done:
    results = st.session_state.results
    if not results:
        st.warning("Inference finished, but no results found.")
    else:
        df_counts = pd.DataFrame(
            [
                {"Image": r["image_name"], "Detected Seal Count": r["count"]}
                for r in results
            ]
        )

        st.subheader("Detection Overview")

        st.dataframe(df_counts)

        csv_path = OUTPUT_DIR / "detection_counts.csv"
        df_counts.to_csv(csv_path, index=False)

        st.subheader("Visualized Results")

        image_names = [r["image_name"] for r in results]
        selected_img_name = st.selectbox(
            "Select an image for detailed view:", image_names
        )

        # Filter for the matching result
        selected_result = next(
            (r for r in results if r["image_name"] == selected_img_name), None
        )

        if selected_result:
            try:
                visual_path = selected_result["visual_path"]
                count = selected_result["count"]

                st.markdown(f"**{count}** seals detected in `{selected_img_name}`.")

                with st.spinner("Loading visualization..."):
                    # Use cached function to speed up switching between images
                    fig = get_plotly_figure(visual_path, USE_DOWNSAMPLING)
                    st.plotly_chart(fig, width="stretch")

            except FileNotFoundError as e:
                st.error(f"Image file not found: {e}")
            except Exception as e:
                st.error(f"Could not load image: {e}")

else:
    st.info("Upload images and start inference to see results.")
