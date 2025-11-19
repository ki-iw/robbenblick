import streamlit as st
from streamlit.errors import StreamlitAPIException
import pandas as pd
from PIL import Image
from pathlib import Path
import time
import shutil
import plotly.express as px

from sahi.models.ultralytics import UltralyticsDetectionModel
from robbenblick.utils import load_config
from robbenblick.inference import load_detection_model, run_inference


st.set_page_config(layout="wide", page_title="Seal Detection")
st.title("🦭 Seal Detection")


@st.cache_resource
def cached_load_model(
    model_path: Path, conf_thresh: float
) -> UltralyticsDetectionModel | None:
    """
    Wrapper, um die externe Ladefunktion in Streamlit zu cachen.
    """
    # Ruft die ausgelagerte Funktion auf
    return load_detection_model(model_path=model_path, conf_thresh=conf_thresh)


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

# Hardcoded model and inference parameters from config
try:
    RUN_ID = "iter_run_model_yolov8m.pt_freeze_10_epochs_150"
    # config_data.get("run_id", "base_run")
    MODEL_PATH = Path(f"runs/detect/{RUN_ID}/weights/best.pt")
    SLICE_SIZE = config_data["yolo_hyperparams"]["imgsz"]
    OVERLAP_RATIO = config_data.get("tile_overlap", 0.2)
    CONF_THRESH = config_data.get("confidence_thresh", 0.25)
except KeyError as e:
    st.error(f"Error in 'base_config.yaml': Missing key {e}")
    st.stop()

# --- Sidebar: Configuration & Model Status ---

st.sidebar.header("Model & Configuration")
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

# --- Main Interface ---

# 1. Image Upload
st.header("1. Upload Images")
uploaded_files = st.file_uploader(
    "Choose JPG, PNG, or TIF images",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)

saved_images = []
if uploaded_files:
    st.info(f"Saving {len(uploaded_files)} images temporarily...")
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
        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(df_counts)
            total_seals = df_counts["Detected Seal Count"].sum()
            st.metric("Total Seals", total_seals)

        with col2:
            st.markdown("Count per Image:")
            chart_data = df_counts.rename(columns={"Detected Seal Count": "Count"})
            st.bar_chart(chart_data.set_index("Image"))

        csv_path = OUTPUT_DIR / "detection_counts.csv"
        df_counts.to_csv(csv_path, index=False)
        st.info(f"Count results saved to: {csv_path}")

        st.subheader("Visualized Results")

        if not results:
            st.info("No images available for visualization.")
            st.stop()

        tab_names = [r["image_name"] for r in results]

        if not all(tab_names):
            st.error("Error: Some images have no names.")
            st.stop()

        try:
            tabs = st.tabs(tab_names)
        except StreamlitAPIException as e:
            st.error(f"Could not create result tabs: {e}")
            st.info("This can happen if image names contain duplicates or are invalid.")
            st.stop()

        for i, tab in enumerate(tabs):
            with tab:
                result_data = results[i]
                try:
                    visual_img = Image.open(result_data["visual_path"])
                    st.markdown(f"**{result_data['count']}** seals detected.")
                    # Lade Bild mit PIL
                    img = Image.open(result_data["visual_path"])

                    # Erstelle interaktives Plotly-Diagramm
                    fig = px.imshow(img)

                    # Konfiguriere Layout: Entferne Ränder, setze Modus auf "Pan" (Verschieben)
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0), height=600, dragmode="pan"
                    )

                    # Zeige das Diagramm an
                    st.plotly_chart(fig, width="stretch")
                except FileNotFoundError as e:
                    st.error(f"Image file not found: {e}")
                except Exception as e:
                    st.error(f"Could not load image: {e}")
else:
    st.info("Upload images and start inference to see results.")
