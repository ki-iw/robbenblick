from pathlib import Path
from PIL import Image
import yaml
import torch

from robbenblick import logger


def convert_tif_to_png(src_folder, dst_folder):
    """
    Converts all .tif images in src_folder to .png and saves them in dst_folder.
    Args:
        src_folder (str): Source folder containing .tif images.
        dst_folder (str): Destination folder to save .png images.
    """
    src_folder = Path(src_folder)
    dst_folder = Path(dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)
    for tif_file in src_folder.iterdir():
        if tif_file.suffix.lower() == ".tif":
            png_file = dst_folder / (tif_file.stem + ".png")
            try:
                with Image.open(tif_file) as im:
                    im.save(png_file)
            except Exception as e:
                logger.error(f"Error converting {tif_file}: {e}")


def convert_xml_tif_to_png(xml_path, xml_path_png):
    """
    Replaces all occurrences of .tif with .png in the given XML file.
    Args:
        xml_path (str): Path to the XML file to modify.
    """
    xml_path = Path(xml_path)
    xml_path_png = Path(xml_path_png)
    xml_content = xml_path.read_text()
    xml_content = xml_content.replace(".tif", ".png")
    xml_content = xml_content.replace(".TIF", ".png")
    xml_path_png.write_text(xml_content)


def load_config(config_path: Path) -> dict | None:
    """Safely loads a YAML configuration file."""
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return None

    with open(config_path, "r") as f:
        try:
            config_data = yaml.safe_load(f)
            logger.info(f"Successfully loaded config from: {config_path}")
            return config_data
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file {config_path}: {e}")
            return None


def get_device() -> str:
    """Detects the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("Found NVIDIA CUDA GPU. Using GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Found Apple Silicon GPU. Using MPS.")
    else:
        device = "cpu"
        logger.warning("No GPU found. Using CPU.")
    return device
