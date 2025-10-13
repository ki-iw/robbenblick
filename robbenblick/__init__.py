from importlib import metadata
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger  # noqa: F401

load_dotenv()

try:
    __version__ = version("robbenblick")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
finally:
    del metadata  # optional, avoids polluting the results of dir(__package__)


DATA_PATH = Path(__file__).resolve().parent.parent / "data"

MODEL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs"

logger.info(f"Data path set to: {DATA_PATH}")
logger.info(f"Config path set to: {MODEL_CONFIG_PATH}")

__all__ = [
    "DATA_PATH",
    "MODEL_CONFIG_PATH",
]
