from importlib import metadata
from importlib.metadata import version

from dotenv import load_dotenv
import os
from loguru import logger  # noqa: F401

load_dotenv()

try:
    __version__ = version("robben_blick")
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
finally:
    del metadata  # optional, avoids polluting the results of dir(__package__)


DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

logger.info("Data path set to: {}", DATA_PATH)

__all__ = [
    "DATA_PATH",
]