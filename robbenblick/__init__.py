from importlib import metadata
from importlib.metadata import version
from pathlib import Path

import yaml
from dotenv import load_dotenv
from dotmap import DotMap
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
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs"

logger.info(f"Data path set to: {DATA_PATH}")
logger.info(f"Config path set to: {CONFIG_PATH}")


dataset_config = DotMap(
    yaml.safe_load((CONFIG_PATH / "create_dataset.yaml").read_text())
)
model_config = DotMap(yaml.safe_load((CONFIG_PATH / "model.yaml").read_text()))

__all__ = ["DATA_PATH", "dataset_config", "model_config"]
