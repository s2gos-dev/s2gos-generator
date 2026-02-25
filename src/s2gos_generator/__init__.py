import logging

from s2gos_utils.scene import SceneDescription

from .core import SceneGenConfig, SceneGenerationPipeline, SceneResourceContext
from .core.config import BackgroundConfig, BufferConfig, SnowConfig, create_scene_config
from .core.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    GeospatialError,
    MaterialError,
    ProcessingError,
    RegridError,
    S2GOSError,
)
from .scene import create_s2gos_scene

# Configure logging for the entire package
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def __getattr__(name):
    if name in ["SceneGenConfig", "SceneGenerationPipeline", "SceneAssets"]:
        from .core import SceneAssets, SceneGenConfig, SceneGenerationPipeline

        return {
            "SceneGenConfig": SceneGenConfig,
            "SceneGenerationPipeline": SceneGenerationPipeline,
            "SceneAssets": SceneAssets,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "SceneGenConfig",
    "SceneGenerationPipeline",
    "SceneResourceContext",
    "SceneAssets",
    "SceneDescription",
    "create_scene_config",
    "create_s2gos_scene",
    "SnowConfig",
    "BufferConfig",
    "BackgroundConfig",
    "S2GOSError",
    "DataNotFoundError",
    "ConfigurationError",
    "ProcessingError",
    "RegridError",
    "GeospatialError",
    "MaterialError",
]
