"""Scene-specific resource context for pipeline execution."""

from pathlib import Path
from typing import Dict, List, Optional

from .assets import SceneAssets
from .config import SceneGenConfig


class SceneResourceContext:
    """Resource context for scene generation pipeline execution."""

    def __init__(
        self,
        config: SceneGenConfig,
        combined_user_assets: List = None,
        additional_material_libraries: List = None,
        **kwargs,
    ):
        """Initialize scene resource context.

        Args:
            config: Scene generation configuration
            combined_user_assets: List combining config + XML assets
            additional_material_libraries: Extra material libraries from XML
        """

        # Core configuration
        self.config = config
        self.dependency_outputs: Dict[str, Path | None] = {}
        self.kwargs = kwargs

        # Asset management
        self.config_assets = list(config.user_assets)
        self.xml_assets = (
            combined_user_assets[len(config.user_assets) :]
            if combined_user_assets
            else []
        )

        # Direct computed properties from config
        self.output_dir = config.scene_output_dir
        self.data_dir = config.data_dir
        self.meshes_dir = config.meshes_dir
        self.textures_dir = config.textures_dir
        self.scene_name = config.scene_name
        self.center_lat = config.location.center_lat
        self.center_lon = config.location.center_lon
        self.aoi_size_km = config.location.aoi_size_km
        self.target_resolution_m = config.processing.target_resolution_m

        # Scene-specific data
        self.assets = SceneAssets()
        self.additional_material_libraries = additional_material_libraries or []
        self.processed_objects: List = []
        self.scene_description: Optional[object] = None
        self.hamster_data_paths: Optional[Dict[str, Path]] = None

        # AOI polygon storage for geometric operations
        self._target_aoi_polygon: Optional[object] = None
        self._buffer_aoi_polygon: Optional[object] = None
        self._background_aoi_polygon: Optional[object] = None

    @property
    def user_assets(self):
        """Get combined user assets (config + XML assets)."""
        return self.config_assets + self.xml_assets

    @property
    def has_buffer(self) -> bool:
        """Check if buffer processing is enabled."""
        return self.config.has_buffer

    @property
    def has_background(self) -> bool:
        """Check if background processing is enabled."""
        return self.config.has_background

    @property
    def has_hamster(self) -> bool:
        """Check if HAMSTER data integration is enabled."""
        return self.config.hamster is not None and self.config.hamster.enabled
