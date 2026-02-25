"""Scene-specific resource context for pipeline execution."""

import logging
import random
from typing import Dict, List, Optional

import numpy as np
from upath import UPath

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
        self.dependency_outputs: Dict[str, UPath | None] = {}
        self.kwargs = kwargs

        # Asset management
        self.config_assets = list(config.user_assets)
        self.xml_assets = (
            combined_user_assets[len(config.user_assets) :]
            if combined_user_assets
            else []
        )

        # Direct computed properties from config
        self.output_dir = config.scene_output_dir.upath
        self.data_dir = config.data_dir.upath
        self.meshes_dir = config.meshes_dir.upath
        self.textures_dir = config.textures_dir.upath
        self.scene_name = config.scene_name
        self.center_lat = config.location.center_lat
        self.center_lon = config.location.center_lon
        self.aoi_size_km = config.location.aoi_size_km
        self.target_resolution_m = config.target_resolution_m

        # Scene-specific data
        self.assets = SceneAssets()
        self.additional_material_libraries = additional_material_libraries or []
        self.processed_objects: List = []
        self.vegetation_exclusion_zones: List = []
        self.scene_description: Optional[object] = None
        self.hamster_data_paths: Optional[Dict[str, UPath]] = None

        # AOI polygon storage for geometric operations
        self._target_aoi_polygon: Optional[object] = None
        self._buffer_aoi_polygon: Optional[object] = None
        self._background_aoi_polygon: Optional[object] = None

        self._coord_system: Optional[object] = None

        # Seed random generators for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            logging.info(
                f"Random seed set to {config.random_seed} for reproducible generation"
            )

    @property
    def user_assets(self):
        """Get combined user assets (config + XML assets)."""
        return self.config_assets + self.xml_assets

    @property
    def has_buffer(self) -> bool:
        """Check if buffer processing is enabled."""
        return self.config.buffer is not None

    @property
    def has_background(self) -> bool:
        """Check if background processing is enabled."""
        return self.config.background is not None

    @property
    def has_hamster(self) -> bool:
        """Check if HAMSTER data integration is enabled."""
        return self.config.hamster is not None and self.config.hamster.enabled

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        """Get cached coordinate system for this scene.

        Returns:
            CoordinateSystem instance for this scene's center location
        """
        if self._coord_system is None:
            from s2gos_utils.coordinates import CoordinateSystem

            self._coord_system = CoordinateSystem(
                center_lat=self.center_lat, center_lon=self.center_lon
            )
        return self._coord_system
