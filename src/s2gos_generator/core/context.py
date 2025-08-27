"""Scene-specific resource context for pipeline execution."""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel
from upath import UPath

from ..resource_graph.resource_registry import ResourceContext

from .config import SceneGenConfig
from .assets import SceneAssets


class SceneResourceContext(ResourceContext):
    """Extended resource context for scene generation with scene-specific data."""
    
    config: SceneGenConfig
    assets: SceneAssets
    
    # Computed properties
    output_dir: Optional[UPath] = None
    data_dir: Optional[UPath] = None
    meshes_dir: Optional[UPath] = None
    textures_dir: Optional[UPath] = None
    
    # Additional material libraries for XML imports
    additional_material_libraries: List = []
    
    # Processed objects storage
    processed_objects: List = []
    
    # Final scene description
    scene_description: Optional[object] = None
    
    # HAMSTER data paths storage
    hamster_data_paths: Optional[Dict[str, Path]] = None
    
    # Cached AOI polygons
    _target_aoi_polygon: Optional[object] = None
    _buffer_aoi_polygon: Optional[object] = None  
    _background_aoi_polygon: Optional[object] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, config: SceneGenConfig, additional_material_libraries=None, **data):
        """Initialize scene resource context."""
        # Set up all required data for parent initialization, including required fields
        context_data = {
            'config': config,
            'assets': SceneAssets(),
            'additional_material_libraries': additional_material_libraries or [],
            'processed_objects': [],
            'scene_description': None,
            'hamster_data_paths': None,
            'kwargs': data,
            'dependency_outputs': {}
        }
        
        super().__init__(**context_data)
        
        # Set additional computed attributes
        self.output_dir = config.scene_output_dir
        self.data_dir = config.data_dir
        self.meshes_dir = config.meshes_dir
        self.textures_dir = config.textures_dir
        
        # Initialize AOI polygon storage
        self._target_aoi_polygon = None
        self._buffer_aoi_polygon = None
        self._background_aoi_polygon = None
    
    @property
    def scene_name(self) -> str:
        """Get scene name from config."""
        return self.config.scene_name
    
    @property
    def center_lat(self) -> float:
        """Get center latitude from config."""
        return self.config.location.center_lat
    
    @property
    def center_lon(self) -> float:
        """Get center longitude from config."""
        return self.config.location.center_lon
    
    @property
    def aoi_size_km(self) -> float:
        """Get AOI size from config."""
        return self.config.location.aoi_size_km
    
    @property
    def target_resolution_m(self) -> float:
        """Get target resolution from config."""
        return self.config.processing.target_resolution_m
    
    @property
    def has_buffer(self) -> bool:
        """Check if buffer is enabled."""
        return self.config.has_buffer
    
    @property
    def has_hamster(self) -> bool:
        """Check if HAMSTER is enabled."""
        return self.config.hamster is not None and self.config.hamster.enabled
    
    @property
    def has_user_assets(self) -> bool:
        """Check if user assets are configured."""
        return len(self.config.user_assets) > 0