"""Assets container for generated scene components."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SceneAssets:
    """Container for generated scene assets."""
    dem_file: Optional[Path] = None
    landcover_file: Optional[Path] = None
    mesh_file: Optional[Path] = None
    selection_texture_file: Optional[Path] = None
    preview_texture_file: Optional[Path] = None
    config_file: Optional[Path] = None
    scene_description_file: Optional[Path] = None
    
    buffer_dem_file: Optional[Path] = None
    buffer_landcover_file: Optional[Path] = None
    buffer_mesh_file: Optional[Path] = None
    buffer_selection_texture_file: Optional[Path] = None
    buffer_preview_texture_file: Optional[Path] = None
    
    background_landcover_file: Optional[Path] = None
    background_selection_texture_file: Optional[Path] = None
    background_preview_texture_file: Optional[Path] = None
    
    def to_dict(self) -> Dict:
        """Convert assets to dictionary."""
        return {
            'dem_file': str(self.dem_file) if self.dem_file else None,
            'landcover_file': str(self.landcover_file) if self.landcover_file else None,
            'mesh_file': str(self.mesh_file) if self.mesh_file else None,
            'selection_texture_file': str(self.selection_texture_file) if self.selection_texture_file else None,
            'preview_texture_file': str(self.preview_texture_file) if self.preview_texture_file else None,
            'config_file': str(self.config_file) if self.config_file else None,
            'scene_description_file': str(self.scene_description_file) if self.scene_description_file else None,
            'buffer_dem_file': str(self.buffer_dem_file) if self.buffer_dem_file else None,
            'buffer_landcover_file': str(self.buffer_landcover_file) if self.buffer_landcover_file else None,
            'buffer_mesh_file': str(self.buffer_mesh_file) if self.buffer_mesh_file else None,
            'buffer_selection_texture_file': str(self.buffer_selection_texture_file) if self.buffer_selection_texture_file else None,
            'buffer_preview_texture_file': str(self.buffer_preview_texture_file) if self.buffer_preview_texture_file else None,
            'background_landcover_file': str(self.background_landcover_file) if self.background_landcover_file else None,
            'background_selection_texture_file': str(self.background_selection_texture_file) if self.background_selection_texture_file else None,
            'background_preview_texture_file': str(self.background_preview_texture_file) if self.background_preview_texture_file else None
        }