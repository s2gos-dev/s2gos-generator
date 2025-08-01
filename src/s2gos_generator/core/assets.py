from dataclasses import dataclass
from upath import UPath
from typing import Dict, Optional

from s2gos_utils.io.paths import optional_str


@dataclass
class SceneAssets:
    """Container for generated scene assets."""

    dem_file: Optional[UPath] = None
    landcover_file: Optional[UPath] = None
    mesh_file: Optional[UPath] = None
    selection_texture_file: Optional[UPath] = None
    preview_texture_file: Optional[UPath] = None
    config_file: Optional[UPath] = None
    scene_description_file: Optional[UPath] = None

    buffer_dem_file: Optional[UPath] = None
    buffer_landcover_file: Optional[UPath] = None
    buffer_mesh_file: Optional[UPath] = None
    buffer_selection_texture_file: Optional[UPath] = None
    buffer_preview_texture_file: Optional[UPath] = None

    background_landcover_file: Optional[UPath] = None
    background_selection_texture_file: Optional[UPath] = None
    background_preview_texture_file: Optional[UPath] = None

    def to_dict(self) -> Dict:
        """Convert assets to dictionary."""
        return {
            "dem_file": optional_str(self.dem_file),
            "landcover_file": optional_str(self.landcover_file),
            "mesh_file": optional_str(self.mesh_file),
            "selection_texture_file": optional_str(self.selection_texture_file),
            "preview_texture_file": optional_str(self.preview_texture_file),
            "config_file": optional_str(self.config_file),
            "scene_description_file": optional_str(self.scene_description_file),
            "buffer_dem_file": optional_str(self.buffer_dem_file),
            "buffer_landcover_file": optional_str(self.buffer_landcover_file),
            "buffer_mesh_file": optional_str(self.buffer_mesh_file),
            "buffer_selection_texture_file": optional_str(self.buffer_selection_texture_file),
            "buffer_preview_texture_file": optional_str(self.buffer_preview_texture_file),
            "background_landcover_file": optional_str(self.background_landcover_file),
            "background_selection_texture_file": optional_str(self.background_selection_texture_file),
            "background_preview_texture_file": optional_str(self.background_preview_texture_file),
        }
