"""Configuration for scene generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class SceneGenerationConfig:
    """Configuration for scene generation pipeline."""
    center_lat: float
    center_lon: float
    aoi_size_km: float
    
    dem_index_path: Path
    dem_root_dir: Path
    landcover_index_path: Path
    landcover_root_dir: Path
    
    output_dir: Path
    scene_name: str
    
    target_resolution_m: float = 30.0
    generate_texture_preview: bool = True
    handle_dem_nans: bool = True
    dem_fillna_value: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            'center_lat': self.center_lat,
            'center_lon': self.center_lon,
            'aoi_size_km': self.aoi_size_km,
            'dem_index_path': str(self.dem_index_path),
            'dem_root_dir': str(self.dem_root_dir),
            'landcover_index_path': str(self.landcover_index_path),
            'landcover_root_dir': str(self.landcover_root_dir),
            'output_dir': str(self.output_dir),
            'scene_name': self.scene_name,
            'target_resolution_m': self.target_resolution_m,
            'generate_texture_preview': self.generate_texture_preview,
            'handle_dem_nans': self.handle_dem_nans,
            'dem_fillna_value': self.dem_fillna_value
        }