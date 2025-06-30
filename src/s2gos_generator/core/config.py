"""Configuration for scene generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, Union
import os


def _serialize_path(path: Union[Path, str, None]) -> Optional[str]:
    """Serialize Path object to string using __fspath__ protocol."""
    if path is None:
        return None
    return os.fspath(path)


def _deserialize_path(path_str: Optional[str]) -> Optional[Path]:
    """Deserialize string to Path object."""
    if path_str is None:
        return None
    return Path(path_str)


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
    
    enable_buffer: bool = False
    buffer_size_km: Optional[float] = None
    buffer_resolution_m: float = 100.0 

    background_elevation: Optional[float] = None
    background_material: str = "water"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'center_lat': self.center_lat,
            'center_lon': self.center_lon,
            'aoi_size_km': self.aoi_size_km,
            'dem_index_path': _serialize_path(self.dem_index_path),
            'dem_root_dir': _serialize_path(self.dem_root_dir),
            'landcover_index_path': _serialize_path(self.landcover_index_path),
            'landcover_root_dir': _serialize_path(self.landcover_root_dir),
            'output_dir': _serialize_path(self.output_dir),
            'scene_name': self.scene_name,
            'target_resolution_m': self.target_resolution_m,
            'generate_texture_preview': self.generate_texture_preview,
            'handle_dem_nans': self.handle_dem_nans,
            'dem_fillna_value': self.dem_fillna_value,
            'enable_buffer': self.enable_buffer,
            'buffer_size_km': self.buffer_size_km,
            'buffer_resolution_m': self.buffer_resolution_m,
            'background_elevation': self.background_elevation,
            'background_material': self.background_material
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneGenerationConfig':
        """Create config from dictionary with proper Path reconstruction."""
        return cls(
            center_lat=data['center_lat'],
            center_lon=data['center_lon'],
            aoi_size_km=data['aoi_size_km'],
            dem_index_path=_deserialize_path(data['dem_index_path']),
            dem_root_dir=_deserialize_path(data['dem_root_dir']),
            landcover_index_path=_deserialize_path(data['landcover_index_path']),
            landcover_root_dir=_deserialize_path(data['landcover_root_dir']),
            output_dir=_deserialize_path(data['output_dir']),
            scene_name=data['scene_name'],
            target_resolution_m=data.get('target_resolution_m', 30.0),
            generate_texture_preview=data.get('generate_texture_preview', True),
            handle_dem_nans=data.get('handle_dem_nans', True),
            dem_fillna_value=data.get('dem_fillna_value', 0.0),
            enable_buffer=data.get('enable_buffer', False),
            buffer_size_km=data.get('buffer_size_km'),
            buffer_resolution_m=data.get('buffer_resolution_m', 100.0),
            background_elevation=data.get('background_elevation'),
            background_material=data.get('background_material', 'water')
        )