"""S2GOS Scene Generator - Generate 3D scenes from earth observation data."""

import logging

__version__ = "0.1.0"

# Configure logging for the entire package
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import lightweight modules directly, defer heavy ones
from .scene import SceneDescription, SceneConfig, SceneMetadata, create_s2gos_scene
from .simulation import SimulationConfig
from .core.exceptions import (
    S2GOSError, DataNotFoundError, ConfigurationError, ProcessingError,
    RegridError, GeospatialError, MaterialError
)


def generate_scene(
    center_lat: float,
    center_lon: float, 
    aoi_size_km: float,
    dem_index_path: str,
    dem_root_dir: str,
    landcover_index_path: str,
    landcover_root_dir: str,
    output_dir: str,
    scene_name: str,
    target_resolution_m: float = 30.0
):
    """Convenience function to generate a complete scene."""
    from pathlib import Path
    from .core import SceneGenerationConfig, SceneGenerationPipeline
    
    config = SceneGenerationConfig(
        center_lat=center_lat,
        center_lon=center_lon,
        aoi_size_km=aoi_size_km,
        dem_index_path=Path(dem_index_path),
        dem_root_dir=Path(dem_root_dir),
        landcover_index_path=Path(landcover_index_path),
        landcover_root_dir=Path(landcover_root_dir),
        output_dir=Path(output_dir),
        scene_name=scene_name,
        target_resolution_m=target_resolution_m
    )
    
    pipeline = SceneGenerationPipeline(config)
    return pipeline.run_full_pipeline()


# Make core classes available with lazy loading
def __getattr__(name):
    if name in ["SceneGenerationConfig", "SceneGenerationPipeline", "SceneAssets"]:
        from .core import SceneGenerationConfig, SceneGenerationPipeline, SceneAssets
        return {"SceneGenerationConfig": SceneGenerationConfig, 
                "SceneGenerationPipeline": SceneGenerationPipeline, 
                "SceneAssets": SceneAssets}[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all available classes
__all__ = [
    "SceneGenerationConfig", "SceneGenerationPipeline", "SceneAssets",
    "SceneDescription", "SceneConfig", "SceneMetadata", "create_s2gos_scene",
    "SimulationConfig", "generate_scene",
    "S2GOSError", "DataNotFoundError", "ConfigurationError", "ProcessingError",
    "RegridError", "GeospatialError", "MaterialError"
]