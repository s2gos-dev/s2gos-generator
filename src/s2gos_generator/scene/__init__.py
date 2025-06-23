"""
S2GOS Scene Generation Module

Scene configuration, pipeline orchestration, and scene description format.
"""

from .pipeline import SceneGenerationPipeline, SceneConfig, SceneAssets
from .description import SceneDescription, MaterialDefinition, SceneGeometry, create_default_materials

__version__ = "0.1.0"

__all__ = [
    # Pipeline orchestrator
    "SceneGenerationPipeline",
    "SceneConfig", 
    "SceneAssets",
    
    # Scene description format
    "SceneDescription",
    "MaterialDefinition",
    "SceneGeometry", 
    "create_default_materials",
]

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
) -> SceneAssets:
    """Convenience function to generate a complete scene with minimal configuration."""
    from pathlib import Path
    
    config = SceneConfig(
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