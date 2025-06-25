"""S2GOS Scene Generator - Generate 3D scenes from earth observation data."""

__version__ = "0.1.0"

# Lazy imports to avoid dependency issues when not needed
def _import_core():
    from .core import SceneGenerationConfig, SceneGenerationPipeline, SceneAssets
    return SceneGenerationConfig, SceneGenerationPipeline, SceneAssets

def _import_scene():
    from .scene import SceneDescription, MaterialDefinition
    return SceneDescription, MaterialDefinition

def _import_simulation():
    from .simulation import SimulationConfig
    from .scene_loading import create_s2gos_scene
    return SimulationConfig, create_s2gos_scene


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
    SceneGenerationConfig, SceneGenerationPipeline, _ = _import_core()
    
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


# Make common classes available for direct import
def __getattr__(name):
    """Lazy import for module attributes."""
    if name in ["SceneGenerationConfig", "SceneGenerationPipeline", "SceneAssets"]:
        return getattr(__import__('s2gos_generator.core', fromlist=[name]), name)
    elif name in ["SceneDescription", "MaterialDefinition"]:
        return getattr(__import__('s2gos_generator.scene', fromlist=[name]), name)
    elif name == "SimulationConfig":
        return getattr(__import__('s2gos_generator.simulation', fromlist=[name]), name)
    elif name == "create_s2gos_scene":
        return getattr(__import__('s2gos_generator.scene_loading', fromlist=[name]), name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")