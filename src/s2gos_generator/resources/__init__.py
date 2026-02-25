"""Scene generation resources for DAG execution."""

# Import all resource modules to register them with the ResourceRegistry
from . import aoi, assets, dem, hamster, landcover, mesh, scene, texture

__all__ = [
    "aoi",
    "dem",
    "landcover",
    "mesh",
    "texture",
    "assets",
    "hamster",
    "scene",
]
