"""Scene generation resources for DAG execution."""

# Import all resource modules to register them with the ResourceRegistry
from . import aoi
from . import dem
from . import landcover
from . import mesh
from . import texture
from . import assets
from . import hamster
from . import scene

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