"""Radiative transfer simulation configurations."""

from .config import SimulationConfig, MaterialRegistry, create_s2gos_scene
from .sensors import Sensor, PerspectiveSensor, DistantSensor

# Conditional import for SceneLoader (requires eradiate)
try:
    from .loader import SceneLoader
    _LOADER_AVAILABLE = True
except ImportError:
    _LOADER_AVAILABLE = False

__all__ = [
    "SimulationConfig", "MaterialRegistry", "create_s2gos_scene",
    "Sensor", "PerspectiveSensor", "DistantSensor"
]

if _LOADER_AVAILABLE:
    __all__.append("SceneLoader")

def __getattr__(name):
    if name == "SceneLoader":
        if not _LOADER_AVAILABLE:
            raise ImportError("SceneLoader requires eradiate dependencies")
        from .loader import SceneLoader
        return SceneLoader
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")