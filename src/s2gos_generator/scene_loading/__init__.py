"""Scene loading and configuration utilities."""

# Conditional import for SceneLoader (requires eradiate)
try:
    from .loader import SceneLoader
    _LOADER_AVAILABLE = True
except ImportError:
    _LOADER_AVAILABLE = False

from .config import SceneConfig, SceneMetadata, create_s2gos_scene

__all__ = [
    "SceneConfig",
    "SceneMetadata", 
    "create_s2gos_scene"
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