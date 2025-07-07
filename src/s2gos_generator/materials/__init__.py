"""Material definitions and JSON-based configuration loader for radiative transfer simulations."""

from .definitions import Material
from .loader import MaterialConfigLoader, load_materials, get_landcover_mapping

__all__ = [
    "Material",
    "MaterialConfigLoader",
    "load_materials",
    "get_landcover_mapping"
]