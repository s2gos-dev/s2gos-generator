"""Material definitions and registry for radiative transfer simulations."""

from .definitions import Material
from .registry import MaterialRegistry, create_s2gos_materials

__all__ = [
    "Material",
    "MaterialRegistry",
    "create_s2gos_materials"
]