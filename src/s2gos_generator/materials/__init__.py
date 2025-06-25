"""Material definitions and registry for radiative transfer simulations."""

from .definitions import Material, MaterialDefinition
from .registry import MaterialRegistry, create_s2gos_materials, create_default_materials

__all__ = [
    "Material",
    "MaterialDefinition", 
    "MaterialRegistry",
    "create_s2gos_materials",
    "create_default_materials"
]