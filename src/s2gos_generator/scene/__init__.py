"""Scene description and materials."""

from .description import SceneDescription, SceneGeometry
from .materials import MaterialDefinition, create_default_materials

__all__ = ["SceneDescription", "SceneGeometry", "MaterialDefinition", "create_default_materials"]