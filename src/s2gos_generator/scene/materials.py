"""Material definitions for scene descriptions."""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MaterialDefinition:
    """Material definition for rendering."""
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


def create_default_materials() -> Dict[str, MaterialDefinition]:
    """Create default material definitions for land cover classes."""
    from ..assets.texture import DEFAULT_MATERIALS
    
    materials = {}
    for mat in DEFAULT_MATERIALS:
        name = mat["name"].lower().replace(" / ", "_").replace(" ", "_")
        materials[name] = MaterialDefinition(
            type="diffuse",
            properties={
                "reflectance": {
                    "type": "rgb",
                    "value": [c/255.0 for c in mat["color_8bit"]]
                },
                "roughness": mat["roughness"]
            }
        )
    
    return materials