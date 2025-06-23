"""Scene description format for S2GOS scenes."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import json


@dataclass
class MaterialDefinition:
    """Material definition for rendering."""
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGeometry:
    """Scene geometry configuration."""
    mesh_path: Path
    texture_path: Optional[Path] = None
    materials: Dict[str, str] = field(default_factory=dict)


@dataclass
class SceneDescription:
    """Complete scene description."""
    name: str
    location: Dict[str, float]
    extent_km: float
    resolution_m: float
    
    materials: Dict[str, MaterialDefinition] = field(default_factory=dict)
    geometry: Optional[SceneGeometry] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_config(cls, config) -> "SceneDescription":
        """Create scene description from pipeline config."""
        return cls(
            name=config.scene_name,
            location={
                "lat": config.center_lat,
                "lon": config.center_lon
            },
            extent_km=config.aoi_size_km,
            resolution_m=config.target_resolution_m,
            metadata={
                "dem_index_path": str(config.dem_index_path),
                "landcover_index_path": str(config.landcover_index_path),
                "output_dir": str(config.output_dir)
            }
        )
    
    def add_geometry(self, mesh_path: Path, texture_path: Optional[Path] = None, 
                    materials: Optional[Dict[str, str]] = None):
        """Add geometry configuration to scene."""
        self.geometry = SceneGeometry(
            mesh_path=mesh_path,
            texture_path=texture_path,
            materials=materials or {}
        )
    
    def add_material(self, name: str, material_type: str, **properties):
        """Add a material definition."""
        self.materials[name] = MaterialDefinition(
            type=material_type,
            properties=properties
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "name": self.name,
            "location": self.location,
            "extent_km": self.extent_km,
            "resolution_m": self.resolution_m,
            "metadata": self.metadata
        }
        
        if self.materials:
            result["materials"] = {
                name: {
                    "type": mat.type,
                    **mat.properties
                }
                for name, mat in self.materials.items()
            }
        
        if self.geometry:
            result["geometry"] = {
                "mesh": str(self.geometry.mesh_path),
                "texture": str(self.geometry.texture_path) if self.geometry.texture_path else None,
                "materials": self.geometry.materials
            }
        
        return result
    
    def save_yaml(self, output_path: Path):
        """Save scene description as YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def save_json(self, output_path: Path):
        """Save scene description as JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_yaml(cls, file_path: Path) -> "SceneDescription":
        """Load scene description from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def load_json(cls, file_path: Path) -> "SceneDescription":
        """Load scene description from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneDescription":
        """Create scene description from dictionary."""
        scene = cls(
            name=data["name"],
            location=data["location"],
            extent_km=data["extent_km"],
            resolution_m=data["resolution_m"],
            metadata=data.get("metadata", {})
        )
        
        if "materials" in data:
            for name, mat_data in data["materials"].items():
                mat_type = mat_data.pop("type")
                scene.add_material(name, mat_type, **mat_data)
        
        if "geometry" in data:
            geom_data = data["geometry"]
            scene.add_geometry(
                mesh_path=Path(geom_data["mesh"]),
                texture_path=Path(geom_data["texture"]) if geom_data.get("texture") else None,
                materials=geom_data.get("materials", {})
            )
        
        return scene


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


def create_scene_from_test_output(output_dir: Path, scene_name: str) -> SceneDescription:
    """Create scene description matching the test_output7 example format."""
    # Read the existing YAML to understand the structure
    scene_yml_path = output_dir / scene_name / f"{scene_name}_scene.yml"
    
    if not scene_yml_path.exists():
        raise FileNotFoundError(f"Scene YAML not found at {scene_yml_path}")
    
    with open(scene_yml_path, 'r') as f:
        existing_data = yaml.safe_load(f)
    
    # Create scene with the same structure as test_output7
    scene = SceneDescription(
        name=existing_data["name"],
        location=existing_data["location"],
        extent_km=existing_data["extent_km"],
        resolution_m=existing_data["resolution_m"],
        metadata=existing_data.get("metadata", {})
    )
    
    # Add materials from existing data
    if "materials" in existing_data:
        for name, mat_data in existing_data["materials"].items():
            scene.materials[name] = MaterialDefinition(
                type=mat_data["type"],
                properties={k: v for k, v in mat_data.items() if k != "type"}
            )
    
    # Add geometry from existing data
    if "geometry" in existing_data:
        geom_data = existing_data["geometry"]
        scene.geometry = SceneGeometry(
            mesh_path=Path(geom_data["mesh"]),
            texture_path=Path(geom_data["texture"]) if geom_data.get("texture") else None,
            materials=geom_data.get("materials", {})
        )
    
    return scene