"""Scene configuration format."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from datetime import datetime

from ..materials import Material, MaterialRegistry
from ..simulation.sensors import Sensor, PerspectiveSensor


@dataclass
class SceneMetadata:
    """Scene generation metadata."""
    center_lat: float
    center_lon: float
    aoi_size_km: float
    resolution_m: float
    generation_date: Optional[str] = None
    
    # Generation parameters (for reproducibility)
    dem_index_path: Optional[str] = None
    landcover_index_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "aoi_size_km": self.aoi_size_km,
            "resolution_m": self.resolution_m
        }
        
        if self.generation_date:
            result["generation_date"] = self.generation_date
            
        if self.dem_index_path or self.landcover_index_path:
            result["generation"] = {}
            if self.dem_index_path:
                result["generation"]["dem_index_path"] = self.dem_index_path
            if self.landcover_index_path:
                result["generation"]["landcover_index_path"] = self.landcover_index_path
            
        return result


@dataclass
class SceneConfig:
    """Complete scene configuration in dreams-scenes compatible format."""
    
    # Scene identification and metadata
    name: str
    metadata: SceneMetadata
    
    # Core definitions (matches Beijing.yml structure)
    landcover_ids: Dict[str, int]
    materials: Dict[str, Material]
    target: Dict[str, Any]  # surface definition
    
    # Simulation parameters (optional)
    atmosphere: Optional[Dict[str, Any]] = None
    illumination: Optional[Dict[str, Any]] = None
    sensors: List[Sensor] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            # Scene metadata at top level
            "scene": {
                "name": self.name,
                **self.metadata.to_dict()
            },
            
            # Core definitions (dreams-scenes structure)
            "landcover_ids": self.landcover_ids,
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
            "target": self.target
        }
        
        # Add simulation parameters if present
        if self.atmosphere:
            result["atmosphere"] = self.atmosphere
        if self.illumination:
            result["illumination"] = self.illumination
        if self.sensors:
            result["measures"] = [s.to_dict() for s in self.sensors]
            
        return result
    
    def save_yaml(self, path: Path, include_comments: bool = True):
        """Save to YAML with optional comments."""
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if include_comments:
                f.write("# S2GOS Generated Scene Configuration\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
            
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, width=120)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'SceneConfig':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract scene metadata
        scene_info = data.get("scene", {})
        metadata = SceneMetadata(
            center_lat=scene_info["center_lat"],
            center_lon=scene_info["center_lon"],
            aoi_size_km=scene_info["aoi_size_km"],
            resolution_m=scene_info["resolution_m"],
            generation_date=scene_info.get("generation_date"),
            dem_index_path=scene_info.get("generation", {}).get("dem_index_path"),
            landcover_index_path=scene_info.get("generation", {}).get("landcover_index_path")
        )
        
        # Parse materials
        materials = {}
        for mat_id, mat_data in data["materials"].items():
            mat_type = mat_data.pop("type")
            spectra = {}
            params = {}
            
            for key, value in mat_data.items():
                if isinstance(value, dict) and "path" in value:
                    filename = Path(value["path"]).name
                    spectra[key] = filename
                else:
                    params[key] = value
            
            materials[mat_id] = Material(mat_type, spectra, params)
        
        # Parse sensors
        sensors = []
        for sensor_data in data.get("measures", []):
            if sensor_data["type"] == "perspective":
                sensors.append(PerspectiveSensor(
                    id=sensor_data.get("id", "perspective"),
                    origin=sensor_data.get("origin", [0, 0, 36612.90342332]),
                    target=sensor_data.get("target", [0, 0, 0]),
                    fov=sensor_data.get("fov", 70),
                    resolution=sensor_data.get("film_resolution", [1024, 768]),
                    spp=sensor_data.get("spp", 128)
                ))
        
        return cls(
            name=scene_info["name"],
            metadata=metadata,
            landcover_ids=data["landcover_ids"],
            materials=materials,
            target=data["target"],
            atmosphere=data.get("atmosphere"),
            illumination=data.get("illumination"),
            sensors=sensors
        )


def create_s2gos_scene(
    scene_name: str, mesh_path: str, texture_path: str,
    center_lat: float, center_lon: float, aoi_size_km: float, resolution_m: float = 30.0,
    sensor_height: float = 36612.90342332, **kwargs
) -> SceneConfig:
    """Create standard S2GOS scene configuration."""
    
    # Scene metadata
    metadata = SceneMetadata(
        center_lat=center_lat,
        center_lon=center_lon,
        aoi_size_km=aoi_size_km,
        resolution_m=resolution_m,
        generation_date=datetime.now().isoformat(),
        dem_index_path=kwargs.get("dem_index_path"),
        landcover_index_path=kwargs.get("landcover_index_path")
    )
    
    # Standard S2GOS landcover mapping
    landcover_ids = {
        "tree_cover": 0, "shrubland": 1, "grassland": 2, "cropland": 3,
        "built_up": 4, "bare_sparse_vegetation": 5, "snow_and_ice": 6,
        "permanent_water_bodies": 7, "herbaceous_wetland": 8, 
        "mangroves": 9, "moss_and_lichen": 10
    }
    
    # Material assignment to landcover
    material_mapping = {
        "tree_cover": "treecover", "shrubland": "shrubland", 
        "grassland": "grassland", "cropland": "cropland",
        "built_up": "concrete", "bare_sparse_vegetation": "baresoil",
        "snow_and_ice": "snow", "permanent_water_bodies": "water",
        "herbaceous_wetland": "wetland", "mangroves": "mangroves",
        "moss_and_lichen": "moss"
    }
    
    # Target surface definition
    target = {
        "mesh": mesh_path,
        "selection_texture": texture_path, 
        "materials": material_mapping
    }
    
    # Default simulation parameters
    atmosphere = kwargs.get("atmosphere", {
        "boa": 0.0, "toa": 40000.0, "aerosol_ot": 0.1, 
        "aerosol_scale": 1000.0, "aerosol_ds": "sixsv-continental"
    })
    
    illumination = kwargs.get("illumination", {
        "type": "directional", "zenith": 30.0, "azimuth": 180.0,
        "irradiance": {"type": "solar_irradiance", "dataset": "thuillier_2003"}
    })
    
    # Default perspective sensor
    sensors = [PerspectiveSensor(
        id="perspective_view",
        origin=[0, 0, sensor_height],
        spp=kwargs.get("spp", 512)
    )]
    
    return SceneConfig(
        name=scene_name,
        metadata=metadata,
        landcover_ids=landcover_ids,
        materials=MaterialRegistry.create_s2gos_materials(),
        target=target,
        atmosphere=atmosphere,
        illumination=illumination,
        sensors=sensors
    )