"""Simulation configuration and material registry."""

from pathlib import Path
from typing import Dict, Any, List
import yaml
from dataclasses import dataclass, field

from .sensors import Sensor, PerspectiveSensor


@dataclass
class Material:
    """Simplified material definition."""
    type: str
    spectra: Dict[str, str] = field(default_factory=dict)  # param -> filename
    params: Dict[str, Any] = field(default_factory=dict)   # static parameters
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        # Add spectral file references
        for param, filename in self.spectra.items():
            result[param] = {"path": f"spectra/{filename}", "variable": param}
        # Add static parameters
        result.update(self.params)
        return result


class MaterialRegistry:
    """Registry of common S2GOS materials with automatic BRDF parameter mapping."""
    
    @staticmethod
    def create_s2gos_materials() -> Dict[str, Material]:
        """Create standard S2GOS material set."""
        return {
            # Vegetation materials - bilambertian with reflectance + transmittance
            "treecover": Material("bilambertian", {
                "reflectance": "spectrum_treecover_foliage_high.nc",
                "transmittance": "spectrum_treecover_foliage_high.nc"
            }),
            "shrubland": Material("bilambertian", {
                "reflectance": "spectrum_shrubland_foliage.nc", 
                "transmittance": "spectrum_shrubland_foliage.nc"
            }),
            "cropland": Material("bilambertian", {
                "reflectance": "spectrum_cropland_foliage_uniform.nc",
                "transmittance": "spectrum_cropland_foliage_uniform.nc"
            }),
            "mangroves": Material("bilambertian", {
                "reflectance": "spectrum_treecover_foliage_high.nc",
                "transmittance": "spectrum_treecover_foliage_high.nc"
            }),
            
            # Surface materials - RPV for rough surfaces
            "grassland": Material("rpv", {
                "rho_0": "spectrum_grassland_rpv_high.nc",
                "k": "spectrum_grassland_rpv_high.nc", 
                "Theta": "spectrum_grassland_rpv_high.nc",
                "rho_c": "spectrum_grassland_rpv_high.nc"
            }),
            "baresoil": Material("rpv", {
                "rho_0": "spectrum_baresoil_low.nc",
                "k": "spectrum_baresoil_low.nc",
                "Theta": "spectrum_baresoil_low.nc", 
                "rho_c": "spectrum_baresoil_low.nc"
            }),
            "wetland": Material("rpv", {  # Use grassland spectra
                "rho_0": "spectrum_grassland_rpv_high.nc",
                "k": "spectrum_grassland_rpv_high.nc",
                "Theta": "spectrum_grassland_rpv_high.nc",
                "rho_c": "spectrum_grassland_rpv_high.nc"
            }),
            
            # Simple materials - diffuse with reflectance only
            "concrete": Material("diffuse", {"reflectance": "spectrum_concrete.nc"}),
            "snow": Material("diffuse", {"reflectance": "spectrum_snow.nc"}),
            "moss": Material("diffuse", {"reflectance": "spectrum_moss.nc"}),
            
            # Water - ocean model with runtime parameters
            "water": Material("ocean_legacy", params={
                "wavelength": 550.0, "chlorinity": 0.0, "pigmentation": 5.0,
                "wind_speed": 2.0, "wind_direction": 90.0
            })
        }


@dataclass
class SimulationConfig:
    """Complete simulation configuration for radiative transfer."""
    
    # Essential components
    landcover_ids: Dict[str, int]
    materials: Dict[str, Material] 
    surface: Dict[str, Any]  # mesh, selection_texture, material_mapping
    
    # Simulation settings
    atmosphere: Dict[str, Any] = field(default_factory=lambda: {
        "boa": 0.0, "toa": 40000.0, "aerosol_ot": 0.1, 
        "aerosol_scale": 1000.0, "aerosol_ds": "sixsv-continental"
    })
    illumination: Dict[str, Any] = field(default_factory=lambda: {
        "type": "directional", "zenith": 30.0, "azimuth": 180.0,
        "irradiance": {"type": "solar_irradiance", "dataset": "thuillier_2003"}
    })
    sensors: List[Sensor] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "landcover_ids": self.landcover_ids,
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
            "target": self.surface,
            "atmosphere": self.atmosphere,
            "illumination": self.illumination,
            "measures": [s.to_dict() for s in self.sensors]
        }
    
    def save_yaml(self, path: Path):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'SimulationConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse materials 
        materials = {}
        for mat_id, mat_data in data["materials"].items():
            mat_type = mat_data.pop("type")
            spectra = {}
            params = {}
            
            for key, value in mat_data.items():
                if isinstance(value, dict) and "path" in value:
                    # Extract filename from path
                    filename = Path(value["path"]).name
                    spectra[key] = filename
                else:
                    params[key] = value
            
            materials[mat_id] = Material(mat_type, spectra, params)
        
        # Parse sensors - simplified for now
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
            landcover_ids=data["landcover_ids"],
            materials=materials,
            surface=data["target"],
            atmosphere=data.get("atmosphere", {}),
            illumination=data.get("illumination", {}),
            sensors=sensors
        )


def create_s2gos_scene(
    scene_name: str, mesh_path: str, texture_path: str,
    sensor_height: float = 36612.90342332, **kwargs
) -> SimulationConfig:
    """Create standard S2GOS simulation configuration."""
    
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
    
    # Surface definition
    surface = {
        "mesh": mesh_path,
        "selection_texture": texture_path, 
        "materials": material_mapping
    }
    
    # Default perspective sensor
    sensors = [PerspectiveSensor(
        id="perspective_view",
        origin=[0, 0, sensor_height],
        spp=kwargs.get("spp", 512)
    )]
    
    return SimulationConfig(
        landcover_ids=landcover_ids,
        materials=MaterialRegistry.create_s2gos_materials(),
        surface=surface,
        sensors=sensors,
        **{k: v for k, v in kwargs.items() if k in ['atmosphere', 'illumination']}
    )