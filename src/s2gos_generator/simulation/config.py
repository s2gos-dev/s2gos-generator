"""Rendering configuration for radiative transfer simulations."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from datetime import datetime

from .sensors import Sensor, PerspectiveSensor, DistantSensor
from .illumination import IlluminationConfig


@dataclass 
class SimulationConfig:
    """Rendering configuration for radiative transfer simulations.
    
    Contains only rendering-specific parameters: sensors and illumination.
    Atmosphere is now part of SceneConfig since it's tied to geographic location.
    Also known as RenderConfig for backwards compatibility.
    
    Attributes:
        illumination: Light source configuration  
        sensors: List of sensor/camera configurations
        name: Optional identifier for this render configuration
    """
    
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig) 
    sensors: List[Sensor] = field(default_factory=list)
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "illumination": self.illumination.to_dict(),
            "measures": [s.to_dict() for s in self.sensors]
        }
        
        if self.name:
            result["name"] = self.name
            
        return result
    
    def save_yaml(self, path: Path, include_comments: bool = True):
        """Save render configuration to YAML file.
        
        Args:
            path: Output file path
            include_comments: Whether to include header comments
        """
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if include_comments:
                f.write("# S2GOS Render Configuration (Sensors + Illumination)\n")
                f.write("# Atmosphere is part of scene configuration\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
            
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, width=120)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'SimulationConfig':
        """Load render configuration from YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            SimulationConfig instance loaded from file
        """
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse illumination
        illum_data = data.get("illumination", {})
        illumination = IlluminationConfig(
            zenith=illum_data.get("zenith", 30.0),
            azimuth=illum_data.get("azimuth", 180.0),
            irradiance_type=illum_data.get("irradiance", {}).get("type", "solar_irradiance"),
            irradiance_dataset=illum_data.get("irradiance", {}).get("dataset", "thuillier_2003")
        )
        
        # Parse sensors
        sensors = []
        for sensor_data in data.get("measures", []):
            if sensor_data["type"] == "perspective":
                sensors.append(PerspectiveSensor(
                    id=sensor_data.get("id", "perspective"),
                    origin=sensor_data.get("origin", [0, 0, 98612.90342332]),
                    target=sensor_data.get("target", [0, 0, 0]),
                    fov=sensor_data.get("fov", 70),
                    resolution=sensor_data.get("film_resolution", [1024, 768]),
                    spp=sensor_data.get("spp", 128)
                ))
            elif sensor_data["type"] == "mpdistant":
                sensors.append(DistantSensor(
                    id=sensor_data.get("id", "distant"),
                    angles=sensor_data.get("angles", [0.0, 0.0]),
                    target_size=sensor_data.get("target", {}).get("xmax", 50.0e3),
                    resolution=sensor_data.get("film_resolution", [256, 256]),
                    spp=sensor_data.get("spp", 128)
                ))
        
        return cls(
            name=data.get("name"),
            illumination=illumination,
            sensors=sensors
        )


# Alias for backwards compatibility and clearer naming
RenderConfig = SimulationConfig


def create_default_render_config(
    name: str = "default",
    sensor_height: float = 98612.90342332,
    sensor_fov: float = 70.0,
    sensor_resolution: List[int] = None,
    spp: int = 32,
    zenith: float = 30.0,
    azimuth: float = 180.0,
    **kwargs
) -> RenderConfig:
    """Create a standard render configuration with sensible defaults.
    
    Args:
        name: Configuration name
        sensor_height: Camera height above ground in meters
        sensor_fov: Field of view in degrees
        sensor_resolution: [width, height] resolution
        spp: Samples per pixel
        zenith: Sun zenith angle in degrees
        azimuth: Sun azimuth angle in degrees
        **kwargs: Additional illumination parameters
        
    Returns:
        Render configuration with sensors and illumination only
    """
    if sensor_resolution is None:
        sensor_resolution = [1024, 768]
    
    illumination = IlluminationConfig(
        zenith=zenith,
        azimuth=azimuth,
        irradiance_type="solar_irradiance",
        irradiance_dataset="thuillier_2003"
    )
    
    sensors = [PerspectiveSensor(
        id="perspective_view",
        origin=[0, 0, sensor_height],
        target=[0, 0, 0],
        fov=sensor_fov,
        resolution=sensor_resolution,
        spp=spp
    )]
    
    return RenderConfig(
        name=name,
        illumination=illumination,
        sensors=sensors
    )
