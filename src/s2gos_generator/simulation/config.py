"""Basic simulation configuration for atmosphere, sensors, and illumination only."""

from dataclasses import dataclass, field
from typing import Dict, Any, List

from .sensors import Sensor
from .illumination import IlluminationConfig


@dataclass
class AtmosphereConfig:
    """Configuration for atmospheric properties."""
    boa: float = 0.0  # Bottom of atmosphere altitude [m]
    toa: float = 40000.0  # Top of atmosphere altitude [m]  
    aerosol_ot: float = 0.1  # Aerosol optical thickness
    aerosol_scale: float = 1000.0  # Aerosol scale height [m]
    aerosol_ds: str = "sixsv-continental"  # Aerosol dataset
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "boa": self.boa,
            "toa": self.toa,
            "aerosol_ot": self.aerosol_ot,
            "aerosol_scale": self.aerosol_scale,
            "aerosol_ds": self.aerosol_ds
        }


@dataclass 
class SimulationConfig:
    """Core simulation configuration - atmosphere, illumination, and sensors only."""
    
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    illumination: IlluminationConfig = field(default_factory=IlluminationConfig) 
    sensors: List[Sensor] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "atmosphere": self.atmosphere.to_dict(),
            "illumination": self.illumination.to_dict(),
            "measures": [s.to_dict() for s in self.sensors]
        }