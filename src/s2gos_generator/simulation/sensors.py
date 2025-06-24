"""Sensor definitions for radiative transfer simulations."""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from abc import ABC, abstractmethod


@dataclass
class Sensor(ABC):
    """Base sensor configuration."""
    type: str
    id: str = "sensor"
    wavelengths: List[float] = field(default_factory=lambda: [440, 550, 660])
    spp: int = 128
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "id": self.id,
            "srf": {"type": "delta", "wavelengths": self.wavelengths},
            "spp": self.spp,
            **self._specific_params()
        }
    
    @abstractmethod
    def _specific_params(self) -> Dict[str, Any]:
        """Sensor-specific parameters."""
        pass


@dataclass
class PerspectiveSensor(Sensor):
    """Perspective camera sensor for scene inspection."""
    type: str = "perspective"
    origin: List[float] = field(default_factory=lambda: [0, 0, 36612.90342332])
    target: List[float] = field(default_factory=lambda: [0, 0, 0])
    up: List[float] = field(default_factory=lambda: [0, 1, 0])
    fov: float = 70.0
    resolution: List[int] = field(default_factory=lambda: [1024, 768])
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "target": self.target, 
            "up": self.up,
            "fov": self.fov,
            "film_resolution": self.resolution
        }


@dataclass  
class DistantSensor(Sensor):
    """Distant sensor for top-of-atmosphere measurements."""
    type: str = "mpdistant"
    angles: List[float] = field(default_factory=lambda: [0.0, 0.0])  # zenith, azimuth
    target_size: float = 50.0e3  # 50km square
    resolution: List[int] = field(default_factory=lambda: [256, 256])
    
    def _specific_params(self) -> Dict[str, Any]:
        return {
            "construct": "from_angles",
            "angles": self.angles,
            "target": {
                "type": "rectangle",
                "xmin": -self.target_size, "xmax": self.target_size,
                "ymin": -self.target_size, "ymax": self.target_size
            },
            "film_resolution": self.resolution
        }