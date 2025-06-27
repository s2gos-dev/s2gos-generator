"""Core radiative transfer simulation components - sensors and illumination only."""

from .config import SimulationConfig
from .sensors import Sensor, PerspectiveSensor, DistantSensor
from .illumination import IlluminationConfig, create_solar_illumination, create_constant_illumination

__all__ = [
    "SimulationConfig", 
    "Sensor", "PerspectiveSensor", "DistantSensor",
    "IlluminationConfig", "create_solar_illumination", "create_constant_illumination"
]