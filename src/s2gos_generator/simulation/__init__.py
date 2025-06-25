"""Core radiative transfer simulation components - sensors, atmosphere, illumination."""

from .config import SimulationConfig, AtmosphereConfig
from .sensors import Sensor, PerspectiveSensor, DistantSensor
from .illumination import IlluminationConfig, create_solar_illumination, create_constant_illumination

# Conditional import for atmosphere (requires eradiate)
try:
    from .atmosphere import create_atmosphere
    _ATMOSPHERE_AVAILABLE = True
except ImportError:
    _ATMOSPHERE_AVAILABLE = False

__all__ = [
    "SimulationConfig", "AtmosphereConfig", 
    "Sensor", "PerspectiveSensor", "DistantSensor",
    "IlluminationConfig", "create_solar_illumination", "create_constant_illumination"
]

if _ATMOSPHERE_AVAILABLE:
    __all__.append("create_atmosphere")

def __getattr__(name):
    if name == "create_atmosphere":
        if not _ATMOSPHERE_AVAILABLE:
            raise ImportError("create_atmosphere requires eradiate dependencies")
        from .atmosphere import create_atmosphere
        return create_atmosphere
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")