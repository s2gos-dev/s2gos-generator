"""Illumination models and configurations."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class IlluminationConfig:
    """Configuration for illumination in radiative transfer simulations."""
    type: str = "directional"
    zenith: float = 30.0  # degrees
    azimuth: float = 180.0  # degrees
    irradiance_type: str = "solar_irradiance"
    irradiance_dataset: str = "thuillier_2003"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "zenith": self.zenith,
            "azimuth": self.azimuth,
            "irradiance": {
                "type": self.irradiance_type,
                "dataset": self.irradiance_dataset
            }
        }


def create_solar_illumination(
    zenith: float = 30.0,
    azimuth: float = 180.0,
    dataset: str = "thuillier_2003"
) -> IlluminationConfig:
    """Create solar illumination configuration."""
    return IlluminationConfig(
        type="directional",
        zenith=zenith,
        azimuth=azimuth,
        irradiance_type="solar_irradiance",
        irradiance_dataset=dataset
    )


def create_constant_illumination(
    zenith: float = 30.0,
    azimuth: float = 180.0,
    radiance: float = 1.0
) -> IlluminationConfig:
    """Create constant illumination configuration."""
    config = IlluminationConfig(
        type="directional",
        zenith=zenith,
        azimuth=azimuth
    )
    config.irradiance_type = "constant"
    config.irradiance_dataset = str(radiance)
    return config