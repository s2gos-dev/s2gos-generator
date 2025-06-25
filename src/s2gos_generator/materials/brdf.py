"""BRDF-specific utilities and models."""

from typing import Dict, Any


class BRDFModels:
    """Common BRDF model configurations."""
    
    @staticmethod
    def bilambertian(reflectance_spectrum: str, transmittance_spectrum: str) -> Dict[str, Any]:
        """Create bilambertian material configuration."""
        return {
            "type": "bilambertian",
            "reflectance": {"path": f"spectra/{reflectance_spectrum}", "variable": "reflectance"},
            "transmittance": {"path": f"spectra/{transmittance_spectrum}", "variable": "transmittance"}
        }
    
    @staticmethod
    def rpv(spectrum_file: str) -> Dict[str, Any]:
        """Create RPV material configuration."""
        return {
            "type": "rpv",
            "rho_0": {"path": f"spectra/{spectrum_file}", "variable": "rho_0"},
            "k": {"path": f"spectra/{spectrum_file}", "variable": "k"},
            "Theta": {"path": f"spectra/{spectrum_file}", "variable": "Theta"},
            "rho_c": {"path": f"spectra/{spectrum_file}", "variable": "rho_c"}
        }
    
    @staticmethod
    def diffuse(reflectance_spectrum: str) -> Dict[str, Any]:
        """Create diffuse material configuration."""
        return {
            "type": "diffuse",
            "reflectance": {"path": f"spectra/{reflectance_spectrum}", "variable": "reflectance"}
        }
    
    @staticmethod
    def ocean_legacy(wavelength: float = 550.0, chlorinity: float = 0.0, 
                    pigmentation: float = 5.0, wind_speed: float = 2.0, 
                    wind_direction: float = 90.0) -> Dict[str, Any]:
        """Create ocean legacy material configuration."""
        return {
            "type": "ocean_legacy",
            "wavelength": wavelength,
            "chlorinity": chlorinity,
            "pigmentation": pigmentation,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction
        }