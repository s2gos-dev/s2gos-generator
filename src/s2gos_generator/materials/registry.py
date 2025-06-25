"""Material registry and S2GOS material catalog."""

from typing import Dict
from .definitions import Material, MaterialDefinition


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


def create_s2gos_materials() -> Dict[str, Material]:
    """Create standard S2GOS material set - convenience function."""
    return MaterialRegistry.create_s2gos_materials()


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