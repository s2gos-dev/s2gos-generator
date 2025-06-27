"""Material registry and S2GOS material catalog."""

from typing import Dict
from .definitions import Material


class MaterialRegistry:
    """Registry of common S2GOS materials."""
    
    @staticmethod
    def create_s2gos_materials() -> Dict[str, Material]:
        """Create standard S2GOS material set.
        
        Returns:
            Dictionary mapping material names to material instances
        """
        
        def spectral_param(filename: str, variable: str) -> Dict[str, str]:
            """Create spectral parameter specification.
            
            Args:
                filename: Name of spectral data file
                variable: Variable name within the file
                
            Returns:
                Dictionary with path and variable keys
            """
            return {"path": filename, "variable": variable}
        
        return {
            "treecover": Material.from_dict({
                "type": "bilambertian",
                "reflectance": spectral_param("spectrum_treecover_foliage_high.nc", "reflectance"),
                "transmittance": spectral_param("spectrum_treecover_foliage_high.nc", "transmittance")
            }, id="treecover"),
            
            "shrubland": Material.from_dict({
                "type": "bilambertian",
                "reflectance": spectral_param("spectrum_shrubland_foliage.nc", "reflectance"),
                "transmittance": spectral_param("spectrum_shrubland_foliage.nc", "transmittance")
            }, id="shrubland"),
            
            "cropland": Material.from_dict({
                "type": "bilambertian", 
                "reflectance": spectral_param("spectrum_cropland_foliage_uniform.nc", "reflectance"),
                "transmittance": spectral_param("spectrum_cropland_foliage_uniform.nc", "transmittance")
            }, id="cropland"),
            
            "mangroves": Material.from_dict({
                "type": "bilambertian",
                "reflectance": spectral_param("spectrum_treecover_foliage_high.nc", "reflectance"),
                "transmittance": spectral_param("spectrum_treecover_foliage_high.nc", "transmittance")
            }, id="mangroves"),
            
            "grassland": Material.from_dict({
                "type": "rpv",
                "rho_0": spectral_param("spectrum_grassland_rpv_high.nc", "rho_0"),
                "k": spectral_param("spectrum_grassland_rpv_high.nc", "k"),
                "Theta": spectral_param("spectrum_grassland_rpv_high.nc", "Theta"),
                "rho_c": spectral_param("spectrum_grassland_rpv_high.nc", "rho_c")
            }, id="grassland"),
            
            "baresoil": Material.from_dict({
                "type": "rpv",
                "rho_0": spectral_param("spectrum_baresoil_low.nc", "rho_0"),
                "k": spectral_param("spectrum_baresoil_low.nc", "k"), 
                "Theta": spectral_param("spectrum_baresoil_low.nc", "Theta"),
                "rho_c": spectral_param("spectrum_baresoil_low.nc", "rho_c")
            }, id="baresoil"),
            
            "wetland": Material.from_dict({
                "type": "rpv",
                "rho_0": spectral_param("spectrum_grassland_rpv_high.nc", "rho_0"),
                "k": spectral_param("spectrum_grassland_rpv_high.nc", "k"),
                "Theta": spectral_param("spectrum_grassland_rpv_high.nc", "Theta"),
                "rho_c": spectral_param("spectrum_grassland_rpv_high.nc", "rho_c")
            }, id="wetland"),
            
            "concrete": Material.from_dict({
                "type": "diffuse",
                "reflectance": spectral_param("spectrum_concrete.nc", "reflectance")
            }, id="concrete"),
            
            "snow": Material.from_dict({
                "type": "diffuse", 
                "reflectance": spectral_param("spectrum_snow.nc", "reflectance")
            }, id="snow"),
            
            "moss": Material.from_dict({
                "type": "diffuse",
                "reflectance": spectral_param("spectrum_moss.nc", "reflectance")
            }, id="moss"),
            
            "water": Material.from_dict({
                "type": "ocean_legacy",
                "chlorinity": 0.0,
                "pigmentation": 5.0,
                "wind_speed": 2.0,
                "wind_direction": 90.0
            }, id="water")
        }
    
    @staticmethod
    def create_material_kdict_kpmap(materials: Dict[str, Material], mode: str = "mono") -> tuple[Dict, Dict]:
        """Generate combined kernel dictionary and parameter map for all materials.
        
        Args:
            materials: Dictionary of material instances
            mode: Eradiate rendering mode ('mono' or 'rgb')
            
        Returns:
            Tuple of (kernel_dict, parameter_map) for use in Eradiate scenes
        """
        kdict = {}
        kpmap = {}
        
        for mat_name, material in materials.items():
            mat_kdict = material.kdict(mode)
            mat_kpmap = material.kpmap(mode)
            
            kdict.update(mat_kdict)
            kpmap.update(mat_kpmap)
        
        return kdict, kpmap


def create_s2gos_materials() -> Dict[str, Material]:
    """Create standard S2GOS material set.
    
    Returns:
        Dictionary mapping material names to material instances
    """
    return MaterialRegistry.create_s2gos_materials()