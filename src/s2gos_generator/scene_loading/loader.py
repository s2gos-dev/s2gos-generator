"""Scene loader for eradiate simulations."""

from pathlib import Path
from typing import Dict, List, Union
import numpy as np
import xarray as xr
from PIL import Image

# Only import eradiate components when needed for loading
try:
    import eradiate
    from eradiate.experiments import AtmosphereExperiment
    from eradiate.units import unit_registry as ureg
    from eradiate.kernel import TypeIdLookupStrategy, UpdateParameter
    from eradiate.scenes.spectra import InterpolatedSpectrum
    import mitsuba as mi
    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False

from .config import SceneConfig
from ..materials import Material
from ..simulation.sensors import Sensor, PerspectiveSensor
from ..simulation.atmosphere import create_atmosphere


class SceneLoader:
    """Load scenes from YAML for eradiate simulations."""
    
    def __init__(self, data_dir: Path):
        if not ERADIATE_AVAILABLE:
            raise ImportError("Eradiate not available. Install eradiate to use SceneLoader.")
        
        self.data_dir = Path(data_dir)
        self.spectra_dir = self.data_dir / "spectra"
    
    def load_eradiate_experiment(self, yaml_path: Path) -> AtmosphereExperiment:
        """Load scene as eradiate experiment."""
        config = SceneConfig.from_yaml(yaml_path)
        
        # Load spectral data and create eradiate components
        spectra = self._load_spectra(config.materials)
        kdict, kpmap = self._create_eradiate_materials(config, spectra)
        
        # Create eradiate experiment
        return AtmosphereExperiment(
            atmosphere=create_atmosphere(**config.atmosphere),
            illumination=self._create_illumination(config.illumination),
            measures=self._create_measures(config.sensors),
            surface=None,  # Added via kdict
            kdict=kdict,
            kpmap=kpmap
        )
    
    def _load_spectra(self, materials: Dict[str, Material]) -> Dict[str, InterpolatedSpectrum]:
        """Load spectral data efficiently."""
        spectra = {}
        
        for mat_id, material in materials.items():
            for param, filename in material.spectra.items():
                spectrum_key = f"{mat_id}_{param}"
                file_path = self.spectra_dir / filename
                
                if file_path.exists():
                    try:
                        ds = xr.load_dataset(file_path)
                        if param in ds.data_vars:
                            spectra[spectrum_key] = InterpolatedSpectrum.from_dataarray(
                                dataarray=ds[param]
                            )
                        else:
                            print(f"Warning: {param} not found in {filename}")
                    except Exception as e:
                        print(f"Warning: Could not load {filename}: {e}")
        
        return spectra
    
    def _create_eradiate_materials(self, config: SceneConfig, spectra: Dict[str, InterpolatedSpectrum]):
        """Create eradiate materials and parameter mapping."""
        kdict = {}
        kpmap = {}
        
        # Create dispatch functions for spectra
        dispatch = {}
        for spec_id, spectrum in spectra.items():
            dispatch[spec_id] = lambda ctx, sid=spec_id: spectra[sid].eval(ctx.si)
        
        # Create materials
        for mat_id, material in config.materials.items():
            # Basic material definition
            if material.type == "bilambertian":
                kdict[mat_id] = {"type": "bilambertian"}
                self._add_spectral_params(kpmap, mat_id, ["reflectance", "transmittance"], dispatch)
                
            elif material.type == "rpv":
                kdict[mat_id] = {"type": "rpv", "rho_c": 0.5}
                param_map = {"rho_0": "rho_0.value", "k": "k.value", "Theta": "g.value", "rho_c": "rho_c.value"}
                self._add_spectral_params(kpmap, mat_id, param_map, dispatch)
                
            elif material.type == "diffuse":
                kdict[mat_id] = {"type": "diffuse"}
                self._add_spectral_params(kpmap, mat_id, ["reflectance"], dispatch)
                
            elif material.type == "ocean_legacy":
                kdict[mat_id] = {"type": "ocean_legacy", **material.params}
                self._add_ocean_params(kpmap, mat_id, material.params)
        
        # Create surface
        self._add_surface(kdict, config)
        
        return kdict, kpmap
    
    def _add_spectral_params(self, kpmap: dict, mat_id: str, params: Union[List[str], Dict[str, str]], dispatch: dict):
        """Add spectral parameter mappings."""
        param_items = params.items() if isinstance(params, dict) else [(p, f"{p}.value") for p in params]
        
        for param, param_path in param_items:
            spec_id = f"{mat_id}_{param}"
            if spec_id in dispatch:
                kpmap[f"{mat_id}.{param_path}"] = UpdateParameter(
                    dispatch[spec_id],
                    lookup_strategy=TypeIdLookupStrategy(
                        node_type=mi.BSDF, node_id=mat_id, parameter_relpath=param_path
                    ),
                    flags=UpdateParameter.Flags.SPECTRAL
                )
    
    def _add_ocean_params(self, kpmap: dict, mat_id: str, params: dict):
        """Add ocean model parameters."""
        ocean_params = {
            "wavelength": lambda ctx: ctx.si.w.m_as("nm"),
            "chlorinity": lambda _: params.get("chlorinity", 0.0),
            "pigmentation": lambda _: params.get("pigmentation", 5.0),
            "wind_speed": lambda _: params.get("wind_speed", 2.0),
            "wind_direction": lambda _: params.get("wind_direction", 90.0)
        }
        
        for param, func in ocean_params.items():
            kpmap[f"{mat_id}.{param}"] = UpdateParameter(
                func,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF, node_id=mat_id, parameter_relpath=param
                ),
                flags=UpdateParameter.Flags.SPECTRAL if param == "wavelength" else 0
            )
    
    def _add_surface(self, kdict: dict, config: SceneConfig):
        """Add surface geometry to kernel dict."""
        surface = config.surface
        texture_path = Path(surface["selection_texture"])
        
        if texture_path.exists():
            # Load texture
            texture_data = np.atleast_3d(np.array(Image.open(texture_path)))
            
            # Create material order matching landcover IDs
            material_order = [
                surface["materials"][lc] 
                for lc in sorted(config.landcover_ids.keys(), key=config.landcover_ids.get)
            ]
            
            # Surface material selector
            kdict["terrain_material"] = {
                "type": "selectbsdf", "id": "terrain_material",
                "indices": {
                    "type": "bitmap", "raw": True, "filter_type": "nearest",
                    "wrap_mode": "clamp", "data": texture_data
                },
                **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id} 
                   for i, mat_id in enumerate(material_order)}
            }
            
            # Terrain mesh
            kdict["terrain"] = {
                "type": "ply", "filename": surface["mesh"],
                "bsdf": {"type": "ref", "id": "terrain_material"}, "id": "terrain"
            }
    
    def _create_illumination(self, config: dict) -> dict:
        """Create eradiate illumination."""
        return {
            "type": config.get("type", "directional"),
            "zenith": config.get("zenith", 30.0) * ureg.deg,
            "azimuth": config.get("azimuth", 180.0) * ureg.deg,
            "irradiance": config.get("irradiance", {
                "type": "solar_irradiance", "dataset": "thuillier_2003"
            })
        }
    
    def _create_measures(self, sensors: List[Sensor]) -> List[dict]:
        """Create eradiate measures from sensors."""
        measures = []
        for sensor in sensors:
            measure = sensor.to_dict()
            # Convert wavelengths to eradiate format
            if "srf" in measure and "wavelengths" in measure["srf"]:
                measure["srf"]["wavelengths"] = [w * ureg.nm for w in measure["srf"]["wavelengths"]]
            measures.append(measure)
        return measures or [PerspectiveSensor().to_dict()]