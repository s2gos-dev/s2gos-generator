"""Material definition classes for radiative transfer simulations."""

import attrs
from typing import Dict, Any, Callable, ClassVar
from pathlib import Path
import xarray as xr

from ..core.paths import open_dataset

try:
    from eradiate import KernelContext
    from eradiate.kernel import TypeIdLookupStrategy, UpdateParameter
    from eradiate.scenes.spectra import InterpolatedSpectrum
    import mitsuba as mi
    ERADIATE_AVAILABLE = True
except ImportError:
    KernelContext = None
    InterpolatedSpectrum = None
    UpdateParameter = None
    TypeIdLookupStrategy = None
    mi = None
    ERADIATE_AVAILABLE = False


def _declare_mono_scene_parameter(
    func: Callable[[KernelContext], float], node_id: str, param: str
) -> UpdateParameter:
    """Create UpdateParameter for monochromatic mode.
    
    Args:
        func: Function that computes parameter value from kernel context
        node_id: Eradiate node identifier
        param: Parameter path relative to the node
        
    Returns:
        UpdateParameter configured for spectral evaluation
    """
    return UpdateParameter(
        func,
        lookup_strategy=TypeIdLookupStrategy(
            node_type=mi.BSDF, node_id=node_id, parameter_relpath=param
        ),
        flags=UpdateParameter.Flags.SPECTRAL,
    )


def _spectral_parameter_converter(value: Any) -> Callable[[KernelContext], float]:
    """Convert spectral parameter specification to callable function.
    
    Args:
        value: Dictionary with 'path' and 'variable' keys specifying spectral data
        
    Returns:
        Function that evaluates spectral data at given wavelength
        
    Raises:
        TypeError: If value type is not supported
    """
    if isinstance(value, dict):
        file_path = value["path"]
        variable = value["variable"]
        
        if not Path(file_path).is_absolute():
            spectra_dir = Path(__file__).parent.parent / "data" / "spectra"
            full_path = spectra_dir / file_path
        else:
            full_path = Path(file_path)
            
        ds = open_dataset(full_path)
        da = ds[variable]
        s = InterpolatedSpectrum.from_dataarray(dataarray=da)

        def spectrum(ctx: KernelContext) -> float:
            return s.eval(ctx.si).m_as("dimensionless")

        return spectrum
    else:
        raise TypeError(f"conversion of {type(value).__name__} is unsupported")


@attrs.define
class Material:
    """Material base class and factory for material subtypes.
    
    Provides factory method to create material instances from dictionary
    specifications and defines the interface all materials must implement.
    """

    __SUBTYPES: ClassVar[dict] = None

    @classmethod
    def __subtypes(cls) -> dict[str, type]:
        """Get the subtype dispatch table.
        
        Returns:
            Dictionary mapping material type names to their classes
        """
        if cls.__SUBTYPES is None:
            cls.__SUBTYPES = {
                "diffuse": DiffuseMaterial,
                "bilambertian": BilambertianMaterial,
                "rpv": RPVMaterial,
                "ocean_legacy": OceanLegacyMaterial,
            }
        return cls.__SUBTYPES

    @classmethod
    def from_dict(cls, d: dict, **kwargs):
        """Create material instance from dictionary specification.
        
        Args:
            d: Dictionary with 'type' key and material parameters
            **kwargs: Additional arguments passed to material constructor
            
        Returns:
            Material instance of appropriate subtype
            
        Raises:
            ValueError: If material type is unknown
        """
        d = d.copy()
        subtype = d.pop("type")

        try:
            subtype = cls.__subtypes()[subtype]
        except KeyError as e:
            raise ValueError(f"unknown material type '{subtype}'") from e

        return subtype(**d, **kwargs)

    @property
    def mat_id(self) -> str:
        """Material ID for use in scene dictionaries.
        
        Returns:
            String identifier with '_mat_' prefix
        """
        return f"_mat_{self.id}"

    def kdict(self, mode: str = "mono") -> dict:
        """Generate kernel dictionary fragment for this material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary fragment for Eradiate kernel scene
        """
        raise NotImplementedError

    def kpmap(self, mode: str = "mono") -> dict:
        """Generate scene parameter map fragment for this material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary fragment for Eradiate parameter map
        """
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for YAML serialization
        """
        raise NotImplementedError


@attrs.define
class DiffuseMaterial(Material):
    """Material with diffuse reflectance properties.
    
    Represents surfaces with Lambertian reflection behavior.
    
    Args:
        id: Unique material identifier
        reflectance: Function returning reflectance value for given wavelength
    """

    id: str = attrs.field(converter=str)
    reflectance: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )

    def kdict(self, mode: str = "mono") -> dict:
        """Generate kernel dictionary for diffuse material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with diffuse BSDF definition
        """
        result = {
            self.mat_id: {"type": "diffuse", "id": self.mat_id, "reflectance": 0.0}
        }
        return result

    def kpmap(self, mode: str = "mono") -> dict:
        """Generate parameter map for diffuse material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with reflectance parameter update specification
        """
        result = {}

        if mode == "mono":
            node_id = self.mat_id
            param = "reflectance.value"
            result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
                self.reflectance, node_id=node_id, param=param
            )

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary with material type and spectral data references
        """
        return {
            "type": "diffuse",
            "reflectance": {"path": "spectrum_diffuse.nc", "variable": "reflectance"}
        }


@attrs.define
class BilambertianMaterial(Material):
    """Material with Lambertian reflection and transmission.
    
    Represents surfaces like vegetation that both reflect and transmit light.
    
    Args:
        id: Unique material identifier
        reflectance: Function returning reflectance value for given wavelength
        transmittance: Function returning transmittance value for given wavelength
    """

    id: str = attrs.field(converter=str)
    reflectance: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )
    transmittance: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )

    def kdict(self, mode: str = "mono") -> dict:
        """Generate kernel dictionary for bilambertian material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with bilambertian BSDF definition
        """
        result = {self.mat_id: {"type": "bilambertian", "id": self.mat_id}}
        return result

    def kpmap(self, mode: str = "mono") -> dict:
        """Generate parameter map for bilambertian material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with reflectance and transmittance parameter updates
        """
        result = {}

        if mode == "mono":
            node_id = self.mat_id
            for param, func in [
                ("reflectance.value", self.reflectance),
                ("transmittance.value", self.transmittance),
            ]:
                result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
                    func, node_id=node_id, param=param
                )

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary with material type and spectral data references
        """
        return {
            "type": "bilambertian",
            "reflectance": {"path": "spectrum_vegetation.nc", "variable": "reflectance"},
            "transmittance": {"path": "spectrum_vegetation.nc", "variable": "transmittance"}
        }


@attrs.define
class RPVMaterial(Material):
    """Material using the RPV reflection model.
    
    Implements the Rahman-Pinty-Verstraete model for rough surface reflection.
    
    Args:
        id: Unique material identifier
        rho_0: Function returning rho_0 parameter value
        k: Function returning k parameter value  
        Theta: Function returning Theta parameter value
        rho_c: Function returning rho_c parameter value
    """

    id: str = attrs.field(converter=str)
    rho_0: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )
    k: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )
    Theta: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )
    rho_c: Callable[[KernelContext], float] = attrs.field(
        converter=_spectral_parameter_converter
    )

    def kdict(self, mode: str = "mono") -> dict:
        """Generate kernel dictionary for RPV material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with RPV BSDF definition
        """
        result = {self.mat_id: {"type": "rpv", "id": self.mat_id, "rho_c": 0.5}}
        return result

    def kpmap(self, mode: str = "mono") -> dict:
        """Generate parameter map for RPV material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with RPV parameter update specifications
        """
        result = {}

        if mode == "mono":
            node_id = self.mat_id
            for param, func in [
                ("rho_0.value", self.rho_0),
                ("k.value", self.k),
                ("g.value", self.Theta),
                ("rho_c.value", self.rho_c),
            ]:
                result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
                    func, node_id=node_id, param=param
                )

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary with material type and spectral data references
        """
        return {
            "type": "rpv",
            "rho_0": {"path": "spectrum_soil.nc", "variable": "rho_0"},
            "k": {"path": "spectrum_soil.nc", "variable": "k"},
            "Theta": {"path": "spectrum_soil.nc", "variable": "Theta"},
            "rho_c": {"path": "spectrum_soil.nc", "variable": "rho_c"}
        }


@attrs.define
class OceanLegacyMaterial(Material):
    """Material using the 6SV ocean reflection model.
    
    Implements the ocean BRDF model from the 6S radiative transfer code.
    
    Args:
        id: Unique material identifier
        chlorinity: Chlorinity content of the ocean water
        pigmentation: Pigmentation level of the ocean water
        wind_speed: Wind speed in m/s
        wind_direction: Wind direction in degrees (North=0, clockwise)
        shininess: Shininess parameter for importance sampling (optional)
        shadowing: Whether to account for shadowing-masking effects
    """

    id: str = attrs.field(converter=str)
    chlorinity = attrs.field(converter=float)
    pigmentation = attrs.field(converter=float)
    wind_speed = attrs.field(converter=float)
    wind_direction = attrs.field(converter=float)
    shininess = attrs.field(default=None, converter=attrs.converters.optional(float))
    shadowing = attrs.field(default=True, converter=bool)

    def default_shininess(self):
        """Calculate default shininess value for multiple importance sampling.
        
        Returns:
            Shininess value computed from wind speed
        """
        return (37.2455 - self.wind_speed) ** 1.15

    def kdict(self, mode: str = "mono") -> dict:
        """Generate kernel dictionary for ocean material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with ocean_legacy BSDF definition
        """
        result = {
            self.mat_id: {
                "type": "ocean_legacy",
                "id": self.mat_id,
                "wavelength": 550.0,
                "chlorinity": self.chlorinity,
                "pigmentation": self.pigmentation,
                "wind_speed": self.wind_speed,
                "wind_direction": self.wind_direction,
            }
        }
        return result

    def kpmap(self, mode: str = "mono") -> dict:
        """Generate parameter map for ocean material.
        
        Args:
            mode: Rendering mode ('mono' or 'rgb')
            
        Returns:
            Dictionary with ocean parameter update specifications
        """
        result = {}

        if mode == "mono":
            node_id = self.mat_id
            for param, func in [
                ("wavelength", lambda ctx: ctx.si.w.m_as("nm")),
                ("chlorinity", self.chlorinity),
                ("pigmentation", self.pigmentation),
                ("wind_speed", self.wind_speed),
                ("wind_direction", self.wind_direction),
            ]:
                if not callable(func):
                    def func(_: KernelContext, value: float = func) -> float:
                        return value

                result[f"{node_id}.{param}"] = _declare_mono_scene_parameter(
                    func, node_id=node_id, param=param
                )

        return result

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization.
        
        Returns:
            Dictionary with material type and parameter values
        """
        return {
            "type": "ocean_legacy",
            "chlorinity": self.chlorinity,
            "pigmentation": self.pigmentation,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction
        }