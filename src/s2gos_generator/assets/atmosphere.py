import eradiate
import joseki
import numpy as np
import xarray as xr
from eradiate.units import unit_registry as ureg


def exp_scale(
    z0: float = 0.0,
    z1: float = 0.0,
    z2: float = 1.0,
    rate: float = 5.0,
):
    """
    Compute a scaling factor such that an exponential distribution with a
    normalized integral between z0 and z2 has, after scaling,
    an integral between z1 and z2 equal to 1.
    """
    n = np.exp(-z0 * rate) - np.exp(-z2 * rate)
    d = np.exp(-z1 * rate) - np.exp(-z2 * rate)
    return n / d


@ureg.wraps(None, ("m", "m", "m", None, "m"), strict=False)
def particle_layer(
    altitude_bottom: float = 0.0,
    altitude_ref: float | None = None,
    altitude_top: float = 120e3,
    tau_ref: float = 0.2,
    scale_ref: float = 1e3,
) -> dict:
    """
    Create a particle layer with an exponential density distribution.
    
    Parameters
    ----------
    altitude_bottom : float
        Bottom altitude of the particle layer.
    altitude_ref : float, optional
        Reference altitude within the particle layer. If unset, the bottom
        altitude is used.
    altitude_top : float
        Top altitude of the particle layer.
    scale_ref : float
        Scaling distance for the exponential density distribution.
    tau_ref : float
        Optical thickness at the reference altitude.
        
    Returns
    -------
    dict
        Particle layer configuration dictionary.
    """
    if altitude_ref is None:
        altitude_ref = altitude_bottom

    height = altitude_top - altitude_bottom
    z0 = 0.0
    z1 = (altitude_ref - altitude_bottom) / height
    z2 = 1.0
    rate = height / scale_ref
    tau = tau_ref * exp_scale(z0, z1, z2, rate=rate)

    return {
        "type": "particle_layer",
        "bottom": altitude_bottom * ureg.m,
        "top": altitude_top * ureg.m,
        "distribution": {"type": "exponential", "rate": rate},
        "tau_ref": tau,
    }


@ureg.wraps(None, ("m", "m", "m", "m", "m", "m", None, "", None), strict=False)
def create_atmosphere(
    boa: float = 0.0,
    toa: float = 4e4,
    aerosol_bottom: float | None = None,
    aerosol_ref: float | None = None,
    aerosol_top: float | None = None,
    aerosol_scale: float = 1e3,
    aerosol_ds: str | xr.Dataset = "sixsv-continental",
    aerosol_ot: float = 0.1,
    is_polarized: bool | None = None,
) -> dict:
    """
    Generate an atmosphere specification with molecular and aerosol components.

    Parameters
    ----------
    boa : float
        Bottom-of-atmosphere altitude [m].
    toa : float
        Top-of-atmosphere altitude [m].
    aerosol_bottom : float, optional
        Altitude of bottom of aerosol layer [m]. If unset, boa is used.
    aerosol_ref : float, optional
        Reference altitude for the aerosol density [m]. If unset,
        aerosol_bottom is used.
    aerosol_top : float, optional
        Altitude of top of aerosol layer [m]. If unset, toa is used.
    aerosol_scale : float
        Characteristic cutoff distance of the exponential aerosol density
        profile [m].
    aerosol_ot : float
        Aerosol optical thickness at reference wavelength (550 nm) and reference
        altitude.
    aerosol_ds : Dataset or str
        Aerosol property dataset (e.g., "sixsv-continental", "sixsv-maritime").
    is_polarized : bool, optional
        If True, use specific settings for polarized scenarios. If unset,
        default based on the currently active mode.
        
    Returns
    -------
    dict
        Atmosphere configuration dictionary.
    """
    if is_polarized is None:
        is_polarized = eradiate.mode().is_polarized

    if aerosol_bottom is None:
        aerosol_bottom = boa

    if aerosol_ref is None:
        aerosol_ref = aerosol_bottom

    if aerosol_top is None:
        aerosol_top = toa

    # Create thermodynamic profile using US Standard atmosphere
    tp = joseki.make(
        identifier="afgl_1986-us_standard",
        z=np.arange(boa, toa + 10.0, 100.0) * ureg.m,
    )
    tp.joseki.rescale_to({"CO2": 360.0 * ureg.ppm})

    # Create aerosol layer
    aerosol = particle_layer(
        altitude_bottom=aerosol_bottom,
        altitude_ref=aerosol_ref,
        altitude_top=aerosol_top,
        tau_ref=aerosol_ot,
        scale_ref=aerosol_scale,
    )
    aerosol.update({
        "dataset": aerosol_ds, 
        "w_ref": 550.0 * ureg.nm, 
        "has_absorption": True
    })

    # Combine molecular and particle components
    atmosphere_config = {
        "type": "heterogeneous",
        "molecular_atmosphere": {
            "has_absorption": False,
            "thermoprops": tp,
            "rayleigh_depolarization": "bodhaine" if is_polarized else 0.0,
        },
        "particle_layers": aerosol,
    }

    return atmosphere_config