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


def create_molecular_atmosphere(
    thermoprops_config,
    absorption_database: str = None,
    has_absorption: bool = True,
    has_scattering: bool = True,
    is_polarized: bool = None,
) -> dict:
    """
    Generate a molecular atmosphere specification using joseki thermophysical properties.

    Parameters
    ----------
    thermoprops_config : ThermophysicalConfig
        Configuration for thermophysical properties.
    absorption_database : str, optional
        Absorption database identifier (e.g., "monotropa").
    has_absorption : bool
        Enable absorption calculations.
    has_scattering : bool
        Enable scattering calculations.
    is_polarized : bool, optional
        If True, use specific settings for polarized scenarios.
        
    Returns
    -------
    dict
        Molecular atmosphere configuration dictionary.
    """
    if is_polarized is None:
        is_polarized = eradiate.mode().is_polarized

    altitudes = np.arange(
        thermoprops_config.altitude_min,
        thermoprops_config.altitude_max + thermoprops_config.altitude_step,
        thermoprops_config.altitude_step
    ) * ureg.m

    tp = joseki.make(
        identifier=thermoprops_config.identifier,
        z=altitudes,
    )
    
    if thermoprops_config.constituent_scaling:
        scaling_dict = {
            species: concentration * ureg.ppm 
            for species, concentration in thermoprops_config.constituent_scaling.items()
        }
        tp.joseki.rescale_to(scaling_dict)

    molecular_config = {
        "type": "molecular",
        "thermoprops": tp,
        "has_absorption": has_absorption,
        "has_scattering": has_scattering,
        "rayleigh_depolarization": "bodhaine" if is_polarized else 0.0,
    }
    
    if absorption_database:
        molecular_config["absorption_data"] = absorption_database

    return molecular_config


def create_particle_layer_from_config(
    particle_config,
    is_polarized: bool = None,
) -> dict:
    """
    Create a particle layer from enhanced configuration.
    
    Parameters
    ----------
    particle_config : ParticleLayerConfig
        Enhanced particle layer configuration.
    is_polarized : bool, optional
        If True, use specific settings for polarized scenarios.
        
    Returns
    -------
    dict
        Particle layer configuration dictionary.
    """
    if is_polarized is None:
        is_polarized = eradiate.mode().is_polarized

    layer_config = {
        "type": "particle_layer",
        "bottom": particle_config.altitude_bottom * ureg.m,
        "top": particle_config.altitude_top * ureg.m,
        "dataset": particle_config.aerosol_dataset.value,
        "w_ref": particle_config.reference_wavelength * ureg.nm,
        "has_absorption": particle_config.has_absorption,
    }

    if particle_config.distribution.type == "exponential":
        height = particle_config.altitude_top - particle_config.altitude_bottom
        rate = height / particle_config.distribution.scale_height
        layer_config["distribution"] = {
            "type": "exponential",
            "rate": rate
        }
        layer_config["tau_ref"] = particle_config.optical_thickness

    elif particle_config.distribution.type == "gaussian":
        normalized_center = (
            (particle_config.distribution.center_altitude - particle_config.altitude_bottom) /
            (particle_config.altitude_top - particle_config.altitude_bottom)
        )
        normalized_width = (
            particle_config.distribution.width /
            (particle_config.altitude_top - particle_config.altitude_bottom)
        )
        layer_config["distribution"] = {
            "type": "gaussian",
            "center": normalized_center,
            "width": normalized_width
        }
        layer_config["tau_ref"] = particle_config.optical_thickness

    elif particle_config.distribution.type == "uniform":
        layer_config["distribution"] = {"type": "uniform"}
        layer_config["tau_ref"] = particle_config.optical_thickness

    return layer_config


def create_heterogeneous_atmosphere(
    molecular_config=None,
    particle_configs=None,
    is_polarized: bool = None,
) -> dict:
    """
    Generate a heterogeneous atmosphere with molecular and particle components.

    Parameters
    ----------
    molecular_config : MolecularAtmosphereConfig, optional
        Molecular atmosphere configuration.
    particle_configs : list of ParticleLayerConfig, optional
        List of particle layer configurations.
    is_polarized : bool, optional
        If True, use specific settings for polarized scenarios.
        
    Returns
    -------
    dict
        Heterogeneous atmosphere configuration dictionary.
    """
    if is_polarized is None:
        is_polarized = eradiate.mode().is_polarized

    atmosphere_config = {"type": "heterogeneous"}

    if molecular_config:
        molecular_atm = create_molecular_atmosphere(
            molecular_config.thermoprops,
            molecular_config.absorption_database.value if molecular_config.absorption_database else None,
            molecular_config.has_absorption,
            molecular_config.has_scattering,
            is_polarized,
        )
        atmosphere_config["molecular_atmosphere"] = molecular_atm

    if particle_configs:
        if len(particle_configs) == 1:
            atmosphere_config["particle_layers"] = create_particle_layer_from_config(
                particle_configs[0], is_polarized
            )
        else:
            atmosphere_config["particle_layers"] = [
                create_particle_layer_from_config(config, is_polarized)
                for config in particle_configs
            ]

    return atmosphere_config