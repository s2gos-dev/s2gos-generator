from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils.io.paths import PathRef

from ._utils import _resolve_asset_path


class AerosolDataset(str, Enum):
    """Comprehensive aerosol datasets from Eradiate."""

    SIXSV_CONTINENTAL = "sixsv-continental"
    SIXSV_MARITIME = "sixsv-maritime"
    SIXSV_URBAN = "sixsv-urban"
    SIXSV_DESERT = "sixsv-desert"
    SIXSV_BIOMASS_BURNING = "sixsv-biomass_burning"
    SIXSV_STRATOSPHERIC = "sixsv-stratospheric"
    GOVAERTS_2021_CONTINENTAL_EXTRAPOLATED = "govaerts_2021-continental-extrapolated"
    GOVAERTS_2021_DESERT_EXTRAPOLATED = "govaerts_2021-desert-extrapolated"


class AbsorptionDatabase(str, Enum):
    """Absorption databases from Eradiate."""

    GECKO = "gecko"
    KOMODO = "komodo"
    MONOTROPA = "monotropa"
    MYCENA = "mycena"
    PANELLUS = "panellus"
    TUBER = "tuber"


class AtmosphereType(str, Enum):
    """Atmosphere types aligned with Eradiate's atmosphere classes."""

    MOLECULAR = "molecular"
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


class ThermophysicalConfig(BaseModel):
    """Configuration for atmospheric thermophysical properties.

    Supports either joseki identifiers (e.g., 'afgl_1986-us_standard') or
    CAMS NetCDF files. Specify one but not both.
    """

    model_config = {"arbitrary_types_allowed": True}

    identifier: Optional[str] = Field(
        None, description="Standard atmosphere identifier (joseki)"
    )
    thermoprops_file: Optional[PathRef] = Field(
        None,
        description="Path to CAMS thermoprops NetCDF file (alternative to identifier)",
    )
    altitude_min: float = Field(0.0, ge=0.0, description="Minimum altitude in meters")
    altitude_max: float = Field(
        120000.0, gt=0.0, description="Maximum altitude in meters"
    )
    altitude_step: float = Field(1000.0, gt=0.0, description="Altitude step in meters")
    constituent_scaling: Optional[dict[str, float]] = Field(
        None, description="Constituent concentration scaling (e.g., {'CO2': 400.0})"
    )

    @model_validator(mode="after")
    def validate_altitude_range(self):
        """Validate altitude configuration."""
        if self.altitude_max <= self.altitude_min:
            raise ValueError("Maximum altitude must be greater than minimum altitude")
        return self

    @field_validator("thermoprops_file", mode="before")
    @classmethod
    def validate_thermaproprs_path(cls, v):
        """Validate and resolve thermaproprs file path using configured search paths."""
        if v is not None:
            resolved = _resolve_asset_path(v, asset_type="NetCDF")
            return resolved

    @model_validator(mode="after")
    def validate_thermoprops_source(self):
        """Ensure exactly one source is used, applying defaults if necessary."""
        has_identifier = self.identifier is not None
        has_file = self.thermoprops_file is not None

        if has_identifier and has_file:
            raise ValueError(
                "Specify either 'identifier' (joseki) OR 'thermoprops_file' (CAMS NetCDF), not both."
            )

        if not has_identifier and not has_file:
            self.identifier = "afgl_1986-us_standard"

        return self


class MolecularAtmosphereConfig(BaseModel):
    """Configuration for a purely molecular (Rayleigh-scattering) atmosphere.

    Attributes:
        type: Discriminator literal fixed to ``"molecular"``.
        thermoprops: Thermophysical profile (identifier or CAMS NetCDF file).
        absorption_database: Database for gas absorption.
        ``None`` disables absorption (equivalent to ``has_absorption=False``).
        has_absorption: Enable gas absorption calculations.
        has_scattering: Enable Rayleigh scattering calculations.
    """

    type: Literal["molecular"] = "molecular"
    thermoprops: ThermophysicalConfig = Field(
        default_factory=ThermophysicalConfig,
        description="Thermophysical properties configuration",
    )
    absorption_database: Optional[AbsorptionDatabase] = Field(
        None, description="Absorption database to use"
    )
    has_absorption: bool = Field(True, description="Enable absorption calculations")
    has_scattering: bool = Field(True, description="Enable scattering calculations")


class HomogeneousAtmosphereConfig(BaseModel):
    """Configuration for a spatially uniform (homogeneous) aerosol atmosphere.

    Attributes:
        type: Discriminator literal fixed to ``"homogeneous"``.
        aerosol_dataset: Aerosol dataset defining scattering (``sigma_s``) and
            absorption (``sigma_a``) phase-function properties.
        optical_thickness: Aerosol optical depth at the reference wavelength.
        scale_height: Exponential decay scale height (metres) for the vertical
            aerosol profile.
        reference_wavelength: Wavelength (nm) at which ``optical_thickness``
            is specified.
        has_absorption: Enable aerosol absorption.
    """

    type: Literal["homogeneous"] = "homogeneous"
    aerosol_dataset: AerosolDataset = Field(
        AerosolDataset.SIXSV_CONTINENTAL, description="Aerosol dataset to use"
    )
    optical_thickness: float = Field(
        0.1, ge=0.0, le=5.0, description="Aerosol optical thickness"
    )
    scale_height: float = Field(
        1000.0, gt=0.0, description="Aerosol scale height in meters"
    )
    reference_wavelength: float = Field(
        550.0, gt=0.0, description="Reference wavelength in nm"
    )
    has_absorption: bool = Field(True, description="Enable absorption by aerosols")


class ParticleDistribution(BaseModel):
    """Base class for particle distribution configurations."""

    type: str = Field(..., description="Distribution type")


class ExponentialDistribution(ParticleDistribution):
    """Exponential vertical decay profile for particle concentration,
    see Eradiate documentation for more details.

    Attributes:
        type: Discriminator literal fixed to ``"exponential"``.
        rate: Decay rate ``λ`` (1/m). Mutually exclusive with ``scale``.
        scale: Scale parameter ``β = 1/λ`` (m). Mutually exclusive with
            ``rate``.
    """

    type: Literal["exponential"] = "exponential"
    rate: Optional[float] = Field(
        None, gt=0.0, description="Eradiate decay rate λ (default 5.0)"
    )
    scale: Optional[float] = Field(None, gt=0.0, description="Eradiate scale β = 1/λ")

    @model_validator(mode="after")
    def validate_exclusive_params(self):
        """Validate that rate and scale are mutually exclusive per Eradiate API."""
        if self.rate is not None and self.scale is not None:
            raise ValueError("rate and scale are mutually exclusive per Eradiate API")
        return self


class GaussianDistribution(ParticleDistribution):
    """Gaussian vertical profile for particle concentration,
    see Eradiate documentation for more details.

    Attributes:
        type: Discriminator literal fixed to ``"gaussian"``.
        center_altitude: Altitude of peak concentration (m).
        width: Standard deviation of the distribution (m).
    """

    type: Literal["gaussian"] = "gaussian"
    center_altitude: float = Field(..., description="Center altitude in meters")
    width: float = Field(..., gt=0.0, description="Distribution width in meters")


class UniformDistribution(ParticleDistribution):
    """Constant (uniform) vertical distribution of particle concentration.

    Attributes:
        type: Discriminator literal fixed to ``"uniform"``.
    """

    type: Literal["uniform"] = "uniform"


DistributionType = Union[
    ExponentialDistribution, GaussianDistribution, UniformDistribution
]


class ParticleLayerConfig(BaseModel):
    """Configuration for a particle layer.

    Attributes:
        aerosol_dataset: Aerosol dataset. Either an ``AerosolDataset`` enum
         value or a path to a custom NetCDF file.
        optical_thickness: Column aerosol optical depth within this layer.
        altitude_bottom: Lower bound of the layer (m above sea level).
        altitude_top: Upper bound of the layer (m above sea level).
        distribution: Vertical distribution of particle concentration within
            the layer bounds.
        reference_wavelength: Wavelength (nm) at which ``optical_thickness``
            is defined.
        has_absorption: Enable particle absorption.
    """

    aerosol_dataset: Union[AerosolDataset, str] = Field(
        ...,
        description="Aerosol dataset: enum value (e.g. 'sixsv-continental') or custom NetCDF path",
    )
    optical_thickness: float = Field(
        ..., ge=0.0, description="Aerosol optical thickness"
    )
    altitude_bottom: float = Field(..., ge=0.0, description="Bottom altitude in meters")
    altitude_top: float = Field(..., gt=0.0, description="Top altitude in meters")
    distribution: DistributionType = Field(
        ..., description="Particle distribution configuration"
    )
    reference_wavelength: float = Field(
        550.0, gt=0.0, description="Reference wavelength in nm"
    )
    has_absorption: bool = Field(True, description="Enable absorption by particles")

    @field_validator("aerosol_dataset")
    @classmethod
    def validate_aerosol_dataset(cls, v):
        """Allow enum or custom file path string."""
        if isinstance(v, AerosolDataset):
            return v.value
        elif isinstance(v, str):
            # Check if it's a valid enum value
            try:
                return AerosolDataset(v).value
            except ValueError:
                # Custom path - validate .nc extension
                if not v.endswith(".nc"):
                    raise ValueError(
                        f"Custom aerosol dataset must be NetCDF file (.nc): {v}"
                    )
                return v
        raise ValueError(
            f"aerosol_dataset must be AerosolDataset enum or str, got {type(v)}"
        )

    @model_validator(mode="after")
    def validate_altitude_range(self):
        """Validate altitude configuration."""
        if self.altitude_top <= self.altitude_bottom:
            raise ValueError("Top altitude must be greater than bottom altitude")
        return self


class HeterogeneousAtmosphereConfig(BaseModel):
    """Configuration for a vertically-resolved heterogeneous atmosphere.

    Combines an optional molecular background (Rayleigh scattering and gas
    absorption) with one or more discrete particle layers.
    At least one of ``molecular`` or ``particle_layers`` must be specified.

    Attributes:
        type: Discriminator literal fixed to ``"heterogeneous"``.
        molecular: Molecular atmosphere providing the Rayleigh-scattering
            background. ``None`` omits the molecular component.
        particle_layers: Ordered list of aerosol or particle layers stacked
            within the atmosphere column.
    """

    type: Literal["heterogeneous"] = "heterogeneous"
    molecular: MolecularAtmosphereConfig = Field(
        None, description="Molecular atmosphere configuration"
    )
    particle_layers: list[ParticleLayerConfig] = Field(
        None, description="Particle layer configurations"
    )

    @model_validator(mode="after")
    def validate_heterogeneous_config(self):
        """Validate that at least one component is configured."""
        if not self.molecular and not self.particle_layers:
            raise ValueError(
                "Heterogeneous atmosphere requires at least molecular atmosphere or particle layers"
            )
        return self


AtmosphereTypeConfig = Union[
    MolecularAtmosphereConfig,
    HomogeneousAtmosphereConfig,
    HeterogeneousAtmosphereConfig,
]


class AtmosphereConfig(BaseModel):
    """Comprehensive atmosphere configuration supporting multiple types."""

    boa: float = Field(
        0.0, ge=0.0, description="Bottom of atmosphere altitude in meters"
    )
    toa: float = Field(
        75000.0, gt=0.0, description="Top of atmosphere altitude in meters"
    )

    details: Annotated[AtmosphereTypeConfig, Field(..., discriminator="type")]

    @model_validator(mode="after")
    def validate_atmosphere_config(self):
        """Validate atmosphere configuration based on type."""
        if self.toa <= self.boa:
            raise ValueError(
                "Top of atmosphere must be higher than bottom of atmosphere"
            )
        return self


def _default_atmosphere_config() -> AtmosphereConfig:
    """Create a default atmosphere configuration matching eradiate defaults."""
    return AtmosphereConfig(
        details=MolecularAtmosphereConfig(
            thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
            absorption_database=None,  # No absorption by default
            has_absorption=False,  # Match eradiate sigma_a=0.0 default
            has_scattering=True,  # Air scattering like eradiate sigma_s default
        ),
    )


def create_molecular_atmosphere_config(
    identifier: str = "afgl_1986-us_standard",
    altitude_max: float = 120000.0,
    absorption_database: Optional[AbsorptionDatabase] = None,
    co2_concentration: Optional[float] = None,
) -> AtmosphereConfig:
    """Create molecular atmosphere configuration.

    Args:
        identifier: Standard atmosphere identifier
        altitude_max: Maximum altitude in meters
        absorption_database: Absorption database to use
        co2_concentration: CO2 concentration in ppm (if different from standard)

    Returns:
        AtmosphereConfig for molecular atmosphere
    """
    thermoprops = ThermophysicalConfig(
        identifier=identifier,
        altitude_max=altitude_max,
        constituent_scaling={"CO2": co2_concentration} if co2_concentration else None,
    )

    molecular_config = MolecularAtmosphereConfig(
        thermoprops=thermoprops, absorption_database=absorption_database
    )

    return AtmosphereConfig(
        boa=0.0,
        toa=altitude_max,
        details=molecular_config,
    )


def create_custom_particle_layer(
    aerosol_dataset: AerosolDataset,
    optical_thickness: float,
    altitude_bottom: float = 0.0,
    altitude_top: float = 10000.0,
    distribution_type: str = "exponential",
    scale_height: float = 1000.0,
) -> ParticleLayerConfig:
    """Create a custom particle layer configuration.

    Args:
        aerosol_dataset: Aerosol dataset to use
        optical_thickness: Aerosol optical thickness
        altitude_bottom: Bottom altitude in meters
        altitude_top: Top altitude in meters
        distribution_type: Distribution type ("exponential", "uniform")
        scale_height: Scale height for exponential distribution

    Returns:
        ParticleLayerConfig
    """
    if distribution_type == "exponential":
        distribution = ExponentialDistribution(rate=1.0 / scale_height)
    elif distribution_type == "uniform":
        distribution = UniformDistribution()
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")

    return ParticleLayerConfig(
        aerosol_dataset=aerosol_dataset,
        optical_thickness=optical_thickness,
        altitude_bottom=altitude_bottom,
        altitude_top=altitude_top,
        distribution=distribution,
    )


def create_heterogeneous_atmosphere_config(
    molecular_config: Optional[MolecularAtmosphereConfig] = None,
    particle_layers: Optional[list[ParticleLayerConfig]] = None,
    toa: float = 75000.0,
) -> AtmosphereConfig:
    """Create heterogeneous atmosphere configuration.

    Args:
        molecular_config: Molecular atmosphere configuration
        particle_layers: List of particle layer configurations
        toa: Top of atmosphere altitude

    Returns:
        AtmosphereConfig for heterogeneous atmosphere
    """
    heterogeneous_config = HeterogeneousAtmosphereConfig(
        molecular=molecular_config, particle_layers=particle_layers
    )
    return AtmosphereConfig(
        boa=0.0,
        toa=toa,
        details=heterogeneous_config,
    )
