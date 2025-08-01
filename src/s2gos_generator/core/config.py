from __future__ import annotations

import json
import logging
import os
import importlib.resources
from datetime import datetime
from enum import Enum
from upath import UPath
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils.io.paths import open_file, read_yaml, exists
from s2gos_utils.io.resolver import resolver
from s2gos_utils.typing import PathLike


def _parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch) tuple."""
    try:
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError("Version must have exactly 3 parts (major.minor.patch)")
        return tuple(int(part) for part in parts)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid version format '{version_str}': {e}")


def _check_config_version_compatibility(config_version: str, current_version: str = "1.0.0") -> None:
    """Check if config version is compatible with current code version.
    
    Args:
        config_version: Version from loaded config
        current_version: Current code version (default: "1.0.0")
        
    Raises:
        ValueError: If major version mismatch (incompatible)
        
    Logs:
        Warning: If minor/patch version differences exist
    """
    try:
        config_major, config_minor, config_patch = _parse_version(config_version)
        current_major, current_minor, current_patch = _parse_version(current_version)
    except ValueError as e:
        logging.warning(f"Could not parse version numbers: {e}")
        return
    
    # Major version mismatch is incompatible
    if config_major != current_major:
        raise ValueError(
            f"Incompatible config version: config uses v{config_version} "
            f"but current code expects v{current_major}.x.x. "
            f"Please update your configuration or use compatible code version."
        )
    
    # Minor/patch differences get warnings
    if config_minor != current_minor or config_patch != current_patch:
        if config_minor > current_minor or (config_minor == current_minor and config_patch > current_patch):
            logging.warning(
                f"Config version v{config_version} is newer than current code v{current_version}. "
                f"Some features may not work as expected."
            )
        else:
            logging.warning(
                f"Config version v{config_version} is older than current code v{current_version}. "
                f"Consider updating your configuration."
            )


class AerosolDataset(str, Enum):
    """Comprehensive aerosol datasets from Eradiate."""

    # SIXSV datasets
    SIXSV_CONTINENTAL = "sixsv-continental"
    SIXSV_MARITIME = "sixsv-maritime"
    SIXSV_URBAN = "sixsv-urban"
    SIXSV_DESERT = "sixsv-desert"

    # Additional Eradiate aerosol datasets
    ELTERMAN_CLEAR = "elterman-clear"
    ELTERMAN_HAZY = "elterman-hazy"
    MCCLATCHY_CLEAR = "mcclatchy-clear"
    MCCLATCHY_HAZY = "mcclatchy-hazy"


class AbsorptionDatabase(str, Enum):
    """Absorption databases from Eradiate."""

    GECKO = "gecko"
    KOMODO = "komodo"
    MONOTROPA = "monotropa"
    HALENIA = "halenia"
    HITRAN_2020 = "hitran-2020"


class AtmosphereType(str, Enum):
    """Atmosphere types aligned with Eradiate's atmosphere classes."""

    MOLECULAR = "molecular"  # MolecularAtmosphere - clear sky, gaseous only
    HOMOGENEOUS = "homogeneous"  # HomogeneousAtmosphere - uniform optical properties
    HETEROGENEOUS = (
        "heterogeneous"  # HeterogeneousAtmosphere - molecular + particle layers
    )


class SceneLocation(BaseModel):
    """Geographic location configuration."""

    center_lat: float = Field(
        ..., ge=-90.0, le=90.0, description="Center latitude in degrees"
    )
    center_lon: float = Field(
        ..., ge=-180.0, le=180.0, description="Center longitude in degrees"
    )
    aoi_size_km: float = Field(
        ..., gt=0.0, description="Area of interest size in kilometers"
    )


def _load_default_data_sources_config() -> Dict[str, Any]:
    """Load default paths from defaults.yaml file.
    
    The defaults file location can be overridden using the S2GOS_DEFAULTS_PATH
    environment variable. If not set, uses the package's defaults.yaml file.
    """
    package_root = importlib.resources.files('s2gos_generator')
    defaults_path = package_root / "defaults.yaml"

    env_path = os.getenv("S2GOS_GEN_DEFAULTS_PATH")
    if env_path:
        defaults_path = UPath(env_path)
        if not exists(defaults_path):
            return {}
        return read_yaml(defaults_path)
    
    if not exists(defaults_path):
        return {}
    
    defaults = read_yaml(defaults_path)
    
    for key, value in defaults.items():
            defaults[key] = str(resolver.resolve(value))
    
    return defaults


class DataSources(BaseModel):
    """Data source configuration using FileResolver."""

    dem_index_path: str = Field(..., description="Path to DEM index file")
    dem_root_dir: str = Field(
        ..., description="Root directory for DEM data"
    )
    landcover_index_path: str = Field(
        ..., description="Path to landcover index file"
    )
    landcover_root_dir: str = Field(
        ..., description="Root directory for landcover data"
    )
    material_config_path: str = Field(
        ..., description="Path to custom material configuration JSON"
    )

    @model_validator(mode="before")
    @classmethod
    def _load_defaults_and_merge_overrides(cls, data: Any) -> Any:
        """
        Load defaults from YAML and merge them with user-provided data.
        This allows a base configuration to be set while still allowing
        users to specify their own paths. The user's data takes precedence.
        """
        if not isinstance(data, dict):
            # Let Pydantic handle validation for non-dictionary inputs.
            return data
        
        default_config = _load_default_data_sources_config()
        default_config.update(data)
        return default_config

    @field_validator(
        "dem_index_path",
        "landcover_index_path",
        "material_config_path",
        "dem_root_dir",
        "landcover_root_dir",
    )
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files or directories exist."""
        path = UPath(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v
    

class ProcessingOptions(BaseModel):
    """Processing options for scene generation."""

    target_resolution_m: float = Field(
        30.0, gt=0.0, description="Target resolution in meters"
    )
    generate_texture_preview: bool = Field(
        True, description="Generate texture preview images"
    )
    handle_dem_nans: bool = Field(True, description="Handle NaN values in DEM data")
    dem_fillna_value: float = Field(0.0, description="Fill value for DEM NaN values")


class ThermophysicalConfig(BaseModel):
    """Configuration for atmospheric thermophysical properties using joseki."""

    identifier: str = Field(
        "afgl_1986-us_standard", description="Standard atmosphere identifier"
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


class MolecularAtmosphereConfig(BaseModel):
    """Configuration for molecular atmosphere using Eradiate's MolecularAtmosphere."""
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
    """Configuration for homogeneous atmosphere with uniform optical properties."""
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


class HeterogeneousAtmosphereConfig(BaseModel):
    """Configuration for heterogeneous atmosphere with molecular background and particle layers."""
    type: Literal["heterogeneous"] = "heterogeneous"
    molecular: Optional[MolecularAtmosphereConfig] = Field(
        None, description="Molecular atmosphere configuration"
    )
    particle_layers: Optional[list[ParticleLayerConfig]] = Field(
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

class ParticleDistribution(BaseModel):
    """Base class for particle distribution configurations."""

    type: str = Field(..., description="Distribution type")


class ExponentialDistribution(ParticleDistribution):
    """Exponential particle distribution - direct Eradiate API mapping."""

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
    """Gaussian particle distribution."""

    type: Literal["gaussian"] = "gaussian"
    center_altitude: float = Field(..., description="Center altitude in meters")
    width: float = Field(..., gt=0.0, description="Distribution width in meters")


class UniformDistribution(ParticleDistribution):
    """Uniform particle distribution."""

    type: Literal["uniform"] = "uniform"


DistributionType = Union[
    ExponentialDistribution, GaussianDistribution, UniformDistribution
]


class ParticleLayerConfig(BaseModel):
    """Enhanced particle layer configuration."""

    aerosol_dataset: AerosolDataset = Field(..., description="Aerosol dataset to use")
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

    @model_validator(mode="after")
    def validate_altitude_range(self):
        """Validate altitude configuration."""
        if self.altitude_top <= self.altitude_bottom:
            raise ValueError("Top altitude must be greater than bottom altitude")
        return self


class AtmosphereConfig(BaseModel):
    """Comprehensive atmosphere configuration supporting multiple types."""

    boa: float = Field(
        0.0, ge=0.0, description="Bottom of atmosphere altitude in meters"
    )
    toa: float = Field(
        40000.0, gt=0.0, description="Top of atmosphere altitude in meters"
    )

    details: Annotated[
        AtmosphereTypeConfig, Field(..., discriminator="type")
    ]

    @model_validator(mode="after")
    def validate_atmosphere_config(self):
        """Validate atmosphere configuration based on type."""
        if self.toa <= self.boa:
            raise ValueError(
                "Top of atmosphere must be higher than bottom of atmosphere"
            )
        return self


def _default_atmosphere_config() -> "AtmosphereConfig":
    """Create a default atmosphere configuration matching eradiate defaults."""
    return AtmosphereConfig(
        details=MolecularAtmosphereConfig(
            thermoprops=ThermophysicalConfig(identifier="afgl_1986-us_standard"),
            absorption_database=None,  # No absorption by default
            has_absorption=False,  # Match eradiate sigma_a=0.0 default
            has_scattering=True,  # Air scattering like eradiate sigma_s default
        ),
    )


class BufferConfig(BaseModel):
    """Combined buffer and background configuration."""

    buffer_size_km: float = Field(..., gt=0.0, description="Buffer size in kilometers")
    buffer_resolution_m: float = Field(
        100.0, gt=0.0, description="Buffer resolution in meters"
    )
    background_elevation: float = Field(
        0.0, description="Background elevation in meters"
    )
    background_size_km: float = Field(
        100.0, gt=0.0, description="Background area size in kilometers"
    )
    background_resolution_m: float = Field(
        200.0, gt=0.0, description="Background resolution in meters"
    )

    @model_validator(mode="after")
    def validate_buffer_config(self):
        """Validate buffer configuration."""
        if self.background_resolution_m < self.buffer_resolution_m:
            raise ValueError(
                "Background resolution must be equal to or lower than buffer resolution"
            )

        return self


class SceneGenConfig(BaseModel):
    """
    Comprehensive scene configuration using Pydantic.

    Provides a modern, validated, and flexible configuration system for scene generation.
    """

    config_version: str = Field(
        "1.0.0", description="Configuration schema version"
    )
    scene_name: str = Field(
        ..., min_length=1, description="Scene name (used for output files)"
    )
    description: Optional[str] = Field(None, description="Scene description")

    location: SceneLocation = Field(..., description="Geographic location")
    data_sources: DataSources = Field(..., description="Data source configuration")
    output_dir: UPath = Field(..., description="Output directory for generated scene")
    processing: ProcessingOptions = Field(
        default_factory=ProcessingOptions, description="Processing options"
    )
    atmosphere: AtmosphereConfig = Field(
        default_factory=_default_atmosphere_config,
        description="Atmosphere configuration",
    )
    buffer: Optional[BufferConfig] = Field(
        None, description="Buffer and background configuration"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Configuration creation time"
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "arbitrary_types_allowed": True,
        "json_encoders": {datetime: lambda v: v.isoformat(), UPath: lambda v: str(v)},
    }

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v):
        """Validate and create output directory if needed."""
        from s2gos_utils.io.paths import mkdir
        v = UPath(v)
        mkdir(v)
        return v

    @model_validator(mode="after")
    def validate_scene_config(self):
        """Validate complete scene configuration."""
        if self.buffer:
            if self.buffer.buffer_size_km <= self.location.aoi_size_km:
                raise ValueError("Buffer size must be larger than AOI size")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def to_json(self, path: Optional[UPath] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            with open_file(path, "w") as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, path: UPath) -> "SceneGenConfig":
        """Load from JSON file with version compatibility checking."""
        with open_file(path, "r") as f:
            data = json.load(f)
        
        config_version = data.get("config_version", "1.0.0")
        _check_config_version_compatibility(config_version)
        
        return cls(**data)

    def enable_buffer_system(
        self,
        buffer_size_km: float,
        buffer_resolution_m: float = 100.0,
        background_elevation: float = 0.0,
        background_size_km: float = 100.0,
        background_resolution_m: float = 200.0,
    ):
        """Enable and configure buffer/background system."""
        self.buffer = BufferConfig(
            buffer_size_km=buffer_size_km,
            buffer_resolution_m=buffer_resolution_m,
            background_elevation=background_elevation,
            background_size_km=background_size_km,
            background_resolution_m=background_resolution_m,
        )

    def disable_buffer_system(self):
        """Disable buffer/background system."""
        self.buffer = None

    def set_atmosphere_homogeneous(
        self,
        aerosol_dataset: AerosolDataset = AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness: float = 0.1,
        scale_height: float = 1000.0,
    ):
        """Set atmosphere using homogeneous configuration."""
        homogeneous_config = HomogeneousAtmosphereConfig(
            aerosol_dataset=aerosol_dataset,
            optical_thickness=optical_thickness,
            scale_height=scale_height,
        )
        self.atmosphere = AtmosphereConfig(
            details=homogeneous_config,
        )

    def set_atmosphere_molecular(self, molecular_config: MolecularAtmosphereConfig):
        """Set atmosphere using molecular configuration."""
        self.atmosphere = AtmosphereConfig(
            details=molecular_config,
        )

    def set_atmosphere_heterogeneous(
        self,
        molecular_config: Optional[MolecularAtmosphereConfig] = None,
        particle_layers: Optional[list[ParticleLayerConfig]] = None,
    ):
        """Set atmosphere using heterogeneous configuration with molecular and particle layers."""
        heterogeneous_config = HeterogeneousAtmosphereConfig(
            molecular=molecular_config, particle_layers=particle_layers
        )
        self.atmosphere = AtmosphereConfig(
            details=heterogeneous_config,
        )

    def validate_configuration(self) -> list[str]:
        """Validate the complete configuration and return any errors."""
        errors = []

        if self.buffer:
            if self.buffer.buffer_size_km <= self.location.aoi_size_km:
                errors.append("Buffer size must be larger than AOI size")

        return errors

    @property
    def scene_output_dir(self) -> UPath:
        """Get the specific output directory for this scene."""
        return self.output_dir / self.scene_name

    @property
    def meshes_dir(self) -> UPath:
        """Get the meshes output directory."""
        return self.scene_output_dir / "meshes"

    @property
    def textures_dir(self) -> UPath:
        """Get the textures output directory."""
        return self.scene_output_dir / "textures"

    @property
    def data_dir(self) -> UPath:
        """Get the data output directory."""
        return self.scene_output_dir / "data"

    @property
    def has_buffer(self) -> bool:
        """Check if buffer/background system is enabled."""
        return self.buffer is not None


def create_scene_config(
    scene_name: str,
    center_lat: float,
    center_lon: float,
    aoi_size_km: float,
    output_dir: UPath,
    target_resolution_m: float = 30.0,
    description: Optional[str] = None,
    data_overrides: Optional[dict] = None,
    atmosphere: Optional[AtmosphereConfig] = None,
    **kwargs
) -> SceneGenConfig:
    """Revolutionary scene configuration using PathResolver.
    
    This modern approach eliminates verbose conditional logic and uses the
    global resolver to find data files elegantly.
    
    Args:
        scene_name: Scene name (used for output files)
        center_lat: Center latitude in degrees
        center_lon: Center longitude in degrees
        aoi_size_km: Area of interest size in kilometers
        output_dir: Output directory for generated scene
        target_resolution_m: Target resolution in meters (default: 30.0)
        description: Optional scene description
        data_overrides: Optional dict with user data overrides:
            - dem_index: Custom DEM index file
            - landcover_index: Custom landcover index file  
            - materials_config: Custom materials config file
        atmosphere: Optional atmosphere configuration
        **kwargs: Additional configuration options
    """
    # Create data sources with resolver-powered elegance
    data_sources = DataSources(**(data_overrides or {}))
    
    return SceneGenConfig(
        scene_name=scene_name,
        description=description,
        location=SceneLocation(
            center_lat=center_lat, 
            center_lon=center_lon, 
            aoi_size_km=aoi_size_km
        ),
        data_sources=data_sources,
        output_dir=output_dir,
        processing=ProcessingOptions(target_resolution_m=target_resolution_m),
        atmosphere=atmosphere or _default_atmosphere_config(),
        **kwargs
    )


def create_clear_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for clear conditions."""
    homogeneous_config = HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.05,  # Low aerosol
        scale_height=1000.0,
    )
    return AtmosphereConfig(
        boa=0.0,
        toa=40000.0,
        details=homogeneous_config,
    )


def create_hazy_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for hazy conditions."""
    homogeneous_config = HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.3,  # High aerosol
        scale_height=1000.0,
    )
    return AtmosphereConfig(
        boa=0.0,
        toa=40000.0,
        details=homogeneous_config,
    )


def create_maritime_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for maritime conditions."""
    homogeneous_config = HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_MARITIME,
        optical_thickness=0.15,
        scale_height=1000.0,
    )
    return AtmosphereConfig(
        boa=0.0,
        toa=40000.0,
        details=homogeneous_config,
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
    toa: float = 40000.0,
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
