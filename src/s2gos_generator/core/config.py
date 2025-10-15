from __future__ import annotations

import importlib.resources
import json
import os
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils import validate_config_version
from s2gos_utils.io.paths import exists, open_file, read_yaml
from s2gos_utils.io.resolver import resolver
from upath import UPath

from .._version import get_version


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
    package_root = importlib.resources.files("s2gos_generator")
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
    dem_root_dir: str = Field(..., description="Root directory for DEM data")
    landcover_index_path: str = Field(..., description="Path to landcover index file")
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
    flatten_dem: bool = Field(
        False, description="Flatten DEM to zero elevation for testing"
    )


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

    details: Annotated[AtmosphereTypeConfig, Field(..., discriminator="type")]

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


class HamsterConfig(BaseModel):
    """HAMSTER albedo data configuration for baresoil material replacement."""

    enabled: bool = Field(True, description="Enable HAMSTER albedo for baresoil")
    data_path: UPath = Field(..., description="Path to HAMSTER NetCDF data file")
    variable_name: str = Field("albedo", description="Variable name in NetCDF file")
    fallback_on_error: bool = Field(
        True, description="Fall back to standard baresoil material on errors"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {UPath: lambda v: str(v)},
    }

    @field_validator("data_path", mode="before")
    @classmethod
    def validate_data_path(cls, v):
        """Validate HAMSTER data file exists."""
        v = UPath(v)
        if not exists(v):
            raise ValueError(f"HAMSTER data file not found: {v}")
        return v


class UserAssets(BaseModel):
    """User assets to be placed on scene."""

    object_id: str = Field(..., description="Unique identifier for the object")
    ply_path: UPath = Field(
        ..., description="Path to PLY file containing 3D object geometry"
    )
    coordinate: list[float] = Field(
        ..., description="Object placement coordinates [lon, lat]"
    )
    material: str = Field(
        ...,
        description="Material reference (string ID) - must exist in scene material library",
    )
    elevation_offset: float = Field(
        0.0, description="Height offset above terrain surface in meters"
    )
    scale: float = Field(1.0, description="Uniform scaling factor for the object")
    rotation_x: float = Field(0.0, description="Rotation around X-axis in degrees")
    rotation_y: float = Field(0.0, description="Rotation around Y-axis in degrees")
    rotation_z: float = Field(0.0, description="Rotation around Z-axis in degrees")
    face_normals: Optional[bool] = Field(
        None,
        description="Mitsuba PLY face normals setting: True=smooth normals, False=per-face normals, None=use PLY file defaults",
    )

    @field_validator("coordinate")
    @classmethod
    def validate_coordinate(cls, v):
        """Validate coordinate format."""
        if len(v) != 2:
            raise ValueError("Coordinate must be [longitude, latitude]")
        lon, lat = v
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} out of valid range [-180, 180]")
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of valid range [-90, 90]")
        return v

    @field_validator("ply_path", mode="before")
    @classmethod
    def validate_ply_path(cls, v):
        """Validate PLY file exists."""
        v = UPath(v)
        if not exists(v):
            raise ValueError(f"PLY file not found: {v}")
        return v

    @field_validator("material")
    @classmethod
    def validate_material(cls, v):
        """Validate material reference is a non-empty string."""
        if not isinstance(v, str):
            raise ValueError(
                f"Material must be a string reference, got {type(v).__name__}. "
                "Inline material definitions are no longer supported. "
                "Define materials in the scene's material library."
            )
        if not v.strip():
            raise ValueError("Material reference cannot be empty")
        return v.strip()

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v):
        """Validate scale is positive."""
        if v <= 0:
            raise ValueError("Scale must be positive")
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {UPath: lambda v: str(v)},
        "validate_assignment": True,
        "extra": "forbid",
    }


class XmlSceneConfig(BaseModel):
    """Configuration for importing assets and materials from XML scene files."""

    xml_path: UPath = Field(..., description="Path to XML scene file")
    base_coordinate: Tuple[float, float] = Field(
        ..., description="Base geographic coordinate [longitude, latitude]"
    )
    object_id_prefix: Optional[str] = Field(
        None, description="Prefix for asset object IDs"
    )
    elevation_offset: float = Field(
        0.0, description="Global elevation offset for all assets in meters"
    )
    scale: float = Field(
        1.0, gt=0.0, description="Global scaling factor for all assets"
    )
    fix_blender_coords: bool = Field(
        True, description="Apply Blender coordinate system correction"
    )
    material_mappings: Optional[Dict[str, str]] = Field(
        None, description="Material name mapping dictionary"
    )
    pattern_type: str = Field(
        "wildcard", description="Pattern matching type for materials"
    )
    validate_materials: bool = Field(
        True, description="Validate that all materials are properly defined"
    )

    @field_validator("xml_path", mode="before")
    @classmethod
    def validate_xml_path(cls, v):
        """Validate XML file exists."""
        v = UPath(v)
        if not exists(v):
            raise ValueError(f"XML file not found: {v}")
        return v

    @field_validator("base_coordinate")
    @classmethod
    def validate_base_coordinate(cls, v):
        """Validate base coordinate format."""
        if len(v) != 2:
            raise ValueError("Base coordinate must be [longitude, latitude]")
        lon, lat = v
        if not (-180 <= lon <= 180):
            raise ValueError(f"Longitude {lon} out of valid range [-180, 180]")
        if not (-90 <= lat <= 90):
            raise ValueError(f"Latitude {lat} out of valid range [-90, 90]")
        return v

    @field_validator("pattern_type")
    @classmethod
    def validate_pattern_type(cls, v):
        """Validate pattern type is supported."""
        valid_types = {"wildcard", "exact", "contains"}
        if v not in valid_types:
            raise ValueError(f"Pattern type must be one of: {valid_types}")
        return v

    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {UPath: lambda v: str(v)},
        "validate_assignment": True,
        "extra": "forbid",
    }


class VegetationSpecies(BaseModel):
    """Configuration for a single vegetation species.

    Defines placement parameters for a vegetation type (e.g., oak trees, shrubs).
    Multiple species can be assigned to the same landcover class for mixed vegetation.
    """

    name: str = Field(
        description="Species identifier (e.g., 'oak_trees', 'berry_bushes')"
    )
    asset_xml_paths: Union[List[str], Dict[str, float]] = Field(
        description="Asset XML file path(s). Use list for uniform distribution or dict for weighted distribution"
    )
    density_per_hectare: float = Field(
        ge=0.0, le=4000.0, description="Density for this species"
    )
    scale_min: float = Field(ge=0.1, description="Minimum scale factor")
    scale_max: float = Field(ge=0.1, description="Maximum scale factor")
    spillover_enabled: bool = Field(
        False, description="Enable spillover into adjacent compatible landcover classes"
    )
    spillover_compatibility: Optional[Dict[int, float]] = Field(
        None,
        description="Per-species spillover compatibility map (overrides global). Maps landcover class to probability 0.0-1.0",
    )

    @field_validator("scale_max")
    @classmethod
    def validate_scale_range(cls, v, info):
        """Ensure scale_max > scale_min."""
        if "scale_min" in info.data and v <= info.data["scale_min"]:
            raise ValueError("scale_max must be greater than scale_min")
        return v

    @field_validator("asset_xml_paths")
    @classmethod
    def validate_asset_paths(cls, v):
        """Validate asset paths and weights."""
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("asset_xml_paths list cannot be empty")
        elif isinstance(v, dict):
            if len(v) == 0:
                raise ValueError("asset_xml_paths dict cannot be empty")
            for path, weight in v.items():
                if weight <= 0:
                    raise ValueError(f"Weight must be positive for {path}: {weight}")
        return v

    def get_asset_paths_and_weights(self) -> Tuple[List[str], List[float]]:
        """Get asset paths and normalized weights for selection.

        Returns:
            (paths, weights) tuple ready for random.choices()
        """
        if isinstance(self.asset_xml_paths, list):
            return (self.asset_xml_paths, [1.0] * len(self.asset_xml_paths))
        else:
            paths = list(self.asset_xml_paths.keys())
            weights = list(self.asset_xml_paths.values())
            return (paths, weights)

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class VegetationPlacementConfig(BaseModel):
    """Configuration for multi-species vegetation placement system.

    Controls how vegetation instances are distributed across the scene based on
    landcover classifications. Supports multiple species per landcover class.

    Configuration levels:
    - Per-species parameters: density, scale, asset (in VegetationSpecies)
    - Global parameters: spacing, variation, limits (this class)

    Example:
        config = VegetationPlacementConfig(
            enabled=True,
            landcover_species_mapping={
                10: [VegetationSpecies(name="oak", asset_xml_path="oak.xml", ...)],
                20: [VegetationSpecies(name="shrub", asset_xml_path="shrub.xml", ...)]
            },
            min_spacing=2.0,
            density_variation=0.3
        )
    """

    enabled: bool = Field(
        True, description="Enable vegetation placement based on landcover data"
    )

    landcover_species_mapping: Dict[int, List[VegetationSpecies]] = Field(
        default_factory=lambda: {
            10: [
                VegetationSpecies(
                    name="oak_trees",
                    asset_xml_path="tree.xml",
                    density_per_hectare=400.0,
                    scale_min=10.0,
                    scale_max=35.0,
                )
            ]
        },
        description="Mapping from landcover class to list of vegetation species",
    )

    min_spacing: float = Field(
        2.0,
        ge=0.1,
        description="Global minimum spacing between any vegetation instances (meters)",
    )
    density_variation: float = Field(
        0.3, ge=0.0, le=1.0, description="Random variation in density (±30% by default)"
    )
    max_instances_per_pixel: int = Field(
        50, ge=1, le=10000, description="Performance limit per pixel across all species"
    )
    rotation_range: float = Field(
        360.0,
        ge=0.0,
        le=360.0,
        description="Random rotation range in degrees (azimuth around vertical axis)",
    )
    tilt_range: float = Field(
        6.0,
        ge=0.0,
        le=23.0,
        description="Random tilt range in degrees (±deviation from vertical for natural variation)",
    )
    spillover_max_distance_m: float = Field(
        30.0,
        ge=0.0,
        le=300.0,
        description="Maximum distance (meters) for spillover from primary landcover class",
    )
    spillover_compatibility: Dict[int, float] = Field(
        default_factory=lambda: {
            20: 0.8,  # Shrubland - high compatibility
            30: 0.7,  # Grassland - moderate compatibility
            40: 0.3,  # Cropland - low compatibility
            90: 0.4,  # Herbaceous Wetland - moderate compatibility
            60: 0.1,  # Bare/sparse vegetation - very low compatibility
        },
        description="Default spillover compatibility map. Maps landcover class to probability 0.0-1.0. Can be overridden per species.",
    )

    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
    }


class SceneGenConfig(BaseModel):
    """
    Comprehensive scene configuration using Pydantic.

    Provides a modern, validated, and flexible configuration system for scene generation.
    """

    config_version: str = Field(
        default_factory=get_version, description="Configuration schema version"
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
    enable_buffer: bool = Field(False, description="Enable buffer area processing")
    enable_background: bool = Field(
        False, description="Enable background area processing"
    )

    buffer_size_km: float = Field(60.0, gt=0.0, description="Buffer size in kilometers")
    buffer_resolution_m: float = Field(
        100.0, gt=0.0, description="Buffer resolution in meters"
    )

    background_size_km: float = Field(
        200.0, gt=0.0, description="Background area size in kilometers"
    )
    background_resolution_m: float = Field(
        200.0, gt=0.0, description="Background resolution in meters"
    )
    background_elevation: float = Field(
        0.0, description="Background elevation in meters"
    )
    hamster: Optional[HamsterConfig] = Field(
        None, description="HAMSTER albedo data configuration for baresoil"
    )
    user_assets: list[UserAssets] = Field(
        [], description="User assets to be placed in generated scene"
    )
    xml_scenes: list[XmlSceneConfig] = Field(
        [], description="XML scene files to import for additional assets and materials"
    )
    vegetation_placement: Optional[VegetationPlacementConfig] = Field(
        None,
        description="Vegetation placement configuration (None disables vegetation)",
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
        if self.enable_buffer:
            if self.buffer_size_km <= self.location.aoi_size_km:
                raise ValueError("Buffer size must be larger than AOI size")

        return self

    @property
    def trees_enabled(self) -> bool:
        """Backward compatibility property for trees_enabled check."""
        return (
            self.vegetation_placement is not None and self.vegetation_placement.enabled
        )

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

        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = UPath(data["output_dir"])

        if (
            "hamster" in data
            and "data_path" in data["hamster"]
            and isinstance(data["hamster"]["data_path"], str)
        ):
            data["hamster"]["data_path"] = UPath(data["hamster"]["data_path"])

        # Simple version validation only
        validate_config_version(
            "scene_config", data, get_version(), "scene generation configuration"
        )

        return cls(**data)

    def enable_hamster_albedo(
        self,
        data_path: UPath,
        variable_name: str = "albedo",
        fallback_on_error: bool = True,
    ):
        """Enable HAMSTER albedo data for baresoil material replacement.

        Args:
            data_path: Path to HAMSTER NetCDF data file
            variable_name: Variable name in NetCDF file (default: "albedo")
            fallback_on_error: Fall back to standard baresoil material on errors
        """
        self.hamster = HamsterConfig(
            enabled=True,
            data_path=data_path,
            variable_name=variable_name,
            fallback_on_error=fallback_on_error,
        )

    def disable_hamster_albedo(self):
        """Disable HAMSTER albedo system."""
        self.hamster = None

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

        if self.enable_buffer:
            if self.buffer_size_km <= self.location.aoi_size_km:
                errors.append("Buffer size must be larger than AOI size")

        for xml_scene_config in self.xml_scenes:
            if not exists(xml_scene_config.xml_path):
                errors.append(f"XML file not found: {xml_scene_config.xml_path}")

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
        """Check if buffer area is enabled."""
        return self.enable_buffer

    @property
    def has_background(self) -> bool:
        """Check if background area is enabled."""
        return self.enable_background


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
    **kwargs,
) -> SceneGenConfig:
    """Scene generation configuration using PathResolver.

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
    data_sources = DataSources(**(data_overrides or {}))

    return SceneGenConfig(
        scene_name=scene_name,
        description=description,
        location=SceneLocation(
            center_lat=center_lat, center_lon=center_lon, aoi_size_km=aoi_size_km
        ),
        data_sources=data_sources,
        output_dir=output_dir,
        processing=ProcessingOptions(target_resolution_m=target_resolution_m),
        atmosphere=atmosphere or _default_atmosphere_config(),
        **kwargs,
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


def load_assets_from_xml(
    xml_path: str,
    base_coordinate: List[float],
    object_id_prefix: str = "asset",
    elevation_offset: float = 0.0,
    scale: float = 1.0,
    fix_blender_coords: bool = True,
    material_mappings: Optional[Dict[str, str]] = None,
    pattern_type: str = "wildcard",
    validate_materials: bool = True,
) -> Tuple[List[UserAssets], Dict[str, Dict[str, Any]]]:
    """Load multi-material assets from Mitsuba XML with material library.

    Args:
        xml_path: Path to Mitsuba XML file
        base_coordinate: [longitude, latitude] for all asset components
        object_id_prefix: Prefix for asset IDs
        elevation_offset: Height offset above terrain (meters)
        scale: Uniform scaling factor
        fix_blender_coords: Apply Blender→Mitsuba coordinate correction (90° X rotation)
        material_mappings: Dict mapping filename patterns to S2GOS material names
        pattern_type: "wildcard", "exact", or "contains" matching for material_mappings
        validate_materials: If True, validate material references and PLY file existence

    Returns:
        Tuple of (assets_list, material_library):
        - assets_list: List of UserAssets with string material references
        - material_library: Dict of material definitions to embed in scene

    Example:
        assets, materials = load_assets_from_xml(
            "fence.xml",
            base_coordinate=[15.1258741, -23.6015431],
            material_mappings={
                "Post_*": "concrete",  # Reference to scene material library
                "*Wire*": "metal_wire", # Will use XML material if available
            }
        )
    """
    # Import using new XML importer
    from ..assets.xml_importer import import_xml_assets

    asset_data_list, material_library = import_xml_assets(
        xml_path=xml_path,
        base_coordinate=base_coordinate,
        object_id_prefix=object_id_prefix,
        elevation_offset=elevation_offset,
        scale=scale,
        fix_blender_coords=fix_blender_coords,
        material_mappings=material_mappings,
        pattern_type=pattern_type,
        validate_materials=validate_materials,
    )

    # Convert asset data to UserAssets objects
    assets = []
    for asset_data in asset_data_list:
        asset = UserAssets(
            object_id=asset_data["object_id"],
            ply_path=UPath(asset_data["ply_path"]),
            coordinate=asset_data["coordinate"],
            material=asset_data["material"],  # Now guaranteed to be string reference
            elevation_offset=asset_data["elevation_offset"],
            scale=asset_data["scale"],
            rotation_x=asset_data["rotation_x"],
            rotation_y=asset_data["rotation_y"],
            rotation_z=asset_data["rotation_z"],
        )
        assets.append(asset)

    return assets, material_library
