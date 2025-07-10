from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import os
import json
import yaml
import fsspec

from .paths import read_yaml, exists


def load_defaults() -> Dict[str, Any]:
    """Load default paths from defaults.yaml file."""
    # Look for defaults.yaml in the generator package directory
    defaults_path = Path(__file__).parent.parent.parent.parent / "defaults.yaml"
    
    if not exists(defaults_path):
        return {}
    
    defaults = read_yaml(defaults_path)
    
    # Convert relative paths to absolute paths (relative to generator package)
    generator_root = Path(__file__).parent.parent.parent.parent
    for key, value in defaults.items():
        if isinstance(value, str) and not value.startswith(("/", "s3://", "gcs://", "azure://")):
            defaults[key] = generator_root / value
    
    return defaults


def open_file(path: Union[Path, str], mode: str = 'r'):
    """Open a file using fsspec for unified access."""
    return fsspec.open(str(path), mode=mode)


def _serialize_path(path: Union[Path, str, None]) -> Optional[str]:
    """Serialize Path object to string using __fspath__ protocol."""
    if path is None:
        return None
    return os.fspath(path)


def _deserialize_path(path_str: Optional[str]) -> Optional[Path]:
    """Deserialize string to Path object."""
    if path_str is None:
        return None
    return Path(path_str)


class MaterialType(str, Enum):
    """Material types for scene surfaces."""
    DIFFUSE = "diffuse"
    RPV = "rpv"
    BILAMBERTIAN = "bilambertian"
    OCEAN_LEGACY = "ocean_legacy"


class BackgroundMaterial(str, Enum):
    """Available background materials."""
    WATER = "water"
    BARESOIL = "baresoil"
    CONCRETE = "concrete"
    SNOW = "snow"
    MOSS = "moss"
    TREECOVER = "treecover"
    SHRUBLAND = "shrubland"
    GRASSLAND = "grassland"
    CROPLAND = "cropland"
    MANGROVES = "mangroves"
    WETLAND = "wetland"


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
    HETEROGENEOUS = "heterogeneous"  # HeterogeneousAtmosphere - molecular + particle layers


class SceneLocation(BaseModel):
    """Geographic location configuration."""
    center_lat: float = Field(..., ge=-90.0, le=90.0, description="Center latitude in degrees")
    center_lon: float = Field(..., ge=-180.0, le=180.0, description="Center longitude in degrees")
    aoi_size_km: float = Field(..., gt=0.0, description="Area of interest size in kilometers")


class DataSources(BaseModel):
    """Data source configuration with validation."""
    dem_index_path: Union[Path, str] = Field(..., description="Path to DEM index file")
    dem_root_dir: Union[Path, str] = Field(..., description="Root directory for DEM data")
    landcover_index_path: Union[Path, str] = Field(..., description="Path to landcover index file")
    landcover_root_dir: Union[Path, str] = Field(..., description="Root directory for landcover data")
    material_config_path: Union[Path, str] = Field(..., description="Path to custom material configuration JSON")
    
    @field_validator('dem_index_path', 'landcover_index_path', 'material_config_path')
    @classmethod
    def validate_files(cls, v):
        """Validate that files exist for local paths."""
        path_str = str(v)
        # Skip validation for remote paths
        if path_str.startswith(("s3://", "gcs://", "azure://", "http://", "https://")):
            return v
        # Validate local paths
        path = Path(v)
        if not path.exists():
            raise ValueError(f"File does not exist: {v}")
        return v
    
    @field_validator('dem_root_dir', 'landcover_root_dir')
    @classmethod
    def validate_dirs(cls, v):
        """Validate that directories exist for local paths."""
        path_str = str(v)
        # Skip validation for remote paths
        if path_str.startswith(("s3://", "gcs://", "azure://", "http://", "https://")):
            return v
        # Validate local paths
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {v}")
        return v


class ProcessingOptions(BaseModel):
    """Processing options for scene generation."""
    target_resolution_m: float = Field(30.0, gt=0.0, description="Target resolution in meters")
    generate_texture_preview: bool = Field(True, description="Generate texture preview images")
    handle_dem_nans: bool = Field(True, description="Handle NaN values in DEM data")
    dem_fillna_value: float = Field(0.0, description="Fill value for DEM NaN values")


class ThermophysicalConfig(BaseModel):
    """Configuration for atmospheric thermophysical properties using joseki."""
    identifier: str = Field("afgl_1986-us_standard", description="Standard atmosphere identifier")
    altitude_min: float = Field(0.0, ge=0.0, description="Minimum altitude in meters")
    altitude_max: float = Field(120000.0, gt=0.0, description="Maximum altitude in meters")
    altitude_step: float = Field(1000.0, gt=0.0, description="Altitude step in meters")
    constituent_scaling: Optional[dict[str, float]] = Field(None, description="Constituent concentration scaling (e.g., {'CO2': 400.0})")
    
    @model_validator(mode='after')
    def validate_altitude_range(self):
        """Validate altitude configuration."""
        if self.altitude_max <= self.altitude_min:
            raise ValueError("Maximum altitude must be greater than minimum altitude")
        return self


class MolecularAtmosphereConfig(BaseModel):
    """Configuration for molecular atmosphere using Eradiate's MolecularAtmosphere."""
    thermoprops: ThermophysicalConfig = Field(default_factory=ThermophysicalConfig, description="Thermophysical properties configuration")
    absorption_database: Optional[AbsorptionDatabase] = Field(None, description="Absorption database to use")
    has_absorption: bool = Field(True, description="Enable absorption calculations")
    has_scattering: bool = Field(True, description="Enable scattering calculations")


class HomogeneousAtmosphereConfig(BaseModel):
    """Configuration for homogeneous atmosphere with uniform optical properties."""
    aerosol_dataset: AerosolDataset = Field(AerosolDataset.SIXSV_CONTINENTAL, description="Aerosol dataset to use")
    optical_thickness: float = Field(0.1, ge=0.0, le=5.0, description="Aerosol optical thickness")
    scale_height: float = Field(1000.0, gt=0.0, description="Aerosol scale height in meters")
    reference_wavelength: float = Field(550.0, gt=0.0, description="Reference wavelength in nm")
    has_absorption: bool = Field(True, description="Enable absorption by aerosols")


class HeterogeneousAtmosphereConfig(BaseModel):
    """Configuration for heterogeneous atmosphere with molecular background and particle layers."""
    molecular: Optional[MolecularAtmosphereConfig] = Field(None, description="Molecular atmosphere configuration")
    particle_layers: Optional[list[ParticleLayerConfig]] = Field(None, description="Particle layer configurations")
    
    @model_validator(mode='after')
    def validate_heterogeneous_config(self):
        """Validate that at least one component is configured."""
        if not self.molecular and not self.particle_layers:
            raise ValueError("Heterogeneous atmosphere requires at least molecular atmosphere or particle layers")
        return self


class ParticleDistribution(BaseModel):
    """Base class for particle distribution configurations."""
    type: str = Field(..., description="Distribution type")


class ExponentialDistribution(ParticleDistribution):
    """Exponential particle distribution - direct Eradiate API mapping."""
    type: Literal["exponential"] = "exponential"
    rate: Optional[float] = Field(None, gt=0.0, description="Eradiate decay rate λ (default 5.0)")
    scale: Optional[float] = Field(None, gt=0.0, description="Eradiate scale β = 1/λ")
    
    @model_validator(mode='after')
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


DistributionType = Union[ExponentialDistribution, GaussianDistribution, UniformDistribution]
    

class ParticleLayerConfig(BaseModel):
    """Enhanced particle layer configuration."""
    aerosol_dataset: AerosolDataset = Field(..., description="Aerosol dataset to use")
    optical_thickness: float = Field(..., ge=0.0, description="Aerosol optical thickness")
    altitude_bottom: float = Field(..., ge=0.0, description="Bottom altitude in meters")
    altitude_top: float = Field(..., gt=0.0, description="Top altitude in meters")
    distribution: DistributionType = Field(..., description="Particle distribution configuration")
    reference_wavelength: float = Field(550.0, gt=0.0, description="Reference wavelength in nm")
    has_absorption: bool = Field(True, description="Enable absorption by particles")
    
    @model_validator(mode='after')
    def validate_altitude_range(self):
        """Validate altitude configuration."""
        if self.altitude_top <= self.altitude_bottom:
            raise ValueError("Top altitude must be greater than bottom altitude")
        return self


AtmosphereTypeConfig = Union[MolecularAtmosphereConfig, HomogeneousAtmosphereConfig, HeterogeneousAtmosphereConfig]


class AtmosphereConfig(BaseModel):
    """Comprehensive atmosphere configuration supporting multiple types."""
    type: AtmosphereType = Field(AtmosphereType.HOMOGENEOUS, description="Atmosphere type")
    
    boa: float = Field(0.0, ge=0.0, description="Bottom of atmosphere altitude in meters")
    toa: float = Field(40000.0, gt=0.0, description="Top of atmosphere altitude in meters")
    
    # Type-specific configurations
    molecular: Optional[MolecularAtmosphereConfig] = Field(None, description="Molecular atmosphere configuration")
    homogeneous: Optional[HomogeneousAtmosphereConfig] = Field(None, description="Homogeneous atmosphere configuration")
    heterogeneous: Optional[HeterogeneousAtmosphereConfig] = Field(None, description="Heterogeneous atmosphere configuration")
    
    
    @model_validator(mode='after')
    def validate_atmosphere_config(self):
        """Validate atmosphere configuration based on type."""
        if self.toa <= self.boa:
            raise ValueError("Top of atmosphere must be higher than bottom of atmosphere")
        
        # Validate type-specific configurations
        if self.type == AtmosphereType.MOLECULAR:
            if not self.molecular:
                raise ValueError("Molecular atmosphere type requires 'molecular' configuration")
        elif self.type == AtmosphereType.HOMOGENEOUS:
            if not self.homogeneous:
                raise ValueError("Homogeneous atmosphere type requires 'homogeneous' configuration")
        elif self.type == AtmosphereType.HETEROGENEOUS:
            if not self.heterogeneous:
                raise ValueError("Heterogeneous atmosphere type requires 'heterogeneous' configuration")
        
        return self
    
    def get_config_for_type(self) -> AtmosphereTypeConfig:
        """Get the appropriate configuration object for the current atmosphere type."""
        if self.type == AtmosphereType.MOLECULAR:
            if self.molecular:
                return self.molecular
            else:
                raise ValueError("No molecular configuration available")
        elif self.type == AtmosphereType.HOMOGENEOUS:
            if self.homogeneous:
                return self.homogeneous
            else:
                raise ValueError("No homogeneous configuration available")
        elif self.type == AtmosphereType.HETEROGENEOUS:
            if self.heterogeneous:
                return self.heterogeneous
            else:
                raise ValueError("No heterogeneous configuration available")
        else:
            raise ValueError(f"Unknown atmosphere type: {self.type}")


def _default_atmosphere_config() -> 'AtmosphereConfig':
    """Create a default atmosphere configuration matching eradiate defaults."""
    return AtmosphereConfig(
        type=AtmosphereType.MOLECULAR,
        molecular=MolecularAtmosphereConfig(
            thermoprops=ThermophysicalConfig(
                identifier="afgl_1986-us_standard"
            ),
            absorption_database=None,  # No absorption by default
            has_absorption=False,      # Match eradiate sigma_a=0.0 default
            has_scattering=True        # Air scattering like eradiate sigma_s default
        )
    )


class BufferConfig(BaseModel):
    """Combined buffer and background configuration."""
    enabled: bool = Field(False, description="Enable buffer/background system")
    buffer_size_km: float = Field(..., gt=0.0, description="Buffer size in kilometers")
    buffer_resolution_m: float = Field(100.0, gt=0.0, description="Buffer resolution in meters")
    background_elevation: float = Field(0.0, description="Background elevation in meters")
    background_size_km: float = Field(100.0, gt=0.0, description="Background area size in kilometers")
    background_resolution_m: float = Field(200.0, gt=0.0, description="Background resolution in meters")
    
    @model_validator(mode='after')
    def validate_buffer_config(self):
        """Validate buffer configuration when enabled."""
        if not self.enabled:
            return self
        
        if self.buffer_size_km <= 0:
            raise ValueError("Buffer size must be positive when buffer is enabled")
        
        if self.background_size_km <= 0:
            raise ValueError("Background size must be positive when buffer is enabled")
        
        if self.background_resolution_m < self.buffer_resolution_m:
            raise ValueError("Background resolution must be equal to or lower than buffer resolution")
        
        return self




class SceneGenConfig(BaseModel):
    """
    Comprehensive scene configuration using Pydantic.
    
    Provides a modern, validated, and flexible configuration system for scene generation.
    """
    
    scene_name: str = Field(..., min_length=1, description="Scene name (used for output files)")
    description: Optional[str] = Field(None, description="Scene description")
    
    location: SceneLocation = Field(..., description="Geographic location")
    data_sources: DataSources = Field(..., description="Data source configuration")
    output_dir: Path = Field(..., description="Output directory for generated scene")
    processing: ProcessingOptions = Field(default_factory=ProcessingOptions, description="Processing options")    
    atmosphere: AtmosphereConfig = Field(default_factory=_default_atmosphere_config, description="Atmosphere configuration")
    buffer: Optional[BufferConfig] = Field(None, description="Buffer and background configuration")
    created_at: datetime = Field(default_factory=datetime.now, description="Configuration creation time")
    
    model_config = {
        "validate_assignment": True,
        "extra": "forbid",
        "json_encoders": {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v)
        }
    }
    
    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v):
        """Validate and create output directory if needed."""
        v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @model_validator(mode='after')
    def validate_scene_config(self):
        """Validate complete scene configuration."""
        if self.buffer and self.buffer.enabled:
            if self.buffer.buffer_size_km <= self.location.aoi_size_km:
                raise ValueError("Buffer size must be larger than AOI size")
        
        return self
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()
    
    def to_json(self, path: Optional[Path] = None, indent: int = 2) -> str:
        """Export to JSON format."""
        json_str = self.model_dump_json(indent=indent)
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
        return json_str
    
    @classmethod
    def from_json(cls, path: Path) -> 'SceneGenConfig':
        """Load from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def enable_buffer_system(self, buffer_size_km: float, buffer_resolution_m: float = 100.0,
                           background_elevation: float = 0.0, background_size_km: float = 100.0,
                           background_resolution_m: float = 200.0):
        """Enable and configure buffer/background system."""
        self.buffer = BufferConfig(
            enabled=True,
            buffer_size_km=buffer_size_km,
            buffer_resolution_m=buffer_resolution_m,
            background_elevation=background_elevation,
            background_size_km=background_size_km,
            background_resolution_m=background_resolution_m
        )
    
    def disable_buffer_system(self):
        """Disable buffer/background system."""
        self.buffer = None
    
    def set_atmosphere_homogeneous(self, aerosol_dataset: AerosolDataset = AerosolDataset.SIXSV_CONTINENTAL,
                                  optical_thickness: float = 0.1, scale_height: float = 1000.0):
        """Set atmosphere using homogeneous configuration."""
        homogeneous_config = HomogeneousAtmosphereConfig(
            aerosol_dataset=aerosol_dataset,
            optical_thickness=optical_thickness,
            scale_height=scale_height
        )
        self.atmosphere = AtmosphereConfig(
            type=AtmosphereType.HOMOGENEOUS,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            homogeneous=homogeneous_config
        )
    
    def set_atmosphere_molecular(self, molecular_config: MolecularAtmosphereConfig):
        """Set atmosphere using molecular configuration."""
        self.atmosphere = AtmosphereConfig(
            type=AtmosphereType.MOLECULAR,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            molecular=molecular_config
        )
    
    def set_atmosphere_heterogeneous(self, molecular_config: Optional[MolecularAtmosphereConfig] = None,
                                   particle_layers: Optional[list[ParticleLayerConfig]] = None):
        """Set atmosphere using heterogeneous configuration with molecular and particle layers."""
        heterogeneous_config = HeterogeneousAtmosphereConfig(
            molecular=molecular_config,
            particle_layers=particle_layers
        )
        self.atmosphere = AtmosphereConfig(
            type=AtmosphereType.HETEROGENEOUS,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            heterogeneous=heterogeneous_config
        )
    
    
    def validate_configuration(self) -> list[str]:
        """Validate the complete configuration and return any errors."""
        errors = []
        
        # Check data sources
        if not self.data_sources.dem_index_path.exists():
            errors.append(f"DEM index file not found: {self.data_sources.dem_index_path}")
        
        if not self.data_sources.landcover_index_path.exists():
            errors.append(f"Landcover index file not found: {self.data_sources.landcover_index_path}")
        
        # Check buffer configuration
        if self.buffer and self.buffer.enabled:
            if self.buffer.buffer_size_km <= self.location.aoi_size_km:
                errors.append("Buffer size must be larger than AOI size")
        
        return errors
    
    @property
    def scene_output_dir(self) -> Path:
        """Get the specific output directory for this scene."""
        return self.output_dir / self.scene_name
    
    @property
    def meshes_dir(self) -> Path:
        """Get the meshes output directory."""
        return self.scene_output_dir / "meshes"
    
    @property
    def textures_dir(self) -> Path:
        """Get the textures output directory."""
        return self.scene_output_dir / "textures"
    
    @property
    def data_dir(self) -> Path:
        """Get the data output directory."""
        return self.scene_output_dir / "data"
    
    @property
    def has_buffer(self) -> bool:
        """Check if buffer/background system is enabled."""
        return self.buffer is not None and self.buffer.enabled


def create_scene_config(
    scene_name: str,
    center_lat: float,
    center_lon: float,
    aoi_size_km: float,
    output_dir: Path,
    target_resolution_m: float = 30.0,
    description: Optional[str] = None,
    dem_index_path: Optional[Union[Path, str]] = None,
    dem_root_dir: Optional[Union[Path, str]] = None,
    landcover_index_path: Optional[Union[Path, str]] = None,
    landcover_root_dir: Optional[Union[Path, str]] = None,
    material_config_path: Optional[Union[Path, str]] = None,
) -> SceneGenConfig:
    """Create a scene configuration using defaults with optional overrides."""
    
    # Load defaults
    defaults = load_defaults()
    
    # Use overrides if provided, otherwise use defaults
    final_dem_index_path = dem_index_path if dem_index_path is not None else defaults.get("dem_index_path")
    final_dem_root_dir = dem_root_dir if dem_root_dir is not None else defaults.get("dem_root_dir")
    final_landcover_index_path = landcover_index_path if landcover_index_path is not None else defaults.get("landcover_index_path")
    final_landcover_root_dir = landcover_root_dir if landcover_root_dir is not None else defaults.get("landcover_root_dir")
    final_material_config_path = material_config_path if material_config_path is not None else defaults.get("material_config_path")
    
    # Check that all required paths are available
    if not final_dem_index_path:
        raise ValueError("dem_index_path not provided and not found in defaults")
    if not final_dem_root_dir:
        raise ValueError("dem_root_dir not provided and not found in defaults")
    if not final_landcover_index_path:
        raise ValueError("landcover_index_path not provided and not found in defaults")
    if not final_landcover_root_dir:
        raise ValueError("landcover_root_dir not provided and not found in defaults")
    if not final_material_config_path:
        raise ValueError("material_config_path not provided and not found in defaults")
    
    return SceneGenConfig(
        scene_name=scene_name,
        description=description,
        location=SceneLocation(
            center_lat=center_lat,
            center_lon=center_lon,
            aoi_size_km=aoi_size_km
        ),
        data_sources=DataSources(
            dem_index_path=Path(final_dem_index_path) if not str(final_dem_index_path).startswith(("s3://", "gcs://", "azure://")) else final_dem_index_path,
            dem_root_dir=Path(final_dem_root_dir) if not str(final_dem_root_dir).startswith(("s3://", "gcs://", "azure://")) else final_dem_root_dir,
            landcover_index_path=Path(final_landcover_index_path) if not str(final_landcover_index_path).startswith(("s3://", "gcs://", "azure://")) else final_landcover_index_path,
            landcover_root_dir=Path(final_landcover_root_dir) if not str(final_landcover_root_dir).startswith(("s3://", "gcs://", "azure://")) else final_landcover_root_dir,
            material_config_path=Path(final_material_config_path) if not str(final_material_config_path).startswith(("s3://", "gcs://", "azure://")) else final_material_config_path
        ),
        output_dir=output_dir,
        processing=ProcessingOptions(
            target_resolution_m=target_resolution_m
        )
    )


def create_clear_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for clear conditions."""
    homogeneous_config = HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.05,  # Low aerosol
        scale_height=1000.0
    )
    return AtmosphereConfig(
        type=AtmosphereType.HOMOGENEOUS,
        boa=0.0,
        toa=40000.0,
        homogeneous=homogeneous_config
    )


def create_hazy_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for hazy conditions."""
    homogeneous_config = HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.3,  # High aerosol
        scale_height=1000.0
    )
    return AtmosphereConfig(
        type=AtmosphereType.HOMOGENEOUS,
        boa=0.0,
        toa=40000.0,
        homogeneous=homogeneous_config
    )


def create_maritime_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for maritime conditions."""
    homogeneous_config = HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_MARITIME,
        optical_thickness=0.15,
        scale_height=1000.0
    )
    return AtmosphereConfig(
        type=AtmosphereType.HOMOGENEOUS,
        boa=0.0,
        toa=40000.0,
        homogeneous=homogeneous_config
    )


def create_molecular_atmosphere_config(
    identifier: str = "afgl_1986-us_standard",
    altitude_max: float = 120000.0,
    absorption_database: Optional[AbsorptionDatabase] = None,
    co2_concentration: Optional[float] = None
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
        constituent_scaling={"CO2": co2_concentration} if co2_concentration else None
    )
    
    molecular_config = MolecularAtmosphereConfig(
        thermoprops=thermoprops,
        absorption_database=absorption_database
    )
    
    return AtmosphereConfig(
        type=AtmosphereType.MOLECULAR,
        boa=0.0,
        toa=altitude_max,
        molecular=molecular_config
    )


def create_custom_particle_layer(
    aerosol_dataset: AerosolDataset,
    optical_thickness: float,
    altitude_bottom: float = 0.0,
    altitude_top: float = 10000.0,
    distribution_type: str = "exponential",
    scale_height: float = 1000.0
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
        distribution = ExponentialDistribution(rate=1.0/scale_height)
    elif distribution_type == "uniform":
        distribution = UniformDistribution()
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    return ParticleLayerConfig(
        aerosol_dataset=aerosol_dataset,
        optical_thickness=optical_thickness,
        altitude_bottom=altitude_bottom,
        altitude_top=altitude_top,
        distribution=distribution
    )


def create_heterogeneous_atmosphere_config(
    molecular_config: Optional[MolecularAtmosphereConfig] = None,
    particle_layers: Optional[list[ParticleLayerConfig]] = None,
    toa: float = 40000.0
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
        molecular=molecular_config,
        particle_layers=particle_layers
    )
    return AtmosphereConfig(
        type=AtmosphereType.HETEROGENEOUS,
        boa=0.0,
        toa=toa,
        heterogeneous=heterogeneous_config
    )