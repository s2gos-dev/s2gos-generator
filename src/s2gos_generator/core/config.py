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
    

class AtmosphereMode(str, Enum):
    """Atmosphere configuration modes."""
    PRESET = "preset"  # Use predefined atmosphere configurations
    DATASET = "dataset"  # Use specific Eradiate datasets
    CUSTOM = "custom"  # Custom molecular/particle layers
    RAW = "raw"  # Raw Eradiate configuration passthrough


class SceneLocation(BaseModel):
    """Geographic location configuration."""
    center_lat: float = Field(..., ge=-90.0, le=90.0, description="Center latitude in degrees")
    center_lon: float = Field(..., ge=-180.0, le=180.0, description="Center longitude in degrees")
    aoi_size_km: float = Field(..., gt=0.0, description="Area of interest size in kilometers")


class DataSources(BaseModel):
    """Data source configuration with validation."""
    dem_index_path: Path = Field(..., description="Path to DEM index file")
    dem_root_dir: Path = Field(..., description="Root directory for DEM data")
    landcover_index_path: Path = Field(..., description="Path to landcover index file")
    landcover_root_dir: Path = Field(..., description="Root directory for landcover data")
    material_config_path: Path = Field(..., description="Path to custom material configuration JSON")
    
    @field_validator('dem_index_path', 'landcover_index_path')
    @classmethod
    def validate_index_files(cls, v):
        """Validate that index files exist."""
        if not v.exists():
            raise ValueError(f"Index file does not exist: {v}")
        return v
    
    @field_validator('dem_root_dir', 'landcover_root_dir')
    @classmethod
    def validate_root_dirs(cls, v):
        """Validate that root directories exist."""
        if not v.exists():
            raise ValueError(f"Root directory does not exist: {v}")
        return v
    
    @field_validator('material_config_path')
    @classmethod
    def validate_material_config(cls, v):
        """Validate that material config file exists if provided."""
        if not v.exists():
            raise ValueError(f"Material config file does not exist: {v}")
        return v


class ProcessingOptions(BaseModel):
    """Processing options for scene generation."""
    target_resolution_m: float = Field(30.0, gt=0.0, description="Target resolution in meters")
    generate_texture_preview: bool = Field(True, description="Generate texture preview images")
    handle_dem_nans: bool = Field(True, description="Handle NaN values in DEM data")
    dem_fillna_value: float = Field(0.0, description="Fill value for DEM NaN values")


class MolecularLayer(BaseModel):
    """Custom molecular atmosphere layer."""
    species: str = Field(..., description="Molecular species (e.g., H2O, CO2, O3)")
    concentration: float = Field(..., ge=0.0, description="Concentration value")
    concentration_unit: str = Field("ppm", description="Concentration unit")
    altitude_range: tuple[float, float] = Field(..., description="(bottom, top) altitude in meters")
    

class ParticleLayer(BaseModel):
    """Custom particle atmosphere layer."""
    aerosol_dataset: AerosolDataset = Field(..., description="Aerosol dataset to use")
    optical_thickness: float = Field(..., ge=0.0, description="Aerosol optical thickness")
    scale_height: float = Field(1000.0, gt=0.0, description="Scale height in meters")
    altitude_range: tuple[float, float] = Field(..., description="(bottom, top) altitude in meters")


class AtmosphereConfig(BaseModel):
    """Comprehensive atmosphere configuration supporting multiple modes."""
    mode: AtmosphereMode = Field(AtmosphereMode.PRESET, description="Atmosphere configuration mode")
    
    boa: float = Field(0.0, ge=0.0, description="Bottom of atmosphere altitude in meters")
    toa: float = Field(40000.0, gt=0.0, description="Top of atmosphere altitude in meters")
    
    aerosol_ot: Optional[float] = Field(0.1, ge=0.0, le=5.0, description="Aerosol optical thickness")
    aerosol_scale: Optional[float] = Field(1000.0, gt=0.0, description="Aerosol scale height in meters")
    aerosol_ds: Optional[AerosolDataset] = Field(AerosolDataset.SIXSV_CONTINENTAL, description="Aerosol dataset")
    
    absorption_db: Optional[AbsorptionDatabase] = Field(None, description="Absorption database")
    aerosol_dataset: Optional[AerosolDataset] = Field(None, description="Specific aerosol dataset")
    
    molecular_layers: Optional[list[MolecularLayer]] = Field(None, description="Custom molecular layers")
    particle_layers: Optional[list[ParticleLayer]] = Field(None, description="Custom particle layers")
    
    raw_config: Optional[dict[str, Any]] = Field(None, description="Raw Eradiate atmosphere configuration")
    
    @model_validator(mode='after')
    def validate_atmosphere_config(self):
        """Validate atmosphere configuration based on mode."""
        if self.toa <= self.boa:
            raise ValueError("Top of atmosphere must be higher than bottom of atmosphere")
        
        if self.mode == AtmosphereMode.PRESET:
            if self.aerosol_ot is None or self.aerosol_ds is None:
                raise ValueError("Preset mode requires aerosol_ot and aerosol_ds")
        elif self.mode == AtmosphereMode.DATASET:
            if self.aerosol_dataset is None and self.absorption_db is None:
                raise ValueError("Dataset mode requires aerosol_dataset or absorption_db")
        elif self.mode == AtmosphereMode.CUSTOM:
            if not self.molecular_layers and not self.particle_layers:
                raise ValueError("Custom mode requires molecular_layers or particle_layers")
        elif self.mode == AtmosphereMode.RAW:
            if self.raw_config is None:
                raise ValueError("Raw mode requires raw_config")
        
        return self


class BufferConfig(BaseModel):
    """Combined buffer and background configuration."""
    enabled: bool = Field(False, description="Enable buffer/background system")
    buffer_size_km: float = Field(..., gt=0.0, description="Buffer size in kilometers")
    buffer_resolution_m: float = Field(100.0, gt=0.0, description="Buffer resolution in meters")
    background_material: BackgroundMaterial = Field(BackgroundMaterial.WATER, description="Background material")
    background_elevation: float = Field(0.0, description="Background elevation in meters")
    
    @model_validator(mode='after')
    def validate_buffer_config(self):
        """Validate buffer configuration when enabled."""
        if not self.enabled:
            return self
        
        if self.buffer_size_km <= 0:
            raise ValueError("Buffer size must be positive when buffer is enabled")
        
        return self




class SceneGenConfig(BaseModel):
    """
    Comprehensive scene configuration using Pydantic.
    
    This replaces the legacy SceneGenerationConfig with a modern,
    validated, and flexible configuration system.
    """
    
    scene_name: str = Field(..., min_length=1, description="Scene name (used for output files)")
    description: Optional[str] = Field(None, description="Scene description")
    
    location: SceneLocation = Field(..., description="Geographic location")
    data_sources: DataSources = Field(..., description="Data source configuration")
    output_dir: Path = Field(..., description="Output directory for generated scene")
    processing: ProcessingOptions = Field(default_factory=ProcessingOptions, description="Processing options")    
    atmosphere: AtmosphereConfig = Field(default_factory=AtmosphereConfig, description="Atmosphere configuration")
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
    
    @property
    def center_lat(self) -> float:
        """Legacy compatibility: center latitude."""
        return self.location.center_lat
    
    @property
    def center_lon(self) -> float:
        """Legacy compatibility: center longitude."""
        return self.location.center_lon
    
    @property
    def aoi_size_km(self) -> float:
        """Legacy compatibility: AOI size."""
        return self.location.aoi_size_km
    
    @property
    def dem_index_path(self) -> Path:
        """Legacy compatibility: DEM index path."""
        return self.data_sources.dem_index_path
    
    @property
    def dem_root_dir(self) -> Path:
        """Legacy compatibility: DEM root directory."""
        return self.data_sources.dem_root_dir
    
    @property
    def landcover_index_path(self) -> Path:
        """Legacy compatibility: landcover index path."""
        return self.data_sources.landcover_index_path
    
    @property
    def landcover_root_dir(self) -> Path:
        """Legacy compatibility: landcover root directory."""
        return self.data_sources.landcover_root_dir
    
    @property
    def target_resolution_m(self) -> float:
        """Legacy compatibility: target resolution."""
        return self.processing.target_resolution_m
    
    @property
    def generate_texture_preview(self) -> bool:
        """Legacy compatibility: generate texture preview."""
        return self.processing.generate_texture_preview
    
    @property
    def handle_dem_nans(self) -> bool:
        """Legacy compatibility: handle DEM NaNs."""
        return self.processing.handle_dem_nans
    
    @property
    def dem_fillna_value(self) -> float:
        """Legacy compatibility: DEM fill value."""
        return self.processing.dem_fillna_value
    
    @property
    def enable_buffer(self) -> bool:
        """Legacy compatibility: enable buffer."""
        return self.buffer is not None and self.buffer.enabled
    
    @property
    def buffer_size_km(self) -> Optional[float]:
        """Legacy compatibility: buffer size."""
        return self.buffer.buffer_size_km if self.buffer and self.buffer.enabled else None
    
    @property
    def buffer_resolution_m(self) -> float:
        """Legacy compatibility: buffer resolution."""
        return self.buffer.buffer_resolution_m if self.buffer else 100.0
    
    @property
    def background_elevation(self) -> Optional[float]:
        """Legacy compatibility: background elevation."""
        return self.buffer.background_elevation if self.buffer and self.buffer.enabled else None
    
    @property
    def background_material(self) -> str:
        """Legacy compatibility: background material."""
        return self.buffer.background_material.value if self.buffer and self.buffer.enabled else "water"
    
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
                           background_material: BackgroundMaterial = BackgroundMaterial.WATER,
                           background_elevation: float = 0.0):
        """Enable and configure buffer/background system."""
        self.buffer = BufferConfig(
            enabled=True,
            buffer_size_km=buffer_size_km,
            buffer_resolution_m=buffer_resolution_m,
            background_material=background_material,
            background_elevation=background_elevation
        )
    
    def disable_buffer_system(self):
        """Disable buffer/background system."""
        self.buffer = None
    
    def set_atmosphere_preset(self, aerosol_ot: float = 0.1, aerosol_scale: float = 1000.0,
                            aerosol_ds: AerosolDataset = AerosolDataset.SIXSV_CONTINENTAL):
        """Set atmosphere using preset configuration."""
        self.atmosphere = AtmosphereConfig(
            mode=AtmosphereMode.PRESET,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            aerosol_ot=aerosol_ot,
            aerosol_scale=aerosol_scale,
            aerosol_ds=aerosol_ds
        )
    
    def set_atmosphere_dataset(self, aerosol_dataset: Optional[AerosolDataset] = None,
                             absorption_db: Optional[AbsorptionDatabase] = None):
        """Set atmosphere using specific datasets."""
        self.atmosphere = AtmosphereConfig(
            mode=AtmosphereMode.DATASET,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            aerosol_dataset=aerosol_dataset,
            absorption_db=absorption_db
        )
    
    def set_atmosphere_custom(self, molecular_layers: Optional[list[MolecularLayer]] = None,
                            particle_layers: Optional[list[ParticleLayer]] = None):
        """Set atmosphere using custom layers."""
        self.atmosphere = AtmosphereConfig(
            mode=AtmosphereMode.CUSTOM,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            molecular_layers=molecular_layers,
            particle_layers=particle_layers
        )
    
    def set_atmosphere_raw(self, raw_config: dict[str, Any]):
        """Set atmosphere using raw Eradiate configuration."""
        self.atmosphere = AtmosphereConfig(
            mode=AtmosphereMode.RAW,
            boa=self.atmosphere.boa,
            toa=self.atmosphere.toa,
            raw_config=raw_config
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
    dem_index_path: Path,
    dem_root_dir: Path,
    landcover_index_path: Path,
    landcover_root_dir: Path,
    material_config_path: Path,
    output_dir: Path,
    target_resolution_m: float = 30.0,
    description: Optional[str] = None
) -> SceneGenConfig:
    """Create a basic scene configuration."""
    
    return SceneGenConfig(
        scene_name=scene_name,
        description=description,
        location=SceneLocation(
            center_lat=center_lat,
            center_lon=center_lon,
            aoi_size_km=aoi_size_km
        ),
        data_sources=DataSources(
            dem_index_path=dem_index_path,
            dem_root_dir=dem_root_dir,
            landcover_index_path=landcover_index_path,
            landcover_root_dir=landcover_root_dir,
            material_config_path=material_config_path
        ),
        output_dir=output_dir,
        processing=ProcessingOptions(
            target_resolution_m=target_resolution_m
        )
    )


def create_clear_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for clear conditions."""
    return AtmosphereConfig(
        mode=AtmosphereMode.PRESET,
        boa=0.0,
        toa=40000.0,
        aerosol_ot=0.05,  # Low aerosol
        aerosol_scale=1000.0,
        aerosol_ds=AerosolDataset.SIXSV_CONTINENTAL
    )


def create_hazy_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for hazy conditions."""
    return AtmosphereConfig(
        mode=AtmosphereMode.PRESET,
        boa=0.0,
        toa=40000.0,
        aerosol_ot=0.3,
        aerosol_scale=1000.0,
        aerosol_ds=AerosolDataset.SIXSV_CONTINENTAL
    )


def create_maritime_atmosphere() -> AtmosphereConfig:
    """Create atmosphere configuration for maritime conditions."""
    return AtmosphereConfig(
        mode=AtmosphereMode.PRESET,
        boa=0.0,
        toa=40000.0,
        aerosol_ot=0.15,
        aerosol_scale=1000.0,
        aerosol_ds=AerosolDataset.SIXSV_MARITIME
    )