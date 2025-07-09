from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import os
from datetime import datetime

from ..materials import Material, load_materials, get_landcover_mapping


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


@dataclass
class SceneMetadata:
    """Scene generation metadata and reproducibility parameters."""
    center_lat: float
    center_lon: float
    aoi_size_km: float
    resolution_m: float
    generation_date: Optional[str] = None
    dem_index_path: Optional[str] = None
    landcover_index_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        result = {
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "aoi_size_km": self.aoi_size_km,
            "resolution_m": self.resolution_m
        }
        
        if self.generation_date:
            result["generation_date"] = self.generation_date
            
        if self.dem_index_path or self.landcover_index_path:
            result["generation"] = {}
            if self.dem_index_path:
                result["generation"]["dem_index_path"] = self.dem_index_path
            if self.landcover_index_path:
                result["generation"]["landcover_index_path"] = self.landcover_index_path
            
        return result


@dataclass
class SceneConfig:
    """Scene configuration for multi-scale earth observation scenes."""
    
    name: str
    metadata: SceneMetadata
    landcover_ids: Dict[str, int]
    material_indices: Dict[int, str]
    materials: Dict[str, Material]
    target: Dict[str, Any]
    buffer: Optional[Dict[str, Any]] = None
    background: Optional[Dict[str, Any]] = None
    atmosphere: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result = {
            "scene": {
                "name": self.name,
                **self.metadata.to_dict()
            },
            "landcover_ids": self.landcover_ids,
            "material_indices": self.material_indices,
            "materials": {k: v.to_dict() for k, v in self.materials.items()},
            "target": self.target
        }
        
        if self.buffer:
            result["buffer"] = self.buffer
        if self.background:
            result["background"] = self.background
        if self.atmosphere:
            result["atmosphere"] = self.atmosphere
            
        return result
    
    def save_yaml(self, path: Path, include_comments: bool = True):
        """Save scene configuration to YAML file."""
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if include_comments:
                f.write("# S2GOS Generated Scene Configuration\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
            
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, 
                     allow_unicode=True, width=120)
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'SceneConfig':
        """Load scene configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Extract scene metadata
        scene_info = data.get("scene", {})
        metadata = SceneMetadata(
            center_lat=scene_info["center_lat"],
            center_lon=scene_info["center_lon"],
            aoi_size_km=scene_info["aoi_size_km"],
            resolution_m=scene_info["resolution_m"],
            generation_date=scene_info.get("generation_date"),
            dem_index_path=scene_info.get("generation", {}).get("dem_index_path"),
            landcover_index_path=scene_info.get("generation", {}).get("landcover_index_path")
        )
        
        # Parse materials
        materials = {}
        for mat_id, mat_data in data["materials"].items():
            mat_type = mat_data.pop("type")
            spectra = {}
            params = {}
            
            for key, value in mat_data.items():
                if isinstance(value, dict) and "path" in value:
                    filename = Path(value["path"]).name
                    spectra[key] = filename
                else:
                    params[key] = value
            
            mat_dict = {"type": mat_type}
            mat_dict.update(params)
            for key, filename in spectra.items():
                mat_dict[key] = {"path": filename, "variable": key}
            
            materials[mat_id] = Material.from_dict(mat_dict, id=mat_id)
        
        return cls(
            name=scene_info["name"],
            metadata=metadata,
            landcover_ids=data["landcover_ids"],
            materials=materials,
            target=data["target"],
            buffer=data.get("buffer"),
            background=data.get("background"),
            atmosphere=data.get("atmosphere"),
            background_elevation=data.get("background_elevation")
        )


def _convert_atmosphere_config_to_dict(atmosphere_config) -> dict:
    """Convert rich AtmosphereConfig to scene description dictionary format.
    
    Args:
        atmosphere_config: AtmosphereConfig object from scene generation configuration
        
    Returns:
        Dictionary format suitable for scene description
    """
    from ..core.config import AtmosphereType
    
    base_dict = {
        "boa": atmosphere_config.boa,
        "toa": atmosphere_config.toa,
        "type": atmosphere_config.type.value
    }
    
    if atmosphere_config.type == AtmosphereType.MOLECULAR:
        # Use molecular atmosphere configuration
        mol_config = atmosphere_config.molecular
        base_dict["molecular_atmosphere"] = {
            "thermoprops_identifier": mol_config.thermoprops.identifier,
            "altitude_min": mol_config.thermoprops.altitude_min,
            "altitude_max": mol_config.thermoprops.altitude_max,
            "altitude_step": mol_config.thermoprops.altitude_step,
            "constituent_scaling": mol_config.thermoprops.constituent_scaling,
            "absorption_database": mol_config.absorption_database.value if mol_config.absorption_database else None,
            "has_absorption": mol_config.has_absorption,
            "has_scattering": mol_config.has_scattering
        }
    
    elif atmosphere_config.type == AtmosphereType.HOMOGENEOUS:
        # Use homogeneous atmosphere configuration
        homogeneous_config = atmosphere_config.homogeneous
        base_dict.update({
            "aerosol_ot": homogeneous_config.optical_thickness,
            "aerosol_scale": homogeneous_config.scale_height,
            "aerosol_ds": homogeneous_config.aerosol_dataset.value,
            "reference_wavelength": homogeneous_config.reference_wavelength,
            "has_absorption": homogeneous_config.has_absorption
        })
    
    elif atmosphere_config.type == AtmosphereType.HETEROGENEOUS:
        # Use heterogeneous atmosphere configuration
        heterogeneous_config = atmosphere_config.heterogeneous
        base_dict.update({
            "has_molecular_atmosphere": heterogeneous_config.molecular is not None,
            "has_particle_layers": heterogeneous_config.particle_layers is not None and len(heterogeneous_config.particle_layers) > 0
        })
        
        # Store molecular atmosphere info
        if heterogeneous_config.molecular:
            mol_config = heterogeneous_config.molecular
            base_dict["molecular_atmosphere"] = {
                "thermoprops_identifier": mol_config.thermoprops.identifier,
                "altitude_min": mol_config.thermoprops.altitude_min,
                "altitude_max": mol_config.thermoprops.altitude_max,
                "altitude_step": mol_config.thermoprops.altitude_step,
                "constituent_scaling": mol_config.thermoprops.constituent_scaling,
                "absorption_database": mol_config.absorption_database.value if mol_config.absorption_database else None,
                "has_absorption": mol_config.has_absorption,
                "has_scattering": mol_config.has_scattering
            }
        
        # Store particle layers info
        if heterogeneous_config.particle_layers:
            base_dict["particle_layers"] = []
            for layer in heterogeneous_config.particle_layers:
                layer_dict = {
                    "aerosol_dataset": layer.aerosol_dataset.value,
                    "optical_thickness": layer.optical_thickness,
                    "altitude_bottom": layer.altitude_bottom,
                    "altitude_top": layer.altitude_top,
                    "distribution_type": layer.distribution.type,
                    "reference_wavelength": layer.reference_wavelength,
                    "has_absorption": layer.has_absorption
                }
                
                # Add distribution-specific parameters
                if layer.distribution.type == "exponential":
                    layer_dict["scale_height"] = layer.distribution.scale_height
                elif layer.distribution.type == "gaussian":
                    layer_dict["center_altitude"] = layer.distribution.center_altitude
                    layer_dict["width"] = layer.distribution.width
                
                base_dict["particle_layers"].append(layer_dict)
    
    else:
        raise ValueError(f"Unknown atmosphere type: {atmosphere_config.type.value}")
    
    return base_dict


def create_s2gos_scene(
    scene_name: str, mesh_path: str, texture_path: str,
    center_lat: float, center_lon: float, aoi_size_km: float, resolution_m: float = 30.0,
    buffer_mesh_path: str = None, buffer_texture_path: str = None, 
    buffer_size_km: float = None, output_dir: Path = None, 
    background_elevation: float = None, buffer_dem_file: str = None, 
    material_config_path: Path = None, material_overrides: Dict[str, Dict[str, Any]] = None,
    landcover_mapping_overrides: Dict[str, str] = None, 
    background_selection_texture: str = None, background_size_km: float = None, **kwargs
) -> SceneConfig:
    """Create standard S2GOS scene configuration.
    
    Args:
        scene_name: Name of the scene
        mesh_path: Path to the terrain mesh file
        texture_path: Path to the selection texture file
        center_lat: Center latitude of the scene
        center_lon: Center longitude of the scene
        aoi_size_km: Size of the area of interest in kilometers
        resolution_m: Target resolution in meters
        buffer_mesh_path: Optional path to buffer mesh file
        buffer_texture_path: Optional path to buffer texture file
        buffer_size_km: Optional buffer size in kilometers
        output_dir: Output directory for the scene
        background_elevation: Background elevation
        buffer_dem_file: Optional buffer DEM file path
        material_config_path: Optional path to custom material configuration JSON
        material_overrides: Optional dictionary of material property overrides
        landcover_mapping_overrides: Optional dictionary of landcover-to-material mapping overrides
        background_selection_texture: Optional path to background selection texture file
        background_size_km: Optional background size in kilometers
        **kwargs: Additional configuration parameters
        
    Returns:
        SceneConfig instance with loaded materials and configuration
    """
    
    metadata = SceneMetadata(
        center_lat=center_lat,
        center_lon=center_lon,
        aoi_size_km=aoi_size_km,
        resolution_m=resolution_m,
        generation_date=datetime.now().isoformat(),
        dem_index_path=kwargs.get("dem_index_path"),
        landcover_index_path=kwargs.get("landcover_index_path")
    )
    
    # Load landcover IDs from configuration
    landcover_ids = {
        "tree_cover": 0, "shrubland": 1, "grassland": 2, "cropland": 3,
        "built_up": 4, "bare_sparse_vegetation": 5, "snow_and_ice": 6,
        "permanent_water_bodies": 7, "herbaceous_wetland": 8, 
        "mangroves": 9, "moss_and_lichen": 10
    }
    
    # Load material mapping from JSON configuration
    material_mapping = get_landcover_mapping(material_config_path)
    
    # Apply any landcover mapping overrides
    if landcover_mapping_overrides:
        material_mapping.update(landcover_mapping_overrides)
    
    # Create material indices mapping for backend use
    material_indices = {}
    for landcover_class_name, texture_index in landcover_ids.items():
        if landcover_class_name in material_mapping:
            material_name = material_mapping[landcover_class_name]
            material_indices[texture_index] = material_name
    
    target = {
        "mesh": mesh_path,
        "selection_texture": texture_path, 
        "materials": material_mapping
    }
    
    # Convert atmosphere configuration to scene description format
    atmosphere_config = kwargs.get("atmosphere_config")
    atmosphere = _convert_atmosphere_config_to_dict(atmosphere_config)
    
    buffer = None
    background = None
    if buffer_mesh_path and buffer_texture_path and buffer_size_km:
        if output_dir is None:
            output_dir = Path(".")
        
        buffer_resolution = int(buffer_size_km * 10)
        target_resolution = int(aoi_size_km * 10)
        
        from ..assets.texture import TextureGenerator
        texture_gen = TextureGenerator()
        mask_path = output_dir / "textures" / f"mask_{scene_name}_{int(buffer_size_km)}km_buffer_{int(aoi_size_km)}km_target.bmp"
        texture_gen.generate_buffer_mask(
            mask_size=buffer_resolution,
            target_size=target_resolution, 
            output_path=mask_path
        )
        
        # Use the same material mapping for buffer as target
        buffer_material_mapping = material_mapping.copy()
        
        buffer = {
            "mesh": buffer_mesh_path,
            "selection_texture": buffer_texture_path,
            "shape_size": buffer_size_km * 1000.0,
            "mask_edge_length": aoi_size_km * 1000.0,
            "mask_texture": _serialize_path(mask_path.relative_to(output_dir)),
            "materials": buffer_material_mapping
        }
        
        bg_elevation = 0.0
        if background_elevation is not None:
            bg_elevation = background_elevation
        elif buffer_dem_file is not None:
            try:
                import xarray as xr
                if output_dir:
                    dem_path = output_dir / buffer_dem_file
                else:
                    dem_path = Path(buffer_dem_file)
                
                if dem_path.exists():
                    dem_data = xr.open_zarr(dem_path)
                    if 'elevation' in dem_data.data_vars:
                        bg_elevation = float(dem_data['elevation'].mean().values)
                    else:
                        var_name = list(dem_data.data_vars.keys())[0]
                        bg_elevation = float(dem_data[var_name].mean().values)
            except Exception as e:
                import logging
                logging.warning(f"Could not calculate average elevation from {buffer_dem_file}: {e}")
                bg_elevation = 0.0
        
        # Use new landcover-based background system
        if background_selection_texture and background_size_km:
            background = {
                "selection_texture": background_selection_texture,
                "elevation": bg_elevation,
                "size_km": background_size_km
            }
        else:
            background = None
    
    # Load materials from JSON configuration
    materials = load_materials(material_config_path)
    
    # Apply any material overrides
    if material_overrides:
        for material_id, overrides in material_overrides.items():
            if material_id in materials:
                # Update existing material with overrides
                current_dict = materials[material_id].to_dict()
                current_dict.update(overrides)
                materials[material_id] = Material.from_dict(current_dict, id=material_id)
            else:
                # Create new material from overrides
                materials[material_id] = Material.from_dict(overrides, id=material_id)
    
    return SceneConfig(
        name=scene_name,
        metadata=metadata,
        landcover_ids=landcover_ids,
        material_indices=material_indices,
        materials=materials,
        target=target,
        buffer=buffer,
        background=background,
        atmosphere=atmosphere
    )