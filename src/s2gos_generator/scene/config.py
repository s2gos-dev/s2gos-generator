"""Scene configuration format."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import yaml
import os
from datetime import datetime

from ..materials import Material, MaterialRegistry


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


def create_s2gos_scene(
    scene_name: str, mesh_path: str, texture_path: str,
    center_lat: float, center_lon: float, aoi_size_km: float, resolution_m: float = 30.0,
    buffer_mesh_path: str = None, buffer_texture_path: str = None, 
    buffer_size_km: float = None, output_dir: Path = None, 
    background_elevation: float = None, background_material: str = "water", 
    buffer_dem_file: str = None, **kwargs
) -> SceneConfig:
    """Create standard S2GOS scene configuration."""
    
    metadata = SceneMetadata(
        center_lat=center_lat,
        center_lon=center_lon,
        aoi_size_km=aoi_size_km,
        resolution_m=resolution_m,
        generation_date=datetime.now().isoformat(),
        dem_index_path=kwargs.get("dem_index_path"),
        landcover_index_path=kwargs.get("landcover_index_path")
    )
    
    landcover_ids = {
        "tree_cover": 0, "shrubland": 1, "grassland": 2, "cropland": 3,
        "built_up": 4, "bare_sparse_vegetation": 5, "snow_and_ice": 6,
        "permanent_water_bodies": 7, "herbaceous_wetland": 8, 
        "mangroves": 9, "moss_and_lichen": 10
    }
    
    material_mapping = {
        "tree_cover": "treecover", "shrubland": "shrubland", 
        "grassland": "grassland", "cropland": "cropland",
        "built_up": "concrete", "bare_sparse_vegetation": "baresoil",
        "snow_and_ice": "snow", "permanent_water_bodies": "water",
        "herbaceous_wetland": "wetland", "mangroves": "mangroves",
        "moss_and_lichen": "moss"
    }
    
    target = {
        "mesh": mesh_path,
        "selection_texture": texture_path, 
        "materials": material_mapping
    }
    
    # Add atmosphere configuration
    atmosphere = {
        "boa": kwargs.get("boa", 0.0),
        "toa": kwargs.get("toa", 40000.0), 
        "aerosol_ot": kwargs.get("aerosol_ot", 0.1),
        "aerosol_scale": kwargs.get("aerosol_scale", 1000.0),
        "aerosol_ds": kwargs.get("aerosol_ds", "sixsv-continental")
    }
    
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
        texture_gen.generate_mask_texture(
            mask_size=buffer_resolution,
            target_size=target_resolution, 
            output_path=mask_path,
            generate_background_mask=True
        )
        
        background_mask_path = mask_path.parent / f"background_{mask_path.name}"
        
        buffer_material_mapping = {
            "tree_cover": "treecover", "shrubland": "shrubland", 
            "grassland": "grassland", "cropland": "cropland",
            "built_up": "concrete", "bare_sparse_vegetation": "baresoil",
            "snow_and_ice": "snow", "permanent_water_bodies": "water",
            "herbaceous_wetland": "wetland", "mangroves": "mangroves",
            "moss_and_lichen": "moss"
        }
        
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
                    dem_data = xr.open_dataset(dem_path)
                    if 'elevation' in dem_data.data_vars:
                        bg_elevation = float(dem_data['elevation'].mean().values)
                    else:
                        var_name = list(dem_data.data_vars.keys())[0]
                        bg_elevation = float(dem_data[var_name].mean().values)
            except Exception as e:
                import logging
                logging.warning(f"Could not calculate average elevation from {buffer_dem_file}: {e}")
                bg_elevation = 0.0
        
        background = {
            "material": background_material,
            "elevation": bg_elevation,
            "mask_edge_length": buffer_size_km * 1000.0,
            "mask_texture": _serialize_path(background_mask_path.relative_to(output_dir))
        }
    
    return SceneConfig(
        name=scene_name,
        metadata=metadata,
        landcover_ids=landcover_ids,
        materials=MaterialRegistry.create_s2gos_materials(),
        target=target,
        buffer=buffer,
        background=background,
        atmosphere=atmosphere
    )