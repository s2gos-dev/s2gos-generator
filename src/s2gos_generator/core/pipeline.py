"""Scene generation pipeline orchestrator."""

import logging
from pathlib import Path
from typing import List, Tuple
import json

from .config import SceneGenerationConfig
from .assets import SceneAssets
from ..assets.dem import DEMProcessor, create_aoi_polygon
from ..assets.landcover import LandCoverProcessor
from ..assets.mesh import MeshGenerator
from ..assets.texture import TextureGenerator
from ..scene import SceneDescription, create_default_materials

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SceneGenerationPipeline:
    """Main pipeline orchestrator for generating 3D scenes from earth observation data."""

    def __init__(self, config: SceneGenerationConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.assets = SceneAssets()
        
        self.dem_processor = DEMProcessor(
            index_path=config.dem_index_path,
            dem_root_dir=config.dem_root_dir
        )
        
        self.landcover_processor = LandCoverProcessor(
            index_path=config.landcover_index_path,
            landcover_root_dir=config.landcover_root_dir
        )
        
        self.mesh_generator = MeshGenerator()
        self.texture_generator = TextureGenerator()
        
        self._setup_output_directories()
        
        logging.info(f"Pipeline initialized for scene '{config.scene_name}'")

    def _setup_output_directories(self) -> None:
        """Create the output directory structure."""
        self.output_dir = self.config.output_dir / self.config.scene_name
        self.data_dir = self.output_dir / "data"
        self.meshes_dir = self.output_dir / "meshes"
        self.textures_dir = self.output_dir / "textures"
        
        for directory in [self.output_dir, self.data_dir, self.meshes_dir, self.textures_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def generate_aoi(self) -> None:
        """Generate the Area of Interest polygon."""
        self.aoi_polygon = create_aoi_polygon(
            center_lat=self.config.center_lat,
            center_lon=self.config.center_lon,
            side_length_km=self.config.aoi_size_km
        )
        logging.info(f"Created AOI polygon: {self.config.aoi_size_km}km x {self.config.aoi_size_km}km")

    def process_dem(self) -> Path:
        """Process DEM data for the AOI."""
        logging.info("=== Processing DEM Data ===")
        
        dem_filename = f"dem_{self.config.scene_name}_{self.config.target_resolution_m}m.nc"
        dem_output_path = self.data_dir / dem_filename
        
        self.dem_processor.generate_dem(
            aoi_polygon=self.aoi_polygon,
            output_path=dem_output_path,
            fillna_value=self.config.dem_fillna_value,
            target_resolution_m=self.config.target_resolution_m,
            center_lat=self.config.center_lat,
            center_lon=self.config.center_lon,
            aoi_size_km=self.config.aoi_size_km
        )
        
        self.assets.dem_file = dem_output_path
        logging.info(f"DEM processing complete: {dem_output_path}")
        return dem_output_path

    def process_landcover(self) -> Path:
        """Process land cover data for the AOI."""
        logging.info("=== Processing Land Cover Data ===")
        
        landcover_filename = f"landcover_{self.config.scene_name}_{self.config.target_resolution_m}m.nc"
        landcover_output_path = self.data_dir / landcover_filename
        
        self.landcover_processor.generate_landcover(
            aoi_polygon=self.aoi_polygon,
            output_path=landcover_output_path,
            target_resolution_m=self.config.target_resolution_m,
            center_lat=self.config.center_lat,
            center_lon=self.config.center_lon,
            aoi_size_km=self.config.aoi_size_km
        )
        
        self.assets.landcover_file = landcover_output_path
        logging.info(f"Land cover processing complete: {landcover_output_path}")
        return landcover_output_path

    def generate_mesh(self, dem_file_path: Path) -> Path:
        """Generate 3D mesh from DEM data."""
        logging.info("=== Generating 3D Mesh ===")
        
        mesh_path = self.meshes_dir / f"{self.config.scene_name}_terrain.ply"
        mesh = self.mesh_generator.generate_mesh_from_dem_file(
            dem_file_path=dem_file_path,
            output_path=mesh_path,
            add_uvs=True,
            handle_nans=self.config.handle_dem_nans
        )
        self.assets.mesh_file = mesh_path
        
        mesh_info = self.mesh_generator.get_mesh_info(mesh)
        logging.info(f"Generated mesh: {mesh_info['vertices']} vertices, {mesh_info['faces']} faces")
        
        return mesh_path

    def generate_textures(self, landcover_file_path: Path) -> Tuple[Path, Path]:
        """Generate texture maps from land cover data."""
        logging.info("=== Generating Textures ===")
        
        selection_texture_path, preview_texture_path = self.texture_generator.generate_textures_from_file(
            landcover_file_path=landcover_file_path,
            output_dir=self.textures_dir,
            base_name=f"{self.config.scene_name}_{self.config.target_resolution_m}m",
            create_preview=self.config.generate_texture_preview
        )
        
        self.assets.selection_texture_file = selection_texture_path
        if preview_texture_path:
            self.assets.preview_texture_file = preview_texture_path
        
        # Log texture analysis
        import xarray as xr
        landcover_data = xr.open_dataarray(landcover_file_path)
        if isinstance(landcover_data, xr.Dataset):
            landcover_data = landcover_data[list(landcover_data.data_vars.keys())[0]]
        
        analysis = self.texture_generator.analyze_landcover_classes(landcover_data)
        logging.info(f"Texture analysis: {analysis['unique_classes']} land cover classes found")
        
        return selection_texture_path, preview_texture_path

    def save_metadata(self) -> Path:
        """Save scene configuration and asset metadata."""
        logging.info("=== Saving Metadata ===")
        
        metadata = {
            'config': self.config.to_dict(),
            'assets': self.assets.to_dict(),
            'aoi_bounds': list(self.aoi_polygon.bounds) if hasattr(self, 'aoi_polygon') else None
        }
        
        metadata_path = self.output_dir / f"{self.config.scene_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.assets.config_file = metadata_path
        logging.info(f"Metadata saved: {metadata_path}")
        return metadata_path

    def generate_scene_description(self) -> Path:
        """Generate scene description file."""
        logging.info("=== Generating Scene Description ===")
        
        scene_desc = SceneDescription.from_config(self.config)
        
        # Add default materials
        default_materials = create_default_materials()
        for name, material in default_materials.items():
            scene_desc.materials[name] = material
        
        # Add geometry if available
        if self.assets.mesh_file and self.assets.selection_texture_file:
            scene_desc.add_geometry(
                mesh_path=self.assets.mesh_file,
                texture_path=self.assets.selection_texture_file
            )
        
        # Save scene description
        scene_desc_path = self.output_dir / f"{self.config.scene_name}_scene.yml"
        scene_desc.save_yaml(scene_desc_path)
        
        self.assets.scene_description_file = scene_desc_path
        logging.info(f"Scene description saved: {scene_desc_path}")
        return scene_desc_path

    def run_full_pipeline(self) -> SceneAssets:
        """Execute the complete scene generation pipeline."""
        logging.info(f"Starting full pipeline for scene '{self.config.scene_name}'")
        
        try:
            # Step 1: Generate AOI
            self.generate_aoi()
            
            # Step 2: Process DEM data
            dem_file = self.process_dem()
            
            # Step 3: Process land cover data
            landcover_file = self.process_landcover()
            
            # Step 4: Generate 3D mesh
            mesh_file = self.generate_mesh(dem_file)
            
            # Step 5: Generate textures
            texture_selection, texture_preview = self.generate_textures(landcover_file)
            
            # Step 6: Save metadata
            metadata_file = self.save_metadata()
            
            # Step 7: Generate scene description
            scene_desc_file = self.generate_scene_description()
            
            logging.info("=== Pipeline Complete ===")
            logging.info(f"Scene assets saved to: {self.output_dir}")
            
            return self.assets
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise

    def run_partial_pipeline(self, steps: List[str]) -> SceneAssets:
        """Execute only specific steps of the pipeline."""
        logging.info(f"Running partial pipeline with steps: {steps}")
        
        if 'aoi' in steps:
            self.generate_aoi()
        
        if 'dem' in steps:
            if not hasattr(self, 'aoi_polygon'):
                self.generate_aoi()
            self.process_dem()
        
        if 'landcover' in steps:
            if not hasattr(self, 'aoi_polygon'):
                self.generate_aoi()
            self.process_landcover()
        
        if 'mesh' in steps:
            if not self.assets.dem_file:
                raise ValueError("DEM file required for mesh generation. Run 'dem' step first.")
            self.generate_mesh(self.assets.dem_file)
        
        if 'textures' in steps:
            if not self.assets.landcover_file:
                raise ValueError("Land cover file required for texture generation. Run 'landcover' step first.")
            self.generate_textures(self.assets.landcover_file)
        
        if 'metadata' in steps:
            self.save_metadata()
        
        if 'scene_description' in steps:
            self.generate_scene_description()
        
        return self.assets