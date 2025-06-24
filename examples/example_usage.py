#!/usr/bin/env python3
"""Example demonstrating the S2GOS scene generation pipeline."""

from pathlib import Path
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from s2gos_generator.core import SceneGenerationConfig, SceneGenerationPipeline

import mitsuba as mi
mi.set_variant("scalar_rgb")
from PIL import Image
import matplotlib.pyplot as plt

ORDERED_MATERIALS = [
    {"name": "Tree cover", "color_8bit": (40, 75, 30), "roughness": 0.6},
    {"name": "Shrubland", "color_8bit": (185, 170, 130), "roughness": 0.7},
    {"name": "Grassland", "color_8bit": (140, 155, 95), "roughness": 0.7},
    {"name": "Cropland", "color_8bit": (240, 150, 255), "roughness": 0.6},
    {"name": "Built-up", "color_8bit": (150, 150, 150), "roughness": 0.3},
    {"name": "Bare / sparse vegetation", "color_8bit": (220, 140, 90), "roughness": 0.8},
    {"name": "Snow and ice", "color_8bit": (240, 240, 240), "roughness": 0.2},
    {"name": "Permanent water bodies", "color_8bit": (0, 100, 200), "roughness": 0.1},
    {"name": "Herbaceous wetland", "color_8bit": (80, 120, 90), "roughness": 0.4},
    {"name": "Mangroves", "color_8bit": (0, 207, 117), "roughness": 0.4},
    {"name": "Moss and lichen", "color_8bit": (250, 230, 160), "roughness": 0.8},
]



def render_scene(mesh_file, texture_file, output_render_file):
    """Render a 3D scene using the generated mesh and texture files."""
    print(f"Rendering scene...")
    print(f"Mesh: {mesh_file}")
    print(f"Texture: {texture_file}")
    
    try:
        # Load texture data
        selection_texture_data = np.array(Image.open(texture_file))
        selection_texture_data = np.atleast_3d(selection_texture_data)
        
        # Create materials
        materials = {}
        for mat_info in ORDERED_MATERIALS:
            name = mat_info["name"]
            color_float = [c / 255.0 for c in mat_info["color_8bit"]]
            roughness = mat_info["roughness"]
            mat_id = f"mat_{name.lower().replace(' / ', '_').replace(' ', '_')}"
            materials[mat_id] = {
                "type": "roughplastic", "distribution": "ggx", "alpha": roughness,
                "diffuse_reflectance": {"type": "rgb", "value": color_float}
            }
        
        # Create terrain material
        terrain_material_definition = {
            "type": "selectbsdf", "id": "terrain_material",
            "indices": {
                "type": "bitmap", "raw": True, "filter_type": "nearest", 
                "wrap_mode": "clamp", "data": selection_texture_data,
            },
            **{f"bsdf_{i:02d}": {"type": "ref", "id": f"mat_{m['name'].lower().replace(' / ', '_').replace(' ', '_')}"}
               for i, m in enumerate(ORDERED_MATERIALS)},
        }
        
        # Create scene
        scene_dict = {
            "type": "scene", "integrator": {"type": "path"},
            **materials, "terrain_material": terrain_material_definition,
            "terrain": {
                "type": "ply", "filename": str(mesh_file),
                "bsdf": {"type": "ref", "id": "terrain_material"}
            },
            "sky_emitter": {"type": "constant", "radiance": {"type": "rgb", "value": 0.35}},
            "sun_emitter": {
                "type": "directional", "direction": [0.5, 0.5, -1],
                "irradiance": {"type": "rgb", "value": 3.0}
            },
            "sensor": {
                "type": "perspective",
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0, 0, 36612.90342332], target=[0,0,0], up=[0,1,0]
                ),
                "far_clip": 80000,
                "fov_axis": "smaller", "fov": 70,
                "film": {"type": "hdrfilm", "width": 1024, "height": 768},
            },
        }
        
        # Render
        scene = mi.load_dict(scene_dict)
        image = mi.render(scene, spp=128)
        mi.util.write_bitmap(str(output_render_file), image)
        
        print(f"Render saved to: {output_render_file}")
        return True
        
    except Exception as e:
        print(f"Rendering failed: {e}")
        return False


def simple_example():
    """Generate a 100km x 100km scene around Gobabeb, Namibia."""
    print("Simple S2GOS Scene Generation Example")
    print("=" * 40)
    
    config = SceneGenerationConfig(
        center_lat=27.960449,
        center_lon=-15.577022,
        aoi_size_km=100.0,
        
        dem_index_path=Path("/home/gonzalezm/s2gos/s2gos-generator/src/s2gos_generator/data/dem_index.feather"),
        dem_root_dir=Path("/media/DATA/DEM"),
        landcover_index_path=Path("/home/gonzalezm/s2gos/s2gos-generator/src/s2gos_generator/data/landcover_index.feather"), 
        landcover_root_dir=Path("/home/gonzalezm/Data"),
        
        output_dir=Path("./example_output"),
        scene_name="example_scene",
        target_resolution_m=30.0
    )
    
    try:
        # Create and run the pipeline
        pipeline = SceneGenerationPipeline(config)
        assets = pipeline.run_full_pipeline()
        
        print(f"\nSuccess! Scene generated at: {assets.config_file.parent}")
        print(f"Key files:")
        print(f"  3D Mesh: {assets.mesh_file}")
        print(f"  Material Texture:   {assets.selection_texture_file}")
        print(f"  Configuration:      {assets.config_file}")
        
        if assets.mesh_file and assets.selection_texture_file:
            render_output = assets.config_file.parent / "scene_render.png"
            render_success = render_scene(
                assets.mesh_file, 
                assets.selection_texture_file, 
                render_output
            )
            if render_success:
                print(f"  Rendered Image:     {render_output}")
        
        return assets
        
    except FileNotFoundError as e:
        print(f"\nMissing required file: {e}")
        print("\nTo fix this:")
        print("  1. Update the paths in this script to match your system")
        print("  2. Ensure you have DEM and land cover index files")
        print("  3. Ensure your data directories are accessible")
        return None
        
    except Exception as e:
        print(f"\nError: {e}")
        return None


def step_by_step_example():
    """Example showing how to run individual pipeline steps."""
    print("\nStep-by-Step Pipeline Example")
    print("=" * 30)
    
    config = SceneGenerationConfig(
        center_lat=-23.6002,
        center_lon=15.1195,
        aoi_size_km=150.0,
        
        dem_index_path=Path("/home/gonzalezm/dreams-scenes/packages/dreams_assets/src/dreams_assets/data/dem_index.feather"),
        dem_root_dir=Path("/media/DATA/DEM"),
        landcover_index_path=Path("/home/gonzalezm/s2gos-generator/s2gos-generator/landcover_index.feather"),
        landcover_root_dir=Path("/home/gonzalezm/Data"),
        
        output_dir=Path("./step_by_step_output"),
        scene_name="step_example",
        target_resolution_m=30.0
    )
    
    try:
        pipeline = SceneGenerationPipeline(config)
        
        print("Step 1: Processing DEM...")
        pipeline.run_partial_pipeline(['aoi', 'dem'])
        print(f"  DEM saved: {pipeline.assets.dem_file}")
        
        print("Step 2: Processing land cover...")
        pipeline.run_partial_pipeline(['landcover'])
        print(f"  Land cover saved: {pipeline.assets.landcover_file}")
        
        print("Step 3: Generating 3D mesh...")
        pipeline.run_partial_pipeline(['mesh'])
        print(f"  Mesh saved: {pipeline.assets.mesh_file}")
        
        print("Step 4: Generating textures...")
        pipeline.run_partial_pipeline(['textures'])
        print(f"  Texture saved: {pipeline.assets.selection_texture_file}")
        
        print("Step 5: Saving metadata...")
        pipeline.run_partial_pipeline(['metadata'])
        print(f"  Metadata saved: {pipeline.assets.config_file}")
        
        print(f"\nStep-by-step generation complete!")
        return pipeline.assets
        
    except Exception as e:
        print(f"\nStep-by-step example failed: {e}")
        return None


if __name__ == "__main__":
    print("S2GOS Scene Generator - Usage Examples")
    print("=" * 50)
    print()
    
    print("IMPORTANT: Before running, update the file paths in this script")
    print("to match your system configuration.")
    print()
    
    simple_assets = simple_example()
    
    # step_assets = step_by_step_example()
    
    print("\n" + "=" * 50)
    print("Example complete!")
    print()
    print("Next steps:")
    print("  - Review the generated files")
    print("  - Check the generated render image")
    print("  - Customize the pipeline for your specific needs")