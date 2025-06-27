#!/usr/bin/env python3
"""Example demonstrating the S2GOS scene generation pipeline with Eradiate rendering using real spectral data."""

from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from s2gos_generator.core import SceneGenerationConfig, SceneGenerationPipeline
from s2gos_generator.assets.atmosphere import create_atmosphere
from s2gos_generator.scene.config import SceneConfig
from s2gos_generator.materials.registry import MaterialRegistry
from s2gos_generator.simulation.config import RenderConfig, create_default_render_config

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg
from eradiate.kernel import TypeIdLookupStrategy, UpdateParameter
from eradiate.scenes.spectra import InterpolatedSpectrum
from eradiate.xarray.interp import dataarray_to_rgb
import xarray as xr
from PIL import Image
import mitsuba as mi

eradiate.set_mode("mono")

def _create_target_surface(mesh_file, texture_file):
    """Create target surface with material selection."""
    texture_image = Image.open(texture_file)
    selection_texture_data = np.array(texture_image)
    selection_texture_data = np.atleast_3d(selection_texture_data)
    
    material_ids = [
        "_mat_treecover", "_mat_shrubland", "_mat_grassland", "_mat_cropland",
        "_mat_concrete", "_mat_baresoil", "_mat_snow", "_mat_water",
        "_mat_wetland", "_mat_mangroves", "_mat_moss"
    ]
    
    return {
        "terrain_material": {
            "type": "selectbsdf",
            "id": "terrain_material",
            "indices": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": selection_texture_data,
            },
            **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
               for i, mat_id in enumerate(material_ids)}
        },
        "terrain": {
            "type": "ply",
            "filename": str(mesh_file),
            "bsdf": {"type": "ref", "id": "terrain_material"},
            "id": "terrain"
        }
    }


def _create_buffer_surface(buffer_mesh_file, buffer_texture_file, mask_file=None):
    """Create buffer surface with material selection and optional masking."""
    buffer_texture_image = Image.open(buffer_texture_file)
    buffer_selection_texture_data = np.array(buffer_texture_image)
    buffer_selection_texture_data = np.atleast_3d(buffer_selection_texture_data)
    
    material_ids = [
        "_mat_treecover", "_mat_shrubland", "_mat_grassland", "_mat_cropland",
        "_mat_concrete", "_mat_baresoil", "_mat_snow", "_mat_water",
        "_mat_wetland", "_mat_mangroves", "_mat_moss"
    ]
    
    result = {
        "buffer_material": {
            "type": "selectbsdf",
            "id": "buffer_material",
            "indices": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": buffer_selection_texture_data,
            },
            **{f"bsdf_{i:02d}": {"type": "ref", "id": mat_id}
               for i, mat_id in enumerate(material_ids)}
        }
    }
    
    buffer_bsdf_id = "buffer_material"
    
    if mask_file and Path(mask_file).exists():
        mask_image = Image.open(mask_file)
        mask_data = np.array(mask_image) / 255.0
        mask_data = np.atleast_3d(mask_data)
        
        result["buffer_mask"] = {
            "type": "mask",
            "opacity": {
                "type": "bitmap",
                "raw": True,
                "filter_type": "nearest",
                "wrap_mode": "clamp",
                "data": mask_data,
            },
            "material": {"type": "ref", "id": "buffer_material"}
        }
        buffer_bsdf_id = "buffer_mask"
    
    result["buffer_terrain"] = {
        "type": "ply",
        "filename": str(buffer_mesh_file),
        "bsdf": {"type": "ref", "id": buffer_bsdf_id},
        "id": "buffer_terrain"
    }
    
    return result


def _create_background_surface(elevation, mask_texture_path, material_id="_mat_water", mask_edge_length=100000.0):
    """Create background surface using dreams-scenes approach."""
    
    shape_size = 1e9
    scale = shape_size / mask_edge_length / 3.0
    offset = -0.002
    
    to_world = mi.ScalarTransform4f.translate(
        [0, 0, elevation + offset]
    ) @ mi.ScalarTransform4f.scale(0.5 * shape_size)
    
    to_uv = mi.ScalarTransform4f.scale(
        [scale, scale, 1]
    ) @ mi.ScalarTransform4f.translate(
        [0.5 * (1.0 / scale - 1.0), 0.5 * (1.0 / scale - 1.0), 0.0]
    )
    
    return {
        "background_surface": {
            "type": "rectangle",
            "to_world": to_world,
            "bsdf": {
                "type": "mask",
                "opacity": {
                    "type": "bitmap",
                    "filename": str(mask_texture_path),
                    "raw": True,
                    "filter_type": "nearest",
                    "wrap_mode": "clamp",
                    "to_uv": to_uv,
                },
                "material": {"type": "ref", "id": material_id},
            },
            "id": "background_surface",
        }
    }


def create_eradiate_scene(scene_config: SceneConfig, render_config: RenderConfig, output_dir: Path):
    """Create an Eradiate scene from SceneConfig and RenderConfig.
    
    Args:
        scene_config: Scene geometry and materials configuration
        render_config: Rendering parameters (atmosphere, illumination, sensors)
        output_dir: Directory containing scene assets
    
    Returns:
        AtmosphereExperiment ready for rendering
    """
    materials = scene_config.materials
    kdict, kpmap = MaterialRegistry.create_material_kdict_kpmap(materials)
    
    # Create target surface
    target_mesh_path = output_dir / scene_config.target["mesh"]
    target_texture_path = output_dir / scene_config.target["selection_texture"]
    kdict.update(_create_target_surface(target_mesh_path, target_texture_path))
    
    # Create buffer surface if configured
    if scene_config.buffer:
        buffer_mesh_path = output_dir / scene_config.buffer["mesh"]
        buffer_texture_path = output_dir / scene_config.buffer["selection_texture"]
        mask_path = output_dir / scene_config.buffer["mask_texture"] if "mask_texture" in scene_config.buffer else None
        kdict.update(_create_buffer_surface(buffer_mesh_path, buffer_texture_path, mask_path))
    
    # Create background surface if configured
    if scene_config.background:
        bg_mask_path = output_dir / scene_config.background["mask_texture"]
        kdict.update(_create_background_surface(
            elevation=scene_config.background["elevation"],
            mask_texture_path=bg_mask_path,
            material_id="_mat_water",
            mask_edge_length=scene_config.background["mask_edge_length"]
        ))
        print(f"Added background surface at elevation {scene_config.background['elevation']:.1f}m")
    
    # Create atmosphere from scene config (tied to geographic location)
    atmosphere = create_atmosphere(
        boa=scene_config.atmosphere.get("boa", 0.0),
        toa=scene_config.atmosphere.get("toa", 40.0e3),
        aerosol_ot=scene_config.atmosphere.get("aerosol_ot", 0.1),
        aerosol_scale=scene_config.atmosphere.get("aerosol_scale", 1e3),
        aerosol_ds=scene_config.atmosphere.get("aerosol_ds", "sixsv-continental")
    )
    
    # Create illumination from render config
    illumination = {
        "type": "directional",
        "zenith": render_config.illumination.zenith * ureg.deg,
        "azimuth": render_config.illumination.azimuth * ureg.deg,
        "irradiance": {
            "type": render_config.illumination.irradiance_type,
            "dataset": render_config.illumination.irradiance_dataset
        }
    }
    
    # Create measures from render config sensors
    measures = []
    for sensor in render_config.sensors:
        measures.append({
            "type": sensor.type,
            "id": sensor.id,
            "origin": sensor.origin,
            "target": sensor.target,
            "up": [0, 1, 0],
            "fov": sensor.fov,
            "film_resolution": tuple(sensor.resolution),
            "srf": {
                "type": "delta",
                "wavelengths": [440, 550, 660] * ureg.nm
            },
            "spp": sensor.spp
        })
    
    return AtmosphereExperiment(
        geometry={"type": "plane_parallel", "toa_altitude": 40.0 * ureg.km},
        atmosphere=atmosphere,
        surface=None,
        illumination=illumination,
        measures=measures,
        kdict=kdict,
        kpmap=kpmap
    )


def render_eradiate_scene(scene_config: SceneConfig, render_config: RenderConfig, scene_dir: Path, output_dir: Path = None):
    """Render scene using Eradiate for physically-based results.
    
    Args:
        scene_config: Scene geometry and materials configuration
        render_config: Rendering parameters (atmosphere, illumination, sensors)
        scene_dir: Directory containing scene assets (meshes, textures, etc.)
        output_dir: Directory for render outputs (defaults to scene_dir/eradiate_renders)
    
    Returns:
        bool: True if rendering succeeded, False otherwise
    """
    if output_dir is None:
        output_dir = scene_dir / "eradiate_renders"
        
    print(f"Rendering scene '{scene_config.name}' with Eradiate...")
    print(f"Scene location: {scene_config.metadata.center_lat}, {scene_config.metadata.center_lon}")
    print(f"Target: {scene_config.target['mesh']} + {scene_config.target['selection_texture']}")
    if scene_config.buffer:
        print(f"Buffer: {scene_config.buffer['mesh']} + {scene_config.buffer['selection_texture']}")
    if scene_config.background:
        print(f"Background: {scene_config.background['material']} at {scene_config.background['elevation']:.1f}m")
    
    try:
        # Create the scene experiment
        exp = create_eradiate_scene(scene_config, render_config, scene_dir)
        
  
        print("Running Eradiate simulation (this may take several minutes)...")
        eradiate.run(exp)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        raw_output = output_dir / "eradiate_results.nc"
        results_ds = exp.results["perspective_view"]
        results_ds.to_netcdf(raw_output)
        
        img = dataarray_to_rgb(
            results_ds["radiance"],
            channels=[("w", 660), ("w", 550), ("w", 440)],
            normalize=False,
        ) * 1.8
        
        img = np.clip(img, 0, 1)
        
        rgb_output = output_dir / "eradiate_rgb.png"
        plt_img = (img * 255).astype(np.uint8)
        rgb_image = Image.fromarray(plt_img)
        rgb_image.save(rgb_output)
        
        print(f"Eradiate simulation complete!")
        print(f"Raw results: {raw_output}")
        print(f"RGB visualization: {rgb_output}")
        return True
        
    except Exception as e:
        print(f"Eradiate rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def simple_eradiate_example():
    """Generate a multi-scale scene with buffer areas using Eradiate rendering."""
    print("S2GOS Scene Generation with Eradiate Example (with Buffer Areas)")
    print("=" * 60)
    print("Target: 10km x 10km at 30m resolution")
    print("Buffer: 100km x 100km total at 100m resolution")
    print()
    
    config = SceneGenerationConfig(
        center_lat=27.978497,
        center_lon=-15.590282,
        aoi_size_km=10.0,
        
        dem_index_path=Path("/home/gonzalezm/s2gos/s2gos-generator/src/s2gos_generator/data/dem_index.feather"),
        dem_root_dir=Path("/media/DATA/DEM"),
        landcover_index_path=Path("/home/gonzalezm/s2gos/s2gos-generator/src/s2gos_generator/data/landcover_index.feather"), 
        landcover_root_dir=Path("/home/gonzalezm/Data"),
        
        output_dir=Path("./eradiate_example_output_la_palma"),
        scene_name="eradiate_example_scene_la_palma",
        target_resolution_m=30.0,
        
        enable_buffer=True,
        buffer_size_km=100.0,
        buffer_resolution_m=100.0
    )
    
    try:
        pipeline = SceneGenerationPipeline(config)
        scene_config = pipeline.run_full_pipeline()
        
        print(f"\nSuccess! Multi-scale scene generated: {scene_config.name}")
        print(f"Location: {scene_config.metadata.center_lat}, {scene_config.metadata.center_lon}")
        print(f"Scene file: {pipeline.output_dir / f'{scene_config.name}.yml'}")
        print()
        print("Generated Assets:")
        print("Target Area (30m resolution):")
        print(f"  Mesh: {pipeline.assets.mesh_file}")
        print(f"  Texture: {pipeline.assets.selection_texture_file}")
        
        if pipeline.assets.buffer_mesh_file:
            print("Buffer Area (100m resolution):")
            print(f"  Mesh: {pipeline.assets.buffer_mesh_file}")
            print(f"  Texture: {pipeline.assets.buffer_selection_texture_file}")
        
        print()
        print("Scene Configuration:")
        print(f"  Target: {scene_config.target['mesh']} + {scene_config.target['selection_texture']}")
        if scene_config.buffer:
            print(f"  Buffer: {scene_config.buffer['mesh']} + {scene_config.buffer['selection_texture']}")
            print(f"  Buffer size: {scene_config.buffer['shape_size']/1000:.1f}km x {scene_config.buffer['shape_size']/1000:.1f}km")
            print(f"  Target mask: {scene_config.buffer['mask_edge_length']/1000:.1f}km x {scene_config.buffer['mask_edge_length']/1000:.1f}km")
        
        # Render with Eradiate
        if pipeline.assets.mesh_file and pipeline.assets.selection_texture_file:
            eradiate_output_dir = pipeline.output_dir / "eradiate_renders"
            
            # Get buffer assets and mask file if available
            buffer_mesh = pipeline.assets.buffer_mesh_file
            buffer_texture = pipeline.assets.buffer_selection_texture_file  
            mask_file = None
            if buffer_mesh and buffer_texture:
                # Find the generated mask file
                mask_file = Path("examples/textures/mask_eradiate_example_scene_la_palma_100km_buffer_10km_target.bmp")
                if not mask_file.exists():
                    mask_file = None
                    
            # Get scene config file path
            scene_yaml_path = pipeline.output_dir / f'{scene_config.name}.yml'
            
            # Create render configuration with the original high camera altitude
            render_config = create_default_render_config(
                name="default_perspective",
                sensor_height=98612.90342332,  # Original high altitude to match previous renders
                spp=32
            )
            
            # Save render config as YAML for easy reuse/modification
            render_config_path = pipeline.output_dir / "render_config.yml"
            render_config.save_yaml(render_config_path)
            print(f"  Render config saved: {render_config_path}")
            
            render_success = render_eradiate_scene(
                scene_config,
                render_config,  # Could also use RenderConfig.from_yaml(yaml_path) here
                pipeline.output_dir,
                eradiate_output_dir
            )
            if render_success:
                print(f"  Eradiate Results: {eradiate_output_dir}")
                
        # Cloud deployment ready
        scene_yaml_path = pipeline.output_dir / f'{scene_config.name}.yml'
        
        return scene_config
        
    except FileNotFoundError as e:
        print(f"\nMissing required file: {e}")
        print("\nTo fix this:")
        print("  1. Update the paths in this script to match your system")
        print("  2. Ensure you have DEM and land cover index files")
        print("  3. Ensure your data directories are accessible")
        print("  4. Make sure Eradiate is properly installed and configured")
        return None
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("S2GOS Scene Generator with Eradiate - Usage Example")
    print("=" * 60)
    print()
    
    print("IMPORTANT: Before running, ensure:")
    print("  - Update file paths in this script to match your system")
    print("  - Eradiate is properly installed and configured")
    print("  - Required data files are accessible")
    print("  - Spectral data files are in src/s2gos_generator/data/spectra/")
    print()
    
    simple_assets = simple_eradiate_example()
    
    print("\n" + "=" * 60)
    print("Example complete!")