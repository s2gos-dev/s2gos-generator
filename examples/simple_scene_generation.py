#!/usr/bin/env python3
"""Simple example demonstrating basic S2GOS scene generation."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from s2gos_generator.core import SceneGenerationConfig, SceneGenerationPipeline


def simple_scene_generation_example():
    """Generate a simple scene using S2GOS scene generator."""
    print("S2GOS Scene Generation Example")
    print("=" * 40)
    print("Target: 10km x 10km at 30m resolution")
    print()
    
    config = SceneGenerationConfig(
        center_lat=27.978497,
        center_lon=-15.590282,
        aoi_size_km=10.0,
        
        # Update these paths to match your system
        dem_index_path=Path("/home/gonzalezm/s2gos/s2gos/packages/s2gos-generator/src/s2gos_generator/data/dem_index.feather"),
        dem_root_dir=Path("/media/DATA/DEM"),
        landcover_index_path=Path("/home/gonzalezm/s2gos/s2gos/packages/s2gos-generator/src/s2gos_generator/data/landcover_index.feather"), 
        landcover_root_dir=Path("/home/gonzalezm/Data"),
        
        output_dir=Path("./simple_scene_output"),
        scene_name="simple_scene_example",
        target_resolution_m=30.0,
    )
    
    try:
        pipeline = SceneGenerationPipeline(config)
        scene_config = pipeline.run_full_pipeline()
        
        print(f"\nSuccess! Scene generated: {scene_config.name}")
        print(f"Location: {scene_config.metadata.center_lat}, {scene_config.metadata.center_lon}")
        print(f"Scene file: {pipeline.output_dir / f'{scene_config.name}.yml'}")
        print()
        print("Generated Assets:")
        print(f"  Mesh: {pipeline.assets.mesh_file}")
        print(f"  Texture: {pipeline.assets.selection_texture_file}")
        
        print(f"\nScene configuration saved to: {pipeline.output_dir / f'{scene_config.name}.yml'}")
        print("\nNote: This is pure scene generation. For rendering/simulation,")
        print("see the integration examples in the main S2GOS project.")
        
        return scene_config
        
    except FileNotFoundError as e:
        print(f"\nMissing required file: {e}")
        print("\nTo fix this:")
        print("  1. Update the paths in this script to match your system")
        print("  2. Ensure you have DEM and land cover index files")
        print("  3. Ensure your data directories are accessible")
        return None
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("S2GOS Scene Generator - Simple Example")
    print("=" * 40)
    print()
    
    print("IMPORTANT: Before running, ensure:")
    print("  - Update file paths in this script to match your system")
    print("  - Required data files are accessible")
    print()
    
    scene_config = simple_scene_generation_example()
    
    print("\n" + "=" * 40)
    if scene_config:
        print("Scene generation complete!")
        print("\nThis example shows pure scene generation.")
        print("For rendering and simulation, check the integration")
        print("examples in the main S2GOS project.")
    else:
        print("Scene generation failed - check data paths.")