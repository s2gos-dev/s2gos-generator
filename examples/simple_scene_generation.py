#!/usr/bin/env python3
"""Simple example demonstrating basic S2GOS scene generation."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from s2gos_generator.core import SceneGenerationPipeline
from s2gos_generator.core.config import (
    SceneGenConfig, create_scene_config, 
    AtmosphereConfig, AtmosphereType, MolecularAtmosphereConfig, HomogeneousAtmosphereConfig, 
    HeterogeneousAtmosphereConfig, ThermophysicalConfig, ParticleLayerConfig,
    AbsorptionDatabase, AerosolDataset, ExponentialDistribution
)


def simple_scene_generation_example():
    """Generate a simple scene using S2GOS scene generator."""
    print("S2GOS Scene Generation Example")
    print("=" * 40)
    print("Target: 10km x 10km at 30m resolution")
    print("Buffer: 60km at 100m resolution")
    print("Background: 200m resolution")
    print()
    
    # Create configuration using the new API - similar to simple_integration_example.py
    # Using Gobabeb location which has good data coverage
    config = create_scene_config(
        scene_name="simple_scene_example",
        center_lat=-23.6002,
        center_lon=15.11956,
        aoi_size_km=10.0,
        output_dir=Path("./simple_scene_output"),
        target_resolution_m=30.0,
        description="Simple scene generation example using S2GOS with buffer/background"
    )
    
    # Enable buffer/background system like in simple_integration_example.py
    config.enable_buffer_system(
        buffer_size_km=60.0,
        buffer_resolution_m=100.0,
        background_elevation=0.0,
        background_resolution_m=200.0
    )
    
    # Configure atmosphere - using heterogeneous atmosphere like simple_integration_example.py
    molecular_config = MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(
            identifier="afgl_1986-us_standard",
        ),
        absorption_database=AbsorptionDatabase.GECKO,
        has_absorption=True,
        has_scattering=True
    )
    
    hazy_layer = ParticleLayerConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.3,  # High aerosol for hazy conditions
        altitude_bottom=0.0,
        altitude_top=1000.0,
        distribution=ExponentialDistribution(rate=5.0),
        reference_wavelength=550.0,
        has_absorption=True
    )
    
    config.set_atmosphere_heterogeneous(
        molecular_config=molecular_config,
        particle_layers=[hazy_layer]
    )
    
    print("Advanced configuration created (with buffer/background and atmosphere)")
    
    # Validate configuration
    errors = config.validate_configuration()
    if errors:
        print(f"Configuration errors: {errors}")
        return None
    else:
        print("Configuration validation passed")
    
    # Display configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Scene: {config.scene_name}")
    print(f"  Location: {config.location.center_lat:.4f}°, {config.location.center_lon:.4f}°")
    print(f"  AOI: {config.location.aoi_size_km} km²")
    print(f"  Resolution: {config.processing.target_resolution_m} m")
    print(f"  Buffer: {config.buffer.buffer_size_km} km at {config.buffer.buffer_resolution_m} m resolution")
    print(f"  Background: at {config.buffer.background_elevation} m elevation, {config.buffer.background_resolution_m} m resolution")
    print(f"  Atmosphere: {config.atmosphere.details.type}")
    print(f"  Output: {config.scene_output_dir}")
    
    try:
        pipeline = SceneGenerationPipeline(config)
        scene_description = pipeline.run_full_pipeline()
        
        print(f"\nSuccess! Scene generated: {scene_description.name}")
        print(f"Location: {config.location.center_lat}, {config.location.center_lon}")
        print(f"Scene file: {config.scene_output_dir / f'{scene_description.name}.yml'}")
        print()
        print("Generated Assets:")
        print(f"  Target mesh: {config.meshes_dir / f'{config.scene_name}_terrain.ply'}")
        print(f"  Target texture: {config.textures_dir / f'{config.scene_name}_{config.processing.target_resolution_m}m_selection.png'}")
        if config.has_buffer:
            print(f"  Buffer mesh: {config.meshes_dir / f'{config.scene_name}_buffer_terrain.ply'}")
            print(f"  Buffer texture: {config.textures_dir / f'{config.scene_name}_buffer_{config.buffer.buffer_resolution_m}m_selection.png'}")
            print(f"  Background texture: {config.textures_dir / f'{config.scene_name}_background_{config.buffer.background_resolution_m}m_selection.png'}")
            print(f"  Buffer mask: {config.textures_dir / f'mask_{config.scene_name}_{config.buffer.buffer_size_km}km_buffer_{config.location.aoi_size_km}km_target.bmp'}")
        
        print(f"\nScene configuration saved to: {config.scene_output_dir / f'{scene_description.name}.yml'}")
        
        return scene_description
        
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
    print("  - Required data files are accessible (DEM and landcover indices)")
    print("  - This example generates a complete scene with buffer/background")
    print("  - Processing may take several minutes due to larger area coverage")
    print()
    
    scene_description = simple_scene_generation_example()
    
    print("\n" + "=" * 40)
    if scene_description:
        print("Scene generation complete!")
    else:
        print("Scene generation failed - check data paths and configuration.")