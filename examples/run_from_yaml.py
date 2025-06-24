#!/usr/bin/env python3
"""Example demonstrating how to load and run an eradiate scene from YAML configuration."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

import eradiate
from eradiate.xarray.interp import dataarray_to_rgb
import numpy as np
from PIL import Image

from s2gos_generator.simulation import SceneLoader

# Set Eradiate mode
eradiate.set_mode("mono")


def run_scene_from_yaml(yaml_path: Path, output_dir: Path):
    """
    Load and run an eradiate scene from YAML configuration.
    
    This demonstrates how the cloud deployment would work:
    1. Upload YAML config and data files to cloud
    2. Load scene configuration from YAML
    3. Run simulation
    4. Save results
    """
    print(f"Loading scene from: {yaml_path}")
    
    # Initialize loader with data directory
    base_data_dir = Path(__file__).parent.parent / "src/s2gos_generator/data"
    loader = SceneLoader(base_data_dir)
    
    try:
        # Load scene from YAML
        exp = loader.load_eradiate_experiment(yaml_path)
        print("✓ Scene loaded successfully from YAML")
        
        # Run simulation
        print("Running Eradiate simulation...")
        eradiate.run(exp)
        print("✓ Simulation complete")
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        raw_output = output_dir / "results_from_yaml.nc"
        results_ds = exp.results["perspective_view"]
        results_ds.to_netcdf(raw_output)
        
        # Create RGB visualization
        img = dataarray_to_rgb(
            results_ds["radiance"],
            channels=[("w", 660), ("w", 550), ("w", 440)],
            normalize=False,
        ) * 1.8
        
        img = np.clip(img, 0, 1)
        
        # Save RGB image
        rgb_output = output_dir / "rgb_from_yaml.png"
        plt_img = (img * 255).astype(np.uint8)
        rgb_image = Image.fromarray(plt_img)
        rgb_image.save(rgb_output)
        
        print(f"✓ Results saved to: {output_dir}")
        print(f"  Raw data: {raw_output}")
        print(f"  RGB image: {rgb_output}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading/running scene: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("S2GOS Eradiate Scene Loader - YAML Example")
    print("=" * 50)
    
    # Check for YAML file argument
    if len(sys.argv) > 1:
        yaml_path = Path(sys.argv[1])
        if not yaml_path.exists():
            print(f"Error: YAML file not found: {yaml_path}")
            sys.exit(1)
    else:
        # Look for example YAML file
        example_yaml = Path("./eradiate_example_output_la_palma/eradiate_example_scene_la_palma/eradiate_scene_config.yml")
        if example_yaml.exists():
            yaml_path = example_yaml
            print(f"Using example YAML: {yaml_path}")
        else:
            print("Usage: python run_from_yaml.py <path_to_scene_config.yml>")
            print("Or run example_usage_eradiate.py first to generate a YAML file")
            sys.exit(1)
    
    # Run simulation
    output_dir = yaml_path.parent / "yaml_simulation_results"
    success = run_scene_from_yaml(yaml_path, output_dir)
    
    if success:
        print("\n" + "=" * 50)
        print("✓ Scene successfully loaded and simulated from YAML!")
        print("\nThis demonstrates how cloud deployment would work:")
        print("1. Upload YAML config + data files to cloud storage")
        print("2. Cloud worker loads scene from YAML")
        print("3. Run simulation and save results")
        print("4. Download results")
    else:
        print("\n✗ Failed to load/run scene from YAML")
        sys.exit(1)