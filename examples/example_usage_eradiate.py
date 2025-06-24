#!/usr/bin/env python3
"""Example demonstrating the S2GOS scene generation pipeline with Eradiate rendering using real spectral data."""

from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from s2gos_generator.core import SceneGenerationConfig, SceneGenerationPipeline

import eradiate
from eradiate.experiments import AtmosphereExperiment
from eradiate.units import unit_registry as ureg
from eradiate.kernel import TypeIdLookupStrategy, UpdateParameter
from eradiate.scenes.spectra import InterpolatedSpectrum
from eradiate.xarray.interp import dataarray_to_rgb
import xarray as xr
from PIL import Image
import mitsuba as mi

# Set Eradiate mode for realistic radiative transfer
eradiate.set_mode("mono")

# Material definitions using proper spectral data
# Following the pattern from dreams-scenes target_visualization_eradiate.ipynb
REALISTIC_MATERIALS = {
    "mat_tree_cover": {"type": "bilambertian"},  # Transmissive for foliage
    "mat_shrubland": {"type": "bilambertian"},   # Transmissive for foliage
    "mat_grassland": {"type": "rpv", "rho_c": 0.5},
    "mat_cropland": {"type": "bilambertian"},    # Transmissive for crops
    "mat_built_up": {"type": "diffuse"},
    "mat_bare_sparse_vegetation": {"type": "rpv", "rho_c": 0.5},
    "mat_snow_and_ice": {"type": "diffuse"},
    "mat_permanent_water_bodies": {"type": "ocean_legacy", "wavelength": 550.0},
    "mat_herbaceous_wetland": {"type": "rpv", "rho_c": 0.5},
    "mat_mangroves": {"type": "bilambertian"},   # Transmissive for foliage
    "mat_moss_and_lichen": {"type": "diffuse"}
}


def load_spectral_data():
    """Load spectral data from NetCDF files following dreams-scenes pattern."""
    spectra_dir = Path(__file__).parent.parent / "src/s2gos_generator/data/spectra"
    
    # Define spectral files and variables following dreams-scenes patterns
    spectra_files = {
        # Tree cover
        "treecover_foliage_high_reflectance": ("spectrum_treecover_foliage_high.nc", "reflectance"),
        "treecover_foliage_high_transmittance": ("spectrum_treecover_foliage_high.nc", "transmittance"),
        # Shrubland foliage for green shrubs
        "shrubland_foliage_reflectance": ("spectrum_shrubland_foliage.nc", "reflectance"),
        "shrubland_foliage_transmittance": ("spectrum_shrubland_foliage.nc", "transmittance"),
        # Grassland RPV parameters
        "grassland_rho_0": ("spectrum_grassland_rpv_dry.nc", "rho_0"),
        "grassland_k": ("spectrum_grassland_rpv_dry.nc", "k"),
        "grassland_Theta": ("spectrum_grassland_rpv_dry.nc", "Theta"),
        "grassland_rho_c": ("spectrum_grassland_rpv_dry.nc", "rho_c"),
        # Cropland foliage (more realistic than just soil)
        "cropland_foliage_uniform_reflectance": ("spectrum_cropland_foliage_uniform.nc", "reflectance"),
        "cropland_foliage_uniform_transmittance": ("spectrum_cropland_foliage_uniform.nc", "transmittance"),
        # Built up areas
        "concrete_reflectance": ("spectrum_concrete.nc", "reflectance"),
        # Bare soil RPV - use low file which has RPV parameters
        "baresoil_rho_0": ("spectrum_baresoil_low.nc", "rho_0"),
        "baresoil_k": ("spectrum_baresoil_low.nc", "k"),
        "baresoil_Theta": ("spectrum_baresoil_low.nc", "Theta"),
        "baresoil_rho_c": ("spectrum_baresoil_low.nc", "rho_c"),
        # Snow and moss
        "snow_reflectance": ("spectrum_snow.nc", "reflectance"),
        "moss_reflectance": ("spectrum_moss.nc", "reflectance"),
    }
    
    # Load spectra using dreams-scenes pattern
    loaded_spectra = {}
    for k, (filename, variable) in spectra_files.items():
        file_path = spectra_dir / filename
        if file_path.exists():
            try:
                # Load following exact dreams-scenes pattern
                loaded_spectra[k] = InterpolatedSpectrum.from_dataarray(
                    dataarray=xr.load_dataset(file_path)[variable]
                )
                print(f"Loaded spectral data: {k} from {filename}[{variable}]")
            except Exception as e:
                print(f"Warning: Could not load {filename}[{variable}]: {e}")
        else:
            print(f"Warning: Spectral file not found: {file_path}")
    
    return loaded_spectra


def create_spectral_parameter_map(spectra):
    """Create parameter map for spectral materials."""
    dispatch_spectra = {}
    
    for spectrum_id, spectrum in spectra.items():
        def _f(ctx, spectrum_id=spectrum_id):
            return spectra[spectrum_id].eval(ctx.si)
        dispatch_spectra[spectrum_id] = _f
    
    # Create parameter map
    kpmap = {}
    
    # Add spectral materials with proper foliage spectra
    for mat_id, param, scene_parameter in [
        # Bilambertian materials - vegetation with transmittance
        ("tree_cover", "reflectance.value", dispatch_spectra.get("treecover_foliage_high_reflectance")),
        ("tree_cover", "transmittance.value", dispatch_spectra.get("treecover_foliage_high_transmittance")),
        ("shrubland", "reflectance.value", dispatch_spectra.get("shrubland_foliage_reflectance")),
        ("shrubland", "transmittance.value", dispatch_spectra.get("shrubland_foliage_transmittance")),
        ("cropland", "reflectance.value", dispatch_spectra.get("cropland_foliage_uniform_reflectance")),
        ("cropland", "transmittance.value", dispatch_spectra.get("cropland_foliage_uniform_transmittance")),
        ("mangroves", "reflectance.value", dispatch_spectra.get("treecover_foliage_high_reflectance")),
        ("mangroves", "transmittance.value", dispatch_spectra.get("treecover_foliage_high_transmittance")),
        # Diffuse materials
        ("built_up", "reflectance.value", dispatch_spectra.get("concrete_reflectance")),
        ("snow_and_ice", "reflectance.value", dispatch_spectra.get("snow_reflectance")),
        ("moss_and_lichen", "reflectance.value", dispatch_spectra.get("moss_reflectance")),
        # RPV materials for grassland and bare soil
        ("grassland", "rho_0.value", dispatch_spectra.get("grassland_rho_0")),
        ("grassland", "k.value", dispatch_spectra.get("grassland_k")),
        ("grassland", "g.value", dispatch_spectra.get("grassland_Theta")),
        ("grassland", "rho_c.value", dispatch_spectra.get("grassland_rho_c")),
        ("bare_sparse_vegetation", "rho_0.value", dispatch_spectra.get("baresoil_rho_0")),
        ("bare_sparse_vegetation", "k.value", dispatch_spectra.get("baresoil_k")),
        ("bare_sparse_vegetation", "g.value", dispatch_spectra.get("baresoil_Theta")),
        ("bare_sparse_vegetation", "rho_c.value", dispatch_spectra.get("baresoil_rho_c")),
        # Wetland uses grassland parameters
        ("herbaceous_wetland", "rho_0.value", dispatch_spectra.get("grassland_rho_0")),
        ("herbaceous_wetland", "k.value", dispatch_spectra.get("grassland_k")),
        ("herbaceous_wetland", "g.value", dispatch_spectra.get("grassland_Theta")),
        ("herbaceous_wetland", "rho_c.value", dispatch_spectra.get("grassland_rho_c")),
        # Ocean legacy for water bodies
        ("permanent_water_bodies", "wavelength", lambda ctx: ctx.si.w.m_as("nm")),
        ("permanent_water_bodies", "chlorinity", lambda _: 0.0),
        ("permanent_water_bodies", "pigmentation", lambda _: 5.0),
        ("permanent_water_bodies", "wind_speed", lambda _: 2.0),
        ("permanent_water_bodies", "wind_direction", lambda _: 90.0),
    ]:
        if scene_parameter is not None:
            node_id = f"mat_{mat_id}"
            kpmap[f"{node_id}.{param}"] = UpdateParameter(
                scene_parameter,
                lookup_strategy=TypeIdLookupStrategy(
                    node_type=mi.BSDF, node_id=node_id, parameter_relpath=param
                ),
                flags=UpdateParameter.Flags.SPECTRAL,
            )
    
    return kpmap


def create_eradiate_scene(mesh_file, texture_file):
    """Create an Eradiate scene with realistic materials and similar viewpoint to Mitsuba."""
    from s2gos_generator.simulation.atmosphere import create_atmosphere
    
    # Load spectral data
    spectra = load_spectral_data()
    kpmap = create_spectral_parameter_map(spectra)
    
    # Load texture data - convert to numpy array from PNG
    texture_image = Image.open(texture_file)
    selection_texture_data = np.array(texture_image)
    selection_texture_data = np.atleast_3d(selection_texture_data)
    
    # Create kernel dictionary with materials and surface
    kdict = {}
    
    # Add material definitions
    kdict.update(REALISTIC_MATERIALS)
    
    # Create material selection surface
    kdict["terrain_material"] = {
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
           for i, mat_id in enumerate(REALISTIC_MATERIALS.keys())}
    }
    
    # Add terrain mesh
    kdict["terrain"] = {
        "type": "ply",
        "filename": str(mesh_file),
        "bsdf": {"type": "ref", "id": "terrain_material"},
        "id": "terrain"
    }
    
    # Create realistic atmosphere with molecular and aerosol components
    atmosphere = create_atmosphere(
        boa=0.0,
        toa=40.0e3,  # 40 km
        aerosol_ot=0.1,  # Optical thickness
        aerosol_scale=1e3,  # 1 km scale height
        aerosol_ds="sixsv-continental"  # Continental aerosol type
    )
    
    # Realistic solar illumination similar to original Mitsuba
    illumination = {
        "type": "directional",
        "zenith": 30.0 * ureg.deg,
        "azimuth": 180.0 * ureg.deg,
        "irradiance": {"type": "solar_irradiance", "dataset": "thuillier_2003"}
    }
    
    # Perspective sensor matching original Mitsuba example exactly
    measures = [
        {
            "type": "perspective",
            "id": "perspective_view", 
            "origin": [0, 0, 48612.90342332],  # Same height as original
            "target": [0, 0, 0],
            "up": [0, 1, 0],
            "fov": 70,
            "film_resolution": (1024, 768),  # Same resolution as original
            "srf": {
                "type": "delta",
                "wavelengths": [440, 550, 660] * ureg.nm  # RGB wavelengths
            },
            "spp": 512  # Match original Mitsuba quality
        }
    ]
    
    return AtmosphereExperiment(
        geometry={"type": "plane_parallel", "toa_altitude": 40.0 * ureg.km},
        atmosphere=atmosphere,
        surface=None,  # Will be added via kdict
        illumination=illumination,
        measures=measures,
        integrator={
            "type": "volpath", 
            "moment": True  # Enable moment-based rendering for accuracy
        },
        kdict=kdict,
        kpmap=kpmap
    )


def render_eradiate_scene(mesh_file, texture_file, output_dir):
    """Render scene using Eradiate for physically-based results."""
    print(f"Rendering with Eradiate...")
    print(f"Mesh: {mesh_file}")
    print(f"Texture: {texture_file}")
    
    try:
        # Create the scene experiment
        exp = create_eradiate_scene(mesh_file, texture_file)
        
        # Run simulation with higher quality settings  
        print("Running Eradiate simulation (this may take several minutes)...")
        eradiate.run(exp)  # Use spp from measure definition
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        raw_output = output_dir / "eradiate_results.nc"
        results_ds = exp.results["perspective_view"]
        results_ds.to_netcdf(raw_output)
        
        # Create RGB visualization similar to dreams-scenes approach
        img = dataarray_to_rgb(
            results_ds["radiance"],
            channels=[("w", 660), ("w", 550), ("w", 440)],  # RGB channels
            normalize=False,
        ) * 1.8  # Same enhancement as dreams-scenes
        
        img = np.clip(img, 0, 1)
        
        # Save RGB visualization
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
    """Generate a 100km x 100km scene around Gobabeb, Namibia with Eradiate rendering."""
    print("S2GOS Scene Generation with Eradiate Example")
    print("=" * 50)
    
    config = SceneGenerationConfig(
        center_lat=27.978497,
        center_lon=-15.590282,
        aoi_size_km=100.0,
        
        dem_index_path=Path("/home/gonzalezm/s2gos/s2gos-generator/src/s2gos_generator/data/dem_index.feather"),
        dem_root_dir=Path("/media/DATA/DEM"),
        landcover_index_path=Path("/home/gonzalezm/s2gos/s2gos-generator/src/s2gos_generator/data/landcover_index.feather"), 
        landcover_root_dir=Path("/home/gonzalezm/Data"),
        
        output_dir=Path("./eradiate_example_output_la_palma"),
        scene_name="eradiate_example_scene_la_palma",
        target_resolution_m=30.0
    )
    
    try:
        # Create and run the pipeline
        pipeline = SceneGenerationPipeline(config)
        assets = pipeline.run_full_pipeline()
        
        print(f"\nSuccess! Scene generated at: {assets.config_file.parent}")
        print(f"Key files:")
        print(f"  3D Mesh: {assets.mesh_file}")
        print(f"  Material Texture: {assets.selection_texture_file}")
        print(f"  Configuration: {assets.config_file}")
        
        # Create and save scene configuration
        if assets.mesh_file and assets.selection_texture_file:
            from s2gos_generator.simulation import create_s2gos_scene
            
            # Create complete scene configuration
            scene_config = create_s2gos_scene(
                scene_name=config.scene_name,
                mesh_path=str(assets.mesh_file),
                texture_path=str(assets.selection_texture_file),
                sensor_height=53612.90342332,
                spp=512
            )
            
            # Save scene configuration
            scene_config_path = assets.config_file.parent / "scene_config.yml"
            scene_config.save_yaml(scene_config_path)
            print(f"  Scene Config: {scene_config_path}")
            
            # Render with Eradiate
            eradiate_output_dir = assets.config_file.parent / "eradiate_renders"
            render_success = render_eradiate_scene(
                assets.mesh_file,
                assets.selection_texture_file, 
                eradiate_output_dir
            )
            if render_success:
                print(f"  Eradiate Results: {eradiate_output_dir}")
                
            # Cloud deployment ready
            print(f"\n  ✓ Scene ready for cloud deployment: {scene_config_path}")
            print(f"  ✓ Upload YAML + spectral data files to run remotely")
        
        return assets
        
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
    print("S2GOS Scene Generator with Eradiate V2 - Usage Example")
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
    print()
    print("Key improvements with Eradiate V2:")
    print("  - Real spectral reflectance data from NetCDF files")
    print("  - Proper BRDF models (RPV for vegetation/soil, ocean model for water)")
    print("  - Perspective sensor matching original Mitsuba viewpoint")
    print("  - Higher quality rendering (128 spp)")
    print("  - Realistic atmospheric scattering")
    print("  - Scientific accuracy for earth observation simulation")