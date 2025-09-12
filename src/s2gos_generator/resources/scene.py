"""Scene description assembly resource."""

import logging
from pathlib import Path
from typing import Optional

from ..core.context import SceneResourceContext
from ..scene import create_s2gos_scene


def create_scene_description(ctx: SceneResourceContext) -> Optional[Path]:
    """Create the complete scene description from all generated assets.

    This is the final resource that assembles all processed components
    into a complete S2GOS scene description.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated scene description YAML file
    """
    logging.info("=== Creating Scene Description ===")

    # Get required dependencies
    target_mesh_path = ctx.dependency_outputs["target_mesh"]
    target_texture_path = ctx.dependency_outputs["target_texture"]

    if target_mesh_path is None or target_texture_path is None:
        raise ValueError(
            "Required target mesh and texture files not found from dependencies"
        )

    # Convert to relative paths
    mesh_path = str(target_mesh_path.relative_to(ctx.output_dir))
    texture_path = str(target_texture_path.relative_to(ctx.output_dir))

    # Get optional buffer components
    buffer_mesh_path = None
    buffer_texture_path = None
    buffer_size_km = None

    buffer_mesh_file = ctx.dependency_outputs.get("buffer_mesh")
    buffer_texture_file = ctx.dependency_outputs.get("buffer_texture")

    if (
        ctx.has_buffer
        and buffer_mesh_file is not None
        and buffer_texture_file is not None
    ):
        buffer_mesh_path = str(buffer_mesh_file.relative_to(ctx.output_dir))
        buffer_texture_path = str(buffer_texture_file.relative_to(ctx.output_dir))
        buffer_size_km = ctx.config.buffer_size_km

    # Get optional background components
    background_selection_texture = None
    background_size_km = None

    background_texture_file = ctx.dependency_outputs.get("background_texture")
    if (
        ctx.has_background
        and background_texture_file is not None
    ):
        background_selection_texture = str(
            background_texture_file.relative_to(ctx.output_dir)
        )
        background_size_km = ctx.config.background_size_km

    # Get buffer DEM file for background elevation calculation
    buffer_dem_file = None
    if ctx.has_buffer and ctx.assets.buffer_dem_file:
        buffer_dem_file = str(ctx.assets.buffer_dem_file.relative_to(ctx.output_dir))

    # Get processed user assets
    processed_objects = getattr(ctx, "processed_objects", None)

    # Get HAMSTER data paths
    hamster_data_paths = getattr(ctx, "hamster_data_paths", None)

    # Get additional material libraries (if any)
    additional_material_libraries = getattr(ctx, "additional_material_libraries", None)

    # Get tree instances (if any)
    tree_instances = getattr(ctx, "tree_instances", None)

    # Create scene description using existing function
    scene_description = create_s2gos_scene(
        scene_name=ctx.scene_name,
        mesh_path=mesh_path,
        texture_path=texture_path,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=ctx.aoi_size_km,
        resolution_m=ctx.target_resolution_m,
        buffer_mesh_path=buffer_mesh_path,
        buffer_texture_path=buffer_texture_path,
        buffer_size_km=buffer_size_km,
        output_dir=ctx.output_dir,
        buffer_dem_file=buffer_dem_file,
        background_elevation=ctx.config.background_elevation
        if ctx.config.enable_background
        else None,
        background_selection_texture=background_selection_texture,
        background_size_km=background_size_km,
        dem_index_path=ctx.config.data_sources.dem_index_path,
        landcover_index_path=ctx.config.data_sources.landcover_index_path,
        material_config_path=ctx.config.data_sources.material_config_path,
        atmosphere_config=ctx.config.atmosphere,
        hamster_data_paths=hamster_data_paths,
        processed_objects=processed_objects,
        additional_material_libraries=additional_material_libraries,
        tree_instances=tree_instances,
    )

    # Save scene description to file
    scene_description_file = ctx.output_dir / f"{ctx.scene_name}.yml"
    scene_description.save_yaml(scene_description_file)

    # Store in assets
    ctx.assets.config_file = scene_description_file
    ctx.assets.scene_description_file = scene_description_file

    # Store scene description in context for pipeline return
    ctx.scene_description = scene_description

    logging.info("=== Scene Generation Complete ===")
    logging.info(f"Scene description saved to: {scene_description_file}")

    # Log summary of generated assets
    logging.info("Generated Assets Summary:")
    logging.info(f"  Target mesh: {target_mesh_path}")
    logging.info(f"  Target texture: {target_texture_path}")

    if buffer_mesh_path:
        logging.info(f"  Buffer mesh: {buffer_mesh_file}")
        logging.info(f"  Buffer texture: {buffer_texture_file}")

    if background_selection_texture:
        logging.info(f"  Background texture: {background_texture_file}")

    if processed_objects:
        logging.info(f"  User assets: {len(processed_objects)} objects")

    if hamster_data_paths:
        logging.info(f"  HAMSTER data: {len(hamster_data_paths)} surface areas")

    return scene_description_file
