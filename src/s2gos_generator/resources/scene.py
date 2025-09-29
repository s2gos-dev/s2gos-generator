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

    target_mesh_path = ctx.dependency_outputs["target_mesh"]
    target_texture_path = ctx.dependency_outputs["target_texture"]

    if target_mesh_path is None or target_texture_path is None:
        raise ValueError(
            "Required target mesh and texture files not found from dependencies"
        )

    mesh_path = str(target_mesh_path.relative_to(ctx.output_dir))
    texture_path = str(target_texture_path.relative_to(ctx.output_dir))

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

    background_selection_texture = None
    background_size_km = None

    background_texture_file = ctx.dependency_outputs.get("background_texture")
    if ctx.has_background and background_texture_file is not None:
        background_selection_texture = str(
            background_texture_file.relative_to(ctx.output_dir)
        )
        background_size_km = ctx.config.background_size_km

    buffer_dem_file = None
    if ctx.has_buffer and ctx.assets.buffer_dem_file:
        buffer_dem_file = str(ctx.assets.buffer_dem_file.relative_to(ctx.output_dir))

    processed_objects = getattr(ctx, "processed_objects", None)

    hamster_data_paths = getattr(ctx, "hamster_data_paths", None)
    if hamster_data_paths:
        logging.info(
            f"Scene description found HAMSTER data paths: {hamster_data_paths}"
        )
    else:
        logging.info("Scene description: No HAMSTER data paths found in context")

    additional_material_libraries = getattr(ctx, "additional_material_libraries", None)

    vegetation_instances = getattr(ctx, "vegetation_instances", None)
    tree_collection_references = []

    if vegetation_instances:
        logging.info(
            f"Processing {len(vegetation_instances)} tree instances for hybrid scene format"
        )

        from .vegetation import save_tree_collection_binary

        binary_filename = f"{ctx.scene_name}_trees.npy"
        binary_path = ctx.output_dir / binary_filename

        tree_metadata = save_tree_collection_binary(vegetation_instances, binary_path)

        # Create reference entry for scene description
        if tree_metadata["count"] > 0:
            tree_collection_references.append(
                {
                    "type": "tree_collection",
                    "name": "target_trees",
                    "material": "forest_tree",
                    "data_file": binary_filename,
                    "count": tree_metadata["count"],
                    "bounds": tree_metadata["bounds"],
                    "file_size_bytes": tree_metadata["file_size_bytes"],
                    "format": "numpy_structured_array",
                    "dtype_info": tree_metadata["dtype_info"],
                }
            )

        logging.info(
            f"Saved tree data to binary format: {binary_path} ({tree_metadata['file_size_bytes']} bytes)"
        )
        vegetation_instances = None

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
        # Use baresoil for tree areas - 3D trees handle the vegetation, surface should be soil
        landcover_mapping_overrides={
            "tree_cover": "baresoil",  # Surface under 3D trees
            "shrubland": "baresoil",  # Surface under 3D shrubs
        },
        atmosphere_config=ctx.config.atmosphere,
        hamster_data_paths=hamster_data_paths,
        processed_objects=processed_objects,
        additional_material_libraries=additional_material_libraries,
        tree_collection_references=tree_collection_references,
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
