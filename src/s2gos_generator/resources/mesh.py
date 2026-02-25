"""3D mesh generation resources."""

import logging
from pathlib import Path
from typing import Optional

from ..assets.mesh import MeshGenerator
from ..core.context import SceneResourceContext


def generate_target_mesh(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate 3D mesh from target area DEM data.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated target mesh PLY file
    """

    dem_file_path = ctx.dependency_outputs["target_dem"]
    if dem_file_path is None:
        raise ValueError("Target DEM file not found from dependencies")

    mesh_generator = MeshGenerator()

    mesh_path = ctx.meshes_dir / f"{ctx.scene_name}_terrain.ply"

    _ = mesh_generator.generate_mesh_from_dem_file(
        dem_file_path=dem_file_path,
        output_path=mesh_path,
        add_uvs=True,
        handle_nans=ctx.config.processing.handle_dem_nans,
    )

    ctx.assets.mesh_file = mesh_path

    logging.info(f"Target mesh: {mesh_path}")
    return mesh_path


def generate_buffer_mesh(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate 3D mesh from buffer area DEM data (if buffer is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer mesh PLY file, or None if buffer disabled
    """
    buffer_dem_file_path = ctx.dependency_outputs["buffer_dem"]
    if buffer_dem_file_path is None:
        logging.warning("Buffer DEM file not found from dependencies")
        return None

    mesh_generator = MeshGenerator()

    mesh_path = ctx.meshes_dir / f"{ctx.scene_name}_buffer_terrain.ply"

    _ = mesh_generator.generate_mesh_from_dem_file(
        dem_file_path=buffer_dem_file_path,
        output_path=mesh_path,
        add_uvs=True,
        handle_nans=ctx.config.processing.handle_dem_nans,
    )

    ctx.assets.buffer_mesh_file = mesh_path

    return mesh_path
