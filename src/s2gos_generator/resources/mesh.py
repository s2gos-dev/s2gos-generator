"""3D mesh generation resources."""

import logging
import sys
from pathlib import Path
from typing import Optional


resource_graph_path = Path("/home/gonzalezm/s2gos/s2gos_resource_graph/src")
if str(resource_graph_path) not in sys.path:
    sys.path.append(str(resource_graph_path))

from ..resource_graph.resource_registry import resource
from upath import UPath

from ..core.context import SceneResourceContext
from ..assets.mesh import MeshGenerator


@resource(id="target_mesh", dependencies=["target_dem"])
def generate_target_mesh(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate 3D mesh from target area DEM data.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated target mesh PLY file
    """
    logging.info("=== Generating Target 3D Mesh ===")
    
    # Get DEM file path from dependency
    dem_file_path = ctx.dependency_outputs["target_dem"]
    if dem_file_path is None:
        raise ValueError("Target DEM file not found from dependencies")
    
    # Initialize mesh generator
    mesh_generator = MeshGenerator()
    
    # Generate output path
    mesh_path = ctx.meshes_dir / f"{ctx.scene_name}_terrain.ply"
    
    # Generate mesh
    mesh = mesh_generator.generate_mesh_from_dem_file(
        dem_file_path=dem_file_path,
        output_path=mesh_path,
        add_uvs=True,
        handle_nans=ctx.config.processing.handle_dem_nans,
    )
    
    # Store in assets
    ctx.assets.mesh_file = mesh_path
    
    # Log mesh info
    mesh_info = mesh_generator.get_mesh_info(mesh)
    logging.info(
        f"Generated target mesh: {mesh_info['vertices']} vertices, {mesh_info['faces']} faces"
    )
    
    logging.info(f"Target mesh generation complete: {mesh_path}")
    return mesh_path


@resource(id="buffer_mesh", dependencies=["buffer_dem"])
def generate_buffer_mesh(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate 3D mesh from buffer area DEM data (if buffer is enabled).
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated buffer mesh PLY file, or None if buffer disabled
    """
    if not ctx.has_buffer:
        logging.info("Buffer system disabled, skipping buffer mesh generation")
        return None
    
    # Get buffer DEM file path from dependency
    buffer_dem_file_path = ctx.dependency_outputs["buffer_dem"]
    if buffer_dem_file_path is None:
        logging.warning("Buffer DEM file not found from dependencies")
        return None
    
    logging.info("=== Generating Buffer 3D Mesh ===")
    
    # Initialize mesh generator
    mesh_generator = MeshGenerator()
    
    # Generate output path
    mesh_path = ctx.meshes_dir / f"{ctx.scene_name}_buffer_terrain.ply"
    
    # Generate buffer mesh
    mesh = mesh_generator.generate_mesh_from_dem_file(
        dem_file_path=buffer_dem_file_path,
        output_path=mesh_path,
        add_uvs=True,
        handle_nans=ctx.config.processing.handle_dem_nans,
    )
    
    # Store in assets
    ctx.assets.buffer_mesh_file = mesh_path
    
    # Log mesh info
    mesh_info = mesh_generator.get_mesh_info(mesh)
    logging.info(
        f"Generated buffer mesh: {mesh_info['vertices']} vertices, {mesh_info['faces']} faces"
    )
    
    logging.info(f"Buffer mesh generation complete: {mesh_path}")
    return mesh_path