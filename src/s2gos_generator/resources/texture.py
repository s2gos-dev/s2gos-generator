"""Texture generation resources."""

import logging
import sys
from pathlib import Path
from typing import Optional

import xarray as xr


resource_graph_path = Path("/home/gonzalezm/s2gos/s2gos_resource_graph/src")
if str(resource_graph_path) not in sys.path:
    sys.path.append(str(resource_graph_path))

from ..resource_graph.resource_registry import resource
from upath import UPath

from ..core.context import SceneResourceContext
from ..assets.texture import TextureGenerator


@resource(id="target_texture", dependencies=["target_landcover"])
def generate_target_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from target area land cover data.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated target selection texture file
    """
    logging.info("=== Generating Target Textures ===")
    
    # Get landcover file path from dependency
    landcover_file_path = ctx.dependency_outputs["target_landcover"]
    if landcover_file_path is None:
        raise ValueError("Target landcover file not found from dependencies")
    
    # Initialize texture generator
    texture_generator = TextureGenerator()
    
    # Generate textures
    selection_texture_path, preview_texture_path = (
        texture_generator.generate_textures_from_file(
            landcover_file_path=landcover_file_path,
            output_dir=ctx.textures_dir,
            base_name=f"{ctx.scene_name}_{ctx.target_resolution_m}m",
            create_preview=ctx.config.processing.generate_texture_preview,
        )
    )
    
    # Store in assets
    ctx.assets.selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.preview_texture_file = preview_texture_path
    
    # Log landcover analysis
    landcover_dataset = xr.open_zarr(landcover_file_path)
    landcover_data = landcover_dataset["landcover"]
    if isinstance(landcover_data, xr.Dataset):
        landcover_data = landcover_data[list(landcover_data.data_vars.keys())[0]]
    
    analysis = texture_generator.analyze_landcover_classes(landcover_data)
    logging.info(
        f"Texture analysis: {analysis['unique_classes']} land cover classes found"
    )
    
    logging.info(f"Target texture generation complete: {selection_texture_path}")
    return selection_texture_path


@resource(id="buffer_texture", dependencies=["buffer_landcover"])
def generate_buffer_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from buffer area land cover data (if buffer is enabled).
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated buffer selection texture file, or None if buffer disabled
    """
    if not ctx.has_buffer:
        logging.info("Buffer system disabled, skipping buffer texture generation")
        return None
    
    # Get buffer landcover file path from dependency
    buffer_landcover_file_path = ctx.dependency_outputs["buffer_landcover"]
    if buffer_landcover_file_path is None:
        logging.warning("Buffer landcover file not found from dependencies")
        return None
    
    logging.info("=== Generating Buffer Textures ===")
    
    # Initialize texture generator
    texture_generator = TextureGenerator()
    
    # Generate buffer textures
    buffer_resolution_m = ctx.config.buffer.buffer_resolution_m
    selection_texture_path, preview_texture_path = (
        texture_generator.generate_textures_from_file(
            landcover_file_path=buffer_landcover_file_path,
            output_dir=ctx.textures_dir,
            base_name=f"{ctx.scene_name}_buffer_{buffer_resolution_m}m",
            create_preview=ctx.config.processing.generate_texture_preview,
        )
    )
    
    # Store in assets
    ctx.assets.buffer_selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.buffer_preview_texture_file = preview_texture_path
    
    logging.info(f"Buffer texture generation complete: {selection_texture_path}")
    return selection_texture_path


@resource(id="background_texture", dependencies=["background_landcover"])
def generate_background_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from background area land cover data (if background is enabled).
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated background selection texture file, or None if background disabled
    """
    if not ctx.has_buffer or not hasattr(ctx.config.buffer, 'background_size_km'):
        logging.info("Background system disabled, skipping background texture generation")
        return None
    
    # Get background landcover file path from dependency
    background_landcover_file_path = ctx.dependency_outputs["background_landcover"]
    if background_landcover_file_path is None:
        logging.warning("Background landcover file not found from dependencies")
        return None
    
    logging.info("=== Generating Background Textures ===")
    
    # Initialize texture generator
    texture_generator = TextureGenerator()
    
    # Generate background textures
    background_resolution_m = ctx.config.buffer.background_resolution_m
    selection_texture_path, preview_texture_path = (
        texture_generator.generate_textures_from_file(
            landcover_file_path=background_landcover_file_path,
            output_dir=ctx.textures_dir,
            base_name=f"{ctx.scene_name}_background_{background_resolution_m}m",
            create_preview=ctx.config.processing.generate_texture_preview,
        )
    )
    
    # Store in assets
    ctx.assets.background_selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.background_preview_texture_file = preview_texture_path
    
    logging.info(f"Background texture generation complete: {selection_texture_path}")
    return selection_texture_path