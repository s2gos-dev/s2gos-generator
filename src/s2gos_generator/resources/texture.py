"""Texture generation resources."""

import logging
from pathlib import Path
from typing import Optional

from ..assets.texture import TextureGenerator
from ..core.context import SceneResourceContext


def generate_target_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from target area land cover data.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated target selection texture file
    """

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

    logging.info(f"Target texture: {selection_texture_path}")
    return selection_texture_path


def generate_buffer_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from buffer area land cover data (if buffer is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer selection texture file, or None if buffer disabled
    """
    buffer_landcover_file_path = ctx.dependency_outputs["buffer_landcover"]
    if buffer_landcover_file_path is None:
        logging.warning("Buffer landcover file not found from dependencies")
        return None

    # Initialize texture generator
    texture_generator = TextureGenerator()

    # Generate buffer textures
    buffer_resolution_m = ctx.config.buffer_resolution_m
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

    return selection_texture_path


def generate_background_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from background area land cover data (if background is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated background selection texture file, or None if background disabled
    """
    background_landcover_file_path = ctx.dependency_outputs["background_landcover"]
    if background_landcover_file_path is None:
        logging.warning("Background landcover file not found from dependencies")
        return None

    # Initialize texture generator
    texture_generator = TextureGenerator()

    # Generate background textures
    background_resolution_m = ctx.config.background_resolution_m
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

    return selection_texture_path
