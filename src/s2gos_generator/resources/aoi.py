"""AOI (Area of Interest) generation resources."""

import logging
from pathlib import Path
from typing import Optional

# Import coordinate transformation system
from s2gos_utils.coordinates import CoordinateSystem

from ..core.context import SceneResourceContext


def generate_aoi(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate the Area of Interest polygon.

    This is the foundation resource that creates the AOI polygon
    used by all other processing steps.

    Args:
        ctx: Scene resource context

    Returns:
        None (AOI polygon is stored in context for other resources to access)
    """

    coords = CoordinateSystem(ctx.center_lat, ctx.center_lon)
    aoi_polygon = coords.create_scene_polygon(ctx.aoi_size_km)

    # Store the AOI polygon in context for other resources
    ctx._target_aoi_polygon = aoi_polygon

    logging.info(
        f"AOI polygon: {ctx.aoi_size_km}km x {ctx.aoi_size_km}km at ({ctx.center_lat:.6f}, {ctx.center_lon:.6f})"
    )

    return None  # AOI is stored in context, no file output


def generate_buffer_aoi(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate buffer AOI polygon.

    Args:
        ctx: Scene resource context

    Returns:
        None (buffer AOI polygon is stored in context)
    """

    buffer_size_km = ctx.config.buffer_size_km
    
    coords = CoordinateSystem(ctx.center_lat, ctx.center_lon)
    buffer_aoi_polygon = coords.create_scene_polygon(buffer_size_km)

    # Store the buffer AOI polygon in context
    ctx._buffer_aoi_polygon = buffer_aoi_polygon

    return None


def generate_background_aoi(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate background AOI polygon.

    Args:
        ctx: Scene resource context

    Returns:
        None (background AOI polygon is stored in context)
    """

    background_size_km = ctx.config.background_size_km
    
    coords = CoordinateSystem(ctx.center_lat, ctx.center_lon)
    background_aoi_polygon = coords.create_scene_polygon(background_size_km)

    # Store the background AOI polygon in context
    ctx._background_aoi_polygon = background_aoi_polygon

    return None
