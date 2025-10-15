"""AOI (Area of Interest) generation resources."""

import logging
from pathlib import Path
from typing import Optional

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

    ctx._target_aoi_polygon = aoi_polygon

    corners = list(aoi_polygon.exterior.coords[:-1])
    logging.info("AOI corners (lon, lat):")
    for i, (lon, lat) in enumerate(corners):
        logging.info(f"  Corner {i + 1}: ({lat:.6f}, {lon:.6f})")

    logging.info(
        f"AOI polygon: {ctx.aoi_size_km}km x {ctx.aoi_size_km}km at ({ctx.center_lat:.6f}, {ctx.center_lon:.6f})"
    )

    return None


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

    ctx._background_aoi_polygon = background_aoi_polygon

    return None
