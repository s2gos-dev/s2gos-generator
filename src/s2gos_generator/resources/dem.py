"""DEM (Digital Elevation Model) processing resources."""

import logging
from pathlib import Path
from typing import Optional

from ..assets.dem import DEMProcessor
from ..core.context import SceneResourceContext


def _process_dem(
    ctx: SceneResourceContext,
    aoi_polygon,
    resolution_m: float,
    filename_prefix: str,  # "dem" | "dem_buffer"
    aoi_size_km: float,
) -> Path:
    processor = DEMProcessor(dataset=ctx.config.data_sources.dem)
    output_path = (
        ctx.data_dir / f"{filename_prefix}_{ctx.scene_name}_{resolution_m}m.zarr"
    )
    processor.generate_dem(
        aoi_polygon=aoi_polygon,
        output_path=output_path,
        fillna_value=ctx.config.processing.dem_fillna_value,
        target_resolution_m=resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=aoi_size_km,
        flatten_dem=ctx.config.processing.flatten_dem,
    )
    return output_path


def process_target_dem(ctx: SceneResourceContext) -> Optional[Path]:
    """Process DEM data for the target area.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated DEM zarr file
    """

    aoi_polygon = ctx._target_aoi_polygon
    if aoi_polygon is None:
        raise ValueError("Target AOI polygon not found in context")

    output_path = _process_dem(
        ctx, aoi_polygon, ctx.target_resolution_m, "dem", ctx.aoi_size_km
    )
    ctx.assets.dem_file = output_path

    logging.info(f"Target DEM: {output_path}")
    return output_path


def process_buffer_dem(ctx: SceneResourceContext) -> Optional[Path]:
    """Process DEM data for the buffer area (if buffer is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer DEM zarr file, or None if buffer disabled
    """

    buffer_aoi_polygon = ctx._buffer_aoi_polygon
    if buffer_aoi_polygon is None:
        logging.warning("Buffer AOI polygon not found in context")
        return None

    buffer_resolution_m = ctx.config.buffer.resolution_m
    output_path = _process_dem(
        ctx,
        buffer_aoi_polygon,
        buffer_resolution_m,
        "dem_buffer",
        ctx.config.buffer.size_km,
    )
    ctx.assets.buffer_dem_file = output_path

    return output_path
