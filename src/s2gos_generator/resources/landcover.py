"""Land cover processing resources."""

import logging
from pathlib import Path
from typing import Optional

from ..assets.landcover import LandCoverProcessor
from ..core.context import SceneResourceContext


def _process_landcover(
    ctx: SceneResourceContext,
    aoi_polygon,
    resolution_m: float,
    filename_prefix: str,  # "landcover" | "landcover_buffer" | "landcover_background"
    aoi_size_km: float,
) -> Path:
    processor = LandCoverProcessor(dataset=ctx.config.data_sources.landcover)
    output_path = (
        ctx.data_dir / f"{filename_prefix}_{ctx.scene_name}_{resolution_m}m.zarr"
    )
    processor.generate_landcover(
        aoi_polygon=aoi_polygon,
        output_path=output_path,
        target_resolution_m=resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=aoi_size_km,
    )
    return output_path


def process_target_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the target area.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated landcover zarr file
    """

    aoi_polygon = ctx._target_aoi_polygon
    if aoi_polygon is None:
        raise ValueError("Target AOI polygon not found in context")

    output_path = _process_landcover(
        ctx, aoi_polygon, ctx.target_resolution_m, "landcover", ctx.aoi_size_km
    )
    ctx.assets.landcover_file = output_path

    logging.info(f"Target landcover ({ctx.target_resolution_m}m): {output_path}")
    return output_path


def process_buffer_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the buffer area (if buffer is enabled).

    Uses configurable buffer resolution for optimal performance.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer landcover zarr file, or None if buffer disabled
    """

    buffer_aoi_polygon = ctx._buffer_aoi_polygon
    if buffer_aoi_polygon is None:
        logging.warning("Buffer AOI polygon not found in context")
        return None

    buffer_resolution_m = ctx.config.buffer.resolution_m
    output_path = _process_landcover(
        ctx,
        buffer_aoi_polygon,
        buffer_resolution_m,
        "landcover_buffer",
        ctx.config.buffer.size_km,
    )
    ctx.assets.buffer_landcover_file = output_path

    logging.info(f"Buffer landcover ({buffer_resolution_m}m): {output_path}")
    return output_path


def process_background_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the background area (if background is enabled).

    Background landcover uses regridded resolution for performance optimization.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated background landcover zarr file, or None if background disabled
    """

    background_aoi_polygon = ctx._background_aoi_polygon
    if background_aoi_polygon is None:
        logging.warning("Background AOI polygon not found in context")
        return None

    background_resolution_m = ctx.config.background.resolution_m
    output_path = _process_landcover(
        ctx,
        background_aoi_polygon,
        background_resolution_m,
        "landcover_background",
        ctx.config.background.size_km,
    )
    ctx.assets.background_landcover_file = output_path

    return output_path
