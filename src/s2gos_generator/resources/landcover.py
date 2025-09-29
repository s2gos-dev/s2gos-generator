"""Land cover processing resources."""

import logging
from pathlib import Path
from typing import Optional

from ..assets.landcover import LandCoverProcessor
from ..core.context import SceneResourceContext


def process_target_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the target area.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated landcover zarr file
    """

    landcover_processor = LandCoverProcessor(
        index_path=ctx.config.data_sources.landcover_index_path,
        landcover_root_dir=ctx.config.data_sources.landcover_root_dir,
    )

    landcover_filename = f"landcover_{ctx.scene_name}_{ctx.target_resolution_m}m.zarr"
    landcover_output_path = ctx.data_dir / landcover_filename

    aoi_polygon = ctx._target_aoi_polygon
    if aoi_polygon is None:
        raise ValueError("Target AOI polygon not found in context")

    landcover_processor.generate_landcover(
        aoi_polygon=aoi_polygon,
        output_path=landcover_output_path,
        target_resolution_m=ctx.target_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=ctx.aoi_size_km,
    )

    ctx.assets.landcover_file = landcover_output_path

    logging.info(
        f"Target landcover ({ctx.target_resolution_m}m): {landcover_output_path}"
    )
    return landcover_output_path


def process_buffer_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the buffer area (if buffer is enabled).

    Uses configurable buffer resolution for optimal performance.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer landcover zarr file, or None if buffer disabled
    """

    landcover_processor = LandCoverProcessor(
        index_path=ctx.config.data_sources.landcover_index_path,
        landcover_root_dir=ctx.config.data_sources.landcover_root_dir,
    )

    buffer_resolution_m = ctx.config.buffer_resolution_m
    landcover_filename = (
        f"landcover_buffer_{ctx.scene_name}_{buffer_resolution_m}m.zarr"
    )
    landcover_output_path = ctx.data_dir / landcover_filename

    buffer_aoi_polygon = ctx._buffer_aoi_polygon
    if buffer_aoi_polygon is None:
        logging.warning("Buffer AOI polygon not found in context")
        return None

    landcover_processor.generate_landcover(
        aoi_polygon=buffer_aoi_polygon,
        output_path=landcover_output_path,
        target_resolution_m=buffer_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=ctx.config.buffer_size_km,
    )

    ctx.assets.buffer_landcover_file = landcover_output_path

    logging.info(f"Buffer landcover ({buffer_resolution_m}m): {landcover_output_path}")
    return landcover_output_path


def process_background_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the background area (if background is enabled).

    Background landcover uses regridded resolution for performance optimization.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated background landcover zarr file, or None if background disabled
    """

    landcover_processor = LandCoverProcessor(
        index_path=ctx.config.data_sources.landcover_index_path,
        landcover_root_dir=ctx.config.data_sources.landcover_root_dir,
    )

    background_resolution_m = ctx.config.background_resolution_m
    landcover_filename = (
        f"landcover_background_{ctx.scene_name}_{background_resolution_m}m.zarr"
    )
    landcover_output_path = ctx.data_dir / landcover_filename

    background_aoi_polygon = ctx._background_aoi_polygon
    if background_aoi_polygon is None:
        logging.warning("Background AOI polygon not found in context")
        return None

    background_size_km = ctx.config.background_size_km
    landcover_processor.generate_landcover(
        aoi_polygon=background_aoi_polygon,
        output_path=landcover_output_path,
        target_resolution_m=background_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=background_size_km,
    )

    ctx.assets.background_landcover_file = landcover_output_path

    return landcover_output_path
