"""DEM (Digital Elevation Model) processing resources."""

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
from ..assets.dem import DEMProcessor


@resource(id="target_dem", dependencies=["aoi"])
def process_target_dem(ctx: SceneResourceContext) -> Optional[Path]:
    """Process DEM data for the target area.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated DEM zarr file
    """
    logging.info("=== Processing Target DEM Data ===")
    
    # Initialize DEM processor
    dem_processor = DEMProcessor(
        index_path=ctx.config.data_sources.dem_index_path,
        dem_root_dir=ctx.config.data_sources.dem_root_dir
    )
    
    # Generate output path
    dem_filename = f"dem_{ctx.scene_name}_{ctx.target_resolution_m}m.zarr"
    dem_output_path = ctx.data_dir / dem_filename
    
    # Get AOI polygon from context
    aoi_polygon = ctx._target_aoi_polygon
    if aoi_polygon is None:
        raise ValueError("Target AOI polygon not found in context")
    
    # Process DEM
    dem_processor.generate_dem(
        aoi_polygon=aoi_polygon,
        output_path=dem_output_path,
        fillna_value=ctx.config.processing.dem_fillna_value,
        target_resolution_m=ctx.target_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=ctx.aoi_size_km,
    )
    
    # Store in assets
    ctx.assets.dem_file = dem_output_path
    
    logging.info(f"Target DEM processing complete: {dem_output_path}")
    return dem_output_path


@resource(id="buffer_dem", dependencies=["buffer_aoi"])
def process_buffer_dem(ctx: SceneResourceContext) -> Optional[Path]:
    """Process DEM data for the buffer area (if buffer is enabled).
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated buffer DEM zarr file, or None if buffer disabled
    """
    if not ctx.has_buffer:
        logging.info("Buffer system disabled, skipping buffer DEM processing")
        return None
    
    logging.info("=== Processing Buffer DEM Data ===")
    
    # Initialize DEM processor
    dem_processor = DEMProcessor(
        index_path=ctx.config.data_sources.dem_index_path,
        dem_root_dir=ctx.config.data_sources.dem_root_dir
    )
    
    # Generate output path
    buffer_resolution_m = ctx.config.buffer.buffer_resolution_m
    dem_filename = f"dem_buffer_{ctx.scene_name}_{buffer_resolution_m}m.zarr"
    dem_output_path = ctx.data_dir / dem_filename
    
    # Get buffer AOI polygon from context
    buffer_aoi_polygon = ctx._buffer_aoi_polygon
    if buffer_aoi_polygon is None:
        logging.warning("Buffer AOI polygon not found in context")
        return None
    
    # Process buffer DEM
    buffer_size_km = ctx.config.buffer.buffer_size_km
    dem_processor.generate_dem(
        aoi_polygon=buffer_aoi_polygon,
        output_path=dem_output_path,
        fillna_value=ctx.config.processing.dem_fillna_value,
        target_resolution_m=buffer_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=buffer_size_km,
    )
    
    # Store in assets
    ctx.assets.buffer_dem_file = dem_output_path
    
    logging.info(f"Buffer DEM processing complete: {dem_output_path}")
    return dem_output_path