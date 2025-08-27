"""Land cover processing resources."""

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
from ..assets.landcover import LandCoverProcessor


@resource(id="target_landcover", dependencies=["aoi"])
def process_target_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the target area.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated landcover zarr file
    """
    logging.info("=== Processing Target Land Cover Data ===")
    
    # Initialize land cover processor
    landcover_processor = LandCoverProcessor(
        index_path=ctx.config.data_sources.landcover_index_path,
        landcover_root_dir=ctx.config.data_sources.landcover_root_dir
    )
    
    # Generate output path
    landcover_filename = f"landcover_{ctx.scene_name}_{ctx.target_resolution_m}m.zarr"
    landcover_output_path = ctx.data_dir / landcover_filename
    
    # Get AOI polygon from context
    aoi_polygon = ctx._target_aoi_polygon
    if aoi_polygon is None:
        raise ValueError("Target AOI polygon not found in context")
    
    # Process land cover
    landcover_processor.generate_landcover(
        aoi_polygon=aoi_polygon,
        output_path=landcover_output_path,
        target_resolution_m=ctx.target_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=ctx.aoi_size_km,
    )
    
    # Store in assets
    ctx.assets.landcover_file = landcover_output_path
    
    logging.info(f"Target land cover processing complete: {landcover_output_path}")
    return landcover_output_path


@resource(id="buffer_landcover", dependencies=["buffer_aoi"])
def process_buffer_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the buffer area (if buffer is enabled).
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated buffer landcover zarr file, or None if buffer disabled
    """
    if not ctx.has_buffer:
        logging.info("Buffer system disabled, skipping buffer landcover processing")
        return None
    
    logging.info("=== Processing Buffer Land Cover Data ===")
    
    # Initialize land cover processor
    landcover_processor = LandCoverProcessor(
        index_path=ctx.config.data_sources.landcover_index_path,
        landcover_root_dir=ctx.config.data_sources.landcover_root_dir
    )
    
    # Generate output path
    buffer_resolution_m = ctx.config.buffer.buffer_resolution_m
    landcover_filename = f"landcover_buffer_{ctx.scene_name}_{buffer_resolution_m}m.zarr"
    landcover_output_path = ctx.data_dir / landcover_filename
    
    # Get buffer AOI polygon from context
    buffer_aoi_polygon = ctx._buffer_aoi_polygon
    if buffer_aoi_polygon is None:
        logging.warning("Buffer AOI polygon not found in context")
        return None
    
    # Process buffer land cover
    buffer_size_km = ctx.config.buffer.buffer_size_km
    landcover_processor.generate_landcover(
        aoi_polygon=buffer_aoi_polygon,
        output_path=landcover_output_path,
        target_resolution_m=buffer_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=buffer_size_km,
    )
    
    # Store in assets
    ctx.assets.buffer_landcover_file = landcover_output_path
    
    logging.info(f"Buffer land cover processing complete: {landcover_output_path}")
    return landcover_output_path


@resource(id="background_landcover", dependencies=["background_aoi"])
def process_background_landcover(ctx: SceneResourceContext) -> Optional[Path]:
    """Process land cover data for the background area (if background is enabled).
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the generated background landcover zarr file, or None if background disabled
    """
    if not ctx.has_buffer or not hasattr(ctx.config.buffer, 'background_size_km'):
        logging.info("Background system disabled, skipping background landcover processing")
        return None
    
    logging.info("=== Processing Background Land Cover Data ===")
    
    # Initialize land cover processor
    landcover_processor = LandCoverProcessor(
        index_path=ctx.config.data_sources.landcover_index_path,
        landcover_root_dir=ctx.config.data_sources.landcover_root_dir
    )
    
    # Generate output path
    background_resolution_m = ctx.config.buffer.background_resolution_m
    landcover_filename = f"landcover_background_{ctx.scene_name}_{background_resolution_m}m.zarr"
    landcover_output_path = ctx.data_dir / landcover_filename
    
    # Get background AOI polygon from context
    background_aoi_polygon = ctx._background_aoi_polygon
    if background_aoi_polygon is None:
        logging.warning("Background AOI polygon not found in context")
        return None
    
    # Process background land cover
    background_size_km = ctx.config.buffer.background_size_km
    landcover_processor.generate_landcover(
        aoi_polygon=background_aoi_polygon,
        output_path=landcover_output_path,
        target_resolution_m=background_resolution_m,
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        aoi_size_km=background_size_km,
    )
    
    # Store in assets
    ctx.assets.background_landcover_file = landcover_output_path
    
    logging.info(f"Background land cover processing complete: {landcover_output_path}")
    return landcover_output_path