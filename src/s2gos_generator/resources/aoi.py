"""AOI (Area of Interest) generation resources."""

import logging
import sys
from pathlib import Path
from typing import Optional

resource_graph_path = Path("/home/gonzalezm/s2gos/s2gos_resource_graph/src")
if str(resource_graph_path) not in sys.path:
    sys.path.append(str(resource_graph_path))

from ..resource_graph.resource_registry import resource

from ..core.context import SceneResourceContext
from ..utils import create_aoi_polygon


@resource(id="aoi")
def generate_aoi(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate the Area of Interest polygon.
    
    This is the foundation resource that creates the AOI polygon
    used by all other processing steps.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        None (AOI polygon is stored in context for other resources to access)
    """
    logging.info("=== Generating AOI ===")
    
    aoi_polygon = create_aoi_polygon(
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        side_length_km=ctx.aoi_size_km,
    )
    
    # Store the AOI polygon in context for other resources
    ctx._target_aoi_polygon = aoi_polygon
    
    logging.info(f"Created AOI polygon: {ctx.aoi_size_km}km x {ctx.aoi_size_km}km")
    logging.info(f"Center: ({ctx.center_lat:.6f}, {ctx.center_lon:.6f})")
    
    return None  # AOI is stored in context, no file output


@resource(id="buffer_aoi", dependencies=["aoi"])  
def generate_buffer_aoi(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate buffer AOI polygon.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        None (buffer AOI polygon is stored in context)
    """
    logging.info("=== Generating Buffer AOI ===")
    
    buffer_size_km = ctx.config.buffer.buffer_size_km
    buffer_aoi_polygon = create_aoi_polygon(
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        side_length_km=buffer_size_km,
    )
    
    # Store the buffer AOI polygon in context
    ctx._buffer_aoi_polygon = buffer_aoi_polygon
    
    logging.info(f"Created buffer AOI polygon: {buffer_size_km}km x {buffer_size_km}km")
    
    return None


@resource(id="background_aoi", dependencies=["buffer_aoi"])
def generate_background_aoi(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate background AOI polygon.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        None (background AOI polygon is stored in context)
    """
    logging.info("=== Generating Background AOI ===")
    
    background_size_km = ctx.config.buffer.background_size_km
    background_aoi_polygon = create_aoi_polygon(
        center_lat=ctx.center_lat,
        center_lon=ctx.center_lon,
        side_length_km=background_size_km,
    )
    
    # Store the background AOI polygon in context
    ctx._background_aoi_polygon = background_aoi_polygon
    
    logging.info(f"Created background AOI polygon: {background_size_km}km x {background_size_km}km")
    
    return None