"""User asset processing resources."""

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional


resource_graph_path = Path("/home/gonzalezm/s2gos/s2gos_resource_graph/src")
if str(resource_graph_path) not in sys.path:
    sys.path.append(str(resource_graph_path))

from ..resource_graph.resource_registry import resource
from upath import UPath

from ..core.context import SceneResourceContext
from ..core.exceptions import ProcessingError


@resource(id="user_assets", dependencies=["target_dem"])
def process_user_assets(ctx: SceneResourceContext) -> Optional[Path]:
    """Process user assets (3D objects) for placement in the scene.
    
    Args:
        ctx: Scene resource context
        
    Returns:
        Path to the objects directory containing processed assets
    """
    logging.info(f"=== Processing {len(ctx.config.user_assets)} user assets ===")
    
    from ..utils.geometry import latlon_to_scene_coordinates, query_elevation_at_coordinate
    from s2gos_utils.io.paths import mkdir
    
    processed_objects = []
    
    # Create objects directory
    objects_dir = ctx.output_dir / "objects"
    mkdir(objects_dir)
    
    # Get target DEM file for elevation queries
    target_dem_path = ctx.dependency_outputs["target_dem"]
    if target_dem_path is None:
        raise ProcessingError("Target DEM data not available for elevation querying", "user_assets", None)
    
    for i, asset in enumerate(ctx.config.user_assets):
        try:
            logging.info(f"Processing object {i+1}/{len(ctx.config.user_assets)}: {asset.object_id}")
            
            # Convert lat/lon to scene coordinates
            lon, lat = asset.coordinate
            scene_x, scene_y = latlon_to_scene_coordinates(
                target_lat=lat,
                target_lon=lon,
                scene_center_lat=ctx.center_lat,
                scene_center_lon=ctx.center_lon
            )
            
            # Query elevation at coordinate
            elevation = query_elevation_at_coordinate(
                dem_zarr_path=target_dem_path,
                latitude=lat,
                longitude=lon,
                scene_center_lat=ctx.center_lat,
                scene_center_lon=ctx.center_lon
            )
            
            final_z = elevation + asset.elevation_offset
            
            # Copy PLY file to objects directory
            ply_filename = f"{asset.object_id}.ply"
            output_ply_path = objects_dir / ply_filename
            shutil.copy2(asset.ply_path, output_ply_path)
            
            # Create object data structure
            object_data = {
                "id": asset.object_id,
                "mesh": f"objects/{ply_filename}",
                "position": [scene_x, scene_y, final_z],
                "scale": asset.scale,
                "rotation": [asset.rotation_x, asset.rotation_y, asset.rotation_z]
            }
            
            if asset.material:
                object_data["material"] = asset.material
            
            processed_objects.append(object_data)
            logging.info(f"Processed object {asset.object_id}: position=({scene_x:.2f}, {scene_y:.2f}, {final_z:.2f})")
            
        except Exception as e:
            raise ProcessingError(f"Failed to process user asset {asset.object_id}: {e}", "user_assets", e) from e
    
    # Store processed objects in context for scene description
    ctx.processed_objects = processed_objects
    
    logging.info(f"Successfully processed {len(processed_objects)} user assets")
    return objects_dir