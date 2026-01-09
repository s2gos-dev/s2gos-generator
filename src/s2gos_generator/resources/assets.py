"""User asset processing resources."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from ..core.context import SceneResourceContext


def process_user_assets(ctx: SceneResourceContext) -> Optional[Path]:
    """Process user assets (3D objects) for placement in the scene.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the objects directory containing processed assets
    """

    from s2gos_utils.coordinates import CoordinateSystem
    from s2gos_utils.io.paths import copy, mkdir

    processed_objects = []
    inline_materials = {}

    objects_dir = ctx.output_dir / "objects"
    mkdir(objects_dir)

    target_dem_path = ctx.dependency_outputs["target_dem"]
    if target_dem_path is None:
        raise RuntimeError("Target DEM data not available for elevation querying")

    coords = CoordinateSystem(ctx.center_lat, ctx.center_lon)
    logging.info("Using cached CoordinateSystem for asset placement")

    for i, asset in enumerate(ctx.user_assets):
        try:
            # Convert lat/lon to scene coordinates
            lon, lat = asset.coordinate

            # Use cached coordinate system
            scene_x, scene_y = coords.latlon_to_scene(lat, lon)

            # Query elevation at coordinate using CoordinateSystem method
            elevation = coords.query_height_from_dem(lat, lon, target_dem_path)

            final_z = elevation + asset.elevation_offset

            # Copy PLY file to objects directory
            ply_filename = f"{asset.object_id}.ply"
            output_ply_path = objects_dir / ply_filename
            copy(asset.ply_path, output_ply_path)

            # Create object data structure
            object_data = {
                "id": asset.object_id,
                "mesh": f"objects/{ply_filename}",
                "position": [scene_x, scene_y, final_z],
                "scale": asset.scale,
                "rotation": [asset.rotation_x, asset.rotation_y, asset.rotation_z],
            }

            if asset.material:
                if isinstance(asset.material, dict):
                    # Generate unique material ID
                    mat_id = asset.get_inline_material_id()
                    inline_materials[mat_id] = asset.material
                    object_data["material"] = mat_id
                    logging.info(
                        f"Added inline material '{mat_id}' for object '{asset.object_id}'"
                    )
                else:
                    object_data["material"] = asset.material

            if asset.face_normals is not None:
                object_data["face_normals"] = asset.face_normals

            processed_objects.append(object_data)

        except Exception as e:
            raise RuntimeError(
                f"Failed to process user asset {asset.object_id}: {e}"
            ) from e

    ctx.processed_objects = processed_objects
    ctx.inline_materials = inline_materials

    logging.info(f"Processed {len(processed_objects)} user assets")
    if inline_materials:
        logging.info(f"Extracted {len(inline_materials)} inline material definitions")
    return objects_dir
