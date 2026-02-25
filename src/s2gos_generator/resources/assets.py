"""User asset processing resources."""

import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

from ..core.context import SceneResourceContext


def _convert_to_scene_coords(
    coordinate: Union[Tuple[float, float], List[float]],
    coord_type: Literal["geographic", "scene"],
    coords,
) -> Tuple[float, float]:
    """Convert coordinates to scene coordinate system.

    Args:
        coordinate: Either (lon, lat) or (x, y) depending on coord_type
        coord_type: "geographic" or "scene"
        coords: CoordinateSystem instance for conversion

    Returns:
        (scene_x, scene_y) in scene coordinates
    """
    if coord_type == "geographic":
        lon, lat = coordinate
        return coords.latlon_to_scene(lat, lon)
    return tuple(coordinate)


def _create_asset_exclusion_zone(
    scene_x: float,
    scene_y: float,
    exclusion_zone: Union[float, Tuple[float, float]],
):
    """Create exclusion zone geometry around asset position.

    Args:
        scene_x, scene_y: Object position in scene coordinates
        exclusion_zone: Either a radius (float) or box dimensions (width, height)

    Returns:
        Shapely Polygon
    """
    from shapely.geometry import Point, box

    if isinstance(exclusion_zone, (int, float)):
        return Point(scene_x, scene_y).buffer(exclusion_zone)
    else:
        width, height = exclusion_zone
        half_w, half_h = width / 2, height / 2
        return box(
            scene_x - half_w,
            scene_y - half_h,
            scene_x + half_w,
            scene_y + half_h,
        )


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
    exclusion_zones = []

    objects_dir = ctx.output_dir / "objects"
    mkdir(objects_dir)

    target_dem_path = ctx.dependency_outputs["target_dem"]
    if target_dem_path is None:
        raise RuntimeError("Target DEM data not available for elevation querying")

    coords = CoordinateSystem(ctx.center_lat, ctx.center_lon)
    logging.info("Using cached CoordinateSystem for asset placement")

    for i, asset in enumerate(ctx.user_assets):
        try:
            if asset.coord_type == "geographic":
                lon, lat = asset.coordinate
                scene_x, scene_y = coords.latlon_to_scene(lat, lon)
                elevation = coords.query_height_from_dem(lat, lon, target_dem_path)
            else:
                scene_x, scene_y = asset.coordinate
                lat, lon = coords.scene_to_latlon(scene_x, scene_y)
                elevation = coords.query_height_from_dem(lat, lon, target_dem_path)

            final_z = elevation + asset.elevation_offset

            if asset.exclusion_zone is not None:
                geometry = _create_asset_exclusion_zone(
                    scene_x, scene_y, asset.exclusion_zone
                )
                exclusion_zones.append(
                    {
                        "source": f"asset_{asset.object_id}",
                        "geometry": geometry,
                    }
                )

                if isinstance(asset.exclusion_zone, (int, float)):
                    logging.info(
                        f"Added {asset.exclusion_zone}m circular exclusion zone for '{asset.object_id}'"
                    )
                else:
                    w, h = asset.exclusion_zone
                    logging.info(
                        f"Added {w}x{h}m box exclusion zone for '{asset.object_id}'"
                    )

            ply_filename = f"{asset.object_id}.ply"
            output_ply_path = objects_dir / ply_filename
            copy(asset.ply_path, output_ply_path)

            object_data = {
                "id": asset.object_id,
                "mesh": f"objects/{ply_filename}",
                "position": [scene_x, scene_y, final_z],
                "scale": asset.scale,
                "rotation": [asset.rotation_x, asset.rotation_y, asset.rotation_z],
                "blender_fix": asset.blender_fix,
            }

            if asset.material:
                if isinstance(asset.material, dict):
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
    ctx.vegetation_exclusion_zones.extend(exclusion_zones)

    logging.info(f"Processed {len(processed_objects)} user assets")
    if inline_materials:
        logging.info(f"Extracted {len(inline_materials)} inline material definitions")
    if exclusion_zones:
        logging.info(
            f"Extracted {len(exclusion_zones)} vegetation exclusion zones from assets"
        )
    return objects_dir


def process_vegetation_exclusion_zones(ctx: SceneResourceContext) -> list[dict]:
    """Process standalone vegetation exclusion zones.

    Args:
        ctx: Scene resource context

    Returns:
        List of exclusion zone dictionaries with shapely geometries
    """
    from shapely.geometry import Point, Polygon, box

    from ..core.config import BoxGeometry, CircleGeometry, PolygonGeometry

    if not ctx.config.vegetation_exclusion_zones:
        logging.info("No standalone vegetation exclusion zones configured")
        return []

    coords = ctx.coordinate_system
    exclusion_zones = []

    for zone_config in ctx.config.vegetation_exclusion_zones:
        try:
            geom = zone_config.geometry
            if isinstance(geom, CircleGeometry):
                scene_x, scene_y = _convert_to_scene_coords(
                    geom.center, geom.coord_type, coords
                )
                geometry = Point(scene_x, scene_y).buffer(geom.radius)

            elif isinstance(geom, BoxGeometry):
                scene_x, scene_y = _convert_to_scene_coords(
                    geom.center, geom.coord_type, coords
                )
                half_w, half_h = geom.width / 2, geom.height / 2
                geometry = box(
                    scene_x - half_w,
                    scene_y - half_h,
                    scene_x + half_w,
                    scene_y + half_h,
                )

            elif isinstance(geom, PolygonGeometry):
                if geom.coord_type == "geographic":
                    scene_coords = [
                        coords.latlon_to_scene(lat, lon)
                        for lon, lat in geom.coordinates
                    ]
                else:
                    scene_coords = list(geom.coordinates)
                geometry = Polygon(scene_coords)

            else:
                logging.warning(
                    f"Unknown geometry type for zone '{zone_config.zone_id}'"
                )
                continue

            exclusion_zones.append(
                {
                    "source": f"zone_{zone_config.zone_id}",
                    "geometry": geometry,
                }
            )
            logging.info(f"Processed exclusion zone '{zone_config.zone_id}'")

        except Exception as e:
            logging.warning(
                f"Failed to process exclusion zone '{zone_config.zone_id}': {e}"
            )

    ctx.vegetation_exclusion_zones.extend(exclusion_zones)
    logging.info(
        f"Processed {len(exclusion_zones)} standalone vegetation exclusion zones"
    )
    return None
