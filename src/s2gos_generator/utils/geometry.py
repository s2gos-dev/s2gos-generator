"""Geometry utilities for S2GOS scene generation."""

import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon
from upath import UPath


def create_aoi_polygon(
    center_lat: float, center_lon: float, side_length_km: float = 200.0
) -> Polygon:
    """Create a square polygon in WGS84 CRS centered at a given point.

    Args:
        center_lat: Center latitude in degrees
        center_lon: Center longitude in degrees
        side_length_km: Side length of the square in kilometers

    Returns:
        Square polygon centered at the given coordinates
    """
    wgs84_crs = CRS("EPSG:4326")
    local_azimuthal_crs = CRS(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +ellps=WGS84 +units=m"
    )

    transformer_to_local = Transformer.from_crs(
        wgs84_crs, local_azimuthal_crs, always_xy=True
    )
    transformer_from_local = Transformer.from_crs(
        local_azimuthal_crs, wgs84_crs, always_xy=True
    )

    center_x, center_y = transformer_to_local.transform(center_lon, center_lat)
    half_side_m = (side_length_km * 1000) / 2

    local_corners = [
        (center_x - half_side_m, center_y - half_side_m),
        (center_x + half_side_m, center_y - half_side_m),
        (center_x + half_side_m, center_y + half_side_m),
        (center_x - half_side_m, center_y + half_side_m),
    ]

    lon_lat_coords = [transformer_from_local.transform(x, y) for x, y in local_corners]

    return Polygon(lon_lat_coords)


def latlon_to_scene_coordinates(
    target_lat: float,
    target_lon: float,
    scene_center_lat: float,
    scene_center_lon: float,
) -> tuple[float, float]:
    """Convert lat/lon coordinates to scene-local coordinates in meters.

    Args:
        target_lat: Latitude of the target point in degrees
        target_lon: Longitude of the target point in degrees
        scene_center_lat: Latitude of scene center in degrees
        scene_center_lon: Longitude of scene center in degrees

    Returns:
        Tuple of (x, y) coordinates in meters relative to scene center
    """
    wgs84_crs = CRS("EPSG:4326")
    local_azimuthal_crs = CRS(
        f"+proj=aeqd +lat_0={scene_center_lat} +lon_0={scene_center_lon} +ellps=WGS84 +units=m"
    )

    transformer_to_local = Transformer.from_crs(
        wgs84_crs, local_azimuthal_crs, always_xy=True
    )

    center_x, center_y = transformer_to_local.transform(
        scene_center_lon, scene_center_lat
    )

    target_x, target_y = transformer_to_local.transform(target_lon, target_lat)

    return (target_x - center_x, target_y - center_y)


def query_elevation_at_coordinate(
    dem_zarr_path: UPath,
    latitude: float,
    longitude: float,
    scene_center_lat: float,
    scene_center_lon: float,
) -> float:
    """Query elevation from DEM dataset at a specific coordinate.

    Args:
        dem_zarr_path: Path to the DEM zarr dataset
        latitude: Latitude coordinate in degrees
        longitude: Longitude coordinate in degrees
        scene_center_lat: Scene center latitude for coordinate transformation
        scene_center_lon: Scene center longitude for coordinate transformation

    Returns:
        Elevation value in meters
    """
    dem_dataset = xr.open_zarr(dem_zarr_path)
    elevation_data = dem_dataset["elevation"]

    scene_x, scene_y = latlon_to_scene_coordinates(
        target_lat=latitude,
        target_lon=longitude,
        scene_center_lat=scene_center_lat,
        scene_center_lon=scene_center_lon,
    )

    try:
        elevation = elevation_data.sel(x=scene_x, y=scene_y, method="nearest").values

        elevation = float(elevation.item() if hasattr(elevation, "item") else elevation)

        return elevation

    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Could not query elevation at ({latitude}, {longitude}) -> ({scene_x}, {scene_y}). "
            f"Coordinate may be outside DEM bounds. Error: {e}"
        )
