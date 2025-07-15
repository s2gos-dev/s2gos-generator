"""Geometry utilities for S2GOS scene generation."""

from pyproj import CRS, Transformer
from shapely.geometry import Polygon


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
