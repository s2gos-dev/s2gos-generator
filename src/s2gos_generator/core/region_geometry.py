"""Geometry classes for defining material regions in scenes.

This module provides classes for defining spatial regions where materials can be overridden.
Supports multiple geometry types (rectangles, polygons) and flexible coordinate specifications
(scene coordinates or geographic coordinates).
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, model_validator
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from s2gos_utils.coordinates import CoordinateSystem
from shapely.geometry import Polygon as ShapelyPolygon


class RegionGeometry(BaseModel, ABC):
    """Abstract base class for region geometry definitions.

    All geometry types must implement the to_mask() method to generate binary masks
    at a specified resolution.
    """

    geometry_type: str = Field(
        ..., description="Type of geometry (rectangle, polygon, etc.)"
    )

    @abstractmethod
    def to_mask(
        self,
        width_px: int,
        height_px: int,
        scene_bounds: dict[str, float],
        coordinate_system: Optional[CoordinateSystem] = None,
    ) -> np.ndarray:
        """Generate binary mask for this region.

        Args:
            width_px: Output mask width in pixels
            height_px: Output mask height in pixels
            scene_bounds: Scene bounds dict with 'xmin', 'xmax', 'ymin', 'ymax' in meters
            coordinate_system: Optional coordinate system for lat/lon conversion

        Returns:
            Binary mask array (0 or 255) with shape (height_px, width_px)
        """
        pass

    @abstractmethod
    def get_bounds(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> dict[str, float]:
        """Get bounding box of this region in scene coordinates (meters).

        Args:
            coordinate_system: Optional coordinate system for lat/lon conversion

        Returns:
            Dictionary with 'xmin', 'xmax', 'ymin', 'ymax' in meters
        """
        pass

    class Config:
        arbitrary_types_allowed = True


class RectangleGeometry(RegionGeometry):
    """Rectangular region defined by center point and dimensions.

    Coordinates can be specified in either:
    - Scene coordinates (meters from scene center): Use center_x, center_y
    - Geographic coordinates (WGS84): Use center_lat, center_lon

    The system will auto-detect based on which fields are provided.
    """

    geometry_type: Literal["rectangle"] = "rectangle"

    # Scene coordinates (meters from scene center)
    center_x: Optional[float] = Field(
        None,
        description="X coordinate of rectangle center in scene coordinates (meters)",
    )
    center_y: Optional[float] = Field(
        None,
        description="Y coordinate of rectangle center in scene coordinates (meters)",
    )

    # Geographic coordinates (WGS84)
    center_lat: Optional[float] = Field(
        None, description="Latitude of rectangle center in decimal degrees"
    )
    center_lon: Optional[float] = Field(
        None, description="Longitude of rectangle center in decimal degrees"
    )

    # Dimensions (always in meters)
    width_m: float = Field(..., description="Rectangle width in meters", gt=0)
    height_m: float = Field(..., description="Rectangle height in meters", gt=0)

    @model_validator(mode="after")
    def validate_coordinates(self):
        """Ensure exactly one coordinate system is specified."""
        has_scene_coords = self.center_x is not None and self.center_y is not None
        has_geo_coords = self.center_lat is not None and self.center_lon is not None

        if not has_scene_coords and not has_geo_coords:
            raise ValueError(
                "Rectangle requires either scene coordinates (center_x, center_y) "
                "or geographic coordinates (center_lat, center_lon)"
            )

        if has_scene_coords and has_geo_coords:
            raise ValueError(
                "Rectangle cannot specify both scene and geographic coordinates. "
                "Use either (center_x, center_y) or (center_lat, center_lon)"
            )

        return self

    def _get_scene_center(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> Tuple[float, float]:
        """Get rectangle center in scene coordinates (meters)."""
        if self.center_x is not None and self.center_y is not None:
            return (self.center_x, self.center_y)
        elif self.center_lat is not None and self.center_lon is not None:
            if coordinate_system is None:
                raise ValueError(
                    "CoordinateSystem required to convert geographic coordinates"
                )
            return coordinate_system.latlon_to_scene(self.center_lat, self.center_lon)
        else:
            raise ValueError("Invalid coordinate specification")

    def get_bounds(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> dict[str, float]:
        """Get bounding box in scene coordinates."""
        center_x, center_y = self._get_scene_center(coordinate_system)

        half_width = self.width_m / 2
        half_height = self.height_m / 2

        return {
            "xmin": center_x - half_width,
            "xmax": center_x + half_width,
            "ymin": center_y - half_height,
            "ymax": center_y + half_height,
        }

    def to_mask(
        self,
        width_px: int,
        height_px: int,
        scene_bounds: dict[str, float],
        coordinate_system: Optional[CoordinateSystem] = None,
    ) -> np.ndarray:
        """Generate binary mask for rectangular region."""
        # Get region bounds in scene coordinates
        region_bounds = self.get_bounds(coordinate_system)

        # Calculate pixel resolution
        pixel_width_m = (scene_bounds["xmax"] - scene_bounds["xmin"]) / width_px
        pixel_height_m = (scene_bounds["ymax"] - scene_bounds["ymin"]) / height_px

        # Create coordinate arrays for pixels (centers)
        x_coords = np.linspace(
            scene_bounds["xmin"] + pixel_width_m / 2,
            scene_bounds["xmax"] - pixel_width_m / 2,
            width_px,
        )
        y_coords = np.linspace(
            scene_bounds["ymin"] + pixel_height_m / 2,
            scene_bounds["ymax"] - pixel_height_m / 2,
            height_px,
        )

        # Create 2D coordinate grids
        X, Y = np.meshgrid(x_coords, y_coords)

        # Check which pixels are inside the rectangle
        inside_x = (X >= region_bounds["xmin"]) & (X <= region_bounds["xmax"])
        inside_y = (Y >= region_bounds["ymin"]) & (Y <= region_bounds["ymax"])
        mask = inside_x & inside_y

        # Convert to uint8 (0 or 255)
        return (mask * 255).astype(np.uint8)


class PolygonGeometry(RegionGeometry):
    """Polygonal region defined by vertices.

    Vertices can be specified in either:
    - Scene coordinates (meters): vertices_xy = [(x1, y1), (x2, y2), ...]
    - Geographic coordinates (WGS84): vertices_latlon = [(lat1, lon1), (lat2, lon2), ...]

    The polygon is automatically closed (first and last points connected).
    """

    geometry_type: Literal["polygon"] = "polygon"

    # Scene coordinates (meters from scene center)
    vertices_xy: Optional[list[Tuple[float, float]]] = Field(
        None,
        description="Polygon vertices as (x, y) tuples in scene coordinates (meters)",
    )

    # Geographic coordinates (WGS84)
    vertices_latlon: Optional[list[Tuple[float, float]]] = Field(
        None,
        description="Polygon vertices as (lat, lon) tuples in decimal degrees",
    )

    @model_validator(mode="after")
    def validate_vertices(self):
        """Ensure exactly one vertex specification is provided."""
        has_scene = self.vertices_xy is not None
        has_geo = self.vertices_latlon is not None

        if not has_scene and not has_geo:
            raise ValueError("Polygon requires either vertices_xy or vertices_latlon")

        if has_scene and has_geo:
            raise ValueError(
                "Polygon cannot specify both scene and geographic vertices. "
                "Use either vertices_xy or vertices_latlon"
            )

        # Check minimum vertices
        vertices = self.vertices_xy if has_scene else self.vertices_latlon
        if len(vertices) < 3:
            raise ValueError(
                f"Polygon requires at least 3 vertices, got {len(vertices)}"
            )

        return self

    def _get_scene_vertices(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> list[Tuple[float, float]]:
        """Get polygon vertices in scene coordinates (meters)."""
        if self.vertices_xy is not None:
            return self.vertices_xy
        elif self.vertices_latlon is not None:
            if coordinate_system is None:
                raise ValueError(
                    "CoordinateSystem required to convert geographic coordinates"
                )
            scene_vertices = []
            for lat, lon in self.vertices_latlon:
                x, y = coordinate_system.latlon_to_scene(lat, lon)
                scene_vertices.append((x, y))
            return scene_vertices
        else:
            raise ValueError("Invalid vertex specification")

    def get_bounds(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> dict[str, float]:
        """Get bounding box of polygon in scene coordinates."""
        vertices = self._get_scene_vertices(coordinate_system)

        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]

        return {
            "xmin": min(x_coords),
            "xmax": max(x_coords),
            "ymin": min(y_coords),
            "ymax": max(y_coords),
        }

    def to_mask(
        self,
        width_px: int,
        height_px: int,
        scene_bounds: dict[str, float],
        coordinate_system: Optional[CoordinateSystem] = None,
    ) -> np.ndarray:
        """Generate binary mask for polygonal region.

        Args:
            width_px: Output mask width in pixels
            height_px: Output mask height in pixels
            scene_bounds: Scene bounds dict with 'xmin', 'xmax', 'ymin', 'ymax' in meters
            coordinate_system: Optional coordinate system for lat/lon conversion

        Returns:
            Binary mask array (0 or 255) with shape (height_px, width_px)
        """
        vertices = self._get_scene_vertices(coordinate_system)

        polygon = ShapelyPolygon(vertices)

        transform = from_bounds(
            scene_bounds["xmin"],
            scene_bounds["ymin"],
            scene_bounds["xmax"],
            scene_bounds["ymax"],
            width_px,
            height_px,
        )

        mask = rasterize(
            [(polygon, 255)],
            out_shape=(height_px, width_px),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )

        return mask


# Type alias for any geometry type
AnyGeometry = Union[RectangleGeometry, PolygonGeometry]


def geometry_from_dict(data: dict) -> AnyGeometry:
    """Create geometry object from dictionary.

    Args:
        data: Dictionary with 'geometry_type' field and type-specific parameters

    Returns:
        Geometry object of appropriate type

    Raises:
        ValueError: If geometry_type is unknown

    Example:
        >>> geom = geometry_from_dict({
        ...     'geometry_type': 'rectangle',
        ...     'center_x': 0,
        ...     'center_y': 0,
        ...     'width_m': 1000,
        ...     'height_m': 1000
        ... })
        >>> isinstance(geom, RectangleGeometry)
        True
    """
    geom_type = data.get("geometry_type")

    if geom_type == "rectangle":
        return RectangleGeometry(**data)
    elif geom_type == "polygon":
        return PolygonGeometry(**data)
    else:
        raise ValueError(
            f"Unknown geometry type: {geom_type}. Supported types: rectangle, polygon"
        )
