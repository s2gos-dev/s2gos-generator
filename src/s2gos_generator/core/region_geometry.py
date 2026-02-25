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
    - Geographic coordinates (WGS84): center=(lon, lat) with coord_type="geographic"
    - Scene coordinates (meters from scene center): center=(x, y) with coord_type="scene"
    """

    geometry_type: Literal["rectangle"] = "rectangle"

    center: Tuple[float, float] = Field(
        ..., description="Center: (lon, lat) if geographic, (x, y) if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    # Dimensions (always in meters)
    width_m: float = Field(..., description="Rectangle width in meters", gt=0)
    height_m: float = Field(..., description="Rectangle height in meters", gt=0)

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        """Ensure coordinate format matches coord_type."""
        if self.coord_type == "geographic":
            lon, lat = self.center
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} out of valid range [-180, 180]")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} out of valid range [-90, 90]")

        return self

    def _get_scene_center(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> Tuple[float, float]:
        """Get rectangle center in scene coordinates (meters)."""
        if self.coord_type == "scene":
            return self.center
        else:
            if coordinate_system is None:
                raise ValueError(
                    "CoordinateSystem required to convert geographic coordinates"
                )
            lon, lat = self.center
            return coordinate_system.latlon_to_scene(lat, lon)

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
    - Geographic coordinates (WGS84): vertices=[(lon, lat), ...] with coord_type="geographic"
    - Scene coordinates (meters): vertices=[(x, y), ...] with coord_type="scene"

    The polygon is automatically closed (first and last points connected).
    """

    geometry_type: Literal["polygon"] = "polygon"

    vertices: list[Tuple[float, float]] = Field(
        ...,
        description="Vertices: [(lon, lat), ...] if geographic, [(x, y), ...] if scene",
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    @model_validator(mode="after")
    def validate_vertices(self):
        """Ensure vertices meet requirements."""
        if len(self.vertices) < 3:
            raise ValueError(
                f"Polygon requires at least 3 vertices, got {len(self.vertices)}"
            )

        if self.coord_type == "geographic":
            for i, (lon, lat) in enumerate(self.vertices):
                if not (-180 <= lon <= 180):
                    raise ValueError(
                        f"Vertex {i}: Longitude {lon} out of range [-180, 180]"
                    )
                if not (-90 <= lat <= 90):
                    raise ValueError(
                        f"Vertex {i}: Latitude {lat} out of range [-90, 90]"
                    )

        return self

    def _get_scene_vertices(
        self, coordinate_system: Optional[CoordinateSystem] = None
    ) -> list[Tuple[float, float]]:
        """Get polygon vertices in scene coordinates (meters)."""
        if self.coord_type == "scene":
            return self.vertices
        else:
            if coordinate_system is None:
                raise ValueError(
                    "CoordinateSystem required to convert geographic coordinates"
                )
            scene_vertices = []
            for lon, lat in self.vertices:
                x, y = coordinate_system.latlon_to_scene(lat, lon)
                scene_vertices.append((x, y))
            return scene_vertices

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
