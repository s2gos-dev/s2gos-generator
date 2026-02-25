from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils.io.paths import PathRef

from ._utils import _resolve_asset_path


class CircleGeometry(BaseModel):
    """Circular geometry definition for vegetation exclusion zones.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): center=(lon, lat) with coord_type="geographic"
    - Scene coordinates (meters from scene center): center=(x, y) with coord_type="scene"
    """

    type: Literal["circle"] = "circle"

    center: Tuple[float, float] = Field(
        ..., description="Center: (lon, lat) if geographic, (x, y) if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    radius: float = Field(..., gt=0, description="Radius in meters")

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


class BoxGeometry(BaseModel):
    """Rectangular box geometry definition for vegetation exclusion zones.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): center=(lon, lat) with coord_type="geographic"
    - Scene coordinates (meters from scene center): center=(x, y) with coord_type="scene"
    """

    type: Literal["box"] = "box"

    center: Tuple[float, float] = Field(
        ..., description="Center: (lon, lat) if geographic, (x, y) if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    width: float = Field(..., gt=0, description="Width in meters (east-west)")
    height: float = Field(..., gt=0, description="Height in meters (north-south)")

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


class PolygonGeometry(BaseModel):
    """Polygon geometry definition for vegetation exclusion zones.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): coordinates=[(lon, lat), ...] with coord_type="geographic"
    - Scene coordinates (meters from scene center): coordinates=[(x, y), ...] with coord_type="scene"
    """

    type: Literal["polygon"] = "polygon"

    coordinates: List[Tuple[float, float]] = Field(
        ...,
        min_length=3,
        description="Vertices: [(lon, lat), ...] if geographic, [(x, y), ...] if scene. Min 3 vertices.",
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        """Ensure coordinate format matches coord_type and vertex count."""
        if len(self.coordinates) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        if self.coord_type == "geographic":
            for i, (lon, lat) in enumerate(self.coordinates):
                if not (-180 <= lon <= 180):
                    raise ValueError(
                        f"Vertex {i}: Longitude {lon} out of range [-180, 180]"
                    )
                if not (-90 <= lat <= 90):
                    raise ValueError(
                        f"Vertex {i}: Latitude {lat} out of range [-90, 90]"
                    )

        return self


class VegetationExclusionZone(BaseModel):
    """Standalone vegetation exclusion zone not tied to objects.

    Defines a geographic area where vegetation placement is disabled.
    Geometry can be a circle, box, or arbitrary polygon.
    """

    zone_id: str = Field(..., description="Unique identifier for this exclusion zone")
    geometry: Union[CircleGeometry, BoxGeometry, PolygonGeometry] = Field(
        ...,
        discriminator="type",
        description="Zone geometry",
    )


class VegetationSpecies(BaseModel):
    """Configuration for a single vegetation species.

    Defines placement parameters for a vegetation type (e.g., oak trees, shrubs).
    Multiple species can be assigned to the same landcover class for mixed vegetation.
    Instances are placed within pixels of the matched landcover class in the target area.
    """

    name: str = Field(
        description="Species identifier (e.g., 'oak_trees', 'berry_bushes')"
    )
    asset_xml_paths: Union[List[PathRef], Dict[PathRef, float]] = Field(
        description="Asset XML file path(s). Use list for uniform distribution or dict for weighted distribution"
    )
    density_per_hectare: float = Field(
        ge=0.0, le=4000.0, description="Density for this species"
    )
    scale_min: float = Field(ge=0.1, description="Minimum scale factor")
    scale_max: float = Field(ge=0.1, description="Maximum scale factor")
    spillover_enabled: bool = Field(
        False, description="Enable spillover into adjacent compatible landcover classes"
    )
    spillover_compatibility: Optional[Dict[int, float]] = Field(
        None,
        description="Per-species spillover compatibility map (overrides global). Maps landcover class to probability 0.0-1.0",
    )

    @field_validator("scale_max")
    @classmethod
    def validate_scale_range(cls, v, info):
        """Ensure scale_max > scale_min."""
        if "scale_min" in info.data and v <= info.data["scale_min"]:
            raise ValueError("scale_max must be greater than scale_min")
        return v

    @field_validator("asset_xml_paths")
    @classmethod
    def validate_asset_paths(cls, v):
        """Validate asset paths and weights."""
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("asset_xml_paths list cannot be empty")
        elif isinstance(v, dict):
            if len(v) == 0:
                raise ValueError("asset_xml_paths dict cannot be empty")
            for path, weight in v.items():
                if weight <= 0:
                    raise ValueError(f"Weight must be positive for {path}: {weight}")
        return v

    @model_validator(mode="after")
    def resolve_asset_paths(self):
        """Resolve all asset XML paths using configured search paths.

        This ensures all vegetation assets can be found before scene generation,
        providing fail-fast behavior with clear error messages.
        """
        if isinstance(self.asset_xml_paths, list):
            resolved_paths = []
            for path in self.asset_xml_paths:
                resolved = _resolve_asset_path(path, asset_type="vegetation XML")
                resolved_paths.append(resolved)
            # Use object.__setattr__ to avoid triggering validate_assignment recursion
            object.__setattr__(self, "asset_xml_paths", resolved_paths)
        elif isinstance(self.asset_xml_paths, dict):
            resolved_dict = {}
            for path, weight in self.asset_xml_paths.items():
                resolved = _resolve_asset_path(path, asset_type="vegetation XML")
                resolved_dict[resolved] = weight
            # Use object.__setattr__ to avoid triggering validate_assignment recursion
            object.__setattr__(self, "asset_xml_paths", resolved_dict)

        return self

    def get_asset_paths_and_weights(self) -> Tuple[List[str], List[float]]:
        """Get asset paths and normalized weights for selection.

        Returns:
            (paths, weights) tuple ready for random.choices()
        """
        if isinstance(self.asset_xml_paths, list):
            return (self.asset_xml_paths, [1.0] * len(self.asset_xml_paths))
        else:
            paths = list(self.asset_xml_paths.keys())
            weights = list(self.asset_xml_paths.values())
            return (paths, weights)

    model_config = {
        "validate_assignment": True,
    }


class VegetationPlacementConfig(BaseModel):
    """Configuration for multi-species vegetation placement system.

    Controls how vegetation instances are distributed across the scene based on
    landcover classifications. Supports multiple species per landcover class.

    Configuration levels:
    - Per-species parameters: density, scale, asset (in VegetationSpecies)
    - Global parameters: spacing, variation, limits (this class)
    """

    enabled: bool = Field(
        True, description="Enable vegetation placement based on landcover data"
    )

    landcover_species_mapping: Dict[int, List[VegetationSpecies]] = Field(
        default_factory=lambda: {
            10: [
                VegetationSpecies(
                    name="oak_trees",
                    asset_xml_path="tree.xml",
                    density_per_hectare=400.0,
                    scale_min=10.0,
                    scale_max=35.0,
                )
            ]
        },
        description="Mapping from landcover class to list of vegetation species",
    )

    min_spacing: float = Field(
        2.0,
        ge=0.1,
        description="Global minimum spacing between any vegetation instances (meters)",
    )
    density_variation: float = Field(
        0.3, ge=0.0, le=1.0, description="Random variation in density (±30% by default)"
    )
    max_instances_per_pixel: int = Field(
        50, ge=1, le=10000, description="Performance limit per pixel across all species"
    )
    rotation_range: float = Field(
        360.0,
        ge=0.0,
        le=360.0,
        description="Random rotation range in degrees (azimuth around vertical axis)",
    )
    tilt_range: float = Field(
        6.0,
        ge=0.0,
        le=23.0,
        description="Random tilt range in degrees (±deviation from vertical for natural variation)",
    )
    spillover_max_distance_m: float = Field(
        30.0,
        ge=0.0,
        le=300.0,
        description="Maximum distance (meters) for spillover from primary landcover class",
    )
    spillover_compatibility: Dict[int, float] = Field(
        default_factory=lambda: {
            20: 0.8,  # Shrubland - high compatibility
            30: 0.7,  # Grassland - moderate compatibility
            40: 0.3,  # Cropland - low compatibility
            90: 0.4,  # Herbaceous Wetland - moderate compatibility
            60: 0.1,  # Bare/sparse vegetation - very low compatibility
        },
        description="Default spillover compatibility map. Maps landcover class to probability 0.0-1.0. Can be overridden per species.",
    )

    model_config = {
        "validate_assignment": True,
    }
