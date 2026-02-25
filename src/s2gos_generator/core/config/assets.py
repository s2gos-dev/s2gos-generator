from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from s2gos_utils.io.paths import PathRef
from upath import UPath

from ._utils import _resolve_asset_path


class HamsterConfig(BaseModel):
    """HAMSTER albedo data configuration for baresoil material replacement."""

    enabled: bool = Field(True, description="Enable HAMSTER albedo for baresoil")
    data_path: PathRef = Field(..., description="Path to HAMSTER NetCDF data file")
    variable_name: str = Field("albedo", description="Variable name in NetCDF file")
    fallback_on_error: bool = Field(
        True, description="Fall back to standard baresoil material on errors"
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @field_validator("data_path", mode="before")
    @classmethod
    def validate_data_path(cls, v):
        """Validate HAMSTER data file exists."""
        resolved = _resolve_asset_path(v, asset_type="netCDF")
        return resolved


class UserAssets(BaseModel):
    """User assets to be placed on scene.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): coordinate=[lon, lat] with coord_type="geographic"
    - Scene coordinates (meters from scene center): coordinate=[x, y] with coord_type="scene"
    """

    object_id: str = Field(..., description="Unique identifier for the object")
    ply_path: PathRef = Field(
        ..., description="Path to PLY file containing 3D object geometry"
    )

    coordinate: list[float] = Field(
        ..., description="Coordinates: [lon, lat] if geographic, [x, y] if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    material: Union[str, Dict[str, Any]] = Field(
        ...,
        description="Material reference (string ID) or inline material definition dict",
    )
    elevation_offset: float = Field(
        0.0, description="Height offset above terrain surface in meters"
    )
    scale: float = Field(1.0, description="Uniform scaling factor for the object")
    rotation_x: float = Field(0.0, description="Rotation around X-axis in degrees")
    rotation_y: float = Field(0.0, description="Rotation around Y-axis in degrees")
    rotation_z: float = Field(0.0, description="Rotation around Z-axis in degrees")
    blender_fix: bool = Field(
        ...,
        description="Informs as to whether a 90 degree around x was added to adjust from blender, useful to know what the actual rotation intended was",
    )
    face_normals: Optional[bool] = Field(
        None,
        description="Mitsuba PLY face normals setting: True=smooth normals, False=per-face normals, None=use PLY file defaults",
    )
    exclusion_zone: Optional[Union[float, Tuple[float, float]]] = Field(
        None,
        description=(
            "Vegetation exclusion zone centered on this object. Can be:\n"
            "- float: Circular radius in meters\n"
            "- (width, height): Rectangular box in meters"
        ),
    )

    @field_validator("coordinate")
    @classmethod
    def validate_coordinate(cls, v):
        """Validate coordinate format."""
        if len(v) != 2:
            raise ValueError("Coordinate must be [value1, value2]")
        return v

    @field_validator("ply_path", mode="before")
    @classmethod
    def validate_ply_path(cls, v):
        """Validate and resolve PLY file path using configured search paths."""
        resolved = _resolve_asset_path(v, asset_type="PLY mesh")
        return resolved

    @field_validator("material")
    @classmethod
    def validate_material(cls, v):
        """Validate material is either a non-empty string reference or a valid dict definition."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Material reference cannot be empty")
            return v.strip()
        elif isinstance(v, dict):
            if "type" not in v:
                raise ValueError(
                    "Inline material definition must have 'type' field. "
                    "Example: {'type': 'diffuse', 'reflectance': 0.5}"
                )
            return v
        else:
            raise ValueError(
                f"Material must be a string reference or dict definition, got {type(v).__name__}"
            )

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v):
        """Validate scale is positive."""
        if v <= 0:
            raise ValueError("Scale must be positive")
        return v

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        """Ensure coordinate format matches coord_type."""
        if self.coord_type == "geographic":
            lon, lat = self.coordinate
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} out of valid range [-180, 180]")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} out of valid range [-90, 90]")
        elif self.coord_type == "scene":
            pass

        return self

    def get_inline_material_id(self) -> Optional[str]:
        """Return generated material ID for inline materials, None otherwise."""
        if isinstance(self.material, dict):
            return f"{self.object_id}_material"
        return None

    def get_inline_material_dict(self) -> Optional[Dict[str, Any]]:
        """Return inline material definition if present, else None."""
        if isinstance(self.material, dict):
            return self.material
        return None

    model_config = {
        # "arbitrary_types_allowed": True,
        "validate_assignment": True,
        # "extra": "forbid",
    }


class MaterialRegion(BaseModel):
    """Material region for spatially-selective material overrides.

    Defines a spatial region where the terrain material will be overridden with a
    specified material. Supports multiple geometry types and coordinate systems.
    """

    region_id: str = Field(..., description="Unique identifier for this region")

    geometry: Dict[str, Any] = Field(
        ...,
        description="Region geometry specification (rectangle or polygon). "
        "See region_geometry module for details.",
    )

    material_name: str = Field(
        ...,
        description="Material reference to apply in this region (must exist in materials)",
    )

    priority: int = Field(
        0,
        description="Priority for overlapping regions (higher priority wins). "
        "Default is 0.",
    )

    applies_to: List[Literal["target", "buffer", "background"]] = Field(
        ["target"],
        description="Which scene areas this region applies to (target/buffer/background)",
    )

    landcover_filter: Optional[List[int]] = Field(
        None,
        description="Optional ESA WorldCover class filter. If specified, only override "
        "pixels matching these landcover classes. If None, override all pixels. "
        "Example: [60, 80] = only bare/sparse vegetation",
    )

    @field_validator("region_id")
    @classmethod
    def validate_region_id(cls, v):
        """Validate region ID is non-empty and valid for filenames."""
        if not v or not v.strip():
            raise ValueError("region_id cannot be empty")
        # Check for filesystem-safe characters
        if any(char in v for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
            raise ValueError(
                f"region_id '{v}' contains invalid characters for filesystem paths"
            )
        return v.strip()

    @field_validator("material_name")
    @classmethod
    def validate_material_name(cls, v):
        """Validate material reference."""
        if not v or not v.strip():
            raise ValueError("material_name cannot be empty")
        return v.strip()

    @field_validator("applies_to")
    @classmethod
    def validate_applies_to(cls, v):
        """Ensure at least one area is specified."""
        if not v:
            raise ValueError("applies_to must specify at least one area")
        return v

    @field_validator("landcover_filter")
    @classmethod
    def validate_landcover_filter(cls, v):
        """Validate landcover class codes."""
        if v is not None:
            for code in v:
                if not (10 <= code <= 100):
                    raise ValueError(
                        f"Invalid ESA WorldCover class code: {code}. "
                        "Valid range is 10-100 (in steps of 10)"
                    )
        return v


class MaterialMapping(BaseModel):
    """Material mapping for XML assets.

    Maps mesh filenames to material IDs using pattern matching.

    Attributes:
        pattern: Filename pattern to match (without .ply extension)
        material: Material ID to assign to matching meshes
        mode: Pattern matching mode ('glob' or 'regex')
    """

    pattern: str = Field(
        ..., description="Filename pattern to match (e.g., 'vegetation_*' or 'tree_.*')"
    )
    material: str = Field(..., description="Material ID to assign to matching meshes")
    mode: Literal["glob", "regex"] = Field(
        default="glob",
        description="Pattern matching mode: 'glob' for wildcards (* and ?), 'regex' for regular expressions",
    )

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, v: str) -> str:
        """Validate pattern is non-empty."""
        if not v or not v.strip():
            raise ValueError(
                "Pattern cannot be empty.\n"
                "Examples:\n"
                "  - Glob: 'vegetation_*', 'tree_?', '*_ground'\n"
                "  - Regex: r'tree_\\d+', r'(oak|pine)_.*'"
            )
        return v.strip()

    @field_validator("material")
    @classmethod
    def validate_material(cls, v: str) -> str:
        """Validate material reference is non-empty."""
        if not v or not v.strip():
            raise ValueError("Material ID cannot be empty")
        return v.strip()

    model_config = {
        "validate_assignment": True,
        # "extra": "forbid",
    }


class XmlSceneConfig(BaseModel):
    """Configuration for importing assets and materials from XML scene files.

    Coordinates can be specified in either:
    - Geographic coordinates (WGS84): base_coordinate=(lon, lat) with coord_type="geographic"
    - Scene coordinates (meters from scene center): base_coordinate=(x, y) with coord_type="scene"
    """

    xml_path: PathRef = Field(..., description="Path to XML scene file")
    base_coordinate: Tuple[float, float] = Field(
        ..., description="Base coordinates: (lon, lat) if geographic, (x, y) if scene"
    )
    coord_type: Literal["geographic", "scene"] = Field(
        ..., description="Coordinate system type"
    )

    object_id_prefix: Optional[str] = Field(
        None, description="Prefix for asset object IDs"
    )
    elevation_offset: float = Field(
        0.0, description="Global elevation offset for all assets in meters"
    )
    scale: float = Field(
        1.0, gt=0.0, description="Global scaling factor for all assets"
    )
    fix_blender_coords: bool = Field(
        True,
        description="Apply Blender coordinate system correction (90° rotation around X-axis)",
    )
    rotation_x: float = Field(
        0.0,
        description="Global rotation around X-axis in degrees (applied after fix_blender_coords)",
    )
    rotation_y: float = Field(
        0.0,
        description="Global rotation around Y-axis in degrees (applied after fix_blender_coords)",
    )
    rotation_z: float = Field(
        0.0,
        description="Global rotation around Z-axis in degrees (applied after fix_blender_coords)",
    )
    material_mappings: list[MaterialMapping] = Field(
        default_factory=list,
        description="List of material mappings for mesh filename patterns",
    )
    validate_materials: bool = Field(
        True, description="Validate that all materials are properly defined"
    )

    @field_validator("xml_path", mode="before")
    @classmethod
    def validate_xml_path(cls, v):
        """Validate and resolve XML file path using configured search paths."""
        resolved = _resolve_asset_path(v, asset_type="XML scene")
        return resolved

    @field_validator("base_coordinate")
    @classmethod
    def validate_base_coordinate(cls, v):
        """Validate coordinate format."""
        if len(v) != 2:
            raise ValueError("Base coordinate must be (value1, value2)")
        return v

    @model_validator(mode="after")
    def validate_coordinate_format(self):
        """Ensure coordinate format matches coord_type."""
        if self.coord_type == "geographic":
            lon, lat = self.base_coordinate
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} out of valid range [-180, 180]")
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} out of valid range [-90, 90]")

        return self

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        # "extra": "forbid",
    }


def load_assets_from_xml(
    xml_path: str,
    base_coordinate: Union[List[float], Tuple[float, float]],
    coord_type: Literal["geographic", "scene"],
    object_id_prefix: str = "asset",
    elevation_offset: float = 0.0,
    scale: float = 1.0,
    fix_blender_coords: bool = True,
    rotation_x: float = 0.0,
    rotation_y: float = 0.0,
    rotation_z: float = 0.0,
    material_mappings: Optional[List[MaterialMapping]] = None,
    validate_materials: bool = True,
) -> Tuple[List[UserAssets], Dict[str, Dict[str, Any]]]:
    """Load multi-material assets from Mitsuba XML with material library.

    Args:
        xml_path: Path to Mitsuba XML file
        base_coordinate: Base coordinates (lon, lat) or (x, y)
        coord_type: "geographic" or "scene"
        object_id_prefix: Prefix for asset IDs
        elevation_offset: Height offset above terrain (meters)
        scale: Uniform scaling factor
        fix_blender_coords: Apply Blender→Mitsuba coordinate correction (90° X rotation)
        rotation_x: Global rotation around X-axis in degrees (applied after fix_blender_coords)
        rotation_y: Global rotation around Y-axis in degrees (applied after fix_blender_coords)
        rotation_z: Global rotation around Z-axis in degrees (applied after fix_blender_coords)
        material_mappings: List of MaterialMapping objects for pattern-based material assignment
        validate_materials: If True, validate material references and PLY file existence

    Returns:
        Tuple of (assets_list, material_library):
        - assets_list: List of UserAssets with string material references
        - material_library: Dict of material definitions to embed in scene
    """
    from ...assets.xml_importer import import_xml_assets

    material_mappings_dicts = None
    if material_mappings:
        material_mappings_dicts = [
            {
                "pattern": mapping.pattern,
                "material": mapping.material,
                "mode": mapping.mode,
            }
            for mapping in material_mappings
        ]

    asset_data_list, material_library = import_xml_assets(
        xml_path=xml_path,
        base_coordinate=base_coordinate,
        coord_type=coord_type,
        object_id_prefix=object_id_prefix,
        elevation_offset=elevation_offset,
        scale=scale,
        fix_blender_coords=fix_blender_coords,
        rotation_x=rotation_x,
        rotation_y=rotation_y,
        rotation_z=rotation_z,
        material_mappings=material_mappings_dicts,
        validate_materials=validate_materials,
    )

    assets = []
    for asset_data in asset_data_list:
        asset_kwargs = {
            "object_id": asset_data["object_id"],
            "ply_path": UPath(asset_data["ply_path"]),
            "material": asset_data["material"],
            "elevation_offset": asset_data["elevation_offset"],
            "scale": asset_data["scale"],
            "rotation_x": asset_data["rotation_x"],
            "rotation_y": asset_data["rotation_y"],
            "rotation_z": asset_data["rotation_z"],
            "blender_fix": asset_data.get(
                "blender_fix", fix_blender_coords
            ),  # Use value from asset_data, fall back to parameter
            "coordinate": base_coordinate,
            "coord_type": coord_type,
        }

        if "face_normals" in asset_data:
            asset_kwargs["face_normals"] = asset_data["face_normals"]

        asset = UserAssets(**asset_kwargs)
        assets.append(asset)

    return assets, material_library
