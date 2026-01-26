import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.io.paths import open_dataset
from upath import UPath

PERMANENT_WATER_MATERIAL_INDEX = 7
UNKNOWN_MATERIAL_PREVIEW_VALUE = -1
GRAY_COLOR = (128, 128, 128)
MAX_PIXEL_VALUE = 255

DEFAULT_MATERIALS = [
    {
        "name": "Tree cover",
        "esa_class": 10,
        "color_8bit": (40, 75, 30),
        "roughness": 0.6,
    },
    {
        "name": "Shrubland",
        "esa_class": 20,
        "color_8bit": (185, 170, 130),
        "roughness": 0.7,
    },
    {
        "name": "Grassland",
        "esa_class": 30,
        "color_8bit": (140, 155, 95),
        "roughness": 0.7,
    },
    {
        "name": "Cropland",
        "esa_class": 40,
        "color_8bit": (240, 150, 255),
        "roughness": 0.6,
    },
    {
        "name": "Built-up",
        "esa_class": 50,
        "color_8bit": (150, 150, 150),
        "roughness": 0.3,
    },
    {
        "name": "Bare / sparse vegetation",
        "esa_class": 60,
        "color_8bit": (220, 140, 90),
        "roughness": 0.8,
    },
    {
        "name": "Snow and ice",
        "esa_class": 70,
        "color_8bit": (240, 240, 240),
        "roughness": 0.2,
    },
    {
        "name": "Permanent water bodies",
        "esa_class": 80,
        "color_8bit": (0, 100, 200),
        "roughness": 0.1,
    },
    {
        "name": "Herbaceous wetland",
        "esa_class": 90,
        "color_8bit": (80, 120, 90),
        "roughness": 0.4,
    },
    {
        "name": "Mangroves",
        "esa_class": 95,
        "color_8bit": (0, 207, 117),
        "roughness": 0.4,
    },
    {
        "name": "Moss and lichen",
        "esa_class": 100,
        "color_8bit": (250, 230, 160),
        "roughness": 0.8,
    },
]


class TerrainMaterialGenerator:
    """
    Generates terrain material selection textures from land cover data for use in 3D rendering.
    """

    def __init__(self, materials: Optional[List[Dict]] = None):
        """
        Initialize the terrain material generator.

        Args:
            materials: List of material definitions. If None, uses default materials.
        """
        self.materials = materials if materials is not None else DEFAULT_MATERIALS
        self.class_to_index = {
            mat["esa_class"]: idx for idx, mat in enumerate(self.materials)
        }

    def landcover_to_selection_texture(
        self,
        landcover_data: xr.DataArray,
        output_path: UPath,
        flip_vertical: bool = False,
        default_material_index: int = PERMANENT_WATER_MATERIAL_INDEX,
        dem_data: Optional[xr.DataArray] = None,
        season_month: Optional[str] = None,
        snow_material_index: Optional[int] = None,
        coordinate_system=None,
        snow_thermoprops: Optional[UPath] = None,
    ) -> np.ndarray:
        """
        Converts land cover classification data to a material selection texture.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            output_path: UPath where the texture PNG will be saved.
            flip_vertical: If True, flips the texture vertically (for Mitsuba compatibility).
            default_material_index: Material index to use for unknown classes.
            dem_data: Optional DEM DataArray for seasonal snow adjustment.
            season_month: Optional month name for seasonal snow adjustment.
            snow_material_index: Optional material index to use for snow.
            coordinate_system: Optional CoordinateSystem for scene-to-latlon conversion.
            snow_thermoprops: Optional path to CAMS thermoprops NetCDF file.

        Returns:
            The selection texture as a numpy array.
        """
        landcover_data.load()
        class_values = landcover_data.values

        if np.any(np.isnan(class_values)):
            nan_count = np.sum(np.isnan(class_values))
            logging.info(
                f"Found {nan_count} NaN values in landcover data, replacing with default material index {default_material_index}"
            )
            class_values = np.where(
                np.isnan(class_values), default_material_index, class_values
            )

        class_values = np.nan_to_num(class_values, nan=default_material_index).astype(
            np.uint8
        )

        selection_texture = np.full_like(
            class_values, default_material_index, dtype=np.uint8
        )

        for esa_class, material_index in self.class_to_index.items():
            mask = class_values == esa_class
            selection_texture[mask] = material_index

        # Apply seasonal snow adjustment if requested
        if (
            dem_data is not None
            and season_month is not None
            and snow_material_index is not None
        ):
            if coordinate_system is None:
                logging.warning(
                    "Seasonal snow requested but coordinate_system not provided, skipping snow adjustment"
                )
            else:
                selection_texture = self._apply_seasonal_snow(
                    selection_texture=selection_texture,
                    landcover_data=landcover_data,
                    dem_data=dem_data,
                    season_month=season_month,
                    snow_material_index=snow_material_index,
                    coordinate_system=coordinate_system,
                    snow_thermoprops=snow_thermoprops,
                )

        if flip_vertical:
            selection_texture = np.flipud(selection_texture)

        self._save_selection_texture(selection_texture, output_path)

        logging.info(f"Texture: {output_path}")
        return selection_texture

    def create_preview_texture(
        self,
        landcover_data: xr.DataArray,
        output_path: UPath,
        flip_vertical: bool = True,
    ) -> np.ndarray:
        """
        Creates a color preview texture showing the actual material colors.

        Args:
            landcover_data: xarray DataArray containing land cover class values.
            output_path: UPath where the preview PNG will be saved.
            flip_vertical: If True, flips the texture vertically.

        Returns:
            The preview texture as a numpy array with shape (height, width, 3).
        """
        landcover_data.load()
        class_values = landcover_data.values

        if np.any(np.isnan(class_values)):
            nan_count = np.sum(np.isnan(class_values))
            logging.info(
                f"Found {nan_count} NaN values in preview texture, using gray {GRAY_COLOR} for unknown areas"
            )
            class_values = np.where(
                np.isnan(class_values), UNKNOWN_MATERIAL_PREVIEW_VALUE, class_values
            )

        class_values = np.nan_to_num(
            class_values, nan=UNKNOWN_MATERIAL_PREVIEW_VALUE
        ).astype(np.int32)

        height, width = class_values.shape
        color_texture = np.zeros((height, width, 3), dtype=np.uint8)

        for material in self.materials:
            esa_class = material["esa_class"]
            color = material["color_8bit"]
            mask = class_values == esa_class
            color_texture[mask] = color

        known_classes = set(mat["esa_class"] for mat in self.materials)
        unknown_mask = ~np.isin(class_values, list(known_classes))
        color_texture[unknown_mask] = GRAY_COLOR

        if flip_vertical:
            color_texture = np.flipud(color_texture)

        self._save_color_texture(color_texture, output_path)

        return color_texture

    def _save_selection_texture(self, texture: np.ndarray, output_path: UPath) -> None:
        """Save selection texture as a grayscale PNG."""
        from s2gos_utils.io.paths import mkdir

        if np.any(np.isnan(texture)) or np.any(np.isinf(texture)):
            logging.warning(
                "Found NaN/inf values in selection texture before saving, cleaning..."
            )
            texture = np.nan_to_num(
                texture,
                nan=PERMANENT_WATER_MATERIAL_INDEX,
                posinf=MAX_PIXEL_VALUE,
                neginf=0,
            ).astype(np.uint8)

        if texture.dtype != np.uint8:
            texture = np.clip(texture, 0, MAX_PIXEL_VALUE).astype(np.uint8)

        mkdir(output_path.parent)
        image = Image.fromarray(texture, mode="L")
        image.save(output_path)

    def _save_color_texture(self, texture: np.ndarray, output_path: UPath) -> None:
        """Save color texture as RGB PNG."""
        from s2gos_utils.io.paths import mkdir

        if np.any(np.isnan(texture)) or np.any(np.isinf(texture)):
            logging.warning(
                "Found NaN/inf values in color texture before saving, cleaning..."
            )
            texture = np.nan_to_num(
                texture, nan=GRAY_COLOR[0], posinf=MAX_PIXEL_VALUE, neginf=0
            ).astype(np.uint8)

        if texture.dtype != np.uint8:
            texture = np.clip(texture, 0, MAX_PIXEL_VALUE).astype(np.uint8)

        mkdir(output_path.parent)
        image = Image.fromarray(texture, mode="RGB")
        image.save(output_path)

    def get_material_info(self) -> Dict:
        """
        Returns information about the configured materials.

        Returns:
            Dictionary containing material configuration details.
        """
        return {
            "num_materials": len(self.materials),
            "materials": self.materials,
            "class_mapping": self.class_to_index,
        }

    def _apply_seasonal_snow(
        self,
        selection_texture: np.ndarray,
        landcover_data: xr.DataArray,
        dem_data: xr.DataArray,
        season_month: str,
        snow_material_index: int,
        coordinate_system,
        snow_thermoprops: Optional[UPath] = None,
    ) -> np.ndarray:
        """
        Apply seasonal snow using temperature-based probability model.

        Args:
            selection_texture: Base material selection texture.
            landcover_data: Land cover DataArray with coordinates.
            dem_data: DEM DataArray with elevation values.
            season_month: "june" or "december".
            snow_material_index: Material index for snow.
            coordinate_system: CoordinateSystem for scene-to-latlon conversion.
            snow_thermoprops: Optional path to CAMS thermoprops NetCDF file.
                             If None, uses synthetic temperature model.

        Returns:
            Snow-adjusted selection texture.
        """
        from ..seasonal.snow import (
            Month,
            calculate_snow_probability_map,
            get_day_of_year,
        )

        y_coords = landcover_data.coords["y"].values  # meters (scene Y)
        x_coords = landcover_data.coords["x"].values  # meters (scene X)

        y_grid_scene, x_grid_scene = np.meshgrid(y_coords, x_coords, indexing="ij")

        center_x = coordinate_system._center_x
        center_y = coordinate_system._center_y

        x_absolute = x_grid_scene + center_x
        y_absolute = y_grid_scene + center_y

        lon_grid, lat_grid = coordinate_system._from_scene_transformer.transform(
            x_absolute, y_absolute
        )

        elevation_grid = dem_data.values

        month_enum = Month(season_month.lower())
        day_of_year = get_day_of_year(month_enum)

        thermoprops_dataset = None
        if snow_thermoprops is not None:
            try:
                thermoprops_dataset = open_dataset(snow_thermoprops).squeeze(drop=True)
                if "t" not in thermoprops_dataset.data_vars:
                    raise ValueError("Missing required variable 't' (temperature)")
                if (
                    "z" not in thermoprops_dataset.coords
                    and "z" not in thermoprops_dataset.data_vars
                ):
                    raise ValueError(
                        "Missing required coordinate/variable 'z' (height)"
                    )

                z_min = float(thermoprops_dataset["z"].min())
                z_max = float(thermoprops_dataset["z"].max())
                logging.info(
                    f"Using CAMS temperature profile: z={z_min:.1f}-{z_max:.1f} km, "
                    f"{len(thermoprops_dataset['z'])} levels"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to load CAMS thermoprops file '{snow_thermoprops}': {e}\n"
                    f"Check that file exists and contains 't' (temperature) and 'z' (height)."
                ) from e

        snow_probs, temps = calculate_snow_probability_map(
            latitudes=lat_grid,
            elevations=elevation_grid,
            day_of_year=day_of_year,
            smooth_sigma=10.0,
            thermoprops=thermoprops_dataset,
        )

        random_field = np.random.uniform(0, 1, snow_probs.shape)
        snow_mask = snow_probs > random_field

        snow_adjusted = selection_texture.copy()
        snow_adjusted[snow_mask] = snow_material_index

        snow_coverage = snow_mask.mean() * 100
        logging.info(
            f"Seasonal snow ({season_month}): {snow_coverage:.1f}% coverage, "
            f"temperature {temps.min():.1f}°C to {temps.max():.1f}°C, "
            f"{snow_mask.sum()} pixels modified"
        )

        return snow_adjusted

    def generate_textures_from_file(
        self,
        landcover_file_path: UPath,
        output_dir: UPath,
        base_name: str,
        create_preview: bool = True,
        dem_file_path: Optional[UPath] = None,
        season_month: Optional[str] = None,
        snow_material_index: Optional[int] = None,
        coordinate_system=None,
        snow_thermoprops: Optional[UPath] = None,
    ) -> Tuple[UPath, Optional[UPath]]:
        """
        Complete pipeline: loads land cover from file and generates textures.

        Args:
            landcover_file_path: UPath to the land cover NetCDF file.
            output_dir: Directory where textures will be saved.
            base_name: Base name for output files.
            create_preview: Whether to create a color preview texture.
            dem_file_path: Optional UPath to DEM file for seasonal snow adjustment.
            season_month: Optional month name for seasonal snow adjustment.
            snow_material_index: Optional material index for snow.
            coordinate_system: Optional CoordinateSystem for scene-to-latlon conversion.
            snow_thermoprops: Optional path to CAMS thermoprops NetCDF file.

        Returns:
            Tuple of (selection_texture_path, preview_texture_path).
        """

        landcover_dataset = xr.open_zarr(landcover_file_path)
        landcover_data = landcover_dataset["landcover"]

        if isinstance(landcover_data, xr.Dataset):
            if "landcover" in landcover_data.data_vars:
                landcover_data = landcover_data["landcover"]
            else:
                landcover_data = landcover_data[
                    list(landcover_data.data_vars.keys())[0]
                ]

        # Load DEM if provided for seasonal snow adjustment
        dem_data = None
        if dem_file_path is not None:
            dem_dataset = xr.open_zarr(dem_file_path)
            dem_data = dem_dataset["elevation"]

        selection_path = output_dir / f"{base_name}_selection.png"
        preview_path = (
            output_dir / f"{base_name}_preview.png" if create_preview else None
        )

        # Generate selection texture with optional snow adjustment
        self.landcover_to_selection_texture(
            landcover_data,
            selection_path,
            dem_data=dem_data,
            season_month=season_month,
            snow_material_index=snow_material_index,
            coordinate_system=coordinate_system,
            snow_thermoprops=snow_thermoprops,
        )

        # Preview texture shows base landcover (no snow adjustment)
        if create_preview:
            self.create_preview_texture(landcover_data, preview_path)

        return selection_path, preview_path
