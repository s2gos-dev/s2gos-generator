"""Texture generation resources."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from PIL import Image

from ..assets.terrain_material import TerrainMaterialGenerator
from ..core.context import SceneResourceContext


def _apply_region_materials_to_texture(
    texture_path: Path,
    landcover_path: Path,
    applicable_regions: list,
    ctx: SceneResourceContext,
    area_name: str = "texture",
) -> None:
    """Apply material region overlays to texture file.

    This is a helper function that extracts common logic from generate_target_texture(),
    generate_buffer_texture(), and generate_background_texture().

    Args:
        texture_path: Path to selection texture PNG file (will be modified in-place)
        landcover_path: Path to landcover zarr file
        applicable_regions: List of MaterialRegion configs to apply
        ctx: Scene resource context
        area_name: Name for logging (e.g., "target texture", "buffer texture")
    """
    # Extract all needed data from zarr within context manager to ensure proper cleanup
    with xr.open_zarr(landcover_path) as ds:
        landcover_data = ds[list(ds.data_vars)[0]]
        width_px = len(landcover_data.coords["x"].values)
        height_px = len(landcover_data.coords["y"].values)
        scene_bounds = {
            "xmin": float(landcover_data.coords["x"].min()),
            "xmax": float(landcover_data.coords["x"].max()),
            "ymin": float(landcover_data.coords["y"].min()),
            "ymax": float(landcover_data.coords["y"].max()),
        }

        # Prepare landcover data if needed for filtering
        landcover_2d = None
        if any(r.landcover_filter is not None for r in applicable_regions):
            landcover_2d = np.flipud(landcover_data.values)

    # Dataset now safely closed, continue with region processing
    from ..core.region_geometry import geometry_from_dict

    coord_system = ctx.coordinate_system  # Use cached coordinate system

    with Image.open(texture_path) as img:
        texture_array = np.array(img)
    texture_2d = (
        texture_array[:, :, 0]
        if len(texture_array.shape) == 3
        else texture_array.copy()
    )

    if not hasattr(ctx, "region_material_indices"):
        ctx.region_material_indices = {}
        region_materials = set(r.material_name for r in ctx.config.material_regions)
        for idx, mat_name in enumerate(sorted(region_materials), start=11):
            ctx.region_material_indices[mat_name] = idx

    modified = False
    for region_config in applicable_regions:
        material_idx = ctx.region_material_indices[region_config.material_name]

        try:
            geometry = geometry_from_dict(region_config.geometry)
            mask = geometry.to_mask(width_px, height_px, scene_bounds, coord_system)
            mask_flipped = np.flipud(mask)
            binary_mask = mask_flipped > 0

            if region_config.landcover_filter is not None and landcover_2d is not None:
                binary_mask = binary_mask & np.isin(
                    landcover_2d, region_config.landcover_filter
                )

            pixels_modified = np.sum(binary_mask)
            if pixels_modified > 0:
                texture_2d[binary_mask] = material_idx
                modified = True
                logging.info(
                    f"Applied region '{region_config.region_id}' to {area_name}: "
                    f"{pixels_modified} pixels → material '{region_config.material_name}' "
                    f"(index {material_idx})"
                )
        except Exception as e:
            logging.error(
                f"Failed to apply region '{region_config.region_id}' to {area_name}: {e}"
            )

    if modified:
        img_to_save = Image.fromarray(texture_2d, mode="L")
        img_to_save.save(texture_path)
        logging.info(f"Updated {area_name} with material regions: {texture_path}")


def _generate_texture(
    ctx: SceneResourceContext,
    landcover_path: Path,
    base_name: str,
    dem_file_path: Optional[Path],
    season_month: Optional[int],
    snow_material_index: Optional[int],
    snow_thermoprops: Optional[Path],
    area_name: str,  # "target" | "buffer" | "background" — used for region filtering and log
) -> tuple[Path, Optional[Path]]:
    material_gen = TerrainMaterialGenerator()
    selection_texture_path, preview_texture_path = (
        material_gen.generate_textures_from_file(
            landcover_file_path=landcover_path,
            output_dir=ctx.textures_dir,
            base_name=base_name,
            create_preview=ctx.config.processing.generate_texture_preview,
            dem_file_path=dem_file_path,
            season_month=season_month,
            snow_material_index=snow_material_index,
            coordinate_system=ctx.coordinate_system,
            snow_thermoprops=snow_thermoprops,
        )
    )
    if ctx.config.material_regions:
        applicable_regions = [
            r for r in ctx.config.material_regions if area_name in r.applies_to
        ]
        if applicable_regions:
            _apply_region_materials_to_texture(
                selection_texture_path,
                landcover_path,
                applicable_regions,
                ctx,
                f"{area_name} texture",
            )
    return selection_texture_path, preview_texture_path


def generate_target_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from target area land cover data.

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated target selection texture file
    """

    landcover_file_path = ctx.dependency_outputs["target_landcover"]
    if landcover_file_path is None:
        raise ValueError("Target landcover file not found from dependencies")

    dem_file_path = None
    season_month = None
    snow_material_index = None
    snow_thermoprops = None

    if ctx.config.snow is not None:
        dem_file_path = ctx.dependency_outputs["target_dem"]
        season_month = ctx.config.snow.season_month
        snow_material_index = ctx.config.snow.material_index
        if ctx.config.snow.thermoprops:
            snow_thermoprops = ctx.config.snow.thermoprops.thermoprops_file

        if dem_file_path is None:
            logging.warning("Seasonal snow requested but DEM not available")

    resolution_str = f"{ctx.target_resolution_m}m"
    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        landcover_file_path,
        f"{ctx.scene_name}_{resolution_str}",
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        "target",
    )

    ctx.assets.selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.preview_texture_file = preview_texture_path

    logging.info(f"Target texture: {selection_texture_path}")
    return selection_texture_path


def generate_buffer_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from buffer area land cover data (if buffer is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated buffer selection texture file, or None if buffer disabled
    """
    buffer_landcover_file_path = ctx.dependency_outputs["buffer_landcover"]
    if buffer_landcover_file_path is None:
        logging.warning("Buffer landcover file not found from dependencies")
        return None

    dem_file_path = None
    season_month = None
    snow_material_index = None
    snow_thermoprops = None

    if ctx.config.snow is not None:
        dem_file_path = ctx.dependency_outputs.get("buffer_dem")
        season_month = ctx.config.snow.season_month
        snow_material_index = ctx.config.snow.material_index
        if ctx.config.snow.thermoprops:
            snow_thermoprops = ctx.config.snow.thermoprops.thermoprops_file

        if dem_file_path is None:
            logging.warning("Seasonal snow requested for buffer but DEM not available")

    buffer_resolution_m = ctx.config.buffer.resolution_m
    resolution_str = f"{buffer_resolution_m}m"
    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        buffer_landcover_file_path,
        f"{ctx.scene_name}_buffer_{resolution_str}",
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        "buffer",
    )

    ctx.assets.buffer_selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.buffer_preview_texture_file = preview_texture_path

    return selection_texture_path


def generate_background_texture(ctx: SceneResourceContext) -> Optional[Path]:
    """Generate texture maps from background area land cover data (if background is enabled).

    Args:
        ctx: Scene resource context

    Returns:
        Path to the generated background selection texture file, or None if background disabled
    """
    background_landcover_file_path = ctx.dependency_outputs["background_landcover"]
    if background_landcover_file_path is None:
        logging.warning("Background landcover file not found from dependencies")
        return None

    # Background is always flat - no snow application
    dem_file_path = None
    season_month = None
    snow_material_index = None
    snow_thermoprops = None

    background_resolution_m = ctx.config.background.resolution_m
    selection_texture_path, preview_texture_path = _generate_texture(
        ctx,
        background_landcover_file_path,
        f"{ctx.scene_name}_background_{background_resolution_m}m",
        dem_file_path,
        season_month,
        snow_material_index,
        snow_thermoprops,
        "background",
    )

    ctx.assets.background_selection_texture_file = selection_texture_path
    if preview_texture_path:
        ctx.assets.background_preview_texture_file = preview_texture_path

    return selection_texture_path
