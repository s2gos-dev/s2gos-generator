"""HAMSTER albedo data processing resources."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from s2gos_utils.io.paths import mkdir
from upath import UPath
from xarray_regrid import Regridder

from ..core.context import SceneResourceContext


def process_hamster_data(ctx: SceneResourceContext) -> Optional[Path]:
    """Process HAMSTER albedo data for scene areas with spatial clipping.

    Args:
        ctx: Scene resource context

    Returns:
        Path to directory containing processed HAMSTER zarr files
    """

    try:
        hamster_path = ctx.config.hamster.data_path.upath
        if not hamster_path.exists():
            if ctx.config.hamster.fallback_on_error:
                logging.warning(
                    f"HAMSTER data file not found: {hamster_path}, falling back to standard baresoil"
                )
                return None
            else:
                raise FileNotFoundError(f"HAMSTER data file not found: {hamster_path}")

        ds = xr.open_dataset(hamster_path)

        if "lat" in ds.dims:
            ds = ds.sel(lat=slice(None, None, -1))

        if "lat" in ds.dims and "lon" in ds.dims:
            ds = ds.swap_dims({"lat": "latitude", "lon": "longitude"})

        var_name = ctx.config.hamster.variable_name
        if var_name not in ds.data_vars:
            if ctx.config.hamster.fallback_on_error:
                logging.warning(
                    f"Variable '{var_name}' not found in HAMSTER data, falling back to standard baresoil"
                )
                return None
            else:
                raise KeyError(f"Variable '{var_name}' not found in HAMSTER dataset")

        albedo_data = ds[var_name]

        center_lat = ctx.center_lat
        center_lon = ctx.center_lon
        proj_string = f"+proj=omerc +lat_0={center_lat} +lonc={center_lon} +alpha=0 +gamma=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"

        albedo_data = albedo_data.transpose("wavelength", "latitude", "longitude")
        albedo_data = albedo_data.rio.set_spatial_dims(
            x_dim="longitude", y_dim="latitude"
        )
        albedo_data = albedo_data.rio.write_crs("EPSG:4326")

        albedo_projected = albedo_data.rio.reproject(proj_string)

        correct_y_coords = np.flip(albedo_projected.y.values)
        albedo_data = albedo_projected.assign_coords(y=correct_y_coords)

        target_x, target_y = 0.0, 0.0

        logging.info(
            f"HAMSTER data projected using scene center ({center_lat:.6f}, {center_lon:.6f})"
        )
        logging.info(f"Target coordinates: ({target_x:.2f}, {target_y:.2f}) meters")
        logging.info(
            f"Data bounds: x=[{albedo_data.x.min().item():.0f}, {albedo_data.x.max().item():.0f}], y=[{albedo_data.y.min().item():.0f}, {albedo_data.y.max().item():.0f}]"
        )

        result_paths = {}

        if ctx._target_aoi_polygon is not None:
            path = _crop_and_save_area(
                albedo_data,
                "target",
                ctx.aoi_size_km,
                f"hamster_{ctx.scene_name}_target_{ctx.target_resolution_m}m.zarr",
                ctx.data_dir,
                var_name,
            )
            if path:
                result_paths["target"] = path

        if ctx.has_buffer and ctx._buffer_aoi_polygon is not None:
            path = _crop_and_save_area(
                albedo_data,
                "buffer",
                ctx.config.buffer.size_km,
                f"hamster_{ctx.scene_name}_buffer_{ctx.config.buffer.resolution_m}m.zarr",
                ctx.data_dir,
                var_name,
            )
            if path:
                result_paths["buffer"] = path

        if ctx.has_background and ctx._background_aoi_polygon is not None:
            path = _crop_and_save_area(
                albedo_data,
                "background",
                ctx.config.background.size_km,
                f"hamster_{ctx.scene_name}_background_{ctx.config.background.resolution_m}m.zarr",
                ctx.data_dir,
                var_name,
            )
            if path:
                result_paths["background"] = path

        if result_paths:
            ctx.hamster_data_paths = result_paths
            logging.info(f"HAMSTER data processed for {len(result_paths)} areas")
            for area, path in result_paths.items():
                logging.info(f"  {area}: {path}")
            return ctx.data_dir
        else:
            logging.warning(
                "No HAMSTER result paths - ctx.hamster_data_paths will not be set"
            )
            return None

    except Exception as e:
        if ctx.config.hamster.fallback_on_error:
            logging.warning(
                f"Could not load HAMSTER data: {e}, falling back to standard baresoil"
            )
            return None
        else:
            raise RuntimeError(f"Failed to load HAMSTER data: {e}") from e


def _save_hamster_dataset(
    dataset: xr.Dataset, output_path: UPath, upscale_factor: int = 1
) -> None:
    """Save HAMSTER dataset to zarr format, with optional upscaling."""

    mkdir(output_path.parent)

    if upscale_factor > 1:
        if "x" in dataset.dims and "y" in dataset.dims:
            new_x_size = len(dataset.x) * upscale_factor
            new_y_size = len(dataset.y) * upscale_factor

            x_coords = dataset.x.values
            y_coords = dataset.y.values

            new_x = np.linspace(x_coords.min(), x_coords.max(), new_x_size)
            new_y = np.linspace(y_coords.min(), y_coords.max(), new_y_size)

            target_grid = xr.Dataset(
                {
                    "x": new_x,
                    "y": new_y,
                }
            )
        else:
            dataset_to_save = dataset

        if "target_grid" in locals():
            regridder = Regridder(dataset)
            dataset_to_save = regridder.cubic(target_grid)
        else:
            dataset_to_save = dataset
    else:
        dataset_to_save = dataset

    dataset_to_save.to_zarr(output_path, mode="w")


def _crop_and_save_area(
    albedo_data: xr.DataArray,
    area_name: str,
    size_km: float,
    filename: str,
    output_dir: UPath,
    var_name: str,
) -> Optional[UPath]:
    """Crop HAMSTER data for specified area and save to file."""

    half_size_m = (size_km * 1000) / 2

    try:
        subset = albedo_data.sel(
            x=slice(-half_size_m, half_size_m), y=slice(-half_size_m, half_size_m)
        )

        if subset.sizes.get("x", 0) == 0 or subset.sizes.get("y", 0) == 0:
            logging.warning(f"No HAMSTER coverage for {area_name} area ({size_km}km)")
            return None

        dataset = subset.to_dataset(name=var_name)
        output_path = output_dir / filename
        _save_hamster_dataset(dataset, output_path)

        actual_size = (subset.x.max().item() - subset.x.min().item()) / 1000
        logging.info(
            f"HAMSTER {area_name}: {actual_size:.1f}km x {actual_size:.1f}km, {subset.sizes}"
        )

        return output_path

    except Exception as e:
        logging.error(f"Error processing {area_name} HAMSTER data: {e}")
        return None


# def _create_hamster_preview(albedo_data_original: xr.DataArray,
#                            albedo_data_projected: xr.DataArray,
#                            output_dir: UPath,
#                            scene_name: str) -> None:
#     """Create before/after plots showing original vs projected HAMSTER data."""

#     try:
#         rgb_bands = [640, 550, 470]
#         fig, axes = plt.subplots(1, 2, figsize=(16, 8))

#         albedo_data_original.sel(wavelength=rgb_bands, method="nearest").plot.imshow(
#             ax=axes[0], rgb="wavelength", robust=True
#         )
#         axes[0].set_title("Original Data (Geographic Coords)")
#         axes[0].set_xlabel("Longitude")
#         axes[0].set_ylabel("Latitude")

#         albedo_data_projected.sel(wavelength=rgb_bands, method="nearest").plot.imshow(
#             ax=axes[1], rgb="wavelength", robust=True
#         )
#         axes[1].set_title("Projected Data (Oblique Mercator)")
#         axes[1].set_xlabel("X Coordinate (meters)")
#         axes[1].set_ylabel("Y Coordinate (meters)")
#         axes[1].set_aspect('equal', adjustable='box')

#         plt.tight_layout()

#         output_path = output_dir / f"hamster_{scene_name}_projection_preview.png"
#         mkdir(output_path.parent)
#         plt.savefig(output_path, dpi=150, bbox_inches='tight')
#         plt.close()

#         logging.info(f"HAMSTER projection preview saved: {output_path}")

#     except Exception as e:
#         logging.warning(f"Could not create HAMSTER projection preview: {e}")
