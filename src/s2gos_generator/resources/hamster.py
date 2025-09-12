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
        hamster_path = ctx.config.hamster.data_path
        if not hamster_path.exists():
            if ctx.config.hamster.fallback_on_error:
                logging.warning(
                    f"HAMSTER data file not found: {hamster_path}, falling back to standard baresoil"
                )
                return None
            else:
                raise FileNotFoundError(f"HAMSTER data file not found: {hamster_path}")

        # Open HAMSTER dataset
        ds = xr.open_dataset(hamster_path)

        # Handle coordinate ordering and naming
        if "lat" in ds.dims:
            ds = ds.sel(lat=slice(None, None, -1))

        if "lat" in ds.dims and "lon" in ds.dims:
            ds = ds.swap_dims({"lat": "latitude", "lon": "longitude"})

        # Check variable exists
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
        result_paths = {}

        # Process target area
        if ctx._target_aoi_polygon is not None:
            target_bounds = ctx._target_aoi_polygon.bounds
            target_lon_slice = slice(target_bounds[0], target_bounds[2])
            target_lat_slice = slice(target_bounds[3], target_bounds[1])

            if "latitude" in albedo_data.dims and "longitude" in albedo_data.dims:
                target_subset = albedo_data.sel(
                    longitude=target_lon_slice, latitude=target_lat_slice
                )
                
                # Check if subset has any spatial data
                if target_subset.sizes.get('latitude', 0) == 0 or target_subset.sizes.get('longitude', 0) == 0:
                    logging.warning(
                        f"HAMSTER data has no coverage for target area "
                        f"(lat: {target_lat_slice}, lon: {target_lon_slice}). "
                        "Falling back to standard baresoil."
                    )
                else:
                    target_dataset = target_subset.to_dataset(name=var_name)
                    target_filename = (
                        f"hamster_{ctx.scene_name}_target_{ctx.target_resolution_m}m.zarr"
                    )
                    target_path = ctx.data_dir / target_filename
                    _save_hamster_dataset(target_dataset, target_path)
                    result_paths["target"] = target_path

        # Process buffer area if enabled
        if ctx.has_buffer and ctx._buffer_aoi_polygon is not None:
            buffer_bounds = ctx._buffer_aoi_polygon.bounds
            buffer_lon_slice = slice(buffer_bounds[0], buffer_bounds[2])
            buffer_lat_slice = slice(buffer_bounds[3], buffer_bounds[1])

            buffer_subset = albedo_data.sel(
                longitude=buffer_lon_slice, latitude=buffer_lat_slice
            )
            
            # Check if subset has any spatial data
            if buffer_subset.sizes.get('latitude', 0) == 0 or buffer_subset.sizes.get('longitude', 0) == 0:
                logging.warning(
                    f"HAMSTER data has no coverage for buffer area "
                    f"(lat: {buffer_lat_slice}, lon: {buffer_lon_slice}). "
                    "Falling back to standard baresoil."
                )
            else:
                buffer_dataset = buffer_subset.to_dataset(name=var_name)
                buffer_filename = f"hamster_{ctx.scene_name}_buffer_{ctx.config.buffer_resolution_m}m.zarr"
                buffer_path = ctx.data_dir / buffer_filename
                _save_hamster_dataset(buffer_dataset, buffer_path)
                result_paths["buffer"] = buffer_path

        # Process background area if enabled
        if (
            ctx.has_background
            and ctx._background_aoi_polygon is not None
        ):
            bg_bounds = ctx._background_aoi_polygon.bounds
            bg_lon_slice = slice(bg_bounds[0], bg_bounds[2])
            bg_lat_slice = slice(bg_bounds[3], bg_bounds[1])

            bg_subset = albedo_data.sel(longitude=bg_lon_slice, latitude=bg_lat_slice)
            
            # Check if subset has any spatial data
            if bg_subset.sizes.get('latitude', 0) == 0 or bg_subset.sizes.get('longitude', 0) == 0:
                logging.warning(
                    f"HAMSTER data has no coverage for background area "
                    f"(lat: {bg_lat_slice}, lon: {bg_lon_slice}). "
                    "Falling back to standard baresoil."
                )
            else:
                bg_dataset = bg_subset.to_dataset(name=var_name)
                bg_filename = f"hamster_{ctx.scene_name}_background_{ctx.config.background_resolution_m}m.zarr"
                bg_path = ctx.data_dir / bg_filename
                _save_hamster_dataset(bg_dataset, bg_path)
                result_paths["background"] = bg_path

        # Store result paths in context for scene description
        if result_paths:
            ctx.hamster_data_paths = result_paths
            logging.info(f"HAMSTER data processed for {len(result_paths)} areas")
            return ctx.data_dir  # Return data directory as the "output"
        else:
            return None

    except Exception as e:
        if ctx.config.hamster.fallback_on_error:
            logging.warning(
                f"Could not load HAMSTER data: {e}, falling back to standard baresoil"
            )
            return None
        else:
            raise RuntimeError(f"Failed to load HAMSTER data: {e}") from e


def _save_hamster_dataset(dataset: xr.Dataset, output_path: UPath, upscale_factor: int = 1) -> None:
    """Save HAMSTER dataset to zarr format, with optional upscaling."""

    mkdir(output_path.parent)

    if upscale_factor > 1 and 'latitude' in dataset.dims and 'longitude' in dataset.dims:
        new_lat_size = len(dataset.latitude) * upscale_factor
        new_lon_size = len(dataset.longitude) * upscale_factor

        lat_coords = dataset.latitude.values
        if lat_coords[0] > lat_coords[-1]:
            new_lat = np.linspace(lat_coords.max(), lat_coords.min(), new_lat_size)
        else:
            new_lat = np.linspace(lat_coords.min(), lat_coords.max(), new_lat_size)

        lon_coords = dataset.longitude.values
        new_lon = np.linspace(lon_coords.min(), lon_coords.max(), new_lon_size)

        target_grid = xr.Dataset({
            'latitude': new_lat,
            'longitude': new_lon,
        })

        regridder = Regridder(dataset)
        dataset_to_save = regridder.cubic(target_grid)
    else:
        dataset_to_save = dataset

    dataset_to_save.to_zarr(output_path, mode="w")
