"""Base tile processor for unified DEM and LandCover processing."""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import psutil
import rioxarray as rxr
import xarray as xr
from s2gos_utils.io.paths import open_dataarray
from shapely.geometry import Polygon
from upath import UPath

from .datautil import regrid_to_projection
from ..dataset import Dataset

# Configure PROJ environment to fix "Cannot find proj.db" warnings
try:
    import pyproj

    # Set PROJ_DATA to the correct location for this environment
    proj_data_dir = pyproj.datadir.get_data_dir()
    os.environ["PROJ_DATA"] = proj_data_dir

    # Clear any conflicting PROJ_LIB environment variable
    if "PROJ_LIB" in os.environ:
        del os.environ["PROJ_LIB"]

    logging.debug(f"PROJ environment configured: PROJ_DATA={proj_data_dir}")
except ImportError:
    logging.warning("pyproj not available, PROJ environment not configured")
except Exception as e:
    logging.warning(f"Failed to configure PROJ environment: {e}")


class BaseTileProcessor(ABC):
    """Base class for tile-based data processors (DEM, LandCover, etc.)."""

    def __init__(
        self,
        dataset: Dataset,
    ):
        """Initialize the base tile processor.

        Args:
            dataset: Dataset containing the data tiles.
        """

        self.dataset = dataset
        self.data_description = dataset.name

    @property
    @abstractmethod
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        pass

    @property
    @abstractmethod
    def default_interpolation_method(self) -> str:
        """Default interpolation method for regridding."""
        pass

    @property
    @abstractmethod
    def data_type(self) -> Optional[str]:
        """Data type to cast the data to (e.g., 'uint8'), or None for no casting."""
        pass

    @property
    @abstractmethod
    def default_fill_value(self) -> Union[float, int]:
        """Default fill value for NaN values."""
        pass

    @property
    @abstractmethod
    def use_context_manager(self) -> bool:
        """Whether to use context manager when opening data arrays."""
        pass

    def _calculate_optimal_chunk_size(self, num_tiles: int) -> int:
        """Calculate optimal chunk size based on available memory and tile count."""
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        # Target: ~1 million elements per chunk (xarray best practice)
        base_chunk_size = int((1_000_000) ** 0.5)

        # Scale down if many tiles or limited memory
        memory_factor = min(1.0, available_memory_gb / 8.0)  # Scale down if < 8GB
        tile_factor = min(1.0, 4.0 / num_tiles)

        chunk_size = int(base_chunk_size * memory_factor * tile_factor)
        chunk_size = max(512, min(chunk_size, 4096))

        return chunk_size

    def _merge_tiles(
        self,
        tile_paths: List[UPath],
        aoi_polygon: Optional[Polygon] = None,
        fillna_value: Optional[Union[float, int]] = None,
    ) -> xr.Dataset:
        """Merge multiple data tiles into a single optimized dataset.

        Args:
            tile_paths: List of paths to data tiles
            aoi_polygon: Optional AOI polygon for early spatial filtering
            fillna_value: Optional fill value for NaN values (uses default_fill_value if None)

        Returns:
            Merged dataset
        """

        chunk_size = self._calculate_optimal_chunk_size(len(tile_paths))

        bbox = None
        if aoi_polygon:
            bbox = aoi_polygon.bounds

        data_arrays = []
        for i, path in enumerate(tile_paths):
            # Open file - use context manager or direct assignment based on processor preference
            if self.use_context_manager:
                with open_dataarray(
                    path, engine="rasterio", chunks={"x": chunk_size, "y": chunk_size}
                ) as da:
                    da = self._process_single_tile(da, bbox)
                    data_arrays.append(da)
            else:
                da = rxr.open_rasterio(
                    str(path), chunks={"x": chunk_size, "y": chunk_size}
                )
                da = self._process_single_tile(da, bbox)
                data_arrays.append(da)

        merged_ds = xr.merge(data_arrays, compat="no_conflicts")

        # Apply fill values
        fill_value = (
            fillna_value if fillna_value is not None else self.default_fill_value
        )
        if fill_value is not None:
            merged_ds = merged_ds.fillna(fill_value)

        return merged_ds

    def _process_single_tile(
        self, da: xr.DataArray, bbox: Optional[tuple]
    ) -> xr.DataArray:
        """Process a single tile: apply spatial filtering, renaming, and type conversion.

        Args:
            da: Input data array
            bbox: Optional bounding box for early spatial filtering

        Returns:
            Processed data array
        """
        # Early spatial filtering if bbox provided
        if bbox:
            try:
                # Rough clip to bounding box first to reduce data volume
                da = da.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))
            except (KeyError, ValueError):
                # If coordinates don't overlap, skip this early filtering
                pass

        # Process the data array
        processed = (
            da.isel(band=0, drop=True)
            .rename({"x": "lon", "y": "lat"})
            .rename(self.data_variable_name)
        )

        # Apply data type conversion if specified
        if self.data_type:
            processed = processed.astype(self.data_type)

        return processed

    def _clip_to_aoi(self, dataset: xr.Dataset, aoi_polygon: Polygon) -> xr.Dataset:
        """Clip the dataset to the exact AOI geometry."""
        try:
            if not hasattr(dataset.rio, "crs") or dataset.rio.crs is None:
                dataset = dataset.rio.write_crs("EPSG:4326")

            bounds = aoi_polygon.bounds

            if "x" in dataset.dims:
                x_dim = "x"
                y_dim = "y"
            elif "lon" in dataset.dims:
                x_dim = "lon"
                y_dim = "lat"

            lon_min, lat_min, lon_max, lat_max = aoi_polygon.bounds

            # Slices depend on the dimension direction, which can flip when merging
            lon = dataset[x_dim]
            lat = dataset[y_dim]
            lon_slice = (
                slice(lon_min, lon_max) if lon[0] < lon[-1] else slice(lon_max, lon_min)
            )
            lat_slice = (
                slice(lat_min, lat_max) if lat[0] < lat[-1] else slice(lat_max, lat_min)
            )
            # Select an area of computation lazily to cater for netcdf and zarr formats
            dataset = dataset.sel({x_dim: lon_slice, y_dim: lat_slice})

            if not hasattr(dataset.rio, "_x_dim") or dataset.rio._x_dim is None:
                dataset = dataset.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

            clipped_ds = dataset.rio.clip([aoi_polygon], crs="EPSG:4326", drop=True)

            return clipped_ds

        except ImportError:
            logging.warning(
                "rioxarray not available, using bounding box clipping instead..."
            )
            bounds = aoi_polygon.bounds  # (minx, miny, maxx, maxy)

            if "x" in dataset.dims and "y" in dataset.dims:
                x_dim, y_dim = "x", "y"
            elif "lon" in dataset.dims and "lat" in dataset.dims:
                x_dim, y_dim = "lon", "lat"
            else:
                raise ValueError(
                    "Dataset must have either (x, y) or (lon, lat) coordinates"
                )

            clipped_ds = dataset.sel(
                {x_dim: slice(bounds[0], bounds[2]), y_dim: slice(bounds[3], bounds[1])}
            )
            return clipped_ds

    def _regrid_data(
        self,
        dataset: xr.Dataset,
        target_resolution_m: float,
        center_lat: float,
        center_lon: float,
        aoi_size_km: float,
        fillna_value: Optional[float] = None,
    ) -> xr.Dataset:
        """Regrid dataset to target resolution using oblique mercator projection."""
        return regrid_to_projection(
            dataset=dataset,
            target_resolution_m=target_resolution_m,
            center_lat=center_lat,
            center_lon=center_lon,
            aoi_size_km=aoi_size_km,
            interpolation_method=self.default_interpolation_method,
            fillna_value=fillna_value,
            data_variable=self.data_variable_name,
        )

    def _save_dataset(self, dataset: xr.Dataset, output_path: UPath) -> None:
        """Save dataset to zarr format with proper directory creation."""
        from s2gos_utils.io.paths import mkdir

        mkdir(output_path.parent)
        dataset.to_zarr(output_path, mode="w")
        logging.info(f"{self.data_description} saved to {output_path}")
