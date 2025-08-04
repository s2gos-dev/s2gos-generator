"""Base tile processor for unified DEM and LandCover processing."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import geopandas as gpd
import psutil
import rioxarray as rxr
import xarray as xr
from s2gos_utils.io.paths import exists, open_dataarray, read_geofeather
from s2gos_utils.typing import PathLike
from shapely.geometry import Polygon
from upath import UPath

from .datautil import regrid_to_projection


class BaseTileProcessor(ABC):
    """Base class for tile-based data processors (DEM, LandCover, etc.)."""

    def __init__(
        self,
        index_path: PathLike,
        data_root_dir: PathLike,
        data_description: str,
    ):
        """Initialize the base tile processor.

        Args:
            index_path: Path to the spatial index file
            data_root_dir: Root directory containing data tiles
            data_description: Type of data being processed (for logging)
        """
        if not exists(index_path):
            raise FileNotFoundError(f"Index file not found at: {index_path}")
        if not exists(data_root_dir):
            raise NotADirectoryError(
                f"{data_description} root directory not found: {data_root_dir}"
            )

        logging.info(f"Loading {data_description} index file...")
        self.index_gdf = read_geofeather(index_path)
        self.data_root_dir = UPath(data_root_dir)
        self.data_description = data_description
        logging.info(f"{self.__class__.__name__} initialized successfully.")

    @property
    @abstractmethod
    def path_column(self) -> str:
        """Column name in index file containing relative paths to data tiles."""
        pass

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

    def _find_intersecting_tiles(self, aoi_polygon: Polygon) -> List[UPath]:
        """Find data tiles that intersect with the AOI.

        Args:
            aoi_polygon: Area of interest polygon

        Returns:
            List of paths to intersecting tiles

        Raises:
            FileNotFoundError: If no intersecting tiles are found
        """
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")
        selected_products = self.index_gdf.sjoin(aoi_gdf.to_crs(self.index_gdf.crs))

        if selected_products.empty:
            raise FileNotFoundError(
                f"No {self.data_description} tiles found for the given AOI."
            )

        relative_paths = selected_products[self.path_column].unique()
        filepaths = [self.data_root_dir / p for p in relative_paths]

        logging.info(
            f"Found {len(filepaths)} intersecting {self.data_description} tile(s)."
        )
        return filepaths

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

        logging.info(
            f"Memory: {available_memory_gb:.1f}GB, {num_tiles} tiles → chunk size: {chunk_size}x{chunk_size}"
        )
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
        logging.info(f"Opening {len(tile_paths)} tiles with optimized Dask chunks...")

        chunk_size = self._calculate_optimal_chunk_size(len(tile_paths))

        bbox = None
        if aoi_polygon:
            bbox = aoi_polygon.bounds
            logging.info(f"Using AOI bbox for early filtering: {bbox}")

        data_arrays = []
        for i, path in enumerate(tile_paths):
            logging.info(f"Opening tile {i + 1}/{len(tile_paths)}: {path}")

            # Open file - use context manager or direct assignment based on processor preference
            if self.use_context_manager:
                with open_dataarray(
                    path, engine="rasterio", chunks={"x": chunk_size, "y": chunk_size}
                ) as da:
                    da = self._process_single_tile(da, bbox)
                    data_arrays.append(da)
            else:
                da = rxr.open_rasterio(path, chunks={"x": chunk_size, "y": chunk_size})
                da = self._process_single_tile(da, bbox)
                data_arrays.append(da)

        logging.info("Merging tiles into a single dataset...")
        merged_ds = xr.merge(data_arrays, compat="no_conflicts")

        logging.info(f"Merged. Shape: {merged_ds[self.data_variable_name].shape}")
        logging.info(f"Data type: {type(merged_ds[self.data_variable_name].data)}")

        # Apply fill values
        fill_value = (
            fillna_value if fillna_value is not None else self.default_fill_value
        )
        if fill_value is not None:
            logging.info(f"Filling NaN values with {fill_value}.")
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

        logging.info(f"  Chunks: {processed.chunks}")
        return processed

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
        logging.info(f"Saving processed {self.data_description} to '{output_path}'")
        from s2gos_utils.io.paths import mkdir

        mkdir(output_path.parent)
        dataset.to_zarr(output_path, mode="w")
        logging.info(f"{self.data_description} generation complete.")
