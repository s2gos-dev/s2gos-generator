import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from .datautil import regrid_to_projection
from ..core.paths import read_geofeather, exists, open_dataarray


def create_aoi_polygon(
    center_lat: float,
    center_lon: float,
    side_length_km: float = 200.0
) -> Polygon:
    """Create a square polygon in WGS84 CRS centered at a given point."""
    wgs84_crs = CRS("EPSG:4326")
    local_azimuthal_crs = CRS(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +ellps=WGS84 +units=m"
    )

    transformer_to_local = Transformer.from_crs(wgs84_crs, local_azimuthal_crs, always_xy=True)
    transformer_from_local = Transformer.from_crs(local_azimuthal_crs, wgs84_crs, always_xy=True)

    center_x, center_y = transformer_to_local.transform(center_lon, center_lat)
    half_side_m = (side_length_km * 1000) / 2

    local_corners = [
        (center_x - half_side_m, center_y - half_side_m),
        (center_x + half_side_m, center_y - half_side_m),
        (center_x + half_side_m, center_y + half_side_m),
        (center_x - half_side_m, center_y + half_side_m),
    ]

    lon_lat_coords = [transformer_from_local.transform(x, y) for x, y in local_corners]

    return Polygon(lon_lat_coords)


class DEMProcessor:
    """Finds, merges, and processes Copernicus GLO-30 DEM tiles for a given AOI."""

    def __init__(self, index_path: Union[Path, str], dem_root_dir: Union[Path, str]):
        """Initialize the DEM processor."""
        if not exists(index_path):
            raise FileNotFoundError(f"Index file not found at: {index_path}")
        if not exists(dem_root_dir):
            raise NotADirectoryError(f"DEM root directory not found: {dem_root_dir}")

        logging.info("Loading DEM index file...")
        self.index_gdf = read_geofeather(index_path)
        self.dem_root_dir = Path(dem_root_dir)  # Keep as Path for joining
        logging.info("DEMProcessor initialized successfully.")

    def _find_intersecting_tiles(self, aoi_polygon: Polygon) -> List[Path]:
        """Find DEM tiles that intersect with the AOI.
        
        Raises:
            FileNotFoundError: If no intersecting DEM tiles are found in the index.
        """
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")
        selected_products = self.index_gdf.sjoin(aoi_gdf.to_crs(self.index_gdf.crs))

        if selected_products.empty:
            raise FileNotFoundError(f"No DEM tiles found for the given AOI.")

        relative_paths = selected_products["path_dem"].unique()
        filepaths = [self.dem_root_dir / p for p in relative_paths]
        
        logging.info(f"Found {len(filepaths)} intersecting DEM tile(s).")
        return filepaths

    def _merge_tiles(
        self,
        tile_paths: List[Path],
        fillna_value: Optional[float] = None
    ) -> xr.Dataset:
        """Merge multiple DEM GeoTIFFs into a single xarray Dataset."""
        logging.info(f"Opening and preparing {len(tile_paths)} tiles for merging...")
        
        data_arrays = []
        for path in tile_paths:
            with open_dataarray(path, engine="rasterio") as da:
                processed_da = (
                    da.isel(band=0, drop=True)
                      .rename("elevation")
                      .rename({"x": "lon", "y": "lat"})
                )
                data_arrays.append(processed_da)

        logging.info("Merging tiles into a single dataset...")
        merged_ds = xr.merge(data_arrays, compat="no_conflicts")

        if fillna_value is not None:
            logging.info(f"Filling NaN values with {fillna_value}.")
            merged_ds = merged_ds.fillna(fillna_value)

        return merged_ds

    def _regrid_dem(
        self,
        dem_ds: xr.Dataset,
        target_resolution_m: float,
        center_lat: float,
        center_lon: float,
        fillna_value: Optional[float],
        aoi_size_km: float
    ) -> xr.Dataset:
        """Regrid DEM dataset to target resolution using oblique mercator projection."""
        return regrid_to_projection(
            dataset=dem_ds,
            target_resolution_m=target_resolution_m,
            center_lat=center_lat,
            center_lon=center_lon,
            aoi_size_km=aoi_size_km,
            interpolation_method="linear",
            fillna_value=fillna_value,
            data_variable="elevation"
        )


    def generate_dem(
        self,
        aoi_polygon: Polygon,
        output_path: Path,
        fillna_value: Optional[float] = 0.0,
        target_resolution_m: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        aoi_size_km: Optional[float] = None,
    ) -> xr.Dataset:
        """Generate DEM data for the AOI."""
        tile_paths = self._find_intersecting_tiles(aoi_polygon)
        merged_dem = self._merge_tiles(tile_paths, fillna_value=fillna_value)

        if target_resolution_m is not None and center_lat is not None and center_lon is not None and aoi_size_km is not None:
            logging.info(f"Regridding DEM to {target_resolution_m}m resolution...")
            merged_dem = self._regrid_dem(merged_dem, target_resolution_m, center_lat, center_lon, fillna_value, aoi_size_km)

        logging.info(f"Saving merged DEM to '{output_path}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged_dem.to_zarr(output_path, mode="w")
        logging.info("DEM generation complete.")
        
        return merged_dem