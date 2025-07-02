"""ESA WorldCover land cover data processing for area of interest."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon

from .datautil import regrid_to_projection



class LandCoverProcessor:
    """Finds, merges, and processes ESA WorldCover land cover tiles for a given AOI."""

    def __init__(self, index_path: Path, landcover_root_dir: Path):
        """Initialize the land cover processor."""
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at: {index_path}")
        if not landcover_root_dir.is_dir():
            raise NotADirectoryError(f"Land cover root directory not found: {landcover_root_dir}")

        logging.info("Loading land cover index file...")
        self.index_gdf = gpd.read_feather(index_path)
        self.landcover_root_dir = landcover_root_dir
        logging.info("LandCoverProcessor initialized successfully.")

    def _find_intersecting_tiles(self, aoi_polygon: Polygon) -> List[Path]:
        """Find land cover tiles that intersect with the AOI."""
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")

        selected_products = self.index_gdf.sjoin(aoi_gdf.to_crs(self.index_gdf.crs))

        if selected_products.empty:
            raise FileNotFoundError(f"No land cover tiles found for the given AOI.")

        relative_paths = selected_products["path_lc"].unique()
        filepaths = [self.landcover_root_dir / p for p in relative_paths]
        
        logging.info(f"Found {len(filepaths)} intersecting land cover tile(s).")
        return filepaths

    def _merge_tiles(self, tile_paths: List[Path]) -> xr.Dataset:
        """Merge multiple land cover GeoTIFFs into a single xarray Dataset."""
        logging.info(f"Opening and preparing {len(tile_paths)} tiles for merging...")
        
        data_arrays = []
        for path in tile_paths:
            with xr.open_dataarray(path, engine="rasterio") as da:
                processed_da = (
                    da.isel(band=0, drop=True)
                      .rename("landcover")
                      .rename({"x": "lon", "y": "lat"})
                      .astype("uint8")
                )
                data_arrays.append(processed_da)

        logging.info("Merging tiles into a single dataset...")
        merged_ds = xr.merge(data_arrays, compat="no_conflicts")
        
        logging.info("Filling NaN values with water landcover class (7)")
        merged_ds = merged_ds.fillna(7)

        return merged_ds

    def _clip_to_aoi(self, dataset: xr.Dataset, aoi_polygon: Polygon) -> xr.Dataset:
        """Clip the dataset to the exact AOI geometry."""
        try:
            logging.info("Clipping dataset to AOI geometry...")
            
            if not hasattr(dataset.rio, 'crs') or dataset.rio.crs is None:
                dataset = dataset.rio.write_crs("EPSG:4326")
            
            if not hasattr(dataset.rio, '_x_dim') or dataset.rio._x_dim is None:
                if 'x' in dataset.dims:
                    dataset = dataset.rio.set_spatial_dims(x_dim='x', y_dim='y')
                elif 'lon' in dataset.dims:
                    dataset = dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
            
            clipped_ds = dataset.rio.clip([aoi_polygon], crs="EPSG:4326", drop=True)
            logging.info("Clipping completed.")
            
            return clipped_ds
            
        except ImportError:
            logging.warning("rioxarray not available, using bounding box clipping instead...")
            bounds = aoi_polygon.bounds  # (minx, miny, maxx, maxy)
            
            if 'x' in dataset.dims and 'y' in dataset.dims:
                x_dim, y_dim = 'x', 'y'
            elif 'lon' in dataset.dims and 'lat' in dataset.dims:
                x_dim, y_dim = 'lon', 'lat'
            else:
                raise ValueError("Dataset must have either (x, y) or (lon, lat) coordinates")
            
            clipped_ds = dataset.sel(
                {x_dim: slice(bounds[0], bounds[2]),
                 y_dim: slice(bounds[3], bounds[1])}
            )
            return clipped_ds

    def _regrid_landcover(
        self,
        landcover_ds: xr.Dataset,
        target_resolution_m: float,
        center_lat: float,
        center_lon: float,
        aoi_size_km: float
    ) -> xr.Dataset:
        """Regrid landcover to target resolution using oblique mercator projection."""
        return regrid_to_projection(
            dataset=landcover_ds,
            target_resolution_m=target_resolution_m,
            center_lat=center_lat,
            center_lon=center_lon,
            aoi_size_km=aoi_size_km,
            interpolation_method="nearest"
        )

    def generate_landcover(
        self,
        aoi_polygon: Polygon,
        output_path: Path,
        target_resolution_m: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        aoi_size_km: Optional[float] = None
    ) -> xr.Dataset:
        """Generate landcover data for the AOI."""
        tile_paths = self._find_intersecting_tiles(aoi_polygon)
        merged_landcover = self._merge_tiles(tile_paths)
        clipped_landcover = self._clip_to_aoi(merged_landcover, aoi_polygon)
        
        if target_resolution_m is not None and center_lat is not None and center_lon is not None and aoi_size_km is not None:
            logging.info(f"Regridding landcover to {target_resolution_m}m resolution...")
            clipped_landcover = self._regrid_landcover(clipped_landcover, target_resolution_m, center_lat, center_lon, aoi_size_km)

        logging.info(f"Saving processed land cover to '{output_path}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {"landcover": {"zlib": True, "complevel": 5, "dtype": "uint8"}}
        clipped_landcover.to_netcdf(output_path, encoding=encoding)
        logging.info("Land cover generation complete.")
        
        return clipped_landcover


# ESA WorldCover class mapping
ESA_LANDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland", 
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}