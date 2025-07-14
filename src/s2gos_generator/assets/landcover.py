import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr
from shapely.geometry import Polygon
import rioxarray as rxr

from .base_processor import BaseTileProcessor


class LandCoverProcessor(BaseTileProcessor):
    """Finds, merges, and processes ESA WorldCover land cover tiles for a given AOI."""

    def __init__(self, index_path: Union[Path, str], landcover_root_dir: Union[Path, str]):
        """Initialize the land cover processor."""
        super().__init__(index_path, landcover_root_dir, "land cover")
        
    @property
    def path_column(self) -> str:
        """Column name in index file containing relative paths to land cover tiles."""
        return "path_lc"
    
    @property
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        return "landcover"
    
    @property
    def default_interpolation_method(self) -> str:
        """Default interpolation method for land cover regridding."""
        return "nearest"
    
    @property
    def data_type(self) -> Optional[str]:
        """Data type to cast the data to (landcover uses uint8)."""
        return "uint8"
    
    @property
    def default_fill_value(self) -> Union[float, int]:
        """Default fill value for NaN values in landcover data."""
        return 7
    
    @property
    def use_context_manager(self) -> bool:
        """Landcover processor uses direct assignment for file opening."""
        return False

    
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
        
        # Pass AOI to merge for early spatial filtering
        merged_landcover = self._merge_tiles(tile_paths, aoi_polygon)
        
        if target_resolution_m is not None:
            logging.info("Persisting merged data before regridding...")
            merged_landcover = merged_landcover.persist()
        
        clipped_landcover = self._clip_to_aoi(merged_landcover, aoi_polygon)
        
        if target_resolution_m is not None and center_lat is not None and center_lon is not None and aoi_size_km is not None:
            logging.info(f"Regridding landcover to {target_resolution_m}m resolution...")
            clipped_landcover = self._regrid_data(clipped_landcover, target_resolution_m, center_lat, center_lon, aoi_size_km)

        self._save_dataset(clipped_landcover, output_path)
        
        return clipped_landcover


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