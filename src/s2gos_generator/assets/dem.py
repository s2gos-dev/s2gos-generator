import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import xarray as xr
from shapely.geometry import Polygon

from .base_processor import BaseTileProcessor


class DEMProcessor(BaseTileProcessor):
    """Finds, merges, and processes Copernicus GLO-30 DEM tiles for a given AOI."""

    def __init__(self, index_path: Union[Path, str], dem_root_dir: Union[Path, str]):
        """Initialize the DEM processor."""
        super().__init__(index_path, dem_root_dir, "DEM")
    
    @property
    def path_column(self) -> str:
        """Column name in index file containing relative paths to DEM tiles."""
        return "path_dem"
    
    @property
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        return "elevation"
    
    @property
    def default_interpolation_method(self) -> str:
        """Default interpolation method for DEM regridding."""
        return "linear"
    
    @property
    def data_type(self) -> Optional[str]:
        """Data type to cast the data to (DEM data stays as float)."""
        return None
    
    @property
    def default_fill_value(self) -> Union[float, int]:
        """Default fill value for NaN values in DEM data."""
        return 0.0
    
    @property
    def use_context_manager(self) -> bool:
        """DEM processor uses context manager for file opening."""
        return True


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
        merged_dem = self._merge_tiles(tile_paths, aoi_polygon, fillna_value=fillna_value)

        if target_resolution_m is not None and center_lat is not None and center_lon is not None and aoi_size_km is not None:
            logging.info(f"Regridding DEM to {target_resolution_m}m resolution...")
            merged_dem = self._regrid_data(merged_dem, target_resolution_m, center_lat, center_lon, aoi_size_km, fillna_value)

        self._save_dataset(merged_dem, output_path)
        
        return merged_dem