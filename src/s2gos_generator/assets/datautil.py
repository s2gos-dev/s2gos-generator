from pathlib import Path
from typing import Union, Optional

import numpy as np
import xarray as xr
from pyproj import Proj

from ..core.exceptions import RegridError, DataNotFoundError


def validate_data_paths(dem_index_path: Path, landcover_index_path: Path, 
                       dem_root_dir: Path, landcover_root_dir: Path) -> bool:
    """Validate that required data paths exist and are accessible.
    
    Raises:
        DataNotFoundError: If required paths don't exist or are inaccessible
    """
    paths_to_check = [dem_index_path, landcover_index_path, dem_root_dir, landcover_root_dir]
    
    for path in paths_to_check:
        if not path.exists():
            raise DataNotFoundError(f"Required path does not exist: {path}", str(path))
    
    if not dem_root_dir.is_dir():
        raise DataNotFoundError(f"DEM root is not a directory: {dem_root_dir}", str(dem_root_dir))
    
    if not landcover_root_dir.is_dir():
        raise DataNotFoundError(f"Landcover root is not a directory: {landcover_root_dir}", str(landcover_root_dir))
    
    return True


def regrid_to_projection(
    dataset: xr.Dataset,
    target_resolution_m: float,
    center_lat: float,
    center_lon: float,
    aoi_size_km: float,
    interpolation_method: str = "linear",
    fillna_value: Optional[float] = None,
    data_variable: Optional[str] = None
) -> xr.Dataset:
    """Regrid dataset to target resolution using oblique mercator projection.
    
    Args:
        dataset: Input xarray Dataset with lat/lon coordinates
        target_resolution_m: Target resolution in meters
        center_lat: Center latitude for projection
        center_lon: Center longitude for projection  
        aoi_size_km: Area of interest size in kilometers
        interpolation_method: Interpolation method ("linear", "nearest", etc.)
        fillna_value: Value to fill NaN values with (optional)
        data_variable: Specific data variable to process for fillna (optional)
        
    Returns:
        Regridded xarray Dataset with oblique mercator coordinates
        
    Raises:
        RegridError: If regridding operation fails
    """
    try:
        domain_width_m = aoi_size_km * 1000.0
        
        half_width = domain_width_m / 2.0
        num_points = int(round(domain_width_m / target_resolution_m))
        start_coord = -half_width + (target_resolution_m / 2.0)
        end_coord = half_width - (target_resolution_m / 2.0)
        target_x = np.linspace(start_coord, end_coord, num_points)
        target_y = target_x.copy()
        
        proj = Proj(f"+proj=omerc +lat_0={center_lat} +lonc={center_lon} +alpha=0 +gamma=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m")
        
        ds = dataset.copy()
        if 'y' in ds.dims and 'x' in ds.dims:
            ds = ds.rename({'y': 'lat', 'x': 'lon'})
        
        ys, xs = np.meshgrid(target_y, target_x, indexing='ij')
        lons, lats = proj(xs.ravel(), ys.ravel(), inverse=True)
        lonlats = xr.Dataset({
            "lon": (("y", "x"), lons.reshape(xs.shape)),
            "lat": (("y", "x"), lats.reshape(xs.shape))
        })
        
        ds_regridded = ds.interp(
            lonlats,
            method=interpolation_method,
            kwargs={"bounds_error": False, "fill_value": None}
        )
        ds_regridded = ds_regridded.assign_coords({"y": target_y, "x": target_x})
        
        if fillna_value is not None and data_variable is not None:
            ds_regridded[data_variable] = ds_regridded[data_variable].fillna(fillna_value)
        
        ds_regridded = ds_regridded.drop_vars(["lon", "lat"], errors='ignore')
        
        return ds_regridded
        
    except Exception as e:
        raise RegridError(f"Failed to regrid dataset to {target_resolution_m}m resolution", "regridding", e)
