"""Path utilities with fsspec integration for unified file access."""

from pathlib import Path
from typing import Union, Any, Dict
import json
import yaml
import fsspec
import pandas as pd
import xarray as xr


def open_file(path: Union[Path, str], mode: str = 'r', **kwargs):
    """Open a file using fsspec for unified access across storage backends.
    
    Args:
        path: Path to file (local or remote)
        mode: File mode ('r', 'rb', 'w', 'wb', etc.)
        **kwargs: Additional arguments for fsspec.open()
    
    Returns:
        fsspec file object
    """
    return fsspec.open(str(path), mode=mode, **kwargs)


def exists(path: Union[Path, str]) -> bool:
    """Check if file or directory exists using fsspec.
    
    Args:
        path: Path to check
    
    Returns:
        True if path exists, False otherwise
    """
    try:
        fs = fsspec.filesystem(fsspec.utils.infer_storage_options(str(path))["protocol"])
        return fs.exists(str(path))
    except Exception:
        return False


def read_feather(path: Union[Path, str], **kwargs) -> pd.DataFrame:
    """Read feather file using fsspec.
    
    Args:
        path: Path to feather file
        **kwargs: Additional arguments for pd.read_feather()
    
    Returns:
        DataFrame
    """
    with open_file(path, 'rb') as f:
        return pd.read_feather(f, **kwargs)


def read_geofeather(path: Union[Path, str], **kwargs):
    """Read feather file as GeoDataFrame using fsspec.
    
    Args:
        path: Path to feather file
        **kwargs: Additional arguments for gpd.read_feather()
    
    Returns:
        GeoDataFrame
    """
    import geopandas as gpd
    with open_file(path, 'rb') as f:
        return gpd.read_feather(f, **kwargs)


def read_json(path: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Read JSON file using fsspec.
    
    Args:
        path: Path to JSON file
        **kwargs: Additional arguments for json.load()
    
    Returns:
        Dictionary
    """
    with open_file(path, 'r') as f:
        return json.load(f, **kwargs)


def read_yaml(path: Union[Path, str], **kwargs) -> Dict[str, Any]:
    """Read YAML file using fsspec.
    
    Args:
        path: Path to YAML file
        **kwargs: Additional arguments for yaml.safe_load()
    
    Returns:
        Dictionary
    """
    with open_file(path, 'r') as f:
        return yaml.safe_load(f, **kwargs)


def open_dataarray(path: Union[Path, str], **kwargs) -> xr.DataArray:
    """Open xarray DataArray using fsspec.
    
    Args:
        path: Path to data file
        **kwargs: Additional arguments for xr.open_dataarray()
    
    Returns:
        xarray DataArray
    """
    # For zarr stores, we can pass the path directly to xarray
    path_str = str(path)
    if path_str.endswith('.zarr') or '://' in path_str:
        return xr.open_dataarray(path_str, **kwargs)
    else:
        # For other formats, pass path directly to xarray (it handles fsspec internally)
        return xr.open_dataarray(path_str, **kwargs)


def open_dataset(path: Union[Path, str], **kwargs) -> xr.Dataset:
    """Open xarray Dataset using fsspec.
    
    Args:
        path: Path to data file
        **kwargs: Additional arguments for xr.open_dataset()
    
    Returns:
        xarray Dataset
    """
    # For zarr stores, we can pass the path directly to xarray
    path_str = str(path)
    if path_str.endswith('.zarr') or '://' in path_str:
        return xr.open_dataset(path_str, **kwargs)
    else:
        # For other formats, pass path directly to xarray (it handles fsspec internally)
        return xr.open_dataset(path_str, **kwargs)