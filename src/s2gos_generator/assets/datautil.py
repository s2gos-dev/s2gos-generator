"""Data utilities for asset processing."""

import importlib.resources
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd


def read_feather_index(file_path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Load a feather index file into a GeoDataFrame."""
    return gpd.read_feather(file_path)


def read_dem_index() -> pd.DataFrame:
    """Load the DEM index (vendored Apache Feather file) into a Pandas DataFrame."""
    with importlib.resources.open_binary(
        "dreams_assets.data", "dem_index.feather"
    ) as f:
        df = gpd.read_feather(f)
    return df


def validate_data_paths(dem_index_path: Path, landcover_index_path: Path, 
                       dem_root_dir: Path, landcover_root_dir: Path) -> bool:
    """Validate that required data paths exist and are accessible."""
    paths_to_check = [dem_index_path, landcover_index_path, dem_root_dir, landcover_root_dir]
    
    for path in paths_to_check:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    
    if not dem_root_dir.is_dir():
        raise NotADirectoryError(f"DEM root is not a directory: {dem_root_dir}")
    
    if not landcover_root_dir.is_dir():
        raise NotADirectoryError(f"Landcover root is not a directory: {landcover_root_dir}")
    
    return True
