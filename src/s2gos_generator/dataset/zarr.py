import logging

import geopandas as gpd
import xarray as xr
from dynaconf.utils.boxing import DynaBox
from pydantic import Field, PrivateAttr, field_validator
from s2gos_utils.io import expand_mapper, resolver
from s2gos_utils.setting import to_upath
from s2gos_utils.typing import PathLike
from shapely import Polygon, box

from .dataset import Dataset


class Zarr(Dataset):
    path: PathLike = Field()
    variable_name: str | None = Field(default=None)
    _xr_engine: str = PrivateAttr("zarr")

    @classmethod
    def from_settings(cls, settings: DynaBox | dict, name: str):
        return cls(
            name=name,
            crs=settings.get("crs","EPSG:4326"),
            path=to_upath(settings["path"]),
            variable_name=settings.get("variable_name",None),
        )

    @field_validator(
        "path",
    )
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files or directories exist."""
        path = resolver.resolve(v)
        if not path.exists() and path.protocol == "file":
            raise ValueError(f"Path does not exist: {v}")
        return v


    def query(self, polygon: Polygon, **kwargs) -> list[PathLike]:
        with self.open() as ds: 
            # Detect coordinate system (fix elif bug)
            if "x" in ds.indexes and "y" in ds.indexes:
                x_dim, y_dim = "x", "y"
            elif "lon" in ds.indexes and "lat" in ds.indexes:
                x_dim, y_dim = "lon", "lat"
            else:
                logging.warning(
                    f"Dataset {self.name} has no valid coordinate system (x,y) or (lon,lat). "
                    "Cannot determine spatial overlap."
                )
                return []
            
            dataset_crs = self.crs  # Default from Dataset base class

            # Compute dataset bounds efficiently
            x_coords = ds[x_dim].values
            y_coords = ds[y_dim].values

            if len(x_coords) == 0 or len(y_coords) == 0:
                return []

            dataset_bounds = (
                float(x_coords.min()), float(y_coords.min()),
                float(x_coords.max()), float(y_coords.max())
            )

            # Create GeoPandas GeoDataFrames
            dataset_box = box(*dataset_bounds)
            dataset_gdf = gpd.GeoDataFrame(geometry=[dataset_box], crs=dataset_crs)
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

            polygon_gdf = polygon_gdf.to_crs(dataset_gdf.crs)
            overlaps = dataset_gdf.intersects(polygon_gdf).any()

        return [self.path] if overlaps else []

     
    def open(self, path=None, **kwargs):
        return xr.open_dataset(
            expand_mapper(self.path), engine=self._xr_engine, **kwargs
        )


