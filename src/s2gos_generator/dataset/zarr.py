import logging
from typing import Any

import geopandas as gpd
import xarray as xr
from dynaconf.utils.boxing import DynaBox
from pydantic import Field, PrivateAttr, field_validator
from s2gos_utils.io import PathRef, resolver
from s2gos_utils.io.paths import open_dataset
from s2gos_utils.setting import to_pathref
from shapely import Polygon, box

from .dataset import Dataset


class Zarr(Dataset):
    """Zarr-backed dataset for cloud-optimised geospatial data.

    Wraps a single Zarr store (local or remote) and provides spatial
    querying by comparing the polygon extent against the dataset's bounding
    box.  The entire store is returned when it intersects the query polygon.

    Attributes:
        path: Path or URL to the Zarr store.
        variable_name: Optional variable name used when slicing the opened
            dataset.
    """

    path: PathRef = Field()
    variable_name: str | None = Field(default=None)
    _xr_engine: str = PrivateAttr("zarr")

    @classmethod
    def from_settings(cls, settings: DynaBox | dict, name: str):
        return cls(
            name=name,
            crs=settings.get("crs", "EPSG:4326"),
            path=to_pathref(settings["path"]),
            variable_name=settings.get("variable_name", None),
        )

    @field_validator("path")
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files or directories exist."""
        path = resolver.resolve(v)
        if not path.exists() and path.protocol == "file":
            raise ValueError(f"Path does not exist: {v}")
        return v

    def query(self, polygon: Polygon, **kwargs: Any) -> list[PathRef]:
        """Return the store path if its spatial extent intersects *polygon*.

        Opens the Zarr store to read coordinate bounds, then checks whether
        the dataset bounding box overlaps the supplied polygon.  Supports
        datasets with ``(x, y)`` or ``(lon, lat)`` coordinate dimensions.

        Args:
            polygon: Query region in EPSG:4326 coordinates.
            **kwargs: Accepted but unused; present for interface compatibility.

        Returns:
            A single-element list ``[self.path]`` when there is spatial
            overlap, or an empty list when there is none.
        """
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
                float(x_coords.min()),
                float(y_coords.min()),
                float(x_coords.max()),
                float(y_coords.max()),
            )

            # Create GeoPandas GeoDataFrames
            dataset_box = box(*dataset_bounds)
            dataset_gdf = gpd.GeoDataFrame(geometry=[dataset_box], crs=dataset_crs)
            polygon_gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")

            polygon_gdf = polygon_gdf.to_crs(dataset_gdf.crs)
            overlaps = dataset_gdf.intersects(polygon_gdf).any()

        return [self.path] if overlaps else []

    def open(self, path: PathRef | None = None, **kwargs: Any) -> xr.Dataset:
        """Open the Zarr store as an ``xarray.Dataset``.

        The ``path`` argument is accepted for interface compatibility but
        ignored; the store is always opened from ``self.path``.

        Args:
            path: Unused. Present for compatibility with the ``Dataset`` base
                class interface.
            **kwargs: Additional keyword arguments forwarded to
                ``open_dataset`` (e.g. ``chunks`` for Dask lazy loading).

        Returns:
            An ``xarray.Dataset`` backed by the Zarr engine.
        """
        return open_dataset(self.path, engine=self._xr_engine, **kwargs)
