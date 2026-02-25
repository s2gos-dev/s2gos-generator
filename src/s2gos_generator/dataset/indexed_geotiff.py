from typing import Any

import geopandas as gpd
import xarray as xr
from dynaconf.utils.boxing import DynaBox
from pydantic import Field, PrivateAttr, field_validator
from s2gos_utils.io import resolver
from s2gos_utils.io.paths import PathRef, open_dataset, read_geofeather
from s2gos_utils.setting import to_pathref
from shapely import Polygon
from upath import UPath

from .dataset import Dataset


def _load_index_gdf(index_path: PathRef):
    """Load the index GeoDataFrame."""
    upath = index_path.upath
    if upath.suffix == ".feather":
        return read_geofeather(upath)
    else:
        raise NotImplementedError(
            f"Index path with extension {upath.suffix} not supported. "
            f"Currently supported: `.feather`"
        )


class IndexedGeoTiff(Dataset):
    """Spatially indexed collection of GeoTIFF tiles.

    Provides efficient spatial querying over a large archive of GeoTIFF files
    using a pre-built GeoDataFrame index (stored as a ``.feather`` file).
    Each row in the index describes one tile and includes a path column
    pointing to the corresponding GeoTIFF relative to ``root_directory``.

    Attributes:
        index_path: Path to the ``.feather`` index file.
        root_directory: Root directory under which tile paths in the index
            are resolved.
        path_column: Column name in the index that holds relative tile paths.
            Auto-detected from any column whose name contains ``"path"`` when
            left as ``None``.
        variable_name: Optional variable name used when opening tiles with
            ``xarray``.
    """

    index_path: PathRef = Field()
    root_directory: PathRef = Field()
    path_column: str | None = Field(default=None)
    variable_name: str | None = Field(default=None)
    _index_gdf: gpd.GeoDataFrame | None = PrivateAttr(default=None)
    _xr_engine: str = PrivateAttr("rasterio")

    def model_post_init(self, __context):
        self._index_gdf = _load_index_gdf(self.index_path)

        # try to infer the path column from the column names
        if self.path_column is None:
            for col in self._index_gdf.columns:
                if "path" in col:
                    self.path_column = col
                    break

    @field_validator("index_path", "root_directory")
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files or directories exist."""
        path = resolver.resolve(v)
        if (not path.exists()) and (path.protocol == "file"):
            raise ValueError(f"Path does not exist: {v}")
        return PathRef(path, v.cid)

    @classmethod
    def from_settings(cls, settings: DynaBox | dict, name: str):
        return cls(
            name=name,
            crs=settings.get("crs", "EPSG:4326"),
            index_path=to_pathref(settings["index_path"]),
            root_directory=to_pathref(settings["root_directory"]),
            path_column=settings.get("path_column", None),
            variable_name=settings.get("variable_name", None),
        )

    def query(self, polygon: Polygon, **kwargs: Any) -> list[UPath]:
        """Return paths of all GeoTIFF tiles that intersect *polygon*.

        Performs a spatial join between the index GeoDataFrame and the
        supplied polygon to identify overlapping tiles.

        Args:
            polygon: Query region in EPSG:4326 coordinates.
            **kwargs: Optional keyword arguments.  ``path_column`` overrides
                the instance-level column name for this query.

        Returns:
            List of authenticated ``UPath`` objects for each matching tile.

        Raises:
            ValueError: If no path column can be determined.
            FileNotFoundError: If no tiles intersect *polygon*.
        """
        # Attempt to find a path column in the index file
        path_column = kwargs.get("path_column", self.path_column)

        if path_column is None:
            raise ValueError(f"Missing index path column to query {self.name}.")

        # use the index to query the file paths.
        gdf = gpd.GeoDataFrame(geometry=[polygon], crs="EPSG:4326")
        selected_products = self._index_gdf.sjoin(gdf.to_crs(self._index_gdf.crs))

        if selected_products.empty:
            raise FileNotFoundError(f"No {self.name} tiles found for the given AOI.")

        relative_paths = selected_products[path_column].unique()
        # Use .upath to get the authenticated UPath, then join with relative paths
        filepaths = [self.root_directory.upath / p for p in relative_paths]

        return filepaths

    def open(self, path: UPath | str, **kwargs: Any) -> xr.Dataset:
        """Open a single GeoTIFF tile as an ``xarray.Dataset``.

        Args:
            path: Path to the GeoTIFF file to open.
            **kwargs: Additional keyword arguments forwarded to
                ``open_dataset`` (e.g. ``chunks`` for Dask).

        Returns:
            An ``xarray.Dataset`` backed by ``rasterio``.
        """
        return open_dataset(path, engine=self._xr_engine, **kwargs)
