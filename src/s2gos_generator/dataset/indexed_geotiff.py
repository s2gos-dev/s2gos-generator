import geopandas as gpd
import xarray as xr
from dynaconf.utils.boxing import DynaBox
from pydantic import Field, PrivateAttr, field_validator
from s2gos_utils.io import expand_mapper, resolver
from s2gos_utils.io.paths import read_geofeather
from s2gos_utils.setting import to_pathref
from s2gos_utils.typing import PathRef
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
            variable_name=settings.get("variable_name",None),
        )

    def query(self, polygon: Polygon, **kwargs) -> list[UPath]:
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

    def open(self, path, **kwargs):
        return xr.open_dataset(expand_mapper(path), engine=self._xr_engine, **kwargs)
