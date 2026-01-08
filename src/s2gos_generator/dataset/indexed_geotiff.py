import geopandas as gpd
import xarray as xr
from dynaconf.utils.boxing import DynaBox
from pydantic import Field, PrivateAttr, field_validator
from s2gos_utils.io import expand_mapper, resolver
from s2gos_utils.io.paths import read_geofeather
from s2gos_utils.setting import to_upath
from s2gos_utils.typing import PathLike
from shapely import Polygon

from .dataset import Dataset


def _load_index_gdf(index_path: PathLike):
    """Load the index GeoDataFrame."""
    if index_path.suffix == ".feather":
        return read_geofeather(index_path)
    else:
        raise NotImplementedError(
            f"Index path with extension {index_path.suffix} not supported."
            f"Currently supported: `.feather`"
        )


class IndexedGeoTiff(Dataset):
    index_path: PathLike | None = Field(default=None)
    root_directory: PathLike | None = Field(default=None)
    path_column: str | None = Field(default=None)
    _index_gdf: gpd.GeoDataFrame | None = PrivateAttr(default=None)
    xr_engine: str = "rasterio"

    def model_post_init(self, __context):
        self._index_gdf = _load_index_gdf(self.index_path)

        # try to infer the path column from the column names
        if self.path_column is None:
            for col in self._index_gdf.columns:
                if "path" in col:
                    self.path_column = col
                    break

    @field_validator(
        "index_path",
        "root_directory",
    )
    @classmethod
    def validate_path_exists(cls, v):
        """Validate that local files or directories exist."""
        path = resolver.resolve(v)
        if not path.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @classmethod
    def from_settings(cls, settings: DynaBox | dict, name: str):
        return cls(
            name=name,
            crs=settings.get("crs", "EPSG:4326"),
            index_path=to_upath(settings["index_path"]),
            root_directory=to_upath(settings["root_directory"]),
            path_column=settings.get("path_column", None),
        )

    def query(self, polygon: Polygon, **kwargs) -> list[PathLike]:
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
        filepaths = [self.root_directory / p for p in relative_paths]

        return filepaths

    def open(self, path, **kwargs):
        return xr.open_dataset(expand_mapper(path), engine=self.xr_engine, **kwargs)
