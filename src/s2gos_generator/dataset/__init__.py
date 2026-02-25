from dynaconf.utils.boxing import DynaBox

from .dataset import Dataset
from .indexed_geotiff import IndexedGeoTiff
from .zarr import Zarr

__all__ = [
    "dataset_factory",
    "Dataset",
    "IndexedGeoTiff",
    "Zarr",
]


# key: str = <dataset type>
# value: Class = <dataset class>, should implement a from_settings function.
# Note: Consider moving to a fully fledge factory if we need external Dataset extension.
_id_to_class = {
    "indexed-geotiff": IndexedGeoTiff,
    "zarr": Zarr,
}


def dataset_factory(settings: dict | DynaBox, name=None) -> Dataset:
    try:
        return _id_to_class[settings["type"]].from_settings(settings, name)
    except ValueError:
        raise ValueError(f"This dataset settings type does not exist: {settings.type}")
