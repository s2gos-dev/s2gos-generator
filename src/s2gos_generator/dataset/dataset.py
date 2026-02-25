from abc import ABC
from typing import Any

from pydantic import BaseModel, Field
from s2gos_utils.typing import PathLike
from shapely import Polygon


class Dataset(ABC, BaseModel):
    name: str = Field(description="Name of the dataset, used for logging.")
    crs: str = Field(default="EPSG:4326", description="Coordinate reference system.")

    def query(self, polygon: Polygon, ctx: dict | None = None) -> list[PathLike]:
        """
        Use this function to query whether data is present within a polygon shape.

        Args:
            polygon: shape to query against.
            ctx: context used to pass additional query information e.g. time.

        Returns:
            List of paths of data files that have overlapping spatial  `polygon`.
        """

        raise NotImplementedError(
            "This dataset does not have the capacity to be queried from a polygon"
        )

    def open(self, path=None) -> Any:
        """
        Use this function to open the dataset.

        Args:
            path (default=None): specific file to open in the dataset.

        Returns:
            The opened dataset. Can be used within a `with` statement.
        """

        raise NotImplementedError(
            "This dataset does not have the capacity to be opened"
        )
