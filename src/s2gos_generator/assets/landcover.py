from typing import Optional, Union

import xarray as xr
from shapely.geometry import Polygon
from upath import UPath

from .base_processor import BaseTileProcessor
from ..dataset import Dataset, IndexedGeoTiff, Zarr


class LandCoverProcessor(BaseTileProcessor):
    """Finds, merges, and processes ESA WorldCover land cover tiles for a given AOI."""

    def __init__(self, dataset: Dataset):
        """Initialize the land cover processor."""
        super().__init__(dataset)

    @property
    def data_variable_name(self) -> str:
        """Name of the data variable in the processed dataset."""
        var_name = self.dataset.variable_name
        return var_name if var_name is not None else "landcover"

    @property
    def default_interpolation_method(self) -> str:
        """Default interpolation method for land cover regridding."""
        return "nearest"

    @property
    def data_type(self) -> Optional[str]:
        """Data type to cast the data to (landcover uses uint8)."""
        return "uint8"

    @property
    def default_fill_value(self) -> Union[float, int]:
        """Default fill value for NaN values in landcover data."""
        return 7

    @property
    def use_context_manager(self) -> bool:
        """Landcover processor uses direct assignment for file opening."""
        return False

    def generate_landcover(
        self,
        aoi_polygon: Polygon,
        output_path: UPath,
        target_resolution_m: float = 10.0,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        aoi_size_km: Optional[float] = None,
    ) -> xr.Dataset:
        """Generate landcover data for the AOI with configurable resolution.

        Args:
            aoi_polygon: Area of interest polygon
            output_path: Path where to save the processed landcover data
            target_resolution_m: Target resolution in meters (default: 10.0 for native WorldCover)
            center_lat: Center latitude for projection (required for non-native resolution)
            center_lon: Center longitude for projection (required for non-native resolution)
            aoi_size_km: Size of the AOI in kilometers (required for non-native resolution)

        Returns:
            Processed landcover dataset
        """

        tile_paths = self.dataset.query(aoi_polygon)

        if isinstance(self.dataset, IndexedGeoTiff):
            # Pass AOI to merge for early spatial filtering
            merged_landcover = self._merge_tiles(tile_paths, aoi_polygon)
            merged_landcover = merged_landcover.persist()
        elif isinstance(self.dataset, Zarr):
            merged_landcover = self.dataset.open()
        else:
            raise NotImplementedError(
                "This type of dataset is not supported for landcovers."
            )

        # Clip to exact AOI geometry
        clipped_landcover = self._clip_to_aoi(merged_landcover, aoi_polygon)

        # Apply regridding if resolution differs from native (10m) or projection is requested
        if target_resolution_m != 10.0 or (
            center_lat is not None
            and center_lon is not None
            and aoi_size_km is not None
        ):
            if center_lat is None or center_lon is None or aoi_size_km is None:
                raise ValueError(
                    "center_lat, center_lon, and aoi_size_km are required for regridding operations"
                )

            clipped_landcover = self._regrid_data(
                clipped_landcover,
                target_resolution_m,
                center_lat,
                center_lon,
                aoi_size_km,
                fillna_value=self.default_fill_value,
            )

        clipped_landcover = clipped_landcover.rename(
            {self.data_variable_name: "landcover"}
        )

        self._save_dataset(clipped_landcover, output_path)

        return clipped_landcover


ESA_LANDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
