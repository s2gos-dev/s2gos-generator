"""DEM processing for area of interest using Copernicus GLO-30 DEM tiles."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import xarray as xr
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_aoi_polygon(
    center_lat: float,
    center_lon: float,
    side_length_km: float = 200.0
) -> Polygon:
    """Create a square polygon in WGS84 CRS centered at a given point."""
    wgs84_crs = CRS("EPSG:4326")
    local_azimuthal_crs = CRS(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +ellps=WGS84 +units=m"
    )

    transformer_to_local = Transformer.from_crs(wgs84_crs, local_azimuthal_crs, always_xy=True)
    transformer_from_local = Transformer.from_crs(local_azimuthal_crs, wgs84_crs, always_xy=True)

    center_x, center_y = transformer_to_local.transform(center_lon, center_lat)
    half_side_m = (side_length_km * 1000) / 2

    local_corners = [
        (center_x - half_side_m, center_y - half_side_m),
        (center_x + half_side_m, center_y - half_side_m),
        (center_x + half_side_m, center_y + half_side_m),
        (center_x - half_side_m, center_y + half_side_m),
    ]

    lon_lat_coords = [transformer_from_local.transform(x, y) for x, y in local_corners]

    return Polygon(lon_lat_coords)


class DEMProcessor:
    """Finds, merges, and processes Copernicus GLO-30 DEM tiles for a given AOI."""

    def __init__(self, index_path: Path, dem_root_dir: Path):
        """Initialize the DEM processor."""
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found at: {index_path}")
        if not dem_root_dir.is_dir():
            raise NotADirectoryError(f"DEM root directory not found: {dem_root_dir}")

        logging.info("Loading DEM index file...")
        self.index_gdf = gpd.read_feather(index_path)
        self.dem_root_dir = dem_root_dir
        logging.info("DEMProcessor initialized successfully.")

    def _find_intersecting_tiles(self, aoi_polygon: Polygon) -> List[Path]:
        """Find DEM tiles that intersect with the AOI.
        
        Raises:
            FileNotFoundError: If no intersecting DEM tiles are found in the index.
        """
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")
        selected_products = self.index_gdf.sjoin(aoi_gdf.to_crs(self.index_gdf.crs))

        if selected_products.empty:
            raise FileNotFoundError(f"No DEM tiles found for the given AOI.")

        relative_paths = selected_products["path_dem"].unique()
        filepaths = [self.dem_root_dir / p for p in relative_paths]
        
        logging.info(f"Found {len(filepaths)} intersecting DEM tile(s).")
        return filepaths

    def _merge_tiles(
        self,
        tile_paths: List[Path],
        fillna_value: Optional[float] = None
    ) -> xr.Dataset:
        """Merge multiple DEM GeoTIFFs into a single xarray Dataset."""
        logging.info(f"Opening and preparing {len(tile_paths)} tiles for merging...")
        
        data_arrays = []
        for path in tile_paths:
            with xr.open_dataarray(path, engine="rasterio") as da:
                processed_da = (
                    da.isel(band=0, drop=True)
                      .rename("elevation")
                      .rename({"x": "lon", "y": "lat"})
                )
                data_arrays.append(processed_da)

        logging.info("Merging tiles into a single dataset...")
        merged_ds = xr.merge(data_arrays, compat="no_conflicts")

        if fillna_value is not None:
            logging.info(f"Filling NaN values with {fillna_value}.")
            merged_ds = merged_ds.fillna(fillna_value)

        return merged_ds

    def _regrid_dem(
        self, 
        dem_ds: xr.Dataset, 
        target_resolution_m: float, 
        center_lat: float, 
        center_lon: float,
        fillna_value: float,
        aoi_size_km: float
    ) -> xr.Dataset:
        """Regrid DEM to target resolution using oblique mercator projection."""
        import numpy as np
        
        domain_width_m = aoi_size_km * 1000.0
        
        half_width = domain_width_m / 2.0
        num_points = int(round(domain_width_m / target_resolution_m))
        start_coord = -half_width + (target_resolution_m / 2.0)
        end_coord = half_width - (target_resolution_m / 2.0)
        target_x = np.linspace(start_coord, end_coord, num_points)
        target_y = target_x.copy()
        
        from pyproj import Proj
        proj = Proj(f"+proj=omerc +lat_0={center_lat} +lonc={center_lon} +alpha=0 +gamma=0 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m")
        
        if 'y' in dem_ds.dims and 'x' in dem_ds.dims:
            dem_ds = dem_ds.rename({'y': 'lat', 'x': 'lon'})
        
        ys, xs = np.meshgrid(target_y, target_x, indexing='ij')
        lons, lats = proj(xs.ravel(), ys.ravel(), inverse=True)
        lonlats = xr.Dataset({
            "lon": (("y", "x"), lons.reshape(xs.shape)), 
            "lat": (("y", "x"), lats.reshape(xs.shape))
        })
        
        ds_regridded = dem_ds.interp(
            lonlats,
            method="linear",
            kwargs={"bounds_error": False, "fill_value": None}
        )
        ds_regridded = ds_regridded.assign_coords({"y": target_y, "x": target_x})
        
        ds_regridded["elevation"] = ds_regridded["elevation"].fillna(fillna_value).drop_vars(["lon", "lat"], errors='ignore')
        
        return ds_regridded

    def generate_dem(
        self,
        aoi_polygon: Polygon,
        output_path: Path,
        fillna_value: Optional[float] = 0.0,
        target_resolution_m: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        aoi_size_km: Optional[float] = None,
    ) -> xr.Dataset:
        """Generate DEM data for the AOI."""
        tile_paths = self._find_intersecting_tiles(aoi_polygon)
        merged_dem = self._merge_tiles(tile_paths, fillna_value=fillna_value)

        # If target resolution specified, regrid like the notebooks do
        if target_resolution_m is not None and center_lat is not None and center_lon is not None and aoi_size_km is not None:
            logging.info(f"Regridding DEM to {target_resolution_m}m resolution...")
            merged_dem = self._regrid_dem(merged_dem, target_resolution_m, center_lat, center_lon, fillna_value, aoi_size_km)

        logging.info(f"Saving merged DEM to '{output_path}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {"elevation": {"zlib": True, "complevel": 5}}
        merged_dem.to_netcdf(output_path, encoding=encoding)
        logging.info("DEM generation complete.")
        
        return merged_dem


if __name__ == '__main__':
    try:
        DEM_INDEX_PATH = Path('/home/gonzalezm/dreams-scenes/packages/dreams_assets/src/dreams_assets/data/dem_index.feather')
        DEM_ROOT_DIR = Path('/media/DATA/DEM/')
        OUTPUT_DIR = Path('./output/dem_assets')
        
        TARGET_LAT = -23.6002  # Namibia, near Gobabeb
        TARGET_LON = 15.1195
        AOI_SIZE_KM = 200.0
        
        OUTPUT_FILENAME = f"Copernicus_DEM_GLO30_Lat{TARGET_LAT}_Lon{TARGET_LON}_{AOI_SIZE_KM}km.nc"
        output_file_path = OUTPUT_DIR / OUTPUT_FILENAME

        processor = DEMProcessor(index_path=DEM_INDEX_PATH, dem_root_dir=DEM_ROOT_DIR)

        target_aoi = create_aoi_polygon(
            center_lat=TARGET_LAT,
            center_lon=TARGET_LON,
            side_length_km=AOI_SIZE_KM
        )

        final_dem = processor.generate_dem(
            aoi_polygon=target_aoi,
            output_path=output_file_path,
            fillna_value=0.0
        )
        
        print("\n--- Summary ---")
        print(f"Successfully generated DEM and saved to:\n{output_file_path.resolve()}")
        print("\nDEM Dataset Info:")
        print(final_dem)

        try:
            import matplotlib.pyplot as plt
            
            print("\nPlotting the generated DEM...")
            fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
            final_dem["elevation"].plot.imshow(ax=ax, cmap='terrain', center=None)
            ax.set_title(f"Generated DEM for AOI around ({TARGET_LAT}, {TARGET_LON})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.show()

        except ImportError:
            logging.warning("Matplotlib not installed. Skipping visualization.")

    except (FileNotFoundError, NotADirectoryError) as e:
        logging.error(f"A configuration error occurred: {e}")
        logging.error("Please check the paths in the CONFIGURATION section.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}")