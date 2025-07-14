import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import geopandas as gpd
import xarray as xr
from shapely.geometry import Polygon
import rioxarray as rxr
import psutil

from .datautil import regrid_to_projection
from s2gos_utils.io.paths import read_geofeather, exists, open_dataarray



class LandCoverProcessor:
    """Finds, merges, and processes ESA WorldCover land cover tiles for a given AOI."""

    def __init__(self, index_path: Union[Path, str], landcover_root_dir: Union[Path, str]):
        """Initialize the land cover processor."""
        if not exists(index_path):
            raise FileNotFoundError(f"Index file not found at: {index_path}")
        if not exists(landcover_root_dir):
            raise NotADirectoryError(f"Land cover root directory not found: {landcover_root_dir}")

        logging.info("Loading land cover index file...")
        self.index_gdf = read_geofeather(index_path)
        self.landcover_root_dir = Path(landcover_root_dir)
        logging.info("LandCoverProcessor initialized successfully.")

    def _find_intersecting_tiles(self, aoi_polygon: Polygon) -> List[Path]:
        """Find land cover tiles that intersect with the AOI."""
        aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_polygon], crs="EPSG:4326")

        selected_products = self.index_gdf.sjoin(aoi_gdf.to_crs(self.index_gdf.crs))

        if selected_products.empty:
            raise FileNotFoundError(f"No land cover tiles found for the given AOI.")

        relative_paths = selected_products["path_lc"].unique()
        filepaths = [self.landcover_root_dir / p for p in relative_paths]
        
        logging.info(f"Found {len(filepaths)} intersecting land cover tile(s).")
        return filepaths
    
    def _calculate_optimal_chunk_size(self, num_tiles: int) -> int:
        """Calculate optimal chunk size based on available memory and tile count."""
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Target: ~1 million elements per chunk (xarray best practice)
        base_chunk_size = int((1_000_000) ** 0.5)
        
        # Scale down if many tiles or limited memory
        memory_factor = min(1.0, available_memory_gb / 8.0)  # Scale down if < 8GB
        tile_factor = min(1.0, 4.0 / num_tiles) 
        
        chunk_size = int(base_chunk_size * memory_factor * tile_factor)
        chunk_size = max(512, min(chunk_size, 4096))
        
        logging.info(f"Memory: {available_memory_gb:.1f}GB, {num_tiles} tiles → chunk size: {chunk_size}x{chunk_size}")
        return chunk_size

    def _merge_tiles(self, tile_paths: List[Path], aoi_polygon: Optional[Polygon] = None) -> xr.Dataset:
        """Merge multiple land cover GeoTIFFs using Dask with optimal chunks."""
        logging.info(f"Opening {len(tile_paths)} tiles with optimized Dask chunks...")
        
        chunk_size = self._calculate_optimal_chunk_size(len(tile_paths))
        
        bbox = None
        if aoi_polygon:
            bbox = aoi_polygon.bounds
            logging.info(f"Using AOI bbox for early filtering: {bbox}")
        
        data_arrays = []
        for i, path in enumerate(tile_paths):
            logging.info(f"Opening tile {i+1}/{len(tile_paths)}: {path}")
            
            # Open with chunks, uses Dask
            da = rxr.open_rasterio(path, chunks={'x': chunk_size, 'y': chunk_size})
            
            # Early spatial filtering if bbox provided
            if bbox:
                try:
                    # Rough clip to bounding box first to reduce data volume
                    da = da.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[3], bbox[1]))
                except (KeyError, ValueError):
                    # If coordinates don't overlap, skip this early filtering
                    pass
            
            # Process lazily
            processed = (
                da.isel(band=0, drop=True)
                .rename({'x': 'lon', 'y': 'lat'})
                .rename('landcover')
                .astype('uint8')
            )
            
            logging.info(f"  Chunks: {processed.chunks}")
            data_arrays.append(processed)
        
        logging.info("Merging with xr.merge (preserves Dask arrays)...")
        
        merged_ds = xr.merge(data_arrays)
        
        logging.info(f"Merged. Shape: {merged_ds.landcover.shape}")
        logging.info(f"Data type: {type(merged_ds.landcover.data)}")
        
        merged_ds = merged_ds.fillna(7)
        
        return merged_ds

    def _clip_to_aoi(self, dataset: xr.Dataset, aoi_polygon: Polygon) -> xr.Dataset:
        """Clip the dataset to the exact AOI geometry."""
        try:
            logging.info("Clipping dataset to AOI geometry...")
            
            if not hasattr(dataset.rio, 'crs') or dataset.rio.crs is None:
                dataset = dataset.rio.write_crs("EPSG:4326")
            
            if not hasattr(dataset.rio, '_x_dim') or dataset.rio._x_dim is None:
                if 'x' in dataset.dims:
                    dataset = dataset.rio.set_spatial_dims(x_dim='x', y_dim='y')
                elif 'lon' in dataset.dims:
                    dataset = dataset.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
            
            clipped_ds = dataset.rio.clip([aoi_polygon], crs="EPSG:4326", drop=True)
            logging.info("Clipping completed.")
            
            return clipped_ds
            
        except ImportError:
            logging.warning("rioxarray not available, using bounding box clipping instead...")
            bounds = aoi_polygon.bounds  # (minx, miny, maxx, maxy)
            
            if 'x' in dataset.dims and 'y' in dataset.dims:
                x_dim, y_dim = 'x', 'y'
            elif 'lon' in dataset.dims and 'lat' in dataset.dims:
                x_dim, y_dim = 'lon', 'lat'
            else:
                raise ValueError("Dataset must have either (x, y) or (lon, lat) coordinates")
            
            clipped_ds = dataset.sel(
                {x_dim: slice(bounds[0], bounds[2]),
                 y_dim: slice(bounds[3], bounds[1])}
            )
            return clipped_ds

    def _regrid_landcover(
        self,
        landcover_ds: xr.Dataset,
        target_resolution_m: float,
        center_lat: float,
        center_lon: float,
        aoi_size_km: float
    ) -> xr.Dataset:
        """Regrid landcover to target resolution using oblique mercator projection."""
        return regrid_to_projection(
            dataset=landcover_ds,
            target_resolution_m=target_resolution_m,
            center_lat=center_lat,
            center_lon=center_lon,
            aoi_size_km=aoi_size_km,
            interpolation_method="nearest"
        )

    def generate_landcover(
        self,
        aoi_polygon: Polygon,
        output_path: Path,
        target_resolution_m: Optional[float] = None,
        center_lat: Optional[float] = None,
        center_lon: Optional[float] = None,
        aoi_size_km: Optional[float] = None
    ) -> xr.Dataset:
        """Generate landcover data for the AOI."""
        tile_paths = self._find_intersecting_tiles(aoi_polygon)
        
        # Pass AOI to merge for early spatial filtering
        merged_landcover = self._merge_tiles(tile_paths, aoi_polygon)
        
        if target_resolution_m is not None:
            logging.info("Persisting merged data before regridding...")
            merged_landcover = merged_landcover.persist()
        
        clipped_landcover = self._clip_to_aoi(merged_landcover, aoi_polygon)
        
        if target_resolution_m is not None and center_lat is not None and center_lon is not None and aoi_size_km is not None:
            logging.info(f"Regridding landcover to {target_resolution_m}m resolution...")
            clipped_landcover = self._regrid_landcover(clipped_landcover, target_resolution_m, center_lat, center_lon, aoi_size_km)

        logging.info(f"Saving processed land cover to '{output_path}'")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clipped_landcover.to_zarr(output_path, mode="w")
        logging.info("Land cover generation complete.")
        
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
    100: "Moss and lichen"
}