"""Tree placement resource."""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from ..core.context import SceneResourceContext


def process_target_trees(ctx: SceneResourceContext) -> Optional[List[Dict[str, Any]]]:
    """Process tree placement for the target area based on landcover data.

    Args:
        ctx: Scene resource context

    Returns:
        List of tree placement dictionaries with position and rotation data
    """
    # Check if trees are enabled (will be added to config)
    trees_enabled = getattr(ctx.config, 'trees_enabled', False)
    if not trees_enabled:
        logging.info("Trees disabled - skipping tree placement")
        return None

    # Get landcover and DEM data paths from dependencies
    landcover_path = ctx.dependency_outputs.get("target_landcover")
    dem_path = ctx.dependency_outputs.get("target_dem")
    
    if landcover_path is None or dem_path is None:
        logging.warning("Landcover or DEM data not available - skipping tree placement")
        return None

    logging.info("Processing tree placement based on landcover data")

    try:
        # Load landcover data
        landcover_data = xr.open_dataarray(landcover_path)
        logging.info(f"Loaded landcover data: {landcover_data.dims}")

        # Load DEM data for elevation queries
        dem_data = xr.open_dataarray(dem_path)
        logging.info(f"Loaded DEM data: {dem_data.dims}")

        # Find treecover pixels (ESA WorldCover: treecover = 10)
        treecover_mask = landcover_data == 10
        
        # Get coordinates where trees can be placed
        tree_locations = np.where(treecover_mask)
        if len(tree_locations[0]) == 0:
            logging.warning("No treecover areas found - no trees will be placed")
            return []

        # Convert array indices to coordinates
        y_coords = landcover_data.y.values[tree_locations[0]]
        x_coords = landcover_data.x.values[tree_locations[1]]

        logging.info(f"Found {len(y_coords)} potential tree locations in treecover areas")

        # Sample tree positions (use a simple density-based approach)
        # Place trees at ~10% of available treecover pixels for reasonable density
        n_trees = max(1, int(len(y_coords) * 0.1))
        
        # Random sampling of tree positions
        random.seed(42)  # Fixed seed for reproducible results
        sampled_indices = random.sample(range(len(y_coords)), min(n_trees, len(y_coords)))
        
        tree_instances = []
        for idx in sampled_indices:
            x, y = float(x_coords[idx]), float(y_coords[idx])
            
            # Query elevation from DEM
            try:
                elevation = float(dem_data.interp(x=x, y=y, method='linear'))
            except Exception:
                # Fallback to nearest neighbor if linear interpolation fails
                elevation = float(dem_data.sel(x=x, y=y, method='nearest'))
            
            # Random rotation around Z-axis (0-360 degrees)
            rotation_z = random.uniform(0, 360)
            
            tree_instance = {
                "position": [x, y, elevation],
                "rotation": rotation_z,
                "scale": random.uniform(0.8, 1.2)  # Slight scale variation
            }
            tree_instances.append(tree_instance)

        logging.info(f"Generated {len(tree_instances)} tree instances")
        
        # Store tree instances in context for scene assembly
        ctx.tree_instances = tree_instances
        
        return tree_instances

    except Exception as e:
        logging.error(f"Error processing tree placement: {e}")
        return None
    finally:
        # Close datasets to free memory
        if 'landcover_data' in locals():
            landcover_data.close()
        if 'dem_data' in locals():
            dem_data.close()