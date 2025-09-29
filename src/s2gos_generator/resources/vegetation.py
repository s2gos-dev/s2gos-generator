"""Vegetation placement resource."""

import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from ..core.context import SceneResourceContext


def process_target_vegetation(
    ctx: SceneResourceContext,
) -> Optional[List[Dict[str, Any]]]:
    """Process multi-species vegetation placement using landcover data.

    Args:
        ctx: Scene resource context

    Returns:
        List of vegetation placement dictionaries with position, rotation, species data
    """
    vegetation_config = getattr(ctx.config, "vegetation_placement", None)
    if vegetation_config is None or not vegetation_config.enabled:
        logging.info("Vegetation disabled - skipping vegetation placement")
        return None

    landcover_path = ctx.dependency_outputs.get("target_landcover")
    dem_path = ctx.dependency_outputs.get("target_dem")

    if landcover_path is None or dem_path is None:
        logging.warning(
            "Landcover or DEM data not available - skipping vegetation placement"
        )
        return None

    vegetation_instances = _process_vegetation_with_shared_datasets(
        landcover_path, dem_path, vegetation_config, "target"
    )

    ctx.vegetation_instances = vegetation_instances

    return vegetation_instances


def _process_vegetation_with_shared_datasets(
    landcover_path, dem_path, vegetation_config, processing_type: str
) -> Optional[List[Dict[str, Any]]]:
    """Multi-species vegetation processing with shared dataset loading for better performance.

    Args:
        landcover_path: Path to landcover data file
        dem_path: Path to DEM data file
        vegetation_config: Vegetation placement configuration with species mapping
        processing_type: Type of processing ("target", "buffer", or "spillover")

    Returns:
        List of vegetation placement dictionaries with position, rotation, and species data
    """
    if not vegetation_config.landcover_species_mapping:
        logging.info(
            f"No species configured for {processing_type} vegetation - skipping"
        )
        return []

    logging.info(
        f"Processing {processing_type} vegetation with {len(vegetation_config.landcover_species_mapping)} landcover classes"
    )

    try:
        with (
            xr.open_dataarray(landcover_path) as landcover_data,
            xr.open_dataarray(dem_path) as dem_data,
        ):
            logging.info(
                f"Loaded datasets - Landcover: {landcover_data.dims}, DEM: {dem_data.dims}"
            )

            y_coords = landcover_data.y.values
            x_coords = landcover_data.x.values
            y_resolution = (
                abs(float(y_coords[1] - y_coords[0])) if len(y_coords) > 1 else 30.0
            )
            x_resolution = (
                abs(float(x_coords[1] - x_coords[0])) if len(x_coords) > 1 else 30.0
            )
            pixel_area_ha = (y_resolution * x_resolution) / 10000.0

            logging.info(
                f"Pixel resolution: {x_resolution:.1f}m × {y_resolution:.1f}m ({pixel_area_ha:.4f} ha per pixel)"
            )

            all_vegetation_instances = []
            random.seed(42)

            for (
                landcover_class,
                species_list,
            ) in vegetation_config.landcover_species_mapping.items():
                if not species_list:
                    continue

                logging.info(
                    f"Processing landcover class {landcover_class} with {len(species_list)} species"
                )

                landcover_mask = landcover_data == landcover_class
                landcover_locations = np.where(landcover_mask)

                if len(landcover_locations[0]) == 0:
                    logging.info(
                        f"No pixels found for landcover class {landcover_class}"
                    )
                    continue

                y_indices, x_indices = landcover_locations
                logging.info(
                    f"Found {len(y_indices)} pixels for landcover class {landcover_class}"
                )

                landcover_instances = _process_landcover_species(
                    y_indices,
                    x_indices,
                    species_list,
                    landcover_data,
                    dem_data,
                    vegetation_config,
                    x_resolution,
                    y_resolution,
                    pixel_area_ha,
                )

                all_vegetation_instances.extend(landcover_instances)
                logging.info(
                    f"Generated {len(landcover_instances)} instances for landcover class {landcover_class}"
                )

            logging.info(
                f"Total vegetation instances before final processing: {len(all_vegetation_instances)}"
            )

            if not all_vegetation_instances:
                logging.info("No vegetation instances generated")
                return []

            logging.info("Applying vectorized elevation lookup...")
            all_vegetation_instances = _batch_elevation_lookup(
                all_vegetation_instances, dem_data
            )
            logging.info(
                f"Completed elevation lookup for {len(all_vegetation_instances)} instances"
            )

            if vegetation_config.min_spacing > 0 and len(all_vegetation_instances) > 1:
                logging.info("Applying optimized spacing filter across all species...")
                all_vegetation_instances = _apply_spacing_filter_optimized(
                    all_vegetation_instances, vegetation_config.min_spacing
                )
                logging.info(
                    f"After spacing filter: {len(all_vegetation_instances)} instances"
                )

            final_instances = []
            for instance in all_vegetation_instances:
                vegetation_instance = {
                    "position": [instance["x"], instance["y"], instance["elevation"]],
                    "rotation": random.uniform(0, vegetation_config.rotation_range),
                    "scale": random.uniform(
                        instance["scale_min"], instance["scale_max"]
                    ),
                    "species": instance["species"],
                    "asset_xml": instance["asset_xml"],
                }
                final_instances.append(vegetation_instance)

            logging.info(
                f"Final vegetation placement: {len(final_instances)} instances across all species"
            )

            return final_instances

    except Exception as e:
        logging.error(f"Error processing vegetation placement: {e}")
        return None


def _process_landcover_species(
    y_indices,
    x_indices,
    species_list,
    landcover_data,
    dem_data,
    vegetation_config,
    x_resolution,
    y_resolution,
    pixel_area_ha,
) -> List[Dict[str, Any]]:
    """Process all species for a specific landcover class.

    Args:
        y_indices, x_indices: Pixel coordinates for this landcover class
        species_list: List of VegetationSpecies for this landcover class
        landcover_data, dem_data: Shared dataset references
        vegetation_config: Global vegetation configuration
        x_resolution, y_resolution, pixel_area_ha: Pixel properties

    Returns:
        List of vegetation instances for this landcover class
    """
    y_coords = landcover_data.y.values
    x_coords = landcover_data.x.values
    landcover_instances = []

    for i in range(len(y_indices)):
        y_idx, x_idx = y_indices[i], x_indices[i]
        pixel_center_y = float(y_coords[y_idx])
        pixel_center_x = float(x_coords[x_idx])

        pixel_instances = []

        for species in species_list:
            base_instances_per_pixel = species.density_per_hectare * pixel_area_ha
            variation = vegetation_config.density_variation * random.uniform(-1, 1)
            instances_per_pixel = max(0, base_instances_per_pixel * (1 + variation))

            max_instances_by_spacing = _calculate_max_instances_per_pixel(
                x_resolution, y_resolution, vegetation_config.min_spacing
            )
            instances_per_pixel = min(instances_per_pixel, max_instances_by_spacing)

            n_instances_base = int(instances_per_pixel)
            n_instances_extra = (
                1 if random.random() < (instances_per_pixel - n_instances_base) else 0
            )
            n_instances = n_instances_base + n_instances_extra

            if n_instances > 0:
                species_positions = _generate_pixel_vegetation_positions(
                    pixel_center_x,
                    pixel_center_y,
                    x_resolution,
                    y_resolution,
                    n_instances,
                    species,
                    vegetation_config,
                )
                pixel_instances.extend(species_positions)

        if len(pixel_instances) > vegetation_config.max_instances_per_pixel:
            pixel_instances = random.sample(
                pixel_instances, vegetation_config.max_instances_per_pixel
            )

        landcover_instances.extend(pixel_instances)

    return landcover_instances


def _generate_pixel_vegetation_positions(
    center_x: float,
    center_y: float,
    x_resolution: float,
    y_resolution: float,
    n_instances: int,
    species,
    vegetation_config,
) -> List[Dict[str, Any]]:
    """Generate vegetation positions for a specific species within a pixel.

    Args:
        center_x, center_y: Pixel center coordinates
        x_resolution, y_resolution: Pixel dimensions
        n_instances: Number of instances to generate
        species: VegetationSpecies configuration
        vegetation_config: Global vegetation configuration

    Returns:
        List of position dictionaries with species information
    """
    positions = []

    half_x = x_resolution / 2.0
    half_y = y_resolution / 2.0
    min_x, max_x = center_x - half_x, center_x + half_x
    min_y, max_y = center_y - half_y, center_y + half_y

    min_spacing = vegetation_config.min_spacing
    max_attempts = min(n_instances * 5, 100)
    attempts = 0

    while len(positions) < n_instances and attempts < max_attempts:
        attempts += 1

        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)

        too_close = False
        if min_spacing > 0 and positions:
            for existing_pos in positions:
                dx = x - existing_pos["x"]
                dy = y - existing_pos["y"]
                if (dx * dx + dy * dy) < (min_spacing * min_spacing):
                    too_close = True
                    break

        if not too_close:
            positions.append(
                {
                    "x": x,
                    "y": y,
                    "elevation": 0.0,
                    "species": species.name,
                    "asset_xml": species.asset_xml_path,
                    "scale_min": species.scale_min,
                    "scale_max": species.scale_max,
                }
            )

    return positions


def save_tree_collection_binary(
    trees: List[Dict[str, Any]], output_path
) -> Dict[str, Any]:
    """Save tree collection as compact NumPy structured array.

    Args:
        trees: List of tree dictionaries with position, rotation, scale
        output_path: Path where to save the binary data

    Returns:
        Dictionary with metadata about the saved tree collection
    """
    if not trees:
        logging.warning("No trees to save")
        return {"count": 0, "bounds": None, "file_size_bytes": 0}

    tree_dtype = np.dtype(
        [("x", "f8"), ("y", "f8"), ("z", "f8"), ("rotation", "f4"), ("scale", "f4")]
    )

    tree_array = np.zeros(len(trees), dtype=tree_dtype)

    for i, tree in enumerate(trees):
        pos = tree["position"]
        tree_array[i] = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(tree["rotation"]),
            float(tree["scale"]),
        )

    np.save(output_path, tree_array)

    bounds = [
        [
            float(np.min(tree_array["x"])),
            float(np.min(tree_array["y"])),
            float(np.min(tree_array["z"])),
        ],
        [
            float(np.max(tree_array["x"])),
            float(np.max(tree_array["y"])),
            float(np.max(tree_array["z"])),
        ],
    ]

    file_size = output_path.stat().st_size if output_path.exists() else 0

    logging.info(
        f"Saved {len(trees)} trees to binary format: {output_path} ({file_size} bytes)"
    )

    return {
        "count": len(trees),
        "bounds": bounds,
        "file_size_bytes": file_size,
        "dtype_info": {
            "x": "float64",
            "y": "float64",
            "z": "float64",
            "rotation": "float32",
            "scale": "float32",
        },
    }


def get_tree_collection_metadata(binary_path) -> Dict[str, Any]:
    """Get metadata about a binary tree collection without loading all data.

    Args:
        binary_path: Path to the binary tree data file

    Returns:
        Dictionary with count, bounds, and other metadata
    """
    try:
        tree_array = np.load(binary_path)

        bounds = [
            [
                float(np.min(tree_array["x"])),
                float(np.min(tree_array["y"])),
                float(np.min(tree_array["z"])),
            ],
            [
                float(np.max(tree_array["x"])),
                float(np.max(tree_array["y"])),
                float(np.max(tree_array["z"])),
            ],
        ]

        file_size = binary_path.stat().st_size if binary_path.exists() else 0

        return {
            "count": len(tree_array),
            "bounds": bounds,
            "file_size_bytes": file_size,
            "dtype_info": {
                "x": "float64",
                "y": "float64",
                "z": "float64",
                "rotation": "float32",
                "scale": "float32",
            },
        }

    except Exception as e:
        logging.error(f"Failed to get metadata from {binary_path}: {e}")
        return {"count": 0, "bounds": None, "file_size_bytes": 0}


def load_tree_collection_binary(binary_path) -> np.ndarray:
    """Load tree collection from binary format as numpy array.

    Args:
        binary_path: Path to the binary tree file (.npy)

    Returns:
        Numpy structured array with tree data (x, y, z, rotation, scale)
        Returns empty array if loading fails
    """
    try:
        tree_data = np.load(binary_path)
        logging.info(
            f"Loaded {len(tree_data)} tree instances from {binary_path} ({tree_data.nbytes} bytes)"
        )
        return tree_data

    except Exception as e:
        logging.error(f"Failed to load tree collection from {binary_path}: {e}")
        tree_dtype = np.dtype(
            [("x", "f8"), ("y", "f8"), ("z", "f8"), ("rotation", "f4"), ("scale", "f4")]
        )
        return np.array([], dtype=tree_dtype)


def _calculate_max_instances_per_pixel(
    x_resolution: float, y_resolution: float, min_spacing: float
) -> int:
    """Calculate maximum instances that can fit in a pixel given minimum spacing.

    Args:
        x_resolution: Pixel width in meters
        y_resolution: Pixel height in meters
        min_spacing: Minimum spacing between instances in meters

    Returns:
        Maximum number of instances that can fit in pixel
    """
    if min_spacing <= 0:
        return 50

    instances_x = max(1, int(x_resolution / min_spacing))
    instances_y = max(1, int(y_resolution / min_spacing))
    max_instances = instances_x * instances_y
    return int(max_instances * 0.75)


def _batch_elevation_lookup(
    positions: List[Dict[str, float]], dem_data: xr.DataArray
) -> List[Dict[str, float]]:
    """Vectorized elevation lookup using scipy's RegularGridInterpolator.

    Args:
        positions: List of vegetation position dictionaries with x, y coordinates
        dem_data: DEM data for elevation queries

    Returns:
        List of vegetation positions with updated elevation values
    """
    if not positions:
        return positions

    x_coords = np.array([pos["x"] for pos in positions])
    y_coords = np.array([pos["y"] for pos in positions])

    try:
        x_grid = dem_data.x.values
        y_grid = dem_data.y.values
        z_values = dem_data.values

        interpolator = RegularGridInterpolator(
            (y_grid, x_grid),
            z_values,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )

        points = np.column_stack([y_coords, x_coords])
        elevations = interpolator(points)
        elevations = np.nan_to_num(elevations, nan=0.0)

    except Exception as e:
        logging.warning(
            f"Vectorized elevation lookup failed: {e}. Falling back to nearest neighbor."
        )
        try:
            interpolator = RegularGridInterpolator(
                (y_grid, x_grid),
                z_values,
                method="nearest",
                bounds_error=False,
                fill_value=0.0,
            )
            points = np.column_stack([y_coords, x_coords])
            elevations = interpolator(points)
            elevations = np.nan_to_num(elevations, nan=0.0)
        except Exception:
            logging.warning("All elevation lookups failed. Using zero elevation.")
            elevations = np.zeros(len(positions))

    for i, pos in enumerate(positions):
        pos["elevation"] = float(elevations[i])

    return positions


def _apply_spacing_filter_optimized(
    positions: List[Dict[str, float]], min_spacing: float
) -> List[Dict[str, float]]:
    """Optimized spacing filter using spatial grid for O(n) average performance.

    Args:
        positions: List of vegetation positions
        min_spacing: Minimum distance between instances in meters

    Returns:
        Filtered list of vegetation positions with minimum spacing enforced
    """
    if len(positions) <= 1 or min_spacing <= 0:
        return positions

    grid_size = max(min_spacing, 1.0)
    position_grid = {}
    filtered_positions = []
    min_spacing_squared = min_spacing * min_spacing

    for pos in positions:
        x, y = pos["x"], pos["y"]

        grid_x = int(x / grid_size)
        grid_y = int(y / grid_size)

        collision_found = False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_key = (grid_x + dx, grid_y + dy)
                if neighbor_key in position_grid:
                    for neighbor_pos in position_grid[neighbor_key]:
                        dx_dist = x - neighbor_pos["x"]
                        dy_dist = y - neighbor_pos["y"]
                        distance_squared = dx_dist * dx_dist + dy_dist * dy_dist

                        if distance_squared < min_spacing_squared:
                            collision_found = True
                            break
                    if collision_found:
                        break
            if collision_found:
                break

        if not collision_found:
            grid_key = (grid_x, grid_y)
            if grid_key not in position_grid:
                position_grid[grid_key] = []
            position_grid[grid_key].append(pos)
            filtered_positions.append(pos)

    return filtered_positions
