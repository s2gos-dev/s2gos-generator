"""Vegetation placement resource."""

import logging
import random
from typing import Any, Dict, List

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from ..core.context import SceneResourceContext
from ..core.exceptions import DataNotFoundError


def _filter_by_exclusion_zones(
    vegetation_instances: List[Dict[str, Any]],
    ctx: SceneResourceContext,
) -> List[Dict[str, Any]]:
    """Filter vegetation instances by exclusion zones.

    Args:
        vegetation_instances: List of vegetation placement dicts
        ctx: Scene resource context with exclusion zones

    Returns:
        Filtered list with instances outside exclusion zones
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    if not vegetation_instances:
        return vegetation_instances

    all_exclusion_zones = getattr(ctx, "vegetation_exclusion_zones", [])
    if not all_exclusion_zones:
        logging.info("No exclusion zones to apply")
        return vegetation_instances

    logging.info(
        f"Applying {len(all_exclusion_zones)} exclusion zones to "
        f"{len(vegetation_instances)} vegetation instances"
    )

    geometries = [zone["geometry"] for zone in all_exclusion_zones]
    spatial_index = STRtree(geometries)

    filtered_instances = []
    excluded_count = 0

    for instance in vegetation_instances:
        pos = instance["position"]
        point = Point(pos[0], pos[1])

        possible_matches = spatial_index.query(point)
        is_excluded = any(geometries[idx].contains(point) for idx in possible_matches)

        if not is_excluded:
            filtered_instances.append(instance)
        else:
            excluded_count += 1

    logging.info(
        f"Exclusion filtering: kept {len(filtered_instances)}, "
        f"excluded {excluded_count} "
        f"({100 * excluded_count / len(vegetation_instances):.1f}%)"
    )

    return filtered_instances


def process_target_vegetation(
    ctx: SceneResourceContext,
) -> List[Dict[str, Any]]:
    """Process multi-species vegetation placement using landcover data.

    Args:
        ctx: Scene resource context

    Returns:
        List of vegetation placement dictionaries with position, rotation, species data.
        Returns empty list if vegetation is disabled or not configured.

    Raises:
        DataNotFoundError: If required landcover or DEM data is missing
    """
    vegetation_config = ctx.config.vegetation_placement
    if vegetation_config is None or not vegetation_config.enabled:
        logging.info("Vegetation disabled - skipping vegetation placement")
        return []

    landcover_path = ctx.dependency_outputs.get("target_landcover")
    dem_path = ctx.dependency_outputs.get("target_dem")

    if landcover_path is None:
        raise DataNotFoundError(
            "Landcover data not available for vegetation placement. "
            "Ensure target_landcover resource is enabled and processed."
        )

    if dem_path is None:
        raise DataNotFoundError(
            "DEM data not available for vegetation placement. "
            "Ensure target_dem resource is enabled and processed."
        )

    from s2gos_utils.io.paths import exists

    if not exists(landcover_path):
        raise DataNotFoundError(f"Landcover file not found: {landcover_path}")
    if not exists(dem_path):
        raise DataNotFoundError(f"DEM file not found: {dem_path}")

    vegetation_instances = _process_vegetation_with_shared_datasets(
        landcover_path, dem_path, vegetation_config
    )
    if getattr(ctx, "vegetation_exclusion_zones", []):
        vegetation_instances = _filter_by_exclusion_zones(vegetation_instances, ctx)

    ctx.vegetation_instances = vegetation_instances

    return vegetation_instances


def _process_vegetation_with_shared_datasets(
    landcover_path, dem_path, vegetation_config
) -> List[Dict[str, Any]]:
    """Multi-species vegetation processing with shared dataset loading for better performance.

    Args:
        landcover_path: Path to landcover data file
        dem_path: Path to DEM data file
        vegetation_config: Vegetation placement configuration with species mapping

    Returns:
        List of vegetation placement dictionaries with position, rotation, and species data.
        Returns empty list if no species configured.
    """
    if not vegetation_config.landcover_species_mapping:
        logging.info("No species configured for vegetation - skipping")
        return []

    logging.info(
        f"Processing vegetation with {len(vegetation_config.landcover_species_mapping)} landcover classes"
    )

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
                logging.info(f"No pixels found for landcover class {landcover_class}")
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

            # Process spillover for each species in this landcover class
            for species in species_list:
                if species.spillover_enabled:
                    logging.info(
                        f"Processing spillover for species '{species.name}' from landcover class {landcover_class}"
                    )
                    spillover_instances = _process_spillover_vegetation(
                        landcover_data,
                        dem_data,
                        landcover_class,
                        species,
                        vegetation_config,
                        x_resolution,
                        y_resolution,
                        pixel_area_ha,
                    )
                    all_vegetation_instances.extend(spillover_instances)

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
            # Shuffle so no species has systematic priority in spacing conflicts.
            random.shuffle(all_vegetation_instances)
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
                "scale": random.uniform(instance["scale_min"], instance["scale_max"]),
                "tilt_x": random.uniform(
                    -vegetation_config.tilt_range, vegetation_config.tilt_range
                ),
                "tilt_y": random.uniform(
                    -vegetation_config.tilt_range, vegetation_config.tilt_range
                ),
                "species": instance["species"],
                "asset_xml": instance["asset_xml"],
            }
            final_instances.append(vegetation_instance)

        logging.info(
            f"Final vegetation placement: {len(final_instances)} instances across all species"
        )

        return final_instances


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


def _process_spillover_vegetation(
    landcover_data: xr.DataArray,
    dem_data: xr.DataArray,
    primary_landcover_class: int,
    species,
    vegetation_config,
    x_resolution: float,
    y_resolution: float,
    pixel_area_ha: float,
) -> List[Dict[str, Any]]:
    """Process spillover vegetation for a species into compatible adjacent landcover classes.

    Spillover allows vegetation to extend naturally from its primary landcover class into
    adjacent compatible classes (e.g., forest trees extending into nearby grassland).

    Performance characteristics:
    - O(n) where n = number of pixels in landcover data (uses scipy distance transform)
    - Efficient numpy operations for mask creation and distance calculation
    - Only processes pixels within max_distance of primary class

    Args:
        landcover_data: Landcover classification data
        dem_data: DEM data for elevation
        primary_landcover_class: The landcover class this species primarily occupies
        species: VegetationSpecies with spillover settings
        vegetation_config: Global vegetation configuration
        x_resolution, y_resolution: Pixel dimensions in meters
        pixel_area_ha: Pixel area in hectares

    Returns:
        List of spillover vegetation instances with position, species metadata
    """
    if not species.spillover_enabled:
        return []

    compatibility = (
        species.spillover_compatibility
        if species.spillover_compatibility is not None
        else vegetation_config.spillover_compatibility
    )

    if not compatibility:
        logging.debug(
            f"Species '{species.name}' has spillover enabled but no compatibility map"
        )
        return []

    primary_mask = (landcover_data == primary_landcover_class).values

    distance_from_primary = distance_transform_edt(~primary_mask)

    pixel_resolution = (x_resolution + y_resolution) / 2.0
    max_distance_pixels = vegetation_config.spillover_max_distance_m / pixel_resolution

    spillover_instances = []

    for target_class, compat_score in compatibility.items():
        if compat_score <= 0:
            continue

        target_mask = landcover_data == target_class
        within_distance = distance_from_primary <= max_distance_pixels
        spillover_candidate_mask = target_mask & within_distance

        y_indices, x_indices = np.where(spillover_candidate_mask)

        if len(y_indices) == 0:
            continue

        logging.info(
            f"Processing spillover for species '{species.name}' into landcover class {target_class}: {len(y_indices)} candidate pixels"
        )

        # Vectorised batch over all candidate pixels
        distances = distance_from_primary[y_indices, x_indices]
        distance_decay = 1.0 - (distances / max_distance_pixels)
        lam = (
            species.density_per_hectare * compat_score * distance_decay * pixel_area_ha
        )
        n_instances_arr = np.random.poisson(lam)
        n_instances_arr = np.minimum(
            n_instances_arr, vegetation_config.max_instances_per_pixel
        )

        active = n_instances_arr > 0
        if not np.any(active):
            continue

        center_y_arr = landcover_data.y.values[y_indices[active]]
        center_x_arr = landcover_data.x.values[x_indices[active]]

        for cy, cx, n in zip(center_y_arr, center_x_arr, n_instances_arr[active]):
            spillover_instances.extend(
                _generate_pixel_vegetation_positions(
                    cx,
                    cy,
                    x_resolution,
                    y_resolution,
                    int(n),
                    species,
                    vegetation_config,
                )
            )

    logging.info(
        f"Generated {len(spillover_instances)} spillover instances for species '{species.name}'"
    )

    return spillover_instances


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

    Uses rejection sampling with spacing constraints. Performance characteristics:
    - Time complexity: O(n²) worst case for dense pixels (each position checks all previous)
    - Limited by max_attempts to prevent infinite loops
    - Typically succeeds quickly for reasonable density/spacing ratios

    Args:
        center_x, center_y: Pixel center coordinates in scene coordinate system
        x_resolution, y_resolution: Pixel dimensions in meters
        n_instances: Number of instances to attempt to place
        species: VegetationSpecies configuration with density and scale parameters
        vegetation_config: Global vegetation configuration (min_spacing, etc.)

    Returns:
        List of position dictionaries with species information.
        May return fewer than n_instances if spacing constraints cannot be satisfied.
    """
    positions = []

    half_x = x_resolution / 2.0
    half_y = y_resolution / 2.0
    min_x, max_x = center_x - half_x, center_x + half_x
    min_y, max_y = center_y - half_y, center_y + half_y

    min_spacing = vegetation_config.min_spacing
    max_attempts = min(n_instances * 5, 100)
    attempts = 0

    asset_paths, asset_weights = species.get_asset_paths_and_weights()

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
            selected_asset = random.choices(asset_paths, weights=asset_weights, k=1)[0]

            positions.append(
                {
                    "x": x,
                    "y": y,
                    "elevation": 0.0,
                    "species": species.name,
                    "asset_xml": selected_asset,
                    "scale_min": species.scale_min,
                    "scale_max": species.scale_max,
                }
            )

    if len(positions) < n_instances:
        logging.warning(
            f"Rejection sampling exhausted {max_attempts} attempts for species "
            f"'{species.name}': placed {len(positions)}/{n_instances}. "
            f"Consider reducing density_per_hectare or min_spacing."
        )

    return positions


def save_vegetation_collection_binary(
    vegetation_instances: List[Dict[str, Any]], output_path
) -> Dict[str, Any]:
    """Save vegetation collection as compact NumPy structured array.

    Args:
        vegetation_instances: List of vegetation dictionaries with position, rotation, scale
        output_path: Path where to save the binary data

    Returns:
        Dictionary with metadata about the saved vegetation collection
    """
    if not vegetation_instances:
        logging.warning("No vegetation instances to save")
        return {"count": 0, "bounds": None, "file_size_bytes": 0}

    vegetation_dtype = np.dtype(
        [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("rotation", "f4"),
            ("scale", "f4"),
            ("tilt_x", "f4"),
            ("tilt_y", "f4"),
        ]
    )

    vegetation_array = np.zeros(len(vegetation_instances), dtype=vegetation_dtype)

    for i, instance in enumerate(vegetation_instances):
        pos = instance["position"]
        vegetation_array[i] = (
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            float(instance["rotation"]),
            float(instance["scale"]),
            float(instance.get("tilt_x", 0.0)),
            float(instance.get("tilt_y", 0.0)),
        )

    np.save(output_path, vegetation_array)

    bounds = [
        [
            float(np.min(vegetation_array["x"])),
            float(np.min(vegetation_array["y"])),
            float(np.min(vegetation_array["z"])),
        ],
        [
            float(np.max(vegetation_array["x"])),
            float(np.max(vegetation_array["y"])),
            float(np.max(vegetation_array["z"])),
        ],
    ]

    file_size = output_path.stat().st_size if output_path.exists() else 0

    logging.info(
        f"Saved {len(vegetation_instances)} vegetation instances to binary format: {output_path} ({file_size} bytes)"
    )

    return {
        "count": len(vegetation_instances),
        "bounds": bounds,
        "file_size_bytes": file_size,
        "dtype_info": {
            "x": "float64",
            "y": "float64",
            "z": "float64",
            "rotation": "float32",
            "scale": "float32",
            "tilt_x": "float32",
            "tilt_y": "float32",
        },
    }


def get_vegetation_collection_metadata(binary_path) -> Dict[str, Any]:
    """Get metadata about a binary vegetation collection without loading all data.

    Args:
        binary_path: Path to the binary vegetation data file

    Returns:
        Dictionary with count, bounds, and other metadata
    """
    try:
        vegetation_array = np.load(binary_path)

        bounds = [
            [
                float(np.min(vegetation_array["x"])),
                float(np.min(vegetation_array["y"])),
                float(np.min(vegetation_array["z"])),
            ],
            [
                float(np.max(vegetation_array["x"])),
                float(np.max(vegetation_array["y"])),
                float(np.max(vegetation_array["z"])),
            ],
        ]

        file_size = binary_path.stat().st_size if binary_path.exists() else 0

        return {
            "count": len(vegetation_array),
            "bounds": bounds,
            "file_size_bytes": file_size,
            "dtype_info": {
                "x": "float64",
                "y": "float64",
                "z": "float64",
                "rotation": "float32",
                "scale": "float32",
                "tilt_x": "float32",
                "tilt_y": "float32",
            },
        }

    except Exception as e:
        logging.error(f"Failed to get metadata from {binary_path}: {e}")
        return {"count": 0, "bounds": None, "file_size_bytes": 0}


def load_vegetation_collection_binary(binary_path) -> np.ndarray:
    """Load vegetation collection from binary format as numpy array.

    Args:
        binary_path: Path to the binary vegetation file (.npy)

    Returns:
        Numpy structured array with vegetation data (x, y, z, rotation, scale, tilt_x, tilt_y)
        Returns empty array if loading fails
    """
    try:
        vegetation_data = np.load(binary_path)
        logging.info(
            f"Loaded {len(vegetation_data)} vegetation instances from {binary_path} ({vegetation_data.nbytes} bytes)"
        )
        return vegetation_data

    except Exception as e:
        logging.error(f"Failed to load vegetation collection from {binary_path}: {e}")
        vegetation_dtype = np.dtype(
            [
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
                ("rotation", "f4"),
                ("scale", "f4"),
                ("tilt_x", "f4"),
                ("tilt_y", "f4"),
            ]
        )
        return np.array([], dtype=vegetation_dtype)


def _calculate_max_instances_per_pixel(
    x_resolution: float,
    y_resolution: float,
    min_spacing: float,
    max_fallback: float = 100,
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
        return max_fallback

    instances_x = max(1, int(x_resolution / min_spacing))
    instances_y = max(1, int(y_resolution / min_spacing))
    max_instances = instances_x * instances_y
    return int(max_instances * 0.75)


def _batch_elevation_lookup(
    positions: List[Dict[str, float]], dem_data: xr.DataArray
) -> List[Dict[str, float]]:
    """Vectorized elevation lookup using scipy's RegularGridInterpolator.

    Performs efficient batch elevation queries using scipy interpolation instead of
    individual xarray selections. This provides significant performance improvement
    for large vegetation datasets.

    Performance characteristics:
    - Time complexity: O(n log m) where n=positions, m=DEM grid points (interpolation)
    - Space complexity: O(n) for coordinate arrays
    - Practical benefit: ~100x faster than per-position xarray lookups

    Args:
        positions: List of vegetation position dictionaries with 'x', 'y' coordinates
        dem_data: DEM data for elevation queries (xarray DataArray)

    Returns:
        List of vegetation positions with 'elevation' field set from DEM interpolation.
        Uses 0.0 for out-of-bounds or invalid positions.
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
    """Optimized spacing filter using spatial grid.

    Args:
        positions: List of vegetation positions with 'x' and 'y' coordinates
        min_spacing: Minimum distance between instances in meters

    Returns:
        Filtered list of vegetation positions with minimum spacing enforced.
        First occurrence is kept, subsequent nearby positions are discarded.
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
