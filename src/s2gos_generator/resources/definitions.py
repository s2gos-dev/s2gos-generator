"""Scene generation pipeline resource definitions.

This module defines all resources for the scene generation pipeline,
including their dependencies, hash specifications for caching, and
conditional inclusion based on configuration.
"""

from typing import Any, Dict, List

from .aoi import generate_aoi, generate_background_aoi, generate_buffer_aoi
from .assets import process_user_assets
from .dem import process_buffer_dem, process_target_dem
from .hamster import process_hamster_data
from .landcover import (
    process_background_landcover,
    process_buffer_landcover,
    process_target_landcover,
)
from .mesh import generate_buffer_mesh, generate_target_mesh
from .scene import create_scene_description
from .texture import (
    generate_background_texture,
    generate_buffer_texture,
    generate_target_texture,
)
from .vegetation import process_target_vegetation
from ..core.cache import HashSpec


def _material_regions_for(area: str):
    """Create custom hash function for material regions filtered by area."""

    def fn(config):
        regions = getattr(config, "material_regions", []) or []
        return [r.model_dump() for r in regions if area in r.applies_to]

    return fn


SCENE_RESOURCES = [
    # === Core resources (always included) ===
    {
        "id": "aoi",
        "dependencies": [],
        "func": generate_aoi,
        "hash_spec": HashSpec(
            config_paths=[
                "location.center_lat",
                "location.center_lon",
                "location.aoi_size_km",
            ],
            include_deps=False,
        ),
        "stateful": True,
    },
    {
        "id": "target_dem",
        "dependencies": ["aoi"],
        "func": process_target_dem,
        "hash_spec": HashSpec(
            config_paths=[
                "target_resolution_m",
                "data_sources.dem.name",
                "processing.dem_fillna_value",
                "processing.flatten_dem",
            ],
        ),
    },
    {
        "id": "target_landcover",
        "dependencies": ["aoi"],
        "func": process_target_landcover,
        "hash_spec": HashSpec(
            config_paths=["target_resolution_m", "data_sources.landcover.name"],
        ),
    },
    {
        "id": "target_mesh",
        "dependencies": ["target_dem"],
        "func": generate_target_mesh,
        "hash_spec": HashSpec(
            config_paths=["processing.handle_dem_nans"],
        ),
    },
    {
        "id": "target_texture",
        "dependencies": ["target_landcover"],
        "func": generate_target_texture,
        "hash_spec": HashSpec(
            config_paths=[
                "processing.generate_texture_preview",
                "apply_seasonal_snow",
                "snow_season_month",
                "snow_material_index",
            ],
            custom_fn=_material_regions_for("target"),
        ),
    },
    # === Buffer resources (conditional on enable_buffer) ===
    {
        "id": "buffer_aoi",
        "dependencies": ["aoi"],
        "func": generate_buffer_aoi,
        "hash_spec": HashSpec(
            config_paths=[
                "location.center_lat",
                "location.center_lon",
                "buffer_size_km",
            ],
        ),
        "stateful": True,
        "condition": lambda cfg: cfg.enable_buffer,
    },
    {
        "id": "buffer_dem",
        "dependencies": ["buffer_aoi"],
        "func": process_buffer_dem,
        "hash_spec": HashSpec(
            config_paths=[
                "buffer_resolution_m",
                "data_sources.dem.name",
                "processing.dem_fillna_value",
                "processing.flatten_dem",
            ],
        ),
        "condition": lambda cfg: cfg.enable_buffer,
    },
    {
        "id": "buffer_landcover",
        "dependencies": ["buffer_aoi"],
        "func": process_buffer_landcover,
        "hash_spec": HashSpec(
            config_paths=["buffer_resolution_m", "data_sources.landcover.name"],
        ),
        "condition": lambda cfg: cfg.enable_buffer,
    },
    {
        "id": "buffer_mesh",
        "dependencies": ["buffer_dem"],
        "func": generate_buffer_mesh,
        "hash_spec": HashSpec(
            config_paths=["processing.handle_dem_nans"],
        ),
        "condition": lambda cfg: cfg.enable_buffer,
    },
    {
        "id": "buffer_texture",
        "dependencies": ["buffer_landcover"],
        "func": generate_buffer_texture,
        "hash_spec": HashSpec(
            config_paths=[
                "processing.generate_texture_preview",
                "apply_seasonal_snow",
                "snow_season_month",
                "snow_material_index",
            ],
            custom_fn=_material_regions_for("buffer"),
        ),
        "condition": lambda cfg: cfg.enable_buffer,
    },
    # === Background resources (conditional on enable_background) ===
    {
        "id": "background_aoi",
        "dependencies": ["aoi"],
        "func": generate_background_aoi,
        "hash_spec": HashSpec(
            config_paths=[
                "location.center_lat",
                "location.center_lon",
                "background_size_km",
            ],
        ),
        "stateful": True,
        "condition": lambda cfg: cfg.enable_background,
    },
    {
        "id": "background_landcover",
        "dependencies": ["background_aoi"],
        "func": process_background_landcover,
        "hash_spec": HashSpec(
            config_paths=["background_resolution_m", "data_sources.landcover.name"],
        ),
        "condition": lambda cfg: cfg.enable_background,
    },
    {
        "id": "background_texture",
        "dependencies": ["background_landcover"],
        "func": generate_background_texture,
        "hash_spec": HashSpec(
            config_paths=["processing.generate_texture_preview"],
            custom_fn=_material_regions_for("background"),
        ),
        "condition": lambda cfg: cfg.enable_background,
    },
    # === Optional resources ===
    {
        "id": "user_assets",
        "dependencies": ["target_dem"],
        "func": process_user_assets,
        "hash_spec": HashSpec(config_paths=["user_assets"]),
        "condition": lambda cfg, ctx: bool(cfg.user_assets) or bool(ctx.get("xml_assets")),
    },
    {
        "id": "hamster_data",
        "dependencies": ["aoi"],  # Will be updated dynamically
        "func": process_hamster_data,
        "hash_spec": HashSpec(config_paths=["hamster"]),
        "condition": lambda cfg: cfg.hamster and cfg.hamster.enabled,
        "dynamic_deps": True,  # Dependencies depend on what's registered
    },
    {
        "id": "target_vegetation",
        "dependencies": ["target_landcover", "target_dem"],
        "func": process_target_vegetation,
        "hash_spec": HashSpec(config_paths=["vegetation_placement"]),
        "stateful": True,
        "condition": lambda cfg: cfg.trees_enabled,
    },
    # === Final scene description (always included) ===
    {
        "id": "scene_description",
        "dependencies": ["target_mesh", "target_texture"],  # Will be updated dynamically
        "func": create_scene_description,
        "hash_spec": HashSpec(config_paths=[], include_deps=True),
        "stateful": True,
        "dynamic_deps": True,  # Dependencies updated based on registered resources
    },
]


def get_resources_for_config(config, xml_assets: List = None) -> List[Dict[str, Any]]:
    """Filter and return resources applicable for the given configuration.

    Args:
        config: SceneGenConfig instance
        xml_assets: List of XML assets (for user_assets condition)

    Returns:
        List of resource definitions that should be registered
    """
    result = []
    ctx = {"xml_assets": xml_assets or []}

    for res in SCENE_RESOURCES:
        condition = res.get("condition")
        if condition is None:
            result.append(res)
        elif callable(condition):
            try:
                # Try (config, ctx) signature first
                if condition(config, ctx):
                    result.append(res)
            except TypeError:
                # Fall back to (config) signature
                if condition(config):
                    result.append(res)

    return result


def get_stateful_resource_ids() -> List[str]:
    """Return list of resource IDs that are stateful (must always execute)."""
    return [res["id"] for res in SCENE_RESOURCES if res.get("stateful", False)]
