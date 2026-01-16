"""Clean resource registry and DAG execution without singleton pattern."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel
from upath import UPath

from .cache import CacheManifest, HashSpec, compute_hash


class ResourceContext(BaseModel):
    """Base context for resource execution."""

    kwargs: Dict = {}
    dependency_outputs: Dict[str, Path | None] = {}

    model_config = {"extra": "allow"}


class Resource:
    """Represents a single resource with dependencies."""

    def __init__(self, id: str, dependencies: List[str], func: Callable):
        self.id = id
        self.dependencies = dependencies or []
        self.func = func

    def __call__(self, ctx: ResourceContext) -> Optional[Path]:
        """Execute the resource function."""
        return self.func(ctx)


class ResourceRegistry:
    """Registry for managing resource definitions without singleton pattern."""

    def __init__(self):
        self.resources: Dict[str, Resource] = {}

    def register(self, id: str, dependencies: List[str], func: Callable):
        """Register a resource explicitly."""
        self.resources[id] = Resource(id, dependencies, func)

    def get_resource(self, id: str) -> Resource:
        """Get a resource by ID."""
        if id not in self.resources:
            raise ValueError(f"Resource '{id}' not found")
        return self.resources[id]

    def get_resource_list(self) -> List[Resource]:
        """Get all registered resources."""
        return list(self.resources.values())

    def get_execution_order(self) -> List[str]:
        """Get resources in dependency order using topological sort with cycle detection."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(resource_id: str):
            if resource_id in temp_visited:
                raise ValueError(
                    f"Circular dependency detected involving: {resource_id}"
                )
            if resource_id in visited:
                return

            temp_visited.add(resource_id)

            if resource_id not in self.resources:
                raise ValueError(f"Missing dependency: {resource_id}")

            resource = self.resources[resource_id]

            for dep in resource.dependencies:
                if dep not in self.resources:
                    raise ValueError(f"Missing dependency: {dep} for {resource_id}")
                visit(dep)

            temp_visited.remove(resource_id)
            visited.add(resource_id)
            result.append(resource_id)

        for resource_id in self.resources:
            if resource_id not in visited:
                visit(resource_id)

        return result

    def filter_by_config(self, config):
        """Remove disabled resources from registry based on configuration."""
        resources_to_remove = []

        if not getattr(config, "enable_buffer", False):
            resources_to_remove.extend(
                [
                    "buffer_aoi",
                    "buffer_dem",
                    "buffer_landcover",
                    "buffer_mesh",
                    "buffer_texture",
                ]
            )

        if not getattr(config, "enable_background", False):
            resources_to_remove.extend(
                ["background_aoi", "background_landcover", "background_texture"]
            )

        # User assets
        if not getattr(config, "user_assets", None):
            resources_to_remove.append("user_assets")

        # HAMSTER data
        if not getattr(config, "hamster", None) or not getattr(
            config.hamster, "enabled", False
        ):
            resources_to_remove.append("hamster_data")

        # Remove disabled resources
        for resource_id in resources_to_remove:
            if resource_id in self.resources:
                del self.resources[resource_id]

    def _categorize_resource(self, resource_id: str) -> str:
        """Categorize a resource by its ID."""
        if resource_id in {"aoi", "scene_description"} or resource_id.startswith(
            "target_"
        ):
            return "core"
        elif resource_id.startswith("buffer_"):
            return "buffer"
        elif resource_id.startswith("background_"):
            return "background"
        else:
            return "optional"

    def update_scene_dependencies(self):
        """Update scene_description dependencies based on currently registered resources."""
        if "scene_description" not in self.resources:
            return

        # Base dependencies that are always required
        base_dependencies = ["target_mesh", "target_texture"]

        # Dynamically find optional dependencies that contribute to scene
        optional_dependencies = []
        for resource_id in self.resources:
            if resource_id == "scene_description":
                continue

            category = self._categorize_resource(resource_id)
            # Include buffer/background meshes and textures, plus optional resources
            if (
                category in ["buffer", "background"]
                and resource_id.endswith(("_mesh", "_texture"))
            ) or category == "optional":
                optional_dependencies.append(resource_id)

        # Include target_vegetation if registered (it provides data to scene_description)
        if "target_vegetation" in self.resources:
            optional_dependencies.append("target_vegetation")

        # Update scene_description dependencies
        scene_resource = self.resources["scene_description"]
        scene_resource.dependencies = base_dependencies + optional_dependencies


class DAGExecutor:
    """Executes resources in dependency order with caching support."""

    def __init__(
        self,
        registry: ResourceRegistry,
        hash_specs: Dict[str, HashSpec] = None,
        stateful_ids: Set[str] = None,
        cache_enabled: bool = True,
    ):
        self.registry = registry
        self.hash_specs = hash_specs or {}
        self.stateful_ids = stateful_ids or set()
        self.cache_enabled = cache_enabled
        self._manifest: Optional[CacheManifest] = None
        self._hashes: Dict[str, str] = {}

    def execute(
        self,
        context: ResourceContext,
        force_rebuild: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Execute all resources in dependency order with cache checking."""
        execution_order = self.registry.get_execution_order()
        results = {}
        force_rebuild = force_rebuild or set()

        # Initialize cache manifest
        if self.cache_enabled and hasattr(context, "config"):
            cache_dir = context.config.scene_output_dir.upath
            self._manifest = CacheManifest(cache_dir)

        for resource_id in execution_order:
            try:
                resource = self.registry.get_resource(resource_id)
                context.dependency_outputs = results

                # Compute hash for this resource
                current_hash = self._compute_hash(resource_id, context)
                self._hashes[resource_id] = current_hash

                # Check cache
                cached_entry = None
                if self._manifest and resource_id not in force_rebuild:
                    cached_entry = self._manifest.get(resource_id, current_hash)

                # Resources that modify context state must always execute
                is_stateful = resource_id in self.stateful_ids

                # cached_entry can be:
                # - UPath: file path that exists (use as cache hit)
                # - True: non-file resource was computed (need to re-execute for state)
                # - None: no cache entry
                is_file_cache_hit = isinstance(cached_entry, (Path, UPath))

                if is_file_cache_hit and not is_stateful:
                    results[resource_id] = cached_entry
                    logging.info(f"Cache hit: {resource_id} ({current_hash[:8]})")
                else:
                    # Set cache hash for filename construction
                    context.cache_hash = current_hash[:8]
                    result = resource(context)
                    results[resource_id] = result

                    if self._manifest:
                        self._manifest.set(resource_id, current_hash, result)

            except Exception as e:
                raise RuntimeError(f"Resource '{resource_id}' failed: {e}") from e

        # Save manifest
        if self._manifest:
            self._manifest.save()

        return results

    def _compute_hash(self, resource_id: str, context: ResourceContext) -> str:
        """Compute hash for a resource based on its config dependencies."""
        spec = self.hash_specs.get(resource_id, HashSpec())
        resource = self.registry.get_resource(resource_id)
        dep_hashes = {dep: self._hashes.get(dep, "") for dep in resource.dependencies}
        return compute_hash(resource_id, context.config, spec, dep_hashes)
