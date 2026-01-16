"""Scene generation pipeline with automatic dependency management."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from s2gos_utils.io.paths import mkdir
from s2gos_utils.scene import SceneDescription

from .config import SceneGenConfig
from .context import SceneResourceContext
from .resource_registry import DAGExecutor, ResourceRegistry
from ..resources.definitions import get_resources_for_config, get_stateful_resource_ids


class SceneGenerationPipeline:
    """Scene generation pipeline with automatic dependency resolution.

    This pipeline automatically manages dependencies between scene generation
    steps, ensuring resources are processed in the correct order. The
    architecture is inspired by modern terrain generation systems like
    Microsoft Flight Simulator.
    """

    def __init__(self, config: SceneGenConfig):
        """Initialize the scene generation pipeline.

        Args:
            config: Scene generation configuration
        """
        self.config = config
        self.xml_assets = []
        self.xml_material_libraries = []
        self.registry = ResourceRegistry()
        self._hash_specs = {}
        self._stateful_ids = set()
        self.executor = None  # Created during run() with proper settings
        self._initialized = False

    def initialize(self):
        """Initialize the pipeline - call this before using the pipeline."""
        if self._initialized:
            return

        self._process_xml_scenes()
        self._register_resources()
        self.registry.update_scene_dependencies()
        self._setup_output_directories()
        self._log_registered_resources()

        self._initialized = True
        logging.info(f"Pipeline initialized for scene '{self.config.scene_name}'")

    def _register_resources(self):
        """Register resources from the pipeline definition."""
        resources = get_resources_for_config(self.config, self.xml_assets)

        for res in resources:
            resource_id = res["id"]

            # Handle dynamic dependencies for hamster_data
            dependencies = res["dependencies"]
            if res.get("dynamic_deps") and resource_id == "hamster_data":
                # HAMSTER adapts to what's been registered so far
                dependencies = ["aoi"]
                if any(r["id"] == "buffer_aoi" for r in resources):
                    dependencies.append("buffer_aoi")
                if any(r["id"] == "background_aoi" for r in resources):
                    dependencies.append("background_aoi")

            self.registry.register(resource_id, dependencies, res["func"])
            self._hash_specs[resource_id] = res["hash_spec"]

        # Collect stateful resource IDs
        self._stateful_ids = set(get_stateful_resource_ids())

    def _get_all_assets(self):
        """Get combined list of config assets + XML assets without mutating config."""
        return list(self.config.user_assets) + list(self.xml_assets)

    def _process_xml_scenes(self) -> None:
        if not self.config.xml_scenes:
            return

        from .config import load_assets_from_xml

        for xml_scene_config in self.config.xml_scenes:
            # Generate meaningful prefix from XML filename if not specified
            object_id_prefix = xml_scene_config.object_id_prefix
            if object_id_prefix is None:
                xml_path = xml_scene_config.xml_path.upath
                object_id_prefix = xml_path.stem  # filename without extension

            assets, materials = load_assets_from_xml(
                xml_path=str(xml_scene_config.xml_path),
                base_coordinate=list(xml_scene_config.base_coordinate),
                object_id_prefix=object_id_prefix,
                elevation_offset=xml_scene_config.elevation_offset,
                scale=xml_scene_config.scale,
                fix_blender_coords=xml_scene_config.fix_blender_coords,
                rotation_x=xml_scene_config.rotation_x,
                rotation_y=xml_scene_config.rotation_y,
                rotation_z=xml_scene_config.rotation_z,
                material_mappings=xml_scene_config.material_mappings,
                validate_materials=xml_scene_config.validate_materials,
            )

            self.xml_assets.extend(assets)
            if materials:
                self.xml_material_libraries.append(materials)

    def _setup_output_directories(self) -> None:
        directories = [
            self.config.scene_output_dir,
            self.config.data_dir,
            self.config.meshes_dir,
            self.config.textures_dir,
        ]

        for directory in directories:
            mkdir(directory)

    def _log_registered_resources(self) -> None:
        if not self.registry.resources:
            logging.warning("No resources registered!")
            return

        core_resources = []
        buffer_resources = []
        background_resources = []
        optional_resources = []

        for resource_id in self.registry.resources.keys():
            category = self.registry._categorize_resource(resource_id)
            if category == "core":
                core_resources.append(resource_id)
            elif category == "buffer":
                buffer_resources.append(resource_id)
            elif category == "background":
                background_resources.append(resource_id)
            else:
                optional_resources.append(resource_id)

        total_resources = len(self.registry.resources)
        logging.info(f"Registered {total_resources} resources:")

        if core_resources:
            logging.info(f"  Core ({len(core_resources)}): {', '.join(core_resources)}")
        if buffer_resources:
            logging.info(
                f"  Buffer ({len(buffer_resources)}): {', '.join(buffer_resources)}"
            )
        if background_resources:
            logging.info(
                f"  Background ({len(background_resources)}): {', '.join(background_resources)}"
            )
        if optional_resources:
            logging.info(
                f"  Optional ({len(optional_resources)}): {', '.join(optional_resources)}"
            )

    def run_full_pipeline(self) -> SceneDescription:
        """Execute the complete scene generation pipeline.

        Returns:
            SceneDescription instance with complete scene configuration
        """
        return self.run()

    def run(
        self,
        force_rebuild: Optional[Set[str]] = None,
        disable_cache: bool = False,
    ) -> SceneDescription:
        """Execute the complete scene generation pipeline.

        Args:
            force_rebuild: Set of resource IDs to force rebuild (ignoring cache)
            disable_cache: If True, disable all caching

        Returns:
            SceneDescription instance with complete scene configuration
        """
        self.initialize()

        # Configure executor with caching settings
        self.executor = DAGExecutor(
            self.registry,
            hash_specs=self._hash_specs,
            stateful_ids=self._stateful_ids,
            cache_enabled=not disable_cache,
        )

        try:
            # Collect region materials if defined
            region_materials = (
                self.config.region_material_defs
                if self.config.region_material_defs
                else None
            )

            ctx = SceneResourceContext(
                config=self.config,
                additional_material_libraries=self.xml_material_libraries,
                combined_user_assets=self._get_all_assets(),
            )

            # Add region materials to context if available
            if region_materials:
                ctx.region_materials = region_materials

            # Execute all resources using DAG executor
            _ = self.executor.execute(ctx, force_rebuild=force_rebuild)
            scene_description = getattr(ctx, "scene_description", None)
            if scene_description is None:
                raise RuntimeError("Scene description not found in pipeline results")

            logging.info(f"Pipeline complete: {scene_description.name}")

            return scene_description

        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise

    @property
    def scene_name(self) -> str:
        """Get scene name from configuration."""
        return self.config.scene_name

    def get_resource_dependencies(self) -> Dict[str, List[str]]:
        """Get the current resource dependency graph.

        Returns:
            Dictionary mapping resource IDs to their dependencies
        """
        self.initialize()
        dependencies = {}

        for resource in self.registry.get_resource_list():
            dependencies[resource.id] = resource.dependencies or []

        return dependencies

    def visualize_dag(
        self, output_path: Optional[Path] = None, format: str = "png"
    ) -> Optional[Path]:
        try:
            import graphviz

            self.initialize()
            resources = self.registry.get_resource_list()

            if output_path is None:
                output_path = self.config.scene_output_dir / f"{self.scene_name}_dag"

            dot = graphviz.Digraph(
                comment=f"Scene Generation Pipeline - {self.scene_name}"
            )
            dot.attr(rankdir="TB")
            dot.attr("graph", bgcolor="white", fontname="Arial", fontsize="14")
            dot.attr("node", fontname="Arial", fontsize="12", style="filled")
            dot.attr("edge", fontname="Arial", fontsize="10")

            resource_colors = {
                "aoi": "#90EE90",
                "buffer_aoi": "#98FB98",
                "background_aoi": "#F0FFF0",
                "target_dem": "#87CEEB",
                "buffer_dem": "#B0E0E6",
                "target_landcover": "#DDA0DD",
                "buffer_landcover": "#E6E6FA",
                "background_landcover": "#F8F8FF",
                "target_mesh": "#FFB6C1",
                "buffer_mesh": "#FFC0CB",
                "target_texture": "#FFFFE0",
                "buffer_texture": "#FFFACD",
                "background_texture": "#FFFFF0",
                "user_assets": "#FFA07A",
                "hamster_data": "#20B2AA",
                "scene_description": "#FF6347",
            }

            for resource in resources:
                color = resource_colors.get(resource.id, "#D3D3D3")

                if "aoi" in resource.id:
                    dot.node(resource.id, resource.id, fillcolor=color, shape="ellipse")
                elif resource.id == "scene_description":
                    dot.node(
                        resource.id, resource.id, fillcolor=color, shape="doubleoctagon"
                    )
                elif "mesh" in resource.id or resource.id == "user_assets":
                    dot.node(resource.id, resource.id, fillcolor=color, shape="diamond")
                else:
                    dot.node(resource.id, resource.id, fillcolor=color, shape="box")

            for resource in resources:
                if resource.dependencies:
                    for dependency in resource.dependencies:
                        dot.edge(dependency, resource.id, color="black")

            legend_text = (
                f"Scene Generation Pipeline: {self.scene_name}\\n"
                f"Generated: {self.config.created_at.strftime('%Y-%m-%d %H:%M')}\\n"
                f"Shapes: ○ AOI, ◊ Mesh, □ Array, ⬢ Final"
            )
            dot.attr(label=legend_text)
            dot.attr(labelloc="t")

            output_file = dot.render(str(output_path), format=format, cleanup=True)
            logging.info(f"Pipeline visualization saved to: {output_file}")
            return Path(output_file)

        except ImportError:
            logging.warning(
                "Pipeline visualization not available (graphviz not installed)"
            )
            return None
        except Exception as e:
            logging.warning(f"Could not create pipeline visualization: {e}")
            return None

    def print_resource_summary(self) -> None:
        """Print a summary of the resource dependency graph."""
        self.initialize()
        resources = self.registry.get_resource_list()

        resource_groups = {"Core": [], "Buffer": [], "Background": [], "Optional": []}

        for resource in resources:
            category = self.registry._categorize_resource(resource.id)
            if category == "core":
                resource_groups["Core"].append(resource.id)
            elif category == "buffer":
                resource_groups["Buffer"].append(resource.id)
            elif category == "background":
                resource_groups["Background"].append(resource.id)
            else:
                resource_groups["Optional"].append(resource.id)

        for group_name, resource_ids in resource_groups.items():
            if resource_ids:
                logging.info(f"{group_name}: {', '.join(resource_ids)}")

    def clear_cache(
        self,
        resource_ids: Optional[List[str]] = None,
        delete_files: bool = True,
    ) -> List[Path]:
        """Clear cache for specified resources or all.

        Args:
            resource_ids: List of resource IDs to clear (None = all)
            delete_files: If True, also delete the cached output files

        Returns:
            List of deleted file paths
        """
        from .cache import CacheManifest

        cache_dir = self.config.scene_output_dir.upath
        manifest = CacheManifest(cache_dir)

        files_to_delete = manifest.clear(resource_ids)

        deleted = []
        if delete_files:
            for path in files_to_delete:
                try:
                    if path.is_dir():
                        import shutil
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    deleted.append(path)
                    logging.info(f"Deleted cached file: {path}")
                except Exception as e:
                    logging.warning(f"Failed to delete {path}: {e}")

        manifest.save()

        if resource_ids:
            logging.info(f"Cleared cache for resources: {resource_ids}")
        else:
            logging.info("Cleared all cached resource hashes")

        return deleted

    def get_cache_status(self) -> Dict[str, Dict[str, str]]:
        """Get current cache status for all resources.

        Returns:
            Dictionary mapping resource IDs to their cached hash→path entries
        """
        from .cache import CacheManifest

        cache_dir = self.config.scene_output_dir.upath
        manifest = CacheManifest(cache_dir)

        self.initialize()
        status = {}
        for resource_id in self.registry.resources.keys():
            entries = manifest.get_all_entries(resource_id)
            status[resource_id] = entries

        return status
