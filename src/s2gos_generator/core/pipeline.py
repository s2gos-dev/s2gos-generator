"""Scene generation pipeline with automatic dependency management."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from s2gos_utils.io.paths import mkdir
from s2gos_utils.scene import SceneDescription

from .config import SceneGenConfig
from .context import SceneResourceContext
from .resource_registry import DAGExecutor, ResourceRegistry


class SceneGenerationPipeline:
    """Scene generation pipeline with automatic dependency resolution.

    This pipeline automatically manages dependencies between scene generation
    steps, ensuring resources are processed in the correct order.
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
        self.executor = DAGExecutor(self.registry)
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
        """Register all pipeline resources and their inter-dependencies."""
        # Import resource functions
        from ..resources.aoi import (
            generate_aoi,
            generate_background_aoi,
            generate_buffer_aoi,
        )
        from ..resources.assets import (
            process_user_assets,
            process_vegetation_exclusion_zones,
        )
        from ..resources.dem import process_buffer_dem, process_target_dem
        from ..resources.hamster import process_hamster_data
        from ..resources.landcover import (
            process_background_landcover,
            process_buffer_landcover,
            process_target_landcover,
        )
        from ..resources.mesh import generate_buffer_mesh, generate_target_mesh
        from ..resources.scene import create_scene_description
        from ..resources.texture import (
            generate_background_texture,
            generate_buffer_texture,
            generate_target_texture,
        )
        from ..resources.vegetation import process_target_vegetation

        # Register resources with their dependencies

        # Core resources - always included
        self.registry.register("aoi", [], generate_aoi)
        self.registry.register("target_dem", ["aoi"], process_target_dem)
        self.registry.register("target_landcover", ["aoi"], process_target_landcover)
        self.registry.register("target_mesh", ["target_dem"], generate_target_mesh)
        self.registry.register(
            "target_texture", ["target_landcover"], generate_target_texture
        )

        if self.config.buffer is not None:
            self.registry.register("buffer_aoi", ["aoi"], generate_buffer_aoi)
            self.registry.register("buffer_dem", ["buffer_aoi"], process_buffer_dem)
            self.registry.register(
                "buffer_landcover", ["buffer_aoi"], process_buffer_landcover
            )
            self.registry.register("buffer_mesh", ["buffer_dem"], generate_buffer_mesh)
            self.registry.register(
                "buffer_texture", ["buffer_landcover"], generate_buffer_texture
            )

        if self.config.background is not None:
            self.registry.register("background_aoi", ["aoi"], generate_background_aoi)
            self.registry.register(
                "background_landcover", ["background_aoi"], process_background_landcover
            )
            self.registry.register(
                "background_texture",
                ["background_landcover"],
                generate_background_texture,
            )

        # Optional resources
        if self.config.user_assets or self.xml_assets:
            self.registry.register("user_assets", ["target_dem"], process_user_assets)

        if self.config.vegetation_exclusion_zones:
            self.registry.register(
                "vegetation_exclusion_zones", [], process_vegetation_exclusion_zones
            )

        if self.config.hamster and self.config.hamster.enabled:
            # HAMSTER adapts to what was registered
            hamster_deps = ["aoi"]
            if "buffer_aoi" in self.registry.resources:
                hamster_deps.append("buffer_aoi")
            if "background_aoi" in self.registry.resources:
                hamster_deps.append("background_aoi")
            self.registry.register("hamster_data", hamster_deps, process_hamster_data)

        if self.config.trees_enabled:
            veg_deps = ["target_landcover", "target_dem"]

            if self.config.vegetation_exclusion_zones:
                veg_deps.append("vegetation_exclusion_zones")

            self.registry.register(
                "target_vegetation",
                veg_deps,
                process_target_vegetation,
            )

        # Scene description (dependencies will be updated by update_scene_dependencies)
        self.registry.register(
            "scene_description",
            ["target_mesh", "target_texture"],
            create_scene_description,
        )

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
                base_coordinate=xml_scene_config.base_coordinate,
                coord_type=xml_scene_config.coord_type,
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

    def run(self) -> SceneDescription:
        """Execute the complete scene generation pipeline.

        Returns:
            SceneDescription instance with complete scene configuration
        """
        self.initialize()

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
            _ = self.executor.execute(ctx)
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
        """Render the pipeline dependency graph to an image file using Graphviz.

        Nodes are colour-coded by resource category (AOI, DEM, landcover,
        mesh, texture, etc.) and shaped by type (ellipse for AOIs, diamond for
        meshes, double-octagon for the final scene description).  The output
        file is written next to the scene output directory when ``output_path``
        is not supplied.

        Args:
            output_path: Destination path for the rendered image (without
                extension). Defaults to
                ``<scene_output_dir>/<scene_name>_dag``.
            format: Graphviz output format (e.g. ``"png"``, ``"svg"``).

        Returns:
            Path to the rendered file, or ``None`` if Graphviz is not
            installed or rendering fails.
        """
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
