"""Scene generation pipeline with automatic dependency management."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from s2gos_utils.scene import SceneDescription
from s2gos_utils.io.paths import mkdir

from .local_pipeline import SceneLocalPipeline
from ..resource_graph.resource_registry import ResourceRegistry

from .context import SceneResourceContext
from .config import SceneGenConfig

# Import all resources to register them
from .. import resources


class SceneGenerationPipeline:
    """Scene generation pipeline with automatic dependency resolution.
    
    This pipeline automatically manages dependencies between scene generation
    steps, ensuring resources are processed in the correct order. The
    architecture is inspired by modern terrain generation systems like
    Microsoft Flight Simulator.
    """
    
    def __init__(self, config: SceneGenConfig, additional_material_libraries: Optional[List[Dict]] = None):
        """Initialize the scene generation pipeline.
        
        Args:
            config: Scene generation configuration
            additional_material_libraries: Optional list of material libraries to merge
        """
        self.config = config
        self.additional_material_libraries = additional_material_libraries or []
        
        from .. import resources
        
        # Clear any test resources that may have been registered
        registry = ResourceRegistry()
        if registry.resources:
            test_resource_ids = {'A', 'B', 'C', 'D', 'root'}
            registry.resources = {
                k: v for k, v in registry.resources.items() 
                if k not in test_resource_ids
            }
        
        # Filter resources based on configuration
        registry.filter_by_config(config)
        registry.update_scene_dependencies()
        
        self._setup_output_directories()
        self._log_registered_resources()
        
        logging.info(f"Pipeline initialized for scene '{config.scene_name}'")
    
    def _setup_output_directories(self) -> None:
        """Create the output directory structure."""
        directories = [
            self.config.scene_output_dir,
            self.config.data_dir,
            self.config.meshes_dir,
            self.config.textures_dir,
        ]
        
        for directory in directories:
            mkdir(directory)
    
    def _log_registered_resources(self) -> None:
        """Log information about the currently registered resources.
        
        This provides visibility into which resources are actually
        included in the DAG based on the configuration.
        """
        registry = ResourceRegistry()
        
        if not registry.resources:
            logging.warning("No resources registered!")
            return
            
        # Categorize resources that are currently registered
        core_resources = []
        buffer_resources = []
        background_resources = []
        optional_resources = []
        
        for resource_id in registry.resources.keys():
            if resource_id in ['aoi', 'target_dem', 'target_landcover', 'target_mesh', 'target_texture', 'scene_description']:
                core_resources.append(resource_id)
            elif resource_id.startswith('buffer_'):
                buffer_resources.append(resource_id)
            elif resource_id.startswith('background_'):
                background_resources.append(resource_id)
            else:
                optional_resources.append(resource_id)
        
        logging.info(f"Core resources: {', '.join(core_resources)}")
        if buffer_resources:
            logging.info(f"Buffer resources: {', '.join(buffer_resources)}")
        if background_resources:
            logging.info(f"Background resources: {', '.join(background_resources)}")
        if optional_resources:
            logging.info(f"Optional resources: {', '.join(optional_resources)}")
            
        total_resources = len(registry.get_resource_list())
        logging.info(f"Total registered resources: {total_resources}")
    
    def run_full_pipeline(self) -> SceneDescription:
        """Execute the complete scene generation pipeline.
        
        Returns:
            SceneDescription instance with complete scene configuration
        """
        return self.run()
    
    def run(self) -> SceneDescription:
        """Execute the complete scene generation pipeline.
        
        Returns:
            SceneDescription instance with complete scene configuration
        """
        logging.info(f"Starting pipeline for scene '{self.config.scene_name}'")
        
        try:
            ctx = SceneResourceContext(
                config=self.config,
                additional_material_libraries=self.additional_material_libraries
            )
            
            pipeline = SceneLocalPipeline()
            results = pipeline.run(ctx)
            
            scene_description = getattr(ctx, 'scene_description', None)
            if scene_description is None:
                raise RuntimeError("Scene description not found in pipeline results")
            
            logging.info("=== Pipeline Complete ===")
            logging.info(f"Scene description: {scene_description.name}")
            logging.info(f"Output directory: {self.config.scene_output_dir}")
            
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
        registry = ResourceRegistry()
        dependencies = {}
        
        # Check if resources are initialized
        if registry.resources is None:
            return dependencies
            
        # Filter out test resources
        test_resource_ids = {'A', 'B', 'C', 'D', 'root'}
        
        for resource in registry.get_resource_list():
            if resource.id not in test_resource_ids:
                dependencies[resource.id] = resource.dependencies or []
        
        return dependencies
    
    def visualize_dag(self, output_path: Optional[Path] = None, format: str = "png") -> Optional[Path]:
        """Visualize the resource dependency structure with scene-specific styling.
        
        Args:
            output_path: Optional path for saving the visualization
            format: Output format (png, svg, pdf, etc.)
            
        Returns:
            Path to the generated visualization file, if available
        """
        try:
            import graphviz
            from ..resource_graph.resource_registry import ResourceRegistry
            
            registry = ResourceRegistry()
            resources = registry.get_resource_list()
            
            if output_path is None:
                output_path = self.config.scene_output_dir / f"{self.scene_name}_dag"
            
            # Create a new directed graph with scene-specific styling
            dot = graphviz.Digraph(comment=f'Scene Generation Pipeline - {self.scene_name}')
            dot.attr(rankdir='TB')
            dot.attr('graph', bgcolor='white', fontname='Arial', fontsize='14')
            dot.attr('node', fontname='Arial', fontsize='12', style='filled')
            dot.attr('edge', fontname='Arial', fontsize='10')
            
            # Define resource type colors
            resource_colors = {
                'aoi': '#90EE90',         # Light green - foundational
                'buffer_aoi': '#98FB98',  # Pale green
                'background_aoi': '#F0FFF0',  # Honeydew
                'target_dem': '#87CEEB',      # Sky blue - DEM
                'buffer_dem': '#B0E0E6',     # Powder blue
                'target_landcover': '#DDA0DD',     # Plum - landcover
                'buffer_landcover': '#E6E6FA',     # Lavender
                'background_landcover': '#F8F8FF', # Ghost white
                'target_mesh': '#FFB6C1',     # Light pink - meshes
                'buffer_mesh': '#FFC0CB',     # Pink
                'target_texture': '#FFFFE0',      # Light yellow - textures
                'buffer_texture': '#FFFACD',      # Lemon chiffon
                'background_texture': '#FFFFF0',  # Ivory
                'user_assets': '#FFA07A',         # Light salmon - assets
                'hamster_data': '#20B2AA',        # Light sea green - HAMSTER
                'scene_description': '#FF6347'    # Tomato - final output
            }
            
            # Add all nodes with appropriate colors and shapes
            for resource in resources:
                color = resource_colors.get(resource.id, '#D3D3D3')  # Default light gray
                
                # Use different shapes for different resource types
                if 'aoi' in resource.id:
                    dot.node(resource.id, resource.id, fillcolor=color, shape='ellipse')
                elif resource.id == 'scene_description':
                    dot.node(resource.id, resource.id, fillcolor=color, shape='doubleoctagon')
                elif 'mesh' in resource.id:
                    dot.node(resource.id, resource.id, fillcolor=color, shape='diamond')
                else:
                    dot.node(resource.id, resource.id, fillcolor=color, shape='box')
            
            # Add edges for dependencies with different styles
            for resource in resources:
                if resource.dependencies:
                    for dependency in resource.dependencies:
                        # Different arrow styles for different dependency types
                        if 'buffer' in resource.id and 'buffer' in dependency:
                            dot.edge(dependency, resource.id, style='dashed', color='blue')
                        elif 'background' in resource.id:
                            dot.edge(dependency, resource.id, style='dotted', color='purple')  
                        else:
                            dot.edge(dependency, resource.id, color='black')
            
            # Add title and legend
            dot.attr(label=f'Scene Generation Pipeline\\n{self.scene_name}\\nGenerated: {self.config.created_at.strftime("%Y-%m-%d %H:%M")}')
            dot.attr(labelloc='t')
            
            # Render the graph
            output_file = dot.render(str(output_path), format=format, cleanup=True)
            logging.info(f"Pipeline visualization saved to: {output_file}")
            return Path(output_file)
            
        except ImportError:
            logging.warning("Pipeline visualization not available (graphviz not installed)")
            return None
        except Exception as e:
            logging.warning(f"Could not create pipeline visualization: {e}")
            return None
    
    def print_resource_summary(self) -> None:
        """Print a summary of the resource dependency graph."""
        registry = ResourceRegistry()
        resources = registry.get_resource_list()
        
        print(f"\\nScene Generation Pipeline Summary - {self.scene_name}")
        print("=" * 50)
        print(f"Total resources: {len(resources)}")
        
        # Group resources by type
        resource_groups = {
            'AOI': [],
            'DEM': [],
            'LandCover': [],
            'Mesh': [],
            'Texture': [],
            'Optional': [],
            'Final': []
        }
        
        for resource in resources:
            if 'aoi' in resource.id:
                resource_groups['AOI'].append(resource.id)
            elif 'dem' in resource.id:
                resource_groups['DEM'].append(resource.id)
            elif 'landcover' in resource.id:
                resource_groups['LandCover'].append(resource.id)
            elif 'mesh' in resource.id:
                resource_groups['Mesh'].append(resource.id)
            elif 'texture' in resource.id:
                resource_groups['Texture'].append(resource.id)
            elif resource.id == 'scene_description':
                resource_groups['Final'].append(resource.id)
            else:
                resource_groups['Optional'].append(resource.id)
        
        for group_name, resource_ids in resource_groups.items():
            if resource_ids:
                print(f"\\n{group_name} Resources:")
                for resource_id in resource_ids:
                    deps = next((r.dependencies for r in resources if r.id == resource_id), [])
                    dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                    print(f"  - {resource_id}{dep_str}")
        print()