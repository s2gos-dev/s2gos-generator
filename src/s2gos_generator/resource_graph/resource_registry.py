from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from .singleton import Singleton


class ResourceContext(BaseModel):
    kwargs: Dict = {}
    dependency_outputs: Dict[str, Path | None] = {}
    
    model_config = {"extra": "allow"}


class Resource(BaseModel):
    id: str
    dependencies: List[str] | None
    func: Callable

    def __call__(self, ctx: ResourceContext) -> Optional[Path]:
        return self.func(ctx)


class ResourceRegistry(metaclass=Singleton):
    resources: Dict[str, Dict[str, Any]] = None

    def register_resource(self, id, dependencies, func: Callable):
        if self.resources is None:
            self.resources = {}

        resource = Resource(id=id, dependencies=dependencies, func=func)
        self.resources[id] = resource

    def get_resource_list(self):
        return list(self.resources.values())

    def get_resource(self, id):
        return self.resources[id]

    def resource(self, id: str, dependencies: List[str] | None = None):
        def resource_decorator(cls):
            cls.id = id
            self.register_resource(id, dependencies, cls)
            return cls

        return resource_decorator

    def remove_resource(self, resource_id: str):
        """Remove a resource from the registry."""
        if self.resources and resource_id in self.resources:
            del self.resources[resource_id]

    def remove_resources(self, resource_ids: List[str]):
        """Remove multiple resources from the registry."""
        for resource_id in resource_ids:
            self.remove_resource(resource_id)

    def filter_by_config(self, config):
        """Remove disabled resources from registry based on configuration."""
        if self.resources is None:
            return

        resources_to_remove = []

        # Buffer resources
        if not getattr(config, 'has_buffer', False):
            resources_to_remove.extend([
                'buffer_aoi', 'buffer_dem', 'buffer_landcover', 
                'buffer_mesh', 'buffer_texture'
            ])

        # Background resources (depend on buffer being enabled)
        if not getattr(config, 'has_buffer', False) or not hasattr(config, 'buffer') or \
           not hasattr(config.buffer, 'background_size_km'):
            resources_to_remove.extend([
                'background_aoi', 'background_landcover', 'background_texture'
            ])

        # User assets
        if not getattr(config, 'user_assets', None):
            resources_to_remove.append('user_assets')

        # HAMSTER data  
        if not getattr(config, 'hamster', None) or not getattr(config.hamster, 'enabled', False):
            resources_to_remove.append('hamster_data')

        # Remove disabled resources
        self.remove_resources(resources_to_remove)

    def update_scene_dependencies(self):
        """Update scene_description dependencies based on currently registered resources."""
        if self.resources is None or 'scene_description' not in self.resources:
            return

        # Base dependencies that are always required
        base_dependencies = ['target_mesh', 'target_texture']
        
        # Optional dependencies that may be present
        optional_resource_ids = {
            'buffer_mesh', 'buffer_texture', 'background_texture', 
            'user_assets', 'hamster_data'
        }
        
        # Add optional dependencies that are actually registered
        optional_dependencies = [
            resource_id for resource_id in optional_resource_ids 
            if resource_id in self.resources
        ]
        
        # Update scene_description dependencies
        scene_resource = self.resources['scene_description']
        scene_resource.dependencies = base_dependencies + optional_dependencies


def resource(id: str, dependencies: List[str] | None = None):
    def resource_decorator(func):
        registry = ResourceRegistry()
        registry.register_resource(id=id, dependencies=dependencies, func=func)
        return func

    return resource_decorator