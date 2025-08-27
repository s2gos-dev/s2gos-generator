"""Pipeline implementation that preserves SceneResourceContext."""

from ..resource_graph.local_pipeline import LocalPipeline
from ..resource_graph.resource_registry import ResourceRegistry
from .context import SceneResourceContext


class SceneLocalPipeline(LocalPipeline):
    """Pipeline that preserves SceneResourceContext during execution."""

    def run(self, ctx: SceneResourceContext):
        """Run the pipeline while preserving the SceneResourceContext."""
        registry = ResourceRegistry()
        
        queue = self._schedule_resources()
        resource_outputs = {}

        for resource_id in queue:
            print(f"Process resource {resource_id}")
            resource = registry.get_resource(resource_id)

            dependency_outputs = {}
            if resource.dependencies:
                for dep_id in resource.dependencies:
                    dependency_outputs[dep_id] = resource_outputs.get(dep_id)
            
            ctx.dependency_outputs = dependency_outputs
            result = resource(ctx)
            resource_outputs[resource_id] = result

            print(f"  -> {resource_id} output: {result}")

        print("complete")
        return resource_outputs