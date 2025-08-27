import pprint
from typing import List, Optional

from pydantic import BaseModel

from .pipeline import Pipeline
from .resource_registry import ResourceContext, ResourceRegistry

registry = ResourceRegistry()


class LocalPipeline(Pipeline):
    class Node(BaseModel):
        id: str
        children: Optional[List[str]] = None
        parents: Optional[List[str]] = None

    def run(self, ctx):
        # 1. Build the dependency graph and generate a resource queue.
        queue = self._schedule_resources()

        # 2. Run the resource functions.
        resource_outputs = {}

        for resource_id in queue:
            print(f"Process resource {resource_id}")
            resource = registry.get_resource(resource_id)

            # Create context with dependency outputs
            dependency_outputs = {}
            if resource.dependencies:
                for dep_id in resource.dependencies:
                    dependency_outputs[dep_id] = resource_outputs.get(dep_id)

            ctx = ResourceContext(dependency_outputs=dependency_outputs)

            # Execute resource and store output
            result = resource(ctx)
            resource_outputs[resource_id] = result

            print(f"  -> {resource_id} output: {result}")

        print("complete")
        return resource_outputs

    def _schedule_resources(self):
        resource_list = registry.get_resource_list()

        node_dict = {}
        leaves = set()
        visited = set()
        queue = []

        # 1. First pass, build nodes and populate Nodes with id and children
        for resource in resource_list:
            id = resource.id
            children = resource.dependencies

            if id not in node_dict:
                node_dict[id] = self.Node(id=id, children=children)

        # 2. Second pass, build nodes and populate Nodes with parents
        # Find all leaves, Nodes with no children
        for resource in resource_list:
            id = resource.id
            children = resource.dependencies if resource.dependencies else []

            for child in children:
                if not node_dict[child].parents:
                    node_dict[child].parents = []

                node_dict[child].parents.append(id)

            if not node_dict[id].children:
                leaves.add(id)

        # 3. Populate the queue with resource ids in the right order
        while len(leaves) != 0:
            parents = set()
            # Add leaf nodes to the queue and find their parents
            for leaf in leaves:
                # Only queue the leaf node if was not added yet
                if leaf not in visited:
                    visited.add(leaf)
                    queue.append(leaf)

            # find the parents of each leaf and consider them to be the next leaves.
            for leaf in leaves:
                node = node_dict[leaf]
                for parent in node.parents if node.parents else []:
                    # Check that all the dependencies of a parent are queued
                    # before adding this one.
                    add_parent = True
                    pnode = node_dict[parent]
                    pchildren = pnode.children if pnode.children else []
                    for pchild in pchildren:
                        add_parent = add_parent & (pchild in visited)

                    if add_parent:
                        parents.add(parent)

            # move to the next leaf level
            leaves = parents

        pprint.pprint(node_dict)
        pprint.pprint(queue)

        return queue