import graphviz
from .resource_registry import ResourceRegistry
from typing import Dict, Set


def visualize_resource_graph(output_path: str = "resource_graph", format: str = "png") -> str:
    """
    Create a graphviz visualization of the resource dependency graph.
    
    Args:
        output_path: Output file path (without extension)
        format: Output format (png, svg, pdf, etc.)
    
    Returns:
        Path to the generated visualization file
    """
    registry = ResourceRegistry()
    resources = registry.get_resource_list()
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='Resource Dependency Graph')
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue')
    
    # Add all nodes first
    for resource in resources:
        dot.node(resource.id, resource.id)
    
    # Add edges for dependencies
    for resource in resources:
        if resource.dependencies:
            for dependency in resource.dependencies:
                # Draw arrow from dependency to resource (dependency -> resource)
                dot.edge(dependency, resource.id)
    
    # Render the graph
    output_file = dot.render(output_path, format=format, cleanup=True)
    print(f"Resource graph visualization saved to: {output_file}")
    
    return output_file


def print_resource_graph() -> None:
    """
    Print a text representation of the resource dependency graph.
    """
    registry = ResourceRegistry()
    resources = registry.get_resource_list()
    
    print("Resource Dependency Graph:")
    print("=" * 30)
    
    for resource in resources:
        deps = resource.dependencies if resource.dependencies else []
        deps_str = ", ".join(deps) if deps else "None"
        print(f"Resource: {resource.id}")
        print(f"  Dependencies: {deps_str}")
        print()


if __name__ == "__main__":
    # Generate visualization
    visualize_resource_graph()
    
    # Print text representation
    print_resource_graph()