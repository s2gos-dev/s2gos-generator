import pytest

from s2gos_generator.core.resource_registry import (
    DAGExecutor,
    ResourceRegistry,
)


class _StubContext:
    """Minimal context stub for DAGExecutor tests."""

    def __init__(self):
        self.dependency_outputs = {}


_NOOP = lambda ctx: None  # noqa: E731


def _make_core_registry() -> ResourceRegistry:
    """Build a registry with the standard core pipeline resources."""
    r = ResourceRegistry()
    r.register("aoi", [], _NOOP)
    r.register("target_dem", ["aoi"], _NOOP)
    r.register("target_landcover", ["aoi"], _NOOP)
    r.register("target_mesh", ["target_dem"], _NOOP)
    r.register("target_texture", ["target_landcover"], _NOOP)
    r.register("scene_description", ["target_mesh", "target_texture"], _NOOP)
    return r


class TestResourceRegistration:
    def test_register_and_retrieve(self):
        r = ResourceRegistry()
        func = lambda ctx: "ok"  # noqa: E731
        r.register("my_resource", ["dep_a"], func)
        res = r.get_resource("my_resource")
        assert res.id == "my_resource"
        assert res.dependencies == ["dep_a"]
        assert res.func is func

    def test_register_overwrites_duplicate_id(self):
        r = ResourceRegistry()
        r.register("x", [], lambda ctx: "first")
        r.register("x", [], lambda ctx: "second")
        res = r.get_resource("x")
        assert res.func(None) == "second"

    def test_get_unknown_id_raises(self):
        r = ResourceRegistry()
        with pytest.raises(ValueError, match="not found"):
            r.get_resource("unknown")

    def test_get_resource_list(self):
        r = ResourceRegistry()
        r.register("a", [], _NOOP)
        r.register("b", ["a"], _NOOP)
        resource_list = r.get_resource_list()
        assert len(resource_list) == 2
        ids = {res.id for res in resource_list}
        assert ids == {"a", "b"}


class TestExecutionOrder:
    def test_single_resource(self):
        r = ResourceRegistry()
        r.register("solo", [], _NOOP)
        assert r.get_execution_order() == ["solo"]

    def test_linear_chain(self):
        r = ResourceRegistry()
        r.register("a", [], _NOOP)
        r.register("b", ["a"], _NOOP)
        r.register("c", ["b"], _NOOP)
        order = r.get_execution_order()
        assert order == ["a", "b", "c"]

    def test_parallel_roots(self):
        r = ResourceRegistry()
        r.register("a", [], _NOOP)
        r.register("b", [], _NOOP)
        r.register("c", ["a", "b"], _NOOP)
        order = r.get_execution_order()
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_diamond_dependency(self):
        r = ResourceRegistry()
        r.register("a", [], _NOOP)
        r.register("b", ["a"], _NOOP)
        r.register("c", ["a"], _NOOP)
        r.register("d", ["b", "c"], _NOOP)
        order = r.get_execution_order()
        assert order.count("d") == 1
        assert order[-1] == "d"
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")

    def test_complex_pipeline_shape(self):
        r = _make_core_registry()
        order = r.get_execution_order()
        assert order[0] == "aoi"
        assert order[-1] == "scene_description"


class TestCycleDetection:
    def test_self_cycle_raises(self):
        r = ResourceRegistry()
        r.register("a", ["a"], _NOOP)
        with pytest.raises(ValueError, match="Circular"):
            r.get_execution_order()

    def test_two_node_cycle_raises(self):
        r = ResourceRegistry()
        r.register("a", ["b"], _NOOP)
        r.register("b", ["a"], _NOOP)
        with pytest.raises(ValueError, match="Circular"):
            r.get_execution_order()

    def test_three_node_cycle_raises(self):
        r = ResourceRegistry()
        r.register("a", ["c"], _NOOP)
        r.register("b", ["a"], _NOOP)
        r.register("c", ["b"], _NOOP)
        with pytest.raises(ValueError, match="Circular"):
            r.get_execution_order()

    def test_indirect_cycle_raises(self):
        # Cycle in the middle: B→C→D→B while A is a root
        r = ResourceRegistry()
        r.register("a", [], _NOOP)
        r.register("b", ["a", "d"], _NOOP)  # B depends on A and D
        r.register("c", ["b"], _NOOP)  # C depends on B
        r.register("d", ["c"], _NOOP)  # D depends on C → creates B→C→D→B
        with pytest.raises(ValueError, match="Circular"):
            r.get_execution_order()


class TestMissingDependencies:
    def test_missing_dependency_raises(self):
        r = ResourceRegistry()
        r.register("a", ["x"], _NOOP)
        with pytest.raises(ValueError, match="Missing"):
            r.get_execution_order()

    def test_missing_upstream_dep_raises(self):
        r = ResourceRegistry()
        r.register("a", ["x"], _NOOP)
        r.register("b", ["a"], _NOOP)
        with pytest.raises(ValueError, match="Missing"):
            r.get_execution_order()


class TestUpdateSceneDependencies:
    def test_base_deps_always_present(self):
        r = _make_core_registry()
        r.update_scene_dependencies()
        deps = r.get_resource("scene_description").dependencies
        assert "target_mesh" in deps
        assert "target_texture" in deps

    def test_buffer_mesh_and_texture_added(self):
        r = _make_core_registry()
        r.register("buffer_mesh", [], _NOOP)
        r.register("buffer_texture", [], _NOOP)
        r.update_scene_dependencies()
        deps = r.get_resource("scene_description").dependencies
        assert "buffer_mesh" in deps
        assert "buffer_texture" in deps

    def test_background_texture_added(self):
        r = _make_core_registry()
        r.register("background_texture", [], _NOOP)
        r.update_scene_dependencies()
        deps = r.get_resource("scene_description").dependencies
        assert "background_texture" in deps

    def test_optional_resources_added(self):
        r = _make_core_registry()
        r.register("user_assets", [], _NOOP)
        r.register("hamster_data", [], _NOOP)
        r.update_scene_dependencies()
        deps = r.get_resource("scene_description").dependencies
        assert "user_assets" in deps
        assert "hamster_data" in deps


class TestDAGExecutor:
    def test_execute_returns_all_results(self):
        r = ResourceRegistry()
        r.register("a", [], lambda ctx: "result_a")
        r.register("b", ["a"], lambda ctx: "result_b")
        r.register("c", ["b"], lambda ctx: "result_c")
        executor = DAGExecutor(r)
        results = executor.execute(_StubContext())
        assert results == {"a": "result_a", "b": "result_b", "c": "result_c"}

    def test_execution_order_respected(self):
        order_log = []
        r = ResourceRegistry()
        r.register("a", [], lambda ctx: order_log.append("a"))
        r.register("b", ["a"], lambda ctx: order_log.append("b"))
        r.register("c", ["b"], lambda ctx: order_log.append("c"))
        DAGExecutor(r).execute(_StubContext())
        assert order_log == ["a", "b", "c"]

    def test_dependency_outputs_available(self):
        received = {}

        def second_func(ctx):
            received["got"] = ctx.dependency_outputs.get("first")
            return "second_result"

        r = ResourceRegistry()
        r.register("first", [], lambda ctx: "first_result")
        r.register("second", ["first"], second_func)
        DAGExecutor(r).execute(_StubContext())
        assert received["got"] == "first_result"

    def test_resource_failure_raises_runtime_error(self):
        def failing_func(ctx):
            raise Exception("kaboom")

        r = ResourceRegistry()
        r.register("bad_resource", [], failing_func)
        executor = DAGExecutor(r)
        with pytest.raises(RuntimeError, match="bad_resource"):
            executor.execute(_StubContext())
