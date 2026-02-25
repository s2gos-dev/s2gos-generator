from unittest.mock import MagicMock

from s2gos_generator.resources.aoi import (
    generate_aoi,
    generate_background_aoi,
    generate_buffer_aoi,
)


def _make_ctx(
    center_lat: float = 0.0,
    center_lon: float = 0.0,
    aoi_size_km: float = 10.0,
    buffer_size_km: float = 20.0,
    background_size_km: float = 50.0,
):
    """Build a minimal mock context for AOI resource tests."""
    ctx = MagicMock()
    ctx.center_lat = center_lat
    ctx.center_lon = center_lon
    ctx.aoi_size_km = aoi_size_km
    # buffer and background sizes live under ctx.config sub-models
    ctx.config.buffer.size_km = buffer_size_km
    ctx.config.background.size_km = background_size_km
    ctx._target_aoi_polygon = None
    ctx._buffer_aoi_polygon = None
    ctx._background_aoi_polygon = None
    return ctx


class TestGenerateAOI:
    def test_generate_aoi_sets_polygon(self):
        ctx = _make_ctx()
        generate_aoi(ctx)
        assert ctx._target_aoi_polygon is not None

    def test_aoi_polygon_is_valid(self):
        ctx = _make_ctx()
        generate_aoi(ctx)
        polygon = ctx._target_aoi_polygon
        assert polygon.is_valid

    def test_generate_aoi_returns_none(self):
        ctx = _make_ctx()
        result = generate_aoi(ctx)
        assert result is None


class TestGenerateBufferAOI:
    def test_buffer_aoi_sets_context(self):
        ctx = _make_ctx()
        generate_aoi(ctx)
        generate_buffer_aoi(ctx)
        assert ctx._buffer_aoi_polygon is not None

    def test_buffer_aoi_is_valid(self):
        ctx = _make_ctx()
        generate_buffer_aoi(ctx)
        assert ctx._buffer_aoi_polygon.is_valid

    def test_buffer_aoi_larger_than_target(self):
        ctx = _make_ctx(aoi_size_km=10.0, buffer_size_km=20.0)
        generate_aoi(ctx)
        generate_buffer_aoi(ctx)
        assert ctx._buffer_aoi_polygon.area > ctx._target_aoi_polygon.area

    def test_buffer_aoi_returns_none(self):
        ctx = _make_ctx()
        result = generate_buffer_aoi(ctx)
        assert result is None


class TestGenerateBackgroundAOI:
    def test_background_aoi_larger_than_buffer(self):
        ctx = _make_ctx(aoi_size_km=10.0, buffer_size_km=20.0, background_size_km=50.0)
        generate_buffer_aoi(ctx)
        generate_background_aoi(ctx)
        assert ctx._background_aoi_polygon.area > ctx._buffer_aoi_polygon.area

    def test_background_aoi_sets_context(self):
        ctx = _make_ctx()
        generate_background_aoi(ctx)
        assert ctx._background_aoi_polygon is not None

    def test_background_aoi_is_valid(self):
        ctx = _make_ctx()
        generate_background_aoi(ctx)
        assert ctx._background_aoi_polygon.is_valid

    def test_background_aoi_returns_none(self):
        ctx = _make_ctx()
        result = generate_background_aoi(ctx)
        assert result is None
