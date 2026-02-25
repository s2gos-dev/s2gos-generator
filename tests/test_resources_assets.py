from unittest.mock import MagicMock

import pytest

from s2gos_generator.resources.assets import (
    _convert_to_scene_coords,
    _create_asset_exclusion_zone,
)


class TestConvertToSceneCoords:
    def test_scene_mode_passthrough(self):
        coordinate = (10.5, 20.3)
        result = _convert_to_scene_coords(coordinate, "scene", None)
        assert result == (10.5, 20.3)

    def test_geographic_mode_lat_lon_argument_order(self):
        # coordinate = [lon, lat] = [15.0, 45.0]
        # must call coords.latlon_to_scene(lat=45.0, lon=15.0), NOT (15.0, 45.0)
        coords = MagicMock()
        coords.latlon_to_scene.return_value = (100.0, 200.0)
        _convert_to_scene_coords([15.0, 45.0], "geographic", coords)
        coords.latlon_to_scene.assert_called_once_with(45.0, 15.0)


class TestCreateAssetExclusionZone:
    def test_float_creates_valid_polygon(self):
        from shapely.geometry import Polygon

        result = _create_asset_exclusion_zone(0.0, 0.0, 5.0)
        assert isinstance(result, Polygon)
        assert not result.is_empty

    def test_circle_contains_center(self):
        from shapely.geometry import Point

        result = _create_asset_exclusion_zone(10.0, 20.0, 5.0)
        assert result.contains(Point(10.0, 20.0))

    def test_circle_excludes_far_point(self):
        from shapely.geometry import Point

        result = _create_asset_exclusion_zone(0.0, 0.0, 5.0)
        assert not result.contains(Point(1000.0, 1000.0))

    def test_box_bounds_correct(self):
        result = _create_asset_exclusion_zone(0.0, 0.0, (10.0, 6.0))
        bounds = result.bounds  # (minx, miny, maxx, maxy)
        assert bounds == pytest.approx((-5.0, -3.0, 5.0, 3.0))

    def test_box_area_correct(self):
        width, height = 10.0, 6.0
        result = _create_asset_exclusion_zone(0.0, 0.0, (width, height))
        assert result.area == pytest.approx(width * height)
