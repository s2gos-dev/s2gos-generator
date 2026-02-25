"""Shared pytest fixtures for s2gos-generator tests."""

from typing import Any, Dict

import pytest


@pytest.fixture
def mock_path_validation(monkeypatch):
    """Mock all file path validation to avoid needing real files.

    Not autouse — apply per-file via pytestmark or explicit fixture dependency.
    """
    monkeypatch.setattr("s2gos_utils.io.paths.exists", lambda p: True)
    monkeypatch.setattr("s2gos_utils.io.paths.mkdir", lambda p: None)

    def mock_upath_exists(self):
        return True

    monkeypatch.setattr("upath.core.UPath.exists", mock_upath_exists)

    def mock_resolve(filename, asset_type: str = "asset"):
        """Mock that properly returns PathRef objects."""
        from s2gos_utils.io import PathRef

        if isinstance(filename, PathRef):
            return filename
        elif isinstance(filename, dict):
            return PathRef(filename.get("value"), filename.get("cid"))
        else:
            return PathRef(filename, None)

    monkeypatch.setattr(
        "s2gos_generator.core.config.atmosphere._resolve_asset_path", mock_resolve
    )
    monkeypatch.setattr(
        "s2gos_generator.core.config.vegetation._resolve_asset_path", mock_resolve
    )
    monkeypatch.setattr(
        "s2gos_generator.core.config.assets._resolve_asset_path", mock_resolve
    )

    class MockResolver:
        def resolve(self, path, strict=True):
            from s2gos_utils.io import PathRef
            from upath import UPath

            if isinstance(path, PathRef):
                return path.upath
            else:
                return UPath(path)

    mock_resolver = MockResolver()
    monkeypatch.setattr(
        "s2gos_generator.dataset.indexed_geotiff.resolver", mock_resolver
    )
    monkeypatch.setattr("s2gos_generator.dataset.zarr.resolver", mock_resolver)
    monkeypatch.setattr("s2gos_generator.core.config.scene.resolver", mock_resolver)

    def mock_settings() -> Dict[str, Any]:
        from s2gos_utils.io import PathRef

        from s2gos_generator.dataset import IndexedGeoTiff

        mock_dem = IndexedGeoTiff(
            name="DEM",
            index_path=PathRef("/mock/dem_index.feather", None),
            root_directory=PathRef("/mock/dem", None),
        )
        mock_landcover = IndexedGeoTiff(
            name="Landcover",
            index_path=PathRef("/mock/landcover_index.feather", None),
            root_directory=PathRef("/mock/landcover", None),
        )

        return {
            "dem": mock_dem,
            "landcover": mock_landcover,
            "material_config_path": PathRef("/mock/materials.json", None),
        }

    def mock_load_index_gdf(index_path):
        import geopandas as gpd

        return gpd.GeoDataFrame({"path": [], "geometry": []}, crs="EPSG:4326")

    monkeypatch.setattr(
        "s2gos_generator.dataset.indexed_geotiff._load_index_gdf",
        mock_load_index_gdf,
    )
    monkeypatch.setattr(
        "s2gos_generator.core.config.scene._load_settings_data_sources_config",
        mock_settings,
    )


@pytest.fixture
def make_minimal_config(tmp_path, mock_path_validation):
    """Factory fixture returning a minimal valid SceneGenConfig backed by temp paths.

    Depends on mock_path_validation so callers don't need to request it separately.
    Uses real temp-path-backed files for path-existence validation to pass reliably.
    """

    def _make(**kwargs):
        from s2gos_utils.io import PathRef

        from s2gos_generator.core.config import SceneGenConfig, SceneLocation
        from s2gos_generator.dataset import IndexedGeoTiff

        # Create real files so DataSources.validate_path_exists passes
        (tmp_path / "dem_index.feather").touch()
        (tmp_path / "dem").mkdir(exist_ok=True)
        (tmp_path / "landcover_index.feather").touch()
        (tmp_path / "landcover").mkdir(exist_ok=True)
        (tmp_path / "materials.json").touch()
        (tmp_path / "output").mkdir(exist_ok=True)

        dem_dataset = IndexedGeoTiff(
            name="DEM",
            index_path=PathRef(tmp_path / "dem_index.feather", None),
            root_directory=PathRef(tmp_path / "dem", None),
        )
        landcover_dataset = IndexedGeoTiff(
            name="Landcover",
            index_path=PathRef(tmp_path / "landcover_index.feather", None),
            root_directory=PathRef(tmp_path / "landcover", None),
        )

        return SceneGenConfig(
            scene_name="test_scene",
            location=SceneLocation(center_lat=45.0, center_lon=15.0, aoi_size_km=10.0),
            data_sources={
                "dem": dem_dataset,
                "landcover": landcover_dataset,
                "material_config_path": PathRef(tmp_path / "materials.json", None),
            },
            output_dir=PathRef(tmp_path / "output", None),
            **kwargs,
        )

    return _make
