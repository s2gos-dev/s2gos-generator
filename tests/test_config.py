import json
from datetime import datetime
from typing import Any, Dict

import pytest

from s2gos_generator.core.config import (
    AbsorptionDatabase,
    AerosolDataset,
    AtmosphereConfig,
    ExponentialDistribution,
    GaussianDistribution,
    HeterogeneousAtmosphereConfig,
    HomogeneousAtmosphereConfig,
    MolecularAtmosphereConfig,
    ParticleLayerConfig,
    ProcessingOptions,
    SceneGenConfig,
    SceneLocation,
    ThermophysicalConfig,
    UniformDistribution,
    UserAssets,
    VegetationPlacementConfig,
    VegetationSpecies,
)


@pytest.fixture(autouse=True)
def mock_path_validation(monkeypatch):
    """Mock all file path validation to avoid needing real files."""
    monkeypatch.setattr("s2gos_utils.io.paths.exists", lambda p: True)
    monkeypatch.setattr("s2gos_utils.io.paths.mkdir", lambda p: None)

    def mock_upath_exists(self):
        return True

    monkeypatch.setattr("upath.core.UPath.exists", mock_upath_exists)

    def mock_resolve(filename, asset_type: str = "asset"):
        """Mock that properly returns PathRef objects."""
        from s2gos_utils.io import PathRef

        # Handle both string input and PathRef input
        if isinstance(filename, PathRef):
            return filename
        elif isinstance(filename, dict):
            # When deserializing from JSON, Pydantic passes the dict representation
            return PathRef(filename.get("value"), filename.get("cid"))
        else:
            # String input
            return PathRef(filename, None)

    monkeypatch.setattr("s2gos_generator.core.config._resolve_asset_path", mock_resolve)

    # Create a mock resolver object with a resolve method
    class MockResolver:
        def resolve(self, path, strict=True):
            """Mock resolver to always return a UPath that exists."""
            from s2gos_utils.io import PathRef
            from upath import UPath

            if isinstance(path, PathRef):
                return path.upath
            else:
                return UPath(path)

    # Replace the resolver instance in both indexed_geotiff and zarr modules
    mock_resolver = MockResolver()
    monkeypatch.setattr(
        "s2gos_generator.dataset.indexed_geotiff.resolver", mock_resolver
    )
    monkeypatch.setattr("s2gos_generator.dataset.zarr.resolver", mock_resolver)
    monkeypatch.setattr("s2gos_generator.core.config.resolver", mock_resolver)

    def mock_settings() -> Dict[str, Any]:
        """Mock settings to return Dataset objects instead of paths."""
        from s2gos_utils.io import PathRef
        from upath import UPath

        from s2gos_generator.dataset import IndexedGeoTiff

        # Create mock datasets
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

    # Mock the _load_index_gdf to avoid actual file loading
    def mock_load_index_gdf(index_path):
        """Mock index loading to return empty GeoDataFrame."""
        import geopandas as gpd

        return gpd.GeoDataFrame({"path": [], "geometry": []}, crs="EPSG:4326")

    monkeypatch.setattr(
        "s2gos_generator.dataset.indexed_geotiff._load_index_gdf",
        mock_load_index_gdf,
    )
    monkeypatch.setattr(
        "s2gos_generator.core.config._load_settings_data_sources_config",
        mock_settings,
    )


@pytest.fixture
def sample_scene_location():
    return SceneLocation(center_lat=45.0, center_lon=15.0, aoi_size_km=10.0)


@pytest.fixture
def sample_processing_options():
    return ProcessingOptions(
        target_resolution_m=30.0,
        generate_texture_preview=True,
        handle_dem_nans=True,
    )


@pytest.fixture
def sample_thermophysical_config():
    return ThermophysicalConfig(
        identifier="afgl_1986-us_standard",
        altitude_min=0.0,
        altitude_max=120000.0,
        altitude_step=1000.0,
    )


@pytest.fixture
def sample_molecular_atmosphere():
    return MolecularAtmosphereConfig(
        thermoprops=ThermophysicalConfig(),
        absorption_database=AbsorptionDatabase.GECKO,
        has_absorption=True,
    )


@pytest.fixture
def sample_homogeneous_atmosphere():
    return HomogeneousAtmosphereConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.1,
        scale_height=1000.0,
    )


@pytest.fixture
def sample_heterogeneous_atmosphere():
    return HeterogeneousAtmosphereConfig(
        molecular=MolecularAtmosphereConfig(), particle_layers=None
    )


@pytest.fixture
def sample_atmosphere(sample_molecular_atmosphere):
    return AtmosphereConfig(boa=0.0, toa=40000.0, details=sample_molecular_atmosphere)


@pytest.fixture
def sample_exponential_distribution():
    return ExponentialDistribution(rate=0.001)


@pytest.fixture
def sample_gaussian_distribution():
    return GaussianDistribution(center_altitude=5000.0, width=1000.0)


@pytest.fixture
def sample_uniform_distribution():
    return UniformDistribution()


@pytest.fixture
def sample_particle_layer():
    return ParticleLayerConfig(
        aerosol_dataset=AerosolDataset.SIXSV_CONTINENTAL,
        optical_thickness=0.2,
        altitude_bottom=0.0,
        altitude_top=10000.0,
        distribution=ExponentialDistribution(rate=0.001),
        reference_wavelength=550.0,
    )


@pytest.fixture
def sample_user_asset():
    return UserAssets(
        object_id="test_object",
        ply_path="test.ply",
        coordinate=[15.0, 45.0],
        material="concrete",
        elevation_offset=0.0,
        scale=1.0,
    )


@pytest.fixture
def sample_vegetation_placement():
    return VegetationPlacementConfig(
        enabled=True,
        landcover_species_mapping={
            10: [
                VegetationSpecies(
                    name="oak_trees",
                    asset_xml_paths=["tree.xml"],
                    density_per_hectare=400.0,
                    scale_min=10.0,
                    scale_max=35.0,
                )
            ]
        },
        min_spacing=2.0,
    )


@pytest.mark.parametrize(
    "model_class,fixture_name,type_value",
    [
        (SceneLocation, "sample_scene_location", None),
        (ProcessingOptions, "sample_processing_options", None),
        (ThermophysicalConfig, "sample_thermophysical_config", None),
        (MolecularAtmosphereConfig, "sample_molecular_atmosphere", "molecular"),
        (HomogeneousAtmosphereConfig, "sample_homogeneous_atmosphere", "homogeneous"),
        (
            HeterogeneousAtmosphereConfig,
            "sample_heterogeneous_atmosphere",
            "heterogeneous",
        ),
        (AtmosphereConfig, "sample_atmosphere", None),
        (ExponentialDistribution, "sample_exponential_distribution", "exponential"),
        (GaussianDistribution, "sample_gaussian_distribution", "gaussian"),
        (UniformDistribution, "sample_uniform_distribution", "uniform"),
        (ParticleLayerConfig, "sample_particle_layer", None),
        (UserAssets, "sample_user_asset", None),
        (VegetationPlacementConfig, "sample_vegetation_placement", None),
    ],
)
def test_model_serialization(model_class, fixture_name, type_value, request):
    config = request.getfixturevalue(fixture_name)
    json_str = config.model_dump_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    data = json.loads(json_str)
    if type_value:
        assert data["type"] == type_value

    reconstructed = model_class(**data)
    assert reconstructed == config

    schema = config.model_json_schema()
    assert "properties" in schema


def test_minimal_scene_config_serialization(tmp_path):
    from s2gos_utils.io import PathRef

    from s2gos_generator.dataset import IndexedGeoTiff

    (tmp_path / "dem_index.feather").touch()
    (tmp_path / "dem").mkdir()
    (tmp_path / "landcover_index.feather").touch()
    (tmp_path / "landcover").mkdir()
    (tmp_path / "materials.json").touch()
    (tmp_path / "output").mkdir()

    # Create Dataset objects
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

    config = SceneGenConfig(
        scene_name="test_scene",
        location=SceneLocation(center_lat=45.0, center_lon=15.0, aoi_size_km=10.0),
        data_sources={
            "dem": dem_dataset,
            "landcover": landcover_dataset,
            "material_config_path": PathRef(tmp_path / "materials.json", None),
        },
        output_dir=PathRef(tmp_path / "output", None),
    )

    json_str = config.model_dump_json()
    assert isinstance(json_str, str)

    data = json.loads(json_str)
    assert data["scene_name"] == "test_scene"
    assert "created_at" in data
    assert isinstance(data["created_at"], str)
    datetime.fromisoformat(data["created_at"])

    schema = config.model_json_schema()
    assert "properties" in schema
    assert "scene_name" in schema["properties"]
    assert "location" in schema["properties"]


def test_scene_config_round_trip(tmp_path):
    from s2gos_utils.io import PathRef

    from s2gos_generator.dataset import IndexedGeoTiff

    (tmp_path / "dem_index.feather").touch()
    (tmp_path / "dem").mkdir()
    (tmp_path / "landcover_index.feather").touch()
    (tmp_path / "landcover").mkdir()
    (tmp_path / "materials.json").touch()
    (tmp_path / "output").mkdir()

    # Create Dataset objects
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

    original = SceneGenConfig(
        scene_name="test_scene",
        location=SceneLocation(center_lat=45.0, center_lon=15.0, aoi_size_km=10.0),
        data_sources={
            "dem": dem_dataset,
            "landcover": landcover_dataset,
            "material_config_path": PathRef(tmp_path / "materials.json", None),
        },
        output_dir=PathRef(tmp_path / "output", None),
    )

    json_str = original.model_dump_json()
    data = json.loads(json_str)
    reconstructed = SceneGenConfig(**data)

    assert reconstructed.scene_name == original.scene_name
    assert reconstructed.location.center_lat == original.location.center_lat
