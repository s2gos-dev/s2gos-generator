import json
from datetime import datetime

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

# Apply mock_path_validation to every test in this file
pytestmark = pytest.mark.usefixtures("mock_path_validation")


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
        coord_type="scene",
        material="concrete",
        elevation_offset=0.0,
        scale=1.0,
        blender_fix=False,
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
