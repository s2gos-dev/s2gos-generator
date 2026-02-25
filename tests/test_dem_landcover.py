from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_ctx(tmp_path):
    ctx = MagicMock()
    ctx.data_dir = tmp_path
    ctx.meshes_dir = tmp_path
    ctx.scene_name = "test_scene"
    ctx.target_resolution_m = 30.0
    ctx.aoi_size_km = 10.0
    ctx.center_lat = 45.0
    ctx.center_lon = 15.0

    ctx._target_aoi_polygon = object()
    ctx.dependency_outputs = {"target_dem": tmp_path / "dem.zarr"}

    # Mock the assets container so we can check side effects
    ctx.assets = MagicMock()
    return ctx


class TestProcessTargetDem:
    def test_validation_error(self, mock_ctx):
        from s2gos_generator.resources.dem import process_target_dem

        mock_ctx._target_aoi_polygon = None
        with pytest.raises(ValueError):
            process_target_dem(mock_ctx)

    def test_success_workflow(self, mock_ctx, monkeypatch):
        from s2gos_generator.resources.dem import process_target_dem

        mock_processor = MagicMock()
        monkeypatch.setattr(
            "s2gos_generator.resources.dem.DEMProcessor",
            MagicMock(return_value=mock_processor),
        )

        result = process_target_dem(mock_ctx)

        expected = mock_ctx.data_dir / "dem_test_scene_30.0m.zarr"
        assert result == expected
        assert mock_processor.generate_dem.call_args.kwargs["output_path"] == expected
        assert mock_ctx.assets.dem_file == expected


class TestProcessTargetLandcover:
    def test_validation_error(self, mock_ctx):
        from s2gos_generator.resources.landcover import process_target_landcover

        mock_ctx._target_aoi_polygon = None
        with pytest.raises(ValueError):
            process_target_landcover(mock_ctx)

    def test_success_workflow(self, mock_ctx, monkeypatch):
        from s2gos_generator.resources.landcover import process_target_landcover

        mock_processor = MagicMock()
        monkeypatch.setattr(
            "s2gos_generator.resources.landcover.LandCoverProcessor",
            MagicMock(return_value=mock_processor),
        )

        result = process_target_landcover(mock_ctx)
        expected = mock_ctx.data_dir / "landcover_test_scene_30.0m.zarr"

        assert result == expected
        assert (
            mock_processor.generate_landcover.call_args.kwargs["output_path"]
            == expected
        )


class TestGenerateTargetMesh:
    def test_validation_error(self, mock_ctx):
        from s2gos_generator.resources.mesh import generate_target_mesh

        mock_ctx.dependency_outputs = {"target_dem": None}
        with pytest.raises(ValueError):
            generate_target_mesh(mock_ctx)

    def test_success_workflow(self, mock_ctx, monkeypatch):
        from s2gos_generator.resources.mesh import generate_target_mesh

        mock_generator = MagicMock()
        monkeypatch.setattr(
            "s2gos_generator.resources.mesh.MeshGenerator",
            MagicMock(return_value=mock_generator),
        )

        result = generate_target_mesh(mock_ctx)
        expected = mock_ctx.meshes_dir / "test_scene_terrain.ply"

        assert result == expected
        assert (
            mock_generator.generate_mesh_from_dem_file.call_args.kwargs["output_path"]
            == expected
        )
        assert mock_ctx.assets.mesh_file == expected
