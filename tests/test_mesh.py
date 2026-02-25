import numpy as np
import pytest
import trimesh
import xarray as xr

from s2gos_generator.assets.mesh import MeshGenerator


@pytest.fixture
def generator():
    return MeshGenerator()


@pytest.fixture
def clean_3x3_dem():
    elevation = np.ones((3, 3), dtype=float)
    return xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
    )


@pytest.fixture
def unit_square_mesh():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class TestCreateGridFaces:
    def test_output_shape_formula(self, generator):
        nx, ny = 3, 4
        faces = generator._create_grid_faces(nx, ny)
        expected_count = 2 * (nx - 1) * (ny - 1)
        assert faces.shape == (expected_count, 3)

    def test_all_indices_in_range(self, generator):
        nx, ny = 4, 5
        faces = generator._create_grid_faces(nx, ny)
        assert faces.max() < nx * ny

    def test_minimal_2x2_exact_faces(self, generator):
        faces = generator._create_grid_faces(2, 2)
        expected = np.array([[0, 1, 3], [0, 3, 2]])
        np.testing.assert_array_equal(faces, expected)

    def test_no_degenerate_faces(self, generator):
        faces = generator._create_grid_faces(5, 5)
        for row in faces:
            assert len(set(row)) == 3


class TestDemToMesh:
    def test_clean_3x3_dem_vertex_and_face_count(self, generator, clean_3x3_dem):
        mesh = generator.dem_to_mesh(clean_3x3_dem)
        assert len(mesh.vertices) == 9
        assert len(mesh.faces) == 8

    def test_nan_removes_affected_faces_and_vertices(self, generator):
        elevation = np.ones((3, 3), dtype=float)
        elevation[0, 0] = np.nan
        da = xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"x": [0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0]},
        )
        mesh = generator.dem_to_mesh(da)
        assert len(mesh.faces) == 6
        assert len(mesh.vertices) == 8

    def test_raises_on_bad_dims(self, generator):
        elevation = np.ones((3, 3), dtype=float)
        da = xr.DataArray(elevation, dims=["row", "col"])
        with pytest.raises(ValueError):
            generator.dem_to_mesh(da)

    def test_lon_lat_dims_accepted(self, generator):
        elevation = np.ones((3, 3), dtype=float)
        da = xr.DataArray(
            elevation,
            dims=["lat", "lon"],
            coords={"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0]},
        )
        mesh = generator.dem_to_mesh(da)
        assert isinstance(mesh, trimesh.Trimesh)


class TestAddUvCoordinates:
    def test_uv_in_unit_range(self, generator, unit_square_mesh):
        result = generator.add_uv_coordinates(unit_square_mesh)
        assert np.all(result.visual.uv >= 0.0)
        assert np.all(result.visual.uv <= 1.0)

    def test_corner_vertices_map_to_extremes(self, generator, unit_square_mesh):
        result = generator.add_uv_coordinates(unit_square_mesh)
        uv = result.visual.uv
        np.testing.assert_allclose(uv[0], [0.0, 0.0])
        np.testing.assert_allclose(uv[2], [1.0, 1.0])

    def test_zero_extent_raises_value_error(self, generator):
        vertices = np.array(
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=float
        )  #  perfectly flat wall on the Y-Z plane
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        expected_error_msg = "Cannot calculate UV coordinates"

        with pytest.raises(ValueError, match=expected_error_msg):
            generator.add_uv_coordinates(mesh)
