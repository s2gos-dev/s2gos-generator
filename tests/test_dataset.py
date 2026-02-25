from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from s2gos_utils.io import PathRef
from shapely.geometry import Polygon, box
from upath import UPath

from s2gos_generator.dataset import (
    IndexedGeoTiff,
    Zarr,
    dataset_factory,
)


@pytest.fixture
def test_polygon_small():
    """Small polygon for testing overlaps."""
    return Polygon([(10.0, 45.0), (10.1, 45.0), (10.1, 45.1), (10.0, 45.1)])


@pytest.fixture
def test_polygon_large():
    """Larger polygon covering different area."""
    return Polygon([(20.0, 50.0), (20.5, 50.0), (20.5, 50.5), (20.0, 50.5)])


@pytest.fixture
def test_polygon_outside():
    """Polygon outside test area."""
    return Polygon([(-10.0, -10.0), (-9.0, -10.0), (-9.0, -9.0), (-10.0, -9.0)])


@pytest.fixture
def mock_index_gdf():
    """Create a mock index GeoDataFrame for IndexedGeoTiff testing."""
    geometries = [
        box(10.0, 45.0, 10.5, 45.5),
        box(10.5, 45.0, 11.0, 45.5),
        box(10.0, 44.5, 10.5, 45.0),
    ]
    paths = ["tile_1.tif", "tile_2.tif", "tile_3.tif"]
    gdf = gpd.GeoDataFrame({"path": paths, "geometry": geometries}, crs="EPSG:4326")
    return gdf


@pytest.fixture
def zarr_path(tmp_path):
    tmp_zarr = tmp_path / "data.zarr"
    tmp_zarr.mkdir()
    return tmp_zarr


@pytest.fixture
def mock_xarray_dataset():
    """Create a mock xarray dataset for testing."""
    x = np.linspace(10.0, 11.0, 100)
    y = np.linspace(45.0, 46.0, 100)
    data = np.random.rand(100, 100)

    ds = xr.Dataset(
        {"band1": (["y", "x"], data)},
        coords={"x": x, "y": y},
    )
    return ds


@pytest.fixture
def mock_zarr_dataset():
    """Create a mock Zarr dataset structure."""
    lon = np.linspace(10.0, 11.0, 50)
    lat = np.linspace(45.0, 46.0, 50)
    data = np.random.rand(50, 50)

    ds = xr.Dataset(
        {"temperature": (["lat", "lon"], data)},
        coords={"lon": lon, "lat": lat},
    )
    return ds


@pytest.fixture
def temp_index_file(tmp_path, mock_index_gdf):
    """Create a temporary feather index file."""
    index_path = tmp_path / "index.feather"
    mock_index_gdf.to_feather(index_path)
    return index_path


@pytest.fixture
def temp_root_dir(tmp_path):
    """Create a temporary root directory."""
    root_dir = tmp_path / "data"
    root_dir.mkdir()
    return root_dir


# Tests for IndexedGeoTiff
class TestIndexedGeoTiff:
    @patch("s2gos_generator.dataset.indexed_geotiff._load_index_gdf")
    def test_initialization(
        self, mock_load_index, mock_index_gdf, temp_index_file, temp_root_dir
    ):
        """Test IndexedGeoTiff initialization."""
        mock_load_index.return_value = mock_index_gdf

        # Create PathRef objects with real paths
        index_path = PathRef(temp_index_file, None)
        root_dir = PathRef(temp_root_dir, None)

        dataset = IndexedGeoTiff(
            name="test_geotiff",
            index_path=index_path,
            root_directory=root_dir,
            path_column="path",
        )

        assert dataset.name == "test_geotiff"
        assert dataset.path_column == "path"
        assert dataset.crs == "EPSG:4326"
        mock_load_index.assert_called_once()

        dataset = IndexedGeoTiff(
            name="test_geotiff",
            index_path=index_path,
            root_directory=root_dir,
        )

        assert dataset.path_column == "path"

    @patch("s2gos_generator.dataset.indexed_geotiff._load_index_gdf")
    def test_query_missing_path_column(
        self, mock_load_index, test_polygon_small, temp_index_file, temp_root_dir
    ):
        """Test query raises error when path column is missing."""
        # Create index without a path column
        gdf = gpd.GeoDataFrame(
            {"tile_id": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]},
            crs="EPSG:4326",
        )
        mock_load_index.return_value = gdf

        index_path = PathRef(temp_index_file, None)
        root_dir = PathRef(temp_root_dir, None)

        dataset = IndexedGeoTiff(
            name="test_geotiff",
            index_path=index_path,
            root_directory=root_dir,
        )

        with pytest.raises(ValueError, match="Missing index path column"):
            dataset.query(test_polygon_small)

    @patch("s2gos_generator.dataset.indexed_geotiff._load_index_gdf")
    def test_query_polygon(
        self,
        mock_load_index,
        mock_index_gdf,
        test_polygon_small,
        test_polygon_outside,
        temp_index_file,
        temp_root_dir,
    ):
        """Test query with an overlapping polygon."""
        mock_load_index.return_value = mock_index_gdf

        index_path = PathRef(temp_index_file, None)
        root_dir = PathRef(temp_root_dir, None)

        dataset = IndexedGeoTiff(
            name="test_geotiff",
            index_path=index_path,
            root_directory=root_dir,
            path_column="path",
        )

        result = dataset.query(test_polygon_small)

        assert len(result) > 0
        assert all(isinstance(p, UPath) for p in result)

        with pytest.raises(FileNotFoundError, match="No test_geotiff tiles found"):
            dataset.query(test_polygon_outside)

    @patch("s2gos_generator.dataset.indexed_geotiff._load_index_gdf")
    @patch("s2gos_generator.dataset.indexed_geotiff.open_dataset")
    def test_open(
        self,
        mock_open_dataset,
        mock_load_index,
        mock_index_gdf,
        mock_xarray_dataset,
        temp_index_file,
        temp_root_dir,
    ):
        """Test opening a file."""
        mock_load_index.return_value = mock_index_gdf
        mock_open_dataset.return_value = mock_xarray_dataset

        index_path = PathRef(temp_index_file, None)
        root_dir = PathRef(temp_root_dir, None)

        dataset = IndexedGeoTiff(
            name="test_geotiff",
            index_path=index_path,
            root_directory=root_dir,
        )

        test_upath = UPath("test_file.tif")
        result = dataset.open(test_upath)

        mock_open_dataset.assert_called_once_with(test_upath, engine="rasterio")
        assert result == mock_xarray_dataset

    @patch("s2gos_generator.dataset.indexed_geotiff._load_index_gdf")
    def test_from_settings(
        self, mock_load_index, mock_index_gdf, temp_index_file, temp_root_dir
    ):
        """Test creating IndexedGeoTiff from settings."""
        mock_load_index.return_value = mock_index_gdf

        settings = {
            "crs": "EPSG:32632",
            "index_path": str(temp_index_file),
            "root_directory": str(temp_root_dir),
            "path_column": "filepath",
            "variable_name": "elevation",
        }

        with patch("s2gos_utils.setting.to_pathref") as mock_to_pathref:
            mock_to_pathref.side_effect = lambda x: PathRef(x, None)
            dataset = IndexedGeoTiff.from_settings(settings, "test_dataset")

        assert dataset.name == "test_dataset"
        assert dataset.crs == "EPSG:32632"
        assert dataset.path_column == "filepath"
        assert dataset.variable_name == "elevation"

    def test_validation_missing_path(self, tmp_path):
        """Test validation fails when path doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent.feather"
        nonexistent_dir = tmp_path / "nonexistent_dir"

        index_path = PathRef(UPath(nonexistent_path), None)
        root_dir = PathRef(UPath(nonexistent_dir), None)

        with pytest.raises((ValueError, FileNotFoundError)):
            IndexedGeoTiff(
                name="test_geotiff",
                index_path=index_path,
                root_directory=root_dir,
            )


# Tests for Zarr
class TestZarr:
    def test_initialization(self, zarr_path):
        """Test Zarr initialization."""
        zarr_pathref = PathRef(zarr_path, None)

        dataset = Zarr(
            name="test_zarr",
            path=zarr_pathref,
            variable_name="temperature",
        )

        assert dataset.name == "test_zarr"
        assert dataset.variable_name == "temperature"
        assert dataset.crs == "EPSG:4326"

    @patch("s2gos_generator.dataset.zarr.open_dataset")
    def test_query_overlapping_polygon(
        self,
        mock_open_dataset,
        test_polygon_small,
        test_polygon_outside,
        mock_zarr_dataset,
        zarr_path,
    ):
        """Test query with an overlapping polygon."""
        mock_open_dataset.return_value.__enter__ = Mock(return_value=mock_zarr_dataset)
        mock_open_dataset.return_value.__exit__ = Mock(return_value=False)

        zarr_pathref = PathRef(zarr_path, None)

        dataset = Zarr(
            name="test_zarr",
            path=zarr_pathref,
        )

        result = dataset.query(test_polygon_small)

        assert len(result) == 1
        assert result[0] == zarr_pathref

        result = dataset.query(test_polygon_outside)

        assert len(result) == 0

    @patch("s2gos_generator.dataset.zarr.open_dataset")
    def test_query_with_xy_coords(
        self,
        mock_open_dataset,
        test_polygon_small,
        zarr_path,
    ):
        """Test query with x/y coordinate system."""
        # Create dataset with x/y coordinates
        x = np.linspace(10.0, 11.0, 50)
        y = np.linspace(45.0, 46.0, 50)
        data = np.random.rand(50, 50)
        ds_xy = xr.Dataset(
            {"data": (["y", "x"], data)},
            coords={"x": x, "y": y},
        )

        mock_open_dataset.return_value.__enter__ = Mock(return_value=ds_xy)
        mock_open_dataset.return_value.__exit__ = Mock(return_value=False)

        zarr_pathref = PathRef(UPath(zarr_path), None)

        dataset = Zarr(
            name="test_zarr",
            path=zarr_pathref,
        )

        result = dataset.query(test_polygon_small)

        assert len(result) == 1

    @patch("s2gos_generator.dataset.zarr.open_dataset")
    def test_query_no_valid_coords(
        self,
        mock_open_dataset,
        test_polygon_small,
        zarr_path,
    ):
        """Test query with dataset having no valid coordinate system."""
        # Create dataset without valid coordinates
        ds_no_coords = xr.Dataset(
            {"data": (["dim1", "dim2"], np.random.rand(10, 10))},
            coords={"dim1": range(10), "dim2": range(10)},
        )

        mock_open_dataset.return_value.__enter__ = Mock(return_value=ds_no_coords)
        mock_open_dataset.return_value.__exit__ = Mock(return_value=False)

        zarr_pathref = PathRef(UPath(zarr_path), None)

        dataset = Zarr(
            name="test_zarr",
            path=zarr_pathref,
        )

        result = dataset.query(test_polygon_small)

        assert len(result) == 0

    @patch("s2gos_generator.dataset.zarr.open_dataset")
    def test_query_empty_coords(
        self,
        mock_open_dataset,
        test_polygon_small,
        zarr_path,
    ):
        """Test query with empty coordinates."""
        # Create dataset with empty coordinates
        ds_empty = xr.Dataset(
            {"data": (["x", "y"], np.empty((0, 0)))},
            coords={"x": [], "y": []},
        )

        mock_open_dataset.return_value.__enter__ = Mock(return_value=ds_empty)
        mock_open_dataset.return_value.__exit__ = Mock(return_value=False)

        zarr_pathref = PathRef(UPath(zarr_path), None)

        dataset = Zarr(
            name="test_zarr",
            path=zarr_pathref,
        )

        result = dataset.query(test_polygon_small)

        assert len(result) == 0

    @patch("s2gos_generator.dataset.zarr.open_dataset")
    def test_open(
        self,
        mock_open_dataset,
        mock_zarr_dataset,
        zarr_path,
    ):
        """Test opening a Zarr dataset."""
        mock_open_dataset.return_value = mock_zarr_dataset

        zarr_pathref = PathRef(UPath(zarr_path), None)

        dataset = Zarr(
            name="test_zarr",
            path=zarr_pathref,
        )

        result = dataset.open()

        # Verify open_dataset was called with the path and zarr engine
        mock_open_dataset.assert_called_once_with(zarr_pathref, engine="zarr")
        assert result == mock_zarr_dataset

    def test_from_settings(self, zarr_path):
        """Test creating Zarr from settings."""
        settings = {
            "crs": "EPSG:32632",
            "path": str(zarr_path),
            "variable_name": "temperature",
        }

        with patch("s2gos_utils.setting.to_pathref") as mock_to_pathref:
            mock_to_pathref.return_value = PathRef(UPath(settings["path"]), None)
            dataset = Zarr.from_settings(settings, "test_zarr")

        assert dataset.name == "test_zarr"
        assert dataset.crs == "EPSG:32632"
        assert dataset.variable_name == "temperature"

    def test_validation_missing_path(self, tmp_path):
        """Test validation fails when path doesn't exist."""
        nonexistent_path = tmp_path / "nonexistent.zarr"

        zarr_pathref = PathRef(UPath(nonexistent_path), None)

        with pytest.raises((ValueError, FileNotFoundError)):
            Zarr(
                name="test_zarr",
                path=zarr_pathref,
            )


# Tests for dataset_factory
class TestDatasetFactory:
    @patch("s2gos_generator.dataset.indexed_geotiff._load_index_gdf")
    def test_factory_indexed_geotiff(
        self, mock_load_index, mock_index_gdf, temp_index_file, temp_root_dir
    ):
        """Test factory creates IndexedGeoTiff."""
        mock_load_index.return_value = mock_index_gdf

        settings = {
            "type": "indexed-geotiff",
            "index_path": str(temp_index_file),
            "root_directory": str(temp_root_dir),
        }

        with patch("s2gos_utils.setting.to_pathref") as mock_to_pathref:
            mock_to_pathref.side_effect = lambda x: PathRef(x, None)
            dataset = dataset_factory(settings, "test_dataset")

        assert isinstance(dataset, IndexedGeoTiff)
        assert dataset.name == "test_dataset"

    def test_factory_zarr(self, tmp_path):
        """Test factory creates Zarr."""
        zarr_path = tmp_path / "data.zarr"
        zarr_path.mkdir()

        settings = {
            "type": "zarr",
            "path": str(zarr_path),
        }

        with patch("s2gos_utils.setting.to_pathref") as mock_to_pathref:
            mock_to_pathref.return_value = PathRef(UPath(settings["path"]), None)
            dataset = dataset_factory(settings, "test_zarr")

        assert isinstance(dataset, Zarr)
        assert dataset.name == "test_zarr"

    def test_factory_invalid_type(self):
        """Test factory raises error for invalid type."""
        settings = {
            "type": "invalid-type",
        }

        with pytest.raises(KeyError):
            dataset_factory(settings, "test_dataset")
