# Configuration

The S2GOS generator is configured through `s2gos_settings.yaml`. The package searches for this file by climbing up the directory tree from your script's location.

## Installation Modes

The configuration structure depends on your installation:

- `common`: Shared settings (see s2gos-utils configuration)
- `generator`: Generator-specific settings (this page)
- `simulator`: Simulator-specific settings (see s2gos-simulator docs)

## Configuration Example

```yaml
# s2gos_settings.yaml
common:
    search_paths:
        - "./resources/data"
        - "./data"

generator:
    dataset:
        dem:
            name: "Copernicus-DEM-30"
            crs: "EPSG:4326"
            type: indexed-geotiff
            root_directory: "/path/to/dem/tiles"
            index_path: "/path/to/dem_index.feather"
            path_column: "path_dem"
            variable_name: elevation

        landcover:
            name: "ESA Worldcover 2021"
            crs: "EPSG:4326"
            type: zarr
            path:
                value: "s3://bucket/worldcover.zarr"
                cid: "my_s3_creds"
            variable_name: landcover

    files:
        material_config: "./materials.json"
```

## Configuration Sections

### `common` - Shared Settings

See s2gos-utils documentation.

### `generator.dataset` - Dataset Sources

#### `dem`, **Dataset**, *required*

Digital Elevation Model dataset. Tested with Copernicus DEM 30m.

#### `landcover`, **Dataset**, *required*

Land cover classification dataset. Tested with ESA WorldCover 2021.

### Dataset Types

Datasets are specified as subobjects with a `type` field. Paths use **PathRef** format:
- `value`: URI string (local or remote)
- `cid`: Credential ID (optional) - see s2gos-utils credentials documentation

#### Common Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | No | Dataset name (defaults to key name) |
| `crs` | string | No | CRS (default: `EPSG:4326`) |
| `type` | string | Yes | Dataset type |

#### Indexed GeoTiff (`type: indexed-geotiff`)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `root_directory` | PathRef | Yes | Directory containing GeoTIFF tiles |
| `index_path` | PathRef | Yes | Feather index file with tile paths |
| `variable_name` | string | No | Data variable name |
| `path_column` | string | No | Column with file paths (auto-detected) |

#### Zarr (`type: zarr`)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | PathRef | Yes | Path to Zarr archive |
| `variable_name` | string | No | Data variable name |

### `generator.files` - Additional Files

#### `material_config`, **PathRef**, *optional*

Path to materials JSON defining optical properties for land cover classes. Default: `materials.json` (resolved via `search_paths`).